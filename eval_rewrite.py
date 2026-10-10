from eval_weights import (load_queries, load_cache, evaluate, by_split, by_bucket, tail_vs, popularity_buckets,
                          default_weights, natural, popularity_file, shipped_collection, depth)
from eval_bm25 import with_lists
from hybrid_search import Retrieval
from rewrite import rewrite_query, rewrite_system
from concurrent.futures import ThreadPoolExecutor
import chromadb
import hashlib
import json
import llm
import numpy as np
import os
import pickle
import re
import sys

# scores the llm query rewrite (retrieval.search with rewrite=true) on the eval, it was shipped without being measured
# the rewrite's keywords are appended to the bm25 text, its genres are boosted and its era filters both lists
# nothing is tuned here, the shipped rewrite is scored as is on val and test, the other variants only take it apart
suffix = '_natural' if natural else ''
# keyed by the prompt, so a prompt change gets fresh rewrites and the old ones stay with their results
rewrites_file = f"movie-info/eval_rewrites{suffix}_{hashlib.sha1(rewrite_system.encode()).hexdigest()[:8]}.pkl"
# what ships (phase 3), frozen so the saved results reproduce after the defaults move
base_weights = dict(default_weights, vector=0.25, preference=0.0, popularity=0.01, pref_sim=0.02, genre_boost=0.002)
variants = {
    'rewrite': {'keywords': True, 'genres': True, 'years': True},
    'keywords_only': {'keywords': True, 'genres': False, 'years': False},
    'no_years': {'keywords': True, 'genres': True, 'years': False},
    # the genre boost cost natural queries about 0.005 with the first prompt, so it is scored without it too
    # what retrieval.search does since the second prompt (0551ffb7): keywords and the era filter, no genre boost
    'no_genres': {'keywords': True, 'genres': False, 'years': True},
}

def get_rewrites(texts):
    # one llm call per distinct query, saved so a rerun scores the same rewrites, failures are retried next run
    rewrites = {}
    if os.path.exists(rewrites_file):
        with open(rewrites_file, 'rb') as f:
            rewrites = pickle.load(f)
    missing = sorted(set(texts) - rewrites.keys())
    if missing:
        assert llm.chat([{'role': 'user', 'content': 'Say ok.'}], max_tokens=5), f'no llm at {llm.base_url}, start serve_llm.sh'
        with ThreadPoolExecutor(llm.max_concurrency) as pool:
            got = list(pool.map(lambda t: rewrite_query(t)[0], missing))
        rewrites.update({t: r for t, r in zip(missing, got) if isinstance(r, dict)})
        with open(rewrites_file, 'wb') as f:
            pickle.dump(rewrites, f)
    failed = len(set(texts) - rewrites.keys())
    print(f'{len(set(texts))} distinct queries, {len(missing)} rewritten this run, {failed} failed')
    return rewrites

def year_range(r):
    years = sorted(y for y in [r.get('year_from'), r.get('year_to')] if isinstance(y, int))
    return (years[0], years[-1]) if years else None

def rewritten_cache(cache, queries, rewrites, retrieval, collection, opts):
    # the shipped cache with bm25 lists from the rewritten text, and the rewrite's genres and era on each entry
    lists, extra = [], []
    for q in queries:
        r = rewrites.get(q['query'])
        years = year_range(r) if r and opts['years'] else None
        text = f"{q['query']} {' '.join(r['keywords'])}" if r and opts['keywords'] else q['query']
        lists.append(retrieval.bm25_rank(text, depth, year_range=years))
        extra.append({'query_genres': set(r['genres']) if r and opts['genres'] else set(), 'year_range': years})
    c = with_lists(cache, lists, collection)
    for e, x in zip(c['entries'], extra):
        e.update(x)
        if x['year_range']:
            # the api filters knn with a where clause, here the cached list is filtered, so it can hold fewer than depth
            lo, hi = x['year_range']
            e['knn'] = {mode: {p: [t for t in l if (y := re.search(r'\((\d{4})\)\s*$', t[1])) and lo <= int(y.group(1)) <= hi]
                               for p, l in by_p.items()} for mode, by_p in e['knn'].items()}
    return c

if __name__ == '__main__':
    config, queries = load_queries()
    with open(popularity_file, 'rb') as f:
        pop_data = pickle.load(f)
    popularity, counts = pop_data['eval'], pop_data['count_eval']
    buckets, cutoffs = popularity_buckets(queries, counts)
    rng = np.random.default_rng(0)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    rewrites = get_rewrites([q['query'] for q in queries])
    cache = load_cache(shipped_collection, queries)
    collection = chromadb.PersistentClient().get_collection(shipped_collection)
    retrieval = Retrieval(collection, 'bm25/eval_bm25.pkl', 'movie-info/eval_movieIds.pkl')
    base = evaluate(cache, queries, base_weights, popularity)
    with_years = sum(1 for q in queries if (r := rewrites.get(q['query'])) and year_range(r))
    results = {'natural_queries': natural, 'collection': shipped_collection, 'weights': base_weights,
               'n_queries': len(queries), 'rewritten': sum(q['query'] in rewrites for q in queries), 'with_year_range': with_years,
               'popularity_buckets': {**cutoffs, 'n': {b: buckets.count(b) for b in ['head', 'mid', 'tail']}},
               'default': {'ndcg@10': by_split(base, queries, rng), 'recall@10': by_split(base, queries, rng, key='recall@10'),
                           'by_popularity': by_bucket(base, buckets, rng)},
               'variants': {}}
    print(f"default ndcg@10 test {results['default']['ndcg@10']['test']['mean']:.4f}, {with_years} queries got a year range")
    for name, opts in variants.items():
        res = evaluate(rewritten_cache(cache, queries, rewrites, retrieval, collection, opts), queries, base_weights, popularity)
        v = results['variants'][name] = {'options': opts, 'ndcg@10': by_split(res, queries, rng),
                                         'recall@10': by_split(res, queries, rng, key='recall@10'),
                                         'by_popularity': by_bucket(res, buckets, rng),
                                         'vs_default': by_split(res, queries, rng, ref=base),
                                         'tail_vs_default': tail_vs(res, base, queries, buckets, rng)}
        for split in ['val', 'test']:
            d, t = v['vs_default'][split], v['tail_vs_default'][split]
            print(f"{name:14} {split:5} ndcg@10 {v['ndcg@10'][split]['mean']:.4f} vs default {d['mean']:+.4f} "
                  f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  tail {t['mean']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]", flush=True)
    results['prompt'] = rewrite_system
    out = args[0] if args else f'results/eval_rewrite{suffix}_{rewrites_file[-12:-4]}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"written to {out}")
