from eval_build import labels_file
from eval_paraphrase import natural_file
from build_popularity import popularity_file
from hybrid_search import Retrieval, default_weights, fuse, with_fallback, boost_era, parse_year, candidate_depth, bm25_params
from rewrite import rewrite_query, rewrite_system
from concurrent.futures import ThreadPoolExecutor
import chromadb
import hashlib
import itertools
import json
import llm
import numpy as np
import os
import pickle
import sys
import time

# the main offline eval. each query is a movielens tag from the consensus label set (eval_build.py), its relevant
# movies are the ones at least 2 held out users gave that tag, and it is ranked the way retrieval.search ranks it:
# bm25 and knn fused by weighted rrf, plus popularity and, for a rewritten query, the era filter and boost.
# scores are NDCG@10, precision@10 and recall@10 with bootstrap CIs, paired against what ships, per split.
# val and test are split by tag: pick on val, score test once
#   --natural            swaps each tag for its llm paraphrase (eval_paraphrase.py) and scores the llm rewrite too,
#                        --rewrite fills rewrites the saved file is missing through the local llm
#   --sweep              grids vector, popularity and bm25 k1/b on val only, test isn't scored
#   --weights=K=V,...    scores those weights (vector, popularity, k1, b) as 'candidate' against what ships
#   --collection=NAME    another eval index, e.g. eval_db_qwen3-embedding-0.6b_clean25_ov
#   --depth=N            candidates per list
natural = '--natural' in sys.argv
sweep = '--sweep' in sys.argv
# the eval index that matches the shipped rag_db: qwen3 on title, genres and the top 25 tags
shipped_collection = 'eval_db_qwen3-embedding-0.6b_clean25'
collection_name = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--collection=')), shipped_collection)
depth = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--depth=')), candidate_depth)
candidate = next(({k: float(v) for k, v in (kv.split('=') for kv in a.split('=', 1)[1].split(','))}
                  for a in sys.argv if a.startswith('--weights=')), None)
suffix = ('_natural' if natural else '') + ('' if collection_name == shipped_collection else f'_{collection_name}') + \
         ('' if depth == candidate_depth else f'_d{depth}')
bootstrap_n = 1000
metric_names = ['ndcg@10', 'precision@10', 'recall@10']
# saved rewrites of the natural queries, keyed by the rewrite prompt so a prompt change gets fresh ones
rewrites_file = f"movie-info/eval_rewrites_natural_{hashlib.sha1(rewrite_system.encode()).hexdigest()[:8]}.pkl"
# fixed references beside what ships, the old hybrid is stated in absolute values so it can't drift with the defaults
references = {
    'old_hybrid': dict(default_weights, vector=1.0, popularity=0.0),
    'bm25_only': dict(default_weights, vector=0.0, popularity=0.0),
    'no_popularity': dict(default_weights, popularity=0.0),
}
grid = {'vector': [0.0, 0.125, 0.25, 0.5, 1.0], 'popularity': [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04],
        'k1': [1.5, 3.0, 5.0], 'b': [0.1, 0.3, 0.75]}

def load_queries():
    with open(labels_file, 'rb') as f:
        queries = pickle.load(f)['queries']
    if not natural:
        return [dict(q, query=q['tag']) for q in queries]
    with open(natural_file, 'rb') as f:
        paraphrases = pickle.load(f)
    missing = sum(q['tag'] not in paraphrases for q in queries)
    if missing:
        print(f"{missing} tags have no paraphrase and are left out, run eval_paraphrase.py")
    return [dict(q, query=paraphrases[q['tag']]) for q in queries if q['tag'] in paraphrases]

def load_rewrites(queries):
    rewrites = {}
    if os.path.exists(rewrites_file):
        with open(rewrites_file, 'rb') as f:
            rewrites = pickle.load(f)
    missing = sorted({q['query'] for q in queries} - rewrites.keys())
    if missing and '--rewrite' in sys.argv:
        assert llm.chat([{'role': 'user', 'content': 'Say ok.'}], max_tokens=5), f'no llm at {llm.base_url}, start serve_llm.sh'
        with ThreadPoolExecutor(llm.max_concurrency) as pool:
            got = list(pool.map(lambda t: rewrite_query(t)[0], missing))
        rewrites.update({t: r for t, r in zip(missing, got) if isinstance(r, dict)})
        with open(rewrites_file, 'wb') as f:
            pickle.dump(rewrites, f)
    return rewrites

def year_range(r):
    years = sorted(y for y in [r.get('year_from'), r.get('year_to')] if isinstance(y, int))
    return (years[0], years[-1]) if years else None

def bm25_lists(retrieval, queries, params, rewrites=None):
    # bm25 lists at these k1 and b, from each query's rewritten text and era when rewrites are given
    retrieval.bm25_data.k1, retrieval.bm25_data.b = params['k1'], params['b']
    lists = []
    for q in queries:
        if rewrites is None:
            lists.append((retrieval.bm25_rank(q['query'], depth), None))
        else:
            # a query without a rewrite keeps its own text, as the api does when the llm fails
            r = rewrites.get(q['query'], {'keywords': []})
            years = year_range(r)
            lists.append((retrieval.bm25_rank(f"{q['query']} {' '.join(r['keywords'])}", depth, year_range=years), years))
    return lists

def rank(knn, bm25, years, weights, popularity):
    # the api filters knn with a where clause on year, here the list is filtered by the year in each title
    if years:
        knn = [t for t in knn if (y := parse_year(t[1])) and years[0] <= y <= years[1]]
    items = fuse(knn, bm25, with_fallback(weights, bm25), popularity)
    return [i['item_id'] for i in boost_era(items, weights, years)]

def metrics(ranked, relevant):
    relevant = set(relevant)
    gains = [1.0 if i in relevant else 0.0 for i in ranked[:10]]
    dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(relevant), 10)))
    return {'ndcg@10': dcg / idcg, 'precision@10': sum(gains) / 10, 'recall@10': sum(gains) / len(relevant)}

def score(knn, bm25, queries, weights, popularity, idx):
    return {i: metrics(rank(knn[i], bm25[i][0], bm25[i][1], weights, popularity), queries[i]['relevant']) for i in idx}

def summarize(values, rng):
    values = np.asarray(values)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(bootstrap_n)]
    return {'mean': float(values.mean()), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

def popularity_buckets(queries, counts):
    # tail, mid or head by tertiles of the median rating count of a query's relevant movies
    med = [np.median([counts.get(int(m), 0) for m in q['relevant']]) for q in queries]
    lo, hi = np.percentile(med, [100 / 3, 200 / 3])
    return ['head' if m > hi else 'tail' if m <= lo else 'mid' for m in med]

def report(rows, ref, queries, buckets, rng, splits=('val', 'test')):
    # per split means of every metric, by popularity bucket, and the paired ndcg difference against ref
    out = {}
    for split in splits:
        idx = [i for i in rows if queries[i]['split'] == split]
        entry = {m: summarize([rows[i][m] for i in idx], rng) for m in metric_names}
        entry['by_popularity'] = {b: summarize([rows[i]['ndcg@10'] for i in idx if buckets[i] == b], rng)
                                  for b in ['head', 'mid', 'tail']}
        if ref is not None:
            entry['vs_shipped'] = summarize([rows[i]['ndcg@10'] - ref[i]['ndcg@10'] for i in idx], rng)
            entry['tail_vs_shipped'] = summarize([rows[i]['ndcg@10'] - ref[i]['ndcg@10'] for i in idx if buckets[i] == 'tail'], rng)
        out[split] = entry
    return out

def show(name, split, entry):
    d = entry.get('vs_shipped')
    vs = f"  vs shipped {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else ''
    print(f"{name:16} {split:4} ndcg@10 {entry['ndcg@10']['mean']:.4f} [{entry['ndcg@10']['ci95'][0]:.4f}, "
          f"{entry['ndcg@10']['ci95'][1]:.4f}]  p@10 {entry['precision@10']['mean']:.3f}  "
          f"r@10 {entry['recall@10']['mean']:.3f}{vs}", flush=True)

if __name__ == '__main__':
    start = time.time()
    queries = load_queries()
    with open(popularity_file, 'rb') as f:
        pop_data = pickle.load(f)
    popularity = pop_data['eval']
    buckets = popularity_buckets(queries, pop_data['count_eval'])
    collection = chromadb.PersistentClient().get_collection(collection_name)
    retrieval = Retrieval(collection, 'bm25/eval_bm25.pkl', 'movie-info/eval_movieIds.pkl')
    rng = np.random.default_rng(0)
    print(f"{len(queries)} {'natural' if natural else 'tag'} queries on {collection_name}, relevant per query median "
          f"{int(np.median([len(q['relevant']) for q in queries]))}")

    # retrieval once per query, every configuration re-fuses the same lists
    vectors = retrieval.embed([q['query'] for q in queries])
    knn = [retrieval.knn_search(k=depth, query_embeddings=v)[1] for v in vectors]
    bm25 = bm25_lists(retrieval, queries, bm25_params)
    every = range(len(queries))
    shipped = score(knn, bm25, queries, default_weights, popularity, every)
    results = {'natural_queries': natural, 'n_queries': len(queries), 'collection': collection_name, 'depth': depth,
               'weights': default_weights, 'bm25_params': bm25_params,
               'popularity_buckets': {b: buckets.count(b) for b in ['head', 'mid', 'tail']}}

    if sweep:
        # val only, every grid row against what ships, bm25 lists are rebuilt per k1 and b
        val = [i for i in every if queries[i]['split'] == 'val']
        rows = []
        for k1, b in itertools.product(grid['k1'], grid['b']):
            lists = bm25_lists(retrieval, queries, {'k1': k1, 'b': b}) if (k1, b) != (bm25_params['k1'], bm25_params['b']) else bm25
            for vector, pop in itertools.product(grid['vector'], grid['popularity']):
                w = dict(default_weights, vector=vector, popularity=pop)
                r = score(knn, lists, queries, w, popularity, val)
                rows.append({'vector': vector, 'popularity': pop, 'k1': k1, 'b': b,
                             'ndcg@10': float(np.mean([r[i]['ndcg@10'] for i in val])),
                             'diff': [r[i]['ndcg@10'] - shipped[i]['ndcg@10'] for i in val]})
        rows.sort(key=lambda r: -r['ndcg@10'])
        for r in rows[:15]:
            r['vs_shipped'] = summarize(r['diff'], rng)
        shipped_val = float(np.mean([shipped[i]['ndcg@10'] for i in val]))
        print(f"shipped val ndcg@10 {shipped_val:.4f}, {len(rows)} rows, best:")
        for r in rows[:15]:
            d = r['vs_shipped']
            print(f"  vector {r['vector']:<5} popularity {r['popularity']:<6} k1 {r['k1']:<3} b {r['b']:<4} ndcg@10 "
                  f"{r['ndcg@10']:.4f}  vs shipped {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
        results.update(split='val', shipped_val=shipped_val, grid=grid,
                       rows=[{k: v for k, v in r.items() if k != 'diff'} for r in rows])
        out = f'results/eval_consensus_sweep{suffix}.json'
    else:
        rows = {'shipped': shipped}
        rows.update({name: score(knn, bm25, queries, w, popularity, every) for name, w in references.items()})
        if candidate:
            w = dict(default_weights, **{k: v for k, v in candidate.items() if k in default_weights})
            params = {'k1': candidate.get('k1', bm25_params['k1']), 'b': candidate.get('b', bm25_params['b'])}
            lists = bm25_lists(retrieval, queries, params) if params != bm25_params else bm25
            rows['candidate'] = score(knn, lists, queries, w, popularity, every)
            results['candidate'] = {'weights': w, 'bm25_params': params}
        if natural:
            rewrites = load_rewrites(queries)
            if rewrites:
                rows['shipped_rewrite'] = score(knn, bm25_lists(retrieval, queries, bm25_params, rewrites), queries,
                                                default_weights, popularity, every)
                results['rewrite'] = {'file': rewrites_file, 'without_rewrite': sum(q['query'] not in rewrites for q in queries)}
        # popularity alone over the shipped candidate pool, and a perfect reorder of that pool
        pools = [list({m for m, _, _ in k} | {m for m, _, _ in b}) for k, (b, _) in zip(knn, bm25)]
        rows['popularity_only'] = {i: metrics([str(m) for m in sorted(p, key=lambda m: -popularity.get(m, 0.0))], q['relevant'])
                                   for i, (p, q) in enumerate(zip(pools, queries))}
        rows['oracle_pool'] = {i: metrics([str(m) for m in sorted(p, key=lambda m: str(m) not in set(q['relevant']))], q['relevant'])
                               for i, (p, q) in enumerate(zip(pools, queries))}
        results['configs'] = {}
        for name, r in rows.items():
            results['configs'][name] = report(r, None if name == 'shipped' else shipped, queries, buckets, rng)
            for split in ['val', 'test']:
                show(name, split, results['configs'][name][split])
        out = f'results/eval_consensus{suffix}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"written to {out}, {time.time()-start:.0f}s")
