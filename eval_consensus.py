from eval_build import pick_test_users, read_ratings, seed
from eval_weights import summarize, popularity_file, shipped_collection, old_weights, depth
from eval_paraphrase import paraphrase, natural_file
from hybrid_search import Retrieval, default_weights, fuse, with_fallback
from preferences import apply_boosts
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

# a second label set where a query has many right answers: a movie is relevant to a tag when at least min_users of
# the eval's held out users applied that tag to it. their tags are left out of the eval index, so a label never
# matches its own tag application, the index only holds other users' tags (the crowd signal production has too)
# queries are per tag, not per user, so they are scored for a cold user (no swipes, no personalization)
# --build makes movie-info/eval_consensus.pkl and paraphrases tags the main eval hasn't, scoring is the default
consensus_file = 'movie-info/eval_consensus.pkl'
min_users = 2
min_relevant = 5
natural = '--natural' in sys.argv
suffix = '_natural' if natural else ''
# what ships (phase 3), frozen so the saved results reproduce after the defaults move
shipped = dict(default_weights, vector=0.25, preference=0.0, popularity=0.01, pref_sim=0.02, genre_boost=0.002)
configs = {
    'shipped': shipped,
    'old_hybrid': dict(old_weights),
    'bm25_only': dict(shipped, vector=0.0, popularity=0.0),
    'no_popularity': dict(shipped, popularity=0.0),
}
# the second rewrite prompt's saved rewrites (eval_rewrite.py), a query without one keeps its text as the api does
rewrites_file = f"movie-info/eval_rewrites_natural_{hashlib.sha1(rewrite_system.encode()).hexdigest()[:8]}.pkl"

def build():
    ratings = read_ratings()
    tags, test_users = pick_test_users(np.random.default_rng(seed), ratings)
    held = tags[tags['userId'].isin(test_users)]
    users = held.groupby(['query', 'movieId'])['userId'].nunique()
    agreed = users[users >= min_users].reset_index()
    queries = []
    for tag, group in agreed.groupby('query'):
        if len(group) >= min_relevant:
            # split by tag, so val and test never share a query
            split = 'val' if int(hashlib.sha1(tag.encode()).hexdigest(), 16) % 2 == 0 else 'test'
            queries.append({'tag': tag, 'relevant': [str(m) for m in group['movieId']],
                            'users': {str(m): int(n) for m, n in zip(group['movieId'], group['userId'])}, 'split': split})
    # paraphrases for tags the main eval didn't have, added as new keys so its queries don't change
    with open(natural_file, 'rb') as f:
        paraphrases = pickle.load(f)
    missing = sorted({q['tag'] for q in queries} - paraphrases.keys())
    assert llm.chat([{'role': 'user', 'content': 'Say ok.'}], max_tokens=5), f'no llm at {llm.base_url}, start serve_llm.sh'
    if missing:
        with ThreadPoolExecutor(llm.max_concurrency) as pool:
            paraphrases.update({t: p for t, p in zip(missing, pool.map(paraphrase, missing)) if p})
        with open(natural_file, 'wb') as f:
            pickle.dump(paraphrases, f)
    # rewrites of every paraphrase with the shipped prompt, added to eval_rewrite.py's file for that prompt
    rewrites = {}
    if os.path.exists(rewrites_file):
        with open(rewrites_file, 'rb') as f:
            rewrites = pickle.load(f)
    texts = sorted({paraphrases[q['tag']] for q in queries if q['tag'] in paraphrases} - rewrites.keys())
    if texts:
        with ThreadPoolExecutor(llm.max_concurrency) as pool:
            rewrites.update({t: r for t, r in zip(texts, pool.map(lambda t: rewrite_query(t)[0], texts)) if isinstance(r, dict)})
        with open(rewrites_file, 'wb') as f:
            pickle.dump(rewrites, f)
    config = {'min_users': min_users, 'min_relevant': min_relevant, 'seed': seed, 'held_out_users': len(test_users)}
    with open(consensus_file, 'wb') as f:
        pickle.dump({'config': config, 'queries': queries}, f)
    n = [len(q['relevant']) for q in queries]
    print(f"{len(queries)} tag queries ({sum(q['split'] == 'val' for q in queries)} val), relevant per query median "
          f"{int(np.median(n))}, max {max(n)}, {len(missing)} tags paraphrased, "
          f"{sum(q['tag'] not in paraphrases for q in queries)} without a paraphrase")

def metrics(ranked, relevant):
    relevant = set(relevant)
    gains = [1.0 if i in relevant else 0.0 for i in ranked[:10]]
    dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(relevant), 10)))
    return {'ndcg@10': dcg / idcg, 'precision@10': sum(gains) / 10, 'recall@10': sum(gains) / len(relevant)}

if __name__ == '__main__':
    if '--build' in sys.argv:
        build()
        sys.exit()
    with open(consensus_file, 'rb') as f:
        queries = pickle.load(f)['queries']
    if natural:
        with open(natural_file, 'rb') as f:
            paraphrases = pickle.load(f)
        queries = [dict(q, query=paraphrases[q['tag']]) for q in queries if q['tag'] in paraphrases]
    else:
        queries = [dict(q, query=q['tag']) for q in queries]
    with open(popularity_file, 'rb') as f:
        popularity = pickle.load(f)['eval']
    collection = chromadb.PersistentClient().get_collection(shipped_collection)
    retrieval = Retrieval(collection, 'bm25/eval_bm25.pkl', 'movie-info/eval_movieIds.pkl')
    rewrites = {}
    if natural and os.path.exists(rewrites_file):
        with open(rewrites_file, 'rb') as f:
            rewrites = pickle.load(f)
    use_rewrite = natural and bool(rewrites)
    rng = np.random.default_rng(0)

    # retrieval once per query, every config re-fuses the same lists
    lists = []
    for q in queries:
        knn = retrieval.knn_search(k=depth, query_embeddings=retrieval.blend(q['query']))[1]
        bm25 = retrieval.bm25_rank(q['query'], depth)
        entry = {'knn': knn, 'bm25': bm25}
        if use_rewrite:
            r = rewrites.get(q['query'], {'keywords': [], 'year_from': None, 'year_to': None})
            years = sorted(y for y in [r.get('year_from'), r.get('year_to')] if isinstance(y, int))
            entry['rewrite'] = (retrieval.bm25_rank(f"{q['query']} {' '.join(r['keywords'])}", depth,
                                                    year_range=(years[0], years[-1]) if years else None),
                                (years[0], years[-1]) if years else None)
        lists.append(entry)

    def rank(e, weights, rewrite=False):
        bm25, years = e['rewrite'] if rewrite else (e['bm25'], None)
        # the api filters knn with a where clause, here the list is filtered by the year in each title
        knn = [t for t in e['knn'] if (y := re.search(r'\((\d{4})\)\s*$', t[1])) and years[0] <= int(y.group(1)) <= years[1]] if years else e['knn']
        items = fuse(knn, bm25, with_fallback(weights, bm25), popularity)
        return [i['item_id'] for i in apply_boosts(items, {}, weights, year_range=years)]

    rows = {name: [metrics(rank(e, w), q['relevant']) for e, q in zip(lists, queries)] for name, w in configs.items()}
    if use_rewrite:
        rows['shipped_rewrite'] = [metrics(rank(e, shipped, True), q['relevant']) for e, q in zip(lists, queries)]
    # popularity alone over the shipped candidate pool, and a perfect reorder of that pool
    pools = [list({m for m, _, _ in e['knn']} | {m for m, _, _ in e['bm25']}) for e in lists]
    rows['popularity_only'] = [metrics([str(m) for m in sorted(p, key=lambda m: -popularity.get(m, 0.0))], q['relevant'])
                               for p, q in zip(pools, queries)]
    rows['oracle_pool'] = [metrics([str(m) for m in sorted(p, key=lambda m: str(m) not in set(q['relevant']))], q['relevant'])
                           for p, q in zip(pools, queries)]

    results = {'natural_queries': natural, 'n_queries': len(queries), 'collection': shipped_collection,
               'min_users': min_users, 'min_relevant': min_relevant, 'rewrite_scored': use_rewrite,
               'without_rewrite': sum(q['query'] not in rewrites for q in queries) if use_rewrite else None, 'configs': {}}
    print(f"{len(queries)} {'natural' if natural else 'tag'} queries, relevant per query median "
          f"{int(np.median([len(q['relevant']) for q in queries]))}")
    for name, r in rows.items():
        results['configs'][name] = {}
        for split in ['val', 'test']:
            idx = [i for i, q in enumerate(queries) if q['split'] == split]
            entry = {m: summarize([r[i][m] for i in idx], rng) for m in ['ndcg@10', 'precision@10', 'recall@10']}
            if name != 'shipped':
                entry['vs_shipped'] = summarize([r[i]['ndcg@10'] - rows['shipped'][i]['ndcg@10'] for i in idx], rng)
            results['configs'][name][split] = entry
        t = results['configs'][name]['test']
        d = f"  vs shipped {t['vs_shipped']['mean']:+.4f} [{t['vs_shipped']['ci95'][0]:+.4f}, {t['vs_shipped']['ci95'][1]:+.4f}]" if 'vs_shipped' in t else ''
        print(f"{name:16} test ndcg@10 {t['ndcg@10']['mean']:.4f} [{t['ndcg@10']['ci95'][0]:.4f}, {t['ndcg@10']['ci95'][1]:.4f}]  "
              f"p@10 {t['precision@10']['mean']:.3f}  r@10 {t['recall@10']['mean']:.3f}{d}")
    out = f'results/eval_consensus{suffix}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"written to {out}")
