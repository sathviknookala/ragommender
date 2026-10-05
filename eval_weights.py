from hybrid_search import Retrieval, default_weights, fuse
from preferences import preferences_from_swipes, apply_boosts
import chromadb
import json
import numpy as np
import os
import pickle
import sys
import time

# scores ranking weights against the offline eval set built by eval_build.py
depth = 100
pref_values = [0.0, default_weights['preference']]
cache_file = 'movie-info/eval_cache.pkl'
bootstrap_n = 1000

def retrieve_all(retrieval, collection, queries):
    # one retrieval pass per query and preference weight, every weight variant re-scores these lists
    cache = []
    start = time.time()
    for n, q in enumerate(queries, 1):
        prefs = preferences_from_swipes(collection, q['swipes'])
        knn = {}
        for p in pref_values:
            vec = retrieval.blend(q['query'], prefs['preference_vector'], p)
            _, knn[p] = retrieval.knn_search(k=depth, query_embeddings=vec)
        cache.append({'knn': knn, 'bm25': retrieval.bm25_rank(q['query'], depth),
                      'genre_preferences': prefs['genre_preferences'], 'n_swipes': len(q['swipes'])})
        if n % 500 == 0:
            print(f"retrieved {n}/{len(queries)} ({time.time()-start:.0f}s)")
    return cache

def rank(cached, weights):
    # same cold start rule as the api: no preference blending under min_swipes
    p = weights['preference'] if cached['n_swipes'] >= weights['min_swipes'] else 0.0
    genre_preferences = cached['genre_preferences'] if weights['genre_boost'] else {}
    items = fuse(cached['knn'][p], cached['bm25'], weights)
    return [item['item_id'] for item in apply_boosts(items, genre_preferences, weights)]

def metrics(ranked, relevant):
    relevant = set(relevant)
    gains = [1.0 if i in relevant else 0.0 for i in ranked[:10]]
    dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(relevant), 10)))
    return {
        'ndcg@10': dcg / idcg,
        'recall@10': len(relevant & set(ranked[:10])) / len(relevant),
        'recall@50': len(relevant & set(ranked[:50])) / len(relevant)
    }

def summarize(values, rng):
    values = np.asarray(values)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(bootstrap_n)]
    return {'mean': float(values.mean()), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

variants = {
    'default': {},
    'vector_only': {'bm25': 0.0},
    'bm25_only': {'vector': 0.0},
    'no_personalization': {'preference': 0.0, 'genre_boost': 0.0},
    'preference_vector_only': {'genre_boost': 0.0},
    'genre_boost_only': {'preference': 0.0},
}

if __name__ == '__main__':
    with open('movie-info/eval_queries.pkl', 'rb') as f:
        data = pickle.load(f)
    queries = data['queries']
    collection = chromadb.PersistentClient().get_collection('eval_db')
    retrieval = Retrieval(collection, 'bm25/eval_bm25.pkl', 'movie-info/eval_movieIds.pkl')

    if os.path.exists(cache_file) and '--refresh' not in sys.argv:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = retrieve_all(retrieval, collection, queries)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)

    rng = np.random.default_rng(0)
    per_query = {}
    for name, overrides in variants.items():
        weights = dict(default_weights, **overrides)
        per_query[name] = [metrics(rank(c, weights), q['relevant']) for c, q in zip(cache, queries)]

    results = {'config': data['config'], 'depth': depth, 'n_queries': len(queries),
               'weights': default_weights, 'variants': {}, 'vs_default': {}, 'by_history': {}}
    for name, rows in per_query.items():
        results['variants'][name] = {split: {m: summarize([r[m] for r, q in zip(rows, queries) if split == 'all' or q['split'] == split], rng)
                                             for m in ['ndcg@10', 'recall@10', 'recall@50']}
                                     for split in ['all', 'val', 'test']}
        if name != 'default':
            # paired difference per query, the ci says whether the change is real
            diff = [r['ndcg@10'] - d['ndcg@10'] for r, d in zip(rows, per_query['default'])]
            results['vs_default'][name] = {'ndcg@10_diff': summarize(diff, rng)}

    buckets = [(0, 4), (5, 19), (20, 49), (50, 10**9)]
    for lo, hi in buckets:
        idx = [i for i, q in enumerate(queries) if lo <= q['n_history'] <= hi]
        if not idx:
            continue
        diff = [per_query['default'][i]['ndcg@10'] - per_query['no_personalization'][i]['ndcg@10'] for i in idx]
        results['by_history'][f'{lo}-{hi if hi < 10**9 else "+"}'] = {
            'n': len(idx),
            'default_ndcg@10': summarize([per_query['default'][i]['ndcg@10'] for i in idx], rng),
            'personalization_gain': summarize(diff, rng)
        }

    os.makedirs('results', exist_ok=True)
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'results/eval_baseline.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)

    print(f"{len(queries)} queries, depth {depth}, written to {out}")
    for name, splits in results['variants'].items():
        a = splits['all']
        diff = results['vs_default'].get(name, {}).get('ndcg@10_diff')
        diff_text = f"  vs default {diff['mean']:+.4f} [{diff['ci95'][0]:+.4f}, {diff['ci95'][1]:+.4f}]" if diff else ''
        print(f"{name:24} ndcg@10 {a['ndcg@10']['mean']:.4f} [{a['ndcg@10']['ci95'][0]:.4f}, {a['ndcg@10']['ci95'][1]:.4f}]"
              f"  r@10 {a['recall@10']['mean']:.4f}  r@50 {a['recall@50']['mean']:.4f}{diff_text}")
    for bucket, b in results['by_history'].items():
        g = b['personalization_gain']
        print(f"history {bucket:6} n={b['n']:5}  ndcg@10 {b['default_ndcg@10']['mean']:.4f}  "
              f"personalization gain {g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]")
