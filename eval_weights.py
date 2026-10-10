from hybrid_search import Retrieval, default_weights, fuse, with_fallback, candidate_depth, bm25_params
from preferences import preferences_from_swipes, apply_boosts
import chromadb
import json
import numpy as np
import os
import pickle
import sys
import time

# scores ranking weights against the offline eval set built by eval_build.py
# shared with the api so the shipped candidate pool is the evaluated one
# --depth=N builds and scores deeper pools, for reranking (eval_rerank.py), in their own cache
depth = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--depth=')), candidate_depth)
# blend values the cache holds knn lists for, fixed because the cache layout depends on them
pref_values = [0.0, 0.3]
# the defaults before phase 2, sweep and final stay relative to these so their saved results reproduce
old_weights = {'vector': 1.0, 'bm25': 1.0, 'rrf_k': 60, 'preference': 0.3, 'genre_boost': 0.002, 'year_boost': 0.005,
               'min_swipes': 5, 'popularity': 0.0, 'pref_sim': 0.0}
# --natural swaps each tag for its llm paraphrase from eval_paraphrase.py
natural = '--natural' in sys.argv
# --collection=NAME scores another eval index, e.g. eval_db_clean25 from eval_build.py --clean-embed=25
# the eval index that matches the shipped rag_db, phase 3 moved it to qwen3 on title, genres and the top 25 tags.
# eval_db is minilm on the full text, what phases 1 and 2 ran on
shipped_collection = 'eval_db_qwen3-embedding-0.6b_clean25'
collection_name = next((a.split('=')[1] for a in sys.argv if a.startswith('--collection=')), shipped_collection)
# --compare=NAME also scores that collection on the same queries and saves the paired difference
compare_name = next((a.split('=')[1] for a in sys.argv if a.startswith('--compare=')), None)
# --sweep tunes the popularity and preference similarity weights on val only, --final scores the picked candidates
sweep = '--sweep' in sys.argv
final = '--final' in sys.argv
# --bm25=K1,B scores with other bm25 parameters, phase 2 ran at rank_bm25's defaults: --bm25=1.5,0.75
bm25_arg = next(({'k1': float(k1), 'b': float(b)} for a in sys.argv if a.startswith('--bm25=')
                 for k1, b in [a.split('=')[1].split(',')]), bm25_params)
phase2_bm25 = {'k1': 1.5, 'b': 0.75}
suffix = ('_natural' if natural else '') + ('' if collection_name == 'eval_db' else f'_{collection_name}')
bootstrap_n = 1000
# 4 is a bump when the cache layout changes, an older cache is rebuilt instead of misread
cache_version = 4
popularity_file = 'movie-info/popularity.pkl'
shuffle_seed = 1

def cache_path(name, bm25=bm25_arg, depth=depth):
    # caches at phase 2's bm25 parameters keep their original names, they predate the parameter
    params = '' if bm25 == phase2_bm25 else f"_k1{bm25['k1']}_b{bm25['b']}"
    pool = '' if depth == candidate_depth else f'_d{depth}'
    return f"movie-info/eval_cache_v{cache_version}{'_natural' if natural else ''}{'' if name == 'eval_db' else f'_{name}'}{params}{pool}.pkl"

def controls(entries, queries):
    # shuffled: each query gets another same-split user's preferences, global: everyone gets the mean preferences
    # both break the link between the user and their history, so any gain left over wasn't personal
    valid = [i for i, e in enumerate(entries) if e['pref_vector'] is not None]
    shuffle = list(range(len(entries)))
    rng = np.random.default_rng(shuffle_seed)
    for split in ['val', 'test']:
        idx = [i for i in valid if queries[i]['split'] == split]
        for i, j in zip(idx, rng.permutation(idx)):
            shuffle[i] = int(j)
    # the global prior comes from val users only and is reused for test, so tuning on val never sees test users
    val = [i for i in valid if queries[i]['split'] == 'val']
    mean = np.mean([entries[i]['pref_vector'] for i in val], axis=0)
    genres = {}
    for i in val:
        for g, v in entries[i]['genre_preferences'].items():
            genres[g] = genres.get(g, 0.0) + v / len(val)
    return shuffle, mean / np.linalg.norm(mean), genres

def retrieve_all(retrieval, collection, queries, depth=depth):
    # one retrieval pass per query and preference weight, every weight variant re-scores these lists
    entries = []
    start = time.time()
    for q in queries:
        prefs = preferences_from_swipes(collection, q['swipes'])
        entries.append({'knn': {}, 'bm25': retrieval.bm25_rank(q['query'], depth),
                        'genre_preferences': prefs['genre_preferences'], 'n_swipes': len(q['swipes']),
                        'pref_vector': prefs['preference_vector']})
    shuffle, mean, genres = controls(entries, queries)
    sources = {'own': lambda i: entries[i]['pref_vector'], 'shuffled': lambda i: entries[shuffle[i]]['pref_vector'],
               'global': lambda i: mean if entries[i]['pref_vector'] is not None else None}
    for n, (q, e) in enumerate(zip(queries, entries), 1):
        for mode, source in sources.items():
            e['knn'][mode] = {}
            # the controls only matter where the user has a preference vector and the blend is on
            if mode != 'own' and e['pref_vector'] is None:
                continue
            for p in (pref_values if mode == 'own' else pref_values[1:]):
                vec = retrieval.blend(q['query'], source(n - 1), p)
                e['knn'][mode][p] = retrieval.knn_search(k=depth, query_embeddings=vec)[1]
        if n % 500 == 0:
            print(f"retrieved {n}/{len(queries)} ({time.time()-start:.0f}s)")
    # every candidate's embedding is stored once, so preference similarity re-scores offline
    ids = sorted({m for e in entries for lists in [e['bm25']] + [l for k in e['knn'].values() for l in k.values()]
                  for m, _, _ in lists})
    emb = None
    index = {m: i for i, m in enumerate(ids)}
    for i in range(0, len(ids), 5000):
        got = collection.get(ids=[str(m) for m in ids[i:i+5000]], include=['embeddings'])
        if emb is None:
            # the dimension comes from the first batch
            emb = np.zeros((len(ids), len(got['embeddings'][0])), dtype=np.float32)
        # chroma returns rows in storage order, not request order, so each row is placed by its returned id
        emb[[index[int(m)] for m in got['ids']]] = np.asarray(got['embeddings'], dtype=np.float32)
    return {'version': cache_version, 'depth': depth, 'collection': collection.name, 'n_queries': len(queries),
            'entries': entries, 'emb_index': index, 'emb': emb,
            'shuffle': shuffle, 'global_vector': mean, 'global_genres': genres}

def load_cache(name, queries, bm25=bm25_arg, depth=depth):
    # a cache from another format, collection, query set or bm25 parameters is rebuilt instead of silently misread
    file = cache_path(name, bm25, depth)
    if os.path.exists(file) and '--refresh' not in sys.argv:
        with open(file, 'rb') as f:
            cache = pickle.load(f)
        if (isinstance(cache, dict) and cache.get('version') == cache_version and cache['collection'] == name
                and cache['n_queries'] == len(queries) and cache['depth'] == depth
                and cache.get('bm25_params', phase2_bm25) == bm25):
            return cache
        print(f"{file} is an old or mismatched cache, rebuilding")
    collection = chromadb.PersistentClient().get_collection(name)
    retrieval = Retrieval(collection, 'bm25/eval_bm25.pkl', 'movie-info/eval_movieIds.pkl', bm25_params=bm25)
    cache = dict(retrieve_all(retrieval, collection, queries, depth), bm25_params=bm25)
    with open(file, 'wb') as f:
        pickle.dump(cache, f)
    return cache

def rank(cache, i, weights, popularity=None, mode='own'):
    # same cold start rule as the api: no preference blending or similarity under min_swipes
    c = cache['entries'][i]
    on = c['n_swipes'] >= weights['min_swipes']
    p = weights['preference'] if on else 0.0
    vector, genre_preferences = c['pref_vector'], c['genre_preferences']
    if mode != 'own' and vector is not None:
        if mode == 'shuffled':
            other = cache['entries'][cache['shuffle'][i]]
            vector, genre_preferences = other['pref_vector'], other['genre_preferences']
        else:
            vector, genre_preferences = cache['global_vector'], cache['global_genres']
    knn = c['knn']['own'][0.0] if p == 0 else c['knn'][mode if c['pref_vector'] is not None else 'own'][p]
    sims = None
    if weights['pref_sim'] and on and vector is not None:
        ids = list({m for m, _, _ in knn + c['bm25']})
        sims = dict(zip(ids, cache['emb'][[cache['emb_index'][m] for m in ids]] @ vector))
    # same fallback as the api, knn fills in when bm25 matched nothing
    items = fuse(knn, c['bm25'], with_fallback(weights, c['bm25']), popularity, sims)
    # genres and an era from the llm rewrite, only set by eval_rewrite.py, as retrieval.search boosts them
    return [item['item_id'] for item in apply_boosts(items, genre_preferences if weights['genre_boost'] else {}, weights,
                                                     c.get('query_genres', ()), c.get('year_range'))]

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

def evaluate(cache, queries, weights, popularity=None, mode='own', idx=None):
    return {i: metrics(rank(cache, i, weights, popularity, mode), queries[i]['relevant'])
            for i in (range(len(queries)) if idx is None else idx)}

def summarize(values, rng):
    values = np.asarray(values)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(bootstrap_n)]
    return {'mean': float(values.mean()), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

def by_split(rows, queries, rng, key='ndcg@10', ref=None, splits=('all', 'val', 'test')):
    # mean per split, or the paired difference against ref (another rows dict over the same queries)
    out = {}
    for split in splits:
        idx = [i for i in rows if split == 'all' or queries[i]['split'] == split]
        if idx:
            out[split] = summarize([rows[i][key] - (ref[i][key] if ref else 0.0) for i in idx], rng)
    return out

def popularity_buckets(queries, counts):
    # tail, mid or head by tertiles of the median rating count of a query's relevant movies
    med = [np.median([counts.get(int(m), 0) for m in q['relevant']]) for q in queries]
    lo, hi = np.percentile(med, [100 / 3, 200 / 3])
    return ['head' if m > hi else 'tail' if m <= lo else 'mid' for m in med], {'tail_max_ratings': float(lo), 'head_min_ratings_exclusive': float(hi)}

def by_bucket(rows, buckets, rng):
    return {b: dict(n=len(idx), **summarize([rows[i]['ndcg@10'] for i in idx], rng))
            for b in ['head', 'mid', 'tail'] if (idx := [i for i in rows if buckets[i] == b])}

def balanced_by_split(rows, queries, buckets, splits=('all', 'val', 'test')):
    # head, mid and tail means within each split, and their plain mean as in the sweep's balanced score
    out = {}
    for split in splits:
        by = {b: float(np.mean([rows[i]['ndcg@10'] for i in rows if buckets[i] == b and (split == 'all' or queries[i]['split'] == split)]))
              for b in ['head', 'mid', 'tail']}
        out[split] = {'balanced': float(np.mean(list(by.values()))), **by}
    return out

def tail_vs(rows, ref, queries, buckets, rng):
    # paired difference on tail queries only, per split
    tail = {i: rows[i] for i in rows if buckets[i] == 'tail'}
    return by_split(tail, queries, rng, ref=ref)

def union_recall(cache, queries, rng):
    # share of relevant movies somewhere in the knn and bm25 candidate lists, the ceiling for any re-ranking
    out = {}
    lists = {'knn_p0': lambda c: c['knn']['own'][0.0], 'knn_p0.3': lambda c: c['knn']['own'][pref_values[1]],
             'bm25': lambda c: c['bm25'], 'union_p0': lambda c: c['knn']['own'][0.0] + c['bm25'],
             'union_p0.3': lambda c: c['knn']['own'][pref_values[1]] + c['bm25']}
    for name, get in lists.items():
        rows = {i: {'r': len({str(m) for m, _, _ in get(c)} & set(q['relevant'])) / len(q['relevant'])}
                for i, (c, q) in enumerate(zip(cache['entries'], queries))}
        out[name] = by_split(rows, queries, rng, key='r')
    return out

# each variant states the weights it is about as absolute values, so it can't drift when the shipped defaults change
# only 'default' follows default_weights, the rest stay fixed
def V(vector, bm25, preference, popularity, pref_sim, genre_boost=0.002, **rest):
    return dict(vector=vector, bm25=bm25, preference=preference, popularity=popularity, pref_sim=pref_sim,
                genre_boost=genre_boost, **rest)
variants = {
    'default': {},
    # the defaults before phase 2, for continuity with the first baseline
    'old_default': V(1.0, 1.0, 0.3, 0.0, 0.0),
    # semantic only keeps the old 0.3 blend so it compares to the first baseline
    'vector_only': V(1.0, 0.0, 0.3, 0.0, 0.0),
    'bm25_only': V(0.0, 1.0, 0.0, 0.0, 0.0),
    # the personalization ablations hold vector and popularity at the shipped values so they differ only in
    # personalization: v2 ran them at vector 0 and popularity 0.01, v3 at vector 0 and 0.005, v4 at 0.25 and 0.01
    'no_personalization': V(0.25, 1.0, 0.0, 0.01, 0.0, genre_boost=0.0),
    'preference_vector_only': V(0.25, 1.0, 0.0, 0.01, 0.02, genre_boost=0.0),
    'genre_boost_only': V(0.25, 1.0, 0.0, 0.01, 0.0),
    'min_swipes_0': dict(min_swipes=0),
    'min_swipes_1': dict(min_swipes=1),
    'min_swipes_3': dict(min_swipes=3),
    'min_swipes_10': dict(min_swipes=10),
}
# variants that use the user's history, each also runs with shuffled and global preferences
personal = ['default', 'old_default', 'preference_vector_only', 'genre_boost_only', 'min_swipes_0', 'min_swipes_1', 'min_swipes_3', 'min_swipes_10']

# picked on val by --sweep, scored on test once by --final
candidates = {
    'final_a': {'vector': 0.0, 'preference': 0.0, 'popularity': 0.005, 'pref_sim': 0.04},
    'final_b': {'vector': 0.0, 'preference': 0.0, 'popularity': 0.01, 'pref_sim': 0.02},
}
# comparison rows scored in the same run, bm25_only is also the reference for the tail check
references = {
    'bm25_only': {'vector': 0.0},
    'pop_only_a': {'vector': 0.0, 'popularity': 0.005},
}

def load_queries():
    with open('movie-info/eval_queries.pkl', 'rb') as f:
        data = pickle.load(f)
    queries = data['queries']
    if natural:
        with open('movie-info/eval_natural.pkl', 'rb') as f:
            paraphrases = pickle.load(f)
        queries = [dict(q, tag=q['query'], query=paraphrases[q['query']]) for q in queries if q['query'] in paraphrases]
    return data['config'], queries

def run_sweep(cache, queries, popularity, rng, buckets):
    # val only, a test query is never scored here
    idx = [i for i, q in enumerate(queries) if q['split'] == 'val']
    W = lambda **o: dict(old_weights, **o)
    base = evaluate(cache, queries, W(), idx=idx)
    # the reference for the tail check: keyword only, other weights at default
    bm25 = evaluate(cache, queries, W(vector=0.0), idx=idx)
    tail = [i for i in idx if buckets[i] == 'tail']
    rows = []

    def diff(res, ref, sub):
        return summarize([res[i]['ndcg@10'] - ref[i]['ndcg@10'] for i in sub], rng)

    def add(name, group, overrides, mode='own', pop=None, controls=False):
        res = evaluate(cache, queries, W(**overrides), pop, mode, idx)
        by = {b: float(np.mean([res[i]['ndcg@10'] for i in idx if buckets[i] == b])) for b in ['head', 'mid', 'tail']}
        row = {'name': name, 'group': group, 'overrides': overrides, 'mode': mode,
               'ndcg@10': summarize([r['ndcg@10'] for r in res.values()], rng), 'by_popularity': by,
               # equal weight per popularity bucket, so head queries can't carry the score
               'balanced': float(np.mean(list(by.values()))),
               'vs_default': diff(res, base, idx), 'tail_vs_bm25_only': diff(res, bm25, tail)}
        # eligible unless clearly worse than keyword only on the tail
        row['eligible'] = row['tail_vs_bm25_only']['ci95'][1] >= 0
        if controls:
            # same weights with the global mean and another user's preferences, own minus control
            for c in ['global', 'shuffled']:
                ref = evaluate(cache, queries, W(**overrides), pop, c, idx)
                row[f'own_vs_{c}'] = diff(res, ref, idx)
                row[f'own_vs_{c}_tail'] = diff(res, ref, tail)
        rows.append(row)
        print(f"{name:50} {row['ndcg@10']['mean']:.4f} bal {row['balanced']:.4f} tail-vs-bm25 {row['tail_vs_bm25_only']['mean']:+.4f} "
              f"[{row['tail_vs_bm25_only']['ci95'][0]:+.4f}, {row['tail_vs_bm25_only']['ci95'][1]:+.4f}] "
              f"{'ok' if row['eligible'] else 'out'}", flush=True)

    pop_weights = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04]
    sim_weights = [0.005, 0.01, 0.02, 0.04]
    shapes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.3), (1.0, 0.0), (1.0, 0.3)]
    # 1. popularity weight on its own, across the vector weight and the existing blend
    for vec, p in shapes:
        for pw in pop_weights:
            add(f'vec={vec} blend={p} pop={pw}', 'popularity', {'vector': vec, 'preference': p, 'popularity': pw}, pop=popularity)
    # 2. the same prior as a global mean preference vector instead of an explicit popularity score
    for sw in sim_weights:
        add(f'vec=0 global-vector prior sim={sw}', 'global_vector_prior', {'vector': 0.0, 'preference': 0.0, 'pref_sim': sw}, 'global')
    # 3. preference similarity on top of the popularity weight, with the global and shuffled controls
    for vec, p in shapes:
        for pw in pop_weights:
            for sw in sim_weights:
                add(f'vec={vec} blend={p} pop={pw} sim={sw}', 'pref_sim',
                    {'vector': vec, 'preference': p, 'popularity': pw, 'pref_sim': sw}, 'own', popularity, True)
    ok = sorted([r for r in rows if r['eligible']], key=lambda r: -r['balanced'])
    return rows, [r['name'] for r in ok]

if __name__ == '__main__':
    config, queries = load_queries()
    cache = load_cache(collection_name, queries)
    with open(popularity_file, 'rb') as f:
        pop_data = pickle.load(f)
    # counts leave out the held out eval users, so the eval can't see the ratings it is tested on
    popularity, counts = pop_data['eval'], pop_data['count_eval']
    buckets, cutoffs = popularity_buckets(queries, counts)
    rng = np.random.default_rng(0)
    os.makedirs('results', exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if sweep:
        out = args[0] if args else f'results/eval_phase2_sweep{suffix}.json'
        results = {'split': 'val', 'natural_queries': natural, 'collection': collection_name,
                   'popularity_buckets': {**cutoffs, 'n': {b: buckets.count(b) for b in ['head', 'mid', 'tail']}}}
        results['rows'], results['eligible_by_balanced'] = run_sweep(cache, queries, popularity, rng, buckets)
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        print(f"written to {out}")
        sys.exit()

    if final:
        # picked candidates scored on val and test against the current default, with controls and popularity buckets
        out = args[0] if args else f'results/eval_phase2_final{suffix}.json'
        default = evaluate(cache, queries, dict(old_weights))
        results = {'natural_queries': natural, 'collection': collection_name, 'candidates': {},
                   'popularity_buckets': {**cutoffs, 'n': {b: buckets.count(b) for b in ['head', 'mid', 'tail']}},
                   'default': {'ndcg@10': by_split(default, queries, rng), 'by_popularity': by_bucket(default, buckets, rng)}}
        ref = evaluate(cache, queries, dict(old_weights, **references['bm25_only']), popularity)
        results['references'] = {}
        for name, overrides in references.items():
            rows = ref if name == 'bm25_only' else evaluate(cache, queries, dict(old_weights, **overrides), popularity)
            results['references'][name] = {'weights': overrides, 'ndcg@10': by_split(rows, queries, rng),
                                           'balanced': balanced_by_split(rows, queries, buckets),
                                           'by_popularity': by_bucket(rows, buckets, rng),
                                           'vs_default': by_split(rows, queries, rng, ref=default),
                                           'tail_vs_bm25_only': tail_vs(rows, ref, queries, buckets, rng)}
        for name, overrides in candidates.items():
            weights = dict(old_weights, **overrides)
            entry = {'weights': overrides, 'modes': {}}
            own = evaluate(cache, queries, weights, popularity, 'own')
            for mode in ['own', 'shuffled', 'global']:
                rows = own if mode == 'own' else evaluate(cache, queries, weights, popularity, mode)
                entry['modes'][mode] = {'ndcg@10': by_split(rows, queries, rng),
                                        'vs_default': by_split(rows, queries, rng, ref=default),
                                        'by_popularity': by_bucket(rows, buckets, rng),
                                        'balanced': balanced_by_split(rows, queries, buckets),
                                        'tail_vs_bm25_only': tail_vs(rows, ref, queries, buckets, rng)}
                if mode == 'global':
                    entry['own_vs_global'] = by_split(own, queries, rng, ref=rows)
                    # the ship-the-global-vector option against the current default
                    entry['global_vs_default'] = entry['modes']['global']['vs_default']
            results['candidates'][name] = entry
            for split, d in entry['modes']['own']['vs_default'].items():
                print(f"{name:24} {split:5} ndcg@10 {entry['modes']['own']['ndcg@10'][split]['mean']:.4f}  vs default "
                      f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  own vs global "
                      f"{entry['own_vs_global'][split]['mean']:+.4f} [{entry['own_vs_global'][split]['ci95'][0]:+.4f}, {entry['own_vs_global'][split]['ci95'][1]:+.4f}]")
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        print(f"written to {out}")
        sys.exit()

    per_query = {}
    for name, overrides in variants.items():
        weights = dict(default_weights, **overrides)
        per_query[name] = evaluate(cache, queries, weights, popularity)
    # the same variants with another user's preferences and with the mean preferences
    control_rows = {(name, mode): evaluate(cache, queries, dict(default_weights, **variants[name]), popularity, mode)
                    for name in personal for mode in ['shuffled', 'global']}

    results = {'config': config, 'depth': depth, 'n_queries': len(queries), 'natural_queries': natural,
               'collection': collection_name, 'cache_version': cache_version, 'shuffle_seed': shuffle_seed,
               'weights': default_weights, 'variants': {}, 'vs_default': {}, 'vs_default_by_split': {},
               'by_history': {}, 'controls': {}, 'by_popularity': {}, 'union_recall@100': union_recall(cache, queries, rng),
               'popularity_buckets': {**cutoffs,
                                      'n': {b: buckets.count(b) for b in ['head', 'mid', 'tail']}}}
    for name, rows in per_query.items():
        results['variants'][name] = {split: {m: summarize([r[m] for i, r in rows.items() if split == 'all' or queries[i]['split'] == split], rng)
                                             for m in ['ndcg@10', 'recall@10', 'recall@50']}
                                     for split in ['all', 'val', 'test']}
        results['by_popularity'][name] = by_bucket(rows, buckets, rng)
        if name != 'default':
            # paired difference per query, the ci says whether the change is real, per split as well
            results['vs_default_by_split'][name] = by_split(rows, queries, rng, ref=per_query['default'])
            results['vs_default'][name] = {'ndcg@10_diff': results['vs_default_by_split'][name]['all']}
    for name in personal:
        own = per_query[name]
        entry = {mode: by_split(rows, queries, rng) for mode, rows in
                 [('own', own), ('shuffled', control_rows[name, 'shuffled']), ('global', control_rows[name, 'global'])]}
        # the personalization gain is own against global, global alone carries any popularity prior
        entry['own_vs_global'] = by_split(own, queries, rng, ref=control_rows[name, 'global'])
        entry['own_vs_shuffled'] = by_split(own, queries, rng, ref=control_rows[name, 'shuffled'])
        results['controls'][name] = entry

    buckets_h = [(0, 4), (5, 19), (20, 49), (50, 10**9)]
    for lo, hi in buckets_h:
        idx = [i for i, q in enumerate(queries) if lo <= q['n_history'] <= hi]
        if not idx:
            continue
        diff = [per_query['default'][i]['ndcg@10'] - per_query['no_personalization'][i]['ndcg@10'] for i in idx]
        results['by_history'][f'{lo}-{hi if hi < 10**9 else "+"}'] = {
            'n': len(idx),
            'default_ndcg@10': summarize([per_query['default'][i]['ndcg@10'] for i in idx], rng),
            'personalization_gain': summarize(diff, rng),
            'gain_vs_global': summarize([per_query['default'][i]['ndcg@10'] - control_rows['default', 'global'][i]['ndcg@10'] for i in idx], rng)
        }

    if compare_name:
        # another collection on the same queries, paired per query: this collection minus the other
        other = load_cache(compare_name, queries)
        results['compare'] = {'other': compare_name, 'vs_other': {}}
        for name in ['default', 'vector_only', 'bm25_only']:
            rows = evaluate(other, queries, dict(default_weights, **variants[name]), popularity)
            results['compare']['vs_other'][name] = by_split(per_query[name], queries, rng, ref=rows)

    out = args[0] if args else ('results/eval_baseline.json' if not suffix else f'results/eval{suffix}.json')
    if compare_name:
        out = args[0] if args else f'results/eval{suffix}_vs_{compare_name}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)

    print(f"{len(queries)} queries, depth {depth}, written to {out}")
    for name, splits in results['variants'].items():
        a = splits['all']
        diff = results['vs_default'].get(name, {}).get('ndcg@10_diff')
        diff_text = f"  vs default {diff['mean']:+.4f} [{diff['ci95'][0]:+.4f}, {diff['ci95'][1]:+.4f}]" if diff else ''
        print(f"{name:24} ndcg@10 {a['ndcg@10']['mean']:.4f} [{a['ndcg@10']['ci95'][0]:.4f}, {a['ndcg@10']['ci95'][1]:.4f}]"
              f"  r@10 {a['recall@10']['mean']:.4f}  r@50 {a['recall@50']['mean']:.4f}{diff_text}")
    for name, c in results['controls'].items():
        g = c['own_vs_global']['all']
        print(f"{name:24} own {c['own']['all']['mean']:.4f}  shuffled {c['shuffled']['all']['mean']:.4f}  global {c['global']['all']['mean']:.4f}"
              f"  own vs global {g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]")
    for bucket, b in results['by_history'].items():
        g = b['personalization_gain']
        print(f"history {bucket:6} n={b['n']:5}  ndcg@10 {b['default_ndcg@10']['mean']:.4f}  "
              f"personalization gain {g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]")
