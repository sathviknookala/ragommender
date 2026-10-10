from eval_weights import (load_queries, load_cache, evaluate, rank, metrics, summarize, by_split, by_bucket, balanced_by_split,
                          tail_vs, popularity_buckets, default_weights, natural, popularity_file, candidate_depth)
from rerank import Reranker
import chromadb
import json
import numpy as np
import os
import pickle
import sys
import time

# phase 4: a cross-encoder reranks the top of the shipped ranking, retrieved 300 deep instead of 100
# --score scores val pools with every model and text, --sweep picks on val only, --final scores the pick on test once
score = '--score' in sys.argv
sweep = '--sweep' in sys.argv
final = '--final' in sys.argv
suffix = '_natural' if natural else ''
pool_depth = 300
base_collection = 'eval_db_qwen3-embedding-0.6b_clean25'
# the reranker reads the eval index's stored documents, which leave out the held out users' tags
texts = {'tags': base_collection, 'overview': base_collection + '_ov'}
models = ['Qwen/Qwen3-Reranker-0.6B', 'BAAI/bge-reranker-v2-m3']
# what ships (phase 3), frozen so the saved results reproduce after the defaults move
base_weights = dict(default_weights, vector=0.25, preference=0.0, popularity=0.01, pref_sim=0.02, genre_boost=0.002)
# how many of the shipped ranking's top movies the reranker reorders, the rest keep their order below them
m_values = [20, 50, 100, 300]
# prior: reranker score plus popularity and cosine to the preference vector, rrf: reranker rank fused with shipped rank
prior_pop = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
prior_sim = [0.0, 0.1, 0.2]
rrf_weights = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
# picked on natural val by --sweep: (model, text, m, mode, a, b), scored on test once by --final
picked = ('BAAI/bge-reranker-v2-m3', 'overview', 50, 'rrf', 0.125, 0.0)

def short(model):
    return model.split('/')[-1].lower()

def score_file(model, text, split):
    return f'movie-info/rerank_scores{suffix}_{short(model)}_{text}_{split}.pkl'

def documents(name, ids):
    # movieId -> stored document, chroma returns rows in storage order so they are matched by id
    collection, out = chromadb.PersistentClient().get_collection(name), {}
    for i in range(0, len(ids), 5000):
        got = collection.get(ids=ids[i:i + 5000], include=['documents'])
        out.update(zip(got['ids'], got['documents']))
    return out

def rerank_scores(model, text, split, pools, queries, reranker=None):
    # reranker score per pool movie, cached per model, text and split, every query's pool must match the cache's
    file = score_file(model, text, split)
    if os.path.exists(file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        if data['pools'] == pools:
            return data['scores']
        print(f'{file} was scored on other pools, rescoring')
    docs = documents(texts[text], sorted({m for p in pools.values() for m in p}))
    # queries repeat across users (same tag, same paraphrase), so each (query, movie) pair is scored once
    pairs = sorted({(queries[i]['query'], m) for i, p in pools.items() for m in p})
    start = time.time()
    reranker = reranker or Reranker(model)
    values = reranker.score([(q, docs[m]) for q, m in pairs])
    lookup = dict(zip(pairs, values))
    print(f'{short(model)} {text} {split}: {len(pairs)} pairs in {time.time()-start:.0f}s', flush=True)
    scores = {i: np.array([lookup[(queries[i]['query'], m)] for m in p], dtype=np.float32) for i, p in pools.items()}
    with open(file, 'wb') as f:
        pickle.dump({'model': model, 'text': text, 'pools': pools, 'scores': scores}, f)
    return scores

def reorder(cache, i, pool, rr, m, mode, a, b, popularity):
    # the top m of the shipped ranking re-sorted, everything below m keeps its shipped order
    top, s = pool[:m], rr[:m]
    if mode == 'pure':
        key = s
    elif mode == 'prior':
        key = s + a * np.array([popularity.get(int(x), 0.0) for x in top])
        e = cache['entries'][i]
        # same cold start rule as the api, no preference similarity under min_swipes
        if b and e['pref_vector'] is not None and e['n_swipes'] >= base_weights['min_swipes']:
            key = key + b * (cache['emb'][[cache['emb_index'][int(x)] for x in top]] @ e['pref_vector'])
    else:
        rr_rank = np.empty(len(s))
        rr_rank[np.argsort(-s, kind='stable')] = np.arange(1, len(s) + 1)
        key = a / (60 + rr_rank) + 1 / (60 + np.arange(1, len(s) + 1))
    order = np.argsort(-key, kind='stable')
    return [top[k] for k in order] + pool[m:]

def grid():
    for m in m_values:
        yield m, 'pure', 0.0, 0.0
        for a in prior_pop:
            for b in prior_sim:
                yield m, 'prior', a, b
        for a in rrf_weights:
            yield m, 'rrf', a, 0.0

if __name__ == '__main__':
    config, queries = load_queries()
    with open(popularity_file, 'rb') as f:
        pop_data = pickle.load(f)
    popularity, counts = pop_data['eval'], pop_data['count_eval']
    buckets, cutoffs = popularity_buckets(queries, counts)
    rng = np.random.default_rng(0)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    # what ships: the 100 deep ranking, every row is paired against it
    shipped = load_cache(base_collection, queries, depth=candidate_depth)
    base_all = evaluate(shipped, queries, base_weights, popularity)
    deep = load_cache(base_collection, queries, depth=pool_depth)
    split_idx = {s: [i for i, q in enumerate(queries) if q['split'] == s] for s in ['val', 'test']}
    pools = {s: {i: rank(deep, i, base_weights, popularity)[:pool_depth] for i in idx} for s, idx in split_idx.items()}

    if score:
        for model in models:
            reranker = Reranker(model)
            for text in texts:
                rerank_scores(model, text, 'val', pools['val'], queries, reranker)
            del reranker
        sys.exit()

    if sweep:
        idx = split_idx['val']
        tail = [i for i in idx if buckets[i] == 'tail']
        diff = lambda res, sub: summarize([res[i]['ndcg@10'] - base_all[i]['ndcg@10'] for i in sub], rng)
        deep_rows = evaluate(deep, queries, base_weights, popularity, idx=idx)
        rows = [{'model': None, 'text': None, 'm': 0, 'mode': 'shipped_300_deep', 'a': 0.0, 'b': 0.0,
                 'ndcg@10': summarize([r['ndcg@10'] for r in deep_rows.values()], rng), 'vs_default': diff(deep_rows, idx),
                 'tail_vs_default': diff(deep_rows, tail)}]
        for model in models:
            for text in texts:
                rr = rerank_scores(model, text, 'val', pools['val'], queries)
                for m, mode, a, b in grid():
                    res = {i: metrics(reorder(deep, i, pools['val'][i], rr[i], m, mode, a, b, popularity), queries[i]['relevant'])
                           for i in idx}
                    by = {bk: float(np.mean([res[i]['ndcg@10'] for i in idx if buckets[i] == bk])) for bk in ['head', 'mid', 'tail']}
                    row = {'model': model, 'text': text, 'm': m, 'mode': mode, 'a': a, 'b': b,
                           'ndcg@10': summarize([r['ndcg@10'] for r in res.values()], rng),
                           'recall@10': float(np.mean([r['recall@10'] for r in res.values()])),
                           'by_popularity': by, 'vs_default': diff(res, idx), 'tail_vs_default': diff(res, tail)}
                    # same rule as phase 3: the tail can't be clearly worse than what ships
                    row['eligible'] = row['tail_vs_default']['ci95'][1] >= 0
                    rows.append(row)
                    d, t = row['vs_default'], row['tail_vs_default']
                    print(f"{short(model):20} {text:8} m={m:<3} {mode:5} a={a:<4} b={b:<4} ndcg {row['ndcg@10']['mean']:.4f} "
                          f"vs default {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] tail {t['mean']:+.4f} "
                          f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] {'ok' if row['eligible'] else 'out'}", flush=True)
        ok = sorted([r for r in rows if r.get('eligible')], key=lambda r: -r['vs_default']['mean'])
        out = args[0] if args else f'results/eval_rerank_sweep{suffix}.json'
        with open(out, 'w') as f:
            json.dump({'split': 'val', 'natural_queries': natural, 'pool_depth': pool_depth,
                       'base': {'collection': base_collection, 'weights': base_weights, 'depth': candidate_depth},
                       'rule': 'highest vs_default mean among rows whose tail_vs_default ci95 upper bound is >= 0',
                       'rows': rows, 'top_eligible': ok[:20]}, f, indent=1)
        print(f"shipped ranking 300 deep, no rerank: vs default {rows[0]['vs_default']}")
        for r in ok[:10]:
            print(f"top: {short(r['model'])} {r['text']} m={r['m']} {r['mode']} a={r['a']} b={r['b']} {r['vs_default']['mean']:+.4f}")
        print(f"written to {out}")
        sys.exit()

    if final:
        assert picked, 'set picked (model, text, m, mode, a, b) from the val sweep first'
        model, text, m, mode, a, b = picked
        reranker = Reranker(model)
        rr = {**rerank_scores(model, text, 'val', pools['val'], queries, reranker),
              **rerank_scores(model, text, 'test', pools['test'], queries, reranker)}
        all_pools = {**pools['val'], **pools['test']}
        res = {i: metrics(reorder(deep, i, all_pools[i], rr[i], m, mode, a, b, popularity), queries[i]['relevant'])
               for i in range(len(queries))}
        results = {'natural_queries': natural, 'config': dict(zip(['model', 'text', 'm', 'mode', 'a', 'b'], picked)),
                   'pool_depth': pool_depth, 'popularity_buckets': {**cutoffs, 'n': {bk: buckets.count(bk) for bk in ['head', 'mid', 'tail']}}}
        for label, rows in [('default', base_all), ('picked', res)]:
            results[label] = {'ndcg@10': by_split(rows, queries, rng), 'recall@10': by_split(rows, queries, rng, key='recall@10'),
                              'balanced': balanced_by_split(rows, queries, buckets), 'by_popularity': by_bucket(rows, buckets, rng)}
        results['picked']['vs_default'] = by_split(res, queries, rng, ref=base_all)
        results['picked']['tail_vs_default'] = tail_vs(res, base_all, queries, buckets, rng)
        out = args[0] if args else f'results/eval_rerank_final{suffix}.json'
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        for split in ['val', 'test']:
            d, t = results['picked']['vs_default'][split], results['picked']['tail_vs_default'][split]
            print(f"{split:5} picked {results['picked']['ndcg@10'][split]['mean']:.4f} default {results['default']['ndcg@10'][split]['mean']:.4f} "
                  f"vs default {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  tail {t['mean']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]")
        print(f"by popularity {results['picked']['by_popularity']}")
        print(f"written to {out}")
