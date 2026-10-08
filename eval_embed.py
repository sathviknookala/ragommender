from eval_weights import (load_queries, load_cache, evaluate, summarize, by_split, by_bucket, balanced_by_split, tail_vs,
                          popularity_buckets, union_recall, default_weights, natural, popularity_file)
import json
import numpy as np
import pickle
import sys

# phase 3: compares embedding models at the shipped bm25 parameters. each collection's knn lists and preference
# vectors come from its own model, so pref_sim changes with the model even when the vector weight is 0
# --sweep scores a weight grid per collection on val only, --final scores the picked row on val and test
sweep = '--sweep' in sys.argv
final = '--final' in sys.argv
suffix = '_natural' if natural else ''
# the shipped index, then minilm and qwen3 on the same text (title, genres, top 25 tags), so the pair isolates the model
base_collection = 'eval_db'
# what shipped when this ran (phase 3 bm25 on minilm), frozen so the saved results reproduce after the defaults moved
base_weights = dict(default_weights, vector=0.0, preference=0.0, popularity=0.005, pref_sim=0.02, genre_boost=0.002)
collections = ['eval_db', 'eval_db_clean25', 'eval_db_qwen3-embedding-0.6b_clean25']
vector_values = [0.0, 0.25, 0.5, 1.0]
# the cache holds knn lists for these blends only (eval_weights.pref_values)
preference_values = [0.0, 0.3]
pop_values = [0.0025, 0.005, 0.01]
sim_values = [0.0, 0.01, 0.02, 0.04]
# at the phase 3 bm25 parameters the genre boost stopped helping in the base eval, so it is re-tuned here on val
genre_values = [0.0, 0.002]
# picked on natural val by --sweep (results/eval_phase3_embed_sweep_natural.json), scored on test once by --final
picked = ('eval_db_qwen3-embedding-0.6b_clean25', {'vector': 0.25, 'preference': 0.0, 'popularity': 0.01, 'pref_sim': 0.02, 'genre_boost': 0.002})

def grid():
    for v in vector_values:
        # the blend only changes the knn list, so with knn off it would repeat the same row
        for p in (preference_values if v else [0.0]):
            for pop in pop_values:
                for s in sim_values:
                    for g in genre_values:
                        yield {'vector': v, 'preference': p, 'popularity': pop, 'pref_sim': s, 'genre_boost': g}

if __name__ == '__main__':
    config, queries = load_queries()
    with open(popularity_file, 'rb') as f:
        pop_data = pickle.load(f)
    popularity, counts = pop_data['eval'], pop_data['count_eval']
    buckets, cutoffs = popularity_buckets(queries, counts)
    rng = np.random.default_rng(0)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    caches = {name: load_cache(name, queries) for name in collections}
    # every row is paired against the shipped ranking on the shipped index
    base_all = evaluate(caches[base_collection], queries, base_weights, popularity)

    if sweep:
        # val only, a test query is never scored here
        idx = [i for i, q in enumerate(queries) if q['split'] == 'val']
        tail = [i for i in idx if buckets[i] == 'tail']
        diff = lambda res, sub: summarize([res[i]['ndcg@10'] - base_all[i]['ndcg@10'] for i in sub], rng)
        rows = []
        for name in collections:
            for overrides in grid():
                res = evaluate(caches[name], queries, dict(base_weights, **overrides), popularity, idx=idx)
                by = {b: float(np.mean([res[i]['ndcg@10'] for i in idx if buckets[i] == b])) for b in ['head', 'mid', 'tail']}
                row = {'collection': name, 'overrides': overrides, 'ndcg@10': summarize([r['ndcg@10'] for r in res.values()], rng),
                       'by_popularity': by, 'balanced': float(np.mean(list(by.values()))),
                       'vs_default': diff(res, idx), 'tail_vs_default': diff(res, tail)}
                # same rule as the bm25 sweep: the tail can't be clearly worse than what ships
                row['eligible'] = row['tail_vs_default']['ci95'][1] >= 0
                rows.append(row)
                d, t = row['vs_default'], row['tail_vs_default']
                print(f"{name[8:] or 'minilm':32} vec={overrides['vector']:<4} blend={overrides['preference']:<3} pop={overrides['popularity']:<6} "
                      f"sim={overrides['pref_sim']:<4} genre={overrides['genre_boost']:<5} ndcg {row['ndcg@10']['mean']:.4f} vs default {d['mean']:+.4f} "
                      f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] tail {t['mean']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] "
                      f"{'ok' if row['eligible'] else 'out'}", flush=True)
        ok = sorted([r for r in rows if r['eligible']], key=lambda r: -r['vs_default']['mean'])
        best = {name: next((r for r in ok if r['collection'] == name), None) for name in collections}
        out = args[0] if args else f'results/eval_phase3_embed_sweep{suffix}.json'
        with open(out, 'w') as f:
            json.dump({'split': 'val', 'natural_queries': natural, 'base': {'collection': base_collection, 'weights': base_weights},
                       'rule': 'highest vs_default mean among rows whose tail_vs_default ci95 upper bound is >= 0',
                       'union_recall@100': {name: union_recall(caches[name], queries, rng) for name in collections},
                       'rows': rows, 'best_by_collection': best,
                       'eligible_by_vs_default': [f"{r['collection']} {r['overrides']}" for r in ok]}, f, indent=1)
        for name, r in best.items():
            print(f"best {name}: {r['overrides'] if r else None} {r['vs_default']['mean'] if r else ''}")
        print(f"written to {out}")
        sys.exit()

    if final:
        assert picked, 'set picked (collection, overrides) from the val sweep first'
        name, overrides = picked
        weights = dict(base_weights, **overrides)
        res = {mode: evaluate(caches[name], queries, weights, popularity, mode) for mode in ['own', 'shuffled', 'global']}
        results = {'natural_queries': natural, 'config': {'collection': name, 'overrides': overrides},
                   'popularity_buckets': {**cutoffs, 'n': {b: buckets.count(b) for b in ['head', 'mid', 'tail']}}}
        for label, rows in [('default', base_all), ('picked', res['own'])]:
            results[label] = {'ndcg@10': by_split(rows, queries, rng), 'balanced': balanced_by_split(rows, queries, buckets),
                              'by_popularity': by_bucket(rows, buckets, rng)}
        results['picked']['vs_default'] = by_split(res['own'], queries, rng, ref=base_all)
        results['picked']['tail_vs_default'] = tail_vs(res['own'], base_all, queries, buckets, rng)
        # whether any preference similarity gain is personal, as in phase 2
        results['picked']['own_vs_global'] = by_split(res['own'], queries, rng, ref=res['global'])
        results['picked']['own_vs_shuffled'] = by_split(res['own'], queries, rng, ref=res['shuffled'])
        # attribution, not a second pick: the same weights on the shipped minilm index, so the gap is the model's
        same = evaluate(caches[base_collection], queries, weights, popularity)
        results['same_weights_minilm'] = {'ndcg@10': by_split(same, queries, rng), 'vs_default': by_split(same, queries, rng, ref=base_all),
                                          'picked_vs_this': by_split(res['own'], queries, rng, ref=same)}
        out = args[0] if args else f'results/eval_phase3_embed_final{suffix}.json'
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        for split in ['val', 'test']:
            d, t, g = (results['picked'][k][split] for k in ['vs_default', 'tail_vs_default', 'own_vs_global'])
            print(f"{split:5} picked {results['picked']['ndcg@10'][split]['mean']:.4f} default {results['default']['ndcg@10'][split]['mean']:.4f} "
                  f"vs default {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  tail {t['mean']:+.4f} "
                  f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]  own vs global {g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]")
        m = results['same_weights_minilm']['picked_vs_this']['test']
        print(f"test  picked vs the same weights on {base_collection}: {m['mean']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]")
        print(f"written to {out}")
