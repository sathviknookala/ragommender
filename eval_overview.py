from eval_weights import (load_queries, load_cache, evaluate, summarize, by_split, by_bucket, balanced_by_split, tail_vs,
                          popularity_buckets, union_recall, default_weights, natural, popularity_file)
from eval_embed import pop_values, sim_values, genre_values, preference_values
import json
import numpy as np
import pickle
import sys

# tmdb overviews in the embedded text: the qwen3 index rebuilt with --overviews against the shipped qwen3 index
# the shipped weights were tuned on the index without overviews, so both indexes get the same val sweep before a pick
# --sweep scores the grid on val only, --final scores the picked row on val and test against the shipped ranking
sweep = '--sweep' in sys.argv
final = '--final' in sys.argv
suffix = '_natural' if natural else ''
base_collection = 'eval_db_qwen3-embedding-0.6b_clean25'
overview_collection = base_collection + '_ov'
collections = [base_collection, overview_collection]
# what ships (phase 3), frozen so the saved results reproduce after the defaults move
base_weights = dict(default_weights, vector=0.25, preference=0.0, popularity=0.01, pref_sim=0.02, genre_boost=0.002)
# overviews may earn the knn list more weight than tags alone did, so the grid goes past phase 3's top value
vector_values = [0.0, 0.25, 0.5, 1.0, 2.0]
# picked on natural val by --sweep, scored on test once by --final
picked = None

def grid():
    for v in vector_values:
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
    base_all = evaluate(caches[base_collection], queries, base_weights, popularity)

    if sweep:
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
                # same rule as phase 3: the tail can't be clearly worse than what ships
                row['eligible'] = row['tail_vs_default']['ci95'][1] >= 0
                rows.append(row)
                d, t = row['vs_default'], row['tail_vs_default']
                print(f"{'overviews' if name == overview_collection else 'shipped':9} vec={overrides['vector']:<4} blend={overrides['preference']:<3} "
                      f"pop={overrides['popularity']:<6} sim={overrides['pref_sim']:<4} genre={overrides['genre_boost']:<5} ndcg {row['ndcg@10']['mean']:.4f} "
                      f"vs default {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] tail {t['mean']:+.4f} "
                      f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] {'ok' if row['eligible'] else 'out'}", flush=True)
        ok = sorted([r for r in rows if r['eligible']], key=lambda r: -r['vs_default']['mean'])
        best = {name: next((r for r in ok if r['collection'] == name), None) for name in collections}
        out = args[0] if args else f'results/eval_overview_sweep{suffix}.json'
        with open(out, 'w') as f:
            json.dump({'split': 'val', 'natural_queries': natural, 'base': {'collection': base_collection, 'weights': base_weights},
                       'rule': 'highest vs_default mean among rows whose tail_vs_default ci95 upper bound is >= 0',
                       'union_recall@100': {name: union_recall(caches[name], queries, rng) for name in collections},
                       'rows': rows, 'best_by_collection': best,
                       'eligible_by_vs_default': [f"{r['collection']} {r['overrides']}" for r in ok]}, f, indent=1)
        for name, r in best.items():
            print(f"best {name}: {r['overrides'] if r else None} {r['vs_default'] if r else ''}")
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
        results['picked']['own_vs_global'] = by_split(res['own'], queries, rng, ref=res['global'])
        results['picked']['own_vs_shuffled'] = by_split(res['own'], queries, rng, ref=res['shuffled'])
        out = args[0] if args else f'results/eval_overview_final{suffix}.json'
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        for split in ['val', 'test']:
            d, t, g = (results['picked'][k][split] for k in ['vs_default', 'tail_vs_default', 'own_vs_global'])
            print(f"{split:5} picked {results['picked']['ndcg@10'][split]['mean']:.4f} default {results['default']['ndcg@10'][split]['mean']:.4f} "
                  f"vs default {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  tail {t['mean']:+.4f} "
                  f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]  own vs global {g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]")
        print(f"written to {out}")
