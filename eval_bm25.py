from eval_weights import (load_queries, load_cache, evaluate, summarize, by_split, by_bucket, balanced_by_split, tail_vs,
                          popularity_buckets, old_weights, candidates, depth, natural, collection_name, popularity_file,
                          phase2_bm25)
import chromadb
import json
import numpy as np
import pickle
import spacy
import sys
import time

# phase 3: tunes bm25 k1 and b on the eval, with the popularity weight re-checked since b changes how bm25 treats
# long (popular, heavily tagged) documents. idf doesn't depend on k1 or b, so the eval index is rescored, not rebuilt
# --sweep scores the grid on val only, --final scores the picked config on val and test against finalist b
sweep = '--sweep' in sys.argv
final = '--final' in sys.argv
suffix = '_natural' if natural else ''
# rank_bm25's defaults, shipped until this sweep, the sweep rescores the eval cache built with them
shipped = (phase2_bm25['k1'], phase2_bm25['b'])
k1_values = [0.5, 0.9, 1.2, 1.5, 2.0, 3.0]
# b=1 is left out: 187 eval documents are empty, so b=1 divides 0 by 0 and every list comes back empty
b_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
pop_values = [0.005, 0.01, 0.02, 0.04]
# finalist b's weights, the sweep varies k1, b and popularity on top of them
B = dict(old_weights, **candidates['final_b'])
keyword_only = dict(old_weights, vector=0.0)
# picked on natural val by --sweep (results/eval_phase3_bm25_sweep_natural.json), scored on test once by --final
picked = (3.0, 0.1, 0.005)

class Rescorer:
    # rank_bm25's get_scores and Retrieval.bm25_rank with k1 and b as arguments, same arithmetic so lists match exactly
    def __init__(self, bm25, idList, movieIds, tokens):
        self.bm25, self.idList, self.movieIds = bm25, idList, movieIds
        self.doc_len = np.array(bm25.doc_len)
        # term frequency per query token over the whole corpus, built once
        self.q_freq = {t: np.zeros(bm25.corpus_size) for t in tokens}
        for d, doc in enumerate(bm25.doc_freqs):
            for t in doc.keys() & self.q_freq.keys():
                self.q_freq[t][d] = doc[t]

    def rank(self, query_tokens, k1, b, k):
        if not query_tokens:
            return []
        score = np.zeros(self.bm25.corpus_size)
        for q in query_tokens:
            q_freq = self.q_freq[q]
            score += (self.bm25.idf.get(q) or 0) * (q_freq * (k1 + 1) / (q_freq + k1 * (1 - b + b * self.doc_len / self.bm25.avgdl)))
        top = np.argsort(score)[-k:][::-1]
        return [(self.idList[i], self.movieIds[self.idList[i]], float(score[i])) for i in top if score[i] > 0]

def tokenize(queries):
    # same tokens as Retrieval.bm25_rank
    nlp = spacy.load('en_core_web_sm')
    return [[t.text for t in doc if t.is_alpha and not t.is_stop] for doc in nlp.pipe([q['query'].lower() for q in queries])]

def with_lists(cache, lists, collection):
    # the cache with its bm25 lists swapped, embeddings of candidates it hasn't seen are fetched for pref_sim
    new = sorted({m for l in lists for m, _, _ in l} - cache['emb_index'].keys())
    emb, index = cache['emb'], cache['emb_index']
    if new:
        # chroma returns rows in storage order, not request order, so each row is placed by its returned id
        index = {**index, **{m: len(emb) + j for j, m in enumerate(new)}}
        emb = np.concatenate([emb, np.zeros((len(new), emb.shape[1]), dtype=np.float32)])
        for i in range(0, len(new), 5000):
            got = collection.get(ids=[str(m) for m in new[i:i+5000]], include=['embeddings'])
            emb[[index[int(m)] for m in got['ids']]] = np.asarray(got['embeddings'], dtype=np.float32)
    entries = [dict(e, bm25=l) for e, l in zip(cache['entries'], lists)]
    return dict(cache, entries=entries, emb=emb, emb_index=index)

if __name__ == '__main__':
    config, queries = load_queries()
    cache = load_cache(collection_name, queries, phase2_bm25)
    with open(popularity_file, 'rb') as f:
        pop_data = pickle.load(f)
    popularity, counts = pop_data['eval'], pop_data['count_eval']
    buckets, cutoffs = popularity_buckets(queries, counts)
    with open('bm25/eval_bm25.pkl', 'rb') as f:
        bm25 = pickle.load(f)
    with open('movie-info/eval_movieIds.pkl', 'rb') as f:
        movieIds = pickle.load(f)
    tokens = tokenize(queries)
    start = time.time()
    rescorer = Rescorer(bm25, list(movieIds.keys()), movieIds, {t for q in tokens for t in q})
    print(f"{len(rescorer.q_freq)} query tokens indexed ({time.time()-start:.0f}s)")
    collection = chromadb.PersistentClient().get_collection(collection_name)
    rng = np.random.default_rng(0)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    # the rescorer must reproduce the cached lists at the shipped k1 and b, or nothing below is comparable
    lists = {shipped: [rescorer.rank(t, *shipped, depth) for t in tokens]}
    mismatch = sum([m for m, _, _ in l] != [m for m, _, _ in e['bm25']] for l, e in zip(lists[shipped], cache['entries']))
    assert mismatch == 0, f'{mismatch} queries rank differently from the cache at the shipped k1 and b'
    print('rescorer matches the cached bm25 lists at the shipped k1 and b')

    if sweep:
        # val only, a test query is never scored here
        idx = [i for i, q in enumerate(queries) if q['split'] == 'val']
        tail = [i for i in idx if buckets[i] == 'tail']
        val = set(idx)
        base = evaluate(cache, queries, B, popularity, idx=idx)
        ref = evaluate(cache, queries, keyword_only, popularity, idx=idx)
        diff = lambda res, other, sub: summarize([res[i]['ndcg@10'] - other[i]['ndcg@10'] for i in sub], rng)
        rows = []
        for k1 in k1_values:
            for b in b_values:
                lists_kb = [rescorer.rank(t, k1, b, depth) for t in tokens] if (k1, b) != shipped else lists[shipped]
                c = with_lists(cache, lists_kb, collection)
                recall = float(np.mean([len({str(m) for m, _, _ in l} & set(queries[i]['relevant'])) / len(queries[i]['relevant'])
                                        for i, l in enumerate(lists_kb) if i in val]))
                variants = [('keyword_only', keyword_only)] + [(f'pop={p}', dict(B, popularity=p)) for p in pop_values]
                for name, weights in variants:
                    res = evaluate(c, queries, weights, popularity, idx=idx)
                    by = {bk: float(np.mean([res[i]['ndcg@10'] for i in idx if buckets[i] == bk])) for bk in ['head', 'mid', 'tail']}
                    row = {'k1': k1, 'b': b, 'variant': name, 'weights': weights, 'bm25_recall@100': recall,
                           'ndcg@10': summarize([r['ndcg@10'] for r in res.values()], rng), 'by_popularity': by,
                           'balanced': float(np.mean(list(by.values()))),
                           'vs_b': diff(res, base, idx), 'tail_vs_b': diff(res, base, tail),
                           'tail_vs_bm25_only': diff(res, ref, tail)}
                    # lower b moves ndcg from tail to head queries, so the tail is held to finalist b, not to keyword only
                    row['eligible'] = row['tail_vs_b']['ci95'][1] >= 0
                    # phase 2's rule, kept for comparison, it let through configs that halve tail ndcg against b
                    row['eligible_phase2_rule'] = row['tail_vs_bm25_only']['ci95'][1] >= 0
                    rows.append(row)
                    print(f"k1={k1:<4} b={b:<5} {name:13} ndcg {row['ndcg@10']['mean']:.4f} bal {row['balanced']:.4f} "
                          f"vs B {row['vs_b']['mean']:+.4f} [{row['vs_b']['ci95'][0]:+.4f}, {row['vs_b']['ci95'][1]:+.4f}] "
                          f"tail vs B {row['tail_vs_b']['mean']:+.4f} [{row['tail_vs_b']['ci95'][0]:+.4f}, {row['tail_vs_b']['ci95'][1]:+.4f}] r@100 {recall:.4f} {'ok' if row['eligible'] else 'out'}", flush=True)
        # selection: highest paired gain over finalist b among eligible rows, personalized variants only
        ok = sorted([r for r in rows if r['eligible'] and r['variant'] != 'keyword_only'], key=lambda r: -r['vs_b']['mean'])
        out = args[0] if args else f'results/eval_phase3_bm25_sweep{suffix}.json'
        with open(out, 'w') as f:
            json.dump({'split': 'val', 'natural_queries': natural, 'collection': collection_name, 'shipped': shipped,
                       'base': 'final_b', 'rows': rows,
                       'rule': 'highest vs_b mean among rows whose tail_vs_b ci95 upper bound is >= 0',
                       'eligible_by_vs_b': [f"k1={r['k1']} b={r['b']} {r['variant']}" for r in ok]}, f, indent=1)
        print(f"top eligible: {[(r['k1'], r['b'], r['variant'], round(r['vs_b']['mean'], 4)) for r in ok[:5]]}")
        print(f"written to {out}")
        sys.exit()

    if final:
        assert picked, 'set picked (k1, b, popularity) from the val sweep first'
        k1, b, pop = picked
        c = with_lists(cache, [rescorer.rank(t, k1, b, depth) for t in tokens], collection)
        base = evaluate(cache, queries, B, popularity)
        ref = evaluate(cache, queries, keyword_only, popularity)
        res = evaluate(c, queries, dict(B, popularity=pop), popularity)
        results = {'natural_queries': natural, 'collection': collection_name, 'config': {'k1': k1, 'b': b, 'popularity': pop},
                   'popularity_buckets': {**cutoffs, 'n': {bk: buckets.count(bk) for bk in ['head', 'mid', 'tail']}}}
        for name, rows in [('final_b', base), ('keyword_only', ref), ('picked', res)]:
            results[name] = {'ndcg@10': by_split(rows, queries, rng), 'balanced': balanced_by_split(rows, queries, buckets),
                             'by_popularity': by_bucket(rows, buckets, rng)}
        results['picked']['vs_b'] = by_split(res, queries, rng, ref=base)
        results['picked']['tail_vs_bm25_only'] = tail_vs(res, ref, queries, buckets, rng)
        results['picked']['tail_vs_b'] = tail_vs(res, base, queries, buckets, rng)
        results['final_b']['tail_vs_bm25_only'] = tail_vs(base, ref, queries, buckets, rng)
        out = args[0] if args else f'results/eval_phase3_bm25_final{suffix}.json'
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        for split in ['val', 'test']:
            d, t = results['picked']['vs_b'][split], results['picked']['tail_vs_b'][split]
            print(f"{split:5} picked {results['picked']['ndcg@10'][split]['mean']:.4f} B {results['final_b']['ndcg@10'][split]['mean']:.4f} "
                  f"vs B {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  tail vs B "
                  f"{t['mean']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]")
        print(f"written to {out}")
