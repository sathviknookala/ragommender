import re

# the ranking math shared by the api (hybrid_search, retrieval) and the eval, kept free of model imports

# vector, popularity and the bm25 parameters below were picked on the consensus eval's natural val queries
# (evaluation/run.py --natural --sweep), natural test +0.0151 [+0.0065, +0.0236] over phase 3's vector 0.25,
# popularity 0.01, k1 3. rrf_k and year_boost are hand-picked
default_weights = {
    # 0 would skip the knn query, qwen3's knn earns a small weight
    'vector': 0.125,
    'bm25': 1.0,
    'rrf_k': 60,
    # added to movies in the era a rewritten query asks for
    'year_boost': 0.005,
    # added per candidate, scaled popularity (0..1)
    'popularity': 0.02
}

# candidates fetched per list, popularity and the era boost re-sort the whole pool so it must match the eval's
candidate_depth = 100

# bm25 k1 and b, picked with the weights above (k1 3 and 5 tie on val). rank_bm25's defaults, which the pickles are
# built with, are 1.5 and 0.75. a low b barely normalizes length, and indexed text grows with tag count, so it leans
# towards popular movies
bm25_params = {'k1': 5.0, 'b': 0.1}

def parse_year(title):
    match = re.search(r'\((\d{4})\)\s*$', title)
    return int(match.group(1)) if match else None

def fuse(knn_results, bm25_results, weights, popularity=None):
    # weighted reciprocal rank fusion, a zero weight drops that retriever's list
    # popularity is a movieId -> value dict, added on top of the fused score
    fused = {}
    for name, results in [('vector', knn_results), ('bm25', bm25_results)]:
        if weights[name] == 0:
            continue
        for rank, (movieId, title, _) in enumerate(results, 1):
            item = fused.setdefault(movieId, {'item_id': str(movieId), 'title': title, 'score': 0.0,
                                              'vector_rank': None, 'bm25_rank': None, 'era_boost': 0.0})
            item['score'] += weights[name]/(weights['rrf_k']+rank)
            item[f'{name}_rank'] = rank
    for movieId, item in fused.items():
        if popularity and weights['popularity']:
            item['score'] += weights['popularity'] * popularity.get(movieId, 0.0)
    return sorted(fused.values(), key=lambda x: x['score'], reverse=True)

def boost_era(items, weights, year_range):
    # lifts fused items released in the era a rewritten query asks for and re-sorts them
    if not year_range:
        return items
    for item in items:
        year = parse_year(item['title'])
        if year and year_range[0] <= year <= year_range[1]:
            item['era_boost'] = weights['year_boost']
            item['score'] += weights['year_boost']
    return sorted(items, key=lambda x: x['score'], reverse=True)

def with_fallback(weights, bm25_results):
    # with knn off, a query bm25 can't match (only stopwords, numbers, typos) would return nothing, so knn fills in
    if not weights['vector'] and not bm25_results:
        return dict(weights, vector=1.0)
    return weights
