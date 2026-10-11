from functools import cache
from ragommender.hybrid_search import Retrieval, default_embed_model
from ragommender.ranking import default_weights, candidate_depth, boost_era, vocab_params, rewrite_vocab_params
from ragommender.vocab import TagVocab, tag_prompt, map_tags, expand
from ragommender.rewrite import rewrite_query, pick_tags
from concurrent.futures import ThreadPoolExecutor
from ragommender import llm, paths
import chromadb
import os

@cache
def index():
    # the shipped index, loaded on first use (the api warms it at startup) so importing this module is cheap
    name = os.environ.get('CHROMA_COLLECTION', 'rag_db')
    collection = chromadb.PersistentClient(path=str(paths.chroma_dir)).get_collection(name)
    retrieval = Retrieval(collection, paths.bm25_file, paths.movieIds_file, paths.popularity_file)
    return collection, retrieval

@cache
def vocab():
    # the tag vocabulary queries are mapped onto, None when the mapping is off
    if not vocab_params['n']:
        return None
    assert paths.vocab_file.exists(), f'vocabulary mapping is on but {paths.vocab_file} is missing, run ingest.build_vocab'
    return TagVocab(paths.vocab_file)

def map_query(query: str, params=vocab_params):
    # the vocabulary tags added to the bm25 text, as the eval adds them (evaluation/run.py bm25_lists)
    if vocab() is None or query.strip().lower() in vocab().tag_set:
        return []
    _, retrieval = index()
    vec = retrieval.embed(query, None if params['prompt'] == 'movie' else tag_prompt)
    return map_tags(vocab(), query, vec, params)

def rewrite_and_pick(query: str):
    # the rewrite and the llm's pick among the nearest tags, run at once so the second call adds little latency
    candidates = [t for t, _ in map_query(query, rewrite_vocab_params)]
    with ThreadPoolExecutor(2) as pool:
        rewritten = pool.submit(rewrite_query, query)
        picked = pool.submit(pick_tags, query, candidates)
        return rewritten.result(), picked.result()

def model_version():
    # the index and embedding model results come from, returned with every search
    collection, _ = index()
    return f"{collection.name}/{(collection.metadata or {}).get('embed_model', default_embed_model)}"

explain_top = 5

explain_system = ("You explain movie search results. For each candidate write one sentence of at most 20 words "
                  "pointing to the candidate's own genres, themes or setting. Never restate the search or say it fits "
                  "the search. Reply as json {item_id: reason}.")

def explain_items(query: str, items: list):
    ids = [item['item_id'] for item in items]
    docs = index()[0].get(ids=ids, include=['documents'])
    descriptions = dict(zip(docs['ids'], docs['documents']))
    candidates = '\n'.join(f"{i}: {descriptions.get(i, '')[:300]}" for i in ids)
    schema = {'type': 'object', 'properties': {i: {'type': 'string'} for i in ids}, 'required': ids}
    messages = [{'role': 'system', 'content': explain_system},
                {'role': 'user', 'content': f"Search: {query}\nCandidates:\n{candidates}"}]
    # real explanations ran 174-191 tokens, 220 truncated some
    # keyed on the prompt inputs so the same search shares its explanations
    return llm.cached(llm.make_key('explain', query, ids),
                      lambda: llm.chat(messages, max_tokens=320, schema=schema, temperature=0.3))

def search(query: str, k: int = 20, explain: bool = False, rewrite: bool = False):
    _, retrieval = index()
    weights = default_weights
    llm_calls = []

    rewritten = None
    if rewrite:
        (rewritten, was_cached), (picked, pick_cached) = rewrite_and_pick(query)
        if not isinstance(rewritten, dict):
            rewritten = None
        llm_calls.append((rewritten is not None, was_cached))
        if picked is not None:
            llm_calls.append((True, pick_cached))
        # the llm's picks replace the embedding mapping, adding both scored lower on val (+0.0242 against +0.0414)
        # when the pick fails, search falls back to the embedding mapping
        mapped = [(t, None) for t in picked] if picked is not None else map_query(query)
    else:
        mapped = map_query(query)
    bm25_text = f"{query} {' '.join(rewritten['keywords'])}" if rewritten else query
    bm25_text = expand(bm25_text, mapped)
    years = sorted(y for y in [rewritten.get('year_from'), rewritten.get('year_to')] if isinstance(y, int)) if rewritten else []
    year_range = (years[0], years[-1]) if years else None

    # boost the whole candidate pool and cut to k after, as the eval does, a pool cut at k would drop candidates the boost lifts
    pool = max(k*3, candidate_depth)
    results = retrieval.hybrid_search(query, pool, weights=weights, bm25_text=bm25_text, year_range=year_range)
    if year_range and len(results) < k:
        # too few movies in the era, top up from the unfiltered search
        seen = {item['item_id'] for item in results}
        extra = retrieval.hybrid_search(query, pool, weights=weights, bm25_text=bm25_text)
        results += [item for item in extra if item['item_id'] not in seen]

    # the rewrite's genres aren't boosted: on the eval the boost cost natural queries about 0.005 (eval_rewrite.py in 7c7c2b7)
    results = boost_era(results, weights, year_range)[:k]
    for item in results:
        reason = []
        if item['vector_rank']:
            reason.append(f"Semantic match for '{query}'")
        if item['bm25_rank']:
            reason.append(f"Keyword match for '{query}'")
        if item['era_boost']:
            reason.append(f"Released in the era you searched for ({year_range[0]}-{year_range[1]})")
        item['reason'] = reason
        if explain:
            item['explain'] = {'vector_rank': item['vector_rank'], 'bm25_rank': item['bm25_rank'],
                               'era_boost': item['era_boost']}

    if explain and results:
        top = results[:explain_top]
        reasons, was_cached = explain_items(query, top)
        if not isinstance(reasons, dict):
            reasons = None
        llm_calls.append((reasons is not None, was_cached))
        # falls back to the template reasons when the llm is busy or down
        if reasons:
            for item in top:
                if reasons.get(item['item_id']):
                    item['reason'] = [reasons[item['item_id']]]

    return {
        'items': results,
        'rewritten_query': rewritten,
        'mapped_tags': [t for t, _ in mapped],
        'llm_used': any(used for used, _ in llm_calls),
        'llm_cached': bool(llm_calls) and all(used and cached for used, cached in llm_calls)
    }
