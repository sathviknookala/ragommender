from hybrid_search import Retrieval, default_weights, candidate_depth
from preferences import record_swipe, get_latest_swipes, get_user_preferences, apply_boosts, genres_by_id
from survey import get_survey_movies
import chromadb
import llm
import os
import time

cName = os.environ.get('CHROMA_COLLECTION', 'rag_db')
client = chromadb.PersistentClient()
collection = client.get_collection(cName)
retrieval = Retrieval(collection, 'bm25/bm25_data.pkl', 'movie-info/movieIds.pkl', 'movie-info/popularity.pkl')

explain_top = 5
all_genres = sorted({g for genres in genres_by_id.values() for g in genres} - {'(no genres listed)'})

explain_system = ("You explain movie recommendations. For each candidate write one sentence of at most 20 words. "
                  "When the candidate relates to a movie the user liked, name that movie and say what they share. "
                  "Otherwise point to the candidate's own genres, themes or setting. Never restate the search or say "
                  "it fits the search or the user's preference. Reply as json {item_id: reason}.")
rewrite_system = ("Rewrite a movie search into json {keywords, genres, year_from, year_to}. keywords are only "
                  "distinctive search terms: titles, themes, settings, people. Drop filler and comparative words such "
                  "as movie, film, something, like, funnier, better, and never put eras or years in keywords. Put an "
                  "era into year_from and year_to (90s is 1990 to 1999), otherwise use null for both. genres must "
                  "come from the allowed list.")
summary_system = "Summarize a viewer's movie taste in two sentences from the movies they liked and disliked."

def get_learned_weights(swipe_count: int):
    # the weights search actually ranks with, global defaults until per-user learning exists
    weights = dict(default_weights)
    # cold start, only blend in the preference vector after min_swipes
    if swipe_count < weights['min_swipes']:
        weights['preference'] = 0.0
        weights['pref_sim'] = 0.0
    return weights

def swiped_titles(user_id: str, direction: str, n: int = 10):
    swipes = [s for s in get_latest_swipes(user_id) if s['direction'] == direction][-n:]
    return [retrieval.movieIds.get(int(s['item_id']), s['item_id']) for s in swipes]

def rewrite_query(query: str):
    schema = {
        'type': 'object',
        'properties': {
            'keywords': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 8},
            'genres': {'type': 'array', 'items': {'type': 'string', 'enum': all_genres}, 'maxItems': 3},
            'year_from': {'type': ['integer', 'null']},
            'year_to': {'type': ['integer', 'null']}
        },
        'required': ['keywords', 'genres', 'year_from', 'year_to']
    }
    messages = [{'role': 'system', 'content': f"{rewrite_system} Allowed genres: {', '.join(all_genres)}."},
                {'role': 'user', 'content': query}]
    # rewrites don't depend on the user so they share one cache entry per query
    return llm.cached(llm.make_key('rewrite', query.lower()),
                      lambda: llm.chat(messages, max_tokens=100, schema=schema, temperature=0.3))

def explain_items(user_id: str, query: str, items: list):
    ids = [item['item_id'] for item in items]
    docs = collection.get(ids=ids, include=['documents'])
    descriptions = dict(zip(docs['ids'], docs['documents']))
    candidates = '\n'.join(f"{i}: {descriptions.get(i, '')[:300]}" for i in ids)
    liked = ', '.join(swiped_titles(user_id, 'like', 8)) or 'none yet'
    schema = {'type': 'object', 'properties': {i: {'type': 'string'} for i in ids}, 'required': ids}
    messages = [{'role': 'system', 'content': explain_system},
                {'role': 'user', 'content': f"User likes: {liked}\nSearch: {query}\nCandidates:\n{candidates}"}]
    # real explanations ran 174-191 tokens, 220 truncated some
    # keyed on the prompt inputs so users with the same likes share explanations
    return llm.cached(llm.make_key('explain', query, liked, ids),
                      lambda: llm.chat(messages, max_tokens=320, schema=schema, temperature=0.3))

def taste_summary(user_id: str, swipe_count: int):
    if swipe_count == 0:
        return None, False
    # sorted so re-swiping a movie doesn't change the prompt or the cache key
    liked = ', '.join(sorted(swiped_titles(user_id, 'like', 25))) or 'none'
    disliked = ', '.join(sorted(swiped_titles(user_id, 'dislike', 15))) or 'none'
    messages = [{'role': 'system', 'content': summary_system},
                {'role': 'user', 'content': f"Liked: {liked}\nDisliked: {disliked}"}]
    # profile reads use the background slots so they never starve searches
    # keyed on the titles actually sent, so it only regenerates when they change
    return llm.cached(llm.make_key('summary', liked, disliked),
                      lambda: llm.chat(messages, max_tokens=100, temperature=0.7, background=True))

def search(user_id: str, query: str, k: int = 20, explain: bool = False, rewrite: bool = False):
    prefs = get_user_preferences(collection, user_id)
    weights = get_learned_weights(prefs['swipe_count'])
    vector = prefs['preference_vector'] if weights['preference'] > 0 or weights['pref_sim'] > 0 else None
    llm_calls = []

    rewritten = None
    if rewrite:
        rewritten, was_cached = rewrite_query(query)
        if not isinstance(rewritten, dict):
            rewritten = None
        llm_calls.append((rewritten is not None, was_cached))
    bm25_text = f"{query} {' '.join(rewritten['keywords'])}" if rewritten else None
    query_genres = set(rewritten['genres']) if rewritten else set()
    years = sorted(y for y in [rewritten.get('year_from'), rewritten.get('year_to')] if isinstance(y, int)) if rewritten else []
    year_range = (years[0], years[-1]) if years else None

    # boost the whole candidate pool and cut to k after, as the eval does, a pool cut at k would drop candidates the boosts lift
    pool = max(k*3, candidate_depth)
    results = retrieval.hybrid_search(query, pool, preference_vector=vector, weights=weights,
                                      bm25_text=bm25_text, year_range=year_range)
    if year_range and len(results) < k:
        # too few movies in the era, top up from the unfiltered search
        seen = {item['item_id'] for item in results}
        extra = retrieval.hybrid_search(query, pool, preference_vector=vector, weights=weights, bm25_text=bm25_text)
        results += [item for item in extra if item['item_id'] not in seen]

    results = apply_boosts(results, prefs['genre_preferences'], weights, query_genres, year_range)[:k]
    for item in results:
        reason = []
        if item['vector_rank']:
            reason.append(f"Semantic match for '{query}'")
        if item['bm25_rank']:
            reason.append(f"Keyword match for '{query}'")
        item['reason'] = reason + item.pop('boost_reasons')
        base_score = item.pop('base_score')
        if explain:
            item['explain'] = {'base_score': base_score, 'vector_rank': item['vector_rank'],
                               'bm25_rank': item['bm25_rank'], 'used_preference_vector': vector is not None}

    if explain and results:
        top = results[:explain_top]
        reasons, was_cached = explain_items(user_id, query, top)
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
        'preference_confidence': prefs['confidence'],
        'learned_weights': weights,
        'rewritten_query': rewritten,
        'llm_used': any(used for used, _ in llm_calls),
        'llm_cached': bool(llm_calls) and all(used and cached for used, cached in llm_calls)
    }

def get_user_profile(user_id: str):
    prefs = get_user_preferences(collection, user_id)
    top_genres = sorted(prefs['genre_preferences'], key=prefs['genre_preferences'].get, reverse=True)
    summary, was_cached = taste_summary(user_id, prefs['swipe_count'])
    return {
        'user_id': user_id,
        'swipe_count': prefs['swipe_count'],
        'preference_confidence': prefs['confidence'],
        'learned_weights': get_learned_weights(prefs['swipe_count']),
        'top_genres': [genre for genre in top_genres if prefs['genre_preferences'][genre] > 0][:5],
        'taste_summary': summary,
        'llm_used': summary is not None,
        'llm_cached': was_cached and summary is not None
    }

def get_similar(user_id: str, k: int = 10):
    prefs = get_user_preferences(collection, user_id)
    if prefs['preference_vector'] is None:
        return {'similar_titles': [], 'cached': False}
    _, results = retrieval.knn_search(k=k, query_embeddings=prefs['preference_vector'])
    return {'similar_titles': [title for _, title, _ in results], 'cached': False}

def start_survey(user_id: str, survey_size: int = 20):
    return {
        'user_id': user_id,
        'movies': get_survey_movies(survey_size),
        'session_id': f'survey_{int(time.time())}'
    }

def swipe(event):
    swipe_count = record_swipe(event.model_dump())
    return {'status': 'recorded', 'preferences_updated': True, 'swipe_count': swipe_count}

if __name__ == '__main__':
    print(search('nobody', 'space adventure', 3))
