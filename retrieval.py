from hybrid_search import Retrieval
from preferences import record_swipe, get_swipes, get_user_preferences, genres_by_id
from survey import get_survey_movies
import chromadb
import llm
import os
import time

cName = os.environ.get('CHROMA_COLLECTION', 'rag_db')
client = chromadb.PersistentClient()
collection = client.get_collection(cName)
retrieval = Retrieval(collection, 'bm25/bm25_data.pkl', 'movie-info/movieIds.pkl')

pref_weight = 0.3
genre_boost = 0.002
min_swipes = 5
explain_top = 5
all_genres = sorted({g for genres in genres_by_id.values() for g in genres} - {'(no genres listed)'})

explain_system = ("You explain movie recommendations. For each candidate write one sentence of at most 20 words "
                  "on why it fits the search and the user's taste. Reply as json {item_id: reason}.")
rewrite_system = ("Rewrite a movie search into json {keywords, genres}. keywords are short search terms "
                  "(titles, themes, people), genres must come from the allowed list.")
summary_system = "Summarize a viewer's movie taste in two sentences from the movies they liked and disliked."

def get_learned_weights(swipe_count: int):
    # fixed for now, rrf weights vector and bm25 ranks equally
    return {
        'vector': 0.5,
        'bm25': 0.5,
        'preference': pref_weight if swipe_count >= min_swipes else 0.0,
        'genre_boost': genre_boost
    }

def swiped_titles(user_id: str, direction: str, n: int = 10):
    swipes = [s for s in get_swipes(user_id) if s['direction'] == direction][-n:]
    return [retrieval.movieIds.get(int(s['item_id']), s['item_id']) for s in swipes]

def rewrite_query(query: str):
    schema = {
        'type': 'object',
        'properties': {
            'keywords': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 8},
            'genres': {'type': 'array', 'items': {'type': 'string', 'enum': all_genres}, 'maxItems': 3}
        },
        'required': ['keywords', 'genres']
    }
    messages = [{'role': 'system', 'content': f"{rewrite_system} Allowed genres: {', '.join(all_genres)}."},
                {'role': 'user', 'content': query}]
    # rewrites don't depend on the user so they share one cache entry per query
    return llm.cached(llm.make_key('rewrite', query.lower()),
                      lambda: llm.chat(messages, max_tokens=80, schema=schema, temperature=0.3))

def explain_items(user_id: str, query: str, items: list, swipe_count: int):
    ids = [item['item_id'] for item in items]
    docs = collection.get(ids=ids, include=['documents'])
    descriptions = dict(zip(docs['ids'], docs['documents']))
    candidates = '\n'.join(f"{i}: {descriptions.get(i, '')[:300]}" for i in ids)
    liked = ', '.join(swiped_titles(user_id, 'like', 5)) or 'unknown'
    schema = {'type': 'object', 'properties': {i: {'type': 'string'} for i in ids}, 'required': ids}
    messages = [{'role': 'system', 'content': explain_system},
                {'role': 'user', 'content': f"User likes: {liked}\nSearch: {query}\nCandidates:\n{candidates}"}]
    # real explanations ran 174-191 tokens, 220 truncated some
    return llm.cached(llm.make_key('explain', user_id, query, swipe_count, ids),
                      lambda: llm.chat(messages, max_tokens=320, schema=schema, temperature=0.3))

def taste_summary(user_id: str, swipe_count: int):
    if swipe_count == 0:
        return None, False
    liked = ', '.join(swiped_titles(user_id, 'like')) or 'none'
    disliked = ', '.join(swiped_titles(user_id, 'dislike')) or 'none'
    messages = [{'role': 'system', 'content': summary_system},
                {'role': 'user', 'content': f"Liked: {liked}\nDisliked: {disliked}"}]
    # profile reads use the background slots so they never starve searches
    return llm.cached(llm.make_key('summary', user_id, swipe_count),
                      lambda: llm.chat(messages, max_tokens=100, temperature=0.7, background=True))

def search(user_id: str, query: str, k: int = 20, explain: bool = False, rewrite: bool = False):
    prefs = get_user_preferences(collection, user_id)
    weights = get_learned_weights(prefs['swipe_count'])
    # cold start, only use the preference vector after min_swipes
    vector = prefs['preference_vector'] if weights['preference'] > 0 else None
    llm_calls = []

    rewritten = None
    if rewrite:
        rewritten, was_cached = rewrite_query(query)
        if not isinstance(rewritten, dict):
            rewritten = None
        llm_calls.append((rewritten is not None, was_cached))
    bm25_text = f"{query} {' '.join(rewritten['keywords'])}" if rewritten else None
    query_genres = set(rewritten['genres']) if rewritten else set()

    results = retrieval.hybrid_search(query, k, preference_vector=vector, pref_weight=pref_weight, bm25_text=bm25_text)

    for item in results:
        boost = 0.0
        reason = []
        if item['vector_rank']:
            reason.append(f"Semantic match for '{query}'")
        if item['bm25_rank']:
            reason.append(f"Keyword match for '{query}'")
        for genre in genres_by_id.get(int(item['item_id']), []):
            weight = prefs['genre_preferences'].get(genre, 0)
            boost += genre_boost * weight
            if weight > 0:
                reason.append(f"Matches your {genre} preference")
            if genre in query_genres:
                boost += genre_boost
                reason.append(f"Matches the {genre} genre you searched for")
        if explain:
            item['explain'] = {'base_score': item['score'], 'vector_rank': item['vector_rank'],
                               'bm25_rank': item['bm25_rank'], 'used_preference_vector': vector is not None}
        item['preference_boost'] = boost
        item['score'] += boost
        item['reason'] = reason
    results.sort(key=lambda x: x['score'], reverse=True)

    if explain and results:
        top = results[:explain_top]
        reasons, was_cached = explain_items(user_id, query, top, prefs['swipe_count'])
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
