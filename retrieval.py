from hybrid_search import Retrieval
from preferences import record_swipe, get_user_preferences, genres_by_id
from survey import get_survey_movies
import chromadb
import os
import time

cName = os.environ.get('CHROMA_COLLECTION', 'rag_db')
client = chromadb.PersistentClient()
collection = client.get_collection(cName)
retrieval = Retrieval(collection, 'bm25/bm25_data.pkl', 'movie-info/movieIds.pkl')

pref_weight = 0.3
genre_boost = 0.002
min_swipes = 5

def get_learned_weights(swipe_count: int):
    # fixed for now, rrf weights vector and bm25 ranks equally
    return {
        'vector': 0.5,
        'bm25': 0.5,
        'preference': pref_weight if swipe_count >= min_swipes else 0.0,
        'genre_boost': genre_boost
    }

def search(user_id: str, query: str, k: int = 20, explain: bool = False):
    prefs = get_user_preferences(collection, user_id)
    weights = get_learned_weights(prefs['swipe_count'])
    # cold start, only use the preference vector after min_swipes
    vector = prefs['preference_vector'] if weights['preference'] > 0 else None
    results = retrieval.hybrid_search(query, k, preference_vector=vector, pref_weight=pref_weight)

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
        if explain:
            item['explain'] = {'base_score': item['score'], 'vector_rank': item['vector_rank'],
                               'bm25_rank': item['bm25_rank'], 'used_preference_vector': vector is not None}
        item['preference_boost'] = boost
        item['score'] += boost
        item['reason'] = reason
    results.sort(key=lambda x: x['score'], reverse=True)
    return {'items': results, 'preference_confidence': prefs['confidence'], 'learned_weights': weights}

def get_user_profile(user_id: str):
    prefs = get_user_preferences(collection, user_id)
    top_genres = sorted(prefs['genre_preferences'], key=prefs['genre_preferences'].get, reverse=True)
    return {
        'user_id': user_id,
        'swipe_count': prefs['swipe_count'],
        'preference_confidence': prefs['confidence'],
        'learned_weights': get_learned_weights(prefs['swipe_count']),
        'top_genres': [genre for genre in top_genres if prefs['genre_preferences'][genre] > 0][:5]
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
