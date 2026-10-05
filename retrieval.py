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

def search(user_id: str, query: str, k: int = 20):
    prefs = get_user_preferences(collection, user_id)
    # cold start, only use the preference vector after 5 swipes
    vector = prefs['preference_vector'] if prefs['swipe_count'] >= 5 else None
    results = retrieval.hybrid_search(query, k, preference_vector=vector)

    for item in results:
        boost = 0.0
        for genre in genres_by_id.get(int(item['item_id']), []):
            boost += 0.002 * prefs['genre_preferences'].get(genre, 0)
        item['preference_boost'] = boost
        item['score'] += boost
    results.sort(key=lambda x: x['score'], reverse=True)
    return {'items': results, 'preference_confidence': prefs['confidence']}

def get_user_profile(user_id: str):
    prefs = get_user_preferences(collection, user_id)
    top_genres = sorted(prefs['genre_preferences'], key=prefs['genre_preferences'].get, reverse=True)
    return {
        'user_id': user_id,
        'swipe_count': prefs['swipe_count'],
        'preference_confidence': prefs['confidence'],
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
