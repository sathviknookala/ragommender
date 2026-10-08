from get_user_profile import movie_file
from survey import parse_year
import numpy as np
import os
import pickle
import threading
import time

swipe_file = 'movie-info/swipes.pkl'
swipe_lock = threading.Lock()
# movieId -> genre list for the indexed movies
genres_by_id = {row.movieId: row.genres.split('|') for row in movie_file.itertuples()}

def load_swipes():
    if not os.path.exists(swipe_file):
        return {}
    with open(swipe_file, 'rb') as f:
        return pickle.load(f)

def record_swipe(swipe: dict):
    # local stand-in for the dynamodb user_swipes table
    swipe = dict(swipe)
    if not swipe.get('timestamp'):
        swipe['timestamp'] = int(time.time())
    # serialize the read-modify-write so concurrent swipes aren't lost
    with swipe_lock:
        swipes = load_swipes()
        user_swipes = swipes.setdefault(swipe['user_id'], [])
        user_swipes.append(swipe)
        # write then rename so readers never see a partial file
        with open(swipe_file + '.tmp', 'wb') as f:
            pickle.dump(swipes, f)
        os.replace(swipe_file + '.tmp', swipe_file)
        return len(user_swipes)

def get_swipes(user_id: str):
    return load_swipes().get(user_id, [])

def get_latest_swipes(user_id: str):
    # one swipe per movie, the latest wins, ordered by when it was last swiped
    latest = {}
    for s in get_swipes(user_id):
        latest.pop(s['item_id'], None)
        latest[s['item_id']] = s
    return list(latest.values())

def centroid(collection, ids):
    # 0 broadcasts against any embedding size
    if not ids:
        return 0.0
    embeddings = collection.get(ids=ids, include=['embeddings'])['embeddings']
    return np.mean(np.asarray(embeddings), axis=0)

def compute_preference_vector(collection, likes, dislikes):
    # likes and dislikes are lists of item ids
    if not likes:
        return None
    vector = centroid(collection, likes) - 0.5 * centroid(collection, dislikes)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm

def get_user_preferences(collection, user_id: str):
    return preferences_from_swipes(collection, get_latest_swipes(user_id), user_id)

def preferences_from_swipes(collection, swipes: list, user_id: str = None):
    # swipes are {item_id, direction} dicts, one per movie, shared by the api and the offline eval
    likes = [s['item_id'] for s in swipes if s['direction'] == 'like']
    dislikes = [s['item_id'] for s in swipes if s['direction'] == 'dislike']
    swipe_count = len(swipes)

    counts = {}
    for s in swipes:
        for genre in genres_by_id.get(int(s['item_id']), []):
            counts[genre] = counts.get(genre, 0) + (1 if s['direction'] == 'like' else -1)
    genre_preferences = {genre: score/swipe_count for genre, score in counts.items()} if swipe_count else {}

    vector = compute_preference_vector(collection, likes, dislikes)
    return {
        'user_id': user_id,
        'swipe_count': swipe_count,
        'preference_vector': vector,
        'genre_preferences': genre_preferences,
        'confidence': min(1, swipe_count/25)
    }

def apply_boosts(items: list, genre_preferences: dict, weights: dict, query_genres=(), year_range=None):
    '''
    Adds genre preference, searched genre and era boosts to fused items and re-sorts them
    '''
    for item in items:
        boost = 0.0
        reasons = []
        for genre in genres_by_id.get(int(item['item_id']), []):
            weight = genre_preferences.get(genre, 0)
            boost += weights['genre_boost'] * weight
            if weight > 0:
                reasons.append(f"Matches your {genre} preference")
            if genre in query_genres:
                boost += weights['genre_boost']
                reasons.append(f"Matches the {genre} genre you searched for")
        year = parse_year(item['title'])
        if year_range and year and year_range[0] <= year <= year_range[1]:
            boost += weights['year_boost']
            reasons.append(f"From {year}, in the era you searched for")
        item['base_score'] = item['score']
        item['preference_boost'] = boost
        item['score'] += boost
        item['boost_reasons'] = reasons
    return sorted(items, key=lambda x: x['score'], reverse=True)

if __name__ == '__main__':
    print(len(genres_by_id))
    print(get_swipes('nobody'))
