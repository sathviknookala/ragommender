from gen_embeds import create_collection
from get_user_profile import movie_file, tags_file
import numpy as np
import pandas as pd
import pickle
import sys
import time

# builds the offline eval set: movielens tag applications become (user, query, relevant movies) triples
seed = 17
test_users_n = 1500
queries_per_user = 2
# only tags many users share, so queries look like a vocabulary people search with
min_tag_users = 20
max_relevant = 20
min_ratings = 20
queries_file = 'movie-info/eval_queries.pkl'
eval_collection = 'eval_db'
eval_bm25_file = 'bm25/eval_bm25.pkl'
eval_movieIds_file = 'movie-info/eval_movieIds.pkl'

def pick_test_users(rng, ratings):
    # the held out users, shared with build_popularity.py so the eval's popularity excludes them
    tags = tags_file.dropna(subset=['tag']).copy()
    tags['query'] = tags['tag'].astype(str).str.strip().str.lower()
    tag_users = tags.groupby('query')['userId'].nunique()
    tags = tags[tags['query'].isin(tag_users[tag_users >= min_tag_users].index)]

    rating_counts = ratings.groupby('userId').size()
    candidates = np.array(sorted(set(tags['userId']) & set(rating_counts[rating_counts >= min_ratings].index)))
    test_users = set(rng.choice(candidates, size=min(test_users_n, len(candidates)), replace=False).tolist())
    return tags, test_users

def read_ratings():
    return pd.read_csv('movie-info/ratings.csv', dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32', 'timestamp': 'int64'})

def build_queries(rng):
    ratings = read_ratings()
    tags, test_users = pick_test_users(rng, ratings)

    pairs = (tags[tags['userId'].isin(test_users)]
             .groupby(['userId', 'query'])
             .agg(movies=('movieId', lambda m: sorted(set(m))), time=('timestamp', 'min'))
             .reset_index())
    pairs = pairs[pairs['movies'].map(len) <= max_relevant]
    ratings = ratings[ratings['userId'].isin(test_users)].sort_values('timestamp')
    ratings_by_user = dict(tuple(ratings.groupby('userId')))

    queries = []
    for user, group in pairs.groupby('userId'):
        picked = group.sample(n=min(queries_per_user, len(group)), random_state=int(rng.integers(1 << 30)))
        history_all = ratings_by_user.get(user)
        for row in picked.itertuples():
            relevant = set(row.movies)
            # only ratings made before the tag, and never the movies being searched for
            history = history_all[(history_all['timestamp'] < row.time) & ~history_all['movieId'].isin(relevant)]
            likes = history[history['rating'] >= 4]['movieId'].tolist()[-50:]
            dislikes = history[history['rating'] <= 2]['movieId'].tolist()[-50:]
            swipes = ([{'item_id': str(m), 'direction': 'like'} for m in likes] +
                      [{'item_id': str(m), 'direction': 'dislike'} for m in dislikes])
            queries.append({
                'user': int(user),
                'query': row.query,
                'relevant': [str(m) for m in row.movies],
                'time': int(row.time),
                'swipes': swipes,
                'n_history': int(((history['rating'] >= 4) | (history['rating'] <= 2)).sum()),
                # split by user so validation and test never share a person
                'split': 'val' if user % 2 == 0 else 'test'
            })
    return queries, test_users

if __name__ == '__main__':
    start = time.time()
    rng = np.random.default_rng(seed)
    queries, test_users = build_queries(rng)
    config = {'seed': seed, 'test_users': len(test_users), 'min_tag_users': min_tag_users,
              'max_relevant': max_relevant, 'min_ratings': min_ratings, 'queries_per_user': queries_per_user}
    with open(queries_file, 'wb') as f:
        pickle.dump({'config': config, 'queries': queries}, f)
    print(f"{len(queries)} queries from {len(test_users)} test users, {time.time()-start:.0f}s")

    # --clean-embed=N builds eval_db_cleanN, embedding only the top N tags, bm25 is unchanged so its pickle is reused
    clean = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--clean-embed=')), None)
    if '--queries-only' not in sys.argv:
        # the eval index leaves out every tag the test users wrote, so a query can't match its own tag
        held_out_tags = tags_file[~tags_file['userId'].isin(test_users)]
        print(f"indexing with {len(held_out_tags)} of {len(tags_file)} tag applications")
        name = f'{eval_collection}_clean{clean}' if clean else eval_collection
        collection, bm25_index, movieIds = create_collection(name, movie_file, held_out_tags, len(movie_file), embed_top_tags=clean)
        if not clean:
            with open(eval_bm25_file, 'wb') as f:
                pickle.dump(bm25_index, f)
            with open(eval_movieIds_file, 'wb') as f:
                pickle.dump(movieIds, f)
    print(f"Time taken: {time.time()-start:.0f}s")
