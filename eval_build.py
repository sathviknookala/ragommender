from gen_embeds import create_collection, overviews_arg
from hybrid_search import default_embed_model
from catalog import movie_file, tags_file
import hashlib
import numpy as np
import pandas as pd
import pickle
import sys
import time

# builds the offline eval set: a label set where each query is a movielens tag and its relevant movies are the ones
# several held out users applied it to, and an eval index without those users' tags
# a movie is relevant to a tag when at least min_users held out users applied it, keeping tags with min_relevant such
# movies. the index only holds other users' tags (the crowd signal production has too), so a label never matches its
# own tag application. queries are per tag, not per user, the service matches queries and has no user state
seed = 17
test_users_n = 1500
# only tags many users share, so queries look like a vocabulary people search with
min_tag_users = 20
min_ratings = 20
min_users = 2
min_relevant = 5
labels_file = 'movie-info/eval_consensus.pkl'
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

def build_labels(rng):
    tags, test_users = pick_test_users(rng, read_ratings())
    held = tags[tags['userId'].isin(test_users)]
    users = held.groupby(['query', 'movieId'])['userId'].nunique()
    agreed = users[users >= min_users].reset_index()
    queries = []
    for tag, group in agreed.groupby('query'):
        if len(group) >= min_relevant:
            # split by tag, so val and test never share a query
            split = 'val' if int(hashlib.sha1(tag.encode()).hexdigest(), 16) % 2 == 0 else 'test'
            queries.append({'tag': tag, 'relevant': [str(m) for m in group['movieId']],
                            'users': {str(m): int(n) for m, n in zip(group['movieId'], group['userId'])}, 'split': split})
    return queries, test_users

if __name__ == '__main__':
    start = time.time()
    queries, test_users = build_labels(np.random.default_rng(seed))
    config = {'min_users': min_users, 'min_relevant': min_relevant, 'seed': seed, 'held_out_users': len(test_users)}
    # --clean-embed=N builds eval_db_cleanN, embedding only the top N tags, bm25 is unchanged so its pickle is reused
    clean = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--clean-embed=')), None)
    # --embed-model=NAME embeds with another model, e.g. Qwen/Qwen3-Embedding-0.6B, into eval_db_<model>[_cleanN]
    embed_model = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--embed-model=')), default_embed_model)
    # --overviews adds tmdb overviews to the embedded text, into eval_db..._ov, overviews hold no movielens tags so
    # they can't leak a held out query
    overviews = overviews_arg()
    variant = clean or embed_model != default_embed_model or overviews
    if variant:
        # a variant index is scored against the existing labels, so they are checked, not rewritten
        with open(labels_file, 'rb') as f:
            assert pickle.load(f)['queries'] == queries, f'rebuilt labels differ from {labels_file}'
    else:
        with open(labels_file, 'wb') as f:
            pickle.dump({'config': config, 'queries': queries}, f)
    n = [len(q['relevant']) for q in queries]
    print(f"{len(queries)} tag queries ({sum(q['split'] == 'val' for q in queries)} val) from {len(test_users)} held out "
          f"users, relevant per query median {int(np.median(n))}, max {max(n)}, {time.time()-start:.0f}s")

    if '--labels-only' not in sys.argv:
        # the eval index leaves out every tag the held out users wrote, so a query can't match its own tag
        held_out_tags = tags_file[~tags_file['userId'].isin(test_users)]
        print(f"indexing with {len(held_out_tags)} of {len(tags_file)} tag applications")
        name = (eval_collection + ('' if embed_model == default_embed_model else '_' + embed_model.split('/')[-1].lower())
                + (f'_clean{clean}' if clean else '') + ('_ov' if overviews else ''))
        collection, bm25_index, movieIds = create_collection(name, movie_file, held_out_tags, len(movie_file),
                                                             embed_top_tags=clean, embed_model=embed_model, overviews=overviews)
        if not variant:
            with open(eval_bm25_file, 'wb') as f:
                pickle.dump(bm25_index, f)
            with open(eval_movieIds_file, 'wb') as f:
                pickle.dump(movieIds, f)
    print(f"Time taken: {time.time()-start:.0f}s")
