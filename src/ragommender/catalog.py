from functools import cache
from ragommender.paths import movie_info
import pandas as pd

# the movielens catalog shared by ingestion, the eval build and the rewrite's genre list, read on first use

@cache
def movies():
    return pd.read_csv(movie_info / 'movies.csv', sep=',', encoding='utf-8')

@cache
def tags():
    return pd.read_csv(movie_info / 'tags.csv', sep=',', encoding='utf-8')

@cache
def genres_by_id():
    # movieId -> genre list
    return {row.movieId: row.genres.split('|') for row in movies().itertuples()}

def read_ratings():
    return pd.read_csv(movie_info / 'ratings.csv', dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32', 'timestamp': 'int64'})
