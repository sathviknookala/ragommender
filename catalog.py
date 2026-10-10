import pandas as pd

# the movielens catalog shared by ingestion (gen_embeds.py), the eval build and the rewrite's genre list
movie_file = pd.read_csv('movie-info/movies.csv', sep=',', encoding='utf-8')
tags_file = pd.read_csv('movie-info/tags.csv', sep=',', encoding='utf-8')
# movieId -> genre list
genres_by_id = {row.movieId: row.genres.split('|') for row in movie_file.itertuples()}
