from get_user_profile import movie_file
import random
import re

core_genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance']
indexed = movie_file

def parse_year(title):
    match = re.search(r'\((\d{4})\)\s*$', title)
    return int(match.group(1)) if match else None

def to_item(movie):
    return {
        'item_id': str(movie.movieId),
        'title': movie.title,
        'genres': movie.genres.split('|'),
        'year': parse_year(movie.title)
    }

def get_survey_movies(survey_size: int = 20):
    # spread picks round robin across core genres
    pools = {genre: indexed[indexed['genres'].str.contains(genre, regex=False)] for genre in core_genres}
    picked = {}
    i = 0
    while len(picked) < survey_size:
        genre = core_genres[i % len(core_genres)]
        movie = next(pools[genre].sample(1).itertuples())
        picked.setdefault(movie.movieId, to_item(movie))
        i += 1
    movies = list(picked.values())
    random.shuffle(movies)
    return movies

if __name__ == '__main__':
    for movie in get_survey_movies(8):
        print(movie)
