import pandas as pd
user_file = pd.read_csv('movie-info/users.csv', sep=',', encoding='utf-8', skipinitialspace=True)
id_file = user_file.set_index('user_id')
tags_file = pd.read_csv('movie-info/tags.csv', sep=',', encoding='utf-8')
movie_file = pd.read_csv('movie-info/movies.csv', sep=',', encoding='utf-8')

def get_user_profile(user_id: str):
    try:
        user = id_file.loc[user_id]
        return {'user_id': user, 'content': [], 'cached': False}
    except Exception as e:
        return 'Error: {e}'
    
def similar(user_id: str):
    return {'similar_titles': [], 'cached': False}

def recommend(user_id: str):
    return {'recommendations': [], 'cached': False}

