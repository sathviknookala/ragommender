from sentence_transformers import SentenceTransformer
import pandas as pd
from get_user_profile import movie_file, tags_file, id_file
import chromadb
import time
start_time = time.time()

description_list = []
for i, movie in movie_file[:100].iterrows():
    movieId = movie['movieId']
    tags = tags_file[tags_file['movieId'] == movieId]
    tag_text = ''.join(str(tags['tag'].tolist()))    

    description = f"movie: {movie['title']}. genre: {movie['genres']}. tags: {tag_text}"
    description_list.append(description)

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
print('Model loaded successfully')

embeddings = model.encode(
    description_list,
    normalize_embeddings=True,
    batch_size=384 
    )

client = chromadb.Client()
collection = client.create_collection('movies')

collection.add(
    embeddings=embeddings.tolist(),
    documents=description_list,
    ids=[str(movie['movieId']) for _, movie in movie_file[:100].iterrows()]
)

first_query = collection.query(
    query_texts=['james bond', '007', 'spy'],
    n_results=10,
    include=['documents']    
)

for k, v in first_query.items():
    print(f"key: {k} | value: {v}")

end_time = time.time()
print(f"Time taken: {end_time-start_time}")
