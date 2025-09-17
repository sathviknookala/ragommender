from sentence_transformers import SentenceTransformer
from get_user_profile import movie_file, tags_file, id_file
import chromadb
import sys
import time
start_time = time.time()

description_list = {}
for _, movie in movie_file.iterrows():
    movieId = movie['movieId']
    tags = tags_file[tags_file['movieId'] == movieId]
    tag_text = ''.join(str(tags['tag'].tolist()))    

    description = f"movie: {movie['title']}. genre: {movie['genres']}. tags: {tag_text}"
    description_list[movie['title']] = description
descriptions_for_embedding = list(description_list.values())

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
print('Model loaded successfully')

embeddings = model.encode(
    descriptions_for_embedding,    
    normalize_embeddings=True,
    batch_size=384 
    )

client = chromadb.PersistentClient()

try:
    collection = client.get_collection('movies')
except:
    collection = client.create_collection('movies')
    print('Creating collection')

if collection.count() == 0:
    collection.add(
        embeddings=embeddings.tolist(),
        documents=descriptions_for_embedding,
        ids=list(description_list.keys())
    )
else:
    print('Collection has data')

first_query = collection.query(
    query_texts=['james bond', '007', 'spy', 'action', 'guns'],
    n_results=10,
    include=['distances']    
)

for k, v in first_query.items():
    print(f"key: {k} | value: {v}")
end_time = time.time()
print(f"Time taken: {end_time-start_time}")
