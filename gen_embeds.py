from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from get_user_profile import movie_file, tags_file, id_file
import chromadb
import pandas as pd
import spacy
import pickle
import time
import torch
import math
start_time = time.time()

# cuda dependent 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device being used: {device}")
spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2',device=device)
print('Model loaded successfully')

def create_collection(collection_name: str, movie_df: pd.DataFrame, tags_df: pd.DataFrame, k: int):
    '''
    Initializes chromadb collection and bm25 corpus for hybrid search
    '''
    description_list = {}
    bm25_corpus = []
    movieIds = []

    for movie in movie_df[:k].itertuples():
        movieId = movie.movieId
        tags = tags_df[tags_df['movieId'] == movieId]
        tag_text = ''.join(str(tags['tag'].tolist()))    

        description = f"movie: {movie.title}. genre: {movie.genres}. tags: {tag_text}"
        description_list[movie.title] = description

        bm25_text = f"{movie.title} {movie.genres} {tag_text}"        

        doc = nlp(bm25_text.lower())
        tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
        bm25_corpus.append(tokens) 
        movieIds.append(movie)
    bm25_index = BM25Okapi(bm25_text)
    descriptions_for_embedding = list(description_list.values())

    embeddings = model.encode(
        descriptions_for_embedding,    
        normalize_embeddings=True,
        batch_size=512,
        show_progress_bar=True,
        convert_to_tensor=True
        ).tolist()

    client = chromadb.PersistentClient()

    try:
        collection = client.create_collection(
            name=collection_name,
            configuration={
                'hnsw': {
                    'space': 'cosine',
                    'max_neighbors': 16,
                    'ef_construction': 200,
                    'ef_search': 100,
                } 
            }
        )
        print('Creating collection')
    except:
        collection = client.get_collection('rag_db') 

    def batches(collection, embeddings, documents, ids, batch_size=5400):
        for index, i in enumerate(range(0, len(embeddings), batch_size), 1):
            batch_end = min(len(embeddings), i+batch_size)

            batch_embeddings = embeddings[i:batch_end]
            batch_documents = documents[i:batch_end]
            batch_ids = ids[i:batch_end]

            collection.add(
                embeddings=batch_embeddings,
                documents=batch_documents,
                ids=batch_ids
            )
            print(f"Added batch {index} of {math.ceil(len(embeddings)/batch_size)}")

    if collection.count() == 0:
        batches(collection, embeddings, descriptions_for_embedding, list(description_list.keys()))
    else:
        print('Collection has data')

    return collection, bm25_index, movieIds        

collection, bm25_index, movieIds = create_collection('rag_db', movie_file, tags_file, 10000)
with open('bm25/bm25_data.pkl', 'wb') as f:
    pickle.dump(bm25_index, f)

end_time = time.time()
print(f"Time taken: {end_time-start_time}")
