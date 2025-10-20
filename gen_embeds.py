from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from get_user_profile import movie_file, tags_file
import chromadb
import pandas as pd
import spacy
import pickle
import time
import torch
import math
import sys
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
    movieIds = {}
    all_texts = []

    tags_grouped = tags_df.groupby('movieId')['tag'].apply(list).to_dict()
    for movie in movie_df[:k].itertuples():
        tags = tags_grouped.get(movie.movieId, [])
        clean_tags = [str(tag) for tag in tags if not pd.isna(tag) and tag!='']
        text = f"{movie.title} {movie.genres} {' '.join(clean_tags)}"

        all_texts.append(text)
        description_list[str(movie.movieId)] = text
        movieIds[movie.movieId] = movie.title

    docs = list(nlp.pipe(all_texts, batch_size=1000))
    tokens_list = [[token.text for token in doc if token.is_alpha and not token.is_stop]
                   for doc in docs]         
    bm25_index = BM25Okapi(tokens_list)

    embeddings = model.encode(
        all_texts,    
        normalize_embeddings=True,
        batch_size=512,
        show_progress_bar=True,
        convert_to_tensor=True
        ).tolist()

    client = chromadb.PersistentClient()
    try:
        client.delete_collection(name=collection_name)
    except:
        pass        

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
        batches(collection, embeddings, all_texts, list(description_list.keys()))
    else:
        print('Collection has data')

    return collection, bm25_index, movieIds        

if __name__ == '__main__':
    collection_name = sys.argv[1]
    collection, bm25_index, movieIds = create_collection(collection_name, movie_file, tags_file, 30000)
    with open('bm25/bm25_data.pkl', 'wb') as f:
        pickle.dump(bm25_index, f)
    with open('movie-info/movieIds.pkl', 'wb') as f:
        pickle.dump(movieIds, f)        

    end_time = time.time()
    print(f"Time taken: {end_time-start_time}")
