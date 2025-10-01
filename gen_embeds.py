from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from get_user_profile import movie_file, tags_file, id_file
import chromadb
import pandas as pd
import spacy
import time
import torch
import math
start_time = time.time()
nlp = spacy.load("en_core_web_sm")

print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def create_corpus(movie_df: pd.DataFrame, tags_df: pd.DataFrame, k: int):
    '''
    Initializes bm25 indexing engine
    ''' 
    corpus = []
    for movie in movie_df[:k].itertuples():
        movieId = movie.movieId


def create_collection(collection_name: str, movie_df: pd.DataFrame, tags_df: pd.DataFrame, k: int):
    '''
    Initializes chromadb collection for KNN search
    '''
    description_list = {}
    for movie in movie_df[:k].itertuples():
        movieId = movie.movieId
        tags = tags_df[tags_df['movieId'] == movieId]
        tag_text = ''.join(str(tags['tag'].tolist()))    

        description = f"movie: {movie.title}. genre: {movie.genres}. tags: {tag_text}"
        description_list[movie.title] = description
    descriptions_for_embedding = list(description_list.values())

    model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda')
    print('Model loaded successfully')

    embeddings = model.encode(
        descriptions_for_embedding,    
        normalize_embeddings=True,
        batch_size=384 
        ).tolist()

    client = chromadb.PersistentClient()


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
        batches(collection, embeddings, descriptions_for_embedding, list(description_list.keys()))
    else:
        print('Collection has data')

    return collection        

collection = create_collection('rag_db', movie_file, tags_file, 10000)

end_time = time.time()
print(f"Time taken: {end_time-start_time}")
