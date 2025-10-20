from sklearn.neighbors import KNeighborsClassifier
from rank_bm25 import BM25Okapi
import chromadb
import spacy
import sys
import numpy as np
import pickle

class Retrieval:
    def __init__(self, collection, bm25_filepath, movieIds_filepath):
        self.collection = collection
        with open(bm25_filepath, 'rb') as f:
            self.bm25_data = pickle.load(f)
        with open(movieIds_filepath, 'rb') as f:
            self.movieIds = pickle.load(f)

    def knn_search(self, query_vector, k=5):
        # chromadb implementation
        response = self.collection.query(
            query_texts=query_vector,
            n_results=k,
            include=['distances', 'documents']
        )
        ids = response['ids']
        return response, [self.movieIds[int(id)] for id in ids[0]]        

    def bm25_rank(self, query_text, k=5):
        # rank_bm25 implementation
        doc = self.nlp(query_text.lower())
        query_tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]

        if not query_tokens:
            return []
        
        scores = self.bm25_data.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            movie = self.movie_ids[idx]

    def hybrid_search(self, query_vector, query_text, k):
        # knn_search + bm25_rank hybrid retrieval logic
        pass 

if __name__ == "__main__":
    client = chromadb.PersistentClient()
    cName = sys.argv[1]
    collection = client.get_collection(cName)
    retrieval = Retrieval(collection, 'bm25/bm25_data.pkl', 'movie-info/movieIds.pkl')

    response, movies = retrieval.knn_search(['james bond movies'], 1)
    print(f"Horror movies: {movies}") 
    print(response['documents'])

