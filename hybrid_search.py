from sklearn.neighbors import KNeighborsClassifier
from rank_bm25 import BM25Okapi
import chromadb
import spacy
import numpy as np
import pickle

class Retrieval:
    def __init__(self, collection, bm25_filepath):
        self.collection = collection
        with open(bm25_filepath, 'rb') as f:
            self.bm25_index = pickle.load(f)

    def knn_search(self, query_vector, k=5):
        # chromadb implementation
        return self.collection.query(
            query_texts=query_vector,
            n_results=k,
            include=['distances']
        )

    def bm25_rank(self, query_text, k=5):
        # rank_bm25 implementation
        doc = self.nlp(query_text.lower())
        query_tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]

        if not query_tokens:
            return []
        
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            movie = self.movie_ids[idx]

    def hybrid_search(self, query_vector, query_text, k):
        # knn_search + bm25_rank hybrid retrieval logic
        pass 

if __name__ == "__main__":
    client = chromadb.PersistentClient()
    collection = client.get_collection('rag_db')
    retrieval = Retrieval(collection, 'bm25bm25_data.pkl')

    query = retrieval.knn_search(['horror movies'])['ids']
    print(f"Horror movies: {query}") 

    query2 = retrieval.knn_search(['sad heartfelt rom coms'])['ids']
    print(f"Sad heartfelt rom coms: {query2}") 
