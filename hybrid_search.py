from sentence_transformers import SentenceTransformer
import chromadb
import spacy
import os
import sys
import torch
import numpy as np
import pickle

class Retrieval:
    def __init__(self, collection, bm25_filepath, movieIds_filepath):
        self.collection = collection
        with open(bm25_filepath, 'rb') as f:
            self.bm25_data = pickle.load(f)
        with open(movieIds_filepath, 'rb') as f:
            self.movieIds = pickle.load(f)
        # bm25 corpus position i -> movieId
        self.idList = list(self.movieIds.keys())
        self.nlp = spacy.load('en_core_web_sm')
        device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def knn_search(self, query_vector=None, k=5, query_embeddings=None):
        # chromadb implementation, query_vector is query text, query_embeddings is a vector
        if query_embeddings is not None:
            response = self.collection.query(
                query_embeddings=[list(map(float, query_embeddings))],
                n_results=k,
                include=['distances', 'documents']
            )
        else:
            response = self.collection.query(
                query_texts=query_vector,
                n_results=k,
                include=['distances', 'documents']
            )
        ids = response['ids']
        distances = response['distances']
        results = [(int(id), self.movieIds[int(id)], dist) for id, dist in zip(ids[0], distances[0])]
        return response, results

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
            if scores[idx] > 0:
                movieId = self.idList[idx]
                results.append((movieId, self.movieIds[movieId], float(scores[idx])))
        return results

    def hybrid_search(self, query_text, k=10, preference_vector=None, pref_weight=0.3):
        # knn_search + bm25_rank fused with reciprocal rank fusion
        query_vec = self.model.encode(query_text, normalize_embeddings=True)
        if preference_vector is not None:
            query_vec = (1 - pref_weight) * query_vec + pref_weight * np.asarray(preference_vector)
            query_vec = query_vec / np.linalg.norm(query_vec)

        _, knn_results = self.knn_search(k=k*3, query_embeddings=query_vec)
        bm25_results = self.bm25_rank(query_text, k*3)

        fused = {}
        for rank, (movieId, title, _) in enumerate(knn_results, 1):
            fused[movieId] = {'item_id': str(movieId), 'title': title, 'score': 1/(60+rank),
                              'vector_rank': rank, 'bm25_rank': None}
        for rank, (movieId, title, _) in enumerate(bm25_results, 1):
            if movieId not in fused:
                fused[movieId] = {'item_id': str(movieId), 'title': title, 'score': 0.0,
                                  'vector_rank': None, 'bm25_rank': None}
            fused[movieId]['score'] += 1/(60+rank)
            fused[movieId]['bm25_rank'] = rank

        return sorted(fused.values(), key=lambda x: x['score'], reverse=True)[:k]

if __name__ == "__main__":
    client = chromadb.PersistentClient()
    cName = sys.argv[1]
    collection = client.get_collection(cName)
    retrieval = Retrieval(collection, 'bm25/bm25_data.pkl', 'movie-info/movieIds.pkl')

    response, movies = retrieval.knn_search(['james bond movies'], 3)
    print(f"knn: {movies}")
    print(response['documents'][0][0][:100])

    print(f"bm25: {retrieval.bm25_rank('space adventure', 3)}")

    for r in retrieval.hybrid_search('funny sci-fi movies', 5):
        print(r)
    pref = retrieval.model.encode('Alien (1979) Horror Sci-Fi', normalize_embeddings=True)
    for r in retrieval.hybrid_search('funny sci-fi movies', 5, preference_vector=pref):
        print(r)
