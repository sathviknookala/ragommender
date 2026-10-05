from sentence_transformers import SentenceTransformer
import chromadb
import spacy
import os
import sys
import torch
import numpy as np
import pickle
import re

# hand-picked ranking weights, phase 1 measures them and phase 2 tunes them
default_weights = {
    'vector': 1.0,
    'bm25': 1.0,
    'rrf_k': 60,
    'preference': 0.3,
    'genre_boost': 0.002,
    'year_boost': 0.005,
    'min_swipes': 5
}

def fuse(knn_results, bm25_results, weights):
    # weighted reciprocal rank fusion, a zero weight drops that retriever's list
    fused = {}
    for name, results in [('vector', knn_results), ('bm25', bm25_results)]:
        if weights[name] == 0:
            continue
        for rank, (movieId, title, _) in enumerate(results, 1):
            item = fused.setdefault(movieId, {'item_id': str(movieId), 'title': title, 'score': 0.0,
                                              'vector_rank': None, 'bm25_rank': None})
            item['score'] += weights[name]/(weights['rrf_k']+rank)
            item[f'{name}_rank'] = rank
    return sorted(fused.values(), key=lambda x: x['score'], reverse=True)

class Retrieval:
    def __init__(self, collection, bm25_filepath, movieIds_filepath):
        self.collection = collection
        with open(bm25_filepath, 'rb') as f:
            self.bm25_data = pickle.load(f)
        with open(movieIds_filepath, 'rb') as f:
            self.movieIds = pickle.load(f)
        # bm25 corpus position i -> movieId
        self.idList = list(self.movieIds.keys())
        # bm25 corpus position i -> release year, 0 when the title has none
        self.years = np.array([int(m.group(1)) if (m := re.search(r'\((\d{4})\)\s*$', self.movieIds[i])) else 0
                               for i in self.idList])
        self.nlp = spacy.load('en_core_web_sm')
        device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def knn_search(self, query_vector=None, k=5, query_embeddings=None, where=None):
        # chromadb implementation, query_vector is query text, query_embeddings is a vector
        if query_embeddings is not None:
            response = self.collection.query(
                query_embeddings=[list(map(float, query_embeddings))],
                n_results=k,
                where=where,
                include=['distances', 'documents']
            )
        else:
            response = self.collection.query(
                query_texts=query_vector,
                n_results=k,
                where=where,
                include=['distances', 'documents']
            )
        ids = response['ids']
        distances = response['distances']
        results = [(int(id), self.movieIds[int(id)], dist) for id, dist in zip(ids[0], distances[0])]
        return response, results

    def bm25_rank(self, query_text, k=5, year_range=None):
        # rank_bm25 implementation
        doc = self.nlp(query_text.lower())
        query_tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]

        if not query_tokens:
            return []

        scores = self.bm25_data.get_scores(query_tokens)
        if year_range:
            scores[(self.years < year_range[0]) | (self.years > year_range[1])] = 0
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                movieId = self.idList[idx]
                results.append((movieId, self.movieIds[movieId], float(scores[idx])))
        return results

    def blend(self, query_text, preference_vector=None, pref_weight=0.0):
        query_vec = self.model.encode(query_text, normalize_embeddings=True)
        if preference_vector is not None and pref_weight > 0:
            query_vec = (1 - pref_weight) * query_vec + pref_weight * np.asarray(preference_vector)
            query_vec = query_vec / np.linalg.norm(query_vec)
        return query_vec

    def hybrid_search(self, query_text, k=10, preference_vector=None, weights=None, bm25_text=None, year_range=None):
        # knn_search + bm25_rank fused with weighted reciprocal rank fusion
        weights = weights or default_weights
        query_vec = self.blend(query_text, preference_vector, weights['preference'])

        where = {'$and': [{'year': {'$gte': year_range[0]}}, {'year': {'$lte': year_range[1]}}]} if year_range else None
        _, knn_results = self.knn_search(k=k*3, query_embeddings=query_vec, where=where)
        bm25_results = self.bm25_rank(bm25_text or query_text, k*3, year_range=year_range)
        return fuse(knn_results, bm25_results, weights)[:k]

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
