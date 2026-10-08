from sentence_transformers import SentenceTransformer
import chromadb
import spacy
import os
import sys
import torch
import numpy as np
import pickle
import re

# phase 3 picked vector, preference, popularity, pref_sim and genre_boost on the eval with the qwen3 index and the
# bm25 parameters below (eval_embed.py), the rest are hand-picked
default_weights = {
    # 0 would skip the knn query, minilm's knn lost to keyword only search, qwen3's earns a small weight
    'vector': 0.25,
    'bm25': 1.0,
    'rrf_k': 60,
    'preference': 0.0,
    'genre_boost': 0.002,
    'year_boost': 0.005,
    'min_swipes': 5,
    # added per candidate, scaled popularity (0..1) and cosine to the preference vector
    'popularity': 0.01,
    'pref_sim': 0.02
}

# candidates fetched per list, popularity, pref_sim and the boosts re-sort the whole pool so it must match the eval's
candidate_depth = 100

# bm25 k1 and b, tuned in phase 3 (eval_bm25.py). rank_bm25's defaults, which the pickles are built with, are 1.5 and
# 0.75. a low b barely normalizes length, and indexed text grows with tag count, so it leans towards popular movies
bm25_params = {'k1': 3.0, 'b': 0.1}

# a collection names its embedding model in its metadata, collections without one were built with minilm
default_embed_model = 'all-MiniLM-L6-v2'
# qwen3 embeddings put an instruction before queries, documents are embedded without one
query_prompts = {'Qwen/Qwen3-Embedding-0.6B': 'Instruct: Given a movie search, retrieve movies that match it\nQuery:'}

def fuse(knn_results, bm25_results, weights, popularity=None, pref_sims=None):
    # weighted reciprocal rank fusion, a zero weight drops that retriever's list
    # popularity and pref_sims are movieId -> value dicts, added on top of the fused score
    fused = {}
    for name, results in [('vector', knn_results), ('bm25', bm25_results)]:
        if weights[name] == 0:
            continue
        for rank, (movieId, title, _) in enumerate(results, 1):
            item = fused.setdefault(movieId, {'item_id': str(movieId), 'title': title, 'score': 0.0,
                                              'vector_rank': None, 'bm25_rank': None})
            item['score'] += weights[name]/(weights['rrf_k']+rank)
            item[f'{name}_rank'] = rank
    for movieId, item in fused.items():
        if popularity and weights['popularity']:
            item['score'] += weights['popularity'] * popularity.get(movieId, 0.0)
        if pref_sims and weights['pref_sim']:
            item['score'] += weights['pref_sim'] * pref_sims.get(movieId, 0.0)
    return sorted(fused.values(), key=lambda x: x['score'], reverse=True)

def with_fallback(weights, bm25_results):
    # with knn off, a query bm25 can't match (only stopwords, numbers, typos) would return nothing, so knn fills in
    if not weights['vector'] and not bm25_results:
        return dict(weights, vector=1.0)
    return weights

class Retrieval:
    def __init__(self, collection, bm25_filepath, movieIds_filepath, popularity_filepath=None, popularity_key='all',
                 bm25_params=bm25_params):
        self.collection = collection
        # movieId -> scaled log rating count from build_popularity.py, empty when the file isn't there
        self.popularity = {}
        if popularity_filepath and os.path.exists(popularity_filepath):
            with open(popularity_filepath, 'rb') as f:
                self.popularity = pickle.load(f)[popularity_key]
        elif popularity_filepath:
            print(f"warning: {popularity_filepath} not found, run build_popularity.py, popularity weight has no effect")
        with open(bm25_filepath, 'rb') as f:
            self.bm25_data = pickle.load(f)
        # get_scores reads k1 and b off the index, and the idf it stores doesn't depend on them
        self.bm25_data.k1, self.bm25_data.b = bm25_params['k1'], bm25_params['b']
        with open(movieIds_filepath, 'rb') as f:
            self.movieIds = pickle.load(f)
        # bm25 corpus position i -> movieId
        self.idList = list(self.movieIds.keys())
        # bm25 corpus position i -> release year, 0 when the title has none
        self.years = np.array([int(m.group(1)) if (m := re.search(r'\((\d{4})\)\s*$', self.movieIds[i])) else 0
                               for i in self.idList])
        self.nlp = spacy.load('en_core_web_sm')
        device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        model_name = (collection.metadata or {}).get('embed_model', default_embed_model)
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prompt = query_prompts.get(model_name)

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
        query_vec = self.model.encode(query_text, prompt=self.query_prompt, normalize_embeddings=True)
        if preference_vector is not None and pref_weight > 0:
            query_vec = (1 - pref_weight) * query_vec + pref_weight * np.asarray(preference_vector)
            query_vec = query_vec / np.linalg.norm(query_vec)
        return query_vec

    def hybrid_search(self, query_text, k=10, preference_vector=None, weights=None, bm25_text=None, year_range=None):
        # knn_search + bm25_rank fused with weighted reciprocal rank fusion
        weights = weights or default_weights
        depth = max(k, candidate_depth)
        bm25_results = self.bm25_rank(bm25_text or query_text, depth, year_range=year_range)
        weights = with_fallback(weights, bm25_results)
        knn_results = []
        # fuse drops a zero weight list anyway, so don't pay for the query embedding and the knn call
        if weights['vector']:
            query_vec = self.blend(query_text, preference_vector, weights['preference'])
            where = {'$and': [{'year': {'$gte': year_range[0]}}, {'year': {'$lte': year_range[1]}}]} if year_range else None
            _, knn_results = self.knn_search(k=depth, query_embeddings=query_vec, where=where)
        pref_sims = None
        ids = sorted({m for m, _, _ in knn_results + bm25_results})
        if weights['pref_sim'] and preference_vector is not None and ids:
            # cosine of every candidate to the preference vector, the embeddings come from chroma
            got = self.collection.get(ids=[str(m) for m in ids], include=['embeddings'])
            sims = np.asarray(got['embeddings']) @ np.asarray(preference_vector)
            pref_sims = {int(i): float(s) for i, s in zip(got['ids'], sims)}
        return fuse(knn_results, bm25_results, weights, self.popularity, pref_sims)[:k]

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
