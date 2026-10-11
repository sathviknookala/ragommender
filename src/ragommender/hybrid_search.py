from sentence_transformers import SentenceTransformer
from ragommender.ranking import default_weights, candidate_depth, bm25_params, parse_year, fuse, with_fallback
import os
import torch
import numpy as np
import pickle
import spacy

# a collection names its embedding model in its metadata, collections without one were built with minilm
default_embed_model = 'all-MiniLM-L6-v2'
# qwen3 embeddings put an instruction before queries, documents are embedded without one
query_prompts = {'Qwen/Qwen3-Embedding-0.6B': 'Instruct: Given a movie search, retrieve movies that match it\nQuery:'}

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
        # term -> (corpus positions, term frequencies), so scoring a term touches only the documents that have it
        # rank_bm25's get_scores loops over every document in python for each term, tens of ms per term
        postings = {}
        for i, freqs in enumerate(self.bm25_data.doc_freqs):
            for term, tf in freqs.items():
                postings.setdefault(term, ([], []))
                postings[term][0].append(i)
                postings[term][1].append(tf)
        self.postings = {t: (np.array(i, dtype=np.int64), np.array(f, dtype=np.int64)) for t, (i, f) in postings.items()}
        self.doc_len = np.array(self.bm25_data.doc_len)
        with open(movieIds_filepath, 'rb') as f:
            self.movieIds = pickle.load(f)
        # bm25 corpus position i -> movieId
        self.idList = list(self.movieIds.keys())
        # bm25 corpus position i -> release year, 0 when the title has none
        self.years = np.array([parse_year(self.movieIds[i]) or 0 for i in self.idList])
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

        scores = self.bm25_scores(query_tokens)
        if year_range:
            scores[(self.years < year_range[0]) | (self.years > year_range[1])] = 0
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                movieId = self.idList[idx]
                results.append((movieId, self.movieIds[movieId], float(scores[idx])))
        return results

    def bm25_scores(self, query_tokens):
        # rank_bm25's BM25Okapi.get_scores over the postings, the same formula per document and the same order of
        # additions, so scores match it exactly (tests/test_integration.py)
        bm = self.bm25_data
        score = np.zeros(bm.corpus_size)
        for q in query_tokens:
            if q not in self.postings:
                continue
            idx, tf = self.postings[q]
            score[idx] += (bm.idf.get(q) or 0) * (tf * (bm.k1 + 1) / (tf + bm.k1 * (1 - bm.b + bm.b * self.doc_len[idx] / bm.avgdl)))
        return score

    def embed(self, query_text, prompt=None):
        # prompt overrides the model's query instruction, e.g. vocab.tag_prompt for tag lookups
        return self.model.encode(query_text, prompt=prompt or self.query_prompt, normalize_embeddings=True)

    def hybrid_search(self, query_text, k=10, weights=None, bm25_text=None, year_range=None):
        # knn_search + bm25_rank fused with weighted reciprocal rank fusion
        weights = weights or default_weights
        depth = max(k, candidate_depth)
        bm25_results = self.bm25_rank(bm25_text or query_text, depth, year_range=year_range)
        weights = with_fallback(weights, bm25_results)
        knn_results = []
        # fuse drops a zero weight list anyway, so don't pay for the query embedding and the knn call
        if weights['vector']:
            query_vec = self.embed(query_text)
            where = {'$and': [{'year': {'$gte': year_range[0]}}, {'year': {'$lte': year_range[1]}}]} if year_range else None
            _, knn_results = self.knn_search(k=depth, query_embeddings=query_vec, where=where)
        return fuse(knn_results, bm25_results, weights, self.popularity)[:k]
