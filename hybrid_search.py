from sklearn.neighbors import KNeighborsClassifier
from rank_bm25 import BM25Okapi
from gen_embeds import collection

class Retrieval:
    def knn_search(self, query_vector, k):
        # chromadb implementation
        collection.query(
            query_vector=query_vector,
            n_results=k,
        )

    def bm25_rank(self, query_text, k):
        # rank_bm25 implementation
        pass

    def hybrid_search(self, query_vector, query_text, k):
        # knn_search + bm25_rank hybrid retrieval logic
        pass 