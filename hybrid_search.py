from sklearn.neighbors import KNeighborsClassifier
from rank_bm25 import BM25Okapi
import chromadb

class Retrieval:
    def __init__(self, collection):
        self.collection = collection

    def knn_search(self, query_vector, k=5):
        # chromadb implementation
        return self.collection.query(
            query_texts=query_vector,
            n_results=k,
            include=['distances']
        )

    def bm25_rank(self, query_text, k):
        # rank_bm25 implementation
        pass

    def hybrid_search(self, query_vector, query_text, k):
        # knn_search + bm25_rank hybrid retrieval logic
        pass 

if __name__ == "__main__":
    client = chromadb.PersistentClient()
    collection = client.get_collection('rag_db')
    retrieval = Retrieval(collection)

    query = retrieval.knn_search(['horror movies'])
    print(query)