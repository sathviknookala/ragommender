from ragommender.ranking import default_weights
from ragommender import paths
import pickle
import pytest

# needs the built indexes and runs the embedding model: pytest -m integration
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not paths.eval_bm25_file.exists() or not paths.labels_file.exists(),
                                 reason='eval index not built')]

def test_api_ranking_matches_eval_ranking():
    # the api's hybrid_search and the eval's rank must order the same pool the same way, or offline numbers
    # wouldn't describe what ships
    import chromadb
    from ragommender.hybrid_search import Retrieval
    from ragommender.evaluation.run import rank, shipped_collection
    collection = chromadb.PersistentClient(path=str(paths.chroma_dir)).get_collection(shipped_collection)
    retrieval = Retrieval(collection, paths.eval_bm25_file, paths.eval_movieIds_file, paths.popularity_file, popularity_key='eval')
    with open(paths.labels_file, 'rb') as f:
        tags = [q['tag'] for q in pickle.load(f)['queries']][:25]
    for tag in tags:
        api = [i['item_id'] for i in retrieval.hybrid_search(tag, 20, weights=default_weights)]
        knn = retrieval.knn_search(k=100, query_embeddings=retrieval.embed(tag))[1]
        bm25 = retrieval.bm25_rank(tag, 100)
        assert api == rank(knn, bm25, None, default_weights, retrieval.popularity)[:20], tag

def test_search_endpoint_on_the_shipped_index():
    from fastapi.testclient import TestClient
    from ragommender.api.main import app
    with TestClient(app) as client:
        body = client.post('/search', json={'query': 'dark comedy', 'k': 10}).json()
    assert len(body['items']) == 10
    assert body['model_version'].startswith('rag_db/')
    assert all(i['reason'] for i in body['items'])
