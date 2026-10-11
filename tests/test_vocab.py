from ragommender.vocab import TagVocab, map_tags, expand
import numpy as np
import pickle
import pytest

@pytest.fixture
def vocab(tmp_path):
    emb = np.array([[1, 0], [0.8, 0.6], [0, 1], [0.6, 0.8]], dtype=np.float32)
    path = tmp_path / 'vocab.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'tags': ['feel-good', 'uplifting', 'gore', 'heartwarming'], 'users': np.array([500, 40, 300, 4]),
                     'embeddings': emb, 'embed_model': 'test'}, f)
    return TagVocab(path)

def test_nearest_orders_by_similarity(vocab):
    assert [t for t, _ in vocab.nearest([1, 0], 3)] == ['feel-good', 'uplifting', 'heartwarming']

def test_nearest_filters(vocab):
    assert [t for t, _ in vocab.nearest([1, 0], 3, min_sim=0.7)] == ['feel-good', 'uplifting']
    assert [t for t, _ in vocab.nearest([1, 0], 3, min_users=10)] == ['feel-good', 'uplifting', 'gore']
    assert vocab.nearest([1, 0], 0) == []

def test_expand():
    assert expand('uplifting movies', [('feel-good', 0.6), ('inspiring', 0.5)]) == 'uplifting movies feel-good inspiring'
    assert expand('uplifting movies', []) == 'uplifting movies'

def test_map_tags_skips_queries_that_are_tags(vocab):
    params = {'n': 2, 'min_sim': 0.0, 'min_users': 1, 'prompt': 'tag'}
    assert map_tags(vocab, 'Feel-Good ', [1, 0], params) == []
    assert [t for t, _ in map_tags(vocab, 'happy films', [1, 0], params)] == ['feel-good', 'uplifting']
    assert map_tags(vocab, 'happy films', [1, 0], dict(params, n=0)) == []
    assert map_tags(None, 'happy films', [1, 0], params) == []
