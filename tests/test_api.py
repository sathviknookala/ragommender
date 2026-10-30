from fastapi.testclient import TestClient
from ragommender.api.main import app
from ragommender import llm, retrieval
import pytest

# the index isn't loaded: TestClient without a with block skips the lifespan, and search is stubbed

@pytest.fixture
def client(monkeypatch):
    calls = []
    def search(query, k, explain, rewrite):
        calls.append((query, k, explain, rewrite))
        item = {'item_id': '608', 'title': 'Fargo (1996)', 'score': 0.025, 'reason': ["Keyword match for 'x'"]}
        return {'items': [item] * k, 'rewritten_query': None, 'mapped_tags': ['dark comedy'], 'llm_used': False, 'llm_cached': False}
    monkeypatch.setattr(retrieval, 'search', search)
    monkeypatch.setattr(retrieval, 'model_version', lambda: 'test_db/test-model')
    c = TestClient(app)
    c.calls = calls
    return c

def test_search_defaults_and_response_shape(client):
    r = client.post('/search', json={'query': 'dark comedy'})
    assert r.status_code == 200
    body = r.json()
    assert client.calls == [('dark comedy', 20, False, False)]
    assert len(body['items']) == 20
    assert body['items'][0] == {'item_id': '608', 'title': 'Fargo (1996)', 'score': 0.025,
                                'reason': ["Keyword match for 'x'"], 'explain': None}
    assert body['model_version'] == 'test_db/test-model'
    assert body['mapped_tags'] == ['dark comedy']
    assert set(body) == {'items', 'model_version', 'rewritten_query', 'mapped_tags', 'llm_used', 'llm_cached'}

def test_search_passes_options(client):
    assert client.post('/search', json={'query': 'heist', 'k': 3, 'explain': True, 'rewrite': True}).status_code == 200
    assert client.calls == [('heist', 3, True, True)]

@pytest.mark.parametrize('body', [{}, {'query': ''}, {'query': '   '}, {'query': 'x' * 501},
                                  {'query': 'x', 'k': 0}, {'query': 'x', 'k': 101}])
def test_search_rejects_bad_requests(client, body):
    assert client.post('/search', json=body).status_code == 422
    assert client.calls == []

def test_unknown_fields_are_ignored(client):
    assert client.post('/search', json={'query': 'x', 'user_id': 'u1'}).status_code == 200

@pytest.mark.parametrize('method, path', [('post', '/swipe'), ('post', '/survey/start'), ('get', '/user/u1/profile')])
def test_removed_endpoints(client, method, path):
    assert getattr(client, method)(path).status_code == 404

@pytest.mark.parametrize('enabled, down_until, state', [(True, 0.0, 'available'), (True, float('inf'), 'cooldown'),
                                                        (False, 0.0, 'disabled')])
def test_health(client, monkeypatch, enabled, down_until, state):
    monkeypatch.setattr(retrieval, 'vocab', lambda: object())
    monkeypatch.setattr(llm, 'enabled', enabled)
    monkeypatch.setattr(llm, 'down_until', down_until)
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'model_version': 'test_db/test-model', 'mapping': True, 'llm': state}
