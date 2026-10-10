from ragommender import llm
import httpx
import pytest

@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(llm, 'enabled', True)
    monkeypatch.setattr(llm, 'down_until', 0.0)
    monkeypatch.setattr(llm, 'model', 'test-model')
    llm.cache.clear()
    yield
    llm.cache.clear()

def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(llm, 'enabled', False)
    assert llm.chat([{'role': 'user', 'content': 'hi'}]) is None

def test_server_down_returns_none_and_cools_down(monkeypatch):
    def post(*a, **kw):
        raise httpx.ConnectError('refused')
    monkeypatch.setattr(llm.client, 'post', post)
    assert llm.chat([{'role': 'user', 'content': 'hi'}]) is None
    assert not llm.is_up()

def test_json_reply_is_parsed_and_think_stripped(monkeypatch):
    request = httpx.Request('POST', 'http://test/chat/completions')
    reply = {'choices': [{'message': {'content': '<think>x</think> {"a": 1}'}}]}
    monkeypatch.setattr(llm.client, 'post', lambda *a, **kw: httpx.Response(200, json=reply, request=request))
    assert llm.chat([{'role': 'user', 'content': 'hi'}], schema={'type': 'object'}) == {'a': 1}

def test_cached_runs_once_and_never_caches_failures():
    calls = []
    def ok():
        calls.append(1)
        return 'value'
    assert llm.cached('k', ok) == ('value', False)
    assert llm.cached('k', ok) == ('value', True)
    assert len(calls) == 1
    assert llm.cached('bad', lambda: None) == (None, False)
    assert 'bad' not in llm.cache
