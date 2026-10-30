from ragommender import retrieval
from ragommender.ranking import default_weights
import pytest

# retrieval.search with the index, the vocabulary mapping and the llm stubbed: what it asks hybrid_search for, and how it
# builds reasons, the era filter and the llm flags from what comes back

def item(movie_id, year, score, vector_rank=None, bm25_rank=None):
    return {'item_id': str(movie_id), 'title': f'Movie {movie_id} ({year})', 'score': score,
            'vector_rank': vector_rank, 'bm25_rank': bm25_rank, 'era_boost': 0.0}

class FakeRetrieval:
    def __init__(self, unfiltered, filtered):
        self.unfiltered, self.filtered = unfiltered, filtered
        # hybrid_search calls, map_query queries and the ids sent for explanations
        self.calls, self.mapped_calls, self.explained = [], [], []

    def hybrid_search(self, query_text, k, weights=None, bm25_text=None, year_range=None):
        self.calls.append({'query': query_text, 'k': k, 'bm25_text': bm25_text, 'year_range': year_range})
        # copies, since boost_era changes items in place
        return [dict(i) for i in (self.filtered if year_range else self.unfiltered)][:k]

@pytest.fixture
def stub(monkeypatch):
    def setup(unfiltered, filtered=(), mapped=(('dark comedy', 0.6),), rewrite=None, explain=None):
        fake = FakeRetrieval(unfiltered, filtered)
        def map_query(query, params=None):
            fake.mapped_calls.append(query)
            return list(mapped)
        def explain_items(query, items):
            fake.explained.append([i['item_id'] for i in items])
            return explain
        monkeypatch.setattr(retrieval, 'index', lambda: (None, fake))
        monkeypatch.setattr(retrieval, 'map_query', map_query)
        monkeypatch.setattr(retrieval, 'rewrite_and_pick', lambda query: rewrite)
        monkeypatch.setattr(retrieval, 'explain_items', explain_items)
        return fake
    return setup

def test_plain_search(stub):
    fake = stub([item(1, 1996, 0.03, 1, 2), item(2, 2004, 0.02, None, 1), item(3, 1999, 0.01, 2, None)])
    result = retrieval.search('funny but bleak', k=2)
    # the whole candidate pool is fetched and cut to k after, the mapped tag joins the bm25 text only
    assert fake.calls == [{'query': 'funny but bleak', 'k': 100, 'bm25_text': 'funny but bleak dark comedy', 'year_range': None}]
    assert [i['item_id'] for i in result['items']] == ['1', '2']
    assert result['items'][0]['reason'] == ["Semantic match for 'funny but bleak'", "Keyword match for 'funny but bleak'"]
    assert result['items'][1]['reason'] == ["Keyword match for 'funny but bleak'"]
    assert 'explain' not in result['items'][0]
    assert result['mapped_tags'] == ['dark comedy']
    assert result['rewritten_query'] is None
    assert (result['llm_used'], result['llm_cached']) == (False, False)
    assert fake.explained == []

def test_pool_grows_with_k(stub):
    fake = stub([item(1, 1996, 0.03, 1, 1)])
    retrieval.search('heist', k=50)
    assert fake.calls[0]['k'] == 150

def test_rewrite_adds_keywords_picked_tags_and_the_era(stub):
    rewritten = {'keywords': ['heist', 'casino'], 'genres': ['Crime'], 'year_from': 1999, 'year_to': 1990}
    in_era = [item(1, 1995, 0.02, 1, 1)]
    everything = [item(2, 2001, 0.022, 1, 1), item(1, 1995, 0.02, 2, 2), item(3, 1980, 0.01, None, 3)]
    fake = stub(everything, filtered=in_era, rewrite=((rewritten, False), (['heist'], True)))
    result = retrieval.search('90s casino heist movies', k=3, rewrite=True)
    # the years come sorted, and with fewer than k movies in the era the unfiltered search tops up the pool
    assert [c['year_range'] for c in fake.calls] == [(1990, 1999), None]
    assert fake.calls[0]['bm25_text'] == '90s casino heist movies heist casino heist'
    # the llm's picks replace the embedding mapping
    assert fake.mapped_calls == []
    assert result['mapped_tags'] == ['heist']
    # the boost lifts the 1995 movie over the 2001 one, and each movie appears once
    assert [i['item_id'] for i in result['items']] == ['1', '2', '3']
    assert result['items'][0]['era_boost'] == default_weights['year_boost']
    assert result['items'][0]['reason'][-1] == 'Released in the era you searched for (1990-1999)'
    assert result['items'][1]['era_boost'] == 0.0
    assert result['rewritten_query'] == rewritten
    # the rewrite was generated, the pick came from the cache
    assert (result['llm_used'], result['llm_cached']) == (True, False)

def test_rewrite_with_enough_movies_in_the_era_skips_the_top_up(stub):
    rewritten = {'keywords': [], 'genres': [], 'year_from': 1990, 'year_to': None}
    fake = stub([item(2, 2001, 0.03, 1, 1)], filtered=[item(1, 1995, 0.02, 1, 1)], rewrite=((rewritten, True), ([], True)))
    result = retrieval.search('movies from 1990 on', k=1, rewrite=True)
    # one year makes a one year range
    assert [c['year_range'] for c in fake.calls] == [(1990, 1990)]
    assert result['mapped_tags'] == []
    assert (result['llm_used'], result['llm_cached']) == (True, True)

def test_failed_rewrite_falls_back_to_the_embedding_mapping(stub):
    fake = stub([item(1, 1996, 0.03, 1, 1)], rewrite=((None, False), (None, False)))
    result = retrieval.search('funny but bleak', k=1, rewrite=True)
    assert fake.calls == [{'query': 'funny but bleak', 'k': 100, 'bm25_text': 'funny but bleak dark comedy', 'year_range': None}]
    assert fake.mapped_calls == ['funny but bleak']
    assert result['rewritten_query'] is None
    assert result['mapped_tags'] == ['dark comedy']
    assert (result['llm_used'], result['llm_cached']) == (False, False)

def test_explain_replaces_reasons_for_the_top_five(stub):
    reasons = {str(i): f'reason {i}' for i in [1, 2, 4, 5]}
    fake = stub([item(i, 2000, 0.1 - i/100, i, None) for i in range(1, 8)], explain=(reasons, False))
    result = retrieval.search('space', k=7, explain=True)
    assert fake.explained == [['1', '2', '3', '4', '5']]
    # a movie the llm skipped keeps its template reason
    template = ["Semantic match for 'space'"]
    assert [i['reason'] for i in result['items']] == [['reason 1'], ['reason 2'], template, ['reason 4'], ['reason 5'],
                                                       template, template]
    assert result['items'][0]['explain'] == {'vector_rank': 1, 'bm25_rank': None, 'era_boost': 0.0}
    assert (result['llm_used'], result['llm_cached']) == (True, False)

def test_explain_keeps_template_reasons_when_the_llm_fails(stub):
    stub([item(1, 2000, 0.1, 1, 1)], explain=(None, False))
    result = retrieval.search('space', k=1, explain=True)
    assert result['items'][0]['reason'] == ["Semantic match for 'space'", "Keyword match for 'space'"]
    assert (result['llm_used'], result['llm_cached']) == (False, False)

def test_explain_skips_the_llm_without_results(stub):
    fake = stub([], explain=({}, False))
    result = retrieval.search('zzz', k=5, explain=True)
    assert result['items'] == []
    assert fake.explained == []
    assert (result['llm_used'], result['llm_cached']) == (False, False)
