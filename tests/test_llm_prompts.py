from ragommender import llm, paths, retrieval
from ragommender.rewrite import rewrite_query, pick_tags
import httpx
import pytest
import re

# the rewrite, tag pick and explain prompts against the running llm server (scripts/serve_llm.sh): pytest -m llm
# the unit tests only check what the code does with a reply, these check the replies the prompts get
pytestmark = [pytest.mark.llm,
              pytest.mark.skipif(not (paths.movie_info / 'movies.csv').exists(), reason='movielens csvs not found')]

@pytest.fixture(scope='module', autouse=True)
def server():
    if not llm.enabled:
        pytest.skip('LLM_ENABLED=0')
    try:
        llm.get_model()
    except httpx.HTTPError:
        pytest.skip(f'no llm server at {llm.base_url}')

@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    # every test asks the server, and one failure doesn't put the rest in cooldown
    monkeypatch.setattr(llm, 'down_until', 0.0)
    llm.cache.clear()
    yield
    llm.cache.clear()

def rewrite(query):
    result, _ = rewrite_query(query)
    assert isinstance(result, dict), f'no rewrite for {query!r}'
    return result

def test_rewrite_turns_a_decade_into_years():
    r = rewrite('90s heist movies')
    assert (r['year_from'], r['year_to']) == (1990, 1999)
    assert any('heist' in k.lower() for k in r['keywords'])
    # release years are a filter, never keywords
    assert not any(re.search(r'\d', k) for k in r['keywords'])

def test_rewrite_keeps_a_story_period_as_a_keyword():
    r = rewrite('world war ii movies')
    assert (r['year_from'], r['year_to']) == (None, None)
    assert any('world war' in k.lower() for k in r['keywords'])

def test_rewrite_drops_filler_words():
    r = rewrite('something like alien but funnier')
    keywords = {k.lower() for k in r['keywords']}
    assert any('alien' in k for k in keywords)
    assert not keywords & {'something', 'like', 'funnier', 'movie', 'movies', 'film'}

def test_rewrite_picks_genres():
    assert {'Horror', 'Sci-Fi'} & set(rewrite('scary space movies')['genres'])

def test_pick_keeps_matching_tags_and_drops_unrelated_ones():
    picked, _ = pick_tags('uplifting movies', ['feel-good', 'inspirational', 'gore', 'zombies'])
    assert picked is not None
    assert 'feel-good' in picked
    assert not {'gore', 'zombies'} & set(picked)

def test_pick_picks_nothing_when_nothing_fits():
    picked, _ = pick_tags('movies about time travel', ['gore', 'romance', 'western'])
    assert picked == []

@pytest.mark.skipif(not paths.chroma_dir.exists(), reason='chroma index not built')
def test_explain_gives_every_item_a_short_reason(monkeypatch):
    import chromadb
    collection = chromadb.PersistentClient(path=str(paths.chroma_dir)).get_collection('rag_db')
    # explain_items reads only the stored documents, so the embedding model isn't loaded
    monkeypatch.setattr(retrieval, 'index', lambda: (collection, None))
    # Fargo, Being John Malkovich, Donnie Darko
    ids = ['608', '2997', '4878']
    reasons, _ = retrieval.explain_items('dark comedy', [{'item_id': i} for i in ids])
    assert isinstance(reasons, dict)
    assert set(reasons) == set(ids)
    # the prompt asks for at most 20 words each
    assert all(reason.strip() and len(reason.split()) <= 30 for reason in reasons.values()), reasons
