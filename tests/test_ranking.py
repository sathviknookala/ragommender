from ragommender.ranking import default_weights, fuse, with_fallback, boost_era, parse_year
import pytest

def lists():
    knn = [(1, 'Alien (1979)', 0.1), (2, 'Aliens (1986)', 0.2)]
    bm25 = [(2, 'Aliens (1986)', 9.0), (3, 'Prometheus (2012)', 5.0)]
    return knn, bm25

def test_parse_year():
    assert parse_year('Alien (1979)') == 1979
    assert parse_year('Alien (1979) ') == 1979
    assert parse_year('2001: A Space Odyssey') is None
    assert parse_year('Babylon 5') is None

def test_fuse_is_weighted_rrf():
    knn, bm25 = lists()
    w = dict(default_weights, popularity=0.0)
    items = {i['item_id']: i for i in fuse(knn, bm25, w)}
    k = w['rrf_k']
    assert items['1']['score'] == pytest.approx(w['vector'] / (k + 1))
    assert items['2']['score'] == pytest.approx(w['vector'] / (k + 2) + w['bm25'] / (k + 1))
    assert items['3']['score'] == pytest.approx(w['bm25'] / (k + 2))
    assert (items['2']['vector_rank'], items['2']['bm25_rank']) == (2, 1)
    assert items['3']['vector_rank'] is None

def test_fuse_sorts_and_zero_weight_drops_a_list():
    knn, bm25 = lists()
    ranked = fuse(knn, bm25, dict(default_weights, vector=0.0, popularity=0.0))
    assert [i['item_id'] for i in ranked] == ['2', '3']

def test_popularity_reorders_close_scores():
    knn, bm25 = lists()
    w = dict(default_weights, vector=0.0)
    assert [i['item_id'] for i in fuse(knn, bm25, w, popularity={3: 1.0})] == ['3', '2']
    assert [i['item_id'] for i in fuse(knn, bm25, dict(w, popularity=0.0), popularity={3: 1.0})] == ['2', '3']

def test_with_fallback_turns_knn_on_only_when_bm25_is_empty():
    w = dict(default_weights, vector=0.0)
    assert with_fallback(w, [])['vector'] == 1.0
    assert with_fallback(w, [(1, 'Alien (1979)', 1.0)]) is w
    assert with_fallback(default_weights, []) is default_weights

def test_boost_era():
    knn, bm25 = lists()
    w = dict(default_weights, popularity=0.0)
    items = boost_era(fuse(knn, bm25, w), w, (2010, 2019))
    assert items[0]['item_id'] == '3' and items[0]['era_boost'] == w['year_boost']
    assert all(i['era_boost'] == 0.0 for i in items[1:])
    same = fuse(knn, bm25, w)
    assert boost_era(same, w, None) is same
