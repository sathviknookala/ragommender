from ragommender.evaluation.run import metrics, summarize, popularity_buckets
import numpy as np
import pytest

def test_perfect_ranking_scores_one():
    m = metrics(['1', '2', '3'], ['1', '2', '3'])
    assert m['ndcg@10'] == pytest.approx(1.0)
    assert m['precision@10'] == pytest.approx(0.3)
    assert m['recall@10'] == pytest.approx(1.0)

def test_single_relevant_at_rank_two():
    assert metrics(['9', '1'], ['1'])['ndcg@10'] == pytest.approx(1 / np.log2(3))

def test_ideal_dcg_caps_at_ten():
    relevant = [str(i) for i in range(30)]
    m = metrics(relevant[:10], relevant)
    assert m['ndcg@10'] == pytest.approx(1.0)
    assert m['recall@10'] == pytest.approx(10 / 30)

def test_nothing_relevant_in_top_ten():
    assert metrics([str(i) for i in range(10, 20)], ['1'])['ndcg@10'] == 0.0

def test_summarize_ci_brackets_mean():
    s = summarize(np.linspace(0, 1, 101), np.random.default_rng(0))
    assert s['ci95'][0] < s['mean'] < s['ci95'][1]

def test_popularity_buckets_tertiles():
    queries = [{'relevant': [str(i)]} for i in range(9)]
    buckets = popularity_buckets(queries, {i: i for i in range(9)})
    assert buckets.count('tail') == buckets.count('mid') == buckets.count('head') == 3
