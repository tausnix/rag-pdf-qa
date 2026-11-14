"""Metric implementations."""
from src.benchmark import answer_in_context, hit_at_k, pct, reciprocal_rank


def test_hit_at_k_boundaries():
    gold = {"g"}
    assert hit_at_k(["x", "g", "y"], gold, 3) is True
    assert hit_at_k(["x", "y", "g"], gold, 3) is True
    assert hit_at_k(["x", "y", "z", "g"], gold, 3) is False
    assert hit_at_k(["x", "y", "z", "g"], gold, 4) is True


def test_hit_at_k_with_multiple_gold_chunks():
    assert hit_at_k(["x", "g2"], {"g1", "g2"}, 2) is True


def test_reciprocal_rank_uses_first_gold_only():
    assert reciprocal_rank(["g", "x"], {"g"}, 10) == 1.0
    assert reciprocal_rank(["x", "g"], {"g"}, 10) == 0.5
    assert abs(reciprocal_rank(["x", "y", "g"], {"g"}, 10) - 1 / 3) < 1e-12
    # beyond the cutoff counts as a miss
    assert reciprocal_rank(["x", "y", "z", "g"], {"g"}, 3) == 0.0


def test_reciprocal_rank_no_gold():
    assert reciprocal_rank(["x", "y"], {"g"}, 10) == 0.0


def test_answer_in_context_normalises_case_and_punctuation():
    texts = ["Reported headcount was 2980 employees, according to the review."]
    assert answer_in_context(texts, "2980 employees") is True
    assert answer_in_context(texts, "2980 EMPLOYEES.") is True
    assert answer_in_context(texts, "4210 employees") is False


def test_answer_in_context_ignores_empty_gold():
    assert answer_in_context(["anything"], "") is False


def test_percentile_helper():
    values = [1.0, 2.0, 3.0, 4.0]
    assert pct(values, 0.0) == 1.0
    assert pct(values, 1.0) == 4.0
    assert pct([], 0.5) == 0.0
