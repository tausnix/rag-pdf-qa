"""Cross-encoder reranking: pure ranking logic, no model download required."""
import numpy as np
import pytest

from src.rerank import CrossEncoderReranker, rerank_hits
from src.retrieve import Hit


def _hit(cid, text="text", score=0.1):
    return Hit(chunk_id=cid, text=text, page=1, source="d.pdf", score=score)


def test_rerank_reorders_by_scorer():
    hits = [_hit("a"), _hit("b"), _hit("c")]
    scores = {"a": 1.0, "b": 9.0, "c": -3.0}
    ranked = rerank_hits("q", hits, k=3, scorer=lambda q, texts: [scores[h.chunk_id] for h in hits])
    assert [h.chunk_id for h in ranked] == ["b", "a", "c"]
    assert ranked[0].rerank_score == 9.0


def test_rerank_respects_k_and_leaves_unscored_hits_alone():
    hits = [_hit(c) for c in "abcde"]
    ranked = rerank_hits(
        "q", hits, k=2, scorer=lambda q, texts: np.arange(len(texts), dtype=np.float32)
    )
    assert [h.chunk_id for h in ranked] == ["e", "d"]
    assert ranked[0].rerank_score == 4.0


def test_rerank_only_scores_top_n_candidates():
    seen = {}

    def scorer(q, texts):
        seen["n"] = len(texts)
        return np.zeros(len(texts), dtype=np.float32)

    hits = [_hit(str(i)) for i in range(30)]
    rerank_hits("q", hits, k=3, scorer=scorer, top_n=15)
    assert seen["n"] == 15
    # hits outside the candidate window must stay unscored
    assert all(h.rerank_score is None for h in hits[15:])


def test_rerank_passes_the_query_through():
    captured = {}

    def scorer(q, texts):
        captured["q"] = q
        return np.zeros(len(texts), dtype=np.float32)

    rerank_hits("what is the headcount?", [_hit("a")], k=1, scorer=scorer)
    assert captured["q"] == "what is the headcount?"


def test_rerank_empty_input():
    assert rerank_hits("q", [], k=5, scorer=lambda q, t: []) == []


def test_scorer_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        rerank_hits("q", [_hit("a"), _hit("b")], k=2, scorer=lambda q, t: [1.0])


def test_score_with_no_passages_returns_empty_without_loading_model():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)  # bypass __init__/download
    assert reranker.score("q", []).shape == (0,)
