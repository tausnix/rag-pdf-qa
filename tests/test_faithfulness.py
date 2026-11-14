"""Faithfulness proxy and cost accounting."""
from src.config import settings
from src.generate import cost_usd, estimate_tokens, faithfulness
from src.retrieve import Hit


def _hit(text, page=1):
    return Hit(chunk_id="c", text=text, page=page, source="d.pdf", score=0.5)


CONTEXT = (
    "For Norvik Plant, the reported headcount was 2980 employees. "
    "The annual budget was USD 34 million."
)


def test_verbatim_answer_is_faithful():
    result = faithfulness("The reported headcount was 2980 employees.", [_hit(CONTEXT)])
    assert result["score"] == 1.0
    assert result["numeric_grounding"] is True
    assert result["unsupported_numbers"] == []


def test_invented_number_is_caught():
    result = faithfulness("The reported headcount was 9999 employees.", [_hit(CONTEXT)])
    assert result["score"] == 0.0
    assert result["unsupported_numbers"] == ["9999"]
    assert result["numeric_grounding"] is False


def test_partially_supported_answer_scores_between_zero_and_one():
    answer = (
        "The reported headcount was 2980 employees. "
        "The plant was acquired by an overseas conglomerate in the spring."
    )
    result = faithfulness(answer, [_hit(CONTEXT)])
    assert result["sentences"] == 2
    assert 0.0 < result["score"] < 1.0


def test_unrelated_answer_scores_zero():
    result = faithfulness(
        "Photosynthesis converts light energy into chemical energy.", [_hit(CONTEXT)]
    )
    assert result["score"] == 0.0


def test_citations_and_extractive_suffix_are_stripped():
    answer = "The reported headcount was 2980 employees [p.3]. [extractive fallback - x]"
    assert faithfulness(answer, [_hit(CONTEXT)])["score"] == 1.0


def test_empty_answer_is_not_faithful():
    assert faithfulness("", [_hit(CONTEXT)])["score"] == 0.0


def test_token_estimate_is_char_based():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


def test_cost_uses_configured_prices():
    one_million = cost_usd(1_000_000, 0)
    assert abs(one_million - settings.price_input_per_mtok) < 1e-9
    assert cost_usd(0, 0) == 0.0
    assert cost_usd(500_000, 500_000) > one_million
