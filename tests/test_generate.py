"""Generation guardrails: refusal detection, prompt grounding, offline fallback."""
from src.config import REFUSAL
from src.generate import answer, build_prompt, is_refusal
from src.retrieve import Hit


def _hit(text, page=3):
    return Hit(chunk_id="c1", text=text, page=page, source="doc.pdf", score=0.5)


def test_is_refusal_variants():
    assert is_refusal(REFUSAL)
    assert is_refusal("I could not find that in the provided text.")
    assert is_refusal("That is not in the document.")
    assert is_refusal("I don't know.")
    assert is_refusal("   ")
    assert not is_refusal("The reported headcount was 2980 employees [p.3].")


def test_prompt_contains_grounding_rules():
    prompt = build_prompt("What is the headcount?", [_hit("Headcount was 2980.")])
    assert REFUSAL in prompt
    assert "=== CONTEXT ===" in prompt
    assert "page 3" in prompt
    assert "What is the headcount?" in prompt


def test_prompt_truncates_huge_context():
    huge = [_hit("x " * 5000, page=i) for i in range(5)]
    prompt = build_prompt("q", huge)
    assert len(prompt) < 20_000


def test_offline_fallback_answers_when_attribute_present():
    hits = [_hit("For Norvik Plant, the reported headcount was 2980 employees. Extra filler text.")]
    result = answer("What was the reported headcount for Norvik Plant?", hits, use_llm=False)
    assert result["refused"] is False
    assert "2980" in result["answer"]


def test_offline_fallback_refuses_when_attribute_absent():
    hits = [_hit("For Norvik Plant, the reported headcount was 2980 employees. Extra filler text.")]
    result = answer("What was the reported carbon offset for Norvik Plant?", hits, use_llm=False)
    assert result["refused"] is True
    assert result["answer"] == REFUSAL


def test_offline_fallback_refuses_when_nothing_retrieved():
    assert answer("Anything?", [], use_llm=False)["refused"] is True
