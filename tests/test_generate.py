"""Generation guardrails: refusal detection, prompt grounding, offline fallback."""
import json
from dataclasses import replace

import pytest

from src import generate
from src.config import REFUSAL, settings
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


# --------------------------------------------------------------------------- #
# Free-tier pacing + on-disk response cache (no network, no API key)
# --------------------------------------------------------------------------- #
def _cache_env(tmp_path, monkeypatch, enabled="1"):
    monkeypatch.setenv("GEMINI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_CACHE", enabled)


def test_cache_roundtrip_is_keyed_by_model_temperature_and_prompt(tmp_path, monkeypatch):
    _cache_env(tmp_path, monkeypatch)
    prompt = "Q: what is the headcount?"
    key = generate.cache_key(prompt)
    assert generate.cache_load(key) is None

    usage = {"input_tokens": 120, "output_tokens": 7}
    generate.cache_store(key, prompt, {"text": "2980 employees [p.3]", "usage": usage})
    assert generate.cache_load(key) == {"text": "2980 employees [p.3]", "usage": usage}

    stored = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    assert stored["version"] == generate.CACHE_VERSION
    assert stored["prompt_sha256"] == key
    assert stored["model"] == settings.gemini_model

    assert len(key) == 64
    assert key == generate.cache_key(prompt)
    assert key != generate.cache_key("other prompt")
    assert key != generate.cache_key(prompt, model="gemini-1.5-flash")
    assert key != generate.cache_key(prompt, temperature=0.7)


def test_cache_can_be_disabled(tmp_path, monkeypatch):
    _cache_env(tmp_path, monkeypatch, enabled="0")
    key = generate.cache_key("p")
    generate.cache_store(key, "p", {"text": "x", "usage": {}})
    assert generate.cache_load(key) is None
    assert not list(tmp_path.glob("*.json"))


class _FakeResponse:
    def __init__(self, status=429, headers=None, body=None):
        self.status_code = status
        self.headers = headers or {}
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def raise_for_status(self):
        pass


def test_retry_delay_prefers_header_then_retry_info_then_backoff():
    body = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                   "retryDelay": "31s"}]}}
    assert generate.retry_delay_seconds(_FakeResponse(headers={"retry-after": "7"}, body=body), 0) == 7.0
    assert generate.retry_delay_seconds(_FakeResponse(body=body), 0) == 31.0

    base = settings.gemini_backoff_base_s
    assert generate.retry_delay_seconds(_FakeResponse(), 0) == base
    assert generate.retry_delay_seconds(_FakeResponse(), 1) == min(60.0, base * 2)
    assert generate.retry_delay_seconds(_FakeResponse(), 10) == 60.0


def test_pace_keeps_requests_at_least_one_interval_apart(monkeypatch):
    clock = [100.0]
    slept: list[float] = []

    def fake_sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(generate, "_LAST_REQUEST_AT", 0.0)
    monkeypatch.setattr(generate.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(generate.time, "sleep", fake_sleep)
    generate.set_min_interval(4.0)
    try:
        generate._pace()          # first call never sleeps
        assert slept == []
        clock[0] += 1.0           # only 1 s of work happened
        generate._pace()
        assert slept == [pytest.approx(3.0)]
    finally:
        generate.set_min_interval(None)


def _fake_post(monkeypatch, responses, calls):
    def post(*args, **kwargs):
        calls.append(kwargs.get("json"))
        return responses.pop(0)

    monkeypatch.setattr("requests.post", post)


def test_gemini_rest_honours_429_then_caches_success(tmp_path, monkeypatch):
    _cache_env(tmp_path, monkeypatch)
    monkeypatch.setattr(generate, "settings", replace(settings, gemini_api_key="test-key"))
    generate.set_min_interval(0.0)
    slept: list[float] = []
    monkeypatch.setattr(generate.time, "sleep", lambda s: slept.append(s))

    responses = [
        _FakeResponse(429, body={"error": {"details": [{"retryDelay": "9s"}]}}),
        _FakeResponse(429, headers={"retry-after": "5"}),
        _FakeResponse(200, body={
            "candidates": [{"content": {"parts": [{"text": " forty two "}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2},
        }),
    ]
    calls: list = []
    _fake_post(monkeypatch, responses, calls)
    try:
        out = generate._gemini_rest("prompt", retries=3)
        assert out["text"] == "forty two"
        assert out["cached"] is False
        assert out["rate_limited"] == 2
        assert out["usage"] == {"input_tokens": 10, "output_tokens": 2}
        assert slept == [9.0, 5.0]

        # The same prompt replays from disk without touching the network.
        monkeypatch.setattr(
            "requests.post",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit")),
        )
        again = generate._gemini_rest("prompt", retries=3)
        assert again["cached"] is True
        assert again["text"] == "forty two"
    finally:
        generate.set_min_interval(None)


def test_gemini_rest_gives_up_and_caches_nothing_after_persistent_429(tmp_path, monkeypatch):
    _cache_env(tmp_path, monkeypatch)
    monkeypatch.setattr(generate, "settings", replace(settings, gemini_api_key="test-key"))
    generate.set_min_interval(0.0)
    monkeypatch.setattr(generate.time, "sleep", lambda s: None)

    responses = [_FakeResponse(429) for _ in range(3)]
    _fake_post(monkeypatch, responses, [])
    try:
        assert generate._gemini_rest("never cached", retries=2) is None
        assert not list(tmp_path.glob("*.json"))
    finally:
        generate.set_min_interval(None)


def test_cached_answer_skips_the_network_and_is_not_billed(tmp_path, monkeypatch):
    _cache_env(tmp_path, monkeypatch)
    hits = [_hit("For Norvik Plant, the reported headcount was 2980 employees.")]
    prompt = build_prompt("What was the headcount?", hits)
    key = generate.cache_key(prompt)
    generate.cache_store(
        key, prompt, {"text": "2980 employees [p.3]", "usage": {"input_tokens": 120, "output_tokens": 7}}
    )
    monkeypatch.setattr(generate, "settings", replace(settings, gemini_api_key="test-key"))
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit"))
    )

    result = answer("What was the headcount?", hits, use_llm=True)
    assert result["used_llm"] is True
    assert result["cached"] is True
    assert result["billable"] is False
    assert result["answer"] == "2980 employees [p.3]"
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 7}


def test_no_cache_lookup_without_an_api_key(tmp_path, monkeypatch):
    """A cached run must not masquerade as a live LLM run when the key is gone."""
    _cache_env(tmp_path, monkeypatch)
    hits = [_hit("For Norvik Plant, the reported headcount was 2980 employees.")]
    prompt = build_prompt("What was the headcount?", hits)
    key = generate.cache_key(prompt)
    generate.cache_store(
        key, prompt, {"text": "2980 employees [p.3]", "usage": {"input_tokens": 1, "output_tokens": 1}}
    )
    monkeypatch.setattr(generate, "settings", replace(settings, gemini_api_key=""))

    result = answer("What was the headcount?", hits, use_llm=True)
    assert result["used_llm"] is False
    assert result["cached"] is False
    assert "extractive fallback" in result["answer"]
