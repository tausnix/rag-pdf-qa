"""Grounded answer generation.

The prompt is the part that decides whether the system is trustworthy:

  * answer ONLY from the provided context,
  * cite page numbers,
  * say a fixed refusal sentence when the context does not contain the answer.

If no ``GEMINI_API_KEY`` is configured (or ``--no-llm`` is used) the system falls
back to an *extractive* answer: the single best-matching sentence from the top
chunks. That keeps retrieval evaluation usable offline and is clearly labelled.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from .config import REFUSAL, settings
from .retrieve import Hit

MAX_CONTEXT_CHARS = 6000

# Bump when the cache payload changes, so stale entries are never reused.
CACHE_VERSION = 1
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)
_MAX_BACKOFF_S = 60.0

SYSTEM_PROMPT = """You are a document question-answering assistant.
Rules:
1. Answer using ONLY the context passages below. Never use outside knowledge.
2. If the context does not contain the answer, reply exactly: "{refusal}"
3. Be concise (1-3 sentences). Quote exact figures, names and units.
4. Cite the page numbers you used in the form [p.N] at the end of the answer.
""".format(refusal=REFUSAL)


def format_context(hits: list[Hit]) -> str:
    blocks, used = [], 0
    for i, h in enumerate(hits, start=1):
        block = f"[{i}] (page {h.page})\n{h.text.strip()}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[Hit]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"=== CONTEXT ===\n{format_context(hits)}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ==="
    )


# --------------------------------------------------------------------------- #
# Pacing and on-disk response cache
#
# Google AI Studio's free tier enforces requests-per-minute across Flash models
# and rejects bursts with 429 RESOURCE_EXHAUSTED. Two cheap defences keep a
# 225-query benchmark alive:
#
#   * pace calls at ``GEMINI_MIN_INTERVAL_S`` (4 s ~= 15 RPM) and honour the
#     server's RetryInfo/Retry-After before backing off exponentially;
#   * write every successful response to ``GEMINI_CACHE_DIR`` keyed by
#     model + temperature + full prompt, so an interrupted run resumes for free
#     and a completed run never pays twice.
# --------------------------------------------------------------------------- #
_LAST_REQUEST_AT = 0.0
_MIN_INTERVAL_OVERRIDE: float | None = None
_CACHE_OVERRIDE: bool | None = None


def set_min_interval(seconds: float | None) -> None:
    """Override the inter-request delay (CLI ``--llm-delay``); None = config."""
    global _MIN_INTERVAL_OVERRIDE
    _MIN_INTERVAL_OVERRIDE = seconds


def set_cache_enabled(enabled: bool | None) -> None:
    """Override the disk cache (CLI ``--no-llm-cache``); None = config."""
    global _CACHE_OVERRIDE
    _CACHE_OVERRIDE = enabled


def min_interval() -> float:
    if _MIN_INTERVAL_OVERRIDE is not None:
        return max(0.0, _MIN_INTERVAL_OVERRIDE)
    return max(0.0, settings.gemini_min_interval_s)


def cache_dir() -> Path:
    """Resolved at call time so tests and one-off runs can redirect it."""
    return Path(os.getenv("GEMINI_CACHE_DIR") or settings.gemini_cache_dir)


def cache_enabled() -> bool:
    if _CACHE_OVERRIDE is not None:
        return _CACHE_OVERRIDE
    raw = os.getenv("GEMINI_CACHE")
    if raw is None:
        return settings.gemini_cache
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def cache_key(prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    """Stable sha256 over everything that can change the completion."""
    payload = "\x00".join(
        (
            f"v{CACHE_VERSION}",
            model or settings.gemini_model,
            f"{settings.temperature if temperature is None else temperature:.4f}",
            prompt,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_load(key: str) -> dict | None:
    """Return ``{"text", "usage"}`` for a previous response, or None."""
    if not cache_enabled():
        return None
    try:
        data = json.loads((cache_dir() / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("version") != CACHE_VERSION or not data.get("text"):
        return None
    return {"text": data["text"], "usage": data.get("usage") or {}}


def cache_store(key: str, prompt: str, result: dict) -> None:
    """Persist one successful response atomically (Ctrl-C safe)."""
    if not cache_enabled():
        return
    path = cache_dir() / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "model": settings.gemini_model,
        "temperature": settings.temperature,
        "prompt_sha256": key,
        "prompt_chars": len(prompt),
        "created_at": time.time(),
        "text": result["text"],
        "usage": result.get("usage") or {},
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _pace() -> None:
    """Sleep so consecutive requests are at least ``min_interval()`` apart."""
    global _LAST_REQUEST_AT
    interval = min_interval()
    now = time.monotonic()
    if _LAST_REQUEST_AT and interval > 0:
        wait = _LAST_REQUEST_AT + interval - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
    _LAST_REQUEST_AT = now


# Retry-After may be plain delta-seconds ("5") and RetryInfo uses "31s".
_DURATION = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*s?\s*$")


def _parse_duration(value) -> float | None:
    match = _DURATION.match(str(value)) if value is not None else None
    return float(match.group(1)) if match else None


def retry_delay_seconds(response, attempt: int) -> float:
    """How long to wait before retrying.

    Priority: ``Retry-After`` header -> ``google.rpc.RetryInfo.retryDelay`` in the
    JSON body -> exponential backoff from ``GEMINI_BACKOFF_BASE_S``.
    """
    try:
        header = _parse_duration(response.headers.get("retry-after"))
    except Exception:  # noqa: BLE001 - a fake/odd response object is not fatal
        header = None
    if header is not None:
        return header

    try:
        body = response.json()
        for detail in (body.get("error") or {}).get("details") or []:
            delay = _parse_duration(detail.get("retryDelay"))
            if delay is not None:
                return delay
    except Exception:  # noqa: BLE001
        pass

    return min(_MAX_BACKOFF_S, settings.gemini_backoff_base_s * (2**attempt))


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def _gemini_sdk(prompt: str) -> str | None:
    """Try google-genai (new) then google-generativeai (legacy)."""
    if not settings.gemini_api_key:
        return None

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=settings.gemini_api_key)
        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config={"temperature": settings.temperature, "max_output_tokens": 512},
        )
        return (resp.text or "").strip()
    except ImportError:
        pass
    except Exception:
        return None

    try:
        import google.generativeai as genai_legacy  # type: ignore

        genai_legacy.configure(api_key=settings.gemini_api_key)
        model = genai_legacy.GenerativeModel(settings.gemini_model)
        resp = model.generate_content(
            prompt, generation_config={"temperature": settings.temperature, "max_output_tokens": 512}
        )
        return (resp.text or "").strip()
    except Exception:
        return None


def _gemini_rest(prompt: str, retries: int | None = None) -> dict | None:
    """Dependency-free REST call with pacing, 429 backoff and a disk cache.

    Returns ``{"text", "usage", "cached", "rate_limited"}`` or None. The cache is
    consulted first, so re-running an interrupted benchmark costs no quota.
    """
    if not settings.gemini_api_key:
        return None

    import requests

    key = cache_key(prompt)
    stored = cache_load(key)
    if stored is not None:
        return {**stored, "cached": True, "rate_limited": 0}

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": settings.temperature, "maxOutputTokens": 512},
    }
    attempts = (settings.gemini_max_retries if retries is None else retries) + 1
    rate_limited = 0

    for attempt in range(attempts):
        _pace()
        try:
            r = requests.post(
                url,
                params={"key": settings.gemini_api_key},
                json=payload,
                timeout=60,
            )
            if r.status_code in _RETRYABLE_STATUS:
                if r.status_code == 429:
                    rate_limited += 1
                if attempt == attempts - 1:
                    return None
                time.sleep(retry_delay_seconds(r, attempt))
                continue
            r.raise_for_status()
            data = r.json()
            usage_meta = data.get("usageMetadata") or {}
            result = {
                "text": data["candidates"][0]["content"]["parts"][0]["text"].strip(),
                "usage": {
                    "input_tokens": int(usage_meta.get("promptTokenCount", 0)),
                    "output_tokens": int(usage_meta.get("candidatesTokenCount", 0)),
                },
            }
            cache_store(key, prompt, result)
            return {**result, "cached": False, "rate_limited": rate_limited}
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(min(_MAX_BACKOFF_S, settings.gemini_backoff_base_s * (2**attempt)))
    return None


# --------------------------------------------------------------------------- #
# Extractive fallback
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "was", "were", "is", "are", "be", "been", "of", "for", "to",
    "in", "on", "at", "by", "with", "and", "or", "what", "which", "who", "when",
    "where", "how", "did", "does", "do", "that", "this", "it", "its", "reported",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2 and t not in _STOP}


def _extractive(question: str, hits: list[Hit]) -> str:
    """Pick the single best-matching sentence from the top chunks.

    Only used when no GEMINI_API_KEY is configured. A candidate must cover at
    least half of the question's *attribute* words (the words that are not part
    of a proper noun) -- otherwise "carbon offset for Halden Institute" would be
    "answered" by any sentence that happens to mention Halden Institute.
    """
    q_tokens = _content_tokens(question)
    q_proper = {w for w in re.findall(r"\b[A-Z][A-Za-z\-]+", question) if w.lower() not in _STOP}
    q_attr = q_tokens - {w.lower() for w in q_proper}

    best, best_score = "", 0.0
    for hit in hits[:3]:
        for sentence in re.split(r"(?<=[.!?])\s+", hit.text):
            s = " ".join(sentence.split())
            if len(s) < 40:
                continue
            s_tokens = _content_tokens(s)

            attr_cov = (len(q_attr & s_tokens) / len(q_attr)) if q_attr else 1.0
            if attr_cov < 0.5:
                continue

            coverage = len(q_tokens & s_tokens) / (len(q_tokens) or 1)
            proper = (
                sum(1 for w in q_proper if re.search(rf"\b{re.escape(w)}\b", s)) / len(q_proper)
                if q_proper
                else 0.0
            )
            score = (coverage + 0.4 * proper) / (1 + len(s) / 600)

            if score > best_score:
                best, best_score = s, score

    if best_score < 0.6:
        return REFUSAL
    return f"{best} [extractive fallback - set GEMINI_API_KEY for LLM answers]"


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token.

    Used when the provider does not return usage metadata (SDK path, extractive
    fallback). The REST path returns exact counts and takes precedence.
    """
    return max(1, int(round(len(text) / 4)))


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """USD cost for one call, from the configured per-million-token prices."""
    return (
        input_tokens / 1_000_000 * settings.price_input_per_mtok
        + output_tokens / 1_000_000 * settings.price_output_per_mtok
    )


# --------------------------------------------------------------------------- #
# Faithfulness
# --------------------------------------------------------------------------- #
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_EXTRACTIVE_SUFFIX = re.compile(r"\s*\[extractive fallback[^\]]*\]", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def faithfulness(answer_text: str, hits: list[Hit]) -> dict:
    """Lexical check that the answer is supported by the retrieved context.

    Per answer sentence: content-word coverage against the best-matching context
    sentence, *and* every number in the sentence must appear in the context. A
    sentence passes when coverage >= ``FAITHFULNESS_MIN_OVERLAP`` and no number
    is unsupported. This is a cheap proxy for entailment: it catches invented
    figures and mismatched entities, but it will not catch a fluent distortion
    that reuses the context's own words.
    """
    text = _EXTRACTIVE_SUFFIX.sub("", re.sub(r"\[p\.\d+\]", "", answer_text)).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 25]
    if not sentences:
        return {"score": 0.0, "sentences": 0, "supported": 0,
                "unsupported_numbers": sorted(_numbers(text)), "numeric_grounding": not _numbers(text)}

    context_text = " ".join(h.text for h in hits)
    context_numbers = _numbers(context_text)
    context_sentences = [
        s for h in hits for s in re.split(r"(?<=[.!?])\s+", h.text) if len(s.strip()) >= 20
    ]

    supported = 0
    unsupported_numbers: set[str] = set()
    for sentence in sentences:
        tokens = _content_tokens(sentence)
        coverage = 0.0
        if tokens:
            for candidate in context_sentences:
                coverage = max(coverage, len(tokens & _content_tokens(candidate)) / len(tokens))
        missing = _numbers(sentence) - context_numbers
        unsupported_numbers |= missing
        if coverage >= settings.faithfulness_min_overlap and not missing:
            supported += 1

    return {
        "score": supported / len(sentences),
        "sentences": len(sentences),
        "supported": supported,
        "unsupported_numbers": sorted(unsupported_numbers),
        "numeric_grounding": not unsupported_numbers,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def is_refusal(text: str) -> bool:
    t = text.strip().lower()
    return (
        not t
        or t.startswith("i couldn't find")
        or t.startswith("i could not find")
        or "not in the document" in t
        or "don't know" in t
        or "do not have enough" in t
    )


def answer(question: str, hits: list[Hit], use_llm: bool = True) -> dict:
    """Return answer, citations, refusal flag and (for LLM calls) token usage."""
    prompt = build_prompt(question, hits)

    text = None
    used_llm = False
    usage = None
    cached = False
    rate_limited = 0
    if use_llm and settings.gemini_api_key:
        result = _gemini_rest(prompt)
        if result:
            text, usage = result["text"], result["usage"]
            used_llm = True
            cached = bool(result.get("cached"))
            rate_limited = int(result.get("rate_limited", 0))
        else:
            text = _gemini_sdk(prompt)
            used_llm = text is not None

    if text is None:
        text = _extractive(question, hits)

    refused = is_refusal(text)
    citations: list[int] = []
    if not refused:
        citations = sorted({int(m) for m in re.findall(r"\[p\.(\d+)\]", text)}) or sorted(
            {h.page for h in hits[:3]}
        )

    # Token accounting: exact usage from the API when available, otherwise an
    # estimate over the same prompt/answer so cost can still be projected.
    if used_llm:
        input_tokens = (usage or {}).get("input_tokens") or estimate_tokens(prompt)
        output_tokens = (usage or {}).get("output_tokens") or estimate_tokens(text)
    else:
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(text)

    return {
        "answer": text,
        "citations": citations,
        "refused": refused,
        "used_llm": used_llm,
        "cached": cached,
        "rate_limited": rate_limited,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "cost_usd": cost_usd(input_tokens, output_tokens),
        # A cache hit consumed no quota in *this* run, so it is not billed here.
        "billable": used_llm and not cached,
    }
