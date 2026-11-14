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

import os
import re
import time

from .config import REFUSAL, settings
from .retrieve import Hit

MAX_CONTEXT_CHARS = 6000

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


def _gemini_rest(prompt: str, retries: int = 3) -> dict | None:
    """Dependency-free REST fallback: works with just ``requests``.

    Returns ``{"text": ..., "usage": {"input_tokens", "output_tokens"}}`` or None.
    """
    if not settings.gemini_api_key:
        return None

    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": settings.temperature, "maxOutputTokens": 512},
    }
    for attempt in range(retries):
        try:
            r = requests.post(
                url,
                params={"key": settings.gemini_api_key},
                json=payload,
                timeout=60,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            usage = data.get("usageMetadata") or {}
            return {
                "text": data["candidates"][0]["content"]["parts"][0]["text"].strip(),
                "usage": {
                    "input_tokens": int(usage.get("promptTokenCount", 0)),
                    "output_tokens": int(usage.get("candidatesTokenCount", 0)),
                },
            }
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
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
    if use_llm and settings.gemini_api_key:
        result = _gemini_rest(prompt)
        if result:
            text, usage = result["text"], result["usage"]
            used_llm = True
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
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "cost_usd": cost_usd(input_tokens, output_tokens),
        "billable": used_llm,
    }
