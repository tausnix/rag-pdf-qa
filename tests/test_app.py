"""AppTest smoke tests for the TraceRAG UI.

The shell/trace tests run against a tiny temporary index, so they need no
embedder, no cross-encoder and no network -- they are CI-safe. The live query
test is skipped unless a real local index exists (``python run_pipeline.py``).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.generate  # noqa: F401 - bind the real settings before the fixture patches config
import src.retrieve  # noqa: F401
from src.config import settings

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
DEMO_PDF = ROOT / "data" / "raw" / "demo_handbook.pdf"


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    """A one-chunk index that makes the app think the handbook is loaded."""
    raw = tmp_path / "raw"
    raw.mkdir()
    if DEMO_PDF.exists():
        shutil.copyfile(DEMO_PDF, raw / DEMO_PDF.name)

    index = tmp_path / "index"
    index.mkdir()
    chunk = {
        "chunk_id": "c1",
        "text": "For Meridian Depot, the reported headcount was 44 employees.",
        "page": 1,
        "source": DEMO_PDF.name,
    }
    (index / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    (index / "bm25.pkl").write_bytes(b"unused by the UI shell tests")

    fake = replace(
        settings,
        raw_dir=raw,
        index_dir=index,
        eval_dir=tmp_path / "eval",
        chroma_dir=tmp_path / "chroma",
    )
    monkeypatch.setattr("src.config.settings", fake)
    return fake


def _hit(rank: int, page: int, prob: float | None) -> dict:
    return {
        "chunk_id": f"c{rank}",
        "page": page,
        "text": f"Page {page} mentions Meridian Depot and its reported headcount.",
        "dense_score": 0.72 - rank * 0.01,
        "bm25_score": 8.5 - rank,
        "rrf_score": 0.032 - rank * 0.001,
        "rrf_rank": rank,
        "rerank_rank": rank,
        "logit": 2.5 - rank * 0.2,
        "prob": prob,
        "movement": f"RRF #{rank} → Rerank #{rank}",
    }


def _trace(refused: bool = False) -> dict:
    result = {
        "answer": (
            "I couldn't find that in the document."
            if refused
            else "For Meridian Depot, the reported headcount was 44 employees. [p.1]"
        ),
        "citations": [] if refused else [1],
        "refused": refused,
        "used_llm": False,
        "cached": False,
        "rate_limited": 0,
        "usage": {"input_tokens": 900, "output_tokens": 14},
        "cost_usd": 0.0001,
        "billable": False,
    }
    faith = None if refused else {
        "score": 1.0,
        "sentences": 1,
        "supported": 1,
        "unsupported_numbers": [],
        "numeric_grounding": True,
    }
    return {
        "question": "What was the reported headcount for Meridian Depot?",
        "mode": "Hybrid + rerank",
        "top_n": 10,
        "chunks_to_model": 5,
        "timings": {"retrieval_ms": 12.5, "rerank_ms": 829.4, "generation_ms": 1.0},
        "cross_encoder": {"pairs": 10, "tokens": 164},
        "result": result,
        "faith": faith,
        "hits": [_hit(i, page=6 + i, prob=(0.9 - i * 0.05) if not refused else 0.02)
                 for i in range(1, 6)],
        "dense_baseline": {
            "chunk_id": "c9",
            "page": 57,
            "text": "A dense-only distractor chunk.",
            "score": 0.61,
        },
    }


def _boot(trace: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    if trace is not None:
        at.session_state["trace"] = trace
        at.session_state["qid"] = 1
        at.session_state["mode"] = "Hybrid + rerank"
        at.session_state["rerank_candidates"] = 10
    return at.run()


def test_app_shell_boots_with_brand_and_split_layout(fake_index):
    at = _boot()
    assert not list(at.exception)
    assert any("TraceRAG" in m.value for m in at.markdown)
    assert [t.label for t in at.tabs] == [
        ":material/search: Ask & Trace",
        ":material/bar_chart: Benchmark",
        ":material/info: How it works",
    ]
    assert len(at.pills) == 1
    assert at.pills[0].options == [
        "Exact lookup · Ember Point budget",
        "Rank rescue · Meridian headcount",
        "Trap · 2026 World Cup winner",
    ]
    assert at.chat_input[0].placeholder == "Ask something about this document..."
    # Benchmark tab still renders all three saved 3-way tables.
    assert [len(df.value) for df in at.dataframe] == [3, 3, 2]
    assert len(at.metric) == 0  # empty state until a question is asked


def test_trace_renders_metrics_table_and_evidence_inspector(fake_index):
    at = _boot(_trace())
    assert not list(at.exception)

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Retrieval latency"] == "12.5 ms"
    assert metrics["Rerank latency"] == "829.4 ms"
    assert metrics["Faithfulness / spans"] == "100%"
    assert metrics["Projected cost / 1k"] == "$0.10"

    table = at.dataframe[0].value
    assert {"#", "Page", "Movement", "Dense", "BM25", "RRF", "CE logit",
            "CE Confidence"} <= set(table.columns)
    assert table["CE Confidence"].between(0.0, 1.0).all()
    assert table["Movement"].str.contains("Rerank #1").any()

    assert any("<div class='tr-answer'>" in m.value for m in at.markdown)
    assert any("Source Evidence Inspector" in m.value for m in at.markdown)
    assert [s.label for s in at.selectbox] == ["Chunk rank"]

    # Compare toggle reveals the dense-vs-reranked callout.
    at.toggle[0].set_value(True).run()
    assert not list(at.exception)
    labels = [m.value for m in at.markdown]
    assert any("Dense-only #1" in value for value in labels)
    assert any("Hybrid + rerank #1" in value for value in labels)
    assert any("Rerank promoted page" in value for value in labels)


def test_trace_explains_an_abstention(fake_index):
    at = _boot(_trace(refused=True))
    assert not list(at.exception)
    labels = [m.value for m in at.markdown]
    assert any("ABSTAINED" in value for value in labels)
    assert any("Why the trace abstained" in value for value in labels)
    assert any("Query-term coverage" in value for value in labels)
    assert {m.label: m.value for m in at.metric}["Faithfulness / spans"] == "abstained"


@pytest.mark.skipif(
    not (ROOT / "data" / "index" / "chunks.jsonl").exists(),
    reason="needs a real local index; run `python run_pipeline.py` first",
)
def test_preset_runs_a_real_query_end_to_end():
    at = AppTest.from_file(str(APP), default_timeout=600).run()
    assert not list(at.exception)

    at.pills[0].set_value("Rank rescue · Meridian headcount").run()
    assert not list(at.exception)
    assert at.session_state["mode"] == "Hybrid + rerank"

    metrics = {m.label: m.value for m in at.metric}
    assert len(metrics) == 4
    assert metrics["Retrieval latency"].endswith(" ms")
    assert metrics["Rerank latency"].endswith(" ms")
    assert "44" in " ".join(m.value for m in at.markdown)

    table = at.dataframe[0].value
    assert "CE Confidence" in table.columns
    assert table["CE Confidence"].max() > 0.5
    assert any("Rerank #1" in str(v) for v in table["Movement"])
