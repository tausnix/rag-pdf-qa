"""TraceRAG: split-screen PDF QA with retrieval observability.

    streamlit run app.py

The **Ask & Trace** tab is a two-column workspace:

* left -- the answer card (faithfulness, citations, rank movement), a 4-metric
  latency/cost strip, the candidate ranking table with cross-encoder confidence
  bars, and an optional dense-vs-reranked diff;
* right -- the Source Evidence Inspector: pick any retrieved chunk, read it with
  the query terms highlighted, and see why a trap question abstained.

The **Benchmark** tab renders the saved 3-way reports straight from
``reports/*.json`` -- no metric is typed into the UI.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from src import retrieve as retrieve_module
from src.config import settings
from src.generate import answer as generate_answer
from src.generate import faithfulness
from src.ingest import ingest
from src.retrieve import get_retriever

st.set_page_config(page_title="TraceRAG", page_icon=":material/verified:", layout="wide")

CORPORA = {
    "Built-in handbook": settings.raw_dir / "demo_handbook.pdf",
    "SQuAD 2.0": settings.raw_dir / "squad_corpus.pdf",
}

MODES = ("Dense (MiniLM)", "Hybrid (RRF)", "Hybrid + rerank")
MODE_HELP = {
    "Dense (MiniLM)": "Cosine similarity over MiniLM embeddings only.",
    "Hybrid (RRF)": "Dense + BM25 fused with Reciprocal Rank Fusion.",
    "Hybrid + rerank": "Fuse, then score the top candidates with a cross-encoder.",
}

# Preset pills are keyed by the indexed source filename. Clicking one runs it.
PRESETS: dict[str, list[dict]] = {
    "demo_handbook.pdf": [
        {
            "label": "Exact lookup · Ember Point budget",
            "question": "What was the reported annual budget for Ember Point?",
            "mode": None,
        },
        {
            "label": "Rank rescue · Meridian headcount",
            "question": "What was the reported headcount for Meridian Depot?",
            "mode": "Hybrid + rerank",
        },
        {
            "label": "Trap · 2026 World Cup winner",
            "question": "Who won the 2026 FIFA World Cup?",
            "mode": None,
        },
    ],
    "squad_corpus.pdf": [
        {
            "label": "Exact lookup · Kublai Khan",
            "question": "What was the system of government created by Kublai Khan a product of?",
            "mode": None,
        },
        {
            "label": "Attribute check · Yuan dynasty",
            "question": "Which groups were given advantages in the Yuan dynasty?",
            "mode": "Hybrid + rerank",
        },
        {
            "label": "Trap · 2026 World Cup winner",
            "question": "Who won the 2026 FIFA World Cup?",
            "mode": None,
        },
    ],
}
CUSTOM_PRESETS = [
    {
        "label": "Trap · 2026 World Cup winner",
        "question": "Who won the 2026 FIFA World Cup?",
        "mode": None,
    }
]

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1280px;}
  .tr-kicker {font-size: 11px; letter-spacing: .2em; text-transform: uppercase;
              color: #9A9385; font-weight: 600; margin-bottom: 2px;}
  .tr-brand {font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
             font-size: 2.9rem; line-height: 1.05; letter-spacing: -.02em;
             color: #1F1E1D; margin: 0 0 2px 0;}
  .tr-sub {color: #5C574D; font-size: 15px; margin: 0 0 6px 0;}
  .tr-side-brand {font-family: "Iowan Old Style", Palatino, Georgia, serif;
                  font-size: 1.55rem; letter-spacing: -.01em; color: #1F1E1D;}
  .tr-q {color: #6B6558; font-size: 13px; margin: 0 0 8px 0;}
  .tr-q b {color: #1F1E1D;}
  .tr-answer {font-family: "Iowan Old Style", Palatino, Georgia, serif;
              font-size: 19px; line-height: 1.62; color: #1F1E1D; margin: 6px 0 4px 0;}
  .tr-pills {display: flex; flex-wrap: wrap; gap: 6px; align-items: center;}
  .tr-pill {display: inline-block; border: 1px solid #DED8C8; border-radius: 999px;
            padding: 2px 10px; font-size: 11.5px; font-weight: 600; letter-spacing: .03em;
            color: #5C574D; background: #FAF9F5;}
  .tr-pill.pass {border-color: #9CC3A6; color: #2F6B48; background: #EEF5EF;}
  .tr-pill.warn {border-color: #E0C08A; color: #8A6220; background: #FBF3E3;}
  .tr-pill.fail {border-color: #D9A2A2; color: #8C2F2F; background: #F9ECEC;}
  .tr-move {font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
            font-size: 11.5px; color: #7A4A2E; background: #F6EDE4;
            border: 1px solid #E5D3C2; border-radius: 6px; padding: 2px 7px;}
  .tr-cite {display: inline; background: #F0EEE6; border: 1px solid #DED8C8;
            color: #8A4B2F; padding: 0 6px; border-radius: 999px;
            font-size: 12px; font-weight: 600; white-space: nowrap;}
  .tr-chunk {background: #FAF9F5; border: 1px solid #E4DFD2; border-radius: 12px;
             padding: 14px 16px; font-size: 13.5px; line-height: 1.7; color: #2B2926;
             max-height: 380px; overflow-y: auto;}
  .tr-badges {display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 10px 0;}
  .tr-badge {display: inline-block; background: #F0EEE6; border: 1px solid #DED8C8;
             border-radius: 6px; padding: 2px 8px; font-size: 11.5px;
             font-weight: 600; color: #5C574D;}
  .tr-abstain {background: #FBF3E3; border: 1px solid #E0C08A; border-radius: 12px;
               padding: 12px 14px; font-size: 13px; color: #5A4317; margin-top: 12px;}
  .tr-abstain b {color: #8A6220;}
  .tr-abstain p {margin: 6px 0 0 0;}
  .tr-empty {border: 1px dashed #D8D2C2; border-radius: 16px; padding: 54px 28px;
             text-align: center; color: #8A8578; background: #FCFBF8;}
  .tr-empty h3 {font-family: "Iowan Old Style", Palatino, Georgia, serif;
                color: #1F1E1D; font-size: 20px; margin: 0 0 6px 0;}
  .tr-diff-label {font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
                  color: #9A9385; font-weight: 600; margin-bottom: 4px;}
  .tr-verdict {font-size: 12.5px; color: #5C574D; margin-top: 4px;}
  .metric-card {border: 1px solid #E4DFD2; border-radius: 12px; padding: 14px 16px;
                background: #FFFFFF;}
  /* Metric labels wrap instead of truncating in narrow 4-up columns */
  [data-testid="stMetricLabel"] {overflow: visible;}
  [data-testid="stMetricLabel"] p {white-space: normal; line-height: 1.2; font-size: 0.8rem;}
  [data-testid="stMetricValue"] {font-size: 1.5rem; overflow: visible;}
  [data-testid="stMetricValue"] > div {overflow: visible; text-overflow: clip; white-space: normal;}
  [data-testid="stMetric"] {height: auto;}
  [data-testid="stMetric"] > div {height: auto;}
  [data-testid="stMetricDelta"] {font-size: 0.75rem;}
  .metric-card h3 {margin: 0 0 4px 0; font-size: 12px; color: #8A8578;
                   font-weight: 600; letter-spacing: .04em;}
  .metric-card p {margin: 0; font-size: 22px; font-weight: 650; color: #1F1E1D;}
  mark {background: #F4D3AC; color: #1F1E1D; border-radius: 3px; padding: 0 2px;}
  code {background: #F0EEE6; border-radius: 4px; padding: 0 4px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resources (ONNX embedder, ONNX cross-encoder, Chroma index)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_retriever():
    return get_retriever()


@st.cache_resource(show_spinner=False)
def load_reranker():
    from src.rerank import get_reranker

    return get_reranker()


def reset_caches() -> None:
    """Drop the retriever/reranker singletons after an index rebuild."""
    import src.rerank as rerank_module

    rerank_module._CACHE = None
    retrieve_module._CACHE = None
    load_retriever.clear()
    load_reranker.clear()


def index_exists() -> bool:
    return settings.bm25_pickle.exists() and settings.chunks_jsonl.exists()


def corpus_stats() -> dict:
    if not settings.chunks_jsonl.exists():
        return {}
    rows = [
        json.loads(line)
        for line in settings.chunks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "chunks": len(rows),
        "pages": len({r["page"] for r in rows}),
        "source": rows[0]["source"] if rows else "-",
    }


def active_source() -> str:
    return corpus_stats().get("source", "")


def index_pdf(pdf: Path) -> bool:
    """(Re)index ``pdf`` and refresh every cache. Returns True on success."""
    try:
        with st.spinner(f"Chunking, embedding and indexing {pdf.name}..."):
            summary = ingest(pdf, reset=True)
        reset_caches()
        clear_query_state()
        st.success(f"Indexed {summary['chunks']} chunks from {summary['pages']} pages.")
        return True
    except SystemExit as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Indexing failed: {exc}")
    return False


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    if "trace" not in st.session_state:
        st.session_state["trace"] = None
    if "qid" not in st.session_state:
        st.session_state["qid"] = 0
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None
    if "corpus" not in st.session_state:
        st.session_state["corpus"] = "Built-in handbook"
    if "mode" not in st.session_state:
        st.session_state["mode"] = "Hybrid (RRF)"
    if "chunks_to_model" not in st.session_state:
        st.session_state["chunks_to_model"] = settings.context_k
    if "rerank_candidates" not in st.session_state:
        st.session_state["rerank_candidates"] = 10
    if "use_gemini" not in st.session_state:
        st.session_state["use_gemini"] = bool(settings.gemini_api_key)


def clear_query_state() -> None:
    st.session_state["trace"] = None
    st.session_state["pending_question"] = None
    st.session_state["preset_pill"] = None


def presets_for(source: str) -> list[dict]:
    if source in PRESETS:
        return PRESETS[source]
    return CUSTOM_PRESETS if source else PRESETS["demo_handbook.pdf"]


def _apply_preset() -> None:
    mapping = st.session_state.get("_preset_map") or {}
    preset = mapping.get(st.session_state.get("preset_pill"))
    if not preset:
        return
    st.session_state["pending_question"] = preset["question"]
    if preset.get("mode"):
        st.session_state["mode"] = preset["mode"]


def _clear_preset() -> None:
    st.session_state["preset_pill"] = None


# --------------------------------------------------------------------------- #
# Trace helpers
# --------------------------------------------------------------------------- #
_STOP = {
    "the", "and", "for", "was", "were", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "did", "does", "do", "is", "are", "of", "in",
    "on", "at", "to", "by", "with", "from", "that", "this", "there", "their",
    "its", "it", "as", "an", "a", "be", "been", "has", "have", "had", "per",
}


def query_terms(question: str) -> list[str]:
    terms = [t for t in re.findall(r"[A-Za-z0-9']+", question.lower())
             if len(t) >= 2 and t not in _STOP]
    # longest first so "headcount" is wrapped before "head"
    return sorted(set(terms), key=len, reverse=True)


def highlight(text: str, question: str, limit: int = 1400) -> str:
    """Escape the chunk, then wrap every query term in <mark>."""
    snippet = text[:limit] + ("..." if len(text) > limit else "")
    escaped = html.escape(snippet)
    terms = query_terms(question)
    if not terms:
        return escaped
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


def answer_markup(text: str) -> str:
    """Escape the model output, then style [p.N] citations."""
    safe = re.sub(
        r"\[p\.(\d+)\]",
        r"<span class='tr-cite'>p.\1</span>",
        html.escape(text),
    )
    return safe.replace("\n", "<br>")


def status_pill(label: str, tone: str = "") -> str:
    return f"<span class='tr-pill {tone}'>{html.escape(label)}</span>"


def fmt_float(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def faith_status(faith: dict | None, result: dict) -> tuple[str, str]:
    if result.get("refused"):
        return "ABSTAINED", "warn"
    if faith is None:
        return "NOT CHECKED", "warn"
    ok = faith["score"] == 1.0 and faith["numeric_grounding"]
    return ("PASS", "pass") if ok else ("FAIL", "fail")


# --------------------------------------------------------------------------- #
# Query execution (one place, so the trace is always consistent)
# --------------------------------------------------------------------------- #
def run_query(question: str, mode: str, k: int, top_n: int | None,
              use_llm: bool) -> dict:
    retriever = load_retriever()
    timings = {"retrieval_ms": 0.0, "rerank_ms": 0.0, "generation_ms": 0.0}
    cross_encoder = None

    t0 = time.perf_counter()
    if mode == "Dense (MiniLM)":
        candidates = retriever.vector_search(question, k=max(k, 10))
    elif mode == "Hybrid (RRF)":
        candidates = retriever.hybrid_search(question, k=max(k, 10))
    else:
        candidates = retriever.hybrid_search(question, k=top_n or settings.rerank_top_n)
    t1 = time.perf_counter()
    timings["retrieval_ms"] = (t1 - t0) * 1000.0

    rrf_order = [h.chunk_id for h in candidates]
    if mode == "Hybrid + rerank":
        reranker = load_reranker()
        ranked = reranker.rerank(question, candidates, k=len(candidates),
                                 top_n=top_n or settings.rerank_top_n)
        t2 = time.perf_counter()
        timings["rerank_ms"] = (t2 - t1) * 1000.0
        cross_encoder = {"pairs": reranker.last_batch, "tokens": reranker.last_seq_len}
        hits = ranked[:k]
        display = ranked
    else:
        hits = candidates[:k]
        display = candidates

    # Dense-only top-1, for the "Compare vs. Dense Baseline" callout. The query
    # embedding is already cached, so this is a few milliseconds.
    baseline = retriever.vector_search(question, k=1)
    dense_baseline = None
    if baseline:
        b = baseline[0]
        dense_baseline = {
            "chunk_id": b.chunk_id,
            "page": b.page,
            "text": b.text,
            "score": b.vector_score,
        }

    t3 = time.perf_counter()
    result = generate_answer(question, hits, use_llm=use_llm)
    timings["generation_ms"] = (time.perf_counter() - t3) * 1000.0

    faith = None
    if not result["refused"]:
        faith = faithfulness(result["answer"], hits)

    hit_rows = []
    for rank, h in enumerate(display, start=1):
        rr_rank = rrf_order.index(h.chunk_id) + 1 if h.chunk_id in rrf_order else None
        movement = None
        if mode == "Hybrid + rerank" and rr_rank:
            delta = rr_rank - rank
            movement = f"RRF #{rr_rank} → Rerank #{rank}"
            if delta:
                movement += f" {'↑' if delta > 0 else '↓'}{abs(delta)}"
        hit_rows.append(
            {
                "chunk_id": h.chunk_id,
                "page": h.page,
                "text": h.text,
                "dense_score": h.vector_score,
                "bm25_score": h.bm25_score,
                "rrf_score": h.score if mode != "Dense (MiniLM)" else None,
                "rrf_rank": rr_rank if mode == "Hybrid + rerank" else None,
                "rerank_rank": rank if mode == "Hybrid + rerank" else None,
                "logit": h.rerank_score,
                "prob": h.rerank_prob,
                "movement": movement,
            }
        )

    return {
        "question": question,
        "mode": mode,
        "top_n": top_n,
        "chunks_to_model": len(hits),
        "timings": timings,
        "cross_encoder": cross_encoder,
        "result": result,
        "faith": faith,
        "hits": hit_rows,
        "dense_baseline": dense_baseline,
    }


# --------------------------------------------------------------------------- #
# Ask & Trace rendering
# --------------------------------------------------------------------------- #
def render_presets(source: str) -> None:
    presets = presets_for(source)
    labels = [p["label"] for p in presets]
    st.session_state["_preset_map"] = {p["label"]: p for p in presets}
    if st.session_state.get("preset_pill") not in labels:
        st.session_state["preset_pill"] = None

    st.caption("One-click demos")
    st.pills(
        "Preset questions",
        labels,
        key="preset_pill",
        on_change=_apply_preset,
        label_visibility="collapsed",
    )


def render_answer_card(trace: dict) -> None:
    result = trace["result"]
    faith = trace["faith"]
    status, tone = faith_status(faith, result)

    with st.container(border=True):
        st.markdown(
            f"<div class='tr-q'><b>Q</b> · {html.escape(trace['question'])}</div>",
            unsafe_allow_html=True,
        )
        pills = [status_pill(status, tone), status_pill(trace["mode"])]
        if result["citations"]:
            pages = ", ".join(f"p.{p}" for p in result["citations"])
            pills.append(status_pill(f"cited {pages}"))
        if trace["mode"] == "Hybrid + rerank" and trace["hits"]:
            movement = trace["hits"][0].get("movement")
            if movement:
                pills.append(f"<span class='tr-move'>{html.escape(movement)}</span>")
        st.markdown("<div class='tr-pills'>" + "".join(pills) + "</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='tr-answer'>{answer_markup(result['answer'])}</div>",
            unsafe_allow_html=True,
        )
        if result.get("cached"):
            st.caption("Replayed from the Gemini response cache - no quota used in this run.")
        if result.get("llm_error"):
            st.caption(
                f"Gemini call failed ({result['llm_error']}) - showing the extractive answer."
            )
        if trace["mode"] == "Hybrid + rerank" and trace["hits"]:
            up = sum(1 for h in trace["hits"]
                     if h.get("rrf_rank") and h["rrf_rank"] > h["rerank_rank"])
            down = sum(1 for h in trace["hits"]
                       if h.get("rrf_rank") and h["rrf_rank"] < h["rerank_rank"])
            st.caption(f"Cross-encoder moved {up} candidate(s) up and {down} down. "
                       "Ranking uses the raw logit; CE confidence = sigmoid(logit).")


def render_metrics(trace: dict) -> None:
    timings = trace["timings"]
    result = trace["result"]
    faith = trace["faith"]
    ce = trace.get("cross_encoder") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval latency", f"{timings['retrieval_ms']:.1f} ms", border=True)
    if trace["mode"] == "Hybrid + rerank":
        c2.metric("Rerank latency", f"{timings['rerank_ms']:.1f} ms", border=True,
                  help=f"{ce.get('pairs', 0)} pairs × {ce.get('tokens', 0)} padded tokens")
    else:
        c2.metric("Rerank latency", "off", border=True,
                  help="Switch the sidebar to Hybrid + rerank to score candidates.")

    if faith is None:
        c3.metric("Faithfulness / spans", "abstained", border=True,
                  help="The model refused, so there is no answer to verify.")
    else:
        c3.metric("Faithfulness / spans", f"{faith['score'] * 100:.0f}%", border=True,
                  help=f"{faith['supported']}/{faith['sentences']} answer sentences supported. "
                       "Lexical support + numeric grounding; see 'How it works'.")

    basis = "actual usage" if result.get("billable") else "projected list price"
    c4.metric("Projected cost / 1k", f"${result['cost_usd'] * 1000:.2f}", border=True,
              help=f"{basis}: {result['usage']['input_tokens']:,} in / "
                   f"{result['usage']['output_tokens']:,} out tokens")


def render_compare(trace: dict) -> None:
    if not st.toggle("Compare vs. Dense Baseline", key=f"compare_{st.session_state['qid']}"):
        return

    base = trace.get("dense_baseline")
    top = trace["hits"][0] if trace["hits"] else None
    if base is None or top is None:
        st.info("No candidates to compare.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='tr-diff-label'>Dense-only #1</div>", unsafe_allow_html=True)
        st.markdown(f"**Page {base['page']}** · cosine {fmt_float(base['score'])}")
        st.caption(base["text"][:260] + ("..." if len(base["text"]) > 260 else ""))
    with right:
        st.markdown("<div class='tr-diff-label'>Hybrid + rerank #1</div>",
                    unsafe_allow_html=True)
        st.markdown(f"**Page {top['page']}** · CE {fmt_float(top.get('logit'), 2)}")
        st.caption(top["text"][:260] + ("..." if len(top["text"]) > 260 else ""))

    if base["chunk_id"] == top["chunk_id"]:
        verdict = "Both retrievers agree on the top chunk; reranking only reordered the tail."
    else:
        verdict = (
            f"Rerank promoted page {top['page']} over the dense favourite page "
            f"{base['page']}"
        )
        if trace["mode"] != "Hybrid + rerank":
            verdict += " (compare is most useful in Hybrid + rerank mode)"
        verdict += "."
    st.markdown(f"<div class='tr-verdict'>{verdict}</div>", unsafe_allow_html=True)


def render_candidates(trace: dict) -> None:
    rows = []
    for i, h in enumerate(trace["hits"], start=1):
        # Compact movement for the table; the full "RRF #n -> Rerank #m" label
        # is already in the answer-card pill.
        move = "-"
        rr, rk = h.get("rrf_rank"), h.get("rerank_rank")
        if rr and rk:
            move = f"#{rr} → #{rk}"
            delta = rr - rk
            if delta:
                move += f" {'↑' if delta > 0 else '↓'}{abs(delta)}"
        rows.append(
            {
                "#": i,
                "Page": h["page"],
                "Movement": move,
                "Dense": round(h["dense_score"], 3) if h["dense_score"] is not None else None,
                "BM25": round(h["bm25_score"], 1) if h["bm25_score"] is not None else None,
                "CE Confidence": h["prob"] if h["prob"] is not None else 0.0,
            }
        )
    st.markdown("#### Candidate ranking")
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Page": st.column_config.NumberColumn("Page", width="small"),
            "Movement": st.column_config.TextColumn("Movement", width="small"),
            "Dense": st.column_config.NumberColumn("Dense", width="small", format="%.3f"),
            "BM25": st.column_config.NumberColumn("BM25", width="small", format="%.1f"),
            "CE Confidence": st.column_config.ProgressColumn(
                "CE Confidence", min_value=0.0, max_value=1.0, format="%.3f",
                width="medium",
            ),
        },
    )
    if trace["mode"] != "Hybrid + rerank":
        st.caption("CE confidence is populated in Hybrid + rerank mode.")


def render_abstention(trace: dict) -> None:
    top = trace["hits"][0] if trace["hits"] else None
    terms = query_terms(trace["question"])
    parts = ["<div class='tr-abstain'><b>Why the trace abstained</b>"]
    if top is None:
        parts.append("<p>Retrieval returned no chunks, so there was no context to answer from.</p>")
    else:
        present = [t for t in terms
                   if re.search(rf"\b{re.escape(t)}\b", top["text"], re.IGNORECASE)]
        if top.get("prob") is not None:
            parts.append(
                f"<p>Top candidate confidence: <b>{top['prob']:.3f}</b> "
                f"(sigmoid of CE logit {top['logit']:+.2f}).</p>"
            )
        else:
            parts.append("<p>Cross-encoder reranking is off, so no CE confidence is available.</p>")
        parts.append(
            f"<p>Query-term coverage in that chunk: <b>{len(present)}/{len(terms)}</b>"
            + (f" ({', '.join(html.escape(t) for t in present)})" if present else "")
            + ".</p>"
        )
    parts.append(
        "<p>The generator returned the refusal sentence instead of answering from "
        "unrelated context. This is the intended behaviour on an out-of-domain trap.</p></div>"
    )
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_inspector(trace: dict) -> None:
    hits = trace["hits"]
    with st.container(border=True):
        st.markdown("#### :material/manage_search: Source Evidence Inspector")
        if not hits:
            st.info("No chunks retrieved for this question.")
            return

        chosen = st.selectbox(
            "Chunk rank",
            options=list(range(len(hits))),
            index=0,
            format_func=lambda i: f"Rank #{i + 1} · Page {hits[i]['page']}",
            key=f"inspector_{st.session_state['qid']}",
        )
        h = hits[chosen]

        badges = [
            f"<span class='tr-badge'>Page {h['page']}</span>",
            f"<span class='tr-badge'>Rank #{chosen + 1}</span>",
        ]
        if h.get("rrf_rank"):
            badges.append(f"<span class='tr-badge'>RRF #{h['rrf_rank']}</span>")
        if h.get("prob") is not None:
            badges.append(f"<span class='tr-badge'>CE {h['prob']:.3f}</span>")
        st.markdown("<div class='tr-badges'>" + "".join(badges) + "</div>",
                    unsafe_allow_html=True)

        st.markdown(
            f"<div class='tr-chunk'>{highlight(h['text'], trace['question'])}</div>",
            unsafe_allow_html=True,
        )

        scores = []
        if h["dense_score"] is not None:
            scores.append(f"dense {h['dense_score']:.3f}")
        if h["bm25_score"] is not None:
            scores.append(f"BM25 {h['bm25_score']:.1f}")
        if h["rrf_score"] is not None:
            scores.append(f"RRF {h['rrf_score']:.4f}")
        if h["logit"] is not None:
            scores.append(f"CE logit {h['logit']:+.2f}")
        if scores:
            st.caption(" · ".join(scores))

        faith = trace["faith"]
        if faith is not None:
            st.caption(
                f"Grounding: {faith['supported']}/{faith['sentences']} answer sentences "
                f"supported · unsupported numbers: "
                f"{', '.join(faith['unsupported_numbers']) or 'none'}"
            )

        if trace["result"]["refused"]:
            render_abstention(trace)


# --------------------------------------------------------------------------- #
# Benchmark tab helpers
# --------------------------------------------------------------------------- #
def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def report_table(data: dict) -> list[dict]:
    rows = []
    for key, label in (
        ("vector_only", "MiniLM dense"),
        ("hybrid", "Hybrid MiniLM + BM25 (RRF)"),
        ("rerank", "Hybrid + cross-encoder rerank"),
    ):
        r = data.get(key)
        if not r:
            continue
        rows.append(
            {
                "Retriever": label,
                "Hit@3": pct(r.get("hit_rate@3")),
                "MRR@10": pct(r.get("mrr@10")),
                "Ans-in-Context@5": pct(r.get("answer_in_context@5")),
                "Abstention": pct(r.get("abstention_accuracy")),
                "Faithfulness": pct(r.get("faithfulness")),
                "p50": f"{r['latency_p50_ms']:.1f} ms",
                "p95": f"{r['latency_p95_ms']:.1f} ms",
            }
        )
    return rows


def render_report(path: Path, title: str, note: str) -> None:
    data = load_report(path)
    st.subheader(title)
    if not data:
        st.info(f"No report at `{path.name}`. Rebuild it with the command in the README.")
        return

    corpus = data.get("corpus", {})
    st.caption(
        f"{note} · {corpus.get('chunks', '-')} chunks · "
        f"embedder `{corpus.get('embedder', '-')}` · "
        f"generator {data.get('config', {}).get('generator', '-')}"
    )
    st.dataframe(report_table(data), hide_index=True, width="stretch")

    rerank = data.get("rerank") or {}
    if rerank:
        cols = st.columns(4)
        cards = (
            ("RERANK HIT@3", pct(rerank.get("hit_rate@3"))),
            ("RERANK MRR@10", pct(rerank.get("mrr@10"))),
            ("RETRIEVAL P50", f"{rerank.get('retrieval_p50_ms', 0):.1f} ms"),
            ("RERANK P50", f"{rerank.get('rerank_p50_ms', 0):.1f} ms"),
        )
        for col, (label, value) in zip(cols, cards):
            col.markdown(
                f"<div class='metric-card'><h3>{label}</h3><p>{value}</p></div>",
                unsafe_allow_html=True,
            )
        delta = data.get("delta_rerank_vs_hybrid", {})
        if delta:
            st.caption(
                f"Rerank lift over RRF - Hit@3 {delta['hit_rate@3'] * 100:+.1f} pts · "
                f"MRR@10 {delta['mrr@10'] * 100:+.1f} pts · "
                f"added {rerank.get('rerank_p50_ms', 0):.0f} ms/query"
            )
        if rerank.get("cost_per_1k_queries_usd") is not None:
            tokens = rerank.get("tokens_per_query", {})
            st.caption(
                f"Tokens/query {tokens.get('input', 0):.0f} in / "
                f"{tokens.get('output', 0):.0f} out · "
                f"${rerank['cost_per_1k_queries_usd']:.2f} per 1,000 queries · "
                f"{rerank.get('cost_basis', '')}"
            )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
init_state()

# Deep link: ?q=<question>&mode=<mode> runs once on load. Handy for README/issue
# links ("open this exact trace"). Must run before the sidebar widgets exist, or
# assigning to st.session_state["mode"] would raise.
_deep_q = st.query_params.get("q")
if _deep_q and st.session_state.get("_deep_link") != _deep_q:
    st.session_state["_deep_link"] = _deep_q
    st.session_state["pending_question"] = _deep_q
    if st.query_params.get("mode") in MODES:
        st.session_state["mode"] = st.query_params["mode"]

with st.sidebar:
    st.markdown("### :material/verified: TraceRAG")
    st.caption("Hybrid RRF + int8 ONNX reranking, traced end to end.")

    st.markdown("#### :material/folder_open: Corpus")
    corpus = st.radio(
        "Corpus",
        list(CORPORA) + ["Custom PDF"],
        key="corpus",
        label_visibility="collapsed",
    )

    if corpus == "Custom PDF":
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded and st.button("Index this PDF", width="stretch", type="primary"):
            dest = settings.raw_dir / uploaded.name
            dest.write_bytes(uploaded.getbuffer())
            if index_pdf(dest):
                st.rerun()
    else:
        target = CORPORA[corpus]
        if not target.exists():
            st.warning(f"`{target.name}` is not built yet.")
            if corpus == "SQuAD 2.0" and st.button("Build SQuAD 2.0 corpus", type="primary"):
                with st.spinner("Sampling SQuAD 2.0 validation contexts..."):
                    proc = subprocess.run(
                        [sys.executable, "-m", "src.build_squad_benchmark",
                         "--contexts", "420", "--target-chunks", "231"],
                        cwd=settings.root, capture_output=True, text=True,
                    )
                if proc.returncode == 0:
                    reset_caches()
                    st.rerun()
                else:
                    st.error("Build failed - it needs the `datasets` package and network access.")
        elif active_source() != target.name:
            # First activation of this corpus: index it once, with a spinner.
            if index_pdf(target):
                st.rerun()

    stats = corpus_stats()
    if stats:
        st.caption(
            f"{stats['chunks']} chunks · {stats['pages']} pages · `{stats['source']}`"
        )
    else:
        st.info("No index yet. Pick a corpus above or run `python run_pipeline.py`.")

    st.divider()

    st.markdown("#### :material/tune: Retrieval")
    mode = st.radio(
        "Mode",
        list(MODES),
        key="mode",
        help="\n\n".join(f"**{m}** - {MODE_HELP[m]}" for m in MODES),
    )
    rerank_top_n = None
    if mode == "Hybrid + rerank":
        rerank_top_n = st.slider(
            "Rerank candidates", 5, 15, key="rerank_candidates",
            help="How many RRF candidates the cross-encoder scores. "
                 "10 keeps ~95% MRR@10 at ~1/3 less CPU than 15.",
        )
        st.caption("First run loads the int8 ONNX model (~22 MB), then it is cached.")

    k = st.slider("Chunks sent to the model", 1, 10, key="chunks_to_model")

    use_llm = st.toggle(
        "Use Gemini",
        key="use_gemini",
        help="Needs GEMINI_API_KEY. Off = extractive answer (free, offline).",
    )
    if use_llm and not settings.gemini_api_key:
        st.warning("No GEMINI_API_KEY found - falling back to extractive answers.")

    st.divider()
    if settings.gemini_api_key:
        st.caption(
            f"Gemini pacing: {settings.gemini_min_interval_s:.0f}s between calls · "
            f"{settings.gemini_max_retries} retries · response cache on disk."
        )
    else:
        st.caption("Offline extractive mode. Add a Gemini key for LLM answers.")


# --------------------------------------------------------------------------- #
# Header + tabs
# --------------------------------------------------------------------------- #
st.markdown(
    "<div class='tr-kicker'>Retrieval observability · MiniLM + BM25 + ONNX cross-encoder</div>"
    "<div class='tr-brand'>TraceRAG</div>"
    "<div class='tr-sub'>Hybrid RRF + int8 ONNX reranked PDF QA with retrieval "
    "observability - every answer ships with its evidence.</div>",
    unsafe_allow_html=True,
)

tab_chat, tab_bench, tab_about = st.tabs(
    [":material/search: Ask & Trace", ":material/bar_chart: Benchmark",
     ":material/info: How it works"]
)

# ------------------------------- ask & trace ------------------------------- #
with tab_chat:
    render_presets(active_source())

    typed = st.chat_input("Ask something about this document...", on_submit=_clear_preset)
    pending = st.session_state.get("pending_question")
    st.session_state["pending_question"] = None
    question = typed or pending

    if question:
        spinner = (
            "Retrieving, reranking and generating..." if mode == "Hybrid + rerank"
            else "Retrieving and generating..."
        )
        with st.spinner(spinner):
            trace = run_query(question, mode, k, rerank_top_n, use_llm)
        st.session_state["qid"] += 1
        st.session_state["trace"] = trace

    trace = st.session_state.get("trace")
    left, right = st.columns([1.3, 1], gap="large")

    with left:
        if trace is None:
            st.markdown(
                "<div class='tr-empty'><h3>Every answer is inspectable</h3>"
                "<p>Pick a preset above or ask your own question. The left column shows "
                "the answer and telemetry; the right column shows the exact chunks the "
                "answer was built from.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            render_answer_card(trace)
            render_metrics(trace)
            render_compare(trace)
            render_candidates(trace)

    with right:
        if trace is None:
            with st.container(border=True):
                st.markdown("#### :material/manage_search: Source Evidence Inspector")
                st.markdown(
                    "<div class='tr-empty'><p>Retrieved chunks, highlighted query terms "
                    "and the reason for an abstention will appear here.</p></div>",
                    unsafe_allow_html=True,
                )
        else:
            render_inspector(trace)

# -------------------------------- benchmark -------------------------------- #
with tab_bench:
    st.header(":material/bar_chart: Retrieval benchmark")
    st.caption("Read directly from `reports/*.json` - the same files the README quotes.")
    render_report(
        settings.reports_dir / "squad_benchmark.json",
        ":material/table_chart: SQuAD 2.0 · 230 chunks",
        "167 passages (129 with questions + 38 distractors) · "
        "150 answerable / 50 unanswerable / 25 out-of-domain",
    )
    st.divider()
    render_report(
        settings.reports_dir / "benchmark.json",
        ":material/table_chart: Synthetic handbook · 230 chunks",
        "61-page generated corpus · 150 answerable / 50 unanswerable / 25 out-of-domain",
    )
    top10 = settings.reports_dir / "squad_benchmark_top10.json"
    if top10.exists():
        with st.expander("Rerank candidate experiment: top-15 vs top-10"):
            d15 = load_report(settings.reports_dir / "squad_benchmark.json") or {}
            d10 = load_report(top10) or {}
            rows = []
            for label, d in (("top 15 (default)", d15), ("top 10", d10)):
                r = d.get("rerank") or {}
                if r:
                    rows.append(
                        {
                            "Candidates": label,
                            "Hit@3": pct(r.get("hit_rate@3")),
                            "MRR@10": pct(r.get("mrr@10")),
                            "rerank p50": f"{r.get('rerank_p50_ms', 0):.1f} ms",
                            "total p50": f"{r.get('latency_p50_ms', 0):.1f} ms",
                        }
                    )
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
                st.caption(
                    "Scoring 10 candidates loses ~0.6 MRR points but cuts rerank time by "
                    "about a third, which is why the interactive slider defaults to 10."
                )

# -------------------------------- how it works ----------------------------- #
with tab_about:
    st.header(":material/info: How it works")
    st.markdown(
        """
**Indexing** - `PyPDFLoader` reads the PDF page by page (page numbers are kept for
citations). Pages are split into ~800-character overlapping chunks, embedded with
`all-MiniLM-L6-v2`, and stored in **ChromaDB**. A **BM25** index is built over the
same chunks for exact-term matching. If Chroma's Rust bindings are unavailable (seen on
Streamlit Cloud + Python 3.12) retrieval falls back to a NumPy cosine index with identical
scores, so the app keeps working. The corpus selector re-indexes the chosen PDF
once; the retriever and cross-encoder are then cached for the session.

**Retrieval modes** - *dense* searches embeddings only; *hybrid* fuses dense + BM25
with **Reciprocal Rank Fusion** (`RRF(d) = Σ 1 / (60 + rank_i(d))`); *rerank* takes
the fused top-N and scores each `(query, chunk)` pair with
`ms-marco-MiniLM-L-6-v2`, an int8 ONNX cross-encoder run on CPU with no PyTorch.

**Why reranking is opt-in** - the cross-encoder reads the question and the chunk
together, which fixes "same entity, wrong attribute" confusions that RRF cannot. But
it is a second transformer pass over every candidate, and on a 2-core CPU that is
hundreds of milliseconds. The slider trades quality for latency: 10 candidates keeps
~95% MRR@10 on SQuAD at roughly two thirds of the top-15 cost. Pairs are padded only
to the longest pair in each batch (capped at 256 tokens), so short chunks are cheap.

**The trace** - every answer records retrieval, rerank and generation timings
separately, the exact RRF rank of each candidate and where the cross-encoder moved
it, both component scores, the relevance probability, the token counts and the
projected USD cost. **Faithfulness** is a lexical check: each answer sentence must
share enough content words with a context sentence, and every number in the answer
must appear in the context. With the offline extractive path it passes by
construction; run with a Gemini key to make it meaningful.

**Presets** - the three pills are a guided tour: an *exact lookup* that both
retrievers get right, a *rank rescue* where dense misses and the cross-encoder
promotes the gold page, and an out-of-domain *trap* that must abstain.

**Benchmark** - the JSON reports are generated by `src/benchmark.py`; the app never
hard-codes a metric. Hit@3 asks whether any gold chunk is in the top 3, MRR@10
measures the rank of the first gold chunk, Ans-in-Context@5 is the generator's
ceiling, and abstention is measured on the 75 negatives.

**Free-tier safety** - Gemini calls are paced (`GEMINI_MIN_INTERVAL_S`, default 4 s),
429s back off using `RetryInfo`/`Retry-After`, and every successful response is
cached by `sha256(model + temperature + prompt)` under `~/.cache/rag-pdf-qa/gemini`,
so an interrupted 225-query benchmark resumes without spending quota twice.
        """
    )
