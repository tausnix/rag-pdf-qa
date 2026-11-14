"""Streamlit UI: switch corpora and retrieval modes, ask questions, inspect the trace.

    streamlit run app.py

Two tabs matter:

* **Ask** -- retrieval mode (dense / hybrid / rerank), a rerank-candidate slider, and
  an expandable *Retrieval & Rerank Trace* under every answer: latency pills, token
  cost, a faithfulness check, the candidate rank table (RRF rank -> rerank rank) and
  the retrieved text with the query terms highlighted.
* **Benchmark** -- the saved 3-way reports for both corpora, read straight from
  ``reports/*.json`` (no numbers are typed into the app).
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

st.set_page_config(page_title="RAG PDF QA", page_icon="📄", layout="wide")

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

CSS = """
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  .metric-card {border:1px solid #2a2f3a; border-radius:12px; padding:14px 16px; background:#11151c;}
  .metric-card h3 {margin:0 0 4px 0; font-size:12px; color:#8b96a8; font-weight:500; letter-spacing:.04em;}
  .metric-card p {margin:0; font-size:22px; font-weight:650;}
  .cite {display:inline-block; background:#1b2430; border:1px solid #2f3b4d; border-radius:999px;
         padding:2px 10px; margin-right:6px; font-size:12px; color:#9fd0ff;}
  .pillbar {display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px 0;}
  .pill {display:inline-block; border:1px solid #2f3b4d; border-radius:999px; padding:3px 10px;
         font-size:12px; color:#c7d1df; background:#141a23;}
  .pill b {color:#8b96a8; font-weight:500; margin-right:4px;}
  .pill.ok {border-color:#2f6b48; color:#b7f0cd;}
  .pill.warn {border-color:#7a5a1f; color:#ffe1a3;}
  .pill.bad {border-color:#7d3340; color:#ffc3cd;}
  .chunk {border-left:3px solid #2f3b4d; padding:6px 12px; margin-bottom:10px; font-size:13px; color:#b9c3d1;}
  mark {background:#3b4f22; color:#e8ffc2; border-radius:3px; padding:0 2px;}
  .muted {color:#8b96a8; font-size:12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resources
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
        with st.spinner(f"Chunking, embedding and indexing {pdf.name}…"):
            summary = ingest(pdf, reset=True)
        reset_caches()
        st.session_state.messages = []
        st.success(f"Indexed {summary['chunks']} chunks from {summary['pages']} pages.")
        return True
    except SystemExit as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Indexing failed: {exc}")
    return False


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


def highlight(text: str, question: str, limit: int = 900) -> str:
    """Escape the chunk, then wrap every query term in <mark>."""
    snippet = text[:limit] + ("…" if len(text) > limit else "")
    escaped = html.escape(snippet)
    terms = query_terms(question)
    if not terms:
        return escaped
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


def pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def fmt_float(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def pill(label: str, value: str, tone: str = "") -> str:
    return f"<span class='pill {tone}'><b>{label}</b> {value}</span>"


def candidate_rows(hits: list[dict]) -> list[dict]:
    """Build the rank table. ``hits`` are already in display order."""
    rows = []
    for i, h in enumerate(hits, start=1):
        rows.append(
            {
                "#": i,
                "Page": h["page"],
                "Dense": fmt_float(h.get("dense_score")),
                "BM25": fmt_float(h.get("bm25_score"), 2),
                "RRF": fmt_float(h.get("rrf_score")),
                "Logit": fmt_float(h.get("logit"), 2),
                "P(relevant)": fmt_float(h.get("prob"), 3),
                "Movement": h.get("movement") or "—",
            }
        )
    return rows


def render_trace(trace: dict) -> None:
    timings = trace["timings"]
    faith = trace.get("faith")
    result = trace["result"]

    with st.expander("🔎 Retrieval & Rerank Trace", expanded=False):
        # ------------------------------------------------ latency & cost pills
        pills = [pill("retrieval", f"{timings['retrieval_ms']:.1f} ms")]
        if trace["mode"] == "Hybrid + rerank":
            pills.append(pill("rerank", f"{timings['rerank_ms']:.1f} ms", "warn"))
            ce = trace.get("cross_encoder") or {}
            if ce.get("pairs"):
                pills.append(pill("pairs", f"{ce['pairs']} × {ce['tokens']} tok"))
        pills.append(pill("generation", f"{timings['generation_ms']:.1f} ms"))

        if faith is None:
            pills.append(pill("faithfulness", "skipped (abstained)", "warn"))
        else:
            ok = faith["score"] == 1.0 and faith["numeric_grounding"]
            detail = f"{faith['supported']}/{faith['sentences']} sentences"
            if faith["unsupported_numbers"]:
                detail += " · unsupported: " + ", ".join(faith["unsupported_numbers"])
            else:
                detail += " · numbers ✓"
            pills.append(pill("faithfulness", ("PASS · " if ok else "FAIL · ") + detail,
                              "ok" if ok else "bad"))

        usage = result["usage"]
        pills.append(
            pill("tokens", f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out")
        )
        basis = "" if result.get("billable") else " projected"
        pills.append(pill("cost", f"${result['cost_usd']:.5f}{basis}"))
        st.markdown("<div class='pillbar'>" + "".join(pills) + "</div>",
                    unsafe_allow_html=True)
        st.caption(
            f"{trace['mode']} · {trace.get('top_n', '—')} rerank candidates · "
            f"{len(trace['hits'])} in trace · {trace['chunks_to_model']} sent to the model"
        )

        # --------------------------------------------------------- rank table
        st.markdown("**Candidate ranks**")
        st.dataframe(
            candidate_rows(trace["hits"]),
            hide_index=True,
            width="stretch",
        )
        if trace["mode"] == "Hybrid + rerank":
            up = sum(1 for h in trace["hits"]
                     if h.get("rrf_rank") and h.get("rerank_rank")
                     and h["rrf_rank"] > h["rerank_rank"])
            down = sum(1 for h in trace["hits"]
                       if h.get("rrf_rank") and h.get("rerank_rank")
                       and h["rrf_rank"] < h["rerank_rank"])
            st.caption(f"Cross-encoder moved {up} candidate(s) up and {down} down. "
                       "Ranking uses the raw logit; P(relevant) = sigmoid(logit).")

        # ---------------------------------------------------------- previews
        st.markdown("**Retrieved text** (query terms highlighted)")
        for i, h in enumerate(trace["hits"], start=1):
            label = f"#{i} · page {h['page']}"
            if h.get("movement"):
                label += f" · {h['movement']}"
            with st.expander(label):
                st.markdown(
                    f"<div class='chunk'>{highlight(h['text'], trace['question'])}</div>",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📄 RAG PDF QA")
    st.caption("MiniLM + BM25 → RRF → cross-encoder → Gemini Flash")

    # ---------------------------------------------------------------- corpus
    st.subheader("Corpus")
    corpus = st.radio(
        "Corpus",
        list(CORPORA) + ["Custom PDF"],
        label_visibility="collapsed",
    )

    if corpus == "Custom PDF":
        up = st.file_uploader("Upload a PDF", type=["pdf"])
        if up and st.button("Index this PDF", width="stretch", type="primary"):
            dest = settings.raw_dir / up.name
            dest.write_bytes(up.getbuffer())
            if index_pdf(dest):
                st.rerun()
    else:
        target = CORPORA[corpus]
        if not target.exists():
            st.warning(f"`{target.name}` is not built yet.")
            if corpus == "SQuAD 2.0" and st.button("Build SQuAD 2.0 corpus", type="primary"):
                with st.spinner("Sampling SQuAD 2.0 validation contexts…"):
                    proc = subprocess.run(
                        [sys.executable, "-m", "src.build_squad_benchmark",
                         "--contexts", "420", "--target-chunks", "231"],
                        cwd=settings.root, capture_output=True, text=True,
                    )
                if proc.returncode == 0:
                    reset_caches()
                    st.rerun()
                else:
                    st.error("Build failed — it needs the `datasets` package and network access.")
        elif active_source() != target.name:
            # First activation of this corpus: index it once, with a spinner.
            if index_pdf(target):
                st.rerun()

    stats = corpus_stats()
    if stats:
        st.caption(f"{stats['chunks']} chunks · {stats['pages']} pages · `{stats['source']}`")
    else:
        st.info("No index yet. Pick a corpus above or run `python run_pipeline.py`.")

    st.divider()

    # ------------------------------------------------------------ retrieval
    st.subheader("Retrieval")
    mode = st.radio("Mode", list(MODES), index=1, help="\n\n".join(
        f"**{m}** — {MODE_HELP[m]}" for m in MODES
    ))
    rerank_top_n = None
    if mode == "Hybrid + rerank":
        rerank_top_n = st.slider(
            "Rerank candidates", 5, 15, 10,
            help="How many RRF candidates the cross-encoder scores. "
                 "10 keeps ~95% MRR@10 at ~1/3 less CPU than 15.",
        )
        st.caption("First run loads the ONNX model (~22 MB), then it is cached.")
    k = st.slider("Chunks sent to the model", 1, 10, settings.context_k)

    use_llm = st.toggle(
        "Use Gemini", value=bool(settings.gemini_api_key),
        help="Needs GEMINI_API_KEY. Off = extractive answer (free, offline).",
    )
    if use_llm and not settings.gemini_api_key:
        st.warning("No GEMINI_API_KEY found — falling back to extractive answers.")


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
    }


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_chat, tab_bench, tab_about = st.tabs(["💬 Ask", "📊 Benchmark", "ℹ️ How it works"])

# ------------------------------- chat -------------------------------------- #
with tab_chat:
    st.header("Ask the document")
    if not index_exists():
        st.info("Index a PDF in the sidebar first — or build the demo corpus:\n\n"
                "```\npython run_pipeline.py\n```")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    st.markdown(
                        " ".join(f"<span class='cite'>p.{p}</span>" for p in msg["citations"]),
                        unsafe_allow_html=True,
                    )
                if msg.get("trace"):
                    render_trace(msg["trace"])

        question = st.chat_input("Ask something about the document…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving…" if mode != "Hybrid + rerank"
                                else "Retrieving and reranking…"):
                    trace = run_query(question, mode, k, rerank_top_n, use_llm)

                result = trace["result"]
                st.markdown(result["answer"])
                if result["citations"]:
                    st.markdown(
                        " ".join(f"<span class='cite'>p.{p}</span>" for p in result["citations"]),
                        unsafe_allow_html=True,
                    )
                if result["refused"]:
                    st.caption("The answer was not present in the retrieved context.")
                render_trace(trace)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                    "trace": trace,
                }
            )


# ------------------------------ benchmark ---------------------------------- #
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
        f"{note} · {corpus.get('chunks', '—')} chunks · "
        f"embedder `{corpus.get('embedder', '—')}` · "
        f"generator {data.get('config', {}).get('generator', '—')}"
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
                f"Rerank lift over RRF — Hit@3 {delta['hit_rate@3'] * 100:+.1f} pts · "
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


with tab_bench:
    st.header("Retrieval benchmark")
    st.caption("Read directly from `reports/*.json` — the same files the README quotes.")
    render_report(
        settings.reports_dir / "squad_benchmark.json",
        "SQuAD 2.0 · 230 chunks",
        "167 passages (129 with questions + 38 distractors) · "
        "150 answerable / 50 unanswerable / 25 out-of-domain",
    )
    st.divider()
    render_report(
        settings.reports_dir / "benchmark.json",
        "Synthetic handbook · 230 chunks",
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


# -------------------------------- about ------------------------------------ #
with tab_about:
    st.header("How it works")
    st.markdown(
        """
**Indexing** — `PyPDFLoader` reads the PDF page by page (page numbers are kept for
citations). Pages are split into ~800-character overlapping chunks, embedded with
`all-MiniLM-L6-v2`, and stored in **ChromaDB**. A **BM25** index is built over the
same chunks for exact-term matching. The corpus selector re-indexes the chosen PDF
once; the retriever is then cached for the session.

**Retrieval modes** — *dense* searches embeddings only; *hybrid* fuses dense + BM25
with **Reciprocal Rank Fusion** (`RRF(d) = Σ 1 / (60 + rank_i(d))`); *rerank* takes
the fused top-N and scores each `(query, chunk)` pair with
`ms-marco-MiniLM-L-6-v2`, an int8 ONNX cross-encoder run on CPU with no PyTorch.

**Why reranking is opt-in** — the cross-encoder reads the question and the chunk
together, which fixes “same entity, wrong attribute” confusions that RRF cannot. But
it is a second transformer pass over every candidate, and on this 2-core CPU that is
hundreds of milliseconds. The slider trades quality for latency: 10 candidates keeps
~95% MRR@10 on SQuAD at roughly two thirds of the top-15 cost. Pairs are padded only
to the longest pair in each batch (capped at 256 tokens), so short chunks are cheap.

**Trace** — every answer records retrieval, rerank and generation timings separately,
the exact RRF rank of each candidate and where the cross-encoder moved it, both
component scores, the relevance probability, the token counts and the projected USD
cost. **Faithfulness** is a lexical check: each answer sentence must share enough
content words with a context sentence, and every number in the answer must appear in
the context. With the offline extractive path it passes by construction; run with a
Gemini key to make it meaningful.

**Benchmark** — the JSON reports are generated by `src/benchmark.py`; the app never
hard-codes a metric. Hit@3 asks whether any gold chunk is in the top 3, MRR@10
measures the rank of the first gold chunk, Ans-in-Context@5 is the generator's
ceiling, and abstention is measured on the 75 negatives.
        """
    )
