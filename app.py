"""Streamlit UI: upload a PDF, ask questions, inspect the evidence.

    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import settings
from src.generate import answer as generate_answer
from src.ingest import ingest
from src.retrieve import get_retriever
from src import retrieve as retrieve_module

st.set_page_config(page_title="RAG PDF QA", page_icon="📄", layout="wide")

CSS = """
<style>
  .block-container {padding-top: 2rem; max-width: 1150px;}
  .metric-card {border:1px solid #2a2f3a; border-radius:12px; padding:14px 16px; background:#11151c;}
  .metric-card h3 {margin:0 0 4px 0; font-size:13px; color:#8b96a8; font-weight:500;}
  .metric-card p {margin:0; font-size:22px; font-weight:650;}
  .cite {display:inline-block; background:#1b2430; border:1px solid #2f3b4d; border-radius:999px;
         padding:2px 10px; margin-right:6px; font-size:12px; color:#9fd0ff;}
  .chunk {border-left:3px solid #2f3b4d; padding:6px 12px; margin-bottom:10px; font-size:13px; color:#b9c3d1;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_retriever():
    return get_retriever()


def index_exists() -> bool:
    return settings.bm25_pickle.exists() and settings.chunks_jsonl.exists()


def corpus_stats() -> dict:
    if not settings.chunks_jsonl.exists():
        return {}
    rows = [json.loads(l) for l in settings.chunks_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {
        "chunks": len(rows),
        "pages": len({r["page"] for r in rows}),
        "source": rows[0]["source"] if rows else "—",
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📄 RAG PDF QA")
    st.caption("MiniLM + BM25 → RRF → Gemini Flash")

    mode = st.radio("Retrieval", ["Hybrid (RRF)", "Vector only"], index=0,
                    help="Hybrid fuses dense and lexical results with Reciprocal Rank Fusion.")
    k = st.slider("Chunks sent to the model", 1, 10, settings.context_k)
    use_llm = st.toggle("Use Gemini", value=bool(settings.gemini_api_key),
                        help="Needs GEMINI_API_KEY. Off = extractive answer.")
    if use_llm and not settings.gemini_api_key:
        st.warning("No GEMINI_API_KEY found — falling back to extractive answers.")

    st.divider()
    st.subheader("Corpus")
    up = st.file_uploader("Upload a PDF", type=["pdf"])
    if up and st.button("Index this PDF", use_container_width=True, type="primary"):
        dest = settings.raw_dir / up.name
        dest.write_bytes(up.getbuffer())
        with st.spinner("Chunking, embedding, writing ChromaDB…"):
            try:
                summary = ingest(dest, reset=True)
                retrieve_module._CACHE = None
                load_retriever.clear()
                st.success(f"Indexed {summary['chunks']} chunks from {summary['pages']} pages.")
                st.rerun()
            except SystemExit as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Indexing failed: {exc}")

    stats = corpus_stats()
    if stats:
        st.caption(f"{stats['chunks']} chunks · {stats['pages']} pages · `{stats['source']}`")
    else:
        st.info("No index yet. Upload a PDF, or run `python run_pipeline.py` for the demo corpus.")

    st.divider()
    toggles = st.expander("Show retrieved chunks")
    show_chunks = toggles.checkbox("On", value=False)

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
                if msg.get("chunks"):
                    with st.expander("Evidence"):
                        for h in msg["chunks"]:
                            st.markdown(
                                f"<div class='chunk'><b>page {h['page']}</b> · score {h['score']:.4f}"
                                f"<br>{h['preview']}</div>",
                                unsafe_allow_html=True,
                            )

        question = st.chat_input("Ask something about the document…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and answering…"):
                    retriever = load_retriever()
                    if mode.startswith("Hybrid"):
                        hits = retriever.hybrid_search(question, k=k)
                    else:
                        hits = retriever.vector_search(question, k=k)
                    result = generate_answer(question, hits, use_llm=use_llm)

                st.markdown(result["answer"])
                if result["citations"]:
                    st.markdown(
                        " ".join(f"<span class='cite'>p.{p}</span>" for p in result["citations"]),
                        unsafe_allow_html=True,
                    )
                if result["refused"]:
                    st.caption("The answer was not present in the retrieved context.")
                if show_chunks:
                    with st.expander("Evidence", expanded=True):
                        for h in hits:
                            st.markdown(
                                f"<div class='chunk'><b>page {h['page']}</b> · score {h['score']:.4f}"
                                f"<br>{h.text[:400]}</div>",
                                unsafe_allow_html=True,
                            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                    "chunks": [
                        {"page": h.page, "score": h.score, "preview": h.text[:300]}
                        for h in hits
                    ] if show_chunks else [],
                }
            )

# ------------------------------ benchmark ---------------------------------- #
with tab_bench:
    st.header("Retrieval benchmark")
    bench_path = settings.reports_dir / "benchmark.json"
    if not bench_path.exists():
        st.info("No results yet. Run `python run_pipeline.py` (or `python -m src.benchmark`).")
    else:
        data = json.loads(bench_path.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        hybrid = data.get("hybrid", {})
        vec = data.get("vector_only", {})
        c1.markdown(f"<div class='metric-card'><h3>CHUNKS</h3><p>{data['corpus']['chunks']}</p></div>",
                    unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>HIT RATE @3 (HYBRID)</h3>"
                    f"<p>{hybrid.get('hit_rate@3', 0) * 100:.1f}%</p></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>MRR @10 (HYBRID)</h3>"
                    f"<p>{hybrid.get('mrr@10', 0) * 100:.1f}%</p></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>MEDIAN LATENCY</h3>"
                    f"<p>{hybrid.get('latency_p50_ms', 0):.1f} ms</p></div>", unsafe_allow_html=True)

        rows = []
        for label, r in (("MiniLM vector only", vec), ("Hybrid MiniLM + BM25 (RRF)", hybrid)):
            if not r:
                continue
            rows.append(
                {
                    "Retriever": label,
                    "Hit@3": f"{r['hit_rate@3'] * 100:.1f}%",
                    "MRR@10": f"{r['mrr@10'] * 100:.1f}%",
                    "Ans-in-Context@5": (
                        f"{r['answer_in_context@5'] * 100:.1f}%"
                        if r.get("answer_in_context@5") is not None else "n/a"
                    ),
                    "Abstention": (
                        f"{r['abstention_accuracy'] * 100:.1f}%"
                        if r.get("abstention_accuracy") is not None else "n/a"
                    ),
                    "p50": f"{r['latency_p50_ms']:.1f} ms",
                    "p95": f"{r['latency_p95_ms']:.1f} ms",
                }
            )
        st.table(rows)
        st.caption(f"Generator: {data['config']['generator']} · "
                   f"chunk {data['config']['chunk_size']}/{data['config']['chunk_overlap']} · "
                   f"RRF k={data['config']['rrf_k']}")
        if "delta" in data:
            d = data["delta"]
            st.caption(f"Hybrid delta — Hit@3 {d['hit_rate@3'] * 100:+.1f} pts · "
                       f"MRR@10 {d['mrr@10'] * 100:+.1f} pts")

# -------------------------------- about ------------------------------------ #
with tab_about:
    st.header("How it works")
    st.markdown(
        """
**Indexing** — `PyPDFLoader` reads the PDF page by page (page numbers are kept for
citations). Pages are split into ~800-character overlapping chunks, embedded with
`all-MiniLM-L6-v2`, and stored in **ChromaDB**. A **BM25** index is built over the
same chunks for exact-term matching.

**Retrieval** — the question is searched twice: dense (semantic) and lexical (BM25).
The two ranked lists are merged with **Reciprocal Rank Fusion**:

```
RRF(d) = Σ 1 / (60 + rank_i(d))
```

RRF uses ranks only, so no score normalisation is needed and neither retriever can
dominate. The top 5 fused chunks become the context.

**Generation** — Gemini Flash receives a strict prompt: answer only from the context,
cite `[p.N]`, and reply *“I couldn't find that in the document.”* when the answer is
absent. That refusal path is what the 50 unanswerable + 25 out-of-domain queries test.

**Metrics** — `Hit@3` (is a gold chunk in the top 3), `MRR@10` (reciprocal rank of the
first gold chunk), `Answer-in-Context@5` (does the gold answer appear in the top 5 —
the ceiling for any generator), and median/p95 retrieval latency.
        """
    )
