"""Benchmark: Hit Rate@k, MRR@k, Answer-in-Context@k, abstention, latency.

    python -m src.benchmark                 # hybrid + vector-only, no LLM
    python -m src.benchmark --llm           # include Gemini abstention scoring

Queries live in ``data/eval/queries.jsonl``, one JSON object per line:

    {"id": "q001",
     "question": "Who founded ...?",
     "type": "answerable" | "unanswerable" | "out_of_domain",
     "gold_chunk_ids": ["a1b2c3..."],   # required for answerable
     "gold_answer": "..."}              # optional, enables Answer-in-Context
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from .config import settings
from .generate import answer as generate_answer
from .retrieve import Retriever, get_retriever

_NORM = re.compile(r"[^a-z0-9 ]+")


def norm(text: str) -> str:
    return " ".join(_NORM.sub(" ", text.lower()).split())


def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No queries file at {path}. Build one with src.build_demo_corpus.")
    queries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for q in queries:
        q.setdefault("type", "answerable")
        q.setdefault("gold_chunk_ids", [])
    return queries


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def hit_at_k(ranked_ids: list[str], gold: set[str], k: int) -> bool:
    return bool(gold & set(ranked_ids[:k]))


def reciprocal_rank(ranked_ids: list[str], gold: set[str], k: int) -> float:
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def answer_in_context(texts: list[str], gold_answer: str) -> bool:
    if not gold_answer:
        return False
    haystack = norm(" ".join(texts))
    needle = norm(gold_answer)
    return bool(needle) and needle in haystack


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def evaluate(
    queries: list[dict],
    retriever: Retriever,
    mode: str,
    use_llm: bool,
    hit_k: int = 3,
    mrr_k: int = 10,
    ctx_k: int = 5,
) -> dict:
    answerable = [q for q in queries if q["type"] == "answerable"]
    negatives = [q for q in queries if q["type"] != "answerable"]

    # warm-up (model load / first Chroma query), excluded from latency stats
    for _ in range(2):
        retriever.vector_search("warm up query")

    hits_ok, rrs, aic_ok, latencies = [], [], [], []
    refused_negatives, answered_negatives = 0, 0

    for q in queries:
        t0 = time.perf_counter()
        if mode == "hybrid":
            ranked = retriever.hybrid_search(q["question"], k=max(mrr_k, ctx_k))
        else:
            ranked = retriever.vector_search(q["question"], k=max(mrr_k, ctx_k))
        latencies.append((time.perf_counter() - t0) * 1000.0)

        ids = [h.chunk_id for h in ranked]
        if q["type"] == "answerable":
            gold = set(q["gold_chunk_ids"])
            hits_ok.append(hit_at_k(ids, gold, hit_k))
            rrs.append(reciprocal_rank(ids, gold, mrr_k))
            if q.get("gold_answer"):
                aic_ok.append(
                    answer_in_context([h.text for h in ranked[:ctx_k]], q["gold_answer"])
                )
        elif use_llm:
            res = generate_answer(q["question"], ranked[:ctx_k], use_llm=True)
            if res["refused"]:
                refused_negatives += 1
            else:
                answered_negatives += 1

    n = len(answerable) or 1
    out = {
        "mode": mode,
        "n_queries": len(queries),
        "n_answerable": len(answerable),
        "n_unanswerable": sum(1 for q in negatives if q["type"] == "unanswerable"),
        "n_out_of_domain": sum(1 for q in negatives if q["type"] == "out_of_domain"),
        f"hit_rate@{hit_k}": sum(hits_ok) / n,
        f"mrr@{mrr_k}": sum(rrs) / n,
        f"answer_in_context@{ctx_k}": (sum(aic_ok) / len(aic_ok)) if aic_ok else None,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": pct(latencies, 0.95),
        "latency_mean_ms": statistics.fmean(latencies),
    }
    if use_llm and negatives:
        out["abstention_accuracy"] = refused_negatives / len(negatives)
        out["negatives_refused"] = refused_negatives
        out["negatives_answered"] = answered_negatives
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _row(name: str, r: dict) -> str:
    aic = r.get(f"answer_in_context@{settings.context_k}")
    aic_s = f"{aic * 100:.1f}%" if aic is not None else "n/a"
    abst = r.get("abstention_accuracy")
    abst_s = f"{abst * 100:.1f}%" if abst is not None else "n/a"
    return (
        f"| {name} | {r['hit_rate@3'] * 100:.1f}% | {r['mrr@10'] * 100:.1f}% | "
        f"{aic_s} | {abst_s} | {r['latency_p50_ms']:.1f} ms | {r['latency_p95_ms']:.1f} ms |"
    )


def write_report(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [
        "# Retrieval benchmark",
        "",
        f"- Corpus chunks: **{results['corpus']['chunks']}**",
        f"- Embedder: `{results['corpus']['embedder']}`",
        f"- Chunking: {results['config']['chunk_size']} chars / "
        f"{results['config']['chunk_overlap']} overlap",
        f"- RRF k: {results['config']['rrf_k']} · context k: {results['config']['context_k']}",
        f"- Generator: {results['config']['generator']}",
        f"- Queries: {results['splits']['answerable']} answerable · "
        f"{results['splits']['unanswerable']} unanswerable · "
        f"{results['splits']['out_of_domain']} out-of-domain",
        "",
        "| Retriever | Hit@3 | MRR@10 | Ans-in-Context@5 | Abstention | p50 latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if "vector_only" in results:
        md.append(_row("MiniLM vector only", results["vector_only"]))
    if "hybrid" in results:
        md.append(_row("Hybrid MiniLM + BM25 (RRF)", results["hybrid"]))
    md.append("")
    md.append("> Latency is **retrieval only** (embed + search + fuse), measured after warm-up.")
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality on a fixed query set")
    ap.add_argument("--queries", type=Path, default=settings.eval_dir / "queries.jsonl")
    ap.add_argument("--mode", choices=["both", "vector", "hybrid"], default="both")
    ap.add_argument("--llm", action="store_true", help="also score abstention with Gemini")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N queries")
    ap.add_argument("--out", type=Path, default=settings.reports_dir / "benchmark.json")
    args = ap.parse_args()

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    chunks = [json.loads(l) for l in settings.chunks_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    retriever = get_retriever()

    results: dict = {
        "corpus": {
            "chunks": len(chunks),
            "pages": len({c["page"] for c in chunks}),
            "embedder": retriever._embedder.name,
        },
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k_vector": settings.top_k_vector,
            "top_k_bm25": settings.top_k_bm25,
            "rrf_k": settings.rrf_k,
            "context_k": settings.context_k,
            "generator": settings.gemini_model if (args.llm and settings.gemini_api_key) else "extractive (no LLM)",
        },
        "splits": {
            "answerable": sum(1 for q in queries if q["type"] == "answerable"),
            "unanswerable": sum(1 for q in queries if q["type"] == "unanswerable"),
            "out_of_domain": sum(1 for q in queries if q["type"] == "out_of_domain"),
        },
    }

    if args.mode in ("both", "vector"):
        results["vector_only"] = evaluate(queries, retriever, "vector", args.llm)
    if args.mode in ("both", "hybrid"):
        results["hybrid"] = evaluate(queries, retriever, "hybrid", args.llm)
    if args.mode == "both":
        v, h = results["vector_only"], results["hybrid"]
        results["delta"] = {
            "hit_rate@3": h["hit_rate@3"] - v["hit_rate@3"],
            "mrr@10": h["mrr@10"] - v["mrr@10"],
        }

    write_report(results, args.out)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
