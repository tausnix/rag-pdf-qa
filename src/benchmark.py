"""Benchmark: Hit Rate@k, MRR@k, Answer-in-Context@k, abstention, faithfulness, cost.

    python -m src.benchmark --mode all              # dense vs hybrid vs rerank
    python -m src.benchmark --mode all --llm        # + Gemini abstention/faithfulness/cost

Queries live in ``data/eval/queries.jsonl``, one JSON object per line:

    {"id": "q001",
     "question": "Who founded ...?",
     "type": "answerable" | "unanswerable" | "out_of_domain",
     "gold_chunk_ids": ["a1b2c3..."],   # required for answerable
     "gold_answer": "..."}              # optional, enables Answer-in-Context

Latency is split into retrieval (embed + search + fuse) and rerank (cross-encoder),
because they have completely different cost profiles.
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
from .generate import (
    cache_dir,
    cache_enabled,
    faithfulness,
    min_interval,
    set_cache_enabled,
    set_min_interval,
)
from .retrieve import Retriever, get_retriever

_NORM = re.compile(r"[^a-z0-9 ]+")


def norm(text: str) -> str:
    return " ".join(_NORM.sub(" ", text.lower()).split())


def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No queries file at {path}. Build one with src.build_demo_corpus.")
    queries = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for q in queries:
        q.setdefault("type", "answerable")
        q.setdefault("gold_chunk_ids", [])
    return queries


# --------------------------------------------------------------------------- #
# metric primitives
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
    reranker=None,
    progress_every: int = 25,
) -> dict:
    answerable = [q for q in queries if q["type"] == "answerable"]
    negatives = [q for q in queries if q["type"] != "answerable"]

    retriever.clear_query_cache()
    # warm-up (model load / first Chroma query), excluded from latency stats
    for _ in range(2):
        retriever.vector_search("warm up query")
    if mode == "rerank" and reranker is not None:
        reranker.score("warm up", ["warm up passage"])

    hits_ok, rrs, aic_ok = [], [], []
    latencies, retrieval_ms, rerank_ms = [], [], []
    gen_latencies, faith_scores = [], []
    refused_negatives = answered_negatives = 0
    false_refusals = 0
    in_tokens = out_tokens = 0
    cost_total = 0.0
    generated = 0
    billable = False
    live_calls = cached_calls = llm_fallbacks = rate_limited = 0

    for i, q in enumerate(queries, start=1):
        t0 = time.perf_counter()
        if mode == "rerank":
            candidates = retriever.hybrid_search(q["question"], k=settings.rerank_top_n)
            t1 = time.perf_counter()
            ranked = reranker.rerank(q["question"], candidates, k=max(mrr_k, ctx_k))
            t2 = time.perf_counter()
            retrieval_ms.append((t1 - t0) * 1000.0)
            rerank_ms.append((t2 - t1) * 1000.0)
        elif mode == "hybrid":
            ranked = retriever.hybrid_search(q["question"], k=max(mrr_k, ctx_k))
            t1 = time.perf_counter()
            retrieval_ms.append((t1 - t0) * 1000.0)
            rerank_ms.append(0.0)
        else:
            ranked = retriever.vector_search(q["question"], k=max(mrr_k, ctx_k))
            t1 = time.perf_counter()
            retrieval_ms.append((t1 - t0) * 1000.0)
            rerank_ms.append(0.0)

        latencies.append(retrieval_ms[-1] + rerank_ms[-1])

        ids = [h.chunk_id for h in ranked]
        if q["type"] == "answerable":
            gold = set(q["gold_chunk_ids"])
            hits_ok.append(hit_at_k(ids, gold, hit_k))
            rrs.append(reciprocal_rank(ids, gold, mrr_k))
            if q.get("gold_answer"):
                aic_ok.append(
                    answer_in_context([h.text for h in ranked[:ctx_k]], q["gold_answer"])
                )

        if use_llm:
            g0 = time.perf_counter()
            res = generate_answer(q["question"], ranked[:ctx_k], use_llm=True)
            gen_latencies.append((time.perf_counter() - g0) * 1000.0)
            generated += 1
            in_tokens += res["usage"]["input_tokens"]
            out_tokens += res["usage"]["output_tokens"]
            cost_total += res["cost_usd"]
            billable = billable or res.get("billable", False)
            rate_limited += int(res.get("rate_limited", 0))
            if res.get("cached"):
                cached_calls += 1
            elif res.get("used_llm"):
                live_calls += 1
            else:
                llm_fallbacks += 1

            if q["type"] == "answerable":
                if res["refused"]:
                    false_refusals += 1
                else:
                    faith_scores.append(faithfulness(res["answer"], ranked[:ctx_k])["score"])
            elif res["refused"]:
                refused_negatives += 1
            else:
                answered_negatives += 1

            if progress_every and i % progress_every == 0:
                print(
                    f"  [{mode}] {i}/{len(queries)} · {live_calls} live · "
                    f"{cached_calls} cached · {rate_limited} rate-limited · "
                    f"{llm_fallbacks} fallback",
                    flush=True,
                )

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
        "retrieval_p50_ms": statistics.median(retrieval_ms),
        "rerank_p50_ms": statistics.median(rerank_ms) if rerank_ms else 0.0,
    }

    if use_llm and negatives:
        out["abstention_accuracy"] = refused_negatives / len(negatives)
        out["negatives_refused"] = refused_negatives
        out["negatives_answered"] = answered_negatives
        out["false_refusal_rate"] = false_refusals / n
        out["faithfulness"] = (statistics.fmean(faith_scores) if faith_scores else None)
        out["generation_p50_ms"] = statistics.median(gen_latencies) if gen_latencies else 0.0
        out["tokens_per_query"] = {
            "input": in_tokens / generated,
            "output": out_tokens / generated,
        } if generated else {"input": 0, "output": 0}
        out["cost_per_1k_queries_usd"] = (cost_total / generated * 1000.0) if generated else 0.0
        out["llm_calls"] = {
            "live": live_calls,
            "cached": cached_calls,
            "fallbacks": llm_fallbacks,
        }
        out["rate_limited_responses"] = rate_limited
        if billable:
            basis = "actual API usage"
            if cached_calls:
                basis += f" ({cached_calls} responses replayed from disk cache)"
        elif cached_calls and not live_calls:
            basis = "projected from list price (all responses replayed from disk cache)"
        elif cached_calls:
            basis = f"projected from list price ({cached_calls} from disk cache)"
        else:
            basis = "projected from list price (no API call was made)"
        out["cost_basis"] = basis
        out["generator"] = (
            settings.gemini_model if settings.gemini_api_key else "extractive (no API key)"
        )

    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _row(name: str, r: dict, ctx_k: int) -> str:
    aic = r.get(f"answer_in_context@{ctx_k}")
    fmt = lambda v, suffix="%": f"{v * 100:.1f}{suffix}" if v is not None else "n/a"  # noqa: E731
    return (
        f"| {name} | {fmt(r['hit_rate@3'])} | {fmt(r['mrr@10'])} | {fmt(aic)} | "
        f"{fmt(r.get('abstention_accuracy'))} | {fmt(r.get('faithfulness'))} | "
        f"{r['latency_p50_ms']:.1f} ms | {r['latency_p95_ms']:.1f} ms |"
    )


def write_report(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    c = results["config"]
    md = [
        "# Retrieval benchmark",
        "",
        f"- Corpus chunks: **{results['corpus']['chunks']}**",
        f"- Embedder: `{results['corpus']['embedder']}`",
        f"- Chunking: {c['chunk_size']} chars / {c['chunk_overlap']} overlap",
        f"- RRF k: {c['rrf_k']} | context k: {c['context_k']} | rerank top-n: {c['rerank_top_n']}",
        f"- Generator: {c['generator']}",
        f"- Queries: {results['splits']['answerable']} answerable | "
        f"{results['splits']['unanswerable']} unanswerable | "
        f"{results['splits']['out_of_domain']} out-of-domain",
        "",
        "| Retriever | Hit@3 | MRR@10 | Ans-in-Context@5 | Abstention | Faithfulness | p50 | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("vector_only", "MiniLM dense"),
        ("hybrid", "Hybrid MiniLM + BM25 (RRF)"),
        ("rerank", f"Hybrid + cross-encoder rerank (top {c['rerank_top_n']})"),
    ):
        if key in results:
            md.append(_row(label, results[key], c["context_k"]))

    md += ["", "## Latency breakdown", "", "| Retriever | retrieval p50 | rerank p50 | total p50 | p95 |",
           "|---|---:|---:|---:|---:|"]
    for key, label in (
        ("vector_only", "MiniLM dense"),
        ("hybrid", "Hybrid RRF"),
        ("rerank", "Hybrid + rerank"),
    ):
        if key in results:
            r = results[key]
            md.append(
                f"| {label} | {r['retrieval_p50_ms']:.1f} ms | {r['rerank_p50_ms']:.1f} ms | "
                f"{r['latency_p50_ms']:.1f} ms | {r['latency_p95_ms']:.1f} ms |"
            )

    if "hybrid" in results and "rerank" in results:
        h, rr = results["hybrid"], results["rerank"]
        md += [
            "",
            "## Rerank lift over RRF",
            "",
            f"- Hit@3: {(rr['hit_rate@3'] - h['hit_rate@3']) * 100:+.1f} pts",
            f"- MRR@10: {(rr['mrr@10'] - h['mrr@10']) * 100:+.1f} pts",
            f"- Added latency: {rr['rerank_p50_ms']:.1f} ms per query",
        ]
    if any("cost_per_1k_queries_usd" in results[k] for k in results if isinstance(results[k], dict)):
        r = next(results[k] for k in ("rerank", "hybrid", "vector_only") if k in results)
        if "cost_per_1k_queries_usd" in r:
            md += [
                "",
                "## Generation cost",
                "",
                f"- Generator: {r['generator']}",
                f"- Tokens/query: {r['tokens_per_query']['input']:.0f} in / "
                f"{r['tokens_per_query']['output']:.0f} out",
                f"- Cost per 1,000 queries: **${r['cost_per_1k_queries_usd']:.2f}** "
                f"({r.get('cost_basis', '')})",
                f"- Generation p50: {r['generation_p50_ms']:.0f} ms",
            ]
            calls = r.get("llm_calls")
            if calls:
                md.append(
                    f"- LLM calls: {calls['live']} live | {calls['cached']} from cache | "
                    f"{calls['fallbacks']} fell back to extractive"
                )
                if r.get("rate_limited_responses"):
                    md.append(
                        f"- 429 responses absorbed: {r['rate_limited_responses']} "
                        f"(paced at {min_interval():.1f} s, retried with backoff)"
                    )

    md += ["", "> Latency is split: retrieval = embed + search + fuse; "
                "rerank = cross-encoder over the candidates."]
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")


MODES = {
    "vector_only": "vector",
    "hybrid": "hybrid",
    "rerank": "rerank",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality on a fixed query set")
    ap.add_argument("--queries", type=Path, default=settings.eval_dir / "queries.jsonl")
    ap.add_argument(
        "--mode", choices=["all", "both", "vector", "hybrid", "rerank"], default="all",
        help="'all' = dense + hybrid + rerank",
    )
    ap.add_argument("--llm", action="store_true", help="score abstention/faithfulness/cost with Gemini")
    ap.add_argument(
        "--llm-delay", type=float, default=None, metavar="SECONDS",
        help="minimum seconds between Gemini calls (default GEMINI_MIN_INTERVAL_S, 4 s)",
    )
    ap.add_argument(
        "--no-llm-cache", action="store_true",
        help="disable the on-disk Gemini response cache (re-running then re-bills every query)",
    )
    ap.add_argument("--limit", type=int, default=0, help="use only the first N queries")
    ap.add_argument("--out", type=Path, default=settings.reports_dir / "benchmark.json")
    args = ap.parse_args()

    if args.llm_delay is not None:
        set_min_interval(args.llm_delay)
    if args.no_llm_cache:
        set_cache_enabled(False)
    if args.llm:
        print(
            f"Gemini: {min_interval():.1f}s between calls, "
            f"{settings.gemini_max_retries} retries, cache "
            f"{cache_dir() if cache_enabled() else 'disabled'}",
            flush=True,
        )

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    chunks = [
        json.loads(line)
        for line in settings.chunks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retriever = get_retriever()
    reranker = None
    if args.mode in ("all", "rerank"):
        from .rerank import get_reranker

        reranker = get_reranker()

    selected = (
        ["vector_only", "hybrid", "rerank"] if args.mode == "all"
        else ["vector_only", "hybrid"] if args.mode == "both"
        else ["vector_only"] if args.mode == "vector"
        else ["hybrid"] if args.mode == "hybrid"
        else ["rerank"]
    )

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
            "rerank_top_n": settings.rerank_top_n,
            "generator": settings.gemini_model if (args.llm and settings.gemini_api_key)
            else "extractive (no LLM)",
            "llm_min_interval_s": min_interval() if args.llm else None,
            "llm_cache": bool(cache_enabled()) if args.llm else None,
        },
        "splits": {
            "answerable": sum(1 for q in queries if q["type"] == "answerable"),
            "unanswerable": sum(1 for q in queries if q["type"] == "unanswerable"),
            "out_of_domain": sum(1 for q in queries if q["type"] == "out_of_domain"),
        },
    }

    for key in selected:
        results[key] = evaluate(
            queries, retriever, MODES[key], args.llm, reranker=reranker
        )

    if "hybrid" in results and "rerank" in results:
        results["delta_rerank_vs_hybrid"] = {
            "hit_rate@3": results["rerank"]["hit_rate@3"] - results["hybrid"]["hit_rate@3"],
            "mrr@10": results["rerank"]["mrr@10"] - results["hybrid"]["mrr@10"],
            "latency_ms": results["rerank"]["latency_p50_ms"] - results["hybrid"]["latency_p50_ms"],
        }

    write_report(results, args.out)
    for key in selected:
        fallbacks = (results[key].get("llm_calls") or {}).get("fallbacks", 0)
        if args.llm and fallbacks:
            hint = (
                f"Re-run the same command to fill them from {cache_dir()}."
                if cache_enabled()
                else "The disk cache is disabled, so those queries must be re-billed."
            )
            print(
                f"WARNING: {key}: {fallbacks} quer{'y' if fallbacks == 1 else 'ies'} fell back to "
                f"extractive (quota exhausted or network error). {hint}",
                flush=True,
            )
    summary = {
        k: {m: round(v, 4) if isinstance(v, float) else v
            for m, v in results[k].items() if m in ("hit_rate@3", "mrr@10", "latency_p50_ms")}
        for k in selected
    }
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
