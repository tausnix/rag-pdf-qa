"""One-command pipeline: demo corpus -> index -> benchmark -> report.

    python run_pipeline.py                # offline demo corpus, dense + hybrid
    python run_pipeline.py --rerank       # include the cross-encoder mode
    python run_pipeline.py --squad        # rebuild the SQuAD 2.0 benchmark instead
    python run_pipeline.py --llm          # include Gemini abstention/cost scoring
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===\n$ python {' '.join(args)}")
    sys.stdout.flush()  # keep CI logs in order (child output is unbuffered)
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", action="store_true", help="build the SQuAD 2.0 corpus")
    ap.add_argument("--contexts", type=int, default=420,
                    help="SQuAD passages to sample from validation")
    ap.add_argument("--target-chunks", type=int, default=231,
                    help="add distractor passages until the SQuAD corpus reaches this size")
    ap.add_argument("--rerank", action="store_true",
                    help="also benchmark the cross-encoder (slower, better MRR)")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    if args.squad:
        step("Build SQuAD 2.0 corpus + queries",
             ["-m", "src.build_squad_benchmark",
              "--contexts", str(args.contexts),
              "--target-chunks", str(args.target_chunks)])
    else:
        step("Build demo corpus + queries", ["-m", "src.build_demo_corpus"])

    bench = ["-m", "src.benchmark", "--mode", "all" if args.rerank else "both"]
    if args.llm:
        bench.append("--llm")
    if args.squad:
        bench += ["--out", "reports/squad_benchmark.json"]
    step("Benchmark", bench)

    print("\nDone.")
    print("  reports/benchmark.md    <- results table")
    print("  reports/benchmark.json  <- full numbers")
    print("  streamlit run app.py    <- interactive demo")


if __name__ == "__main__":
    main()
