"""One-command pipeline: demo corpus -> index -> benchmark -> report.

    python run_pipeline.py                # offline demo corpus
    python run_pipeline.py --squad        # rebuild the SQuAD 2.0 benchmark instead
    python run_pipeline.py --llm          # include Gemini abstention scoring
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===\n$ python {' '.join(args)}")
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", action="store_true", help="build the SQuAD 2.0 corpus")
    ap.add_argument("--contexts", type=int, default=280)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    if args.squad:
        step("Build SQuAD 2.0 corpus + queries",
             ["-m", "src.build_squad_benchmark", "--contexts", str(args.contexts)])
    else:
        step("Build demo corpus + queries", ["-m", "src.build_demo_corpus"])

    bench = ["-m", "src.benchmark", "--mode", "both"]
    if args.llm:
        bench.append("--llm")
    step("Benchmark", bench)

    print("\nDone.")
    print("  reports/benchmark.md    <- results table")
    print("  reports/benchmark.json  <- full numbers")
    print("  streamlit run app.py    <- interactive demo")


if __name__ == "__main__":
    main()
