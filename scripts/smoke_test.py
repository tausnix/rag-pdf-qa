"""Health check: index present, retriever loads, one question answers.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ask import run  # noqa: E402
from src.config import settings  # noqa: E402


def main() -> int:
    if not settings.bm25_pickle.exists():
        print("FAIL: no index. Run: python run_pipeline.py")
        return 1

    result = run("What was the annual budget?", use_llm=False)
    assert result["retrieved"], "retriever returned nothing"
    print(f"chunks retrieved : {len(result['retrieved'])}")
    print(f"top page         : {result['retrieved'][0]['page']}")
    print(f"answer           : {result['answer'][:160]}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
