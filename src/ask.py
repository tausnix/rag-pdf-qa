"""Ask a question against the indexed PDF.

    python -m src.ask "What was the headcount at Norvik Plant?"
    python -m src.ask "..." --no-llm --show-chunks
"""
from __future__ import annotations

import argparse
import json

from .config import settings
from .generate import answer
from .retrieve import get_retriever


def run(question: str, k: int | None = None, use_llm: bool = True) -> dict:
    retriever = get_retriever()
    hits = retriever.hybrid_search(question, k=k or settings.context_k)
    result = answer(question, hits, use_llm=use_llm)
    result["retrieved"] = [
        {"page": h.page, "score": round(h.score, 6), "chunk_id": h.chunk_id,
         "preview": h.text[:220].replace("\n", " ")}
        for h in hits
    ]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a question about the indexed PDF")
    ap.add_argument("question")
    ap.add_argument("-k", "--top-k", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true", help="extractive answer, no API call")
    ap.add_argument("--show-chunks", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    result = run(args.question, args.top_k, use_llm=not args.no_llm)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"\nQ: {args.question}\n")
    print(f"A: {result['answer']}")
    if result["citations"]:
        print(f"\n   pages: {', '.join(str(p) for p in result['citations'])}")
    if args.show_chunks:
        print("\n   retrieved chunks:")
        for h in result["retrieved"]:
            print(f"   [p.{h['page']}] {h['score']:.5f}  {h['preview']}")


if __name__ == "__main__":
    main()
