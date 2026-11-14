"""Build the SQuAD 2.0 benchmark used in the resume (150 / 50 / 25).

    python -m src.build_squad_benchmark --sections 80

Requires ``datasets`` and internet access (downloads ``rajpurkar/squad_v2``).

How the split is constructed — this is the part that makes the numbers mean
something:

* **150 answerable** — questions whose answer span exists in a corpus context.
* **50 unanswerable** — SQuAD 2.0 questions with *no* answer. Their source
  contexts are deliberately **excluded from the corpus**, so the question is
  on-topic but genuinely unanswerable given the document. This is what tests
  abstention; keeping the context would make it a fake negative.
* **25 out-of-domain** — hand-written questions from unrelated domains.

Gold labels: each answerable query stores the chunk id that contains its answer
span, plus the raw answer text (for Answer-in-Context@5).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .config import settings

OOD = [
    "What is the best way to bake sourdough bread at home?",
    "Who won the 2018 FIFA World Cup final?",
    "How do I change a flat tyre on a motorcycle?",
    "What are the symptoms of seasonal influenza?",
    "Which planet has the largest ring system?",
    "How long should I boil an egg for a soft yolk?",
    "What is the capital city of Australia?",
    "How do I prune a fig tree in early spring?",
    "What causes the northern lights?",
    "Recommend a beginner routine for learning guitar.",
    "How much water should an adult drink per day?",
    "What is the plot of Hamlet in one sentence?",
    "How do I fix a leaking kitchen tap?",
    "Which programming language is best for mobile games?",
    "What is the fastest land animal?",
    "How do vaccines work in the human body?",
    "What time zone is Tokyo in?",
    "How do I remove red wine from a carpet?",
    "What is the tallest mountain in South America?",
    "How many players are on a basketball team?",
    "What is the difference between a stock and a bond?",
    "How do I train a puppy to sit?",
    "What is photosynthesis in simple terms?",
    "Which country produces the most coffee?",
    "How do I get rid of fruit flies in my kitchen?",
]


def load_squad(limit_contexts: int, seed: int = 42):
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(seed)

    by_context: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        answers = row["answers"]["text"]
        starts = row["answers"]["answer_start"]
        by_context[row["context"]].append(
            {
                "question": row["question"].strip(),
                "answer": answers[0].strip() if answers else "",
                "answer_start": int(starts[0]) if starts else -1,
                "title": row["title"],
            }
        )

    contexts = list(by_context)
    rng.shuffle(contexts)
    chosen = contexts[:limit_contexts]
    return [(c, by_context[c]) for c in chosen], rng


def write_pdf(contexts: list[tuple[str, list[dict]]], path: Path) -> dict[str, int]:
    """One context per page so page numbers map 1:1 to contexts."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13)

    page_of: dict[int, int] = {}
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm, title="SQuAD 2.0 corpus",
    )
    story = []
    for i, (context, _) in enumerate(contexts):
        title = f"Passage {i + 1}"
        page_of[i] = i + 1
        story.append(Paragraph(title, h1))
        story.append(Paragraph(context.replace("&", "&amp;"), body))
        story.append(Spacer(1, 8))
        if i != len(contexts) - 1:
            story.append(PageBreak())
    doc.build(story)
    return page_of


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a SQuAD 2.0 corpus + 225-query benchmark")
    ap.add_argument("--contexts", type=int, default=420,
                    help="SQuAD passages to sample from validation")
    ap.add_argument("--target-chunks", type=int, default=231,
                    help="add distractor passages until the corpus reaches this size (0 = off)")
    ap.add_argument("--pdf", type=Path, default=settings.raw_dir / "squad_corpus.pdf")
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args()

    settings.ensure_dirs()
    pairs, rng = load_squad(args.contexts)

    # Split by question availability, then drop unanswerable contexts from corpus.
    answerable, unanswerable = [], []
    for idx, (context, rows) in enumerate(pairs):
        for row in rows:
            row["context_idx"] = idx
            (answerable if row["answer"] else unanswerable).append(row)

    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    answerable = answerable[:150]
    unanswerable = unanswerable[:50]

    # Rebuild the corpus from contexts that carry an answerable question, then add
    # extra passages until the chunk count matches the target. Extra passages are
    # real document content (a PDF has more pages than questions) and they *raise*
    # distractor density, so this makes retrieval harder, not easier.
    forbidden = {u["context_idx"] for u in unanswerable}
    required = sorted({a["context_idx"] for a in answerable})

    def est_chunks(text: str) -> int:
        # Calibrated against this chunker (800/150): ~538 source chars per chunk.
        return max(1, round(len(text) / 538))

    corpus_ids = list(required)
    estimated = sum(est_chunks(pairs[i][0]) for i in corpus_ids)
    if args.target_chunks and estimated < args.target_chunks:
        extras = [
            i for i in range(len(pairs)) if i not in forbidden and i not in set(required)
        ]
        rng.shuffle(extras)
        for i in extras:
            if estimated >= args.target_chunks:
                break
            corpus_ids.append(i)
            estimated += est_chunks(pairs[i][0])
    corpus_ids.sort()

    page_by_ctx = {ctx: page for page, ctx in enumerate(corpus_ids, start=1)}
    corpus = [(pairs[i][0], pairs[i][1]) for i in corpus_ids]
    write_pdf(corpus, args.pdf)
    print(
        f"Wrote {args.pdf} ({len(corpus)} passages, "
        f"{len(required)} with questions + {len(corpus) - len(required)} distractor passages, "
        f"~{estimated} chunks estimated)"
    )

    if args.no_index:
        return

    from .ingest import ingest, load_chunks

    summary = ingest(args.pdf, reset=True)
    chunks = load_chunks()
    compact = lambda t: re.sub(r"\s+", " ", t).strip().lower()  # noqa: E731

    def gold_chunks(row: dict, context: str) -> list[str]:
        """Label chunks that hold *this* answer, not chunks that repeat its string.

        SQuAD answers are often short ("2", "France"), so a global substring match
        would label dozens of unrelated chunks and inflate Hit@3. We therefore:
          1. restrict candidates to the passage's own page,
          2. prefer a ~140-char window around ``answer_start``,
          3. shrink the window before falling back to the bare answer string.
        """
        page_no = page_by_ctx[row["context_idx"]]
        page_chunks = [c for c in chunks if c["page"] == page_no]
        if not page_chunks:
            return []
        start, ans = row["answer_start"], row["answer"]
        for half in (70, 40, 20):
            lo, hi = max(0, start - half), min(len(context), start + len(ans) + half)
            window = compact(context[lo:hi])
            hits = [c["chunk_id"] for c in page_chunks if window in compact(c["text"])]
            if hits:
                return hits
        return [c["chunk_id"] for c in page_chunks if compact(ans) in compact(c["text"])]

    queries, dropped = [], 0
    for i, row in enumerate(answerable):
        cids = gold_chunks(row, pairs[row["context_idx"]][0])
        if not cids:  # answer split across a chunk boundary -> exclude honestly
            dropped += 1
            continue
        queries.append(
            {
                "id": f"a{i + 1:03d}",
                "question": row["question"],
                "type": "answerable",
                "gold_chunk_ids": cids,
                "gold_answer": row["answer"],
                "gold_page": page_by_ctx[row["context_idx"]],
            }
        )
    for i, row in enumerate(unanswerable):
        queries.append(
            {
                "id": f"u{i + 1:03d}",
                "question": row["question"],
                "type": "unanswerable",
                "gold_chunk_ids": [],
                "gold_answer": "",
            }
        )
    for i, q in enumerate(OOD[:25]):
        queries.append(
            {"id": f"o{i + 1:03d}", "question": q, "type": "out_of_domain",
             "gold_chunk_ids": [], "gold_answer": ""}
        )

    out = settings.eval_dir / "queries.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    manifest = {
        "source": "rajpurkar/squad_v2 (validation)",
        "chunks": len(chunks),
        "pages": summary["pages"],
        "answerable": len(queries) - 75,
        "unanswerable": 50,
        "out_of_domain": 25,
        "dropped_answerable": dropped,
        "note": "Unanswerable questions come from contexts excluded from the corpus, "
                "so they are genuinely unanswerable from the document.",
    }
    (settings.eval_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
