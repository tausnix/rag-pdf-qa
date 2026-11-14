"""Generate a self-contained demo corpus + gold-labelled query set (offline).

Why this exists: the SQuAD-based benchmark needs a download and a carefully
matched corpus. This script produces the same *shape* of data with zero network
access, so the pipeline can be run and debugged immediately:

    python -m src.build_demo_corpus

It writes:
    data/raw/demo_handbook.pdf      ~60-page synthetic operations handbook
    data/eval/queries.jsonl         150 answerable / 50 unanswerable / 25 OOD
    data/eval/corpus_manifest.json  how the gold labels were derived

The metrics from this corpus are *plumbing checks*, not research results.
Use ``src/build_squad_benchmark.py`` for the SQuAD 2.0 protocol.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .config import settings

RNG = random.Random(42)

ENTITIES = [
    "Halden Institute", "Norvik Plant", "Cobalt Ridge", "Sable Works",
    "Meridian Depot", "Ardent Labs", "Kestrel Terminal", "Juniper Facility",
    "Larkspur Yard", "Tamarack Works", "Vantage Annex", "Osprey Station",
    "Quarry Nine", "Basalt Complex", "Northwind Silo", "Ember Point",
    "Granite Reach", "Silverline Hub", "Thornfield Site", "Wexford Mill",
    "Yarrow Campus", "Zenith Bunker", "Alder Grove", "Blackfen Works",
    "Calder Wharf", "Dunmore Plant", "Elmsworth Depot", "Fallow Ridge",
    "Gorse Hollow", "Harrow Field", "Ivory Quay", "Jasper Flats",
    "Kelvin Yard", "Linden Terminal", "Marlowe Works", "Nettle Bank",
    "Orchard Line", "Pinehurst Site", "Redwood Vault", "Sorrel Point",
    "Tamarisk Yard", "Umber Works", "Verdant Depot", "Willow Reach",
    "Xenia Plant", "Yew Hollow", "Zephyr Station", "Ashcombe Mill",
    "Briar Glen", "Cedar Wharf",
]

ATTRIBUTES = [
    ("annual budget", "USD {n} million", (8, 240)),
    ("headcount", "{n} employees", (40, 4200)),
    ("median response time", "{n} milliseconds", (9, 900)),
    ("throughput", "{n} tonnes per day", (120, 9800)),
    ("energy consumption", "{n} megawatt hours", (400, 41000)),
    ("compliance score", "{n} out of 100", (61, 99)),
    ("retention rate", "{n} percent", (68, 97)),
    ("open audit findings", "{n} findings", (0, 41)),
]

FILLER = [
    "The review board reconfirmed the operating assumptions during the autumn cycle.",
    "Regional supervisors submitted revised schedules after the quarterly briefing.",
    "Documentation for this section is retained under the standard seven-year policy.",
    "Cross-functional teams continue to align their forecasts with the central plan.",
    "Measurements are taken at the same point in the production cycle each quarter.",
    "Any variance beyond the tolerance band is escalated to the duty manager.",
    "Figures were independently verified before publication of this handbook.",
    "The procedure described here supersedes the previous revision.",
    "Staff receive refresher training on this topic every eighteen months.",
    "Local conditions may justify a temporary deviation, subject to written approval.",
]

OOD_QUESTIONS = [
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


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
def build_sections(n_sections: int) -> list[dict]:
    sections = []
    for s in range(n_sections):
        entity = ENTITIES[s % len(ENTITIES)]
        title = f"Section {s + 1}: {entity}"
        paragraphs, facts = [], []
        for a, (attr, value_fmt, (lo, hi)) in enumerate(ATTRIBUTES):
            value = value_fmt.format(n=RNG.randint(lo, hi))
            sentence = f"For {entity}, the reported {attr} was {value}."
            # 2-3 filler sentences per paragraph keeps the chunk realistic
            noise = " ".join(RNG.sample(FILLER, k=RNG.randint(2, 3)))
            paragraphs.append(f"{sentence} {noise}")
            facts.append({"entity": entity, "attribute": attr, "value": value, "sentence": sentence})
        sections.append({"title": title, "entity": entity, "paragraphs": paragraphs, "facts": facts})
    return sections


def write_pdf(sections: list[dict], path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceAfter=10)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, 1.2 * cm, "Aurora Operations Handbook - internal circulation")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Aurora Operations Handbook",
    )
    story = [Paragraph("Aurora Operations Handbook", h1), Spacer(1, 12)]
    story.append(Paragraph("Revision 7 - consolidated operating figures. " + RNG.choice(FILLER), body))
    story.append(PageBreak())
    for sec in sections:
        story.append(Paragraph(sec["title"], h1))
        for para in sec["paragraphs"]:
            story.append(Paragraph(para.replace("&", "&amp;"), body))
            story.append(Spacer(1, 6))
        story.append(PageBreak())
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------- #
# Gold labels
# --------------------------------------------------------------------------- #
def find_chunks_for(sentence: str, chunks: list[dict], value: str) -> list[str]:
    """All chunks that contain the fact.

    Overlapping chunks mean the same sentence can appear twice; labelling only
    one of them would count a correct retrieval as a miss.
    """
    compact = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731
    target = compact(sentence)
    exact = [c["chunk_id"] for c in chunks if target in compact(c["text"])]
    if exact:
        return exact
    return [c["chunk_id"] for c in chunks if compact(value) in compact(c["text"])]


def build_queries(sections: list[dict], chunks: list[dict]) -> list[dict]:
    facts = [f for sec in sections for f in sec["facts"]]
    RNG.shuffle(facts)

    answerable: list[dict] = []
    for fact in facts:
        if len(answerable) >= 150:
            break
        sentence = fact["sentence"]
        cids = find_chunks_for(sentence, chunks, fact["value"])
        if not cids:
            continue
        question = f"What was the reported {fact['attribute']} for {fact['entity']}?"
        answerable.append(
            {
                "id": f"a{len(answerable) + 1:03d}",
                "question": question,
                "type": "answerable",
                "gold_chunk_ids": cids,
                "gold_answer": fact["value"],
            }
        )

    # Unanswerable: real entity, attribute that the handbook never reports for it.
    unanswerable: list[dict] = []
    extra_attrs = [
        "carbon offset", "board size", "insurance premium", "patent portfolio",
        "customer satisfaction index", "overtime cost", "water usage", "incident count",
    ]
    for i in range(50):
        entity = ENTITIES[i % len(ENTITIES)]
        attr = extra_attrs[i % len(extra_attrs)]
        unanswerable.append(
            {
                "id": f"u{i + 1:03d}",
                "question": f"What was the reported {attr} for {entity}?",
                "type": "unanswerable",
                "gold_chunk_ids": [],
                "gold_answer": "",
            }
        )

    ood = [
        {
            "id": f"o{i + 1:03d}",
            "question": q,
            "type": "out_of_domain",
            "gold_chunk_ids": [],
            "gold_answer": "",
        }
        for i, q in enumerate(OOD_QUESTIONS[:25])
    ]

    return answerable + unanswerable + ood


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the offline demo corpus + queries")
    ap.add_argument("--sections", type=int, default=60)
    ap.add_argument("--pdf", type=Path, default=settings.raw_dir / "demo_handbook.pdf")
    ap.add_argument("--no-index", action="store_true", help="write the PDF only")
    args = ap.parse_args()

    settings.ensure_dirs()
    sections = build_sections(args.sections)
    write_pdf(sections, args.pdf)
    print(f"Wrote {args.pdf}")

    if args.no_index:
        return

    from .ingest import ingest, load_chunks

    summary = ingest(args.pdf, reset=True)
    chunks = load_chunks()
    queries = build_queries(sections, chunks)

    out = settings.eval_dir / "queries.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    manifest = {
        "corpus": args.pdf.name,
        "chunks": len(chunks),
        "pages": summary["pages"],
        "splits": {
            "answerable": sum(1 for q in queries if q["type"] == "answerable"),
            "unanswerable": sum(1 for q in queries if q["type"] == "unanswerable"),
            "out_of_domain": sum(1 for q in queries if q["type"] == "out_of_domain"),
        },
        "gold_rule": "gold_chunk = chunk containing the exact fact sentence; "
                     "falls back to a chunk containing the answer value",
    }
    (settings.eval_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
