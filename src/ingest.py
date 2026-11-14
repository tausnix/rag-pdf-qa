"""Indexing: PDF -> pages -> chunks -> ChromaDB + BM25.

Run once per corpus:

    python -m src.ingest --pdf data/raw/corpus.pdf

Artefacts written
-----------------
data/index/chunks.jsonl   one JSON record per chunk (text + metadata)
data/index/bm25.pkl       pickled BM25 index + tokenised corpus
chroma/                   persistent vector collection
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

from .config import settings
from .embeddings import Embedder

# --------------------------------------------------------------------------- #
# PDF loading
# --------------------------------------------------------------------------- #
def load_pdf(path: Path) -> list[dict]:
    """Return ``[{"text", "page", "source"}]``.

    Uses LangChain's ``PyPDFLoader`` when available (the loader named on the
    resume); otherwise falls back to plain ``pypdf`` with identical semantics.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(str(path)).load()
        return [
            {
                "text": d.page_content,
                "page": int(d.metadata.get("page", 0)) + 1,  # 0-based -> 1-based
                "source": Path(str(d.metadata.get("source", path))).name,
            }
            for d in docs
        ]
    except Exception:  # langchain missing or loader failed -> direct pypdf
        from pypdf import PdfReader

        return [
            {
                "text": page.extract_text() or "",
                "page": i + 1,
                "source": path.name,
            }
            for i, page in enumerate(PdfReader(str(path)).pages)
        ]


def strip_boilerplate(
    pages: list[dict], min_page_ratio: float = 0.3, head_lines: int = 2, tail_lines: int = 2
) -> list[dict]:
    """Remove repeated headers/footers (running titles, page numbers, legal lines).

    Only lines near the top or bottom of a page are candidates: a line that
    repeats in the *body* is content (templated letters, forms), not boilerplate.
    A candidate is removed when the same shape (digits masked) appears on at
    least ``min_page_ratio`` of pages.
    """
    from collections import Counter

    def edges(text: str) -> list[str]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # A page must have a middle for "top/bottom" to mean anything.
        if len(lines) < head_lines + tail_lines + 1:
            return []
        return lines[:head_lines] + lines[-tail_lines:]

    counts: Counter[str] = Counter()
    for page in pages:
        for key in {_line_key(line) for line in edges(page["text"])}:
            if key:
                counts[key] += 1

    threshold = max(3, int(min_page_ratio * len(pages)))
    boilerplate = {key for key, n in counts.items() if n >= threshold}

    cleaned = []
    for page in pages:
        keep = [
            line for line in page["text"].splitlines() if _line_key(line) not in boilerplate
        ]
        cleaned.append({**page, "text": "\n".join(keep)})
    return cleaned


def _line_key(line: str) -> str:
    """Normalise a line so ``Page 3`` and ``Page 4`` share a key."""
    compact = re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))
    return compact if len(compact) >= 4 else ""


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def split_pages(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split pages into overlapping chunks, preserving page provenance."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        pieces_for = lambda text: splitter.split_text(text)  # noqa: E731
    except Exception:  # langchain absent: use the native equivalent
        pieces_for = lambda text: _recursive_chunks(text, chunk_size, overlap)  # noqa: E731

    chunks: list[dict] = []
    for page in pages:
        for local_idx, piece in enumerate(pieces_for(page["text"])):
            piece = piece.strip()
            if len(piece) < 30:  # drop headers, page numbers, stray whitespace
                continue
            chunk_id = hashlib.sha1(
                f"{page['source']}:{page['page']}:{local_idx}:{piece[:80]}".encode()
            ).hexdigest()[:16]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": len(chunks),
                }
            )
    return chunks


def _recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Native stand-in for LangChain's RecursiveCharacterTextSplitter.

    Splits on the largest boundary that fits (paragraph -> line -> sentence ->
    word -> character), then merges neighbours back up to ``size`` with
    ``overlap`` characters of context. Never cuts mid-word unless a single word
    is longer than the whole budget.
    """
    atoms = _split_at(text, size, ["\n\n", "\n", ". ", " ", ""])
    chunks: list[str] = []
    cur = ""
    for atom in atoms:
        if not atom:
            continue
        if len(cur) + len(atom) <= size:
            cur += atom
            continue
        if cur:
            chunks.append(cur)
            cur = (cur[-overlap:] if overlap > 0 else "") + atom
        else:
            cur = atom
        while len(cur) > size:
            chunks.append(cur[:size])
            cur = cur[size - overlap:] if overlap > 0 else cur[size:]
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def _split_at(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively break ``text`` into pieces no longer than ``size``."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep == "":
            return [text[j : j + size] for j in range(0, len(text), size)]
        if sep in text:
            parts = text.split(sep)
            pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
            out: list[str] = []
            for piece in pieces:
                if len(piece) <= size:
                    out.append(piece)
                else:
                    out.extend(_split_at(piece, size, separators[i + 1 :]))
            return out
    return [text[j : j + size] for j in range(0, len(text), size)]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
_STOP = set(
    "a an and are as at be by for from has have how in is it its of on or that "
    "the this to was were what when where which who why will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def build_bm25(chunks: list[dict]):
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus), corpus


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def ingest(pdf_path: Path, embed_backend: str | None = None, reset: bool = True) -> dict:
    settings.ensure_dirs()

    pages = load_pdf(pdf_path)
    if not any(p["text"].strip() for p in pages):
        raise SystemExit(f"No extractable text in {pdf_path} (scanned PDF? needs OCR).")

    pages = strip_boilerplate(pages)
    chunks = split_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise SystemExit("Chunking produced 0 chunks.")

    embedder = Embedder(backend=embed_backend)
    vectors = embedder.encode([c["text"] for c in chunks])

    # ---- persist raw chunks (also the source of truth for BM25 + gold mapping)
    with settings.chunks_jsonl.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- BM25
    bm25, corpus = build_bm25(chunks)
    with settings.bm25_pickle.open("wb") as fh:
        pickle.dump(
            {"bm25": bm25, "corpus": corpus, "ids": [c["chunk_id"] for c in chunks]}, fh
        )

    # ---- ChromaDB
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "l2"}
    )
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ],
    )

    summary = {
        "pdf": pdf_path.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "embedder": embedder.name,
        "vector_dim": int(vectors.shape[1]),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    print(json.dumps(summary, indent=2))
    return summary


def load_chunks() -> list[dict]:
    if not settings.chunks_jsonl.exists():
        raise SystemExit("No index found. Run:  python -m src.ingest --pdf <file.pdf>")
    return [
        json.loads(line)
        for line in settings.chunks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Index a PDF into ChromaDB + BM25")
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--backend", default=None, choices=["st", "onnx", "hash"])
    ap.add_argument("--no-reset", action="store_true", help="append instead of rebuilding")
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"File not found: {args.pdf}")
    ingest(args.pdf, embed_backend=args.backend, reset=not args.no_reset)


if __name__ == "__main__":
    main()
