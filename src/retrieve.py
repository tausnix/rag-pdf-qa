"""Retrieval: dense (Chroma) + lexical (BM25), fused with Reciprocal Rank Fusion.

    RRF(d) = sum over retrievers of 1 / (k + rank(d)),  k = 60

RRF only uses *ranks*, so the two retrievers never need score normalisation.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

from .config import settings
from .embeddings import Embedder
from .ingest import tokenize


def fuse_rrf(
    ranked_lists: list[list[str]], k: int, rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over ranked id lists.

    Pure function, so it can be unit-tested without a vector store:

        fuse_rrf([["a","b"], ["b","c"]], k=2, rrf_k=60)
        -> [("b", 1/61 + 1/61), ("a", 1/61)]
    """
    scores: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


@dataclass
class Hit:
    chunk_id: str
    text: str
    page: int
    source: str
    score: float
    rank_vector: int | None = None
    rank_bm25: int | None = None


@dataclass
class Retriever:
    """Holds the loaded Chroma collection, BM25 index and embedder."""

    _embedder: Embedder | None = None
    _collection: object | None = None
    _bm25: object | None = None
    _ids: list[str] = field(default_factory=list)
    _texts: dict[str, str] = field(default_factory=dict)
    _meta: dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------ load
    def load(self) -> "Retriever":
        import chromadb

        if not settings.bm25_pickle.exists():
            raise SystemExit("Index missing. Run:  python -m src.ingest --pdf <file.pdf>")

        with settings.bm25_pickle.open("rb") as fh:
            blob = pickle.load(fh)
        self._bm25 = blob["bm25"]
        self._ids = blob["ids"]

        self._embedder = Embedder()
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = client.get_collection(settings.collection_name)

        # warm the text/metadata cache from the chunk store
        import json

        for line in settings.chunks_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            self._texts[c["chunk_id"]] = c["text"]
            self._meta[c["chunk_id"]] = {
                "page": c["page"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
            }
        return self

    # ------------------------------------------------------------- retrieve
    def vector_search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or settings.top_k_vector
        qvec = self._embedder.encode([query])
        res = self._collection.query(
            query_embeddings=qvec.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # Chroma returns L2 distance; convert to a similarity for readability.
            hits.append(
                Hit(
                    chunk_id=cid,
                    text=doc,
                    page=int(meta["page"]),
                    source=str(meta["source"]),
                    score=1.0 / (1.0 + float(dist)),
                )
            )
        return hits

    def bm25_search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or settings.top_k_bm25
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits: list[Hit] = []
        for i in order:
            if scores[i] <= 0:
                continue
            cid = self._ids[i]
            meta = self._meta.get(cid, {})
            hits.append(
                Hit(
                    chunk_id=cid,
                    text=self._texts.get(cid, ""),
                    page=int(meta.get("page", 0)),
                    source=str(meta.get("source", "")),
                    score=float(scores[i]),
                )
            )
        return hits

    def hybrid_search(self, query: str, k: int | None = None) -> list[Hit]:
        """Fuse the two ranked lists with RRF; returns top ``k`` (default context_k)."""
        k = k or settings.context_k
        vec = self.vector_search(query)
        lex = self.bm25_search(query)

        by_id: dict[str, Hit] = {h.chunk_id: h for h in vec}
        for hit in lex:
            by_id.setdefault(hit.chunk_id, hit)

        fused = fuse_rrf(
            [[h.chunk_id for h in vec], [h.chunk_id for h in lex]],
            k=k,
            rrf_k=settings.rrf_k,
        )

        ranked: list[Hit] = []
        for cid, score in fused:
            hit = by_id[cid]
            hit.score = score
            hit.rank_vector = next(
                (i for i, h in enumerate(vec, start=1) if h.chunk_id == cid), None
            )
            hit.rank_bm25 = next(
                (i for i, h in enumerate(lex, start=1) if h.chunk_id == cid), None
            )
            ranked.append(hit)
        return ranked


_CACHE: Retriever | None = None


def get_retriever() -> Retriever:
    """Process-wide singleton so the Streamlit app loads the index once."""
    global _CACHE
    if _CACHE is None:
        _CACHE = Retriever().load()
    return _CACHE
