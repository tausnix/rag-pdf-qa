"""Vector storage: ChromaDB when it works, a NumPy cosine index when it does not.

Chroma 1.x routes collection calls through a Rust bindings layer. On some
Linux + Python 3.12 wheels the native module is present but never initialised,
so *every* client call dies with::

    AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'

That is an environment problem, not a data problem, so the pipeline must not
stop for it. When Chroma cannot be opened we fall back to an in-memory index
built from ``data/index/vectors.npz``.

Score compatibility
-------------------
Embeddings are L2-normalised, so a dot product *is* the cosine similarity. We
report Chroma's squared-L2 distance (``2 - 2*cos``) rather than the cosine
itself, which makes the fallback numerically indistinguishable from the Chroma
path: ``Retriever.vector_search`` keeps using ``1 / (1 + distance)`` and gets
the same numbers and the same ranking from either backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VECTORS_FILENAME = "vectors.npz"


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
def is_chroma_bindings_error(exc: BaseException) -> bool:
    """True for the RustBindingsAPI / missing-bindings AttributeError."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "bindings" in text or "rustbindings" in text


def should_fall_back(exc: BaseException) -> bool:
    """Whether a Chroma failure is an environment issue we can route around.

    The bindings crash is the one we hit on Streamlit Cloud, but a missing
    native library or an unreadable persist directory is equally fatal to
    Chroma and equally irrelevant to retrieval, so those fall back too. Genuine
    programming errors (TypeError, KeyError, ...) are re-raised by the caller.
    """
    return is_chroma_bindings_error(exc) or isinstance(
        exc, (AttributeError, ImportError, OSError, RuntimeError)
    )


# --------------------------------------------------------------------------- #
# NumPy fallback
# --------------------------------------------------------------------------- #
class NumpyVectorIndex:
    """Cosine index over a ``(n_chunks, dim)`` float32 matrix.

    Implements the small slice of Chroma's ``Collection`` API that this project
    uses -- ``query`` and ``count`` -- and returns Chroma-shaped result dicts,
    so ``Retriever`` does not need to know which backend answered.
    """

    def __init__(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self.ids = list(ids)
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embeddings = matrix / np.maximum(norms, 1e-9)
        self.documents = list(documents)
        self.metadatas = list(metadatas)

    # ------------------------------------------------------------------ io
    def save(self, index_dir: Path) -> Path:
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / VECTORS_FILENAME
        np.savez(
            path,
            ids=np.asarray(self.ids, dtype=str),
            embeddings=self.embeddings.astype(np.float32),
        )
        return path

    @classmethod
    def from_chunks(
        cls,
        index_dir: Path,
        ids: list[str],
        embeddings: np.ndarray,
        chunks: list[dict],
        save: bool = True,
    ) -> "NumpyVectorIndex":
        by_id = {c["chunk_id"]: c for c in chunks}
        documents = [by_id[i]["text"] for i in ids]
        metadatas = [
            {
                "source": by_id[i]["source"],
                "page": by_id[i]["page"],
                "chunk_index": by_id[i]["chunk_index"],
            }
            for i in ids
        ]
        index = cls(ids, embeddings, documents, metadatas)
        if save:
            index.save(index_dir)
        return index

    @classmethod
    def load(cls, index_dir: Path) -> "NumpyVectorIndex":
        path = index_dir / VECTORS_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No vector index at {path}. Rebuild with: python run_pipeline.py"
            )
        blob = np.load(path, allow_pickle=False)
        ids = [str(i) for i in blob["ids"]]
        embeddings = blob["embeddings"]

        rows: dict[str, dict] = {}
        chunks_path = index_dir / "chunks.jsonl"
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                rows[chunk["chunk_id"]] = chunk

        documents = [rows[i]["text"] for i in ids]
        metadatas = [
            {
                "source": rows[i]["source"],
                "page": rows[i]["page"],
                "chunk_index": rows[i]["chunk_index"],
            }
            for i in ids
        ]
        return cls(ids, embeddings, documents, metadatas)

    def count(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------- querying
    def _mask(self, where: dict | None) -> np.ndarray | None:
        if not where:
            return None
        mask = np.ones(len(self.ids), dtype=bool)
        for key, cond in where.items():
            if isinstance(cond, dict):
                if "$in" in cond:
                    allowed = set(cond["$in"])
                    mask &= np.asarray([m.get(key) in allowed for m in self.metadatas])
                elif "$ne" in cond:
                    mask &= np.asarray([m.get(key) != cond["$ne"] for m in self.metadatas])
                else:
                    raise ValueError(f"Unsupported metadata operator in {cond!r}")
            else:
                mask &= np.asarray([m.get(key) == cond for m in self.metadatas])
        return mask

    def query(
        self,
        *,
        query_embeddings: list[list[float]] | np.ndarray,
        n_results: int = 10,
        include: list[str] | None = None,
        where: dict | None = None,
    ) -> dict:
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-9)

        # cosine similarity, then Chroma's squared-L2 distance (2 - 2cos)
        similarities = queries @ self.embeddings.T
        mask = self._mask(where)

        ids: list[list[str]] = []
        documents: list[list[str]] = []
        metadatas: list[list[dict]] = []
        distances: list[list[float]] = []

        for row in similarities:
            order = np.argsort(-row, kind="stable")
            if mask is not None:
                order = order[mask[order]]
            order = order[:n_results]
            ids.append([self.ids[i] for i in order])
            documents.append([self.documents[i] for i in order])
            metadatas.append([self.metadatas[i] for i in order])
            distances.append([float(max(0.0, 2.0 - 2.0 * row[i])) for i in order])

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
