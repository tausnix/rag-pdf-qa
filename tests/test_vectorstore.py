"""NumPy fallback vector index and the ChromaDB bindings crash we hit on Cloud.

Streamlit Community Cloud (Linux + Python 3.12) raised::

    Indexing failed: 'RustBindingsAPI' object has no attribute 'bindings'

Those tests pin the behaviour that keeps the app usable when that happens.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import replace

import numpy as np
import pytest

import src.retrieve as retrieve_mod
from src.config import settings as real_settings
from src.vectorstore import (
    NumpyVectorIndex,
    is_chroma_bindings_error,
    should_fall_back,
)

CHUNKS = [
    {"chunk_id": "a", "text": "alpha text", "source": "doc.pdf", "page": 1, "chunk_index": 0},
    {"chunk_id": "b", "text": "beta text", "source": "doc.pdf", "page": 2, "chunk_index": 1},
    {"chunk_id": "c", "text": "gamma text", "source": "other.pdf", "page": 3, "chunk_index": 2},
]
# a is the query direction; b is close to it; c is orthogonal
EMBEDDINGS = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _index() -> NumpyVectorIndex:
    return NumpyVectorIndex(
        ids=[c["chunk_id"] for c in CHUNKS],
        embeddings=EMBEDDINGS,
        documents=[c["text"] for c in CHUNKS],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in CHUNKS],
    )


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
def test_bindings_error_is_recognised_and_routable():
    exc = AttributeError("'RustBindingsAPI' object has no attribute 'bindings'")
    assert is_chroma_bindings_error(exc)
    assert should_fall_back(exc)


def test_unrelated_programming_error_is_not_routed_away():
    assert not is_chroma_bindings_error(TypeError("bad argument"))
    assert not should_fall_back(KeyError("missing"))


# --------------------------------------------------------------------------- #
# Fallback index semantics
# --------------------------------------------------------------------------- #
def test_query_orders_by_cosine_and_reports_chroma_distances():
    res = _index().query(query_embeddings=[1.0, 0.0, 0.0], n_results=3)
    assert res["ids"] == [["a", "b", "c"]]
    distances = res["distances"][0]
    # squared L2 on unit vectors: d = 2 - 2*cos
    assert distances[0] == pytest.approx(0.0)
    assert distances[2] == pytest.approx(2.0)
    cos_b = float(np.dot([1, 0, 0], [0.9, 0.1, 0]) / np.linalg.norm([0.9, 0.1, 0]))
    assert distances[1] == pytest.approx(2 - 2 * cos_b)


def test_query_matches_chroma_result_shape():
    res = _index().query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2)
    assert set(res) == {"ids", "documents", "metadatas", "distances"}
    assert len(res["ids"]) == 1  # one row per query
    assert res["documents"][0][0] == "alpha text"
    assert res["metadatas"][0][0]["page"] == 1


def test_metadata_filter_restricts_candidates():
    res = _index().query(
        query_embeddings=[1.0, 0.0, 0.0], n_results=5, where={"source": "other.pdf"}
    )
    assert res["ids"] == [["c"]]


def test_metadata_in_filter_supports_multiple_values():
    res = _index().query(
        query_embeddings=[1.0, 0.0, 0.0],
        n_results=5,
        where={"page": {"$in": [2, 3]}},
    )
    assert res["ids"] == [["b", "c"]]


def test_save_load_round_trip(tmp_path):
    _index().save(tmp_path)
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in CHUNKS) + "\n", encoding="utf-8"
    )

    restored = NumpyVectorIndex.load(tmp_path)
    assert restored.count() == 3
    assert restored.query(query_embeddings=[0.0, 1.0, 0.0], n_results=1)["ids"] == [["c"]]


# --------------------------------------------------------------------------- #
# Retriever integration: Chroma raises, NumPy answers
# --------------------------------------------------------------------------- #
class _FakeBM25:
    def get_scores(self, tokens):
        return [0.0, 5.0, 0.0]


class _FakeEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode(self, texts):
        return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))

    @property
    def name(self) -> str:
        return "fake:test"


def _write_index(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in CHUNKS) + "\n", encoding="utf-8"
    )
    with (index_dir / "bm25.pkl").open("wb") as fh:
        pickle.dump(
            {"bm25": _FakeBM25(), "corpus": [], "ids": [c["chunk_id"] for c in CHUNKS]}, fh
        )
    NumpyVectorIndex(
        ids=[c["chunk_id"] for c in CHUNKS],
        embeddings=EMBEDDINGS,
        documents=[c["text"] for c in CHUNKS],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in CHUNKS],
    ).save(index_dir)
    return index_dir


def test_retriever_falls_back_when_chroma_bindings_are_missing(tmp_path, monkeypatch):
    index_dir = _write_index(tmp_path)

    monkeypatch.setattr(
        retrieve_mod,
        "settings",
        replace(
            real_settings,
            index_dir=index_dir,
            chroma_dir=tmp_path / "chroma",
            eval_dir=tmp_path / "eval",
        ),
    )
    monkeypatch.setattr(retrieve_mod, "Embedder", _FakeEmbedder)

    class _BrokenClient:
        def __init__(self, *args, **kwargs) -> None:
            raise AttributeError("'RustBindingsAPI' object has no attribute 'bindings'")

    monkeypatch.setattr("chromadb.PersistentClient", _BrokenClient)

    retriever = retrieve_mod.Retriever().load()

    # Chroma never produced a collection; the NumPy index took over.
    assert isinstance(retriever._collection, NumpyVectorIndex)
    hits = retriever.vector_search("anything", k=2)
    assert [h.chunk_id for h in hits] == ["a", "b"]
    assert hits[0].vector_score == pytest.approx(1.0)

    # Hybrid retrieval still works end to end (BM25 + dense -> RRF).
    fused = retriever.hybrid_search("anything", k=3)
    assert fused[0].chunk_id == "b"  # BM25 #1 + dense #2 wins RRF
    assert {h.chunk_id for h in fused} == {"a", "b", "c"}


def test_retriever_rebuilds_a_missing_checkpoint_from_the_chunk_store(tmp_path, monkeypatch):
    """The deployed app already had chunks.jsonl + bm25.pkl but no vectors.npz,
    because the crashing ingest died before writing it. Loading must self-heal
    instead of demanding a re-index."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.jsonl").write_text(
        json.dumps(CHUNKS[0]) + "\n", encoding="utf-8"
    )
    with (index_dir / "bm25.pkl").open("wb") as fh:
        pickle.dump({"bm25": _FakeBM25(), "corpus": [], "ids": ["a"]}, fh)

    monkeypatch.setattr(
        retrieve_mod,
        "settings",
        replace(real_settings, index_dir=index_dir, chroma_dir=tmp_path / "chroma"),
    )
    monkeypatch.setattr(retrieve_mod, "Embedder", _FakeEmbedder)

    class _BrokenClient:
        def __init__(self, *args, **kwargs) -> None:
            raise AttributeError("'RustBindingsAPI' object has no attribute 'bindings'")

    monkeypatch.setattr("chromadb.PersistentClient", _BrokenClient)

    retriever = retrieve_mod.Retriever().load()

    assert isinstance(retriever._collection, NumpyVectorIndex)
    assert retriever._collection.count() == 1
    assert [h.chunk_id for h in retriever.vector_search("anything", k=1)] == ["a"]
    # the rebuilt checkpoint is persisted for the next session
    assert (index_dir / "vectors.npz").exists()
