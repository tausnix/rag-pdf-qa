"""Embedding backends, in order of preference.

1. ``st``   - sentence-transformers, the real ``all-MiniLM-L6-v2`` (384-d).
2. ``onnx`` - the ONNX export of *the same* MiniLM weights, run directly through
              onnxruntime + tokenizers. No torch, and ~20x faster per query than
              the wrapper bundled with chromadb (which re-creates work per call).
3. ``hash`` - deterministic feature hashing. Fast, offline, and deliberately
              dumb: use it only to smoke-test plumbing, never to report metrics.

All backends return L2-normalised float32 vectors, so L2 distance in Chroma
ranks identically to cosine similarity.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

from .config import settings

_DIM = 384
_MAX_TOKENS = 256
_ONNX_SUBDIR = Path("onnx_models") / "all-MiniLM-L6-v2" / "onnx"


class Embedder:
    """Thin wrapper exposing ``encode(list[str]) -> np.ndarray``."""

    def __init__(self, backend: str | None = None, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embed_model
        self.backend = backend or settings.embed_backend
        self._impl = None

        if self.backend == "auto":
            self.backend = self._pick_backend()
        self._load()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _pick_backend() -> str:
        import importlib.util

        if importlib.util.find_spec("sentence_transformers"):
            return "st"
        return "onnx"  # onnxruntime + tokenizers are chromadb dependencies

    def _load(self) -> None:
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer

            self._impl = SentenceTransformer(self.model_name)
            return

        if self.backend == "onnx":
            self._impl = _OnnxMiniLM(_find_onnx_dir())
            return

        if self.backend == "hash":
            return

        raise ValueError(f"Unknown EMBED_BACKEND={self.backend!r} (use st|onnx|hash|auto)")

    # --------------------------------------------------------------- encoding
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)

        if self.backend == "st":
            vecs = self._impl.encode(texts, batch_size=32, show_progress_bar=False)
        elif self.backend == "onnx":
            vecs = self._impl.encode(texts)
        else:
            vecs = np.vstack([self._hash_one(t) for t in texts])

        return _l2_normalise(np.asarray(vecs, dtype=np.float32))

    def _hash_one(self, text: str) -> np.ndarray:
        """Bag-of-words hashing: deterministic, no model, obviously not semantic."""
        vec = np.zeros(_DIM, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
            vec[h % _DIM] += 1.0
        return vec

    @property
    def name(self) -> str:
        return f"{self.backend}:{self.model_name}"


class _OnnxMiniLM:
    """MiniLM inference: tokenize -> transformer -> masked mean pooling."""

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=_MAX_TOKENS)
        self.tokenizer.enable_padding()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ONNX_THREADS", max(1, (os.cpu_count() or 4) // 2)))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(list(texts))
        ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = self.session.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        summed = (hidden * m).sum(axis=1)
        return summed / np.maximum(m.sum(axis=1), 1e-9)


def _find_onnx_dir() -> Path:
    """Locate the MiniLM ONNX export, downloading it via chromadb if absent."""
    candidates = []
    if os.getenv("ONNX_MODEL_DIR"):
        candidates.append(Path(os.environ["ONNX_MODEL_DIR"]))
    candidates.append(Path.home() / ".cache" / "chroma" / _ONNX_SUBDIR)
    candidates.append(Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "chroma" / _ONNX_SUBDIR)

    for candidate in candidates:
        if (candidate / "model.onnx").exists():
            return candidate

    # First run: let chromadb fetch and cache the weights, then reuse the folder.
    # NOTE: the ONNXMiniLM_L6_V2 constructor is lazy -- it only imports onnxruntime,
    # so the actual download happens in _download_model_if_not_exists(). Constructing
    # the object alone would silently do nothing on a clean machine (e.g. CI).
    from chromadb.utils import embedding_functions

    embedder = embedding_functions.ONNXMiniLM_L6_V2()
    try:
        embedder._download_model_if_not_exists()
    except Exception as exc:  # noqa: BLE001 - network/SHA failures all land here
        raise SystemExit(
            "Could not download the MiniLM ONNX model "
            f"({type(exc).__name__}: {exc}). Set ONNX_MODEL_DIR to a folder "
            "containing model.onnx and tokenizer.json, or run with EMBED_BACKEND=hash."
        ) from exc

    for candidate in candidates:
        if (candidate / "model.onnx").exists():
            return candidate
    raise SystemExit(
        "Could not locate the MiniLM ONNX model. Set ONNX_MODEL_DIR to a folder "
        "containing model.onnx and tokenizer.json."
    )


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)
