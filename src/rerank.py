"""Cross-encoder reranking (ONNX, no PyTorch).

A bi-encoder (MiniLM) embeds the query and each chunk *independently*, so it can
only compare two summaries. A cross-encoder reads ``(query, chunk)`` together and
scores relevance directly -- much more accurate, far too slow to run over a whole
corpus. So: retrieve a handful of candidates cheaply, then rerank those.

Pair encoding is *dynamic*: each batch is padded to the longest ``(query, chunk)``
pair it actually contains (never a fixed width), and truncated at 256 tokens.
That keeps interactive reranks cheap on CPU -- padding is computed work.

Model: ``Xenova/ms-marco-MiniLM-L-6-v2`` (int8 ONNX, ~22 MB) -- the same
ms-marco MiniLM cross-encoder, exported for transformers.js.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import settings
from .retrieve import Hit

_MODEL_FILES = ("tokenizer.json", "onnx/model_quantized.onnx")
_FALLBACK_FILES = ("tokenizer.json", "onnx/model.onnx")

# Hard cap on a (query, chunk) pair. Longer inputs are truncated; shorter ones are
# padded only to the longest pair in the batch, so short chunks cost less.
MAX_PAIR_TOKENS = 256


class CrossEncoderReranker:
    """Scores query/passage pairs with a 6-layer MiniLM cross-encoder."""

    def __init__(
        self,
        model_dir: Path | None = None,
        max_length: int | None = None,
        top_n: int | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else _ensure_model()
        # Never exceed the hard cap, even if the environment asks for more.
        self.max_length = min(max_length or settings.rerank_max_length, MAX_PAIR_TOKENS)
        self.top_n = top_n or settings.rerank_top_n
        self._threads = threads or int(
            os.getenv("ONNX_THREADS", max(1, (os.cpu_count() or 4) // 2))
        )
        self._session = None
        self._tokenizer = None
        # Exposed for the UI trace: what the last batch actually cost.
        self.last_batch = 0
        self.last_seq_len = 0

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self.max_length)
        # No `length=`: pad to the longest pair in *this* batch. A fixed 512-width
        # tensor would do ~2-3x the work for 160-token pairs.
        self._tokenizer.enable_padding()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

    # ---------------------------------------------------------------- scoring
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance logits for each passage; higher is better."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        self._load()

        encoded = self._tokenizer.encode_batch([(query, p) for p in passages])
        self.last_batch = len(encoded)
        self.last_seq_len = max(len(e.ids) for e in encoded)
        feed = {
            "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.asarray(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        logits = self._session.run(None, feed)[0]
        return np.asarray(logits, dtype=np.float32).reshape(-1)

    def rerank(self, query: str, hits: list[Hit], k: int, top_n: int | None = None) -> list[Hit]:
        return rerank_hits(
            query,
            hits,
            k=k,
            scorer=self.score,
            top_n=top_n if top_n is not None else self.top_n,
        )


def rerank_hits(
    query: str,
    hits: list[Hit],
    k: int,
    scorer,
    top_n: int = 15,
) -> list[Hit]:
    """Pure ranking step: score the first ``top_n`` hits, return the best ``k``.

    ``scorer`` is injected so this can be unit-tested without a model.
    """
    if not hits:
        return []
    candidates = hits[:top_n]
    scores = np.asarray(scorer(query, [h.text for h in candidates]), dtype=np.float32)
    if scores.shape[0] != len(candidates):
        raise ValueError("scorer returned the wrong number of scores")

    for hit, score in zip(candidates, scores):
        hit.rerank_score = float(score)
        # ms-marco heads emit a single logit; sigmoid turns it into a 0-1 relevance
        # probability for display. Ranking uses the raw logit (monotonic).
        hit.rerank_prob = float(1.0 / (1.0 + np.exp(-float(score))))

    ranked = sorted(candidates, key=lambda h: h.rerank_score, reverse=True)[:k]
    return ranked


# --------------------------------------------------------------------------- #
# Model download / cache
# --------------------------------------------------------------------------- #
def cache_root() -> Path:
    override = os.getenv("RERANK_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "rag-pdf-qa" / "ms-marco-MiniLM-L-6-v2"


def _ensure_model() -> Path:
    """Return a folder containing ``model.onnx`` + ``tokenizer.json``.

    Downloads the int8 ONNX export from the Hub on first use and copies it into a
    stable cache path (so later runs work offline).
    """
    dest = cache_root()
    model = dest / "model.onnx"
    tok = dest / "tokenizer.json"
    if model.exists() and tok.exists():
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        repo = settings.rerank_model
        for files in (_MODEL_FILES, _FALLBACK_FILES):
            try:
                for remote in files:
                    local = hf_hub_download(
                        repo_id=repo, filename=remote, cache_dir=str(dest / "hf")
                    )
                    target = dest / Path(remote).name
                    if Path(remote).name == "tokenizer.json":
                        target = tok
                    elif not model.exists():
                        target = model
                    else:
                        continue
                    if not target.exists():
                        Path(target).write_bytes(Path(local).read_bytes())
                if model.exists() and tok.exists():
                    return dest
            except Exception:
                continue
    except ImportError:
        pass

    raise SystemExit(
        f"Cross-encoder model not found in {dest}.\n"
        f"Run once with network access to download {settings.rerank_model}, or set "
        f"RERANK_MODEL_DIR to a folder containing model.onnx and tokenizer.json."
    )


_CACHE: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _CACHE
    if _CACHE is None:
        _CACHE = CrossEncoderReranker()
    return _CACHE
