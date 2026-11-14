"""Central configuration: paths, retrieval knobs and model names.

Every value can be overridden with an environment variable, so the benchmark
protocol (chunk size, k, RRF constant) is explicit and reproducible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- paths ---
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    index_dir: Path = ROOT / "data" / "index"
    eval_dir: Path = ROOT / "data" / "eval"
    reports_dir: Path = ROOT / "reports"
    chroma_dir: Path = ROOT / "chroma"

    # --- chunking ---
    chunk_size: int = _int("CHUNK_SIZE", 800)          # characters
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)

    # --- retrieval ---
    embed_model: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    embed_backend: str = os.getenv("EMBED_BACKEND", "auto")
    top_k_vector: int = _int("TOP_K_VECTOR", 10)
    top_k_bm25: int = _int("TOP_K_BM25", 10)
    rrf_k: int = _int("RRF_K", 60)                     # Reciprocal Rank Fusion constant
    context_k: int = _int("CONTEXT_K", 5)              # chunks handed to the LLM

    # --- reranking (cross-encoder) ---
    rerank_model: str = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    rerank_top_n: int = _int("RERANK_TOP_N", 15)       # candidates scored by the cross-encoder
    rerank_max_length: int = _int("RERANK_MAX_LENGTH", 256)

    # --- generation ---
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    temperature: float = _float("TEMPERATURE", 0.0)

    # --- cost model (USD per 1M tokens, Gemini 2.0 Flash list price) ---
    price_input_per_mtok: float = _float("PRICE_INPUT_PER_MTOK", 0.10)
    price_output_per_mtok: float = _float("PRICE_OUTPUT_PER_MTOK", 0.40)

    # --- faithfulness ---
    faithfulness_min_overlap: float = _float("FAITHFULNESS_MIN_OVERLAP", 0.6)

    @property
    def collection_name(self) -> str:
        return os.getenv("COLLECTION", "pdf_chunks")

    @property
    def chunks_jsonl(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def bm25_pickle(self) -> Path:
        return self.index_dir / "bm25.pkl"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.index_dir, self.eval_dir, self.reports_dir, self.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()

# Text used by the extractive fallback (--no-llm) to signal "not in the document".
REFUSAL = "I couldn't find that in the document."
