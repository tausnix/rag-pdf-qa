# Retrieval benchmark

- Corpus chunks: **172**
- Embedder: `onnx:all-MiniLM-L6-v2`
- Chunking: 800 chars / 150 overlap
- RRF k: 60 · context k: 5
- Generator: extractive (no LLM)
- Queries: 150 answerable · 50 unanswerable · 25 out-of-domain

| Retriever | Hit@3 | MRR@10 | Ans-in-Context@5 | Abstention | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM vector only | 94.7% | 89.5% | 98.0% | 88.0% | 16.6 ms | 43.7 ms |
| Hybrid MiniLM + BM25 (RRF) | 97.3% | 93.1% | 98.7% | 86.7% | 15.4 ms | 26.8 ms |

> Latency is **retrieval only** (embed + search + fuse), measured after warm-up.