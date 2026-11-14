# Retrieval benchmark

- Corpus chunks: **230**
- Embedder: `onnx:all-MiniLM-L6-v2`
- Chunking: 800 chars / 150 overlap
- RRF k: 60 · context k: 5
- Generator: extractive (no LLM)
- Queries: 150 answerable · 50 unanswerable · 25 out-of-domain

| Retriever | Hit@3 | MRR@10 | Ans-in-Context@5 | Abstention | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM vector only | 92.7% | 75.1% | 98.7% | n/a | 13.6 ms | 17.4 ms |
| Hybrid MiniLM + BM25 (RRF) | 92.7% | 74.5% | 99.3% | n/a | 14.4 ms | 17.8 ms |

> Latency is **retrieval only** (embed + search + fuse), measured after warm-up.