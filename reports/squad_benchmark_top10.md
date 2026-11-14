# Retrieval benchmark

- Corpus chunks: **230**
- Embedder: `onnx:all-MiniLM-L6-v2`
- Chunking: 800 chars / 150 overlap
- RRF k: 60 | context k: 5 | rerank top-n: 10
- Generator: extractive (no LLM)
- Queries: 150 answerable | 50 unanswerable | 25 out-of-domain

| Retriever | Hit@3 | MRR@10 | Ans-in-Context@5 | Abstention | Faithfulness | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid + cross-encoder rerank (top 10) | 98.0% | 95.4% | 98.0% | 93.3% | 100.0% | 843.7 ms | 1142.1 ms |

## Latency breakdown

| Retriever | retrieval p50 | rerank p50 | total p50 | p95 |
|---|---:|---:|---:|---:|
| Hybrid + rerank | 12.5 ms | 829.4 ms | 843.7 ms | 1142.1 ms |

## Generation cost

- Generator: extractive (no API key)
- Tokens/query: 910 in / 22 out
- Cost per 1,000 queries: **$0.10**
- Generation p50: 0 ms

> Latency is split: retrieval = embed + search + fuse; rerank = cross-encoder over the candidates.