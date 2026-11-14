# Retrieval benchmark

- Corpus chunks: **230**
- Embedder: `onnx:all-MiniLM-L6-v2`
- Chunking: 800 chars / 150 overlap
- RRF k: 60 | context k: 5 | rerank top-n: 15
- Generator: extractive (no LLM)
- Queries: 150 answerable | 50 unanswerable | 25 out-of-domain

| Retriever | Hit@3 | MRR@10 | Ans-in-Context@5 | Abstention | Faithfulness | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM dense | 92.7% | 75.1% | 98.7% | 100.0% | 100.0% | 10.2 ms | 10.9 ms |
| Hybrid MiniLM + BM25 (RRF) | 92.7% | 74.5% | 99.3% | 100.0% | 100.0% | 11.0 ms | 11.9 ms |
| Hybrid + cross-encoder rerank (top 15) | 100.0% | 88.3% | 100.0% | 100.0% | 100.0% | 1125.3 ms | 1353.0 ms |

## Latency breakdown

| Retriever | retrieval p50 | rerank p50 | total p50 | p95 |
|---|---:|---:|---:|---:|
| MiniLM dense | 10.2 ms | 0.0 ms | 10.2 ms | 10.9 ms |
| Hybrid RRF | 11.0 ms | 0.0 ms | 11.0 ms | 11.9 ms |
| Hybrid + rerank | 12.1 ms | 1113.1 ms | 1125.3 ms | 1353.0 ms |

## Rerank lift over RRF

- Hit@3: +7.3 pts
- MRR@10: +13.8 pts
- Added latency: 1113.1 ms per query

## Generation cost

- Generator: extractive (no API key)
- Tokens/query: 1015 in / 25 out
- Cost per 1,000 queries: **$0.11**
- Generation p50: 0 ms

> Latency is split: retrieval = embed + search + fuse; rerank = cross-encoder over the candidates.