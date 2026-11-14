# Retrieval benchmark

- Corpus chunks: **230**
- Embedder: `onnx:all-MiniLM-L6-v2`
- Chunking: 800 chars / 150 overlap
- RRF k: 60 | context k: 5 | rerank top-n: 15
- Generator: extractive (no LLM)
- Queries: 150 answerable | 50 unanswerable | 25 out-of-domain

| Retriever | Hit@3 | MRR@10 | Ans-in-Context@5 | Abstention | Faithfulness | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM dense | 92.7% | 75.1% | 98.7% | 100.0% | 100.0% | 11.2 ms | 14.1 ms |
| Hybrid MiniLM + BM25 (RRF) | 92.7% | 74.5% | 99.3% | 100.0% | 100.0% | 11.9 ms | 15.0 ms |
| Hybrid + cross-encoder rerank (top 15) | 100.0% | 88.3% | 100.0% | 100.0% | 100.0% | 1084.6 ms | 1387.0 ms |

## Latency breakdown

| Retriever | retrieval p50 | rerank p50 | total p50 | p95 |
|---|---:|---:|---:|---:|
| MiniLM dense | 11.2 ms | 0.0 ms | 11.2 ms | 14.1 ms |
| Hybrid RRF | 11.9 ms | 0.0 ms | 11.9 ms | 15.0 ms |
| Hybrid + rerank | 12.3 ms | 1072.6 ms | 1084.6 ms | 1387.0 ms |

## Rerank lift over RRF

- Hit@3: +7.3 pts
- MRR@10: +13.8 pts
- Added latency: 1072.6 ms per query

## Generation cost

- Generator: extractive (no API key)
- Tokens/query: 1015 in / 25 out
- Cost per 1,000 queries: **$0.11**
- Generation p50: 0 ms

> Latency is split: retrieval = embed + search + fuse; rerank = cross-encoder over the candidates.