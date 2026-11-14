# Retrieval benchmark

- Corpus chunks: **230**
- Embedder: `onnx:all-MiniLM-L6-v2`
- Chunking: 800 chars / 150 overlap
- RRF k: 60 | context k: 5 | rerank top-n: 15
- Generator: extractive (no LLM)
- Queries: 150 answerable | 50 unanswerable | 25 out-of-domain

| Retriever | Hit@3 | MRR@10 | Ans-in-Context@5 | Abstention | Faithfulness | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM dense | 95.3% | 88.5% | 97.3% | 93.3% | 100.0% | 10.2 ms | 15.7 ms |
| Hybrid MiniLM + BM25 (RRF) | 96.7% | 91.4% | 98.7% | 93.3% | 100.0% | 11.9 ms | 17.3 ms |
| Hybrid + cross-encoder rerank (top 15) | 98.7% | 96.0% | 98.7% | 93.3% | 100.0% | 1340.2 ms | 1885.1 ms |

## Latency breakdown

| Retriever | retrieval p50 | rerank p50 | total p50 | p95 |
|---|---:|---:|---:|---:|
| MiniLM dense | 10.2 ms | 0.0 ms | 10.2 ms | 15.7 ms |
| Hybrid RRF | 11.9 ms | 0.0 ms | 11.9 ms | 17.3 ms |
| Hybrid + rerank | 13.0 ms | 1331.8 ms | 1340.2 ms | 1885.1 ms |

## Rerank lift over RRF

- Hit@3: +2.0 pts
- MRR@10: +4.6 pts
- Added latency: 1331.8 ms per query

## Generation cost

- Generator: extractive (no API key)
- Tokens/query: 916 in / 22 out
- Cost per 1,000 queries: **$0.10**
- Generation p50: 0 ms

> Latency is split: retrieval = embed + search + fuse; rerank = cross-encoder over the candidates.