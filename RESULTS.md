# Evaluation Results

Offline IR evaluation of each ranking strategy against the WANDS relevance
judgments (480 queries, top-50 retrieved per query). Regenerate with `./scripts/run-eval.sh`.

| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency (ms) |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.6707 | 0.8793 | 0.0615 | 0.2485 | 0.7942 | 991 |
| Semantic (bge-small-en-v1.5) | 0.6990 | 0.8872 | 0.0580 | 0.2303 | 0.7988 | 248 |
