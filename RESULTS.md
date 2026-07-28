# Evaluation Results

Offline IR evaluation of each ranking strategy against the WANDS relevance
judgments (480 queries, top-50 retrieved per query). Regenerate with `./scripts/run-eval.sh`.

| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency (ms) |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.6707 | 0.8793 | 0.0615 | 0.2485 | 0.7942 | 2 |
| Semantic (bge-small-en-v1.5) | 0.6990 | 0.8872 | 0.0580 | 0.2303 | 0.7988 | 6 |
| Hybrid (RRF, k=60) | 0.7308 | 0.9226 | 0.0638 | 0.2506 | 0.8431 | 9 |
| Hybrid + Cross-Encoder Rerank | 0.7456 | 0.9015 | 0.0642 | 0.2506 | 0.8358 | 1299 |

## RRF constant sweep

Fusion of the same lexical/semantic candidate lists at each k (see `results/rrf-sweep.csv`); the Hybrid row above uses whichever k scored highest on nDCG@10.

| k | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 |
|---|---|---|---|---|---|
| 10 | 0.7220 | 0.9244 | 0.0625 | 0.2506 | 0.8258 |
| 20 | 0.7289 | 0.9252 | 0.0636 | 0.2506 | 0.8365 |
| 40 | 0.7307 | 0.9226 | 0.0638 | 0.2506 | 0.8427 |
| 60 | 0.7308 | 0.9226 | 0.0638 | 0.2506 | 0.8431 |
| 100 | 0.7305 | 0.9230 | 0.0638 | 0.2506 | 0.8435 |
| 200 | 0.7302 | 0.9230 | 0.0638 | 0.2506 | 0.8438 |
