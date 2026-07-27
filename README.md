# Semantic Product Search

A Java service for semantic product search over the [WANDS](https://github.com/wayfair/WANDS)
(Wayfair) furniture catalog. Built as a long-term portfolio project focused on one
thing: **retrieval and ranking quality** — hybrid lexical/semantic search plus
cross-encoder reranking, proven with real offline IR evaluation (nDCG, MRR,
Recall@k) rather than eyeballed results.

## Why this project

Most semantic-search portfolios are thin wrappers around a vector DB with no
evaluation story. This one is built to demonstrate the opposite: that each
stage of the ranking pipeline (lexical → semantic → hybrid → reranked)
measurably improves search quality, with the numbers to back it up.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Dataset | [WANDS](https://github.com/wayfair/WANDS) — Wayfair furniture catalog | Ships with `product.csv`, `query.csv`, and `label.csv` (human relevance judgments: Exact/Partial/Irrelevant), which is what makes real IR metrics possible instead of a "looks reasonable" demo. |
| Stack | Java, embeddings + reranking run in-process via **ONNX Runtime** | No Python inference server anywhere — a deliberate differentiator, since most semantic-search portfolios are Python end to end. |
| Search engine | **Elasticsearch** | Considered OpenSearch first (its native RRF fusion is fully open-source, vs. Elasticsearch's being Platinum-gated). But this project hand-rolls RRF fusion client-side in Java rather than using either engine's built-in fusion endpoint (see below), so that licensing distinction turned out not to matter. Elasticsearch was chosen as the more resume-recognizable name; free/Basic license fully covers what's needed here (BM25 + `dense_vector`/kNN). |
| Hybrid fusion | Hand-rolled Reciprocal Rank Fusion (RRF) in Java, not a built-in engine endpoint | More demoable and independently unit-testable than flipping a config flag — it's the artifact that proves understanding of the algorithm, and it gives the eval harness full control to sweep the RRF constant. |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim), via pre-exported ONNX weights from `Xenova/bge-small-en-v1.5` | Purpose-built for retrieval, small, and requires no local Python export step. Unlike most sentence-embedding models, BGE is trained with CLS-token pooling (first token of `last_hidden_state`) rather than mean pooling, and needs an asymmetric instruction prefix on queries only, not documents — both easy to get wrong silently since a mean-pooled embedding still "works," just worse; documented in `EmbeddingService`. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, via `Xenova/ms-marco-MiniLM-L-6-v2`'s **INT8-quantized** ONNX export | Standard MS MARCO cross-encoder. Scoring a 50-candidate pool is one forward pass over a batch of 50 (query, document) pairs — real single-query cost measured at ~1.3s p95 even quantized (quantization roughly halved it from the fp32 export's ~3s), nowhere near "well under 100ms"; a materially heavier workload than the embedding model's batch-of-1 query encode, so it gets the INT8 export while the embedding model stays fp32 (quantizing that would change every stored vector, invalidating the already-recorded M1–M3 numbers). |
| Tokenization in Java | `ai.djl.huggingface:tokenizers` (standalone, not the full DJL engine) | The one piece Java lacks natively — loads `tokenizer.json` directly via JNI binding to the Rust `tokenizers` library, paired with raw ONNX Runtime for the forward pass. |
| Build tool | Maven, multi-module reactor | Declarative XML dependency graphs are easy for a reviewer to skim; no build-script logic to audit. |

## Architecture

Six-module Maven reactor:

```
search/
├── search-common/      domain models + WANDS CSV parsing (no framework deps)
├── search-inference/   ONNX Runtime + tokenizer wrappers — embedding & reranker services
├── search-retrieval/   Elasticsearch client + 4 pluggable SearchStrategy impls + RRF fusion
├── search-ingestion/   CLI: index creation + bulk indexing with embeddings
├── search-api/         thin Spring Boot REST API over search-retrieval
└── search-eval/        CLI: offline IR evaluation harness against label.csv
```

`search-retrieval` is framework-agnostic and shared by both `search-api` and
`search-eval`, so the evaluation harness scores the *exact same* strategy code
the API serves — no risk of eval measuring something different from
production.

Four ranking strategies, each a `SearchStrategy` implementation:
1. **Lexical** — BM25 (`multi_match` across name/class/category/description/features)
2. **Semantic** — dense kNN search over `bge-small-en-v1.5` embeddings
3. **Hybrid** — client-side RRF fusion of the two ranked lists above
4. **Hybrid + Rerank** — top-50 from Hybrid, rescored by the cross-encoder, top-10 returned

## Roadmap

- [x] **M0 — Environment + scaffolding.** Toolchain installed (JDK 21, Maven,
      Colima/Docker), Maven reactor scaffolded, Elasticsearch + Kibana running
      via Docker Compose, WANDS dataset downloaded, `search-common` domain
      models + CSV parsing + unit tests.
- [x] **M1 — Lexical baseline + eval harness.** `LexicalSearchStrategy`, index
      mapping/creation, bulk ingestion of all ~43K products, full
      `search-eval` pipeline, first row of the results table.
- [x] **M2 — Embeddings + semantic search + eval.** `EmbeddingService`
      (ONNX + tokenizer, CLS-pool, normalize, BGE query-prefix handling),
      `dense_vector` field, `SemanticSearchStrategy`, second results row.
- [x] **M3 — Hybrid + RRF + eval.** `RrfFusionService`,
      `HybridRrfSearchStrategy`, sweep of the RRF constant, third results row.
- [x] **M4 — Cross-encoder rerank + eval.** `RerankerService`,
      `HybridRerankStrategy`, fourth results row, latency measurement.
- [ ] **M5 — Polish.** Swagger UI, `/api/search/compare` side-by-side
      endpoint, minimal demo surface, Dockerized `search-api`, architecture
      diagram, full write-up.
- [ ] **M6+ — ongoing.** Server-side RRF comparison, query understanding,
      learning-to-rank layer, alternate embedding models, CI-enforced eval,
      observability.

Target results table (to be filled in as milestones land):

| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.6707 | 0.8793 | 0.0615 | 0.2485 | 0.7942 | 2ms |
| Semantic (bge-small-en-v1.5) | 0.6990 | 0.8872 | 0.0580 | 0.2303 | 0.7988 | 7ms |
| Hybrid (RRF, k=60) | 0.7308 | 0.9226 | 0.0638 | 0.2506 | 0.8431 | 8ms |
| Hybrid + Cross-Encoder Rerank | 0.7456 | 0.9015 | 0.0642 | 0.2506 | 0.8358 | 1288ms |

See `RESULTS.md` (regenerate with `./scripts/run-eval.sh`) for the
canonical, always-current version of this table plus per-query CSVs
under `results/`.

## Data notes

WANDS files have a `.csv` extension but are actually **tab-delimited**, and
the category column is literally named `category hierarchy` (space, not
underscore) — both discovered by inspecting the raw bytes, not the (comma-
delimited) assumption in the original design doc. `WandsCsvLoader` parses
accordingly. Category values also have inconsistent whitespace around `/`
(e.g. `"Furniture / Beds"`), so `EmbeddingTextBuilder` normalizes with a
regex rather than a plain string replace — a naive replace produced
double-spaced text and was caught by a unit test.

`product_features` (pipe-delimited attribute:value pairs) is indexed as a
separate lexical field but deliberately excluded from the embedding text —
attribute noise dilutes sentence-embedding quality more than it helps.

`search-eval` measures p95 latency serially, one query at a time over a
fixed 50-query sample — separately from the metrics computation, which
still runs all 480 queries concurrently since that's just correctness, not
timing. Firing all 480 queries at once as a single burst was the original
design, and it's fine for the cheap strategies, but for the reranker
(~1.3s of genuine per-query cross-encoder work) it measured queueing delay
behind the burst, not realistic single-query latency — a first pass showed
a nonsensical multi-minute p95 that was actually ~480 queries stacked up
behind each other on one CPU, not the model being slow. `RerankerService`
also pins its ONNX session to a single intra-op thread and gates concurrent
forward passes with a semaphore sized to the core count, so that concurrent
callers (e.g. multiple simultaneous API requests) share the CPU instead of
each spawning their own full-width thread pool and thrashing it.

## Local setup

Prerequisites (installed via Homebrew): JDK 26, Maven, Colima + Docker +
Docker Compose plugin.

```bash
# start Elasticsearch + Kibana
docker compose up -d

# fetch the WANDS dataset into dataset/
./scripts/download-dataset.sh

# fetch the embedding model's ONNX weights + tokenizer into models/
./scripts/download-models.sh

# build + test everything
mvn test

# create the products index and bulk-index the catalog
mvn -q -pl search-ingestion -am package -DskipTests
java -jar search-ingestion/target/search-ingestion.jar

# run the offline eval harness, writing RESULTS.md + results/*.csv
./scripts/run-eval.sh
```

Homebrew's `openjdk` formula is keg-only, so `java` may not be on `PATH`
even after `brew install openjdk` — if `mvn`/`java` can't find a runtime,
set `JAVA_HOME` to `$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home`.

Elasticsearch runs with security disabled (`xpack.security.enabled=false`)
for local dev simplicity — not for production use.
