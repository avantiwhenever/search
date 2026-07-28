# Semantic Product Search — Write-up

This is the long-form account of building a hybrid lexical/semantic product
search engine over the WANDS furniture catalog, in Java, with every ranking
stage measured against real relevance judgments rather than eyeballed. The
[README](README.md) is the reference doc (architecture, key-decisions
table, CI); [HOWTO.md](HOWTO.md) has step-by-step local setup; this is the
narrative — what was tried, what broke, and what the numbers actually
showed at each stage.

## The premise

Most "semantic search" portfolio projects are a thin wrapper around a vector
database with a few example queries that look reasonable. That's not a
demonstration of anything — it's a demo of the fact that embeddings exist.
The interesting question is whether each added piece of complexity (semantic
retrieval, hybrid fusion, cross-encoder reranking) actually earns its keep in
measured search quality, and what it costs in latency to get there. WANDS
ships with 480 real queries and ~233K graded relevance judgments
(Exact/Partial/Irrelevant) against a ~43K-product Wayfair catalog, which is
what makes that question answerable instead of a matter of taste.

The plan from the start was four ranking strategies, each strictly building
on the last, each scored with the same offline IR metrics (nDCG@10, MRR,
Recall@10/50, Precision@10) against the same 480 queries:

1. **Lexical** — plain BM25.
2. **Semantic** — dense kNN over sentence embeddings.
3. **Hybrid** — Reciprocal Rank Fusion of the two ranked lists.
4. **Hybrid + Rerank** — a cross-encoder rescoring the hybrid candidate pool.

## M0–M1: scaffolding and the lexical baseline

The Maven reactor came together in six modules with `search-retrieval`
deliberately framework-agnostic, shared by both the eval harness and the API
— the point being that `search-eval`'s numbers describe *exactly* the code
`search-api` serves, never a parallel implementation that could quietly
drift from what's measured.

WANDS' own files immediately punished the assumption that a `.csv` extension
means comma-delimited: they're tab-delimited, and the category column header
is `category hierarchy` (a literal space, not an underscore). Both were
caught by looking at the raw bytes rather than trusting the extension.
Category values also had inconsistent spacing around `/` (`"Furniture / Beds"`
vs `"Furniture/Beds"`), which a naive string replace turned into
double-spaced text — caught by a unit test, fixed with a regex.

First real number: Lexical (BM25) landed at **nDCG@10 = 0.6707**, a strong
baseline that later stages would need to clearly beat to justify their added
complexity and latency.

## M2: embeddings, and the pooling detail that's easy to get silently wrong

`bge-small-en-v1.5` was picked for being purpose-built for retrieval, small,
and available as pre-exported ONNX weights (no local Python export step
required — a deliberate constraint, since the entire inference stack runs
in-process via ONNX Runtime with no Python server anywhere).

The detail worth flagging: BGE is trained with **CLS-token pooling** (the
first token of `last_hidden_state`), not the mean-pooling most
sentence-transformer setups default to. A mean-pooled embedding from this
same model still produces *a* vector that *looks* like a valid embedding —
cosine similarity still returns numbers between -1 and 1, nothing throws —
it's just a worse embedding, silently. Same story for the asymmetric query
prefix ("Represent this sentence for searching relevant passages: ", queries
only, never documents) — the kind of requirement that's easy to skip because
skipping it doesn't produce an error, just degraded ranking that's hard to
notice without an eval harness already in place to catch it.

Semantic alone scored **nDCG@10 = 0.6990** — better than lexical on ranking
quality, but with lower Recall@50 (0.2303 vs 0.2485), the first hint that
lexical and semantic retrieval are making genuinely different kinds of
mistakes, not just weaker/stronger versions of the same one.

## M3: hybrid fusion, and letting the eval harness pick the constant instead of guessing it

Reciprocal Rank Fusion is hand-rolled client-side in Java rather than using
Elasticsearch's or OpenSearch's built-in fusion endpoint. That's a real
tradeoff — a config flag would be less code — but it's also the artifact
that proves the algorithm is understood rather than invoked, and it's what
lets the eval harness sweep the fusion constant `k` directly against offline
metrics instead of trusting the textbook default of 60.

The sweep candidate-generation is the one piece of this project's evaluation
design worth calling out as an actual efficiency win rather than just a
correctness one: fusing a lexical+semantic candidate pair at a different `k`
is pure in-memory arithmetic, so the harness fetches each query's candidate
lists from Elasticsearch/ONNX exactly once and reuses them across all six
swept values of `k`, rather than re-querying per candidate.

The sweep curve turned out flat from k=40 to k=200 (nDCG@10 between 0.7302
and 0.7308), with k=60 — the standard default from the original RRF paper —
sitting right at the empirical peak. That's a nice validation of the
default, but it's a validation earned by measuring, not assumed in advance.

Hybrid reached **nDCG@10 = 0.7308**, clearly ahead of both individual
strategies, at Recall@50 = 0.2506 — recovering the recall that semantic
alone had given up, which is exactly what fusing two different retrieval
signals should do.

## M4: reranking, and two bugs a first "successful" eval run would have hidden

This milestone is where the project's insistence on measuring rather than
trusting paid off hardest, because the *first* full eval run after wiring up
the cross-encoder reranker "succeeded" — exit code 0, a results table
written — and the results table was nonsense. `Hybrid + Cross-Encoder Rerank`
reported a p95 latency of **265,761 ms**. Four and a half minutes. Per query.

It would have been easy to write that number down as "reranking is
expensive" and move on. Instead: reproduce it in isolation. A single
reranker call, run with no concurrency at all, cost ~3.2 seconds — real, but
nowhere near 4.5 minutes. The gap was the eval harness itself: it evaluates
all 480 queries concurrently via a virtual-thread-per-query pool, which is
exactly right for measuring *correctness* (nDCG, MRR, etc. don't care about
concurrency) but wrong for measuring *latency*, because it means firing 480
simultaneous cross-encoder forward passes at a 10-core machine. The reported
"p95 latency" was actually queueing delay behind that self-inflicted burst,
not the model being slow. Lexical and semantic hadn't shown the same
distortion only because their per-query cost was cheap enough (sub-second)
that 480-way contention didn't compound into anything visible.

Two independent fixes followed, not one:

- **The real, measurable cost was still real.** ~3.2s per query for a
  50-candidate cross-encoder batch is genuinely far from the "well under
  100ms" a cross-encoder card promises for a single pair — the mismatch is
  the batch size, not the model. Switching to the INT8-quantized ONNX export
  (vs. the fp32 export used for the embedding model) cut it to ~1.5s, a real
  ~2x, not a rounding error. The embedding model stayed on fp32 deliberately:
  quantizing it would change every stored vector and invalidate the
  already-recorded M1–M3 numbers for no benefit, since its own per-query cost
  (one short-text batch-of-1 encode) was never the bottleneck.
- **The measurement methodology was wrong regardless.** Latency is now
  measured serially, one query at a time, over a fixed 50-query sample —
  decoupled from the (still concurrent, still fast) correctness pass over
  all 480. `RerankerService` also pins its ONNX session to a single intra-op
  thread and gates concurrent forward passes with a semaphore sized to the
  core count, so *real* concurrent callers (multiple simultaneous API
  requests, not an artificial eval burst) share the CPU instead of each
  spawning a full-width thread pool and thrashing it.

The corrected, honest numbers: **nDCG@10 = 0.7456** (best of all four
strategies) at a real single-query p95 of **~1.3 seconds** — a latency cost
that's now an honest engineering tradeoff to weigh, not an artifact to
explain away.

## M5: the API layer, and the toolchain problems that only show up at packaging time

Adding `search-api` — a thin Spring Boot layer exposing `/api/search` and a
side-by-side `/api/search/compare`, a Swagger UI, and a minimal static demo
page — surfaced three unrelated toolchain issues in quick succession, each
only visible once you actually try to *package and run* the thing rather
than just compile it:

1. **`spring-boot-maven-plugin` 3.3.4 couldn't repackage JDK 26 bytecode** —
   its bundled ASM version predates class file major version 70. The fix
   wasn't to downgrade the project's JDK target (that would silently undo
   an explicit earlier decision to standardize the whole reactor on JDK 26);
   it was to bump Spring Boot to 3.5.16 (and springdoc to the matching
   2.8.17), the current release in the same major line.
2. **That bump then exposed a version conflict already latent in the
   project**: an explicit `httpclient5` pin (5.3.1, added early on for the
   Elasticsearch client's transport) was shadowing whatever newer version
   Spring's dependency BOM would otherwise have supplied — one that Spring
   Boot 3.5's own `HttpClientAutoConfiguration` now requires a class from
   (`TlsSocketStrategy`) that 5.3.1 doesn't have. The fix was to bring the
   explicit pin up to what Spring's BOM already recommends (5.5.2) rather
   than delete the pin and hope.
3. **Every `@RequestParam` without an explicit name 500'd at request time**,
   not at compile time — Spring resolves unnamed parameters via reflection,
   which requires the `-parameters` javac flag. `spring-boot-starter-parent`
   sets that by default; this project doesn't inherit from it, so it had to
   be added explicitly to `maven-compiler-plugin`'s configuration.

None of these three would show up in `mvn compile`, or even `mvn test` — they
only surface when you actually package the Spring Boot jar and hit a live
endpoint, which is exactly why each one was found by doing that rather than
trusting a green build.

A fourth issue surfaced once the app was actually running in Docker rather
than as a local `java -jar`: the `/api/search?strategy=RERANK` endpoint
threw an intermittent `ConnectTimeoutException` trying to reach
`elasticsearch:9200` — from inside the *same* container that was supposedly
too slow to open a socket to its neighbor. The actual cause was one level
down: Colima's VM was allocated 2 CPUs by default, against a 10-core host. A
single cross-encoder forward pass, even correctly pinned to one intra-op
thread, can occupy an entire core for over a second; on a 2-CPU VM that's
half the machine, and Docker's own CFS scheduling jitter under that
contention was apparently enough to occasionally blow past the Elasticsearch
client's default 1-second connect timeout for a completely unrelated
request. The honest fix wasn't to lengthen that timeout — that would have
quietly rewarded an under-provisioned VM by making requests wait longer
without addressing why they were waiting. It was to give Colima the
resources the workload actually needs (`colima start --cpu 6 --memory 8`),
after which the same request that had been flaky became a boring, consistent
~1.4 seconds every time.

## Final results

| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency |
|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.6707 | 0.8793 | 0.0615 | 0.2485 | 0.7942 | 2ms |
| Semantic (bge-small-en-v1.5) | 0.6990 | 0.8872 | 0.0580 | 0.2303 | 0.7988 | 7ms |
| Hybrid (RRF, k=60) | 0.7308 | 0.9226 | 0.0638 | 0.2506 | 0.8431 | 8ms |
| Hybrid + Cross-Encoder Rerank | 0.7456 | 0.9015 | 0.0642 | 0.2506 | 0.8358 | 1288ms |

Every stage of the pipeline (lexical → semantic → hybrid → reranked) measurably
improved nDCG@10 over the one before it, which was the entire thesis of the
project. The one strategy that *doesn't* strictly improve on every metric is
the reranker: MRR and Precision@10 both dip slightly versus Hybrid alone
(0.9015 vs 0.9226, 0.8358 vs 0.8431) even as nDCG@10 improves — a real,
plausible outcome (the cross-encoder occasionally reorders a near-tied exact
match below a more topically-relevant result) and a genuine cost/quality
tradeoff to weigh against the ~1.3s latency, not something to paper over.

## What's next

The README's roadmap tracks this as "M6+ — ongoing": a server-side RRF
comparison (Elasticsearch's/OpenSearch's built-in fusion vs. the hand-rolled
version, now that both are provably correct), query understanding
(spelling/synonyms), a learning-to-rank layer instead of a fixed RRF
constant, alternate embedding models, and CI-enforced eval so a regression
in ranking quality fails a build the same way a broken test does.

The recurring lesson across M4 and M5 was the same one twice: a green build
and a 200 response are necessary, not sufficient. The reranker's first eval
run "succeeded" with a nonsensical number; the API's first packaged jar
"built" and then failed at startup, then at request time. Both were only
caught by actually running the thing and looking at what came back — the
same discipline the whole project is built to demonstrate at the ranking
layer, applied to the engineering around it too.
