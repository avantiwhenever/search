# How to run this locally

Step-by-step instructions to get Elasticsearch, the WANDS catalog, and
`search-api` running on your machine, plus how to confirm each piece
actually came up before moving to the next step.

## Prerequisites

Installed via Homebrew: JDK 26, Maven, Colima + Docker + Docker Compose
plugin.

Colima's default VM allocation (2 CPUs) is too little once `search-api`'s
reranker is under any load — a single cross-encoder forward pass can occupy
the one active core long enough that the *same container's* outbound call to
Elasticsearch misses its 1s connect timeout, surfacing as a flaky 500 that
looks like a networking bug but is actually CPU starvation.

```bash
colima start --cpu 6 --memory 8
```

(or edit `~/.colima/default/colima.yaml`).

Homebrew's `openjdk` formula is keg-only, so `java`/`mvn` may not be on
`PATH` even after `brew install openjdk` — if they can't find a runtime, set:

```bash
export JAVA_HOME="$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
```

Note this only needs to be exported in shells that don't already source it
from your shell profile — IntelliJ's integrated terminal and its run
configurations pick it up from your interactive shell's profile
automatically; a plain non-interactive script invocation may not.

## Step-by-step

Run these in order — each step depends on the one before it. Step numbers
match the IntelliJ run configs in the table further down.

### 0. Start Elasticsearch + Kibana

```bash
docker compose up -d
```

**Verify:** `curl -sf http://localhost:9200/_cluster/health` should return
JSON with `"status":"green"` or `"yellow"` (not connection-refused, and not
`"status":"red"`).

### 1. Fetch the WANDS dataset

```bash
./scripts/download-dataset.sh
```

Downloads `product.csv`, `query.csv`, `label.csv` into `dataset/`.

**Verify:** `wc -l dataset/*.csv` — expect ~42,995 products, 480 queries,
233,448 labels (plus one header row each).

### 2. Fetch the embedding + reranker models

```bash
./scripts/download-models.sh
```

Downloads ONNX weights + tokenizers for `bge-small-en-v1.5` (embedding) and
`ms-marco-MiniLM-L-6-v2` (reranker) into `models/`.

**Verify:** `models/` should contain a subdirectory per model, each with a
`model.onnx` and `tokenizer.json`.

### 3. Ingest the catalog into Elasticsearch

```bash
mvn -q -pl search-ingestion -am package -DskipTests
java -jar search-ingestion/target/search-ingestion.jar --recreate
```

This embeds and bulk-indexes all ~43K products — the slow, CPU-bound step
(ONNX inference, not I/O).

**Verify:**

```bash
curl -sf http://localhost:9200/products/_count
```

`count` should match the product row count from step 1's verification
(~42,995).

### 4. Run the offline eval harness (optional at this point)

```bash
./scripts/run-eval.sh
```

Writes `RESULTS.md` and per-strategy CSVs under `results/`. Add
`--baseline-file ci/eval-baseline.json` to gate against the same regression
floor CI used to enforce (see [README's CI section](README.md#ci) — this
eval is no longer run in CI itself, only locally).

**Verify:** exits 0 and prints `All 4 strategies met their baseline floor`
when given `--baseline-file`; otherwise just check `RESULTS.md` was written.

### 5. Run search-api

```bash
mvn -q -pl search-api -am package -DskipTests
java -jar search-api/target/search-api.jar
```

Or run `com.avanti.search.api.SearchApiApplication`'s `main` directly from
your IDE instead of building a jar first.

**Verify the process is actually up and pollable on `localhost:8080`:**

```bash
./scripts/wait-for-search-api.sh
```

Polls `http://localhost:8080/` until it gets *any* HTTP response (a 404 on
an unmapped path still counts — the point is confirming the process is
listening, not that a specific route exists) or times out after 120s. Then
confirm the real endpoints work end to end:

```bash
curl -sf "http://localhost:8080/api/search?query=modern+oak+dining+table&strategy=RERANK" | head -c 500
```

Or visit in a browser:

- `http://localhost:8080/index.html` — the search-comparison demo page
- `http://localhost:8080/swagger-ui.html` — call `/api/search` and
  `/api/search/compare` directly

## Running search-api in Docker instead of locally

```bash
docker compose up -d --build search-api
```

Build stage matches the local toolchain (Maven 3.9.16 + JDK 26); runtime is
a glibc-based JRE image, since ONNX Runtime's and the tokenizer's native
libraries aren't musl-compatible. This mounts the already-downloaded
`models/` directory read-only into the container rather than baking ~150MB
of weights into the image, and points `SEARCH_ELASTICSEARCH_HOST` at the
`elasticsearch` compose service by name — no extra configuration needed
beyond having already run `download-models.sh`.

**Verify:** same as step 5 above — `./scripts/wait-for-search-api.sh`, or
`docker compose logs -f search-api` and watch for the Spring Boot startup
banner.

## Sanity-checking the whole build

```bash
mvn test
```

Not part of the numbered flow above (no step depends on it, and it has no
dedicated IntelliJ config) — run it any time to build and test every
module. Tests that need Elasticsearch or the models skip gracefully
(`Assumptions.assumeTrue`) if steps 0–2 haven't been done yet.

## Running from IntelliJ

Shared run configurations live under `.idea/runConfigurations/` (note:
`.idea/` itself is gitignored in this repo, so these are local-only unless
you deliberately commit that subfolder — see the repo's `.gitignore`):

| # | Config | Runs |
|---|---|---|
| 0 | Start Elasticsearch | `scripts/start-elasticsearch.sh` — `docker compose up -d` for ES + Kibana, then polls `_cluster/health` |
| 1 | Download WANDS dataset | `scripts/download-dataset.sh` |
| 2 | Download ONNX models | `scripts/download-models.sh` |
| 3 | Ingest catalog | `scripts/ingest-catalog.sh` — builds `search-ingestion` and runs it with `--recreate` |
| 4 | Run eval (vs CI baseline) | `scripts/run-eval.sh --baseline-file ci/eval-baseline.json` |
| 5 | Capture demo snapshots | `scripts/capture-demo-snapshots.sh` — hits a running `search-api` to regenerate the GitHub Pages demo data |
| 6 | search-api (app) | `SearchApiApplication.main` directly, no jar build |
| 7 | Wait for search-api :8080 | `scripts/wait-for-search-api.sh` — polls until the app responds |
| 8 | search-api (start + wait until ready) | Compound: runs 6 + 7 together |
| 9 | Stop search-api | `scripts/stop-search-api.sh` — stops it whether it's running as a docker-compose container or a plain local process on :8080 |

Configs 0–4 mirror the numbered steps above one-to-one. Config 5 is a
follow-on task that needs `search-api` already running (see step 5). Configs
6–8 are all ways to run step 5 itself — use 8 when you just want to launch
`search-api` and know as soon as it's actually reachable, without watching
the console for the Spring Boot startup banner yourself. Config 9 stops it
again — useful mainly for the Docker path (`docker compose up -d --build
search-api` has no equivalent "stop" button in the IDE the way a foreground
run configuration does); for config 6/8, IntelliJ's own Stop button on that
running configuration works too.

## Elasticsearch security note

Elasticsearch runs with security disabled (`xpack.security.enabled=false`)
for local dev simplicity — not for production use.

## Where to go next

See [WRITEUP.md](WRITEUP.md) for the full narrative: what was tried at each
milestone, what broke, and how the final numbers were arrived at. See
[README.md](README.md) for architecture, key decisions, and CI.
