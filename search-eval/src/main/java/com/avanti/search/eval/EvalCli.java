package com.avanti.search.eval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.common.RelevanceGrade;
import com.avanti.search.common.WandsCsvLoader;
import com.avanti.search.common.WandsDataset;
import com.avanti.search.common.WandsQuery;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.LexicalSearchStrategy;
import com.avanti.search.retrieval.ScoredResult;
import com.avanti.search.retrieval.SearchStrategy;
import com.avanti.search.retrieval.SemanticSearchStrategy;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Runs every currently-implemented SearchStrategy against all 480 WANDS
 * queries and reports nDCG@10 / MRR / Recall@10 / Recall@50 / Precision@10,
 * writing a fresh RESULTS.md each run. Strategies are added to STRATEGY
 * count as later milestones implement them (M2: semantic, M3: hybrid,
 * M4: hybrid+rerank) — this file is re-run in full each time rather than
 * patching individual rows, so RESULTS.md can never go stale relative to
 * the strategies that actually exist.
 */
@Command(name = "search-eval", mixinStandardHelpOptions = true,
        description = "Evaluates all implemented ranking strategies against WANDS relevance judgments.")
public class EvalCli implements Callable<Integer> {

    private static final Logger log = LoggerFactory.getLogger(EvalCli.class);

    private static final int TOP_K = 50;
    private static final int NDCG_K = 10;
    private static final int PRECISION_K = 10;
    private static final int RECALL_SMALL_K = 10;

    @Option(names = "--dataset-dir", defaultValue = "dataset")
    private Path datasetDir;

    @Option(names = "--host", defaultValue = "http://localhost:9200")
    private String host;

    @Option(names = "--results-dir", defaultValue = "results")
    private Path resultsDir;

    @Option(names = "--results-md", defaultValue = "RESULTS.md")
    private Path resultsMdPath;

    @Option(names = "--model-dir", description = "Directory containing the embedding model's model.onnx/tokenizer.json", defaultValue = "models/bge-small-en-v1.5")
    private Path modelDir;

    public static void main(String[] args) {
        int exitCode = new CommandLine(new EvalCli()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        WandsDataset dataset = WandsCsvLoader.load(datasetDir);
        log.info("Loaded {} queries, {} labels", dataset.queries().size(), dataset.labels().size());

        Files.createDirectories(resultsDir);

        try (ElasticsearchClient client = ElasticsearchClients.create(host);
             EmbeddingService embeddingService = new EmbeddingService(modelDir)) {
            List<SearchStrategy> strategies = List.of(
                    new LexicalSearchStrategy(client),
                    new SemanticSearchStrategy(client, embeddingService)
            );

            List<StrategySummary> summaries = new ArrayList<>();
            for (SearchStrategy strategy : strategies) {
                summaries.add(evaluate(strategy, dataset));
            }

            writeResultsMarkdown(summaries);
        }

        return 0;
    }

    private StrategySummary evaluate(SearchStrategy strategy, WandsDataset dataset) throws InterruptedException, IOException {
        log.info("Evaluating strategy: {}", strategy.name());
        List<WandsQuery> queries = dataset.queries();
        List<QueryMetrics> perQuery = new ArrayList<>();
        List<Long> latenciesMs = new ArrayList<>();

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<QueryEvalResult>> futures = new ArrayList<>();
            for (WandsQuery query : queries) {
                futures.add(executor.submit(() -> runOneQuery(strategy, query, dataset)));
            }
            for (Future<QueryEvalResult> future : futures) {
                QueryEvalResult result;
                try {
                    result = future.get();
                } catch (ExecutionException e) {
                    throw new RuntimeException("Query evaluation failed for strategy " + strategy.name(), e.getCause());
                }
                perQuery.add(result.metrics());
                latenciesMs.add(result.latencyMs());
            }
        }

        writePerQueryCsv(strategy.name(), perQuery);
        return StrategySummary.aggregate(strategy.name(), perQuery, latenciesMs);
    }

    private QueryEvalResult runOneQuery(SearchStrategy strategy, WandsQuery query, WandsDataset dataset) {
        Map<String, RelevanceGrade> judgments = dataset.judgmentsFor(query.queryId());

        long start = System.nanoTime();
        List<ScoredResult> results = strategy.search(query.queryText(), TOP_K);
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;

        List<String> rankedIds = results.stream().map(ScoredResult::productId).toList();

        QueryMetrics metrics = new QueryMetrics(
                query.queryId(),
                MetricsCalculator.ndcgAtK(rankedIds, judgments, NDCG_K),
                MetricsCalculator.reciprocalRank(rankedIds, judgments),
                MetricsCalculator.recallAtK(rankedIds, judgments, RECALL_SMALL_K),
                MetricsCalculator.recallAtK(rankedIds, judgments, TOP_K),
                MetricsCalculator.precisionAtK(rankedIds, judgments, PRECISION_K)
        );

        return new QueryEvalResult(metrics, elapsedMs);
    }

    private void writePerQueryCsv(String strategyName, List<QueryMetrics> perQuery) throws IOException {
        String fileName = strategyName.toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9]+", "-")
                .replaceAll("(^-|-$)", "") + ".csv";
        Path path = resultsDir.resolve(fileName);

        CSVFormat format = CSVFormat.Builder.create(CSVFormat.DEFAULT)
                .setHeader("query_id", "ndcg_at_10", "mrr", "recall_at_10", "recall_at_50", "precision_at_10")
                .build();

        try (Writer writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
             CSVPrinter printer = new CSVPrinter(writer, format)) {
            for (QueryMetrics m : perQuery) {
                printer.printRecord(m.queryId(), m.ndcgAt10(), m.mrr(), m.recallAt10(), m.recallAt50(), m.precisionAt10());
            }
        }
        log.info("Wrote per-query metrics to {}", path);
    }

    private void writeResultsMarkdown(List<StrategySummary> summaries) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("# Evaluation Results\n\n");
        sb.append("Offline IR evaluation of each ranking strategy against the WANDS relevance\n");
        sb.append("judgments (480 queries, top-").append(TOP_K).append(" retrieved per query). ");
        sb.append("Regenerate with `./scripts/run-eval.sh`.\n\n");
        sb.append("| Strategy | nDCG@10 | MRR | Recall@10 | Recall@50 | Precision@10 | p95 latency (ms) |\n");
        sb.append("|---|---|---|---|---|---|---|\n");
        for (StrategySummary s : summaries) {
            sb.append("| ").append(s.strategyName())
                    .append(" | ").append(format(s.ndcgAt10()))
                    .append(" | ").append(format(s.mrr()))
                    .append(" | ").append(format(s.recallAt10()))
                    .append(" | ").append(format(s.recallAt50()))
                    .append(" | ").append(format(s.precisionAt10()))
                    .append(" | ").append(s.p95LatencyMs())
                    .append(" |\n");
        }
        Files.writeString(resultsMdPath, sb.toString());
        log.info("Wrote {}", resultsMdPath);
    }

    private static String format(double value) {
        return String.format(Locale.ROOT, "%.4f", value);
    }

    private record QueryEvalResult(QueryMetrics metrics, long latencyMs) {
    }
}
