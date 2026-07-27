package com.avanti.search.api;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.nio.file.Path;

@ConfigurationProperties(prefix = "search")
public record SearchProperties(
        Elasticsearch elasticsearch,
        Models models,
        Hybrid hybrid
) {
    public record Elasticsearch(String host) {
    }

    public record Models(Path embeddingDir, Path rerankerDir) {
    }

    /**
     * rrfK=60 is the constant search-eval's offline sweep found best on the
     * WANDS eval set (see RESULTS.md's RRF sweep table — the curve is flat
     * from k=40-200, so this isn't a fragile choice); a real deployment
     * would periodically re-sweep against fresh judgments rather than
     * trust a value picked once.
     */
    public record Hybrid(int rrfK, int candidatePoolSize) {
    }
}
