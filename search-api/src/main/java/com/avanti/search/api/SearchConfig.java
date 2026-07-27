package com.avanti.search.api;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.inference.EmbeddingService;
import com.avanti.search.inference.RerankerService;
import com.avanti.search.retrieval.ElasticsearchClients;
import com.avanti.search.retrieval.HybridRerankStrategy;
import com.avanti.search.retrieval.HybridRrfSearchStrategy;
import com.avanti.search.retrieval.LexicalSearchStrategy;
import com.avanti.search.retrieval.SearchStrategy;
import com.avanti.search.retrieval.SemanticSearchStrategy;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.util.Map;

/** Wires the four SearchStrategy implementations exactly as search-eval does, over one shared ElasticsearchClient. */
@Configuration
public class SearchConfig {

    @Bean
    public ElasticsearchClient elasticsearchClient(SearchProperties properties) {
        return ElasticsearchClients.create(properties.elasticsearch().host());
    }

    @Bean
    public EmbeddingService embeddingService(SearchProperties properties) throws IOException {
        return new EmbeddingService(properties.models().embeddingDir());
    }

    @Bean
    public RerankerService rerankerService(SearchProperties properties) throws IOException {
        return new RerankerService(properties.models().rerankerDir());
    }

    @Bean
    public LexicalSearchStrategy lexicalSearchStrategy(ElasticsearchClient client) {
        return new LexicalSearchStrategy(client);
    }

    @Bean
    public SemanticSearchStrategy semanticSearchStrategy(ElasticsearchClient client, EmbeddingService embeddingService) {
        return new SemanticSearchStrategy(client, embeddingService);
    }

    @Bean
    public HybridRrfSearchStrategy hybridRrfSearchStrategy(
            LexicalSearchStrategy lexical, SemanticSearchStrategy semantic, SearchProperties properties) {
        return new HybridRrfSearchStrategy(lexical, semantic, properties.hybrid().candidatePoolSize(), properties.hybrid().rrfK());
    }

    @Bean
    public HybridRerankStrategy hybridRerankStrategy(
            HybridRrfSearchStrategy hybrid, ElasticsearchClient client, RerankerService rerankerService, SearchProperties properties) {
        return new HybridRerankStrategy(hybrid, client, rerankerService, properties.hybrid().candidatePoolSize());
    }

    @Bean
    public Map<StrategyType, SearchStrategy> strategiesByType(
            LexicalSearchStrategy lexical, SemanticSearchStrategy semantic,
            HybridRrfSearchStrategy hybrid, HybridRerankStrategy rerank) {
        return Map.of(
                StrategyType.LEXICAL, lexical,
                StrategyType.SEMANTIC, semantic,
                StrategyType.HYBRID, hybrid,
                StrategyType.RERANK, rerank
        );
    }
}
