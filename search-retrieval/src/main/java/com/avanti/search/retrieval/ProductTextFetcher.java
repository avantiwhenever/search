package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import com.avanti.search.common.EmbeddingTextBuilder;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fetches product text for a candidate set of ids, for reranking — the only
 * place a SearchStrategy needs document content rather than just an id/score,
 * since a cross-encoder scores (query, document text) pairs directly.
 */
public final class ProductTextFetcher {

    private static final List<String> TEXT_FIELDS = List.of(
            "product_name", "product_class", "category_hierarchy", "product_description");

    private ProductTextFetcher() {
    }

    public static Map<String, String> fetchTexts(ElasticsearchClient client, List<String> productIds) {
        if (productIds.isEmpty()) {
            return Map.of();
        }

        try {
            SearchResponse<ProductDocument> response = client.search(s -> s
                            .index(SearchConstants.PRODUCTS_INDEX)
                            .size(productIds.size())
                            .query(q -> q.ids(i -> i.values(productIds)))
                            .source(src -> src.filter(f -> f.includes(TEXT_FIELDS))),
                    ProductDocument.class);

            Map<String, String> texts = new HashMap<>();
            for (Hit<ProductDocument> hit : response.hits().hits()) {
                ProductDocument doc = hit.source();
                if (doc != null) {
                    texts.put(hit.id(), EmbeddingTextBuilder.build(
                            doc.productName(), doc.productClass(), doc.categoryHierarchy(), doc.productDescription()));
                }
            }
            return texts;
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to fetch product texts for reranking", e);
        }
    }
}
