package com.avanti.search.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.avanti.search.common.EmbeddingTextBuilder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Builds embedding-style text for a candidate set of ids, for reranking. */
public final class ProductTextFetcher {

    private ProductTextFetcher() {
    }

    public static Map<String, String> fetchTexts(ElasticsearchClient client, List<String> productIds) {
        Map<String, ProductDocument> documents = ProductLookup.fetchByIds(client, productIds);

        Map<String, String> texts = new HashMap<>();
        for (Map.Entry<String, ProductDocument> entry : documents.entrySet()) {
            ProductDocument doc = entry.getValue();
            texts.put(entry.getKey(), EmbeddingTextBuilder.build(
                    doc.productName(), doc.productClass(), doc.categoryHierarchy(), doc.productDescription()));
        }
        return texts;
    }
}
