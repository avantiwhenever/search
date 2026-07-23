package com.avanti.search.common;

/**
 * Builds the text passed to the embedding model for a product. Deliberately
 * excludes {@code productFeatures} (pipe-delimited attribute:value pairs) —
 * that attribute noise dilutes sentence-embedding quality more than it helps;
 * it is still indexed as a separate lexical field for BM25.
 */
public final class EmbeddingTextBuilder {

    private static final int MAX_DESCRIPTION_CHARS = 1200;

    private EmbeddingTextBuilder() {
    }

    public static String build(WandsProduct product) {
        StringBuilder text = new StringBuilder(product.productName());
        if (product.productClass() != null) {
            text.append(". ").append(product.productClass());
        }
        if (product.categoryHierarchy() != null) {
            // Source category hierarchies are inconsistently spaced around "/" (e.g. "Furniture / Beds"
            // vs "Furniture/Beds"), so collapse any surrounding whitespace rather than a plain replace.
            text.append(". ").append(product.categoryHierarchy().replaceAll("\\s*/\\s*", " > "));
        }
        if (product.productDescription() != null) {
            text.append(". ").append(truncate(product.productDescription(), MAX_DESCRIPTION_CHARS));
        }
        return text.toString();
    }

    private static String truncate(String text, int maxChars) {
        return text.length() <= maxChars ? text : text.substring(0, maxChars);
    }
}
