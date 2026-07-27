package com.avanti.search.api;

import java.util.List;
import java.util.Map;

/** Keyed by strategy display name (e.g. "Hybrid (RRF, k=60)"), in the same order as StrategyType's declared values. */
public record CompareResult(String query, Map<String, List<ProductResult>> resultsByStrategy) {
}
