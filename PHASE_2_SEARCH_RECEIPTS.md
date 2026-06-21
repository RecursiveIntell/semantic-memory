# Phase 2 - Deterministic search context and search receipts

## Objective

Make search deterministic under an explicit context and make search behavior explainable through receipt-ready metadata.

## P1 issues covered

- F-004: filtered HNSW under-return fallback, if not completed in Phase 1
- F-006: replay-clean recency
- F-009: search receipt skeleton

## Design target

Add a search context similar to:

```rust
#[derive(Debug, Clone)]
pub struct SearchContext {
    pub evaluation_time: DateTime<Utc>,
    pub receipt_mode: ReceiptMode,
    pub exactness_profile: ExactnessProfile,
    pub request_id: Option<String>,
}

pub enum ReceiptMode {
    Disabled,
    ExplainOnly,
    ReturnReceipt,
}

pub enum ExactnessProfile {
    Default,
    PreferExact,
    AllowApproximate,
}
```

Default public APIs may create a context once:

```rust
let context = SearchContext::default_now();
```

Internal ranking must use `context.evaluation_time`, not repeated `Utc::now()` calls.

## Search receipt skeleton

Add a type equivalent to:

```rust
pub struct VectorSearchReceiptV1 {
    pub receipt_id: String,
    pub evaluation_time: DateTime<Utc>,
    pub query_embedding_digest: Option<String>,
    pub search_profile: String,
    pub candidate_backend: String,
    pub requested_candidates: usize,
    pub returned_candidates: usize,
    pub post_filter_candidates: usize,
    pub fallback: Option<String>,
    pub exact_rerank: bool,
    pub result_ids: Vec<String>,
    pub degradations: Vec<String>,
}
```

This can live as an internal struct first. It does not need full v11 artifact machinery.

## Filtered HNSW under-return rule

When HNSW returns candidates but post-filter candidates are insufficient, do not silently return weak results.

Minimum behavior:

```text
if post_filter_vector_hits < requested_top_k and filter_scope_has_more_possible_rows:
    run exact/brute-force vector search over filtered scope
    merge/rerank according to existing behavior
    set receipt fallback = "hnsw_filtered_underreturn_fallback"
```

If determining `filter_scope_has_more_possible_rows` is hard, use a conservative fallback when post-filter hits `< top_k` and filters are active.

## Tests to add

1. Recency score is identical under same `SearchContext.evaluation_time`.
2. Recency score changes predictably with different evaluation times.
3. Search no longer calls wall clock inside scoring helpers.
4. Sparse namespace/source HNSW search falls back when post-filter hits are too low.
5. Receipt records backend, requested/returned/post-filter counts, fallback, exact rerank, result IDs.
6. Receipt mode disabled preserves old API behavior/performance as much as possible.

## Acceptance criteria

- Internal ranking deterministic given context.
- Existing search APIs still work.
- New context-aware API exists.
- Filtered HNSW under-return falls back.
- Receipt data is returned or internally available.
- Tests cover frozen time and filtered fallback.

## Codex prompt

```text
Run Phase 2: deterministic search context and search receipts.

Add a SearchContext or equivalent so recency/ranking uses an explicit evaluation_time captured at the API boundary. Remove internal repeated Utc::now() behavior from scoring helpers. Add a receipt-ready search metadata type that records candidate backend, candidate counts, post-filter counts, fallback/degradation, exact rerank, and result IDs.

Add filtered-HNSW under-return fallback: if HNSW candidates exist but active namespace/source/session filters reduce vector hits below top_k, run exact/brute-force filtered fallback and mark the receipt/degradation.

Do not add TurboQuant in this phase. Keep old APIs working by wrapping them around default SearchContext.

Run search/HNSW tests plus cargo fmt/check. Report exact commands and remaining risks.
```

---
