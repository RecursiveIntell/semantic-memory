# P32 Benchmark and Release Gates

## Benchmark output schema

Every benchmark gate must emit JSON with:

```json
{
  "schema_version": "retrieval_benchmark_summary_v1",
  "run_id": "P32_RESEARCH_MAX_RETRIEVAL_RUNTIME",
  "created_at": "...",
  "profile": {
    "codec_family": "turbo_quant",
    "bits": 8,
    "projections": 64,
    "seed": 0,
    "exact_rerank": true
  },
  "cases": [],
  "classification": "green|yellow|red",
  "debt": []
}
```

Each case must include:

```json
{
  "dim": 384,
  "corpus_size": 10000,
  "query_count": 100,
  "filtered": false,
  "filter_kind": null,
  "recall_at_10": 0.99,
  "ndcg_at_10": 0.99,
  "mean_rank_drift": 0.1,
  "mean_abs_score_error": 0.05,
  "p95_abs_score_error": 0.1,
  "p50_ms": 10.0,
  "p95_ms": 25.0,
  "p99_ms": 40.0,
  "encoded_bytes_per_vector": 1022.0,
  "raw_bytes_per_vector": 1536.0,
  "raw_rows_loaded_p95": 50,
  "fallback_rate": 0.0,
  "degradation_rate": 0.0
}
```

## Gate classes

### Smoke gate

Purpose: cheap CI validation.

- dim: 384
- corpus: 1k
- queries: 50
- filter cases: none
- recall@10 >= 0.99
- ndcg@10 >= 0.99
- encoded bytes/vector < raw bytes/vector
- no corrupt/stale fallback

### Internal gate

Purpose: feature-gated internal readiness.

- dims: 384, 768, 1536
- corpus: 10k
- queries: 100
- filter cases: none + one namespace/source/session case
- recall@10 >= 0.98
- ndcg@10 >= 0.98
- encoded bytes/vector < raw bytes/vector
- p95 latency recorded and non-regressed vs previous accepted run
- raw rows loaded p95 <= exact rerank candidate budget * 1.25

### Release-candidate gate

Purpose: public beta / serious integration.

- dims: 384, 768, 1536
- corpus: 100k
- queries: 500
- filter cases: namespace, source type, session, mixed
- recall@10 >= 0.98
- ndcg@10 >= 0.98
- recall@50 >= 0.995
- fallback rate <= 1% on clean artifact generation
- p99 latency threshold accepted by owner
- memory footprint recorded
- rebuild time recorded
- stale/corrupt artifact scenarios tested

### Default-eligibility gate

Purpose: future gate, not P32.

- multi-corpus real embedding model samples;
- production-like update/delete/rebuild workload;
- no unconditional filter fallback;
- strict release owner thresholds;
- stable benchmark trend across at least three runs;
- no release-blocking workspace debt;
- public claim review complete.

## Regression policy

Any regression from last accepted internal gate must be classified:

- algorithmic regression;
- benchmark noise;
- changed dataset/corpus;
- changed threshold;
- dependency/toolchain drift;
- intentional tradeoff with proof debt.

No regression can be ignored silently.

