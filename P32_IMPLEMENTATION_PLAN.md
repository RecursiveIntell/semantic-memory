# P32 Implementation Plan

## Dependency order

```text
provenance cleanup
  -> public-claim cleanup
  -> wire canonicality
  -> artifact generation manifest
  -> invalidation/repair
  -> query-time performance rewrite
  -> filter-aware candidate path
  -> receipts/conformance/reference harness
  -> benchmarks/release gates
  -> v11A/v11B draft hooks
  -> workspace debt closure
```

## Step 1 — Repair provenance and package evidence

- Update `CURRENT_RUN.md` to P32.
- Add active P32 baseline/final docs.
- Add evidence summaries that remain in package even if detailed logs archive out.
- Create a `RunEvidenceSummaryV1` JSON shape with:
  - run id;
  - package hash;
  - command list;
  - command status;
  - log path or archived digest;
  - dirty tree status;
  - proof debt.

## Step 2 — Fix public claims

- Edit `turbo-quant/README.md`.
- Edit semantic-memory docs if needed.
- Do not say “zero accuracy loss.”
- Do not say default-ready.
- Keep the exact-rerank story front and center.

## Step 3 — Wire canonicality

Patch `turbo-quant/src/wire.rs`:

```rust
let seed = cursor.read_u64()?;
if seed != profile.seed() {
    return Err(TurboQuantError::MalformedCode {
        reason: format!("wire seed {seed} does not match profile seed {}", profile.seed()),
    });
}
```

After reading packed signs:

```rust
let extra_bits = profile.projections() % 8;
if extra_bits != 0 {
    let mask = !((1u8 << extra_bits) - 1);
    if packed[sign_bytes - 1] & mask != 0 { reject }
}
```

Add tests before changing behavior to prove they fail, then make them pass.

## Step 4 — Generation manifest

Add table:

```sql
CREATE TABLE IF NOT EXISTS derived_vector_artifact_generations (
    generation_id TEXT PRIMARY KEY,
    codec_family TEXT NOT NULL,
    codec_profile_digest TEXT NOT NULL,
    source_snapshot_digest TEXT NOT NULL,
    artifact_manifest_digest TEXT NOT NULL,
    source_row_count INTEGER NOT NULL,
    artifact_count INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    encoding TEXT NOT NULL,
    status TEXT NOT NULL,
    build_receipt_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

Add `generation_id` to `derived_vector_artifacts` or create a linking table. Prefer adding it to the artifact row so search can select current generation cheaply.

## Step 5 — Source snapshot digest

Compute a stable snapshot digest over authoritative rows:

```text
semantic-memory.vector_source_snapshot.v1
for each sorted (item_key, source_embedding_digest, source_type, namespace/session/doc metadata)
```

Do not use query-time raw-row validation for every artifact once a generation is current.

## Step 6 — Invalidation

Add functions:

```rust
mark_derived_artifacts_dirty_for_item(conn, item_key)
delete_derived_vector_artifact(conn, item_key)
current_generation_for_profile(conn, codec_family, profile_digest)
validate_generation(conn, generation_id)
```

Call these from embedding mutation paths.

## Step 7 — Query-time rewrite

Replace:

```text
for each artifact -> load raw row -> recompute source digest -> score -> sort all
```

With:

```text
validate current generation once
load current artifacts with lightweight metadata
score encoded bytes using prepared query
bounded top-k approximate candidates
batch-load raw f32 rows for selected candidates
exact rerank
receipt
```

## Step 8 — Filter support

Minimum P32 acceptable implementation: adaptive oversampling.

Algorithm:

1. Score artifacts by approximate score using heap.
2. Select top `pool_size * widen_factor` candidate keys.
3. Batch-load metadata for candidate keys.
4. Apply SQL-equivalent filters.
5. If post-filter under-return and budget remains, widen candidate pool.
6. If still under-return, fallback or disclose degraded under-return.

Better P32 implementation: materialize filter metadata in derived artifact rows and filter before scoring.

## Step 9 — Reference conformance

Add `retrieval_reference.rs` test helper:

```rust
exact_raw_vector_reference(...)
exact_raw_hybrid_reference(...)
assert_approx_candidate_path(...)
```

Conformance tests:

- exact preference bypasses TurboQuant;
- fallback matches exact reference;
- successful TurboQuant path exact-reranks selected candidates;
- stale artifact cannot influence final result silently;
- filtered path discloses widening/fallback.

## Step 10 — Benchmark harness

Add `examples/retrieval_benchmark_gate.rs` or expand existing benchmark gate.

Output JSON:

```json
{
  "schema_version": "retrieval_benchmark_summary_v1",
  "run_id": "P32",
  "dim": 384,
  "corpus_size": 10000,
  "query_count": 100,
  "mode": "turbo_quant_candidate_then_exact_f32",
  "filtered": false,
  "recall_at_10": 0.99,
  "ndcg_at_10": 0.99,
  "p50_ms": 12.3,
  "p95_ms": 30.2,
  "p99_ms": 41.0,
  "encoded_bytes_per_vector": 1022,
  "raw_bytes_per_vector": 1536,
  "raw_rows_loaded_per_query_p95": 50,
  "fallback_rate": 0.0,
  "classification": "green"
}
```

## Step 11 — v11 hooks

Do not implement full v11. Add draft hooks:

- `RetrievalOperatorContractV1` docs/schema.
- `RetrievalOperatorInvocationReceiptV1` adapter/wrapper.
- `RetrievalGraphSurfaceDeclarationV1` docs/schema.
- `RetrievalDegradationRecordV1` replacing free-text-only degradation.

## Step 12 — Finalization

Final report must answer:

- Is TurboQuant still disabled by default?
- Are public claims cleaned?
- Are filters supported without unconditional fallback?
- Are benchmark gates green, yellow, or red?
- Are workspace gates green or proof-debted?
- Does the final package include active evidence summaries?
- What label is assigned?

