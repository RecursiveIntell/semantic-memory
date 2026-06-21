# P31 TurboQuant Readiness Final Report

## Status label

`tq-live-feature-gated-exact-rerank-ready`

## Summary

- What landed: workspace TurboQuant package visibility, BLAKE3 codec/search/receipt digests, encoded artifact digest validation, binary TurboQuant wire bytes, hostile-code validation, prepared query scoring, receipt evidence fields, derived artifact storage/rebuild, optional TurboQuant candidate retrieval with exact f32 rerank, HNSW sidecar manifests, and deterministic P31 benchmark gates.
- What did not land: default TurboQuant retrieval.
- Why default TurboQuant is still blocked: the live path is feature-gated, SQL-filtered TurboQuant search currently degrades to exact f32, and P31 classifies TurboQuant only as an experimental candidate-generation backend.

## Baseline

See `P31_TURBOQUANT_READINESS_BASELINE.md`.

## Changes

### TurboQuant core

- Added `validate_for` methods for `PolarCode`, `QjlSketch`, and `TurboCode`.
- Rejected non-finite input vectors, malformed radii, out-of-range angles, bad QJL signs, and bad QJL norms.
- Added deterministic binary wire helpers:
  - `encode_to_bytes`
  - `decode_code_from_bytes`
  - `score_inner_product_from_bytes`
- Added prepared query scoring for Polar, QJL, and TurboQuant.
- Removed public "zero accuracy loss" wording in TurboQuant crate docs.

### semantic-memory vector codec

- Replaced FNV-style profile/artifact digests with `stack-ids::ContentDigest` BLAKE3 digests rendered as `blake3:<64-hex>`.
- Added persisted `artifact_digest` to `VectorArtifactV1` with serde default compatibility.
- Validated encoded artifact digest before decode/score.
- Switched `TurboQuantCodec` from JSON artifacts to binary TurboQuant wire bytes.

### Receipts and Context

- Replaced search receipt row digest with BLAKE3 over stored receipt JSON.
- Replaced query embedding digest with BLAKE3 over a domain-separated dimension plus little-endian f32 bytes.
- Added backwards-compatible search context and receipt hook fields for trace, attempt family, attempt, replay, query/filter digests, redaction state, budget/deadline, artifact manifest digest, artifact counters, approximate candidate count, exact rerank count, and fallback reason.

### Derived artifact storage

- Added migration V19 for `derived_vector_artifacts`.
- Added rebuild support via `MemoryStore::rebuild_vector_artifacts()` under `turbo-quant-codec`.
- Added `VectorArtifactBuildReceiptV1` with counts, profile digest, elapsed time, and degradation notes.
- Stored source embedding digests with domain-separated BLAKE3 over authoritative raw f32 blobs.

### Search backend

- Added `DerivedVectorBackendPolicy::{Disabled, TurboQuantCandidateOnly}`.
- Default remains `Disabled`.
- TurboQuant candidate generation uses stored derived artifacts only when the configured profile is complete and valid.
- Final vector order uses exact raw f32 cosine by default.
- Missing, incomplete, stale, corrupt, or filtered artifact paths degrade to brute-force raw f32 and record fallback/degradation in receipts.

### HNSW manifest

- Added `memory.hnsw.manifest.json` with schema version, generation ID, basename, graph/data file names, BLAKE3 digests, dimensions, vector count, sidecar format version, source epoch, and created timestamp.
- HNSW save now writes graph/data temp files, computes digests, renames graph/data, writes and fsyncs manifest temp, then renames the manifest last and fsyncs the directory.
- HNSW load validates manifest digests before loading graph/data and rebuilds from SQLite on mismatch.
- Legacy graph/data sidecars without a manifest still load deterministically.

### tests/benchmarks

- Added Criterion bench `turbo_quant_search`.
- Bench modes cover encode throughput, raw dot/cosine-style scoring vs TurboQuant unprepared vs prepared scoring, and candidate generation over 1k/10k corpora at dimensions 32, 384, 768, and 1536.
- Existing deterministic quality harness reports recall@k, rank drift, score error, and storage bytes/vector smoke metrics.

### Workspace and Docs

- Added `turbo-quant` to workspace members; default members remain scoped to `semantic-memory`.
- Added `docs/TURBOQUANT_READINESS.md`.

## Migrations

Added V19/V20:

```sql
CREATE TABLE IF NOT EXISTS derived_vector_artifacts (
    item_key                TEXT NOT NULL,
    codec_family            TEXT NOT NULL,
    codec_profile_digest    TEXT NOT NULL,
    source_embedding_digest TEXT NOT NULL,
    encoded_digest          TEXT NOT NULL,
    artifact_digest         TEXT NOT NULL,
    encoding                TEXT NOT NULL,
    dim                     INTEGER NOT NULL,
    encoded                 BLOB NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    status                  TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (item_key, codec_family, codec_profile_digest)
);

CREATE INDEX IF NOT EXISTS idx_derived_vector_artifacts_profile
ON derived_vector_artifacts(codec_family, codec_profile_digest, status);

CREATE INDEX IF NOT EXISTS idx_derived_vector_artifacts_source_digest
ON derived_vector_artifacts(source_embedding_digest);
```

Backward compatibility: raw f32 embeddings remain authoritative; the table is rebuildable acceleration state and can be empty on existing DBs.

## Gates run

Passed:

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features
cargo test -p turbo-quant
cargo check -p semantic-memory --features turbo-quant-codec --all-targets
cargo test -p semantic-memory --features turbo-quant-codec --test vector_codec -- --nocapture
cargo test -p semantic-memory --features hnsw,turbo-quant-codec
cargo test -p semantic-memory --features hnsw,turbo-quant-codec --test vector_codec -- --nocapture
cargo test -p semantic-memory --features turbo-quant-codec search_tests::turbo_quant -- --nocapture
cargo test -p semantic-memory --features turbo-quant-codec derived_vector -- --nocapture
cargo test -p semantic-memory --features turbo-quant-codec migration -- --nocapture
cargo test -p semantic-memory --features testing,turbo-quant-codec search_tests::turbo_quant::corrupt -- --nocapture
cargo test -p semantic-memory --features hnsw --test hnsw_persistence -- --nocapture
cargo test -p semantic-memory --features hnsw --test hnsw_hotswap -- --nocapture
cargo test -p semantic-memory digest_tests::query_embedding_digest_includes_dimension_and_bytes
cargo test -p semantic-memory --features testing context_search_receipt_records_exact_backend_and_result_ids
cargo test -p semantic-memory --features testing durable_receipt_id_conflict_fails_closed
cargo test -p turbo-quant prepared
cargo check -p turbo-quant --benches
cargo bench -p turbo-quant --bench turbo_quant_search -- --sample-size 10
cargo check --workspace --all-targets
```

Known failures:

```bash
cargo test --workspace
```

Fails in `contract-schema-gen` on schema drift for
`schemas/verification-case-v1.schema.json`.

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Fails on existing non-scoped warnings, mainly `expect_used` in other crates.
Focused semantic-memory clippy with `-D warnings` now gets past the TurboQuant
wire-format lint but still fails on pre-existing `expect_used` patterns in
semantic-memory integration tests outside this pass scope.

## Benchmark/quality metrics

Smoke codec harness:

- dimensions: 32
- corpus size: 48
- recall@5: 1.000
- mean rank drift: 0.792
- mean absolute score error: 0.076
- p95 score error: 0.152
- bytes/vector: 124.0

Criterion local run:

- dimensions: 32, 384, 768, 1536
- corpus sizes: 1k and 10k
- encode throughput examples:
  - dim 32, 8 bits, 4 projections: ~3.31 us/vector
  - dim 768, 8 bits, 96 projections: ~1.64 ms/vector
  - dim 1536, 8 bits, 192 projections: ~8.44 ms/vector
- prepared scoring examples:
  - dim 32: ~227 ns/candidate
  - dim 384: ~2.35 us/candidate
  - dim 768: ~4.66 us/candidate
  - dim 1536: ~9.35 us/candidate
- 10k candidate generation:
  - dim 32: ~5.28 ms
  - dim 384: ~50.08 ms
  - dim 768: ~98.62 ms
  - dim 1536: ~186.88 ms
- p50/p95 latency: not separately measured beyond Criterion timing summaries.

## Remaining blockers

- Default retrieval blockers: TurboQuant remains feature-gated, SQL-filtered searches currently fall back to raw f32, and benchmark gates are not sufficient for default enablement.
- Product/public claim blockers: benchmark harness exists, but no release-owner accepted thresholds.
- Workspace blockers: pre-existing `contract-schema-gen` schema drift and workspace `-D warnings` clippy blockers.

## Operator notes

The tree was already dirty at start. I avoided reverting unrelated changes. The
`~/Documents/turbo-quant` copy differs from the workspace dependency copy; this
pass changed the dependency copy at `../turbo-quant`.
