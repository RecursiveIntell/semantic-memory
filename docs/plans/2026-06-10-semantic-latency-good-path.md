# Semantic Memory Latency Good-Path Implementation Plan

> For Hermes: implement directly with strict TDD where behavior changes are testable. Do not pursue low-ROI GPU/custom-IVF/PQ work in this pass.

Goal: reduce real semantic-memory retrieval latency on the recommended paths while preserving usearch as the primary latency backend and proveKV/TurboQuant as exact-reranked derived candidate paths.

Architecture: Keep usearch as the hot ANN path. Improve derived-backend overhead by caching generation search state, using flat/norm-aware scoring for decoded proveKV vectors, batching exact-rerank row loads, and adding a contained usearch scalar-kind spike/benchmark knob. Defer proveKV compressed-space scoring to a separate research pass after the measurable systems wins land.

Tech Stack: Rust, rusqlite, usearch 2.25, semantic-memory derived vector generations, existing cargo tests/benchmarks.

Repository: /home/sikmindz/Coding/Libraries/semantic-memory
Plan date: 2026-06-10 UTC

Current measured baseline:
- Release command: `cargo run --release --example provekv_vs_usearch_benchmark --features poly-kv-pool results/provekv_vs_usearch_benchmark_release.json`
- proveKV warm total p50: 0.560 ms; p95: 0.590 ms
- proveKV cold decode_once: 4686 ms
- proveKV compression ratio vs f32: 23.99x
- usearch p50: 0.103 ms; p95: 0.132 ms
- exact f32 full scan p50: 0.509 ms; p95: 0.531 ms

Claim boundary:
- Certified now: usearch is the fastest measured current backend on this harness.
- Candidate improvements: generation cache, flat scoring, batch rerank, usearch scalar-kind configurability.
- Non-claims: proveKV does not become a latency winner until compressed-space scoring exists and beats usearch under measured gates.

Out of scope / killed for this pass:
- GPU search.
- hand-rolled IVF/PQ.
- replacing usearch with proveKV decode-then-scan.
- large fib-quant internal rewrite.

---

## Phase 1: Benchmark and config foundation

### Task 1.1: Parameterize proveKV/usearch benchmark

Objective: Make the benchmark useful for latency work beyond the fixed 2k-vector fixture.

Files:
- Modify: `examples/provekv_vs_usearch_benchmark.rs`
- Test: run cargo example with small and default args.

Steps:
1. Add optional args/env for corpus size, query count, k, candidate multiplier.
2. Keep defaults identical to current behavior: corpus=2000, queries=100, k=10, multiplier=20.
3. Verify current default still writes the same schema.
4. Run:
   - `cargo run --release --example provekv_vs_usearch_benchmark --features poly-kv-pool,usearch-backend results/provekv_vs_usearch_benchmark_release.json`
   - `SEMANTIC_MEMORY_BENCH_CORPUS=256 SEMANTIC_MEMORY_BENCH_QUERIES=8 cargo run --release --example provekv_vs_usearch_benchmark --features poly-kv-pool,usearch-backend results/provekv_vs_usearch_benchmark_small.json`

### Task 1.2: Add test coverage for usearch scalar-kind parser

Objective: Support a safe benchmark/config knob without changing default F32 behavior.

Files:
- Modify: `src/config.rs` or `src/vector_backend.rs` if public config owns this.
- Modify: `src/usearch_backend.rs`
- Test: `src/usearch_backend.rs` unit tests.

Steps:
1. Write failing tests for parsing `f32`, `f16`, and `f8`, plus rejecting unknown values.
2. Implement parser and keep default as F32.
3. Wire parser to env/config only if usearch crate exposes compatible `ScalarKind` variants.
4. Run focused tests.

Gate:
- If usearch F16/F8 variants are absent in this crate version, document and keep F32-only with a failing-spike avoided. Do not invent unsupported variants.

---

## Phase 2: Derived backend hot-path cache and flat scoring

### Task 2.1: Add a proveKV decoded generation search cache type

Objective: Cache more than decoded `Vec<Vec<f32>>` so queries avoid repeated item-key construction and can use flat/normed scoring.

Files:
- Modify: `src/provekv_pool.rs`
- Modify: `src/search.rs`
- Test: `tests/search_tests.rs` or `tests/pool_generation_types.rs`

Cache shape:
- `generation_cache_key: String`
- `vectors_flat: Vec<f32>`
- `dim: usize`
- `row_count: usize`
- `row_norms: Vec<f32>`
- `item_keys: Vec<String>`

Steps:
1. Write failing unit test that builds a tiny compact pool and verifies the cached search matrix has expected row_count, dim, item_keys length, and finite norms.
2. Implement `ProveKvDecodedSearchCache` creation from payload + item map.
3. Keep existing `decode_compact_pool_payload` API for compatibility.
4. Add cache lookup by generation/manifest key.
5. Run focused tests.

### Task 2.2: Replace proveKV candidate cosine loop with flat/norm-aware dot scoring

Objective: Remove per-row norm recomputation and Vec<Vec> traversal from warm proveKV candidate scoring.

Files:
- Modify: `src/search.rs`
- Test: `tests/search_tests.rs`

Steps:
1. Write failing parity test: proveKV flat scorer returns the same top candidate ordering as existing cosine scorer on normalized and non-normalized vectors.
2. Implement helper:
   - compute query norm once
   - use row_norms
   - dot over contiguous row slice
   - preserve existing non-finite behavior during cache construction
3. Use bounded heap top-k as current production path does.
4. Run focused tests.

Gate:
- Candidate order must remain deterministic on ties using existing sequence ordering.

---

## Phase 3: Batch exact-rerank row loading

### Task 3.1: Add batch vector-row loader by item key

Objective: Replace one SQL query per candidate with grouped source-table batch loads.

Files:
- Modify: `src/search.rs`
- Test: `tests/search_tests.rs`

Steps:
1. Write failing test that creates fact/chunk/message/episode/projection vector rows, asks for a mixed item-key batch, and verifies all rows are returned by key.
2. Implement `load_vector_rows_by_item_keys(conn, &[String]) -> HashMap<String, VectorRow>`.
3. Internally group by source prefix: `fact:`, `chunk:`, `msg:`, `episode:`, `projection:`.
4. Use SQLite `IN` placeholders or prepared per-key fallback if batch SQL is awkward; correctness first.
5. Preserve source/content/blob/updated_at fields exactly.

### Task 3.2: Use batch loader in proveKV and TurboQuant exact rerank

Objective: Remove per-candidate exact-row queries from derived backends.

Files:
- Modify: `src/search.rs`
- Test: existing derived backend tests.

Steps:
1. Write/adjust test that ensures missing candidates still increment missing_count and candidate order/ranks are preserved.
2. Replace loop-local `load_vector_row_by_item_key` calls with one batch load before rerank loop.
3. Preserve metadata:
   - `raw_rows_loaded_count`
   - `missing_count`
   - `exact_rerank_count`
4. Run focused tests.

---

## Phase 4: usearch scalar-kind benchmark spike

### Task 4.1: Wire scalar kind to benchmark-only path if supported

Objective: Allow release benchmark to compare F32/F16/F8 without changing production default.

Files:
- Modify: `src/usearch_backend.rs` if a public config path exists.
- Modify: `examples/provekv_vs_usearch_benchmark.rs` or create `examples/usearch_scalar_kind_benchmark.rs`.
- Test: unit parser tests and cargo check with usearch feature.

Steps:
1. Inspect `usearch::ffi::ScalarKind` actual variants.
2. If F16/F8 exist, add benchmark env var `SEMANTIC_MEMORY_USEARCH_SCALAR_KIND`.
3. Include scalar kind in benchmark JSON.
4. If not supported by current crate, record in benchmark report and keep F32.

Gate:
- Production default remains F32.
- Sidecar manifest must reject incompatible scalar-kind loads if production wiring stores non-F32 sidecars.

---

## Phase 5: Verification and evidence

### Task 5.1: Run focused correctness gates

Commands:
- `cargo fmt --all -- --check`
- `cargo test --features poly-kv-pool,usearch-backend search_tests -- --nocapture`
- `cargo test --features poly-kv-pool,usearch-backend pool_generation_types -- --nocapture`
- `cargo check --features poly-kv-pool,usearch-backend --examples`

### Task 5.2: Run latency evidence gate

Commands:
- `cargo run --release --example provekv_vs_usearch_benchmark --features poly-kv-pool,usearch-backend results/provekv_vs_usearch_benchmark_release.json`
- small fast check with env-overrides if implemented.

Acceptance:
- No correctness regression.
- usearch remains default latency answer.
- proveKV warm path should not regress; target is measurable reduction from ~0.560ms p50 if flat cache lands.
- If p50 does not improve, report honestly and keep the architectural cleanup only if tests prove it simplifies future compressed-space scoring.

---

## Deferred follow-up plan

After this pass, implement the true proveKV research path separately:
1. Fib/proveKV compressed-space dot scoring API.
2. Differential test against decoded-vector scorer.
3. Exact rerank unchanged.
4. Benchmark against usearch at increasing corpus sizes.

Do not claim proveKV latency win until this exists and wins measured gates.
