# Phase 5 - Conformance, benchmarks, CI gates

## Objective

Turn the implementation into a release candidate with falsifiable gates.

## Conformance fixture categories

### HNSW/index integrity fixtures

- missing key
- stale key
- swapped key with equal counts
- wrong domain prefix
- deleted/tombstoned row
- wrong dimension
- unsupported sidecar version
- empty/corrupt sidecar
- pending-op save failure if injectable

### Search determinism fixtures

- frozen evaluation time
- different evaluation time
- filtered under-return fallback
- receipt disabled/enabled modes
- exact rerank flag
- source/namespace/session filters

### Codec fixtures

- raw f32 reference
- SQ8 deterministic digest
- TurboQuant deterministic digest under feature
- profile mismatch
- malformed code bytes
- non-finite vector rejection
- wrong dimension rejection

### Drift/benchmark fixtures

- 1k vectors / 100 queries
- 10k vectors / 500 queries, if practical
- namespace-sparse corpus
- mixed source kinds
- deleted/tombstoned rows
- model/profile mismatch corpus

## Benchmark outputs

Write benchmark outputs to ignored/generated directory, e.g.:

```text
target/giga-pass/benchmarks/*.json
```

Optional checked-in examples can live under:

```text
docs/giga-pass/examples/*.json
```

## Required metrics

```text
recall@1
recall@5
recall@10
NDCG@10 or MRR if practical
mean rank drift
max top-k loss
mean absolute score error
p95 absolute score error
storage bytes per vector
index rebuild time
search latency p50/p95 if practical
receipt overhead bytes
```

## CI gate suggestion

Add or update CI to run:

```bash
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo check --workspace --all-features
cargo test --workspace --all-features
```

If benchmarks are too slow for CI, keep them as manual or nightly.

## Acceptance criteria

- P0 fixtures fail before fix and pass after fix, where applicable.
- Raw/SQ8/TurboQuant paths are differentially testable.
- Drift report exists.
- CI or documented local release bar exists.
- Release gate explicitly says TurboQuant default is blocked unless drift thresholds pass.

## Suggested default thresholds

Do not hardcode these without measuring, but use them as initial release discussion targets:

```text
recall@10 vs raw reference >= 0.90 for 8-bit TurboQuant profile on benchmark corpus
mean rank drift <= 2.0 positions for top-10 overlap
no silent malformed-code scoring
no sidecar corruption can serve HNSW without rebuild/fallback
filtered under-return must equal brute-force top-k under test fixture
```

## Codex prompt

```text
Run Phase 5: conformance and benchmark gates.

Create conformance fixtures for HNSW integrity, deterministic search, codec mismatch, malformed compressed artifacts, filtered fallback, and TurboQuant drift if the feature exists. Add a benchmark/report harness that compares raw reference, HNSW/raw, SQ8, TurboQuant, and TurboQuant+rerank where available.

Produce machine-readable benchmark JSON and a human-readable docs/giga-pass/CONFORMANCE_AND_BENCHMARKS.md summary. Do not force slow benchmarks into normal test runs unless they are small and stable.

Add or update CI/release-bar docs with exact commands. Report commands run, metrics observed, and thresholds that remain tentative.
```

---
