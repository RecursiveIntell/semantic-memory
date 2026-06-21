# P32 Test and Conformance Plan

## Gate philosophy

P32 must separate four claims:

1. package/certifier clean;
2. feature-gated candidate backend works;
3. evidence-grade retrieval runtime works;
4. default-ready retrieval works.

Only the first three are in scope. The fourth is explicitly out of scope.

## Required test groups

### A. Wire format conformance

```bash
cargo test -p turbo-quant wire_format
cargo test -p turbo-quant malformed_artifacts
```

New required tests:

- seed mismatch rejected;
- padding-bit mismatch rejected;
- wrong magic rejected;
- wrong version rejected;
- wrong dim rejected;
- wrong projection count rejected;
- wrong bit width rejected;
- trailing bytes rejected;
- payload length mismatch rejected;
- nonzero reserved bytes rejected.

### B. Derived artifact generation

Tests:

- migration creates `derived_vector_artifact_generations`;
- rebuild creates generation manifest;
- rebuild supersedes prior generation;
- generation digest changes after authoritative source change;
- artifact row links to generation;
- empty DB rebuild emits receipt with zero counts;
- corrupt artifact generation fails closed.

### C. Invalidation and repair

Tests:

- fact embedding update invalidates/deletes derived artifact;
- chunk embedding update invalidates/deletes derived artifact;
- message embedding update invalidates/deletes derived artifact;
- episode embedding update invalidates/deletes derived artifact;
- deletion invalidates artifact;
- rebuild repairs dirty state;
- stale artifact cannot be used without receipt degradation.

### D. Search behavior

Tests:

- default config uses exact raw f32;
- TurboQuant config uses candidate generation;
- `PreferExact` bypasses TurboQuant;
- successful TurboQuant receipt names `turbo_quant_candidate_then_exact_f32`;
- fallback uses `turbo_quant_then_brute_force_f32` only when fallback occurs;
- selected candidates exact-rerank against raw f32;
- returned result IDs are deterministic for fixed evaluation time;
- approximate scanned/returned/exact-rerank counts are correct;
- raw rows loaded count is <= selected exact-rerank candidate count plus tolerated overhead.

### E. Filter-aware behavior

Tests:

- namespace filter;
- source type filter;
- session filter;
- combined filters;
- under-return widening;
- budget exhaustion fallback;
- filtered receipt discloses strategy.

### F. Receipt/evidence behavior

Tests:

- query embedding digest stable and dimension-sensitive;
- query/filter digests propagate from `SearchContext`;
- trace/attempt/replay IDs propagate;
- artifact generation ID appears when TurboQuant used;
- degradation record emitted for fallback;
- no done-state without receipt for rebuild/search material operations;
- replay compares against reference digest and result order.

### G. HNSW sidecar behavior

Regression tests:

- manifest digest mismatch rejects and rebuilds;
- manifest graph/data mismatch rejects;
- manifest dimensions mismatch rejects;
- legacy manifestless sidecar emits explicit degradation/warning;
- manifest written last under save path.

### H. Reference interpreter

Fixtures:

- exact vector-only raw f32;
- exact hybrid raw f32 + BM25;
- filtered exact vector-only;
- filtered exact hybrid;
- TurboQuant candidate exact-rerank comparison;
- HNSW candidate exact-rerank comparison;
- stale/corrupt fallback comparison.

### I. Workspace debt gates

Run:

```bash
cargo fmt --all --check
cargo check --workspace --all-targets --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

If failure remains, produce:

```text
ProofDebtLedgerEntryV1
- gate
- failure summary
- owner/scope
- reason for waiver
- risk
- expiry condition
- next action
```

## Conformance labels

| Label | Requirements |
|---|---|
| `p32-shadow-only` | Wire, artifact, receipt, and fallback tests pass; benchmark red/yellow or filters incomplete. |
| `p32-evidence-grade-retrieval-runtime-rc` | All P32 scoped gates pass; workspace debt either fixed or ledgered; internal benchmark gates green. |
| `p32-default-blocked` | Default eligibility not met, but P32 runtime ready. This is expected. |
| `not-ready` | Wire/artifact/search safety gates fail. |

