# P31 Final Report — TurboQuant Evidence-Grade Retrieval

## Verdict
- Final label: tq-live-feature-gated-exact-rerank-ready
- Default-ready: no
- Feature-gated live candidate backend: yes
- Exact rerank default: yes
- Benchmark classification: green

## Baseline
- Starting run: P30 reported by the prompt; local codex metadata previously reported P29.
- Starting dirty tree: dirty, 504 `git status --short` lines recorded in `baseline.md`.
- Baseline failures: `cargo test --workspace --all-features` failed in `contract-schema-gen` schema drift; `cargo clippy --workspace --all-targets --all-features -- -D warnings` failed in pre-existing `continuity-runtime` test `expect_used` lint errors.

## Changes landed
- Added `turbo-quant` to the packaged workspace template and generated manifest body.
- Added `scripts/p31_turboquant_gates.sh`.
- Migrated new semantic-memory vector/search/HNSW evidence digest tags from `b3:<hex>` to `blake3:<64-hex>`.
- Added `MemoryError::DigestError`.
- Added public `turbo_quant::wire` with `TURBO_CODE_WIRE_MAGIC = TQW1` and `TurboCodeWireV1`.
- Routed `TurboQuantizer::encode_to_bytes` and `decode_code_from_bytes` through the `TQW1` wire format.
- Added malformed artifact tests and wire-format tests for `turbo-quant`.
- Updated TurboQuant readiness docs and codex run metadata to P31.
- Added a deterministic benchmark gate script and `turboquant_benchmark_gate` example that emit P31 recall/latency/storage JSON.
- Expanded `derived_vector_artifacts` with `encoded_digest`, `encoding`, `dim`, `status`, profile/status indexing, and source digest indexing.
- Expanded search receipts with attempt-family, query-input, budget/deadline, vector-artifact manifest/counts, exact-rerank count, approximate-candidate count, and explicit fallback reason fields.
- Repaired generated schema drift for the `query_turn` verification case class in committed JSON schemas.
- Added test-only clippy allowances and narrow production lint fixes needed for the full workspace P31 gate script to pass.

## Gates
| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --all --check` | pass | final log captured |
| `cargo check --workspace --all-features` | pass | final log captured |
| `cargo test -p turbo-quant` | pass | includes malformed and wire-format tests |
| `cargo test -p semantic-memory --features turbo-quant-codec` | pass | includes HNSW manifest and TurboQuant live-path tests |
| `cargo clippy -p turbo-quant --all-targets --all-features -- -D warnings` | pass | final log captured |
| `cargo clippy -p semantic-memory --all-targets --all-features -- -D warnings` | pass | rerun after test-lint cleanup |
| `scripts/p31_turboquant_gates.sh` | pass | clean rerun captured |
| `scripts/p31_turboquant_benchmark_gate.sh` | pass | emitted green benchmark JSON |

## Behavioral proof
| Test | Result | Evidence |
|---|---:|---|
| Default backend remains non-TurboQuant | pass | existing `prefer_exact_bypasses_candidate_path` and default config tests |
| Enabled with no artifacts falls back | pass | `search_tests::turbo_quant::missing_artifacts_fallback_to_brute_force` |
| Enabled after rebuild uses TurboQuant candidate path | pass | `search_tests::turbo_quant::candidate_path_sets_receipt_profile` |
| Malformed Polar/QJL artifacts rejected | pass | `turbo-quant/tests/malformed_artifacts.rs` |
| Deterministic compact wire roundtrip | pass | `turbo-quant/tests/wire_format.rs` |
| HNSW manifest digest mismatch rebuilds | pass | `tests/hnsw_persistence.rs` |

## Benchmark results
| Dim | Corpus | Recall@10 | NDCG@10 | bytes/vector | p95 score ms | Classification |
|---:|---:|---:|---:|---:|---:|---|
| 384 | 1000 | 1.000 | 1.000 | 1022.0 | 148.129 | green |

## Fallback/degradation behavior
TurboQuant remains feature/config gated and is not default. Existing live path falls back to authoritative raw f32 search when artifacts are missing, incomplete, stale, corrupt, or when filters cannot be enforced before candidate scoring. Receipts record backend, codec family/profile, artifact manifest digest, artifact counts, fallback reason, exact rerank count, approximate candidate count, and degradation notes.

## HNSW manifest behavior
HNSW graph/data sidecars are manifest-bound and digest-checked. Manifest, graph digest, data digest, missing file, and legacy sidecar cases are covered by `tests/hnsw_persistence.rs`.

## Digest policy
New semantic-memory vector profile, vector artifact, query embedding, receipt, source embedding, and HNSW file digests now use `stack_ids::ContentDigest` rendered as `blake3:<64-hex>`. No FNV writer path was found in `semantic-memory/src`.

## Remaining proof debt
- None blocking for the P31 target label. TurboQuant remains experimental, feature-gated, and not default-ready.

## Public claims allowed
- TurboQuant is experimental and feature-gated.
- TurboQuant can be used as a compressed-vector candidate-generation backend.
- TurboQuant results are exact-reranked against authoritative raw f32 embeddings by default.
- Receipts disclose codec profile, fallback, degradation, and exact rerank state.

## Public claims forbidden
- TurboQuant is production-ready.
- TurboQuant is default-ready.
- TurboQuant has zero accuracy loss.
- TurboQuant compresses with no tradeoff.
