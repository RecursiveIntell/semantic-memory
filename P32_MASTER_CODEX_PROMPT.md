# P32 Master Codex Prompt — Research-Max Retrieval Runtime

## Pass identity

```text
P32_RESEARCH_MAX_RETRIEVAL_RUNTIME
```

## Target label

```text
p32-evidence-grade-retrieval-runtime-rc
```

## Explicit non-target labels

Do not claim any of the following at the end of this pass:

```text
turbo-quant-default-ready
v11A-conformant-core
v11B-conformant-runtime
v11plus-release-candidate
production-compression-claim-ready
```

## Current state summary

P31 landed feature-gated TurboQuant candidate generation with exact f32 rerank, BLAKE3-style digests, deterministic wire bytes, derived vector storage, HNSW sidecar manifest, richer receipts, and benchmark scaffolding.

P32 must convert that into an evidence-grade retrieval runtime by hardening:

1. provenance/run metadata;
2. public claim discipline;
3. wire canonicality;
4. query-time performance shape;
5. artifact generation manifests and incremental invalidation;
6. filter-aware candidate generation;
7. conformance fixtures;
8. benchmark/release thresholds;
9. v11A-compatible operator/receipt hooks;
10. v11B-compatible graph/view disclosure hooks.

## Non-negotiable laws

- Raw SQLite f32 embeddings remain authoritative.
- TurboQuant artifacts are derived acceleration artifacts only.
- TurboQuant remains feature-gated and disabled by default.
- Exact f32 rerank remains mandatory for TurboQuant candidates in P32.
- Any fallback/degradation must be recorded in receipts.
- No material operation may emit a done state without receipts or explicit proof debt.
- Do not hide baseline workspace failures; classify proof debt explicitly.
- Do not delete or rewrite historical P31 artifacts. Reclassify, summarize, or index them.

## Required outputs

Create or update:

```text
semantic-memory/docs/codex-runs/CURRENT_RUN.md
semantic-memory/docs/codex-runs/P32_RESEARCH_MAX_RETRIEVAL_BASELINE.md
semantic-memory/docs/codex-runs/P32_RESEARCH_MAX_RETRIEVAL_FINAL.md
semantic-memory/docs/audits/p32-research-max-retrieval-runtime-20260513/
semantic-memory/docs/RETRIEVAL_RUNTIME_READINESS.md
semantic-memory/docs/TURBOQUANT_READINESS.md
semantic-memory/docs/VECTOR_ARTIFACT_GENERATION.md
semantic-memory/docs/RETRIEVAL_BENCHMARK_GATES.md
semantic-memory/scripts/p32_retrieval_runtime_gates.sh
semantic-memory/scripts/p32_retrieval_benchmark_gate.sh
```

Add active evidence summaries even if detailed logs are archived:

```text
semantic-memory/docs/audits/p32-research-max-retrieval-runtime-20260513/RUN_EVIDENCE_INDEX.md
semantic-memory/docs/audits/p32-research-max-retrieval-runtime-20260513/RUN_EVIDENCE_SUMMARY.json
```

## Phase 0 — Baseline and provenance repair

1. Set active run to P32.
2. Record `git status --short`, `git diff --stat`, and dirty-tree scope.
3. Run baseline gates and classify failures.
4. Do not mark P31 material as active instruction.
5. Keep P31 evidence reachable via digest-indexed archive pointers.
6. Ensure P31 artifacts are not unclassified if the run marker can determine P31.

Acceptance:

- `CURRENT_RUN.md` says P32.
- `CODEX_RUN_INDEX.md` includes P31 archive manifest and P32 active evidence summary.
- Final report explains whether tree was dirty at start/end.

## Phase 1 — Public claim cleanup

Fix all public overclaims in TurboQuant and semantic-memory docs.

Required replacements:

- Remove or qualify “zero accuracy loss”.
- Remove “no dataset-specific calibration” as unconditional claim.
- Replace “ICLR 2026 / AISTATS 2026 / AAAI 2025” certainty with either verified citation context or neutral algorithm-family language.
- State that P32 is feature-gated candidate-generation only.
- State default-readiness criteria separately.

Acceptance:

```bash
grep -RIn "zero accuracy loss\|default-ready\|production-ready" turbo-quant semantic-memory/docs semantic-memory/README.md
```

must return only explicitly negated/safe statements.

## Phase 2 — TurboCodeWireV1 canonicality

Harden `turbo-quant/src/wire.rs`.

Required changes:

1. Validate wire seed equals profile seed.
2. Reject non-zero padding bits in the final QJL sign byte when projections % 8 != 0.
3. Validate payload length before large allocation or field expansion where practical.
4. Keep trailing-byte rejection.
5. Add malformed-wire tests:
   - seed mismatch rejected;
   - QJL padding bits rejected;
   - payload length too small/large rejected;
   - reserved bytes rejected;
   - wrong dimension/bit/projection rejected;
   - wrong magic/version rejected.

Acceptance:

```bash
cargo test -p turbo-quant wire_format malformed_artifacts
```

## Phase 3 — Artifact generation manifest

Add a stable generation-level manifest for derived vector artifacts.

Preferred logical contract:

```rust
DerivedVectorArtifactGenerationV1 {
  schema_version: "derived_vector_artifact_generation_v1",
  generation_id: String,
  codec_family: String,
  codec_profile_digest: String,
  source_snapshot_digest: String,
  source_row_count: usize,
  artifact_count: usize,
  source_tables: Vec<String>,
  dim: usize,
  encoding: String,
  created_at: DateTime<Utc>,
  build_receipt_id: Option<String>,
  artifact_manifest_digest: String,
  status: active|superseded|invalidated|failed,
  degradations: Vec<String>
}
```

Persist as either a new table:

```sql
CREATE TABLE IF NOT EXISTS derived_vector_artifact_generations (...)
```

or as a compact schema-compatible JSON manifest table.

Acceptance:

- Rebuild creates one generation row.
- Rebuild supersedes old generation for same codec/profile.
- Search receipt records generation id and manifest digest.
- Query path can validate generation-level completeness without loading raw rows for every artifact.

## Phase 4 — Incremental invalidation and repair

Do not require full manual rebuild after every write forever.

Add invalidation paths for writes/re-embeds/deletes:

- `fact` embedding insert/update/delete invalidates its derived artifact.
- `chunk` embedding insert/update/delete invalidates its derived artifact.
- `message` embedding insert/update/delete invalidates its derived artifact.
- `episode` embedding insert/update/delete invalidates its derived artifact.

P32 may implement eager deletion, dirty status, or invalidation queue. It must be explicit.

Acceptance:

- After updating an authoritative embedding, stale derived artifact is not silently used.
- Receipt records stale/invalidation state.
- Rebuild repairs dirty artifacts and emits a build receipt.

## Phase 5 — Query-time performance shape

Remove O(n) raw-row validation from the TurboQuant candidate loop when a current artifact generation is valid.

Required changes:

1. Validate generation manifest once.
2. Score from encoded bytes without loading raw rows for all artifacts.
3. Use a bounded top-k heap or partial selection instead of full sort for approximate candidate selection.
4. Load raw f32 rows only for selected approximate candidates to exact-rerank.
5. Record:
   - `approximate_scanned_count`;
   - `approximate_returned_count`;
   - `exact_rerank_count`;
   - `raw_rows_loaded_count`;
   - `artifact_generation_id`.

Acceptance:

- 10k/1536 candidate generation improves materially from P31 local reference (~187 ms).
- Receipts disclose scanned vs returned vs exact-reranked counts.

## Phase 6 — Filter-aware TurboQuant path

P31 falls back when SQL filters are active. P32 should support at least one safe filter-aware mode.

Option A: store filter metadata with derived artifacts:

```text
item_key, source_type, namespace, session_id, document_id, created_at/updated_at
```

Then filter before scoring or during candidate selection.

Option B: adaptive oversampling:

- score approximate candidates;
- apply filters via batch metadata load;
- if under-return, widen approximate candidate pool up to budget;
- fallback only if budget exhausted.

Option C: hybrid mode:

- use exact filtered candidate ID set from SQL;
- score only matching derived artifacts.

Acceptance:

- SQL filters do not automatically force full brute-force fallback.
- Receipts disclose filter strategy and widening/budget behavior.
- Under-return emits degradation.

## Phase 7 — Retrieval reference interpreter

Define an exact reference evaluator for retrieval conformance.

Minimum surfaces:

- vector-only exact raw f32;
- hybrid BM25 + exact raw f32;
- filtered vector-only exact raw f32;
- filtered hybrid exact raw f32;
- HNSW candidate + exact rerank comparison;
- TurboQuant candidate + exact rerank comparison;
- stale/corrupt artifact fallback comparison.

Acceptance:

- Tests assert TurboQuant exact-reranked final ordering against reference within accepted candidate-recall thresholds.
- Exactness profile `PreferExact` always bypasses derived candidates.
- Degraded answers disclose degradation.

## Phase 8 — Benchmark gate expansion

Create benchmark classes.

### Smoke gate

- dim 384
- corpus 1k
- queries 50
- recall@10 >= 0.99
- ndcg@10 >= 0.99
- encoded bytes/vector < raw bytes/vector

### Internal gate

- dims 384, 768, 1536
- corpus 10k
- queries >= 100
- recall@10 >= 0.98
- ndcg@10 >= 0.98
- p95 candidate latency budget recorded
- exact rerank count recorded
- encoded bytes/vector < raw bytes/vector

### Release-candidate gate

- dims 384, 768, 1536
- corpus 100k
- filtered and unfiltered queries
- stale/corrupt artifact scenarios
- rebuild time
- memory footprint
- p50/p95/p99 latency
- exact baseline comparison
- regression threshold from previous accepted run

### Default-eligibility gate

Must be defined but should not be required to pass in P32.

## Phase 9 — v11A-compatible operator receipts

Add logical contracts or stubs for:

- `OperatorContractV1` for `BuildVectorArtifacts`, `ValidateArtifactGeneration`, `TurboQuantCandidateSearch`, `ExactF32Rerank`, `HnswSidecarSave`, `HnswSidecarLoad`.
- `OperatorInvocationReceiptV1`-compatible wrappers for rebuild/search/sidecar operations.
- `DegradationRecordV1`-compatible structured degradation objects rather than free-text only.
- `ProofDebtLedgerEntryV1` for workspace failures that remain out of scope.

Do not claim v11A compliance. Claim `v11A-draft-retrieval-surface` if gates pass.

## Phase 10 — v11B-compatible right-graph declarations

Retrieval now has multiple graph surfaces. Declare them explicitly:

- storage graph: SQLite rows/tables;
- retrieval graph: candidate expansion;
- derived artifact graph: codec profile + artifact generation;
- execution/receipt graph: search/build/sidecar receipts;
- exact reference graph: raw f32 oracle;
- future inference graph: not active.

Add `GraphSurfaceDeclarationV1`-compatible stubs or docs for retrieval paths.

## Phase 11 — Workspace debt burn-down

P32 should attempt to clear or explicitly quarantine:

- `contract-schema-gen` schema drift;
- workspace clippy `expect_used` debt;
- focused semantic-memory clippy test warnings;
- stale/unclassified run artifact issues.

If not fixed, emit proof debt entries with owner, scope, risk, and waiver conditions.

## Final status labels

One of:

```text
p32-evidence-grade-retrieval-runtime-rc
p32-retrieval-runtime-shadow-only
p32-retrieval-runtime-not-ready
```

Do not use `turbo-quant-default-ready`.

## Drop-in command

```text
Execute P32_RESEARCH_MAX_RETRIEVAL_RUNTIME.

Start from the latest semantic-memory package. Preserve raw SQLite f32 embeddings as authoritative. Keep TurboQuant disabled by default and exact-reranked. Repair run provenance, remove public overclaims, harden TurboCodeWireV1 canonicality, add derived artifact generation manifests, implement incremental invalidation/repair, remove per-query raw-row validation from the full candidate loop, replace full-corpus sort with top-k selection, add filter-aware TurboQuant candidate generation, define exact reference retrieval conformance, expand benchmark gates, add v11A operator/receipt hooks and v11B graph-surface declarations, and either fix or explicitly ledger workspace gate debt. Produce active evidence summaries and final status.
```

