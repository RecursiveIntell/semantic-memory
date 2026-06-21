# v11 Draft Retrieval Surfaces

Status: `v11A-draft-retrieval-surface`, not v11A or v11B compliance.

## OperatorContractV1 Drafts

- `BuildVectorArtifacts`: rebuilds derived vector artifacts from authoritative f32 rows.
- `ValidateArtifactGeneration`: validates active generation status, counts, dimension, encoding, and manifest digest.
- `TurboQuantCandidateSearch`: scores encoded artifacts and returns approximate candidates.
- `ExactF32Rerank`: loads selected raw f32 rows and reranks exactly.
- `HnswSidecarSave`: persists HNSW sidecar state.
- `HnswSidecarLoad`: loads HNSW sidecar state and records legacy/degraded paths.

## OperatorInvocationReceiptV1 Compatibility Fields

- `receipt_id`
- `evaluation_time`
- `candidate_backend`
- `artifact_generation_id`
- `vector_artifact_manifest_digest`
- `approximate_scanned_count`
- `approximate_returned_count`
- `exact_rerank_count`
- `raw_rows_loaded_count`
- `fallback_reason`
- `degradations`

## DegradationRecordV1 Drafts

Current code stores degradation strings in receipts. P32-compatible degradation IDs are:

- `turbo_quant_generation_missing_or_invalidated`
- `turbo_quant_generation_incomplete_or_stale`
- `turbo_quant_artifact_validation_failed`
- `filter_aware_candidate_under_return`
- `hnsw_legacy_manifest_missing`

Structured object migration remains proof debt.

## GraphSurfaceDeclarationV1 Drafts

- Storage graph: SQLite rows/tables for facts, chunks, messages, episodes, receipts, and artifact generations.
- Retrieval graph: BM25, vector candidate expansion, fusion, and rerank edges.
- Derived artifact graph: codec profile to artifact generation to artifact rows.
- Execution/receipt graph: search/build/sidecar invocation receipts.
- Exact reference graph: raw f32 brute-force oracle.
- Future inference graph: not active.
