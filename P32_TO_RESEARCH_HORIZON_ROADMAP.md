# Research Horizon Roadmap — From P32 to v11+ Runtime

## Current position

P31 made TurboQuant admissible as a feature-gated exact-reranked candidate backend. P32 should make retrieval evidence-grade. After that, the research line points toward a v11A/B/C-shaped runtime:

- v11A: typed artifact runtime with receipts, operator contracts, proof debt, boundary law.
- v11B: regional recursive/subtractive runtime with right-graph declarations, convergence/repair/subtraction law.
- v11C: future-admission hooks for self-hosting, federation, mechanism search, and agency preservation.

## P32 — Evidence-grade retrieval runtime

Goal: make the retrieval subsystem honest, measurable, performant, and receipt-bearing.

Deliverables:

- artifact generation manifests;
- incremental invalidation;
- filter-aware TurboQuant candidate path;
- top-k candidate selection;
- expanded benchmark gates;
- retrieval reference interpreter;
- structured degradation records;
- active run evidence summaries;
- public claim cleanup.

Label if successful:

```text
p32-evidence-grade-retrieval-runtime-rc
```

## P33 — Constitutional artifact microkernel slice

Goal: implement the v11A core on retrieval/build/sidecar operations.

Deliverables:

- `ArtifactEnvelopeV1` active contracts for vector artifacts, HNSW sidecars, search receipts, build receipts.
- `OperatorContractV1` for retrieval operators.
- `OperatorInvocationReceiptV1` for rebuild/search/sidecar operations.
- `DegradationRecordV1` replacing free-text-only degradation.
- `ProofDebtLedgerEntryV1` for unresolved workspace and benchmark debt.
- Schema generation and compatibility gate for retrieval artifacts.
- First production path differentially checked against retrieval reference interpreter.

Label if successful:

```text
v11A-draft-retrieval-surface
```

Do not claim full `v11A-conformant-core` until all declared material operations, not just retrieval, satisfy gates.

## P34 — Right-graph and regional retrieval runtime

Goal: implement v11B-compatible graph-surface discipline around retrieval.

Deliverables:

- `GraphSurfaceDeclarationV1` for storage, retrieval, artifact, receipt, exact reference, and repair graphs.
- `RegionContractV1` for retrieval regions.
- `RegionResultV1` for search/build side effects.
- residual/syndrome records for stale/corrupt/missing artifacts.
- local repair candidates for artifact generation problems.
- support core for search result explanation.
- invalidation cone for embedding updates/deletes.

Label if successful:

```text
v11B-draft-retrieval-region
```

## P35 — Lawful subtraction for retrieval artifacts

Goal: make compaction, artifact deletion, receipt pruning, and history budgeting lawful.

Deliverables:

- `SupportCoreV1` for result/evidence support.
- `RemovalFrontierV1` for safe artifact/log pruning.
- `InvariantPreservationReceiptV1` for compaction.
- `HistoricalLossBudgetV1` for receipt/artifact retention.
- tests proving retired/subtracted artifacts remain queryable within declared history budget.

Label if successful:

```text
subtractive-retrieval-runtime-draft
```

## P36 — Future-admission hooks

Goal: reserve v11C-compatible surfaces without smuggling authority.

Deliverables:

- external vector artifact quarantine;
- attested artifact envelope stubs;
- remote retrieval result admission policy;
- mechanism/theory artifact stubs for retrieval algorithm variants;
- human challenge/veto receipt for generated spec/test changes;
- agency-risk classification for user-facing personalized retrieval/advice.

Label if successful:

```text
v11C-reserved-retrieval-hooks
```

## Strategic warning

Do not jump to P34/P35 before P32 is green. A right-graph regional runtime on top of unresolved artifact-generation and benchmark debt is just a better-organized way to be wrong.

