# Codex Giga-Pass Master Prompt

## Mission

You are working on the RecursiveIntell semantic-memory stack and preparing it for future fusion with `turbo-quant`.

Your mission is to transform the current codebase into a **local-first AI memory substrate with compressed retrieval readiness and verifiable search/execution receipts**.

This is a **layered, gated stabilization + fusion-readiness pass**. Do not treat this as a normal feature request. Treat it as a release-hardening program.

## Target repos / packages

Primary package roots:

```text
semantic-memory/
semantic-memory-forge/
forge-memory-bridge/
stack-ids/
```

Supporting future integration repo:

```text
turbo-quant/
```

If the workspace contains these roots in one archive, work at the archive workspace root. If working inside a GitHub repo, honor the repo’s existing workspace layout.

## Strategic objective

The final architecture should support this claim:

> semantic-memory is a Rust substrate for local-first AI memory: SQLite + FTS + vector search, episode identity, source-grounded imports, compressed-vector readiness, and receipt-backed retrieval.

TurboQuant must eventually plug in as a **derived vector codec backend**, not as a source of truth.

## Non-negotiable invariants

### Authority

1. SQLite/projected memory rows remain authoritative for queryable memory.
2. Raw evidence/import lanes remain authoritative for promoted truth.
3. HNSW, q8, TurboQuant codes, cached scores, and sidecars are derived acceleration artifacts only.
4. A derived vector/index artifact must be rebuildable from authoritative rows and codec/index profiles.
5. No runtime cache, sidecar, compressed vector, or receipt may become a shadow database.

### Search and vector integrity

6. Every embedding write path must validate dimension and finite values.
7. Every vector/index key must round-trip to a live authoritative row.
8. Count parity is not enough. Use key-level parity.
9. HNSW post-filter under-return must fallback deterministically or disclose degradation.
10. Search ranking must be deterministic when supplied an explicit evaluation time.

### Persistence

11. Never persist `usize` as an on-disk format field.
12. Version sidecars and persistent binary formats.
13. Reject unsupported sidecar/header versions explicitly.
14. Failed HNSW sidecar writes must produce a safe rebuild/degraded state, not silent partial trust.

### Boundary discipline

15. Import/export/bridge/defaulting paths must not silently invent semantic content.
16. `unwrap_or_default` at semantic boundaries must become a typed error, explicit default policy, or degradation record.
17. Lossy transformations must be explicit and testable.
18. Approximate retrieval must be visible in receipts/explanations.

### TurboQuant admission

19. Do not add TurboQuant until P0 vector/index/search integrity gates pass.
20. TurboQuant must be behind a `VectorCodec` abstraction.
21. TurboQuant must have profile identity: dim, bits, projections, seed, codec version, scoring semantics, normalization.
22. TurboQuant score/rank drift must be measured against raw reference scoring.
23. Profile mismatch must fail closed.

### Product/career layer

24. Public-facing docs should say “receipts,” “replay,” “source grounding,” and “trustworthy AI memory,” not lead with academic provenance terms.
25. Every provenance feature must answer a visible question: why this result, which source, which action, approximate or exact, can it be replayed, can it be rebuilt.

## Required execution model

Work in phases. Do not skip a gate.

At the start of every pass:

1. Inspect relevant source files and tests.
2. Run baseline commands if the environment supports them.
3. Record failures before changing code.
4. Make focused changes only.
5. Add tests for every correctness fix.
6. Run the acceptance commands.
7. Produce a final report using `06_FINAL_REPORT_TEMPLATE.md`.

If the environment lacks Rust/Cargo, do not fake success. Report that and still make static-safe code changes when possible.

## Global acceptance commands

Run the strongest available subset:

```bash
cargo fmt --all --check
cargo check --workspace --all-targets --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --all-features --no-deps
```

If some features are unavailable, run narrower equivalents and explain why.

## Layered build order

### Layer 0 — Baseline, ledger, and AGENTS discipline

Create/update a working ledger:

```text
docs/audits/codex-giga-pass-20260511.md
docs/audits/codex-giga-pass-20260511-status.json
```

Record:

- starting commit/hash,
- toolchain versions,
- commands run,
- failures observed,
- pass status,
- P0/P1/P2 closure evidence,
- unresolved/deferred findings.

Also install/update an `AGENTS.md` if missing, based on `05_DROP_IN_AGENTS.md`.

### Layer 1 — semantic-memory vector/index truth hardening

Close P0 integrity risks before feature work:

1. HNSW key-level parity.
2. HNSW fixed-width/versioned sidecar header.
3. HNSW graph/data/keymap validation or rebuild-on-suspicion.
4. HNSW filtered under-return fallback.
5. Pending mutation failure recovery.
6. Central vector validation.
7. Raw brute-force oracle path.

### Layer 2 — deterministic search and minimal receipts

Add deterministic query context and receipt scaffolding:

1. `SearchContext { evaluation_time, receipt_mode, exactness_profile }`.
2. Boundary `Utc::now()` capture once, not inside ranking internals.
3. `VectorSearchReceiptV1` or equivalent internal struct.
4. Fallback/degradation recording.
5. “Why this result?” explainability surface.

### Layer 3 — boundary/defaulting cleanup

Audit import/export/bridge/projection defaulting:

1. Convert semantic-boundary defaulting to typed errors or explicit default policies.
2. Add ugly-input tests.
3. Ensure bridge does not invent meaning.
4. Ensure repair/default behavior emits receipts/degradation where relevant.

### Layer 4 — codec abstraction without TurboQuant

Add the vector codec boundary:

1. `VectorCodecProfileV1`.
2. `VectorArtifactV1`.
3. Object-safe codec trait, preferably byte-oriented for persistence.
4. `RawF32Codec` reference implementation.
5. `Sq8Codec` wrapping existing q8 path.
6. Stable profile digest tests.
7. Profile mismatch rejection tests.

### Layer 5 — TurboQuant optional backend

Only after Layers 1-4 pass:

1. Add optional `turbo-quant` feature/dependency.
2. Implement `TurboQuantCodec`.
3. Store `TurboCode` encoded bytes through vector artifact surface.
4. Add raw-vs-TurboQuant differential harness.
5. Add benchmark fixtures for recall@k, rank drift, score error, latency, storage bytes.
6. Ensure all TurboQuant paths emit receipts/degradations.

### Layer 6 — product-facing receipt UX/API

Expose practical answers:

- why this result appeared,
- which source produced it,
- what vector/index/codec profile was used,
- whether retrieval was approximate,
- whether fallback occurred,
- whether exact rerank was run,
- replay/search receipt ID.

### Layer 7 — docs, release, portfolio story

Update README/docs to frame the work as:

> local-first AI memory with receipts and compressed retrieval readiness.

Do not lead with “provenance runtime.” Lead with visible value.

## Stop conditions

Stop and report instead of continuing if:

- HNSW integrity cannot be proven or rebuilt safely.
- Persistent sidecar format cannot be migrated safely.
- TurboQuant integration requires making compressed vectors authoritative.
- Exact/reference scoring cannot be retained.
- Tests require deleting meaningful functionality.
- The environment cannot run required commands and static review is insufficient to validate the change.

## Required final output

At completion, report:

1. Phases completed.
2. Files changed.
3. Tests added.
4. Commands run with results.
5. P0/P1 issues closed.
6. Remaining risks.
7. Whether TurboQuant integration is now eligible.
8. Next recommended pass.

Use `06_FINAL_REPORT_TEMPLATE.md`.
