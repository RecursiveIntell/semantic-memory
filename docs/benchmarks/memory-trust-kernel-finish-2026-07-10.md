# Memory trust-kernel finish handoff — 2026-07-10

## Claim boundary

This pass closes the two remaining verified benchmark gaps: current/historical supersession
search validity and authoritative rejection of unsupported model fact admission. It also retains
the previously shipped canonical-contract and SQLite authority work described below.

## Shipped

- The hostile benchmark uses real `MemoryStore::list_facts_with_view` calls for `Current` and
  `HistoricalAt`, and a real bitemporal `add_graph_edge_at` supersession edge.
- Current-state filtering accepts both the canonical externally-tagged `GraphEdgeType::Entity`
  serialization and the legacy flat relation JSON. This repaired the public-API supersession test.
- The hostile benchmark threshold is strict: 100% tested-scenario pass rate, zero stale retrievals,
  unsupported admissions, and namespace leakage, plus replay equivalence.
- The hostile benchmark now exercises `MemoryAuthority::append` with an append-capable model permit
  that carries neither evidence nor operator/system admission. The operation is rejected and no
  fact is persisted. The semantic-memory receipt records 9/9 and `thresholds_met=true`.
- `AuthorityPermit` now carries an explicit admission basis: non-empty evidence, the clearly named
  `operator_system` permit, or unspecified. Authoritative append rejects unspecified/empty evidence.
- Ordinary hybrid, FTS, and vector fact search default to `StateView::Current`. Search cache entries
  are cleared by fact append, graph-edge mutation, and authority mutation so supersession cannot
  return a cached stale head.
- `MemoryStore::search_with_view` provides explicit historical retrieval. `HistoricalAt` excludes
  replacement heads whose supersession edge becomes valid after the cutoff, including same-second
  fact inserts, while retaining the old head.
- Current/historical filtering recognizes canonical externally tagged and legacy flat entity-edge
  serialization. Direct MCP `sm_search`, MCP `sm_search_as_of`, and HTTP `/search` are regression tested.
- Benchmark artifacts are written below the semantic-memory crate regardless of invocation cwd.
- The MCP `search` Cargo profile now composes every capability referenced by its unified tool router.
  Documentation no longer describes this as a minimal surface.
- Added canonical versioned contract types: `MemoryEnvelopeV1`, `CapabilityManifestV1`,
  `RetrievalResponseV1`, `RetrievalWitnessV1`, `StageOutcomeV1`, `InjectionDecisionV1`,
  `InjectionDisposition`, `SupersessionReceiptV1`, `AuthoritySnapshotId`, and `RetrievalEpoch`.
- Added bounded `Probability`, `Confidence`, `CosineSimilarity`, and `NonNegativeWeight` newtypes.
  Constructors and serde reject non-finite and out-of-range values; boundary round trips are tested.
- Added `MemoryStore::authority()` with capability-gated append, supersede, and redact operations.
  Facts and FTS rows remain append-only; supersession/redaction edges and the exactly-one active
  head are persisted in the existing SQLite graph alongside authority lineage versions.
- Added idempotent V29 initialization for `authority_state`, `authority_lineages`,
  `authority_versions`, `operation_journal`, and `authority_receipts`. Each mutation commits the
  fact/version, lineage edge/head transition, retrieval epoch, journal row, and receipt in one
  SQLite transaction. The operation UUID is separate from the content digest.
- Added receipt lookup by operation ID and caller idempotency key. Same-payload retries return the
  stored receipt bytes and do not advance the epoch; conflicting payloads fail closed.
- Added typed test-only fault gates before/after append, lineage, journal, epoch, and receipt
  stages. The integration suite exercises all ten gates and verifies complete rollback.
- The backend invariant remains intact. The supported minimal semantic-memory profile is
  `--no-default-features --features brute-force`; bare no-feature compilation is not claimed.

## Remaining honest gaps — not implemented or claimed

- The authority lane is not yet the mandatory write path for legacy `MemoryStore::add_fact`, direct
  graph-edge APIs, projection import, or administrative mutation APIs.
- Redaction is represented by an append-only `[REDACTED]` tombstone and a redaction lineage edge;
  historical rows remain in SQLite for audit and are filtered from the current fact view.
- Receipt recovery after a process loses its response is covered by durable lookup, but there is
  no distributed permit issuer or cross-process authority service.
- Cross-host permit issuance and injection conformance remain outside this bounded local authority lane.
- End-to-end adapter round-trip conformance against a live authority snapshot/epoch provider; the
  current semantic-memory API does not expose those values, so the MCP response truthfully returns
  them as unavailable with explicit degraded stage outcomes.
- Phase 6 research and any universal provenance, safety, bitemporality, or superiority claim.

## Receipts

Passed during this closure pass:

- `cargo test -p semantic-memory --all-features` — passed.
- `cargo test --manifest-path semantic-memory-mcp/Cargo.toml` — passed: 2 unit tests and 19 integration tests.
- `cargo run -p semantic-memory --example hostile_memory_integrity --quiet` — semantic-memory 9/9,
  strict thresholds met, zero unsupported admissions and zero stale retrievals.
- `benchmark-memory-trust-kernel.py ... --launch-local` against the freshly built MCP binary —
  `state_validity=pass`, `poisoning=pass`; the four unsupported suites remain truthfully `not_tested`.
- Live artifacts: `/home/sikmindz/Coding/agent-memory-kits/docs/benchmarks/memory-trust-kernel-live-2026-07-10.json`
  and the adjacent Markdown report.

Earlier receipts retained below:

- `cargo test -p semantic-memory --test hostile_benchmark_receipt` — 3 passed.
- `cargo run -p semantic-memory --example hostile_memory_integrity --quiet` — executable completed;
  semantic-memory 8/9, admission failure explicit, strict thresholds not met.
- `cargo check --manifest-path semantic-memory-mcp/Cargo.toml --no-default-features --features search`
  — passed.
- `cargo test -p semantic-memory authority_contracts --all-features` — 2 passed; 372 unrelated unit
  tests filtered by the focused test selector.
- `cargo check -p semantic-memory --no-default-features --features brute-force` — passed.
- `cargo check -p semantic-memory --no-default-features --features usearch-backend` — passed.
- `cargo check -p semantic-memory --all-features` — passed.
- `cargo test -p semantic-memory --all-features` — passed: 371 unit tests passed, 3 ignored;
  every integration-test binary and doc test passed (two scale tests and one doc test ignored).
- `cargo fmt --manifest-path Cargo.toml -- --check` — passed.
- `cargo test --all-features authority` — passed.
- `cargo test --all-features --test authority_transactions` — passed: 5 tests, including all ten
  fault gates, duplicate retry, conflicting idempotency payload, unauthorized permit, and
  append/supersede/redact head behavior.
- `cargo test --all-features` — passed: 371 library tests, 3 ignored, and all integration suites;
  the authority transaction suite passed with 5 tests.
- `cargo check --no-default-features --features brute-force` — passed.
- `cargo check --no-default-features --features usearch-backend` — passed.
- `cargo check --all-features` — passed.
- `cargo test --manifest-path semantic-memory-mcp/Cargo.toml` — passed: 16 integration tests.
- Changed-file `rustfmt --check` — passed.
- `cargo fmt --all -- --check` — **not green** because clean, unrelated `claim-ledger` and
  `receipt-bench` files already differ from rustfmt. They were not modified by this pass.
- Shared semantic-memory handoff fact: `c8ed670f-150b-4687-8d84-007eea8792fa` in `handoffs`.

## MCP P1 correctness hardening receipt — 2026-07-10

Shipped in `semantic-memory-mcp` without modifying `semantic-memory`:

- `sm_graph_path` now returns one of four explicit terminal outcomes: `Found`,
  `NoPathWithinCompleteSearch`, `BudgetExceeded`, or `InvalidEndpoint`. Adapter-owned bounded BFS
  proves component exhaustion before reporting no path and preserves legacy `path` fields.
- MCP and HTTP statistics report health/error independently for core statistics and graph-edge
  statistics. Failed queries produce `null` values plus their error; graph failures never become a
  fabricated zero.
- Routing feedback is marked mutating in MCP metadata and responses. Caller feedback is typed as a
  `ProxyLabel`, not a verified outcome, and both routing-policy load and save errors now surface.
- `lean` and `standard` autonomous profiles expose exactly one tool,
  `sm_search_witnessed`; mutation, destructive/admin, and unwitnessed retrieval tools require the
  explicit `full` profile.
- Witnessed retrieval now has explicit stage outcome/degradation pairs and top-level `state_view`,
  `current_snapshot_id`, and `retrieval_epoch` fields. Snapshot/epoch values remain `null` and are
  explicitly degraded because the current API does not provide evidence for them.
- Added contract tests for all four graph outcomes, autonomous profile exposure, routing mutation
  metadata/proxy response, and healthy per-component stats.

Verification receipts:

- `cargo fmt --check` — passed.
- `cargo test` / focused tests — blocked before compiling `semantic-memory-mcp` by the concurrently
  owned `semantic-memory` source: `AuthorityFaultStage` does not implement `Display` for the
  `thiserror` format in `semantic-memory/src/error.rs:255`. This pass did not modify that crate.
- `cargo check --no-default-features --features search` — blocked by the same upstream compile
  error before the MCP crate compiled.
- `cargo clippy --all-targets -- -D warnings` — blocked by the same upstream compile error before
  MCP linting began.

Deferred:

- Re-run the full MCP test suite and search-profile check after the owning semantic-memory agent
  restores its crate to a compiling state. No passing test claim is made for this P1 pass.
