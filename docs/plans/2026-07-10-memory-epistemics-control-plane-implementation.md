# Memory Epistemics and Control Plane Implementation Plan

> **For Hermes:** Use subagent-driven-development task-by-task. Use strict RED/GREEN TDD. Controller owns final cargo, benchmark, binary installation, service restart, and live receipts.

**Goal:** Extend the existing semantic-memory authority and witnessed-retrieval substrate into verified memory formation, explicit state epistemics, diagnostic evaluation, causal influence measurement, and origin-bounded answer/action authority.

**Architecture:** SQLite and raw evidence remain canonical. New components are typed verification and interpretation layers around the existing `MemoryAuthority`, `StateView`, and witnessed retrieval paths—not new stores. Learned or LLM components may propose candidates in shadow; deterministic code owns commitment, authority, promotion, and rollback.

**Tech stack:** Rust, SQLite/rusqlite, serde, blake3, chrono, semantic-memory, semantic-memory-mcp, Python benchmark adapters in agent-memory-kits.

## Source inventory checked

- `/home/sikmindz/Downloads/recursiveintell_semantic_memory_research_bundle_2026-07-10.zip`
- `RECURSIVEINTELL_SEMANTIC_MEMORY_RESEARCH_ATLAS_2026-07-10.md` — 1,277 lines, 90-source synthesis.
- `CURRENT_CAPABILITY_TO_RESEARCH_GAP_MATRIX.md`
- `RESEARCH_SOURCE_MATRIX.tsv` and `.json`
- `/home/sikmindz/Coding/Libraries/semantic-memory/docs/research/RESEARCH_ATLAS_LIVE_RECONCILIATION_2026-07-10.md`
- Existing live implementation: `src/authority.rs`, `src/authority_contracts.rs`, `src/knowledge.rs`, `src/search.rs`, `tests/authority_transactions.rs`, hostile benchmark and receipts.
- Existing MCP implementation: witnessed retrieval, authority-backed mutation, lean-profile containment.
- Existing host implementation: fail-closed provenance framing and witnessed recall.

## Verified starting state

- Hostile memory-integrity benchmark: 9/9, strict 100% threshold met.
- Core authority supports atomic append/supersede/redact, receipts, epochs, idempotency, and fault rollback.
- Current/historical state views are implemented and future-state leakage is fixed.
- Lean/standard autonomous MCP profiles expose mandatory witnessed retrieval only.
- Host auto-injection fails closed on missing provenance/state/receipt.
- Working trees are dirty from the trust-kernel implementation; preserve all unrelated changes and do not commit without explicit instruction.

## Sprint A — Verified Memory Transition Compiler

### A1. Define source-span and transition contracts

**Files:**
- Create: `src/transition_contracts.rs`
- Modify: `src/lib.rs`
- Test: `tests/transition_compiler.rs`

**RED tests:**
- Candidate rejects missing source references.
- Source spans reject empty artifact IDs and invalid ranges.
- Verification serializes deterministic schema/version/disposition.
- Unsupported and omitted spans remain distinct.

**Implementation:**
- `SourceSpanRefV1`
- `AssertionDraftV1`
- `SupersessionDraftV1`
- `MemoryTransitionCandidateV1`
- `MemoryTransitionVerificationV1`
- `TransitionOperation`, `TransitionDisposition`, `VerificationScore`

### A2. Deterministic coverage/preservation/faithfulness verifier

**Files:**
- Create: `src/transition_verifier.rs`
- Modify: `src/lib.rs`
- Test: `tests/transition_compiler.rs`

**RED tests:**
- Exact source-backed draft passes.
- Unsupported text is identified and quarantined.
- Omitted required source span is identified.
- Still-valid active assertions cannot disappear without an explicit supersession/retraction draft.
- Hashes and results are stable across replay.

**Implementation rule:** Deterministic checks only. No LLM judgment in canonical admission.

### A3. Quarantine persistence and inspection

**Files:**
- Modify: `src/db.rs`
- Modify: `src/authority.rs`
- Modify: `src/error.rs`
- Test: `tests/transition_compiler.rs`

**RED tests:**
- Failed candidate persists in quarantine, not facts/Current.
- Quarantine record retains candidate, verification, source refs, digest, and timestamp.
- Same idempotency key returns the same quarantine record.
- Conflicting retry fails closed.

### A4. Compile-and-commit authority path

**Files:**
- Modify: `src/authority.rs`
- Modify: `src/authority_contracts.rs`
- Test: `tests/transition_compiler.rs`
- Test: `tests/authority_transactions.rs`

**RED tests:**
- Verified candidate commits through the existing authority transaction.
- Verification failure cannot call canonical mutation.
- Fault injection leaves either zero transition or one complete transition.
- Existing append/supersede/redact behavior remains compatible.

**Sprint A gate:**
```bash
cargo test --test transition_compiler -- --nocapture
cargo test --test authority_transactions -- --nocapture
cargo test --lib
cargo test --all-targets
cargo check --no-default-features --features brute-force
cargo check --all-features
git diff --check
```

## Sprint B — State Epistemics Kernel

### B1. State-resolution contracts

**Files:**
- Create: `src/state_epistemics.rs`
- Modify: `src/lib.rs`
- Test: `tests/state_epistemics.rs`

Add:
- `StateResolutionReceiptV1`
- `PremiseStatus`
- `AnswerPolicy`
- `ResolvedAssertionV1`
- `BeliefAlternativeV1`
- Transition/trajectory request modes without weakening existing `StateView` compatibility.

### B2. State-dependency edges and propagation

Add typed relationships:
- `invalidates`
- `weakens`
- `requires_reevaluation`
- `scope_changes`
- `derived_from_state`

**RED tests:** supersession closure propagates deterministic valid/invalid/uncertain/pending states; cycles and conflicting heads fail closed.

### B3. Premise classification and answer policy

**RED tests:** supported, stale, contradicted, unsupported, and ambiguous premises map to explicit policies. Missing evidence, budget exhaustion, and unresolved conflict never map to confident negative answers.

### B4. Witnessed state-resolution MCP path

**Files:**
- Modify sibling `semantic-memory-mcp/src/server.rs`
- Modify sibling MCP tests.

Add a state-resolved witnessed retrieval envelope that references the existing retrieval witness and includes resolution receipt/digest. Keep lean profile read-only.

**Sprint B gate:** core all-targets + MCP default/search profiles + hostile benchmark 9/9 with no historical regression.

## Sprint C — Diagnostic Memory Evaluation Lab

### C1. Versioned benchmark manifest and failure taxonomy

**Files:** agent-memory-kits `shared/scripts/`, `shared/fixtures/`, `tests/`, `docs/benchmarks/`.

Add phase-local statuses for ingestion, extraction, transition proposal, verification, commit, indexing, retrieval, rerank, state resolution, evidence sufficiency, admission, answer use, tool arguments, testimony, and forgetting.

### C2. State benchmark adapters

Implement deterministic/local subsets for STALE, A-TMA/LTP, MemTrace, MemConflict, and Supersede. Dataset licenses and hashes must be recorded; unavailable datasets become `not_tested`, never pass.

### C3. Transition/security/governance adapters

Implement locally reproducible TRUSTMEM/HaluMem operation subsets, MPBench/GhostWriter-style attacks, and minimal GateMem/GroupMemBench fixtures.

### C4. Baseline matrix

No memory, full context, BM25, vector, exact hybrid, witnessed hybrid, state-resolved path, and each feature ablation. Named competitors only when identical adapters install and execute successfully.

**Sprint C gate:** clean-run receipt includes environment, commit/dirty status, dataset/model/config hashes, raw predictions, stage outcomes, thresholds, confidence intervals where applicable, costs, and explicit failures.

## Sprint D — Causal Memory Influence and Tool Drift

### D1. Influence contracts

Add `MemoryInfluenceReceiptV1`, claim deltas, tool-selection/argument deltas, utility/risk vectors, and influence classes.

### D2. Offline counterfactual runner

Cells: no-memory, gold, retrieved, unlabeled, witnessed/state-labeled, distractors, poison, governed. Store output hashes and model/prompt configs.

### D3. Admission calibration

Produce offline calibration data only. No synchronous double inference in normal requests. Promotion requires held-out quality, state, poisoning, safety, and cost gates.

**Sprint D gate:** at least one deterministic fixture and one real-model bounded run; no broad causal claim from deterministic fixtures alone.

## Sprint E — Origin-Bound Authority

### E1. Immutable origin and separated scopes

Extend—not replace—`AuthorityPermit` with `OriginAuthorityLabelV1`: origin class/principal/channel/digest, recall/assertion/action scopes, elevation requirements, revocation.

### E2. Laundering-resistant derivation

Derived summaries inherit maximum ancestor risk and minimum ancestor authority. Rephrasing, trusted-tool echo, and repeated corroboration cannot elevate authority.

### E3. Uniform access-path enforcement

Apply identical scope policy to search, direct get, graph traversal, cache, export, replay, and action/tool-argument use.

### E4. Attack suite

Direct poison, sleeper activation, summary laundering, trusted-tool echo, manufactured corroboration, direct-ID bypass, cache/export/replay bypass, and cross-principal leakage.

**Sprint E gate:** attack thresholds pass while benign recall stays above declared floor; origin survives import/export/consolidation/compression/re-embedding.

## Sprint F — Evidence Gaps, Forgetting, Multi-Principal, and Shadow Policies

### F1. Evidence-gap controller

Typed `EvidenceGapV1`, `EvidenceInsufficient`, and `BudgetExceeded`; bounded iterative lexical/vector/graph/episodic routes; explicit changed-order/stage receipts.

### F2. Forgetting closure

`ForgettingClosureReceiptV1`; dependency closure across canonical facts and derived FTS/vector/graph/cache/export artifacts; logical invalidation distinct from physical deletion; adversarial post-forget checks.

### F3. Multi-principal/audience model

Writer, subject, audience-at-write, current reader, role, resource scope, consent basis, and access-time authorization. Contradictions must survive dedup for adjudication.

### F4. Shadow policy governor

Learned policies may propose retrieval/write/retention actions but cannot mutate canonical state, schemas, or authority. Promotion is content-addressed, benchmark-gated, canaried, and rollbackable.

### F5. Procedural skill artifact pilot

Versioned, tested, revocable workflow artifact with preconditions, phase graph, tool manifest, postconditions, tested model/tool envelope, failure modes, and rollback. Keep separate from user facts.

## Full verification gauntlet

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo check --no-default-features --features brute-force
cargo check --no-default-features --features usearch-backend
cargo check --all-features
cargo run --example hostile_memory_integrity --quiet
cd ../semantic-memory-mcp
cargo test --quiet
cargo check --no-default-features --features search --quiet
cd ../../agent-memory-kits
python3 -m pytest -q
```

Controller then rebuilds/installs MCP binaries, restarts both user services, verifies one-tool lean discovery, performs authority-write/idempotency/witnessed-recall smoke tests, and removes or invalidates test fixtures according to receipt constraints.

## Claim boundaries

Safe after Sprint A: transitions can be deterministically verified/quarantined and committed atomically under tested local invariants.

Safe after Sprint B: queries receive explicit local state-resolution receipts under covered fixture classes.

Not safe until Sprint C external runs: superiority over Mem0, Graphiti/Zep, Letta, A-MEM, HippoRAG, or other named systems.

Not safe until Sprint D real-model runs: causal improvement in answer quality or tool safety.

Not safe from external deletion alone: parametric model unlearning.

## Hard no list

- No shadow truth store.
- No LLM unilateral authority to commit canonical state.
- No online RL canonical writes.
- No graph/decoder stage credited unless it changes final outcomes.
- No similarity-only action authorization.
- No hidden-state/neural memory as canonical truth.
- No performance or superiority claim without identical-input receipts.
- No destructive migration of raw evidence or exact vectors.
- No commits while preserving the current dirty remediation tree unless explicitly requested.
