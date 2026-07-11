# Semantic-Memory Research Atlas: Live Reconciliation

Date: 2026-07-10
Source bundle: `/home/sikmindz/Downloads/recursiveintell_semantic_memory_research_bundle_2026-07-10.zip`
Evidence boundary: The bundle synthesizes 90 sources. Paper/preprint mechanisms are research inputs, not locally reproduced results. This reconciliation compares its recommendations to the live local code and receipts after the 2026-07-10 trust-kernel remediation.

## Verdict

The atlas is directionally strong. Its central thesis is correct: the highest-value frontier is no longer generic retrieval quality. It is proof-governed memory state formation, state interpretation, causal influence, and action authority.

The atlas is partly stale against the live implementation. Today’s remediation already closed substantial portions of temporal correctness, atomic authority, witnessed retrieval, injection containment, idempotency, and fault rollback. The next build should extend those primitives rather than create parallel replacements.

## Live capability delta

| Atlas recommendation | Live status | Evidence | Actual remaining gap |
|---|---|---|---|
| Explicit current/historical state views | Implemented | `StateView`, `list_facts_with_view`; hostile benchmark 9/9 | Transition/trajectory views, query-premise classification, dependency propagation |
| No stale/future leakage | Implemented and benchmarked | Hostile benchmark: zero stale retrieval and temporal correctness pass | External STALE/A-TMA/MemTrace adapters |
| Atomic canonical transitions | Partially implemented | `MemoryAuthority` append/supersede/redact, journal, epochs, receipts, idempotency, ten fault gates | Pre-commit coverage/preservation/faithfulness verifier and quarantine lane |
| Unique active head/fail-closed lineage | Implemented | branching lineage failure tests and Current filtering | Typed downstream state-dependency closure |
| Mandatory retrieval witness | Implemented | `sm_search_witnessed`, `RetrievalResponseV1`, receipt verification, lean profile containment | State-resolution receipt above retrieval; evidence-gap loop |
| Provenance-safe injection | Implemented | fail-closed host hooks and DATA ONLY framing; 154 Python tests | Immutable origin authority carried through derivations |
| Origin/action authority | Partial | `AuthorityPermit`, principal/caller/capability/evidence, autonomous tool containment | Separate recall/assertion/action scopes, laundering-resistant inheritance, revocation |
| Poisoning evaluation | Partial | bounded poisoning/governance suite passes | MPBench/GhostWriter/sleeper/laundering reproduction and benign-utility measurement |
| Diagnostic evaluation lab | Partial | hostile benchmark, StateValidityBench, receipts, typed not-tested outcomes | Named external benchmark adapters and feasible competitor baselines |
| Causal memory influence | Missing | Current benchmark marks reasoning drift not tested | Counterfactual no-memory/memory runner, claim/tool-argument deltas |
| Evidence-gap retrieval | Missing | Current retrieval is witnessed but fixed candidate retrieval | Typed sufficiency decision, missing-evidence diagnosis, bounded iterative routes |
| Multi-principal governance | Mostly missing | Principal exists in authority permits | Subject/audience/read principal, access-time policy across every path |
| Selective forgetting closure | Partial | redact/delete/supersession and lawful subtraction exist | Derived-artifact dependency closure and post-forget adversarial verification |

## Highest-ROI implementation order

### P0-1: Verified Memory Transition Compiler

Build on `MemoryAuthority`; do not create another store or journal.

Add:
- `MemoryTransitionCandidateV1`
- `MemoryTransitionVerificationV1`
- deterministic coverage, preservation, and source-faithfulness checks
- unsupported/omitted span reporting
- quarantine disposition
- simulation of active-head and dependency effects before authority commit

Acceptance gates:
- Existing atomic/idempotency/fault tests remain green.
- Raw evidence remains immutable and sufficient for reconstruction.
- Every derived assertion maps to exact source spans.
- Omission, corruption, and unsupported-content rates are independently measurable.
- Failed candidates never enter Current state.

Why first: authority now guarantees atomicity, but it does not prove that the proposed semantic transition is faithful or complete.

### P0-2: State Epistemics Kernel

Build above `StateView` and witnessed retrieval.

Add:
- `StateResolutionReceiptV1`
- `PremiseStatus`: supported, stale, contradicted, unsupported, ambiguous
- Transition and trajectory views
- `StateDependencyEdgeV1`: invalidates, weakens, requires_reevaluation, scope_changes, derived_from_state
- explicit answer policy: answer, correct premise, disclose conflict, abstain, request evidence

Acceptance gates:
- STALE/A-TMA/MemTrace-style fixtures separate current, historical, trajectory, and false-premise behavior.
- Historical accuracy does not regress.
- Missing evidence, unresolved conflict, and budget exhaustion never become confident negatives.
- Every memory-influenced answer declares its state view and resolution receipt.

Why second: temporal storage is now correct, but query-conditioned interpretation is not yet a first-class artifact.

### P0-3: Diagnostic Memory Evaluation Lab

Implement adapters before broad feature work:
- STALE
- A-TMA/LTP
- MemTrace
- MemConflict
- TRUSTMEM/HaluMem transition subsets
- MPBench/GhostWriter-style poisoning
- minimal GateMem/GroupMemBench
- minimal MemoryArena action loops

Required baselines:
- no memory
- full context
- BM25 only
- dense only
- current exact hybrid
- each new stage in isolation
- named competitors only where identical, reproducible adapters are feasible

Acceptance gate: no stage receives credit unless it changes the final evidence packet, answer, or action and improves held-out outcomes.

Why third: the current hostile benchmark validates integrity primitives, not external agent-memory superiority.

### P0-4: Causal Memory Influence and Tool Drift

Add offline counterfactual cells:
- no memory
- gold memory
- retrieved memory
- unlabeled memory
- witnessed/state-labeled memory
- distractors
- poison
- governed admission

Emit `MemoryInfluenceReceiptV1` with claim deltas, tool selection/argument deltas, unsupported-claim delta, risk delta, latency, and token cost.

Keep this offline or risk-triggered. Do not double inference on every ordinary request.

### P0-5: Origin-Bound Authority

Extend existing `AuthorityPermit`; do not replace it.

Add immutable write-time origin plus separate:
- recall scope
- assertion scope
- action scope
- elevation requirements
- revocation reference

Derived artifacts inherit maximum ancestor risk and minimum ancestor authority. Summarization, rephrasing, trusted-tool echo, and repeated corroboration cannot elevate authority.

Blocking tests: direct poison, sleeper activation, summary laundering, trusted-tool echo, manufactured corroboration, direct-ID bypass, cache/export/replay bypass, and cross-principal leakage.

## P1 after the five P0 items

1. Evidence-gap retrieval and state-aware reranking with typed `EvidenceInsufficient` versus `BudgetExceeded`.
2. Selective forgetting closure across canonical state, FTS/vector/graph/cache/export projections, with post-forget attacks.
3. Multi-principal/audience-aware policy applied uniformly to search, direct get, graph, cache, replay, and export.
4. Shadow-learned routing/write/retention proposals with deterministic promotion and rollback.
5. Procedural skill artifacts with tested envelopes; keep separate from factual memory.

## Defer

- More graph expansion without answer-level ablation.
- More ANN/compression work before state-critical and contradiction-critical recall metrics exist.
- Online RL controlling canonical writes.
- Neural/test-time memory as canonical truth.
- Autonomous consolidation without transition verification.
- Broad superiority claims.

## Strongest defensible product/research thesis

A local-first memory epistemics and control plane for persistent agents: verified transitions into authoritative state, query-conditioned state resolution, witnessed evidence retrieval, causal influence measurement, and origin-bounded authority for answers and actions.

This is stronger and more differentiated than “agent memory,” but it is only a validated comparative advantage after external benchmark adapters and identical-baseline results exist.

## Immediate implementation slice

The next concrete slice should be `MemoryTransitionCandidateV1` + deterministic verifier + quarantine, integrated directly before `MemoryAuthority` commit. It has the best leverage because every later state, influence, security, and benchmark feature depends on trustworthy memory formation.
