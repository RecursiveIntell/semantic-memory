# Issue Matrix

This matrix compresses the re-audit into implementation tickets. P0 and P1 items should block TurboQuant default integration.

| ID | Priority | Layer | Issue | Required fix | Acceptance evidence |
|---|---:|---|---|---|---|
| GP-000 | P0 | 0 | No durable pass ledger | Add audit ledger and status JSON | Ledger records baseline, commands, pass status |
| GP-001 | P0 | 1 | Invalid vectors can reach derived paths if validation is missed | Centralize finite/dim/blob validation | Tests reject NaN/inf/wrong dim/malformed blobs |
| GP-002 | P0 | 1 | HNSW integrity can be count-only | Key-level parity live rows ↔ keymap ↔ typed keys | Swapped/stale/wrong-prefix keys fail tests |
| GP-003 | P0 | 1 | HNSW graph sidecar validation shallow | Header/profile/digest validation or rebuild-on-suspicion | Corrupted/empty/wrong-dim sidecar not trusted |
| GP-004 | P0 | 1 | Persisted `usize` in sidecar format | Fixed-width versioned header | Static check + persistence tests |
| GP-005 | P0 | 1 | Filtered HNSW under-return | Adaptive overfetch or brute-force scoped fallback | Sparse namespace test matches brute force |
| GP-006 | P0/P1 | 1 | Pending HNSW mutation ambiguity | Rebuild-on-failure or snapshot swap | Simulated save failure degrades/rebuilds safely |
| GP-007 | P1 | 2 | Search ranking samples wall clock internally | Add `SearchContext.evaluation_time` | Frozen-time ranking tests |
| GP-008 | P1 | 2 | Fallback/degradation invisible | Add receipt/explanation skeleton | Receipt captures backend/fallback/result IDs |
| GP-009 | P1 | 3 | Boundary defaulting can invent meaning | Convert semantic defaults to typed errors/policies | Ugly import/bridge tests |
| GP-010 | P1 | 4 | `embedding_q8` could be overloaded with TurboQuant semantics | Add explicit codec profiles/artifacts | Raw + SQ8 through codec interface |
| GP-011 | P1 | 4 | No stable codec profile identity | Add digest/canonical profile | Stable profile digest tests |
| GP-012 | P1 | 4 | Profile mismatch may score silently | Fail closed on dim/profile/codec mismatch | Mismatch tests |
| GP-013 | P1 | 4 | No raw oracle abstraction | `RawF32Codec` reference path | Raw scoring retained and used in tests |
| GP-014 | P1 | 5 | TurboQuant could be fused prematurely | Feature-gated `TurboQuantCodec` only after gates | Optional feature; default build independent |
| GP-015 | P1 | 5 | TurboQuant approximation could be hidden | Search receipt marks approximate/rerank/drift | Receipt + drift metrics tests |
| GP-016 | P1 | 5 | TurboQuant rank drift unknown | Deterministic harness vs raw reference | recall/rank drift/score error report |
| GP-017 | P1 | 6 | Users cannot answer “why this result?” | Product-facing explanation API | Example result explanation test/doc |
| GP-018 | P2 | 7 | Docs overuse internal doctrine | Public docs use receipts/replay/source grounding | README updated with practical framing |
| GP-019 | P2 | 7 | Portfolio story scattered | Add flagship language: local-first memory with receipts | Case-study copy exists |
| GP-020 | P2 | 7 | Internal prompt/spec docs leak into release packages | Move/exclude internal docs as appropriate | Package manifest excludes stale sidecars/prompts |

## P0 closure rule

A P0 is closed only when:

1. The relevant code path is fixed.
2. A regression test exists.
3. The acceptance command passes or the environment blocker is recorded.
4. The ledger maps the issue to evidence.

## P1 closure rule

A P1 can be deferred only if:

1. TurboQuant/default release is also blocked.
2. The defer rationale is explicit.
3. The risk is visible in docs/ledger.

## P2 closure rule

P2 can be batched, but not silently ignored.
