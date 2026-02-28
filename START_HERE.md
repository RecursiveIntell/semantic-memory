# START HERE — semantic-memory v0.2.0 Upgrade

## What To Do

Read `CLAUDE.md`, then read `UPGRADE_SPEC.md` in its entirety, then read `AGENTS.md`, then read `HNSWLIB_RS_REFERENCE.md`. These four documents contain the complete specification for upgrading semantic-memory from v0.1.0 to v0.2.0.

Begin with `@researcher` as defined in AGENTS.md. Execute each agent phase sequentially, writing state to `.agent_context/` between phases. Each phase must compile and pass all tests before proceeding to the next.

## The Upgrade In One Paragraph

Add HNSW approximate nearest neighbor search with SQ8 scalar quantization to the semantic-memory Rust crate. Replace brute-force vector scan as the default backend using the `hnswlib-rs` crate. Change storage from single SQLite file to a directory with three files (SQLite + HNSW graph + quantized vectors). Gate backends behind feature flags (`hnsw` default, `brute-force` opt-in). Fix 4 known bugs. Write comprehensive integration tests. The SQLite schema for existing tables does NOT change — SQLite remains the source of truth for content and f32 embeddings.

## File Map

```
CLAUDE.md                  → Project rules, build commands, hard constraints
AGENTS.md                  → 7 specialized agents, one per phase, with handoff protocol
UPGRADE_SPEC.md            → 1,800 line specification: every type, method, data flow, test case
HNSWLIB_RS_REFERENCE.md    → Dependency API reference for the HNSW crate
```

## Phase Summary

| Phase | Agent | What Happens | Key Deliverable |
|---|---|---|---|
| 0 | @researcher | Verify hnswlib-rs exists on crates.io, confirm API | `.agent_context/researcher.md` |
| 1 | @foundation | Cargo.toml, storage.rs, quantize.rs, error types | Compiles in all feature combos |
| 2 | @hnsw | src/hnsw.rs wrapper module | HNSW works in isolation |
| 3 | @integrator | Wire into MemoryStore, search, all write paths | Full pipeline functional |
| 4 | @bugfixer | Fix 4 identified bugs | Clean codebase |
| 5 | @tester | 5 integration test files | All tests pass |
| 6 | @reviewer | README, clippy, final verification | Ship-ready v0.2.0 |

## Begin

Start with `@researcher`. Verify the `hnswlib-rs` dependency, then proceed through each phase. Do not skip phases. Do not proceed to the next phase until the current phase compiles and passes tests.
