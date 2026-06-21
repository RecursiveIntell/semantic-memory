# Suggested AGENTS.md for semantic-memory

## Role

You are working on a Rust local-first AI memory substrate. The project goal is trustworthy memory: source-grounded retrieval, deterministic vector/index behavior, and receipt-backed search/execution paths.

## Build and test commands

Prefer running the strongest available subset:

```bash
cargo fmt --all --check
cargo check --workspace --all-targets --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --all-features --no-deps
```

If the workspace is not packaged at the archive root, run the equivalent commands from the relevant package root and document limitations.

## Non-negotiable design rules

1. SQLite/projected memory rows are authoritative for queryable memory.
2. HNSW, q8, TurboQuant codes, cached scores, and sidecars are derived artifacts only.
3. Derived vector/index artifacts must be rebuildable from authoritative rows.
4. Do not make compressed vectors or indexes the source of truth.
5. Do not silently invent import/export/bridge semantics.
6. Every semantic-boundary parse/default failure must be a typed error, explicit compatibility policy, or degradation record.
7. Every embedding write path must validate dimension and finite values.
8. Never persist `usize` in an on-disk format.
9. Sidecars and binary formats must be versioned and reject unsupported versions.
10. Search must be deterministic when supplied an explicit evaluation time.
11. Approximation/fallback/degradation must be visible in explanations or receipts.
12. TurboQuant integration must be feature-gated and admitted through a vector codec interface.
13. Raw/reference scoring must remain available for differential tests.
14. Every correctness fix needs a regression test.
15. Do not remove meaningful functionality to make tests pass.

## HNSW rules

- Count parity is not enough.
- Verify live row -> expected key and active key -> live row.
- Malformed keys, stale keys, duplicate keys, wrong prefixes, and swapped IDs must fail integrity checks.
- If sidecar save/load/validation fails, mark HNSW degraded and rebuild from SQLite before trusting it.
- Filtered HNSW under-return must fallback or disclose degradation.

## Codec rules

- Use codec profiles for all compressed vector representations.
- Profile identity must include family, version, dim, score semantics, normalization, and codec-specific parameters.
- Profile mismatch must fail closed.
- `embedding_q8` is legacy/current scalar quantization and must not be silently redefined as TurboQuant.

## Public language

In user-facing docs, prefer:

- receipts,
- replay,
- source grounding,
- “why this result?”,
- trustworthy AI memory,
- local-first memory.

Avoid leading with internal terms such as constitutional runtime, artifact law, bitemporal doctrine, decoder kernel, or lawful subtraction.

## Final report requirement

Every pass must report:

- files changed,
- tests added,
- commands run,
- results,
- unresolved risks,
- whether TurboQuant eligibility changed.
