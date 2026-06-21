# P31 TurboQuant Readiness Baseline

Date: 2026-05-12

## Starting State

The working tree was already heavily dirty before this pass, including many
parent-workspace edits, deleted archived control files, and existing
`semantic-memory` changes. Relevant starting state included modified
`semantic-memory` source/tests and an existing optional `turbo-quant-codec`
dependency on `../turbo-quant`.

`CLAUDE.md`, `02_MASTER_ISSUE_MATRIX.md`, and `04_EXACT_FILE_TOUCH_MAP.md` were
not present in the crate directory. I read `../CLAUDE.md` and the archived V29
matrix/touch-map copies under `../docs/archive/superseded_packs/`.

`~/Documents/turbo-quant` exists, but the dependency used by this workspace is
`../turbo-quant`, which was already the newer copy and already compilable through
the workspace dependency graph.

## Baseline Commands

```bash
git status --short
cargo fmt --all --check
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets --all-features
```

## Results

- `cargo fmt --all --check`: passed.
- `cargo check --workspace --all-targets`: passed with existing warnings.
- `cargo test --workspace`: failed in `contract-schema-gen`.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: failed
  before this pass's scoped code on existing `expect_used` warnings in other
  crates.
- `cargo clippy --workspace --all-targets --all-features`: passed with many
  existing warnings.

## Baseline Blockers

`cargo test --workspace` failure:

```text
contract-schema-gen tests::canonical_schemas_match_generator_output
schema drift detected for '../schemas/verification-case-v1.schema.json'
```

`cargo clippy ... -D warnings` blockers include existing test `expect()` /
`expect_err()` use in crates such as `continuity-runtime`,
`authority-delegation`, and `assurance-runtime`.

These blockers are outside the scoped TurboQuant readiness changes.
