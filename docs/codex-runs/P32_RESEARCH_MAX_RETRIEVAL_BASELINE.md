# P32 Research-Max Retrieval Baseline

Run: `P32_RESEARCH_MAX_RETRIEVAL_RUNTIME`
Date: `2026-05-13`

## Start State

The tree was dirty at start. `git status --short` showed broad parent-workspace edits, deleted root control-plane files, modified semantic-memory source/tests, and untracked P32/P31 artifacts. The missing files requested by the session protocol were:

- `CLAUDE.md` in `semantic-memory/`
- `02_MASTER_ISSUE_MATRIX.md` in `semantic-memory/`
- `04_EXACT_FILE_TOUCH_MAP.md` in `semantic-memory/`

The package-local `AGENTS.md`, P32 prompt files, and current source were used instead.

## Baseline Commands

- `cargo test -p turbo-quant wire_format malformed_artifacts`: command shape is invalid for Cargo because two test filters were supplied before `--`.
- `cargo check -p semantic-memory --features turbo-quant-codec --all-targets`: passed before the P32 patch completed.
- Claim scan found unsafe TurboQuant public wording in `../turbo-quant/README.md`.

## Baseline Classification

- Workspace cleanliness: red, pre-existing broad dirty tree.
- Focused semantic-memory feature check: green.
- Public claim discipline: red before cleanup.
- Wire canonicality: incomplete before cleanup.
- Full workspace gates: proof debt, not rerun as a release claim because unrelated parent workspace changes are out of P32 scope.
