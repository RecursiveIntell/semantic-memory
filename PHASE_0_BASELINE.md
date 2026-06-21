# Phase 0 - Baseline proof and task inventory

## Objective

Establish a clean, reproducible starting point before Codex edits anything. This phase should produce evidence, not major code changes.

## Files/modules to inspect

Likely locations:

```text
Cargo.toml
semantic-memory/Cargo.toml
semantic-memory/src/lib.rs
semantic-memory/src/db.rs
semantic-memory/src/search.rs
semantic-memory/src/hnsw*.rs
semantic-memory/src/quantize.rs
semantic-memory/src/projection_storage*.rs
semantic-memory/tests/*
semantic-memory-forge/src/envelope.rs
forge-memory-bridge/src/transform.rs
stack-ids/src/*
```

## Implementation tasks

1. Create `docs/giga-pass/BASELINE_AUDIT.md`.
2. Record workspace members and feature flags.
3. Record current test commands and whether they pass.
4. Inventory HNSW files, vector storage files, search files, projection/import files, and quantization files.
5. Inventory existing tests relevant to HNSW, search, episode identity, quantization, and bridge/import.
6. Add no behavior changes unless needed to make test output recordable.

## Acceptance criteria

- Baseline audit doc exists.
- Current test state is recorded with exact commands.
- Existing relevant modules/tests are listed.
- No unrelated implementation changes.

## Codex prompt

```text
Run Phase 0 of CODEX_GIGA_PASS_MASTER.md.

Produce docs/giga-pass/BASELINE_AUDIT.md containing:
- workspace members
- feature flags
- relevant modules for HNSW/search/vector/projection/import
- existing tests by category
- commands run and results
- immediate blockers preventing later phases

Do not refactor code in this phase unless a trivial doc/test command issue blocks baseline recording.
```

---
