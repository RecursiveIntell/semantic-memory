# P32 Run Evidence Index

Run: `P32_RESEARCH_MAX_RETRIEVAL_RUNTIME`
Date: `2026-05-13`

## Active Evidence

- Baseline report: `docs/codex-runs/P32_RESEARCH_MAX_RETRIEVAL_BASELINE.md`
- Final report: `docs/codex-runs/P32_RESEARCH_MAX_RETRIEVAL_FINAL.md`
- Readiness: `docs/RETRIEVAL_RUNTIME_READINESS.md`
- TurboQuant readiness: `docs/TURBOQUANT_READINESS.md`
- Artifact generation: `docs/VECTOR_ARTIFACT_GENERATION.md`
- Benchmark gates: `docs/RETRIEVAL_BENCHMARK_GATES.md`
- Gate scripts: `scripts/p32_retrieval_runtime_gates.sh`, `scripts/p32_retrieval_benchmark_gate.sh`

## Command Evidence

- `cargo test -p turbo-quant --test wire_format --test malformed_artifacts`: passed.
- `cargo check -p semantic-memory --features turbo-quant-codec --all-targets`: passed.
- `scripts/p32_retrieval_runtime_gates.sh`: passed.
- `scripts/p32_retrieval_benchmark_gate.sh`: passed and wrote `turboquant-smoke-benchmark-summary.json`.
- `cargo test -p semantic-memory --features turbo-quant-codec --test vector_codec`: passed.
- `cargo test -p semantic-memory --features turbo-quant-codec --test search_tests turbo_quant`: passed.
- Public claim grep was repaired for active README/doc surfaces; archived P31 evidence still contains forbidden phrases only as historical/negated material.

## Archive Pointers

- `docs/codex-runs/archive/P31/ARCHIVE_MANIFEST.json`
- `docs/codex-runs/archive/unclassified/20260513T023602Z/ARCHIVE_MANIFEST.json`
