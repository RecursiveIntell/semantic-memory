# P32 Research-Max Retrieval Final

Run: `P32_RESEARCH_MAX_RETRIEVAL_RUNTIME`
Date: `2026-05-13`

## Final Label

`p32-retrieval-runtime-shadow-only`

The implementation now has the P32 runtime hooks, but the full 10k/100k benchmark suite and workspace-wide debt burn-down remain proof debt. Do not claim any default-readiness, production-compression, v11A conformance, v11B conformance, or v11-plus release-candidate label.

## Completed

- Active run marker moved to P32 and P31 archive pointers retained.
- TurboQuant public overclaims were qualified or removed.
- `TurboCodeWireV1` now rejects seed mismatch, QJL padding bits, reserved bytes, wrong header profile, wrong magic/version, trailing bytes, and invalid payload lengths.
- Derived vector artifact rebuilds now create generation manifests with generation ID, source snapshot digest, artifact manifest digest, counts, status, and degradations.
- Authoritative embedding writes/re-embeds/deletes invalidate active derived generations.
- TurboQuant search validates the generation once, scores encoded artifacts without per-artifact raw f32 validation, uses a bounded top-k heap, loads raw f32 only for selected candidates, and records scanned/returned/rerank/raw-load counts.
- Filtered TurboQuant search no longer automatically falls back; it uses adaptive oversampling and discloses under-return as degradation.
- v11A/v11B-compatible retrieval hooks are documented as draft surfaces, not compliance.

## Evidence

- `cargo test -p turbo-quant --test wire_format --test malformed_artifacts`: passed.
- `cargo check -p semantic-memory --features turbo-quant-codec --all-targets`: passed after the runtime patch.
- `scripts/p32_retrieval_runtime_gates.sh`: passed.
- `scripts/p32_retrieval_benchmark_gate.sh`: passed smoke gate with recall@10 `1.0`, ndcg@10 `1.0`, encoded bytes/vector `1022.0`, raw bytes/vector `1536.0`, candidate p95 `111.280556 ms`.
- `cargo test -p semantic-memory --features turbo-quant-codec --test vector_codec`: passed.
- `cargo test -p semantic-memory --features turbo-quant-codec --test search_tests turbo_quant`: passed after updating P32 receipt expectations.

## Proof Debt

- Full workspace check/test/clippy/doc were not used as release gates because the parent workspace was already broadly dirty at start.
- The smoke benchmark gate passed; internal and release-candidate benchmark classes are defined but not executed.
- Default eligibility is defined but intentionally not required for P32.
- Existing archived P31 material contains negated forbidden-claim examples; it remains evidence, not active release copy.

## End State

The tree remains dirty. P32 changed semantic-memory runtime/docs/scripts and turbo-quant claim/wire files; unrelated parent-workspace changes were preserved.
