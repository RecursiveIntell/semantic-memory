# Patch Order Summary

1. Baseline ledger and audit import
2. Hermetic workspace/root Cargo.toml/archive reproducibility
3. Embedding and vector invariants
4. Explicit little-endian f32 codec
5. Deletion/cascade stale-state correctness
6. Integrity engine key/content checks
7. HNSW sidecar persistence hardening
8. Search correctness/filter fallback/async blocking fix
9. Projection/episode/bridge consistency
10. API validation/error hygiene/P2-P3 generated cleanup
11. Docs/CI/bench harness
12. Repackage and clean-extraction gate

Do not start TurboQuant until this pass is complete.
