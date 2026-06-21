# Giga-Pass Issue Matrix

| ID | Severity | Track | Issue | Exit condition | Phase |
|---|---:|---|---|---|---|
| GP-001 | P0 | HNSW | Keymap count parity hides swapped/stale keys | Full key-level parity tests pass | 1 |
| GP-002 | P0 | HNSW | Graph sidecar validation too shallow | Header/digest/rebuild-on-suspicion implemented | 1 |
| GP-003 | P0 | HNSW | Persisted `usize` in sidecar | Fixed-width versioned header or migration/rebuild path | 1 |
| GP-004 | P0 | Search | Filtered HNSW under-return | Exact/brute-force filtered fallback with receipt marker | 1/2 |
| GP-005 | P0/P1 | HNSW | Pending op save failure ambiguity | Dirty/untrusted + rebuild/fallback policy | 1 |
| GP-006 | P1 | Search | Wall-clock recency | `SearchContext.evaluation_time` controls ranking | 2 |
| GP-007 | P1 | Boundary | Risky `unwrap_or_default` at semantic boundaries | Typed errors/degradations/compatibility handling | 2/3 |
| GP-008 | P1 | Codec | `embedding_q8` overloading risk | New codec profile/artifact surface | 3 |
| GP-009 | P1 | Receipt | Search explainability not durable | Receipt-ready metadata type/API | 2 |
| GP-010 | P1 | Codec | Raw exact oracle not first-class | RawF32 codec/reference scoring path | 3 |
| GP-011 | P1 | TurboQuant | Compressed retrieval drift unknown | Fixed-corpus drift report exists | 4/5 |
| GP-012 | P2 | Docs | Public story too broad | README/case study centered on memory with receipts | 6 |
| GP-013 | P2 | Package | Internal docs/spec sprawl | Package/public docs curated | 6 |
| GP-014 | P2 | CI | Release bar not codified | CI/release commands documented and ideally enforced | 5 |
