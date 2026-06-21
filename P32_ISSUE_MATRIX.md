# P32 Issue Matrix — Research-Max Retrieval Runtime

| ID | Severity | Area | Problem | Required fix | Acceptance evidence |
|---|---:|---|---|---|---|
| P32-001 | Critical | Provenance | Active run says P30 while P31 work is active. | Set CURRENT_RUN to P32 and index P31 archive correctly. | CURRENT_RUN, CODEX_RUN_INDEX, final report. |
| P32-002 | Critical | Evidence | P31 logs archived out of active package; hostile audit sees summaries only. | Add active RUN_EVIDENCE_INDEX and digest summary. | Active evidence index with hashes. |
| P32-003 | Critical | Public claims | TurboQuant README still says “zero accuracy loss.” | Remove or qualify all unconditional claims. | grep over docs/README returns safe statements only. |
| P32-004 | High | Wire format | Seed header is written but not validated. | Reject seed mismatch in TurboCodeWireV1 decode. | seed mismatch test. |
| P32-005 | High | Wire format | QJL padding bits allow non-canonical wire bytes. | Reject non-zero unused packed-sign padding bits. | padding malformed-wire test. |
| P32-006 | High | Runtime perf | TurboQuant query loop loads raw row for every artifact. | Trust validated artifact generation; load raw only for selected candidates. | benchmark + receipt raw_rows_loaded_count. |
| P32-007 | High | Runtime perf | Candidate selection sorts full corpus. | Use bounded top-k heap/partial selection. | 10k/1536 and 100k benchmarks. |
| P32-008 | High | Filters | Any SQL filter forces raw fallback. | Implement filter-aware derived candidate path. | filtered TurboQuant tests + receipt strategy. |
| P32-009 | High | Artifact lifecycle | Derived artifacts lack generation-level manifest/state. | Add generation table/manifest with status, source snapshot, counts, digests. | generation tests. |
| P32-010 | High | Invalidation | Writes/re-embeds do not automatically invalidate artifacts. | Add invalidation on authoritative embedding mutation. | update/delete tests. |
| P32-011 | High | Benchmarking | P31 gate too narrow for release. | Add smoke/internal/RC/default-eligibility gate classes. | benchmark summary JSON. |
| P32-012 | Medium | Receipts | Degradations are mostly free text. | Add structured degradation records or v11-compatible wrapper. | JSON receipt tests. |
| P32-013 | Medium | Receipts | Candidate backend naming is misleading. | Rename successful Turbo path to candidate-then-exact. | receipt tests. |
| P32-014 | Medium | Replay | Receipt can check replay only with caller-supplied inputs. | Add input manifest/redaction semantics. | replay fixtures. |
| P32-015 | Medium | Reference behavior | No formal retrieval reference interpreter. | Add exact raw f32 reference conformance harness. | reference differential tests. |
| P32-016 | Medium | HNSW | Legacy manifestless load still allowed. | Keep for migration, but emit degradation and add migration doc. | hnsw legacy receipt/degradation test. |
| P32-017 | Medium | v11A hooks | Material ops not wrapped in operator contracts. | Add draft OperatorContract/InvocationReceipt for retrieval ops. | schema/tests. |
| P32-018 | Medium | v11B hooks | Graph surfaces are implicit. | Add GraphSurfaceDeclarationV1-compatible docs/stubs. | conformance fixture. |
| P32-019 | Medium | Workspace debt | contract-schema-gen schema drift blocks workspace test. | Fix or ledger as proof debt. | cargo test --workspace or debt ledger. |
| P32-020 | Medium | Workspace debt | clippy workspace warnings block release claim. | Fix or ledger as proof debt. | clippy pass or debt ledger. |
| P32-021 | Low | Config | `turbo_quant_require_exact_rerank` can mislead if false is unsupported. | Enforce true or document as reserved. | config validation test. |
| P32-022 | Low | Docs | P31 final says README claim removed, but README still has it. | Align final report and docs. | grep + doc check. |
| P32-023 | Low | Metrics | `returned_candidates` can exceed requested pool because it means all scored artifacts. | Split scanned/returned fields. | receipt field tests. |
| P32-024 | Low | Bench | Criterion candidate generation decodes bytes each score; good realism, but separate direct-wire scoring benchmark needed. | Add scoring_from_wire_no_allocation benchmark. | bench output. |

