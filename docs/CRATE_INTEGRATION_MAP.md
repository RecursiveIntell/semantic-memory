# Semantic-Memory Crate Integration Map

Generated: 2026-06-20
Scope: All crates in the Libraries monorepo and their integration opportunities with semantic-memory.

## Current integration status

| Crate | Status | semantic-memory Usage | Integration Opportunities |
|---|---|---|---|
| stack-ids | FULL | Identity primitives, digests, scope keys | None needed — fully integrated |
| forge-memory-bridge | FULL | Bridge import, projection lane | None needed — fully integrated |
| boundary-compiler | FULL | Structured output hardening | None needed — fully integrated |
| bitemporal-runtime | FULL | Bitemporal truth | None needed — fully integrated |
| turbo-quant | PARTIAL | Codec via `turbo-quant-codec` feature | Wire into compression_governor and factor_graph quantization (P0) |
| quant-governor | PARTIAL | Available via `turbo-quant-codec` feature | Use as compression policy layer for factor_graph quantization recommendations (P1) |
| scr-runtime-compression | PARTIAL | Available via `turbo-quant-codec` feature | Use for compression runtime decisions in subgraph pruning (P1) |
| poly-kv | PARTIAL | Available via `poly-kv-pool` feature | KV cache pool integration for embedding storage (P2) |
| fib-quant | NONE | Not integrated | Geometry-preserving compression for embeddings at aggressive bit rates (P2) |
| llm-pipeline | NONE | Not integrated | LLM response parsing → semantic-memory ingestion pipeline (P1) |
| llm-tool-runtime | NONE | Not integrated | Tool execution receipts → semantic-memory provenance records (P1) |
| knowledge-runtime | NONE | Not integrated | Query classification and routing → semantic-memory adaptive routing (P1) |
| forge-pilot | NONE | Not integrated | Execution pilot receipts → semantic-memory episode tracking (P2) |
| claim-ledger | NONE | Spec only, not built | Claim/evidence spine → semantic-memory provenance + contradiction detection (P0 when built) |
| llm-output-parser | NONE | Not integrated | Structured output parsing → semantic-memory chunk ingestion (P2) |
| agent-guard | NONE | Not integrated | Agent security scanning → semantic-memory MCP tool provenance (P2) |
| agent-graph | NONE | Not integrated | Graph orchestration → semantic-memory graph traversal (P2) |
| ai-batch-queue | NONE | Not integrated | Batch embedding queue → semantic-memory bulk ingestion (P2) |
| job-queue | NONE | Not integrated | Async job execution → semantic-memory receipt tracking (P2) |
| tauri-queue | NONE | Not integrated | UI job queue → semantic-memory search job tracking (P2) |
| gpu-backend | NONE | Not integrated | GPU-accelerated embedding → semantic-memory embedder backend (P2) |
| hnsw-bench | NONE | Not integrated | HNSW benchmarking → semantic-memory vector backend evaluation (P2) |
| quant-eval | NONE | Not integrated | Quantization evaluation → semantic-memory compression governor benchmarks (P2) |
| receipt-bench | NONE | Not integrated | Receipt benchmarking → semantic-memory receipt replay evaluation (P2) |

## Priority integration targets

### P0: Turbo-quant enablement everywhere

Current state: TurboQuant is wired into search.rs as a candidate backend and into vector_codec.rs as a codec. But it is NOT wired into:
- compression_governor.rs — the governor scores importance but doesn't apply TurboQuant compression
- factor_graph.rs — the new factor graph quantization function produces recommendations but doesn't apply them via TurboQuant
- integration.rs — confidence_aware_quantization produces level strings but doesn't call TurboQuant

Required changes:
1. compression_governor.rs: add a `compress_with_turbo_quant` function that takes ImportanceScore + raw embedding and produces a TurboQuant VectorArtifactV1
2. factor_graph.rs: the `factor_graph_quantization` function should return TurboQuant-compatible recommendations (already returns ConfidenceQuantizationRecommendation which has level strings — need to map to TurboQuant profiles)
3. config.rs: add `turbo_quant_default_compression: bool` flag

### P0: ClaimLedger integration (when built)

ClaimLedger is the evidence spine. When built, it should:
- Export claims to semantic-memory as facts with provenance
- Use semantic-memory's contradiction detection to find claim conflicts
- Use semantic-memory's graph edges to link claims to evidence to source spans
- Use semantic-memory's retrieval as the search layer for claim lookup

### P1: LLM pipeline integration

llm-pipeline processes LLM responses. Integration:
- LLM responses → semantic-memory as documents (chunked, embedded, searchable)
- Tool call receipts → semantic-memory as provenance records
- Parse failures → semantic-memory as error episodes for debugging

### P1: Knowledge-runtime integration

knowledge-runtime has query classification and routing. Integration:
- knowledge-runtime routes queries to semantic-memory
- semantic-memory's adaptive routing (RoutingDecision) is the execution plan
- knowledge-runtime's entity resolution feeds into semantic-memory's graph traversal

### P1: LLM tool runtime integration

llm-tool-runtime executes tools. Integration:
- Tool execution receipts → semantic-memory as episodes
- Tool descriptors → semantic-memory as facts for audit
- Tool provenance → semantic-memory provenance records

### P2: Fib-quant integration

fib-quant does geometry-preserving compression. Integration:
- Alternative codec for semantic-memory embeddings at aggressive bit rates
- Would require a new feature flag `fib-quant-codec` similar to `turbo-quant-codec`
- Compression governor could choose between TurboQuant and FibQuant based on vector characteristics

## Dependency order

1. Turbo-quant enablement (P0, no new deps)
2. LLM pipeline integration (P1, requires llm-pipeline to expose ingestion API)
3. Knowledge-runtime integration (P1, requires knowledge-runtime routing API)
4. LLM tool runtime integration (P1, requires llm-tool-runtime receipt API)
5. ClaimLedger integration (P0 when built, blocked on ClaimLedger implementation)
6. Fib-quant integration (P2, requires fib-quant codec API)
7. Poly-kv deeper integration (P2, already partially wired)
8. GPU backend integration (P2, requires gpu-backend embedding API)

## Conflict risks

- **turbo-quant + fib-quant**: Both want to be the compression codec. Need a codec selection policy in compression_governor.
- **knowledge-runtime + semantic-memory routing**: Both have routing logic. Need clear boundary: knowledge-runtime decides WHERE to route, semantic-memory routing decides HOW to search.
- **claim-ledger + semantic-memory provenance**: Both track evidence. Need clear boundary: ClaimLedger owns claim truth, semantic-memory owns retrieval witnesses.
- **llm-pipeline + semantic-memory ingestion**: Both can ingest documents. Need clear boundary: llm-pipeline produces structured output, semantic-memory stores and indexes it.