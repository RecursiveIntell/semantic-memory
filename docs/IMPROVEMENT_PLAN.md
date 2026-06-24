# semantic-memory improvement plan
Date: 2026-06-23
Author: Hermes Agent (compiled from semantic memory research corpus)

## CURRENT STATE

- v0.5.3, 36K lines Rust, 642 tests passing (all features)
- 52 source modules
- Architecture: SQLite + FTS5 + usearch 2.25 (default) or hnsw_rs (opt-in) or brute-force
- Hybrid search: BM25 + dense vector + RRF fusion
- 7 novel research combinations ALL IMPLEMENTED (2026-06-20):
  1. factor_graph.rs -- heterogeneous edge BP (13 tests)
  2. late_interaction.rs -- ColBERT MaxSim
  3. topology.rs -- persistent homology void detection
  4. matryoshka.rs -- multi-resolution embeddings
  5. community.rs -- Leiden community detection
  6. rl_routing.rs -- RL routing on receipts
  7. subgraph_pruning.rs -- lawful subgraph pruning
- MCP server v0.3.1: 30 tools, Candle embedder default, works with Hermes/Claude Code/Codex/Cursor
- Provenance semiring, temporal weight, compression governor, decoder/contradiction detection
- Integration wires connect all modules (topology+contradictions, community+compression, subgraph+lifecycle, matryoshka+routing)

## IMPROVEMENT TIERS

### Tier 1: High ROI, low effort (can do now)

#### 1A. Content-aware chunking
Current chunker is naive text splitting. SimpleMem research shows structured compression + consolidation gives large token reduction and F1 gains.
- Add sentence-boundary-aware chunking with overlap
- Add code-aware chunking (don't split inside function bodies)
- Add markdown-header-based chunking
- Estimated impact: 10-20% better recall on long documents, fewer malformed chunks

#### 1B. Search result dedup at the DB level
The dedup we added at the hook layer is a symptom fix. The DB should dedup search results before returning them.
- Add content-hash dedup in the search pipeline (before RRF fusion)
- If two chunks from different documents have identical content, keep only the highest-scoring one
- Estimated impact: eliminates the duplicate injection problem at the source, saves tokens on every search

#### 1C. Ingestion dedup guard in the library
The pre_tool_call hook catches duplicates at the MCP layer, but direct library users (Gloss, Recall, any Rust consumer) get no protection.
- Add content-hash check in `add_fact()` and `ingest_document()` before insert
- If duplicate content exists, return the existing ID instead of creating a new row
- Configurable: `MemoryConfig.dedup_on_write: bool` (default true)
- Estimated impact: prevents the 5% DB bloat from ever recurring

#### 1D. Search receipt improvements
Current search returns results but the receipt metadata is thin. The research corpus calls for:
- Per-result provenance: which retrieval signal contributed (BM25, vector, both)
- Confidence score decomposition: BM25 score, cosine similarity, RRF rank
- Degradation flags: was this result from a quantized vector? approximate?
- Estimated impact: agents can make better decisions about when to trust results

### Tier 2: Medium effort, high impact

#### 2A. Adaptive routing integration with the agent layer
The sm_route_query tool exists but the adaptive router skill (Hermes-side) is manual. The research says:
- Self-RAG (arxiv 2310.11511): retrieve-or-not gate based on confidence
- Adaptive-RAG (arxiv 2406.19387): A/B/C query complexity classifier
- FLARE (arxiv 2305.06983): forward-looking confidence threshold
- RAGRouter-Bench (arxiv 2602.00296): lightweight TF-IDF+SVM router achieves 93.2%

Action: Wire sm_route_query into the pre_llm_call hook so the agent automatically gets routed results, not just raw sm_search. The router should:
- Classify query complexity (A=simple lookup, B=multi-hop, C=contradiction, D=synthesis, E=temporal, F=creative)
- Select tools by class (A: search only, B/D: +discord+graph, C: +routing+decoder)
- Gate graph expansion behind corpus density >0.3
- Estimated impact: graph retrieval stops hurting simple lookups, complex queries get better results

#### 2B. BGE-M3 multi-function embeddings
Currently using nomic-embed-text (768d dense only). BGE-M3 produces dense + sparse + ColBERT multi-vector from a single model.
- Replace 3-model pipeline (BM25 + dense + separate ColBERT) with single BGE-M3
- RRF fusion becomes 3-way: dense + sparse + multi-vector
- Local-first: BGE-M3 runs via Ollama or ONNX
- The late_interaction.rs module already exists, just needs real BGE-M3 vectors
- Estimated impact: 5-15% recall improvement on long-tail queries, architectural simplification

#### 2C. SimpleMem-style semantic compression
The compression_governor.rs exists but only handles vector quantization. SimpleMem's three-stage pipeline adds:
- Structured compression: extract entities, relations, claims from raw text before storing
- Consolidation: merge near-duplicate facts across sessions
- Intent-aware retrieval planning: compress differently based on likely query patterns
- Estimated impact: 30-50% token reduction in stored content, better retrieval precision

#### 2D. Temporal decay and staleness tracking
temporal.rs exists but the temporal_weight is not used in search ranking. Research says:
- Stale facts (not accessed in 30+ days) should be scored lower
- Superseded facts should be filtered from search results (already done in hooks, should be in library)
- Contradicted facts should carry a degradation flag
- Estimated impact: search results favor current information, less noise from old data

### Tier 3: High effort, novel research contributions

#### 3A. Provenance-attributable token-level matching
ColBERT late interaction + provenance semiring = each token-level match carries confidence and source-span metadata. No existing system does this.
- late_interaction.rs exists but needs provenance integration
- Each MaxSim score contribution would carry: source document, chunk, token position, confidence
- Enables: "this specific sentence in the response is backed by this specific passage in the KB"
- Estimated impact: research contribution, enables trust-scored retrieval

#### 3B. MemRL self-evolving routing
rl_routing.rs exists but trains on synthetic data. MemRL proposes:
- Record actual retrieval outcomes as receipts
- Train routing policy on receipt replay data
- Separate frozen LLM from plastic memory policy
- Estimated impact: router gets better over time based on real usage, not benchmarks

#### 3C. Persistent homology guided exploration
topology.rs exists and computes Betti numbers. The novel combination:
- Use topological voids to guide retrieval: "explore toward the gap"
- Identify clusters that should be connected but aren't
- Trigger contradiction detection in structurally weak areas
- Estimated impact: system can identify what it doesn't know but should

#### 3D. AutoPrunedRetriever with lawful subtraction
subgraph_pruning.rs exists. The novel combination:
- Persist reasoning subgraphs across sessions
- Prune using provenance-preserving subtraction (not deletion)
- Formal guarantees: pruned graph preserves all provenance paths
- Estimated impact: graph stays manageable as it grows, no orphaned edges

### Tier 4: Infrastructure

#### 4A. Benchmark harness
The benchmark.rs module exists but has no real-world workload data.
- Build semantic-memory-bench: deterministic JSONL fixtures, recall@k, nDCG, MRR, p95 latency
- Run against the current 6815-fact / 877-doc / 15413-chunk DB
- Establish baseline numbers before changes, detect regressions
- Estimated impact: know if changes actually improve things

#### 4B. Multi-agent shared access
The DB is shared across Hermes + Codex + Claude Code. Current architecture has no locking for concurrent writes.
- SQLite WAL mode handles concurrent readers, but concurrent writers can conflict
- Add advisory locking or queue-based write coordination
- Estimated impact: no more "database is locked" errors

## PRIORITY ORDERING

1. 1C (ingestion dedup in library) -- prevents future bloat, 2-3 hours
2. 1B (search dedup in library) -- fixes root cause of duplicate results, 1-2 hours
3. 1A (content-aware chunking) -- improves recall immediately, 3-4 hours
4. 4A (benchmark harness) -- establishes baselines, 2-3 hours
5. 2A (adaptive routing in hooks) -- automates tool selection, 4-6 hours
6. 1D (search receipts) -- enables better agent decisions, 2-3 hours
7. 2D (temporal decay in ranking) -- favors current info, 2 hours
8. 2C (SimpleMem compression) -- big token reduction, 6-8 hours
9. 2B (BGE-M3) -- embedding quality, 4-6 hours
10. 3A-3D (novel research) -- longer term, each 8+ hours