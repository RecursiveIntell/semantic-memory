# AGENTS.md — Semantic-Memory 7-Combination Enhancement Pass

## Present-tense codebase facts

### Workspace layout

- `semantic-memory/` — Core crate. 47 source files, 539 tests passing (all features).
- `semantic-memory/src/factor_graph.rs` — NEW: Factor graph unification (Combination 1, COMPLETE).
- `semantic-memory/src/decoder.rs` — BP on contradiction graphs (ConflictGraph + pass_messages).
- `semantic-memory/src/graph_edges.rs` — Stored graph edges (V27 table, 4 edge types).
- `semantic-memory/src/routing.rs` — Adaptive retrieval routing (QueryProfile + RoutingDecision).
- `semantic-memory/src/compression_governor.rs` — Per-vector importance scoring.
- `semantic-memory/src/integration.rs` — 15 FEUT-inspired cross-feature wires.
- `semantic-memory/src/vector_codec.rs` — TurboQuant codec profile + artifact boundary.
- `semantic-memory/src/search.rs` — Hybrid BM25+vector+RRF search pipeline.
- `semantic-memory-mcp/` — MCP server with 15 tools.

### Feature flags

```
default = ["usearch-backend"]
turbo-quant-codec = [dep:turbo-quant, dep:quant-governor, dep:scr-runtime-compression]
provenance, temporal, multiscale, discord, decoder, subtraction,
compression-governor, routing, benchmark, integration
integration = [provenance, temporal, multiscale, discord, decoder,
               subtraction, compression-governor, routing]
```

### What NOT to do

- Do NOT refactor, rename, or reorganize existing code beyond the specific additions.
- Do NOT add new dependencies without checking workspace Cargo.toml first.
- Do NOT change existing function signatures.
- Do NOT modify tests that are already passing.
- Do NOT touch factor_graph.rs — it is COMPLETE and tested.
- Do NOT break the `default` feature set — `cargo test` (no features) must still pass.
- Do NOT use `unwrap`, `expect`, `todo!`, `unimplemented!` in production code.
- The patch tool's standalone lint on Rust emits false `async fn Rust 2015` errors.
  `cargo check --manifest-path Cargo.toml` is the source of truth.

### Verification commands

```bash
cd /home/sikmindz/Coding/Libraries/semantic-memory

# All features (must pass with 539+ tests)
ALL="provenance,temporal,multiscale,discord,decoder,subtraction,compression-governor,routing,benchmark,integration,turbo-quant-codec"
cargo test --features "$ALL" 2>&1 | grep "test result:"

# Default features (must still pass)
cargo test 2>&1 | grep "test result:"

# Check compilation
cargo check --features "$ALL" 2>&1 | tail -3
```

══════════════════════════════════════════════
COMBINATION 2: ColBERT Late Interaction as 3rd RRF Signal
══════════════════════════════════════════════

## Goal

Add ColBERT-style late interaction multi-vector retrieval as a third RRF signal alongside BM25 and dense vector search. Enable token-level provenance attribution on match results.

## Files to create

- `semantic-memory/src/late_interaction.rs` — ColBERT-style per-token MaxSim scoring

## Files to modify

- `semantic-memory/src/search.rs` — Add late_interaction as 3rd RRF signal when feature enabled
- `semantic-memory/src/lib.rs` — Register new module
- `semantic-memory/Cargo.toml` — Add `late-interaction` feature flag
- `semantic-memory/src/config.rs` — Add late interaction config fields

## Architecture

1. New `late_interaction` feature flag (additive, opt-in)
2. `LateInteractionIndex` struct that stores per-token embeddings (N vectors per document)
3. `MaxSim` scoring: for each query token, find max similarity with all document tokens, sum across query tokens
4. Token-level provenance: each match contribution carries (query_token, doc_token, doc_id, source_span, similarity)
5. RRF fusion: late_interaction_rank joins bm25_rank and vector_rank in the RRF formula
6. Config: `LateInteractionConfig { enabled, max_tokens_per_doc, similarity_threshold }`

## Constraints

- Must NOT change existing search() signature — late interaction is an additive RRF signal
- Must NOT require a new embedding model — reuse existing Ollama embedder for per-token embeddings
- Token-level provenance must be optional (only when provenance feature enabled)
- Must degrade gracefully when late interaction index is empty

## Acceptance gates

- [ ] `cargo test --features "late-interaction"` passes with new tests
- [ ] `cargo test` (default features) still passes
- [ ] Late interaction adds a third RRF signal without breaking existing hybrid search
- [ ] Token-level match receipts are produced when provenance feature is also enabled

══════════════════════════════════════════════
COMBINATION 3: Persistent Homology Void Detection
══════════════════════════════════════════════

## Goal

Add persistent homology (topological data analysis) to identify structural gaps/voids in the knowledge graph. Combined with discord retrieval and contradiction detection, this enables "topology-aware knowledge gap detection" — the system can discover what it should know but doesn't.

## Files to create

- `semantic-memory/src/topology.rs` — Persistent homology computations on graph

## Files to modify

- `semantic-memory/src/lib.rs` — Register new module
- `semantic-memory/Cargo.toml` — Add `topology` feature flag
- `semantic-memory/src/integration.rs` — Add wire 16: topology + discord + contradiction

## Architecture

1. New `topology` feature flag (additive, opt-in)
2. Pure-Rust implementation of Betti numbers and persistence diagrams (NO external dependencies — implement from scratch using the graph adjacency matrix)
3. `compute_betti_numbers(adjacency: &HashMap<String, Vec<String>>) -> BettiNumbers` — Betti_0 (connected components), Betti_1 (cycles/loops)
4. `find_voids(graph_edges: &[StoredGraphEdge]) -> Vec<TopologicalVoid>` — identify regions where connections are missing
5. `Void` struct: { description, nearby_items, suggested_connections, void_type (MissingLink, MissingContext, ContradictionGap) }
6. Integration wire 16: `topology_aware_gap_detection(voids, discord_results, contradictions) -> Vec<GapReport>` — combines topology + discord + contradictions to identify knowledge gaps

## Constraints

- NO new dependencies — implement Betti number computation from scratch in pure Rust
- Must work on the existing graph_edges table (86+ edges)
- Must be O(n^2) or better for the graph sizes involved (<1000 edges)
- Must produce human-readable gap descriptions

## Acceptance gates

- [ ] `cargo test --features "topology,integration"` passes with new tests
- [ ] `cargo test` (default features) still passes
- [ ] Betti numbers computed correctly on test graphs (verified against known examples)
- [ ] Void detection finds at least one gap in a graph with known missing edges

══════════════════════════════════════════════
COMBINATION 4: Matryoshka Representation Learning + TurboQuant
══════════════════════════════════════════════

## Goal

Add Matryoshka Representation Learning (MRL) support: nested embeddings where vector prefixes are valid representations. Combined with TurboQuant, this enables multi-resolution retrieval where the router selects both compression level AND embedding dimensionality.

## Files to create

- `semantic-memory/src/matryoshka.rs` — MRL embedding truncation and multi-resolution search

## Files to modify

- `semantic-memory/src/search.rs` — Add multi-resolution candidate generation path
- `semantic-memory/src/routing.rs` — Add dimensionality selection to routing decision
- `semantic-memory/src/lib.rs` — Register new module
- `semantic-memory/Cargo.toml` — Add `matryoshka` feature flag
- `semantic-memory/src/config.rs` — Add MRL config fields

## Architecture

1. New `matryoshka` feature flag (additive, opt-in)
2. `MatryoshkaConfig { dimensions: Vec<usize>, candidate_dim, exact_dim }` — e.g. [64, 256, 768]
3. `truncate_embedding(embedding: &[f32], target_dim: usize) -> Vec<f32>` — extract prefix
4. Multi-resolution search: 64-dim candidates via TurboQuant compressed vectors → 768-dim exact rerank
5. Router extension: `RoutingDecision` gets `embedding_dim: usize` field
6. Router logic: high-stakes query → full 768-dim; quick lookup → 64-dim candidates + 256-dim rerank
7. TurboQuant integration: compressed vectors at each resolution level

## Constraints

- Must NOT require re-embedding existing facts — MRL truncation works on existing 768-dim embeddings
- Must NOT change existing search() signature
- TurboQuant integration must go through the existing `turbo-quant-codec` feature
- Router changes must be backward-compatible (new field defaults to full dimension)

## Acceptance gates

- [ ] `cargo test --features "matryoshka,routing,turbo-quant-codec"` passes with new tests
- [ ] `cargo test` (default features) still passes
- [ ] Truncated embeddings produce valid search results (lower recall at lower dims, but non-zero)
- [ ] Router selects appropriate dimensionality based on query profile

══════════════════════════════════════════════
COMBINATION 5: Leiden Community Detection
══════════════════════════════════════════════

## Goal

Add Leiden community detection on the graph_edges table to identify knowledge communities. Combined with provenance and contradiction detection, this enables hierarchical knowledge compression with contradiction tracking at every community level.

## Files to create

- `semantic-memory/src/community.rs` — Leiden algorithm implementation + community-aware compression

## Files to modify

- `semantic-memory/src/integration.rs` — Add wire 17: community + provenance + contradiction + compression
- `semantic-memory/src/lib.rs` — Register new module
- `semantic-memory/Cargo.toml` — Add `community` feature flag

## Architecture

1. New `community` feature flag (additive, opt-in)
2. Pure-Rust Leiden algorithm implementation (NO external dependencies)
3. `detect_communities(edges: &[StoredGraphEdge]) -> Vec<Community>` — Leiden clustering
4. `Community { id, members: Vec<String>, level: usize, parent: Option<String> }`
5. Hierarchical: communities at multiple levels of granularity
6. `community_contradiction_scan(communities, contradictions) -> Vec<CommunityContradiction>` — check for intra-community conflicts
7. `community_aware_compression(communities, importance_scores) -> Vec<CompressionDecision>` — tight communities with high provenance → aggressive compression; isolated facts → full precision
8. Integration wire 17 connecting community detection to compression governor

## Constraints

- NO new dependencies — implement Leiden from scratch in pure Rust
- Must work on existing graph_edges (86+ edges, 6900+ facts)
- Must be deterministic (fixed seed for Leiden randomness)
- Community detection must be O(n*m) or better where n=nodes, m=edges

## Acceptance gates

- [ ] `cargo test --features "community,integration"` passes with new tests
- [ ] `cargo test` (default features) still passes
- [ ] Communities detected on test graph match expected clustering
- [ ] Community-aware compression produces different quantization levels than uniform compression

══════════════════════════════════════════════
COMBINATION 6: MemRL-Style RL Routing on Receipt Replay
══════════════════════════════════════════════

## Goal

Use receipt-driven replay data as training signal for an RL-trained retrieval routing policy. The system's receipts (query, routing decision, results, outcome) become training data for improving routing decisions over time.

## Files to create

- `semantic-memory/src/rl_routing.rs` — RL routing policy trained on receipt replay

## Files to modify

- `semantic-memory/src/routing.rs` — Add `RLL routing mode` alongside heuristic routing
- `semantic-memory/src/integration.rs` — Add wire 18: receipts → RL training → routing
- `semantic-memory/src/lib.rs` — Register new module
- `semantic-memory/Cargo.toml` — Add `rl-routing` feature flag

## Architecture

1. New `rl-routing` feature flag (additive, opt-in, depends on `routing`)
2. `RoutingPolicy` struct: a simple tabular/linear policy mapping QueryProfile features → stage selection probabilities
3. `ReceiptOutcome` extraction: parse search receipts to get (query_profile, routing_decision, outcome_score)
4. `update_policy(policy, receipt_batch) -> RoutingPolicy` — simple gradient-free update (reward = outcome_score - baseline)
5. `route_with_rl(policy, profile) -> RoutingDecision` — use learned policy instead of heuristic
6. Fallback: if policy is untrained (no receipts), fall back to heuristic routing
7. Integration wire 18: `receipt_replay_to_training_data(receipts) -> TrainingBatch`

## Constraints

- NO deep learning dependencies — use simple tabular or linear policy
- Must NOT change existing RoutingDecision struct
- Must degrade gracefully to heuristic routing when untrained
- Training must be incremental (update from receipt batches, not full retraining)

## Acceptance gates

- [ ] `cargo test --features "rl-routing,routing,integration"` passes with new tests
- [ ] `cargo test` (default features) still passes
- [ ] RL routing falls back to heuristic when untrained
- [ ] After training on synthetic receipts, RL routing produces different decisions than heuristic

══════════════════════════════════════════════
COMBINATION 7: AutoPrunedRetriever-Style Reasoning Subgraph Pruning
══════════════════════════════════════════════

## Goal

Apply lawful subtraction to reasoning subgraphs, not just individual facts. The KB automatically maintains its own relevance by pruning unused reasoning paths while preserving their reconstruction history via receipts.

## Files to create

- `semantic-memory/src/subgraph_pruning.rs` — Reasoning subgraph identification and pruning

## Files to modify

- `semantic-memory/src/integration.rs` — Add wire 19: subgraph pruning + lawful subtraction + compression
- `semantic-memory/src/lib.rs` — Register new module
- `semantic-memory/Cargo.toml` — Add `subgraph-pruning` feature flag (depends on `subtraction`)

## Architecture

1. New `subgraph-pruning` feature flag (additive, opt-in, depends on `subtraction`)
2. `ReasoningSubgraph { root: String, nodes: Vec<String>, edges: Vec<(String, String)>, access_count: usize, last_accessed: String }`
3. `identify_subgraphs(edges: &[StoredGraphEdge], access_logs: &[AccessLog]) -> Vec<ReasoningSubgraph>` — find connected subgraphs that form reasoning paths
4. `prune_subgraph(subgraph, subtraction_engine) -> PruningReceipt` — apply lawful subtraction to entire subgraph
5. Pruning preserves: contradiction history, provenance chain, reconstruction receipts
6. `reconstruct_subgraph(receipt) -> Option<ReasoningSubgraph>` — rebuild from receipts if needed
7. Integration wire 19: `autonomous_subgraph_maintenance(subgraphs, access_logs, subtraction, compression) -> MaintenanceReport`

## Constraints

- Must NOT delete any fact without a subtraction receipt
- Must preserve contradiction history even for pruned subgraphs
- Must be reversible — pruned subgraphs can be reconstructed from receipts
- Access frequency determines pruning priority (least-accessed first)

## Acceptance gates

- [ ] `cargo test --features "subgraph-pruning,subtraction,integration"` passes with new tests
- [ ] `cargo test` (default features) still passes
- [ ] Pruning produces receipts for every removed node
- [ ] Reconstruction from receipts produces the original subgraph

══════════════════════════════════════════════
TURBO-QUANT ENABLEMENT AUDIT
══════════════════════════════════════════════

## Goal

Audit every place in semantic-memory where TurboQuant could be useful but is not currently wired in. Enable it everywhere it adds value.

## Current state

- `turbo-quant-codec` feature exists in Cargo.toml
- `vector_codec.rs` has `TurboQuantCodec` implementation
- `search.rs` has `turbo_quant_vector_outcome()` for TurboQuant candidate search
- `compression_governor.rs` does NOT use turbo-quant (it only scores importance, doesn't compress)
- `factor_graph.rs` (new) does NOT use turbo-quant
- `integration.rs` confidence_aware_quantization produces recommendations but does NOT apply them via turbo-quant

## Files to audit and modify

1. `compression_governor.rs` — Wire TurboQuant as the compression backend for low-importance vectors
2. `integration.rs` — `confidence_aware_quantization()` should produce TurboQuant-compatible recommendations
3. `factor_graph.rs` — `factor_graph_quantization()` should route through TurboQuant when available
4. `search.rs` — Ensure TurboQuant candidate generation is the default when `turbo-quant-codec` is enabled
5. `config.rs` — Add a `turbo_quant_default` config option

## Acceptance gates

- [ ] `cargo test --features "turbo-quant-codec,compression-governor,integration"` passes
- [ ] TurboQuant is used as the compression backend when both features are enabled
- [ ] Factor graph quantization recommendations are TurboQuant-compatible
- [ ] No regression in search quality when TurboQuant is enabled

══════════════════════════════════════════════
CRATE INTEGRATION SURVEY
══════════════════════════════════════════════

## Goal

Survey all crates in the Libraries monorepo and identify how semantic-memory can integrate with them.

## Crates to survey

- `turbo-quant/` — Codec integration (already partially done)
- `fib-quant/` — Geometry-preserving compression integration
- `poly-kv/` — KV cache pool integration
- `stack-ids/` — Identity primitives (already used)
- `forge-memory-bridge/` — Bridge integration (already used)
- `boundary-compiler/` — Structured output hardening
- `bitemporal-runtime/` — Bitemporal truth integration
- `quant-governor/` — Codec policy layer
- `scr-runtime-compression/` — Compression runtime
- `llm-pipeline/` — LLM execution pipeline
- `llm-tool-runtime/` — Tool execution runtime
- `knowledge-runtime/` — Query classification and routing
- `forge-pilot/` — Execution pilot
- `claim-ledger/` — Claim/evidence ledger (spec only, not built)

## Output

Produce a crate integration map as a markdown file at `semantic-memory/docs/CRATE_INTEGRATION_MAP.md` with:
- For each crate: current integration status (none/partial/full), integration opportunities, required changes, priority
- Dependency order for integration
- Conflict risks (shared types, duplicate functionality)

## Acceptance gates

- [ ] Integration map exists and covers all crates listed above
- [ ] Each entry has a clear priority (P0/P1/P2) and estimated effort
- [ ] No conflicts identified between proposed integrations