# semantic-memory

Local-first hybrid semantic search backed by authoritative SQLite
state and a high-performance vector sidecar.

`semantic-memory` stores facts, chunked documents, conversation
messages, and searchable episodes in SQLite. Search combines
**BM25 (FTS5)** and **vector retrieval** with **Reciprocal Rank
Fusion**, and `search_explained()` returns the exact scoring
breakdown from the live pipeline.

The vector sidecar is **usearch 2.25** (default), with hnsw_rs 0.3
and brute-force as opt-in alternatives. All three implement the
`VectorBackend` trait.

## Why this crate

Most "vector databases" treat the index as authoritative and
the metadata as a side table. `semantic-memory` is the
opposite: **SQLite is authoritative** for all durable state
(records, embeddings, content, conversations, links). The
vector index is an **acceleration sidecar** that can be
rebuilt from SQLite at any time. If the sidecar corrupts, you
call `reconcile()` and you have a fresh index from
authoritative state.

This makes the crate suitable for local-first AI systems that
need:

- **Durable, recoverable state** — SQLite + WAL, one writer +
  pooled readers.
- **Fast vector search** — usearch 2.25 (default), hnsw_rs 0.3
  (opt-in), or brute-force (no C++ toolchain needed).
- **Hybrid search** — BM25 + vector + RRF, with the breakdown
  exposed via `search_explained()`.
- **Bitemporal truth** — every fact carries a `valid_time` and
  `recorded_time` via the [bitemporal-runtime](https://crates.io/crates/bitemporal-runtime)
  foundation.
- **Receipt-bearing operations** — every state transition
  emits a typed, blake3-digested receipt.
- **Provenance** — algebraic confidence scores (semiring-based)
  with support counts.
- **Contradiction detection** — syndrome detection and belief
  propagation on conflict graphs.
- **Adaptive routing** — query profiling and stage selection.
- **Lawful subtraction** — safe forgetting with invariant
  verification and recovery.
- **Typed graph edges** — durable, append-only edges (semantic,
  temporal, causal, entity) with invalidation.
- **Factor graph reasoning** — unified belief propagation over
  all four edge types.
- **Topological analysis** — Betti numbers and void detection.
- **Community detection** — Leiden-inspired with contradiction
  scanning.
- **Late-interaction RRF** — a third retrieval signal (proxy
  ColBERT MaxSim) fused alongside BM25 and vector.
- **Temporal scoring** — stale facts are automatically downranked
  via temporal weight in the RRF score.
- **Provenance-weighted scoring** — high-confidence facts rank
  higher via provenance confidence multiplier.
- **Namespace-weighted scoring** — configurable per-namespace boost
  in `SearchConfig`.
- **Self-RAG gating** — `should_retrieve()` skips search for
  greetings, confirmations, and trivial queries.
- **Embedding cache** — LRU cache (256 entries) skips model
  compute for repeated query texts.
- **Search result cache** — LRU cache (64 entries) returns identical
  queries instantly. Invalidated on any mutation.
- **Query expansion** — hyphenated variants (turbo-quant /
  turboquant) automatically matched in BM25.
- **Result diversity** — max 2 chunks per document prevents
  narrow single-source results.
- **Embedding similarity dedup** — Jaccard + cosine heuristic
  removes near-duplicate results.
- **SimpleMem compression** — `compress_search_results()`
  shortens result content to first sentence + key terms (opt-in
  via `SearchConfig.compress_results`).
- **Matryoshka 2-stage search** — 64d truncated embeddings for
  fast candidate generation, 768d exact rerank (opt-in via
  `SearchConfig.candidate_dims`).
- **RL routing feedback** — `record_routing_outcome()` feeds
  good/bad/neutral signals to the tabular routing policy.
- **BM25 tuning** — configurable `k1` and `b` parameters in
  `SearchConfig`.
- **TurboQuant codec** — optional compressed vector sidecar via
  the `turbo-quant-codec` feature flag.

## Quick start

```rust
use semantic_memory::{MemoryConfig, MemoryStore};

#[tokio::main]
async fn main() -> Result<(), semantic_memory::MemoryError> {
    let store = MemoryStore::open(MemoryConfig::default())?;

    // Store a fact.
    store.add_fact("general", "Rust was first released in 2015", None, None).await?;

    // Hybrid search (BM25 + vector + RRF).
    let results = store.search("when was Rust released", None, None, None).await?;
    for hit in &results {
        println!("  score={:.4}  {}", hit.score, hit.content);
    }

    // Get the exact scoring breakdown.
    let explained = store.search_explained("when was Rust released", None, None).await?;
    for hit in explained {
        println!(
            "  rrf={:.4}  bm25={:?}  vec={:?}  -> {}",
            hit.breakdown.rrf_score,
            hit.breakdown.bm25_score,
            hit.breakdown.vector_score,
            hit.result.content,
        );
    }
    Ok(())
}
```

`MemoryConfig::default()` uses the in-process Candle embedder
(pure-Rust, CPU-only) with `nomic-embed-text-v1.5` at 768 dimensions.
The model downloads automatically from HuggingFace on first use and
is cached in `~/.cache/huggingface/hub`. No external process or
server required.

To use Ollama instead, enable the `ollama` path by constructing
`OllamaEmbedder` explicitly (see below).

## Examples

The crate ships with several runnable examples in `examples/`:

### Basic search (`basic_search.rs`)

Creates a store, adds 5 facts across two namespaces, demonstrates
`search()`, `search_explained()`, `search_fts_only()`, `stats()`,
and `graph_view().neighbors()`. Uses `MockEmbedder` — no Ollama
or model download needed.

```bash
cargo run --example basic_search
```

### Conversation memory (`conversation_memory.rs`)

Creates a conversation session, stores system/user/assistant
messages, retrieves messages within a token budget, runs
`search_conversations()`. Uses `MockEmbedder` — no Ollama needed.

```bash
cargo run --example conversation_memory
```

### Hybrid retrieval recall gate (`hybrid_retrieval_recall_gate.rs`)

Measures end-to-end recall@10 for the hybrid BM25+vector+RRF
pipeline against a golden 30-document corpus and 12 ground-truth
queries. Emits a JSON receipt. Uses `MockEmbedder`.

```bash
cargo run --example hybrid_retrieval_recall_gate
```

### Real benchmark (`real_bench.rs`)

Inserts 500 facts across 10 namespaces with `MockEmbedder`
(384-dim), runs 50 benchmark queries comparing hybrid vs
FTS-only vs vector-only search. Reports avg latency, top-5
overlap analysis, and a feature comparison vs Qdrant.

```bash
cargo run --example real_bench
```

### Routing benchmark (`run_bench.rs`)

Runs the RAGRouter-Bench default benchmark using
`benchmark::run_default_benchmark()`. Prints a human-readable
report and a JSON reproducibility manifest.

```bash
cargo run --example run_bench --features benchmark
```

### TurboQuant benchmark (`turboquant_benchmark_gate.rs`)

Benchmarks the TurboQuant codec: encodes 1000 vectors at 8 bits,
runs 50 queries, measures recall@10, NDCG@10, rank drift, score
error, and latency. Writes a JSON summary with green/amber/red
classification.

```bash
cargo run --example turboquant_benchmark_gate --features turbo-quant-codec
```

## Embedding backends

`semantic-memory` supports three embedding backends:

### Candle (default with `candle-embedder` feature)

In-process pure-Rust ML framework (CPU-only). No external process
or server required. Downloads `nomic-embed-text-v1.5` from
HuggingFace on first use (cached after).

```rust
use semantic_memory::{MemoryConfig, MemoryStore};

// CandleEmbedder is the default when candle-embedder is enabled.
let store = MemoryStore::open(MemoryConfig::default())?;
```

### Ollama (external server)

If you prefer using an external Ollama instance (e.g. for GPU
acceleration or shared model serving):

```rust
use semantic_memory::{MemoryConfig, MemoryStore, embedder::OllamaEmbedder};

let config = MemoryConfig::default();
let embedder = Box::new(OllamaEmbedder::try_new(&config.embedding)?);
let store = MemoryStore::open_with_embedder(config, embedder)?;
```

Requires Ollama running with the model pulled:
```bash
ollama pull nomic-embed-text
```

### Mock (testing/CI)

Deterministic hash-based embeddings for tests and CI — no network,
no model download:

```rust
use semantic_memory::{MemoryConfig, MemoryStore, embedder::MockEmbedder};

let config = MemoryConfig {
    base_dir: std::env::temp_dir().join("sm-test"),
    ..Default::default()
};
let embedder = Box::new(MockEmbedder::new(768));
let store = MemoryStore::open_with_embedder(config, embedder)?;
```

## Storing and searching documents

```rust
// Ingest a document — it's automatically chunked and each chunk
// is embedded and indexed independently.
let doc_id = store
    .ingest_document("Tokio Tutorial", long_text, "docs", None, None)
    .await?;

// Search across facts AND document chunks.
let results = store
    .search("how does tokio scheduling work", None, None, None)
    .await?;

// Search only document chunks (filter by source type).
use semantic_memory::SearchSourceType;
let chunks = store
    .search(
        "tokio spawn",
        Some(5),
        None,
        Some(&[SearchSourceType::Chunks]),
    )
    .await?;
```

## Conversation memory

```rust
use semantic_memory::{MemoryStore, MemoryConfig, embedder::MockEmbedder, Role};

let store = MemoryStore::open_with_embedder(
    MemoryConfig { base_dir: dir, ..Default::default() },
    Box::new(MockEmbedder::new(768)),
)?;

// Create a conversation session.
let session_id = store.create_session("chat", None).await?;

// Store messages.
store.add_message(&session_id, Role::User, "What is RAG?", None).await?;
store.add_message(&session_id, Role::Assistant, "RAG stands for...", None).await?;

// Search conversation history.
let messages = store
    .search_conversations("what did we discuss about RAG", &session_id, Some(5))
    .await?;
```

## Graph edges

```rust
use semantic_memory::GraphEdgeType;

// Add a causal edge between two facts.
let edge = store
    .add_graph_edge(
        "fact:abc123-...",
        "fact:def456-...",
        GraphEdgeType::Causal {
            confidence: 0.85,
            evidence_ids: vec!["fact:ev1-...".to_string()],
        },
        1.0,
        None,
    )
    .await?;

// Add an entity edge with a relation name.
let edge = store
    .add_graph_edge(
        "fact:abc123-...",
        "namespace:rust-facts",
        GraphEdgeType::Entity { relation: "belongs_to".to_string() },
        1.0,
        None,
    )
    .await?;

// Add a historical edge with explicit bitemporal timestamps.
let historical = store
    .add_graph_edge_at(
        "fact:abc123-...",
        "fact:def456-...",
        GraphEdgeType::Causal {
            confidence: 0.70,
            evidence_ids: vec!["fact:ev-old".to_string()],
        },
        1.0,
        None,
        "2026-01-01 00:00:00.000000", // valid/domain time
        "2026-01-15 00:00:00.000000", // recorded/system time
    )
    .await?;

// List current edges for a node.
let edges = store
    .list_graph_edges_for_node("fact:abc123-...")
    .await?;

// Reconstruct the graph as of domain time + recorded/system time.
let as_of_edges = store
    .list_graph_edges_for_node_as_of(
        "fact:abc123-...",
        "2026-02-01 00:00:00.000000",
        "2026-02-01 00:00:00.000000",
    )
    .await?;

// Find the shortest path between two items.
let g = store.graph_view();
if let Some(path) = g.path("fact:abc123-...", "fact:xyz789-...", 5)? {
    println!("Path: {:?}", path);
}
```

## Provenance

```rust
use semantic_memory::provenance::{ConfidenceSemiring, ConfidenceValue, ProvenanceItemType};

// Set evidence confidence for a fact.
let value = ConfidenceValue::new(0.92, 3); // confidence=0.92, support_count=3
let receipt = store
    .set_provenance::<ConfidenceSemiring>(
        &ProvenanceItemType::Fact,
        "fact:abc123-...",
        &value,
        &[],
        None,
    )
    .await?;
println!("Provenance recorded: {}", receipt.provenance_id);
```

## Integrity and reconciliation

```rust
// Strict integrity check — surfaces malformed data, sidecar drift,
// invalid embeddings, broken FTS indexes.
let report = store.verify_integrity(None).await?;
if !report.errors.is_empty() {
    for err in &report.errors {
        eprintln!("Integrity error: {err}");
    }
}

// Rebuild everything from authoritative SQLite state.
store.reconcile(None).await?;
```

## HNSW to usearch migration (June 2026)

The vector sidecar was migrated from `hnsw_rs 0.3` to
`usearch 2.25` based on a head-to-head benchmark on
**2026-06-02**.

### Headline @ D=768 (production case)

| Metric | hnsw_rs 0.3 | usearch 2.25 | Advantage |
|---|---:|---:|---:|
| **Insert throughput** | 265 vec/s | 770 vec/s | **2.9x** |
| **Search p50** | 9,992 us | 529 us | **18.9x** |
| **Search p99** | 54,110 us | 692 us | **78x** |
| **Search mean** | 14,524 us | 538 us | **27x** |
| **Recall@10** | 0.885 | 0.925 | **+4 pp** |
| **Load time** | 34,484 ms | 11 ms | **3,134x** |
| **p99/p50 ratio** | 5.4x | 1.3x | usearch stable |

The key wins: 78x better search p99 (hnsw_rs had pathological tail
behavior), 3,134x faster load (hnsw_rs re-runs slow on-disk decode),
and +4pp recall@10 at production scale.

### Reproduce the benchmark

```bash
cargo build -p hnsw-bench --bin hnsw-bench \
    --no-default-features --features hnsw --release
./target/release/hnsw-bench            # hnsw_rs run

cargo build -p hnsw-bench --bin hnsw-bench \
    --no-default-features --features usearch-backend --release
./target/release/hnsw-bench            # usearch run
```

Receipts: `hnsw-bench-receipt-{hnsw_rs,usearch}-20260602-*.json`.

### Choosing a backend

```toml
# Default — usearch 2.25 (recommended)
semantic-memory = "0.5"

# Legacy — hnsw_rs 0.3 (opt-in)
semantic-memory = { version = "0.5", default-features = false, features = ["hnsw"] }

# No C++ toolchain — pure-Rust brute-force
semantic-memory = { version = "0.5", default-features = false, features = ["brute-force"] }
```

## What's in the box

### Storage

- **SQLite + WAL** — authoritative for all durable state.
  One writer connection + pooled reader connections.
- **FTS5** — BM25 full-text search over content, episode
  titles, message bodies.
- **Vector sidecar** — usearch 2.25 (default), hnsw_rs 0.3
  (opt-in), or brute-force (opt-in). All implement the
  `VectorBackend` trait. Pending sidecar mutations are
  journaled in SQLite and replayed on open / flush / rebuild
  / reconcile.
- **Bitemporal truth** — every fact carries a `valid_time` and
  `recorded_time` via the [bitemporal-runtime](https://crates.io/crates/bitemporal-runtime)
  foundation.
- **Typed graph edges** — durable, append-only edges with
  invalidation and bitemporal `valid_time` / `recorded_time`
  reconstruction. Four edge types: semantic, temporal, causal,
  entity. Stored in the `graph_edges` SQLite table.

### Search

- **`search()`** — hybrid (BM25 + vector + RRF) over facts,
  document chunks, and episodes by default.
- **`search_explained()`** — same as `search()` but with the
  per-signal scores exposed (BM25, vector, recency, RRF,
  weights, contributions).
- **`search_conversations()`** — message-level retrieval.
- **`search_fts_only()`** — BM25 only, no vector path.
- **`search_vector_only()`** — vector only, no BM25 path.
- **`reconcile()`** — rebuild FTS, re-embed, rebuild the
  sidecar from authoritative SQLite state.

### Integrity

- **`verify_integrity()`** — strict check for malformed stored
  data (invalid roles, JSON, enums, embedding blobs, quantized
  blobs, sidecar drift). Surfaces errors instead of silently
  converting to defaults.
- **Strict deserialization** — invalid stored data is an
  error, not a fallback.

### Graph

- **`store.graph_view()`** — deterministic traversal over
  namespaces, facts, documents, chunks, sessions, messages,
  episodes, and semantic/temporal/causal/entity edges derived
  from SQLite state.
- **`add_graph_edge()` / `add_graph_edge_at()`** — add typed
  edges with per-type metadata (cosine_similarity, delta_secs,
  confidence, relation). Normal insertion defaults bitemporal
  fields to the insertion timestamp; `add_graph_edge_at()` imports
  historical relationships with explicit valid/recorded timestamps.
  Insertion remains idempotent.
- **`list_graph_edges_for_node_as_of()`** — reconstruct a node's
  relationship set at explicit domain/recorded-time cutoffs,
  including edges that were invalidated after the as-of recorded
  time.
- **`list_graph_edges()` / `invalidate_graph_edge()`** —
  append-only edge lifecycle. Edges are never deleted, only
  invalidated with a reason.

### Receipts

- Every state transition (add_fact, search, reconcile,
  add_graph_edge, set_provenance, ...) emits a typed receipt.
  Receipts are content-addressed (blake3-digested) and
  reproducible. The `SearchContext` struct carries audit
  metadata (request_id, trace_id, attempt_family_id,
  replay_of) for full replay support.

## Cargo features

| Feature | Default | What it enables |
|---|---|---|
| `usearch-backend` | yes | usearch 2.25 vector backend (C++ cxx-bridge) |
| `candle-embedder` | no | In-process Candle embedder (pure-Rust, CPU-only, no Ollama required) |
| `hnsw` | no | hnsw_rs 0.3 vector backend (legacy, opt-in) |
| `brute-force` | no | Pure-Rust brute-force backend (no ANN, no C++ needed) |
| `provenance` | no | Semiring provenance (Boolean, Tropical, Probability, Confidence) |
| `temporal` | no | Temporal weight scoring (requires provenance) |
| `multiscale` | no | Multiscale retrieval scheduling pipeline |
| `discord` | no | Second-order graph-neighbor retrieval |
| `decoder` | no | Syndrome detection + belief propagation contradiction correction |
| `subtraction` | no | Lawful forgetting with invariant verification |
| `compression-governor` | no | Importance-driven quantization level decisions |
| `routing` | no | Adaptive query-aware retrieval stage selection |
| `benchmark` | no | Benchmark harness for routing (requires routing) |
| `integration` | no | Cross-feature wiring (requires all constituent features) |
| `late-interaction` | no | ColBERT-style late interaction multi-vector retrieval |
| `topology` | no | Persistent homology and topological void detection |
| `matryoshka` | no | Matryoshka Representation Learning (multi-resolution embeddings) |
| `community` | no | Leiden community detection with contradiction tracking |
| `rl-routing` | no | MemRL-style RL routing over receipts (requires routing) |
| `subgraph-pruning` | no | Reasoning subgraph pruning with lawful subtraction (requires subtraction) |
| `turbo-quant-codec` | no | TurboQuant codec integration for compressed vector search |
| `admin-ops` | no | Admin-only hard delete/update of truth-bearing rows |
| `testing` | no | Internal testing utilities |

The `integration` feature requires all constituent features
(provenance, temporal, multiscale, discord, decoder, subtraction,
compression-governor, routing, topology, community,
subgraph-pruning, matryoshka) and wires cross-feature bridges:
routing to decoder, decoder to subtraction, provenance to
temporal, subtraction to compression, discord to provenance.

The `turbo-quant-codec` feature is an opt-in experimental codec
integration that adds [turbo-quant](https://crates.io/crates/turbo-quant),
[quant-governor](https://crates.io/crates/quant-governor), and
[scr-runtime-compression](https://crates.io/crates/scr-runtime-compression)
as dependencies. It enables approximate candidate generation with
exact f32 rerank. Not needed for standard search — usearch already
handles vector indexing.

## Public API surface

### Core types

| Type | Description |
|------|-------------|
| `MemoryStore` | The main store handle. All operations go through this. |
| `MemoryConfig` | Configuration (base_dir, embedding, search, chunking, pool, limits). |
| `EmbeddingConfig` | Embedding backend config (model, dimensions, batch_size, timeout, ollama_url for Ollama). |
| `SearchConfig` | BM25/vector weights, RRF k, recency, candidate pool, derived backend policy. |
| `SearchResult` | One search hit: content, source, score, bm25_rank, vector_rank, cosine_similarity. |
| `ExplainedResult` | SearchResult + ScoreBreakdown with all per-signal scores. |
| `SearchSource` | Enum: Fact, Chunk, Message, Episode, Projection — tells you what type of thing the result is. |
| `GraphEdgeType` | Enum: Semantic, Temporal, Causal, Entity — with per-type metadata. |
| `GraphView` | Deterministic graph traversal over the store's state. |
| `MemoryStats` | Facts, documents, chunks, sessions, messages counts + DB size + embedding info. |
| `VectorIndex` | The vector sidecar handle (usearch/hnsw/brute-force). |

### Embedder

| Type | Description |
|------|-------------|
| `Embedder` | Trait for embedding providers. |
| `CandleEmbedder` | In-process pure-Rust embedder (CPU-only, no Ollama). Feature `candle-embedder`. |
| `OllamaEmbedder` | External — calls Ollama's /api/embed endpoint. |
| `MockEmbedder` | Deterministic embedder for tests/CI. No network. |

### Error

| Type | Description |
|------|-------------|
| `MemoryError` | The unified error type for all operations. |

### Modules (feature-gated)

| Module | Feature | What it provides |
|--------|---------|-----------------|
| `provenance` | `provenance` | ConfidenceSemiring, BooleanSemiring, TropicalSemiring, ProbabilitySemiring, ConfidenceValue, ProvenanceItemType |
| `temporal` | `temporal` | Temporal weight computation (age, supersession, support, contradiction) |
| `pipeline` | `multiscale` | Multiscale retrieval scheduling (staged search with budgets) |
| `discord` | `discord` | DiscordScorer, GraphEdgeRef — second-order retrieval |
| `decoder` | `decoder` | detect_syndromes, compute_correction, pass_messages, ConflictGraph |
| `subtraction` | `subtraction` | SubtractionCandidate, invariant verification |
| `compression_governor` | `compression-governor` | decide_quantization, QuantizationLevel, ImportanceConfig |
| `routing` | `routing` | RetrievalRouter, QueryProfile, RoutingDecision |
| `benchmark` | `benchmark` | run_default_benchmark, BenchmarkReport |
| `integration` | `integration` | plan_execution, corrections_to_subtraction_candidates, should_trigger_recompression, autonomous_subgraph_maintenance |
| `factor_graph` | `integration` | FactorGraph, FactorGraphConfig, factors_from_edges |
| `topology` | `topology` | compute_betti_numbers, find_voids, gap_report |
| `matryoshka` | `matryoshka` | MatryoshkaConfig, multi_resolution_route |
| `community` | `community` | detect_communities, community_contradiction_scan, community_aware_compression |
| `subgraph_pruning` | `subgraph-pruning` | AccessLog, autonomous_subgraph_maintenance |
| `rl_routing` | `rl-routing` | RL-trained routing policy |
| `late_interaction` | `late-interaction` | ColBERT-style multi-vector retrieval |

## MSRV

Rust 1.75 (2021 edition). The `usearch` cxx-bridge requires
C++17 to build, which is a documented `build.rs` prerequisite.

## Test coverage

- **321 tests** pass with default features (`cargo test`).
  **500+ tests** pass with all features enabled.
- Tests cover: SQLite schema, WAL concurrency, FTS5 rebuild,
  usearch backend (insert, search, save, load, hot-swap), hnsw
  backend (opt-in), bitemporal as-of queries, hybrid search
  with score breakdown, receipt emission, integrity checks,
  graph view traversal, provenance (all four semirings), temporal
  weight, decoder syndromes, subtraction, compression governor,
  routing, discord, graph edges, factor graph, topology,
  community, and cross-feature integration.

## Dependencies

### Runtime

| Crate | Purpose |
|-------|---------|
| `rusqlite` | SQLite + FTS5 bindings (bundled) |
| `usearch` | Vector sidecar (default backend) |
| `reqwest` | HTTP client for Ollama embeddings (when using OllamaEmbedder) |
| `blake3` | Content-addressed receipts |
| `tokio` | Async runtime |
| `serde` / `serde_json` | Serialization |
| `chrono` | Timestamps |
| `uuid` | ID generation |
| `tracing` | Structured logging |
| `thiserror` | Error derive |
| `bytemuck` | Zero-copy byte conversions |
| `schemars` | JSON Schema generation |

### Stack crates (from the same monorepo)

| Crate | Purpose |
|-------|---------|
| [stack-ids](https://crates.io/crates/stack-ids) | Typed IDs, scopes, trace context, BLAKE3 digests |
| [bitemporal-runtime](https://crates.io/crates/bitemporal-runtime) | Bitemporal truth primitives |
| [boundary-compiler](https://crates.io/crates/boundary-compiler) | RFC 8785 JSON Canonicalization (JCS) |
| [forge-memory-bridge](https://crates.io/crates/forge-memory-bridge) | Projection import transforms |

## Where it's used

`semantic-memory` is the search engine for:

- [semantic-memory-mcp](https://crates.io/crates/semantic-memory-mcp)
  — MCP server exposing 48 tools (33 lean / 39 standard / 48 full) and 14 HTTP endpoints for agent
  integration (Hermes Agent, Claude Desktop, Cursor, Windsurf).
- The LLM agent stack (forge-pilot, llm-pipeline) — every
  retrieval over a knowledge base.
- The LLM tool runtime — long-term tool-call memory.
- The verification runtime — fact storage with bitemporal
  truth.
- `fib-quant`, `turbo-quant`, `quant-eval` — recall measurements
  in benchmarks run through `semantic-memory::search` against the
  raw-vector baseline.

Any system that needs **local-first, hybrid, bitemporal,
receipt-bearing** search can adopt `semantic-memory`
directly.

**No cloud dependencies.** Every component runs locally: SQLite
for storage, usearch for vector search, Candle for embeddings (or
Ollama if you choose). No calls to OpenAI, Anthropic, Pinecone,
Weaviate, Supabase, or any hosted service. The only network call
is the one-time HuggingFace model download (cached after) when
using Candle, or your local Ollama instance when using Ollama.
Your data never leaves your machine.

## Scope and limits

- The crate is a library, not a server. The MCP server
  ([semantic-memory-mcp](https://crates.io/crates/semantic-memory-mcp))
  wraps it for agent integration.
- Requires an embedding backend. With the `candle-embedder` feature,
  `MemoryStore::open()` defaults to `CandleEmbedder` (in-process,
  pure-Rust, CPU-only, no Ollama). Without it, defaults to
  `OllamaEmbedder`. Use `MockEmbedder` for tests.
- The `integration` feature wires cross-feature bridges
  (routing to decoder, decoder to subtraction, provenance to
  temporal, subtraction to compression, discord to provenance)
  but the decoder does not yet re-rank search results in the
  live search path. The factor graph runs independently.
- Graph-based reasoning tools (discord, factor graph, topology,
  community) require stored graph edges to produce meaningful
  results. With zero edges they return empty results — not
  broken, just no graph to work with.
- The `turbo-quant-codec` and `admin-ops` features are opt-in
  and not enabled by default. `turbo-quant-codec` adds external
  codec dependencies; `admin-ops` enables hard delete/update
  operations that bypass supersession.

## License

Apache-2.0. See `LICENSE` for the full text.

## Changelog

See `CHANGELOG.md` for the release history.

## Acknowledgments

The HNSW to usearch migration was a 2-day investigation that
included the full benchmark harness, the `VectorBackend`
trait refactor, the default switch, and the sidecar-format
migration. The benchmark receipts (machine fingerprint, git
commit, full per-row payload) are in
`hnsw-bench-receipt-{hnsw_rs,usearch}-20260602-*.json` for
independent verification.