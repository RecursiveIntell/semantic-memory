# semantic-memory

[![Crates.io](https://img.shields.io/crates/v/semantic-memory.svg)](https://crates.io/crates/semantic-memory)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Local-first hybrid semantic search for Rust, built for AI agents.

`semantic-memory` combines SQLite (FTS5) and HNSW approximate nearest-neighbor search into one embedded crate. It handles facts, documents, conversation history, and causal episodes without a separate vector database.

## Features

- **Hybrid search** — BM25 full-text + vector similarity fused with exact, explainable Reciprocal Rank Fusion (RRF)
- **HNSW ANN** — approximate nearest-neighbor sidecar with brute-force fallback and SQLite-backed recovery
- **Fact storage** — namespaced key-value facts with embeddings
- **Document ingestion** — chunking with overlap, per-chunk embeddings
- **Conversation memory** — sessions, messages, token budgets, BM25 search, vector search
- **Episode tracking** — searchable causal records with verification status and outcomes
- **Explainable results** — score breakdowns from the real search pipeline, including weights, ranks, rerank state, and recency
- **Integrity & repair** — verify, reconcile, detect malformed rows, repair FTS drift, replay/rebuild HNSW, re-embed
- **Graph view** — derived graph traversal across namespaces, facts, documents, chunks, sessions, messages, episodes, and semantic links
- **Embedding quantization** — int8 quantization helpers for compact storage
- **Resource limits** — caps on namespace size, content length, embedding concurrency
- **Connection pooling** — one writer + configurable pooled readers under SQLite WAL mode

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
semantic-memory = "0.4"
tokio = { version = "1", features = ["rt", "macros"] }
```

### Hybrid Search (requires Ollama)

```rust
use semantic_memory::{MemoryConfig, MemoryStore};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryStore::open(MemoryConfig {
        base_dir: PathBuf::from("./my-memory"),
        ..Default::default()
    })?;

    // Store facts in namespaces
    store.add_fact("general", "Rust was first released in 2015", None, None).await?;
    store.add_fact("general", "Python is popular for data science", None, None).await?;
    store.add_fact("user", "Prefers Rust for systems work", None, None).await?;

    // Hybrid search (BM25 + vector, fused with RRF)
    let results = store.search("systems programming", Some(5), None, None).await?;
    for r in &results {
        println!("{:.4} | {}", r.score, r.content);
    }

    Ok(())
}
```

### Without Ollama (FTS-only or MockEmbedder)

```rust
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder};

let store = MemoryStore::open_with_embedder(
    MemoryConfig::default(),
    Box::new(MockEmbedder::new(768)),
)?;

// BM25 search works without a live embedding service, and MockEmbedder
// enables vector search, explainable search, and conversation indexing locally.
let results = store.search_fts_only("Python", Some(5), None, None).await?;
```

### Conversation Memory

```rust
use semantic_memory::{MemoryStore, MemoryConfig, MockEmbedder, Role};

let store = MemoryStore::open_with_embedder(
    MemoryConfig::default(),
    Box::new(MockEmbedder::new(768)),
)?;

let session = store.create_session("chat").await?;
store.add_message(&session, Role::User, "What is Rust?", Some(10), None).await?;
store.add_message(&session, Role::Assistant, "A systems language.", Some(8), None).await?;

// Retrieve messages within a token budget
let msgs = store.get_messages_within_budget(&session, 100).await?;
```

## Architecture

```
┌─────────────────────────────────────────┐
│              MemoryStore                │
├──────────┬──────────┬───────────────────┤
│  Facts   │  Docs    │  Conversations    │
│          │ (chunks) │  (sessions/msgs)  │
├──────────┴──────────┴───────────────────┤
│           Search Layer                  │
│  BM25 (FTS5)  +  Vector (HNSW/BF)      │
│         → Reciprocal Rank Fusion        │
├─────────────────────────────────────────┤
│ SQLite (authoritative) + HNSW sidecar  │
└─────────────────────────────────────────┘
```

All data lives under a single `base_dir`. SQLite stores every durable record and embedding. HNSW is a recoverable acceleration sidecar: writes are journaled in SQLite, sidecar failures do not roll back committed data, and pending index work is replayed on open, flush, rebuild, or reconcile.

## Operational Semantics

- SQLite is the source of truth for facts, documents, chunks, messages, and episodes.
- HNSW is derived state. Pending sidecar mutations are recorded durably in SQLite and replayed after commit.
- The pool uses one writer connection plus `max_read_connections` pooled readers. Under WAL mode, readers can proceed concurrently while writes serialize on the writer connection.
- `search_explained()` returns the exact decomposition used by the active search pipeline, not a reconstructed approximation.
- Generic `search()` includes facts, chunks, and episodes by default. Messages are opt-in via `SearchSourceType::Messages` or `search_conversations()`.
- `verify_integrity()` surfaces malformed JSON/enums/roles, missing embeddings, quantization drift, pending sidecar work, and HNSW/keymap drift instead of laundering bad rows into defaults.
- `reconcile(RebuildFts)` repairs FTS drift and replays pending sidecar work. `reconcile(ReEmbed)` re-embeds facts, chunks, messages, and episodes from SQLite, repairs missing q8 data, and rebuilds HNSW.
- `store.graph_view()` exposes deterministic traversal over namespace, fact, document, chunk, session, message, and episode nodes derived from SQLite state.

## API Overview

### Initialization

| Method | Description |
|---|---|
| `MemoryStore::open(config)` | Open with default Ollama embedder |
| `MemoryStore::open_with_embedder(config, embedder)` | Open with custom/mock embedder |
| `store.config()` | Access the active configuration |
| `store.graph_view()` | Create a derived graph traversal view |

### Facts

| Method | Description |
|---|---|
| `add_fact(namespace, content, source, metadata)` | Store a fact with auto-embedding |
| `add_fact_with_embedding(namespace, content, embedding, source, metadata)` | Store with pre-computed embedding |
| `update_fact(fact_id, content)` | Update content and re-embed |
| `delete_fact(fact_id)` | Delete a single fact |
| `delete_namespace(namespace)` | Delete all facts in a namespace |
| `get_fact(fact_id)` | Retrieve a fact by ID |
| `get_fact_embedding(fact_id)` | Retrieve raw embedding vector |
| `list_facts(namespace, limit, offset)` | Paginated listing |

### Documents

| Method | Description |
|---|---|
| `ingest_document(title, content, namespace, source_path, metadata)` | Chunk, embed, and store |
| `delete_document(document_id)` | Delete document and all its chunks |
| `list_documents(namespace, limit, offset)` | Paginated listing |
| `count_chunks_for_document(document_id)` | Get chunk count |
| `chunk_text(text)` | Preview chunking without storing |

### Conversations

| Method | Description |
|---|---|
| `create_session(channel)` | Create a new session |
| `rename_session(session_id, new_channel)` | Rename a session |
| `delete_session(session_id)` | Delete session and its messages |
| `list_sessions(limit, offset)` | Paginated listing |
| `add_message(session_id, role, content, token_count, metadata)` | Add with FTS + embedding |
| `add_message_fts(...)` | Add with FTS indexing only |
| `add_message_embedded(...)` | Add with embedding only |
| `get_recent_messages(session_id, limit)` | Most recent N messages |
| `get_messages_within_budget(session_id, max_tokens)` | Fit messages into token budget |
| `session_token_count(session_id)` | Total tokens in a session |

### Search

| Method | Description |
|---|---|
| `search(query, top_k, namespaces, source_types)` | Hybrid BM25 + vector with exact RRF explanations available via `search_explained` |
| `search_fts_only(...)` | BM25 full-text search only |
| `search_vector_only(...)` | Vector similarity only |
| `search_conversations(query, top_k, session_ids)` | Search across conversation messages |
| `search_explained(...)` | Hybrid search with exact score breakdowns from the live pipeline |

### Episodes

| Method | Description |
|---|---|
| `ingest_episode(document_id, meta)` | Upsert a searchable episode while preserving original `created_at` |
| `update_episode_outcome(document_id, outcome, confidence, experiment_id)` | Update outcome, verification state, search text, and embeddings |
| `search_episodes(effect_type, outcome, limit)` | Query episodes by type/outcome |

### Embeddings & Analysis

| Method | Description |
|---|---|
| `embed(text)` | Get embedding vector for text |
| `embed_batch(texts)` | Batch embed multiple texts |
| `embedding_displacement(text_a, text_b)` | Compare two texts (cosine, euclidean) |
| `MemoryStore::embedding_displacement_from_vecs(a, b)` | Compare precomputed vectors, returning an error on dimension mismatch |

### Integrity & Maintenance

| Method | Description |
|---|---|
| `verify_integrity(mode)` | Check schema, SQLite integrity, FTS drift, malformed rows, quantized blobs, and HNSW drift |
| `reconcile(action)` | Repair issues (`ReportOnly`, `RebuildFts`, `ReEmbed`) |
| `embeddings_are_dirty()` | Check if any embeddings need refresh |
| `reembed_all()` | Re-embed facts, chunks, messages, and episodes, then rebuild HNSW |
| `stats()` | Database statistics |
| `vacuum()` | SQLite VACUUM |
| `rebuild_hnsw_index()` | Rebuild HNSW from scratch |
| `flush_hnsw()` | Persist HNSW to disk, replaying pending SQLite-journaled sidecar work first |
| `compact_hnsw()` | Compact HNSW deleted entries |

### Graph View

`store.graph_view()` returns a `GraphView` over the authoritative SQLite state. The derived graph currently includes:

- `namespace:{name}` → facts and documents in that namespace
- `document:{uuid}` → chunks and attached episode
- `session:{uuid}` → messages in that session
- `episode:{document_id}` → causal links to referenced facts/chunks/messages
- semantic chunk/message/episode edges when cosine similarity clears the configured threshold

```rust
use semantic_memory::{GraphDirection, MemoryConfig, MemoryStore, MockEmbedder};

# fn example() -> Result<(), semantic_memory::MemoryError> {
let store = MemoryStore::open_with_embedder(
    MemoryConfig::default(),
    Box::new(MockEmbedder::new(768)),
)?;

let graph = store.graph_view();
let edges = graph.neighbors("namespace:general", GraphDirection::Outgoing, 1)?;
for edge in edges {
    println!("{} -> {}", edge.source, edge.target);
}
# Ok(())
# }
```

### Integrity Workflow

```rust
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder, ReconcileAction, VerifyMode};

# async fn example() -> Result<(), semantic_memory::MemoryError> {
let store = MemoryStore::open_with_embedder(
    MemoryConfig::default(),
    Box::new(MockEmbedder::new(768)),
)?;

let report = store.verify_integrity(VerifyMode::Full).await?;
if !report.ok {
    let repaired = store.reconcile(ReconcileAction::ReEmbed).await?;
    assert!(repaired.ok);
}
# Ok(())
# }
```

## Configuration

### MemoryConfig

```rust
MemoryConfig {
    base_dir: PathBuf::from("memory"),    // Storage directory
    embedding: EmbeddingConfig { .. },     // Ollama connection + model
    search: SearchConfig { .. },           // Fusion weights + thresholds
    chunking: ChunkingConfig { .. },       // Document chunking params
    pool: PoolConfig { .. },               // SQLite connection settings
    limits: MemoryLimits { .. },           // Resource caps
    hnsw: HnswConfig { .. },              // HNSW index params (when enabled)
}
```

### EmbeddingConfig (defaults)

| Field | Default | Description |
|---|---|---|
| `ollama_url` | `http://localhost:11434` | Ollama API endpoint |
| `model` | `nomic-embed-text` | Embedding model |
| `dimensions` | `768` | Vector dimensionality |
| `batch_size` | `32` | Max texts per API call |
| `timeout_secs` | `30` | Request timeout |

### SearchConfig (defaults)

| Field | Default | Description |
|---|---|---|
| `bm25_weight` | `1.0` | BM25 weight in RRF |
| `vector_weight` | `1.0` | Vector weight in RRF |
| `rrf_k` | `60.0` | RRF rank decay constant |
| `candidate_pool_size` | `50` | Candidates per search method |
| `default_top_k` | `5` | Default result count |
| `min_similarity` | `0.3` | Minimum cosine similarity |
| `recency_half_life_days` | `None` | Optional recency boost half-life |
| `recency_weight` | `0.5` | Recency boost weight in RRF |
| `rerank_from_f32` | `true` | Rerank HNSW hits with exact f32 similarity |

### ChunkingConfig (defaults)

| Field | Default | Description |
|---|---|---|
| `target_size` | `1000` | Target chunk size (chars) |
| `min_size` | `100` | Minimum chunk size |
| `max_size` | `2000` | Maximum chunk size |
| `overlap` | `200` | Overlap between chunks |

### PoolConfig (defaults)

| Field | Default | Description |
|---|---|---|
| `busy_timeout_ms` | `5000` | SQLite busy timeout |
| `wal_autocheckpoint` | `1000` | WAL checkpoint threshold (pages) |
| `enable_wal` | `true` | Enable WAL mode |
| `max_read_connections` | `4` | Number of pooled reader connections |

### MemoryLimits (defaults)

| Field | Default | Description |
|---|---|---|
| `max_facts_per_namespace` | `100,000` | Max facts per namespace |
| `max_chunks_per_document` | `1,000` | Max chunks per document |
| `max_content_bytes` | `1 MB` | Max content size per entry |
| `max_embedding_concurrency` | `8` | Concurrent embedding requests (hard cap: 32) |
| `max_db_size_bytes` | `0` (unlimited) | Max database file size |
| `embedding_timeout` | `30s` | Embedding request timeout |

## Key Types

```rust
// Data types
Fact        // Namespaced fact with optional source and metadata
Document    // Ingested document with chunk count
TextChunk   // Individual chunk from document splitting
Session     // Conversation session
Message     // Chat message with role and optional token count
Role        // System | User | Assistant | Tool

// Search types
SearchResult     // Hybrid result with score and source info
ExplainedResult  // SearchResult + ScoreBreakdown
SearchSource     // Fact | Chunk | Message | Episode source details
SearchSourceType // Filter: Facts | Chunks | Messages | Episodes
ScoreBreakdown   // RRF, BM25, vector, recency scores and ranks

// Graph types
GraphView        // Graph traversal over derived memory relationships
GraphDirection   // Outgoing | Incoming | Both
GraphEdge        // Edge returned from graph traversal
GraphEdgeType    // Semantic | Temporal | Causal | Entity

// Episode types
EpisodeMeta         // Causal metadata (cause_ids, effect_type, outcome)
EpisodeOutcome      // Confirmed | Refuted | Inconclusive | Pending
VerificationStatus  // Unverified | Verified | Failed

// Analysis types
EmbeddingDisplacement  // Cosine similarity, euclidean distance, magnitudes
MemoryStats            // Counts and database size summary

// Embedding types
Embedder         // Trait: embed(), embed_batch(), model_name(), dimensions()
OllamaEmbedder   // Production embedder (Ollama API)
MockEmbedder      // Deterministic fake embedder for tests

// Quantization
Quantizer        // f32 → int8 quantizer
QuantizedVector  // Quantized data + scale + zero_point

// Integrity
IntegrityReport   // Health check results
VerifyMode        // Quick | Full
ReconcileAction   // ReportOnly | RebuildFts | ReEmbed

// Auditability
TraceId           // Stable write correlation ID propagated into metadata

// HNSW (feature = "hnsw")
HnswConfig  // Index parameters (m, ef_construction, ef_search, etc.)
HnswIndex   // ANN index with insert/delete/update/search
HnswHit     // Search hit with key and distance
```

## Cargo Features

| Feature | Default | Description |
|---|---|---|
| `hnsw` | Yes | HNSW approximate nearest-neighbor search via `hnsw_rs` |
| `brute-force` | No | Brute-force vector search (no external deps) |
| `testing` | No | Exposes `raw_execute()` for test harnesses |

At least one of `hnsw` or `brute-force` must be enabled.

## Error Handling

All fallible operations return `Result<T, MemoryError>`. Error variants include:

- `Database` — SQLite errors
- `EmbeddingRequest` — HTTP/network errors from embedding provider
- `DimensionMismatch` — embedding size doesn't match config
- `InvalidEmbedding` — malformed embedding or quantized blob payloads
- `ModelMismatch` — stored model differs from configured model
- `SessionNotFound`, `FactNotFound`, `DocumentNotFound` — missing entities
- `ContentTooLarge` — content exceeds `max_content_bytes`
- `NamespaceFull` — namespace exceeds `max_facts_per_namespace`
- `DatabaseSizeLimitExceeded` — configured SQLite growth ceiling would be exceeded
- `InvalidConfig` — config normalization rejected an invalid runtime value
- `CorruptData` — stored JSON, enum, role, or sidecar state is malformed
- `SchemaAhead` — database was created by a newer version
- `HnswError`, `QuantizationError`, `StorageError` — subsystem errors
- `IntegrityError` — HNSW/SQLite index mismatch

Use `error.kind()` for stable string discriminants suitable for programmatic matching.

## Examples

```bash
# Hybrid search with Ollama (requires Ollama + nomic-embed-text)
cargo run --example basic_search

# Conversation memory (no Ollama needed)
cargo run --example conversation_memory
```

## License

MIT
