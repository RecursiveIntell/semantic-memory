# semantic-memory

[![Crates.io](https://img.shields.io/crates/v/semantic-memory.svg)](https://crates.io/crates/semantic-memory)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Local-first hybrid semantic search for Rust, built for AI agents.

`semantic-memory` combines SQLite (FTS5) and HNSW approximate nearest-neighbor search into one embedded crate. It handles facts, documents, conversation history, and causal episodes — no external vector database required.

## Features

- **Hybrid search** — BM25 full-text + vector similarity fused with Reciprocal Rank Fusion (RRF)
- **HNSW ANN** — approximate nearest-neighbor index with brute-force fallback
- **Fact storage** — namespaced key-value facts with embeddings
- **Document ingestion** — chunking with overlap, per-chunk embeddings
- **Conversation memory** — sessions, messages, token budgets
- **Episode tracking** — causal records with verification status and outcomes
- **Explainable results** — full score breakdowns (BM25 rank, vector rank, RRF, recency)
- **Integrity & repair** — verify, reconcile, detect dirty embeddings, re-embed
- **Embedding quantization** — int8 quantization helpers for compact storage
- **Resource limits** — caps on namespace size, content length, embedding concurrency
- **Connection pooling** — SQLite WAL mode, busy timeout, auto-checkpoint tuning

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

// FTS-only search works without any embedding provider
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
│  SQLite (WAL) + HNSW sidecar files     │
└─────────────────────────────────────────┘
```

All data lives under a single `base_dir`. SQLite stores facts, documents, chunks, sessions, messages, and embeddings. The optional HNSW index is kept as sidecar files alongside the database.

## API Overview

### Initialization

| Method | Description |
|---|---|
| `MemoryStore::open(config)` | Open with default Ollama embedder |
| `MemoryStore::open_with_embedder(config, embedder)` | Open with custom/mock embedder |
| `store.config()` | Access the active configuration |

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
| `search(query, top_k, namespaces, source_types)` | Hybrid BM25 + vector with RRF |
| `search_fts_only(...)` | BM25 full-text search only |
| `search_vector_only(...)` | Vector similarity only |
| `search_conversations(query, top_k, session_ids)` | Search across conversation messages |
| `search_explained(...)` | Hybrid search with full score breakdowns |

### Episodes

| Method | Description |
|---|---|
| `ingest_episode(document_id, meta)` | Attach causal metadata to a document |
| `update_episode_outcome(document_id, outcome, confidence, experiment_id)` | Update outcome |
| `search_episodes(effect_type, outcome, limit)` | Query episodes by type/outcome |

### Embeddings & Analysis

| Method | Description |
|---|---|
| `embed(text)` | Get embedding vector for text |
| `embed_batch(texts)` | Batch embed multiple texts |
| `embedding_displacement(text_a, text_b)` | Compare two texts (cosine, euclidean) |

### Integrity & Maintenance

| Method | Description |
|---|---|
| `verify_integrity(mode)` | Check database health (`Quick` or `Full`) |
| `reconcile(action)` | Repair issues (`ReportOnly`, `RebuildFts`, `ReEmbed`) |
| `embeddings_are_dirty()` | Check if any embeddings need refresh |
| `reembed_all()` | Re-embed all content |
| `stats()` | Database statistics |
| `vacuum()` | SQLite VACUUM |
| `rebuild_hnsw_index()` | Rebuild HNSW from scratch |
| `flush_hnsw()` | Persist HNSW to disk |
| `compact_hnsw()` | Compact HNSW deleted entries |

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
- `ModelMismatch` — stored model differs from configured model
- `SessionNotFound`, `FactNotFound`, `DocumentNotFound` — missing entities
- `ContentTooLarge` — content exceeds `max_content_bytes`
- `NamespaceFull` — namespace exceeds `max_facts_per_namespace`
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
