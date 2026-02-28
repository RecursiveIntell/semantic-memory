# semantic-memory

Hybrid semantic search library for Rust, backed by **SQLite + FTS5 + HNSW**. Built for AI agent memory systems.

Combines BM25 full-text search with approximate nearest neighbor vector search via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), giving you the best of both lexical and semantic retrieval in a single query.

## Features

- **Hybrid search** — BM25 (FTS5) + cosine similarity fused with RRF
- **HNSW indexing** — Fast approximate nearest neighbor via [`hnsw_rs`](https://crates.io/crates/hnsw_rs), with brute-force fallback
- **Knowledge store** — Add, update, delete, and search facts organized by namespace
- **Document chunking** — Ingest long documents with configurable overlap chunking and per-chunk embeddings
- **Conversation memory** — Session-based message history with token budgeting and message search
- **SQ8 quantization** — Quantized embeddings stored alongside f32 for space-efficient persistence
- **Recency boosting** — Optional time-decay weighting with configurable half-life
- **Single-file storage** — Everything lives in one SQLite database + optional HNSW sidecar files
- **Async API** — All public methods are async, with SQLite I/O on `spawn_blocking`
- **Zero external services** — Only requires [Ollama](https://ollama.com) for embeddings (or use `MockEmbedder` for testing)

## Quick Start

### Prerequisites

- Rust 1.75+
- [Ollama](https://ollama.com) running locally with an embedding model:

```bash
ollama pull nomic-embed-text
```

### Installation

```toml
[dependencies]
semantic-memory = "0.3"
```

### Usage

```rust
use semantic_memory::{MemoryConfig, MemoryStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryStore::open(MemoryConfig::default())?;

    // Store facts
    store.add_fact("general", "Rust was first released in 2015", None, None).await?;
    store.add_fact("general", "Python is great for data science", None, None).await?;

    // Hybrid search (BM25 + vector similarity)
    let results = store.search("systems programming language", Some(5), None, None).await?;
    for r in &results {
        println!("[{:.4}] {}", r.score, r.content);
    }

    // FTS-only search (no embedding model needed)
    let results = store.search_fts_only("Rust", Some(5), None, None).await?;

    Ok(())
}
```

### Conversation Memory

```rust
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder, Role};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryStore::open_with_embedder(
        MemoryConfig::default(),
        Box::new(MockEmbedder::new(768)),
    )?;

    let session = store.create_session("repl").await?;

    store.add_message(&session, Role::User, "What is Rust?", Some(10), None).await?;
    store.add_message(&session, Role::Assistant, "A systems language.", Some(8), None).await?;

    // Get messages within a token budget
    let messages = store.get_messages_within_budget(&session, 500).await?;

    // Search across all conversations
    let results = store.search_messages("systems language", &session, Some(5)).await?;

    Ok(())
}
```

### Document Ingestion

```rust
let doc_id = store.add_document(
    "Rust Book Ch.1",
    &long_text,
    "docs",
    None, // source path
    None, // metadata
).await?;

// Chunks are automatically created, embedded, and searchable
let results = store.search("ownership and borrowing", Some(5), None, None).await?;
```

## Architecture

```
MemoryStore
├── SQLite (FTS5)        — BM25 full-text search, f32 + SQ8 embeddings, all metadata
├── HNSW Index           — Approximate nearest neighbor (optional, feature-gated)
└── Ollama / MockEmbedder — Embedding generation
```

**Search pipeline:**

1. FTS5 `MATCH` produces BM25-ranked candidates
2. HNSW ANN (or brute-force) produces vector-similarity candidates
3. Reciprocal Rank Fusion merges both lists into a single scored ranking

**Storage layout:**

```
base_dir/
├── memory.db            — SQLite database (content, metadata, FTS5, embeddings)
├── memory.hnsw.graph    — HNSW graph topology (optional)
└── memory.hnsw.data     — HNSW vector data (optional)
```

SQLite is the single source of truth. The HNSW index is a performance accelerator that can be rebuilt from SQLite at any time.

## Configuration

All configuration is done through `MemoryConfig`:

```rust
use semantic_memory::{MemoryConfig, SearchConfig, EmbeddingConfig, ChunkingConfig};
use std::path::PathBuf;

let config = MemoryConfig {
    base_dir: PathBuf::from("/data/agent-memory"),
    embedding: EmbeddingConfig {
        ollama_url: "http://localhost:11434".into(),
        model: "nomic-embed-text".into(),
        dimensions: 768,
        batch_size: 32,
        timeout_secs: 30,
    },
    search: SearchConfig {
        bm25_weight: 1.0,
        vector_weight: 1.0,
        rrf_k: 60.0,
        default_top_k: 5,
        min_similarity: 0.3,
        recency_half_life_days: Some(30.0), // Enable recency boosting
        ..Default::default()
    },
    chunking: ChunkingConfig {
        target_size: 1000,
        min_size: 100,
        max_size: 2000,
        overlap: 200,
    },
    ..Default::default()
};
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `hnsw` | Yes | HNSW approximate nearest neighbor search via `hnsw_rs` |
| `brute-force` | No | Exact cosine similarity search (no external index) |
| `testing` | No | Enables test utilities and `MockEmbedder` helpers |

At least one of `hnsw` or `brute-force` must be enabled.

```bash
# Default (HNSW enabled)
cargo build

# Brute-force only (no HNSW dependency)
cargo build --no-default-features --features brute-force

# Run tests
cargo test --features "hnsw,testing"
```

## API Overview

### Knowledge Store

| Method | Description |
|--------|-------------|
| `add_fact(namespace, content, source, metadata)` | Store a searchable fact |
| `update_fact(id, content, metadata)` | Update fact content and re-embed |
| `delete_fact(id)` | Remove a fact |
| `get_fact(id)` | Retrieve a fact by ID |
| `list_facts(namespace, limit, offset)` | List facts with pagination |

### Document Store

| Method | Description |
|--------|-------------|
| `add_document(title, content, namespace, source, meta)` | Ingest and chunk a document |
| `delete_document(id)` | Remove document and all its chunks |
| `list_documents(namespace, limit, offset)` | List documents with pagination |

### Search

| Method | Description |
|--------|-------------|
| `search(query, top_k, namespace, domain)` | Hybrid BM25 + vector search |
| `search_fts_only(query, top_k, namespace, domain)` | BM25-only search (no embeddings) |
| `search_vector_only(query, top_k, namespace, domain)` | Vector-only search |
| `search_messages(query, session_id, top_k)` | Search conversation history |

### Conversations

| Method | Description |
|--------|-------------|
| `create_session(channel)` | Start a new conversation session |
| `add_message(session, role, content, tokens, meta)` | Append a message |
| `get_recent_messages(session, limit)` | Get latest messages |
| `get_messages_within_budget(session, budget)` | Get messages fitting a token budget |
| `session_token_count(session)` | Total tokens in a session |
| `list_sessions(limit, offset)` | List all sessions |
| `delete_session(id)` | Remove a session and its messages |

### Maintenance

| Method | Description |
|--------|-------------|
| `stats()` | Database statistics |
| `rebuild_hnsw_index()` | Rebuild HNSW from SQLite (hot-swap) |
| `compact_hnsw()` | Clean up HNSW tombstones |

## License

MIT
