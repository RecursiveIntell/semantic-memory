# semantic-memory

Local-first semantic memory for Rust.

`semantic-memory` combines SQLite, FTS5, and optional HNSW search into one crate for facts, documents, conversation history, and causal episodes. It is built for agent systems that need durable retrieval without standing up a separate vector database.

## What It Supports Today

- Hybrid search: BM25 + vector retrieval fused with RRF
- Optional HNSW ANN with brute-force fallback path
- Fact storage by namespace
- Document ingestion with chunking and overlap
- Conversation/session memory with token budgeting
- Episode storage for causal or experimental records
- Explainable search breakdowns
- Embedding quantization helpers
- Integrity verification and FTS reconciliation
- Dirty-embedding detection and full re-embedding

## Quick Start

```rust
use semantic_memory::{MemoryConfig, MemoryStore};
use std::path::PathBuf;

# async fn run() -> Result<(), semantic_memory::MemoryError> {
let store = MemoryStore::open(MemoryConfig {
    base_dir: PathBuf::from("./memory"),
    ..Default::default()
})?;

store
    .add_fact("general", "Rust was first released in 2015", None, None)
    .await?;

let results = store
    .search("systems programming language", Some(5), None, None)
    .await?;

for result in results {
    println!("{:.4} {}", result.score, result.content);
}
# Ok(()) }
```

## Core Concepts

| Type | Role |
| --- | --- |
| `MemoryStore` | Main handle for all storage and retrieval operations |
| `MemoryConfig` | Base dir, embedding, chunking, pool, and limit settings |
| `Embedder` | Text-to-vector abstraction (`OllamaEmbedder`, `MockEmbedder`) |
| `SearchResult` | Hybrid retrieval result |
| `ExplainedResult` | Search result plus scoring breakdown |
| `EpisodeMeta` | Causal/verification metadata attached to documents |
| `IntegrityReport` | Verification and reconciliation summary |

## Config Shape

Current configs are rooted at `base_dir`, not a single `database_path`.
The crate manages SQLite plus optional HNSW sidecar files under that directory.

Key config areas:

- `embedding`
- `search`
- `chunking`
- `pool`
- `limits`
- `hnsw` when the feature is enabled

## Higher-Level Features

### Conversation Memory

Create sessions, add messages, retrieve recent history, or enforce token budgets.
Search helpers include `search_fts_only()`, `search_vector_only()`, and `search_conversations()`.

### Episodes and Verification

Attach `EpisodeMeta` to documents, update outcomes, search episodes, and use `VerificationStatus` to model validation state.

### Integrity and Repair

- `verify_integrity()`
- `reconcile()`
- `embeddings_are_dirty()`
- `reembed_all()`

These make the crate more operationally useful than a simple demo vector store.

Open the store with `open()` for the default Ollama-backed embedder or `open_with_embedder()` when you want a mock or custom embedder.

## Features

- Default feature: `hnsw`
- Optional: `brute-force`
- Optional: `testing`

At least one search backend must be enabled.

## Examples

- `basic_search`
- `conversation_memory`

## License

MIT
