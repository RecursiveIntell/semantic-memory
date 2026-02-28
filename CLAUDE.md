# CLAUDE.md — semantic-memory Project Context

## Project Overview

`semantic-memory` is a Rust library providing hybrid semantic search (BM25 + vector similarity) backed by SQLite + FTS5 + HNSW. It's designed for AI agent memory systems.

**Current version:** 0.3.0
**Rust edition:** 2021
**MSRV:** 1.75 (stable only)

## Architecture

```
MemoryStore (Arc<MemoryStoreInner>)
├── conn: Mutex<rusqlite::Connection>     — Single SQLite connection, all I/O via spawn_blocking
├── embedder: Box<dyn Embedder>           — Ollama (prod) or Mock (test)
├── config: MemoryConfig                  — All tuning knobs
├── paths: StoragePaths                   — File layout convention
├── token_counter: Arc<dyn TokenCounter>  — chars/4 estimator by default
└── hnsw_index: RwLock<HnswIndex>         — Approximate nearest neighbor (feature-gated, hot-swappable)
```

**Storage layout:**
```
base_dir/
├── memory.db            — SQLite (content, metadata, FTS5, f32 embeddings)
├── memory.hnsw.graph    — HNSW graph topology
└── memory.hnsw.data     — HNSW vector data
```

**Search pipeline:**
1. FTS5 MATCH → BM25 candidates
2. HNSW ANN search → vector candidates (or brute-force fallback)
3. Reciprocal Rank Fusion → merged, scored results

## Key Design Decisions

- SQLite is the single source of truth. HNSW is a performance index that can be rebuilt from SQLite.
- All DB access goes through `with_conn()` + `spawn_blocking` to keep the tokio runtime unblocked.
- Feature flags: `hnsw` (default) vs `brute-force`. At least one required at compile time.
- `MockEmbedder` uses deterministic hash-seeded xorshift for reproducible tests.

## v0.3.0 Changes (completed)

- Fixed SQ8 quantization math (symmetric [-127, 127])
- HNSW key mappings persisted in SQLite (`hnsw_keymap` table) — survive process restart
- Hot-swap HNSW index via `RwLock` — `rebuild_hnsw_index()` updates live instance
- `search_vector_only()` routes through HNSW when feature enabled
- SQ8 quantized embeddings stored alongside f32 in all insert paths (`embedding_q8` columns)
- Batched HNSW→SQLite lookups (eliminated N+1 queries in hybrid search)
- HNSW compaction via `compact_hnsw()` — tombstone cleanup when deleted ratio exceeds threshold
- Fixed `Box::leak` in HNSW load — reloader kept alive via `_reloader_keepalive`
- Migration V5: `embedding_q8` columns on facts/chunks/messages, `hnsw_keymap` table

## Common Commands

```bash
cargo check --all-features
cargo test --all-features
cargo test --features "hnsw,testing"
cargo test --features "brute-force,testing"
cargo clippy --all-features -- -D warnings
cargo doc --all-features --no-deps
```

## Code Conventions

- Error handling: `MemoryError` enum with `thiserror`, `#[from]` for auto-conversion
- Async: All public `MemoryStore` methods are async. Internal SQLite work is sync inside `with_conn()`
- Feature gating: `#[cfg(feature = "hnsw")]` for all HNSW-specific code paths
- IDs: UUID v4 strings for facts/documents/chunks/sessions, i64 autoincrement for messages
- HNSW keys: `"{domain}:{id}"` format, e.g., `"fact:abc-123"`, `"chunk:def-456"`, `"msg:42"`

## Don't

- Don't add connection pooling (future work)
- Don't use nightly Rust features
- Don't develop directly on the server — all development is local
