# semantic-memory

Local-first semantic memory for Rust, backed by SQLite with FTS5 and an optional recoverable HNSW sidecar.

## Status

This README describes the shipped crate source, public API, examples, tests, and on-disk files in this crate. It does not treat the historical spec/archive files in this directory as active runtime API.

- Crate version: `0.5.0`
- Minimum Rust: `1.75`
- Current SQLite schema version: `17`
- Runtime shape: library crate only, no built-in daemon or CLI
- Stability posture: implemented and broadly tested, but still a `0.x` crate
- Compatibility posture: legacy envelope import remains present only for migration and is visibly deprecated/hidden

## What The Crate Owns

`semantic-memory` owns local, queryable memory state:

- facts
- documents and text chunks
- conversation sessions and messages
- searchable causal episodes
- imported projection rows and their import receipts

It does not own raw export truth or export transformation. The bridge boundary is `forge-memory-bridge`; this crate owns the imported/queryable representation after that handoff.

## What It Does

- SQLite is the authoritative store for all durable rows and f32 embeddings.
- FTS5 provides BM25 full-text search.
- Vector search uses cosine similarity.
- Reciprocal Rank Fusion combines BM25 and vector results.
- HNSW is an optional acceleration sidecar, not a source of truth.
- Sidecar mutations are journaled in SQLite and replayed on open, flush, rebuild, or reconcile.
- Quantized `q8` copies of embeddings are stored for facts, chunks, messages, and episodes.
- WAL mode plus a single writer and pooled readers allows concurrent read-heavy use.
- Integrity and repair tooling is built in.
- Projection imports are tracked with durable success and failure receipts.

## What It Does Not Do

- It does not transform Forge exports itself.
- It does not make imported projection rows part of the generic `search()` hybrid results today.
- It does not treat the legacy envelope import path as a normal integration seam.
- It does not expose evidence payloads as part of generic retrieval; evidence refs are queried through their own projection API.

## Search Semantics

The search surface is slightly narrower than the full storage surface:

- `search()` targets facts, document chunks, and searchable episodes by default.
- Conversation messages are queried through `search_conversations()` or by explicitly opting into `SearchSourceType::Messages`.
- `search_fts_only()` uses BM25/FTS only.
- `search_vector_only()` uses vector similarity only.
- `search_explained()` returns the exact score breakdown used by the live ranking pipeline.
- Namespace filters and session filters are parameterized, not interpolated.
- When recency weighting is enabled in config, recency contributes to fused ranking.
- Results are deduplicated by normalized content after fusion.
- If embeddings are marked dirty after an embedding-model change, searches still run but warn that quality is degraded until `reembed_all()` completes.

Imported projection rows are queried through dedicated read APIs rather than generic hybrid search.

## Data Model

### Native memory rows

- Facts: free-text assertions grouped by namespace
- Documents: top-level source records with namespace, title, source path, and metadata
- Chunks: chunked document text with token estimates and embeddings
- Sessions: conversation containers
- Messages: ordered session messages with roles, token counts, metadata, and optional embeddings
- Episodes: searchable causal/verification records attached to documents

### Imported projection rows

- `claim_versions`
- `relation_versions`
- `entity_aliases`
- `evidence_refs`
- `episode_links`
- `projection_import_log`
- `projection_import_failures`
- `derivation_edges`

The projection read surface is scope-aware and includes valid-time and recorded-time filtering where that makes sense.

## Public API Overview

### Opening and configuration

- `MemoryStore::open()` builds the default Ollama-backed store
- `MemoryStore::open_with_embedder()` accepts a custom embedder such as `MockEmbedder`
- `MemoryConfig` groups embedding, search, chunking, pool, and limit settings

### Facts

- `add_fact()`
- `add_fact_with_trace()`
- `add_fact_with_embedding()`
- `update_fact()`
- `delete_fact()`
- `delete_namespace()`
- `get_fact()`
- `get_fact_embedding()`
- `list_facts()`

### Documents and chunks

- `ingest_document()`
- `ingest_document_with_trace()`
- `delete_document()`
- `list_documents()`
- `count_chunks_for_document()`
- `chunk_text()`

### Conversations

- `create_session()`
- `create_session_with_metadata()`
- `rename_session()`
- `list_sessions()`
- `delete_session()`
- `add_message()`
- `add_message_fts()`
- `add_message_embedded()`
- `get_recent_messages()`
- `get_messages_within_budget()`
- `session_token_count()`
- `search_conversations()`

### Episodes

- `ingest_episode()`
- `create_episode()`
- `get_episode()`
- `update_episode_outcome_by_id()`
- `update_episode_outcome()` for the legacy document-keyed seam
- `search_episodes()`

### Search and ranking

- `search()`
- `search_fts_only()`
- `search_vector_only()`
- `search_explained()`
- `embedding_displacement()`
- `embedding_displacement_from_vecs()`

### Projection import and reads

- `import_projection_batch()` is the canonical import seam
- `import_projection_batch_json_compat()` keeps the JSON compatibility path
- `query_projection_imports()`
- `latest_rebuildable_kernel_projection_import_for_scope()`
- `query_projection_import_failures()`
- `query_claim_versions()`
- `query_relation_versions()`
- `query_episodes()`
- `query_entity_aliases()`
- `query_evidence_refs()`
- `invalidate_derivations()`

### Diagnostics and maintenance

- `verify_integrity()`
- `reconcile()`
- `embeddings_are_dirty()`
- `reembed_all()`
- `stats()`
- `vacuum()`
- `graph_view()`
- `embed()`
- `embed_batch()`
- `flush_hnsw()`
- `rebuild_hnsw_index()`
- `compact_hnsw()`

## Projection Import Boundary

The canonical import path is:

`ExportEnvelopeV3 -> forge-memory-bridge transform -> ProjectionImportBatchV3 -> semantic-memory import transaction`

Important details from the current implementation:

- imports are idempotent by `source_envelope_id + schema_version + content_digest`
- successful imports write durable import-log receipts
- failed imports write durable failure receipts
- receipts preserve `source_exported_at` and `transformed_at`
- when present, receipts also preserve `evidence_bundle`, `episode_bundle`, `execution_context`, and `kernel_payload`
- `query_claim_versions()`, `query_relation_versions()`, `query_episodes()`, `query_entity_aliases()`, and `query_evidence_refs()` are the supported public read surfaces for imported rows
- the old `import_envelope()` path is still callable only as compatibility/migration scaffolding

## Source Layout

| Path | Responsibility |
|------|----------------|
| `src/lib.rs` | public facade, store open/reconcile/search/stats maintenance surface |
| `src/db.rs` | SQLite open/configuration, migrations, schema versioning, integrity verification, FTS rebuild helpers |
| `src/pool.rs` | single-writer plus pooled-reader SQLite access model |
| `src/embedder.rs` | `Embedder` trait, `OllamaEmbedder`, `MockEmbedder`, response parsing |
| `src/tokenizer.rs` | token counting trait and default estimate-based counter |
| `src/chunker.rs` | recursive UTF-8-safe text chunking with overlap |
| `src/knowledge.rs` | fact CRUD plus FTS/HNSW synchronization |
| `src/documents.rs` | document ingestion, chunk persistence, chunk cleanup |
| `src/conversation.rs` | session/message CRUD, token-budget retrieval, conversation search |
| `src/episodes.rs` | episode ingest/update/search and document-linked causal state |
| `src/search.rs` | BM25 search, brute-force vector search, HNSW-assisted ranking, RRF fusion, dedup |
| `src/hnsw.rs` | HNSW wrapper and sidecar keymap logic |
| `src/hnsw_ops.rs` | rebuild/recovery/flush/replay operations for the sidecar |
| `src/quantize.rs` | q8 quantization pack/unpack logic |
| `src/graph.rs` | deterministic graph view over namespaces, documents, chunks, facts, messages, and episodes |
| `src/storage.rs` | storage-path conventions under the base directory |
| `src/types.rs` | public data types and search/graph/projection structs |
| `src/error.rs` | crate error model |
| `src/projection_batch.rs` | import-batch trait helpers |
| `src/projection_lane.rs` | canonical projection import transaction and receipt query surface |
| `src/projection_storage.rs` | projection table definitions and query/storage helpers |
| `src/projection_derivation.rs` | derivation invalidation surface and derivation-focused tests |
| `src/json_compat_import.rs` | JSON compatibility decoder for projection batches |
| `src/projection_import.rs` | legacy envelope import compatibility module |
| `src/projection_legacy_compat.rs` | wrappers for legacy import compatibility calls |
| `src/store_support.rs` | shared helpers for trace metadata, episode search text, and string-slice capture |

### Non-source directories in this crate

- `examples/` contains runnable examples
- `tests/` contains integration coverage
- `reference/` contains local reference material used during implementation, not runtime API

## On-Disk Layout

By default a store lives under `MemoryConfig.base_dir` and uses:

- `memory.db` for SQLite data, FTS tables, metadata, receipts, and embeddings
- `memory.hnsw.graph` for HNSW graph topology when the `hnsw` feature is enabled
- `memory.hnsw.data` for HNSW vector data when the `hnsw` feature is enabled

SQLite stays authoritative even when the sidecar is missing or stale. The sidecar is recoverable from SQLite state.

## Default Configuration

The defaults in the current source are:

- base directory: `memory`
- embedding URL: `http://localhost:11434`
- embedding model: `nomic-embed-text`
- embedding dimensions: `768`
- embedding batch size: `32`
- embedding timeout: `30s`
- chunk target/min/max/overlap: `1000 / 100 / 2000 / 200`
- search default top-k: `5`
- search candidate pool: `50`
- search minimum similarity: `0.3`
- recency weighting: disabled by default
- SQLite WAL: enabled
- reader pool size: `4`
- max embedding concurrency: `8`

## Quick Start

For local development and tests, `MockEmbedder` is the easiest way to avoid depending on Ollama:

```rust
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder, Role};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), semantic_memory::MemoryError> {
    let config = MemoryConfig {
        base_dir: PathBuf::from("memory-example"),
        ..Default::default()
    };

    let store = MemoryStore::open_with_embedder(config, Box::new(MockEmbedder::new(768)))?;

    store
        .add_fact("general", "Rust 1.0 was released in 2015.", None, None)
        .await?;

    let session_id = store.create_session("repl").await?;
    store
        .add_message(&session_id, Role::User, "When did Rust 1.0 ship?", None, None)
        .await?;

    let hits = store.search("Rust 1.0 release", Some(3), None, None).await?;
    println!("{hits:#?}");

    Ok(())
}
```

If you use `MemoryStore::open()` instead, the default runtime expects a reachable Ollama server and the configured model to exist.

## Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `hnsw` | yes | enables the HNSW acceleration sidecar via `hnsw_rs` |
| `brute-force` | no | allows building without HNSW and using exact vector scans only |
| `testing` | no | exposes extra testing-only helpers such as `raw_execute()` |

At least one search backend feature must be enabled.

## Operational Notes

- `verify_integrity(VerifyMode::Quick)` checks structural health cheaply.
- `verify_integrity(VerifyMode::Full)` also validates stored JSON/enums/blobs and checks SQLite integrity.
- `reconcile(ReconcileAction::RebuildFts)` rebuilds FTS tables from source rows.
- `reconcile(ReconcileAction::ReEmbed)` regenerates embeddings and then rebuilds the HNSW sidecar.
- `reembed_all()` processes facts, chunks, embedded messages, and episodes.
- `stats()` reports counts for facts, documents, chunks, sessions, and messages, plus database size and embedding metadata.
- `graph_view()` derives graph edges from SQLite state; it is not a separate graph store.

## Examples And Tests

- `examples/basic_search.rs` uses the default embedder path and is meant for an Ollama-backed setup.
- `examples/conversation_memory.rs` uses `MockEmbedder` and demonstrates sessions, token budgeting, and conversation search.
- `tests/` cover chunking, DB migrations, facts, conversations, documents, search, HNSW persistence/hot-swap, episodes, projection imports, derivation invalidation, quantization, concurrency, and integrity repair paths.
- `cargo test -- --list` in the current tree enumerates a large integration surface, including unit tests and doc-tests.

## Dependencies

- [`stack-ids`](../stack-ids) for canonical IDs, scopes, trace context, and digests
- [`forge-memory-bridge`](../forge-memory-bridge) for projection import batch types

## License

This repository uses a custom source-available license in [LICENSE](LICENSE).

Unless you have separate written permission from RecursiveIntell:

- any software, library, application, or service that uses, links to, imports, bundles, modifies, or incorporates this code must make its full corresponding source publicly available
- that corresponding source must be released under an OSI-approved open source license
- hosted or network use is treated the same way as distributed use

This is not an OSI-approved license for this repository itself; it is a control-retaining source-available license.
