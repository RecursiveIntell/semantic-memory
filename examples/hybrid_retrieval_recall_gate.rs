//! Hybrid retrieval recall gate — measures end-to-end recall@10 for the
//! hybrid BM25+vector+RRF search pipeline against a golden corpus.
//!
//! Run with:
//!   cargo run --example hybrid_retrieval_recall_gate --features usearch-backend
//!
//! Emits a JSON receipt to stdout with per-query recall and overall recall.

use semantic_memory::embedder::{Embedder, MockEmbedder};
use semantic_memory::{EmbeddingConfig, MemoryConfig, MemoryStore, SearchConfig};
use std::collections::HashMap;

/// A golden document with a title and content.
struct GoldenDoc {
    title: &'static str,
    content: &'static str,
}

/// A query with its expected relevant document titles.
struct GoldenQuery {
    query: &'static str,
    relevant_titles: &'static [&'static str],
}

#[tokio::main]
async fn main() {
    let tmp = std::env::temp_dir().join("sm_recall_gate");
    let _ = std::fs::remove_dir_all(&tmp);

    // Open store with MockEmbedder (deterministic embeddings from text hash).
    let embedder: Box<dyn Embedder> = Box::new(MockEmbedder::new(768));
    let config = MemoryConfig {
        base_dir: tmp.clone(),
        embedding: EmbeddingConfig::default(),
        search: SearchConfig::default(),
        ..Default::default()
    };
    let store = MemoryStore::open_with_embedder(config, embedder).unwrap();

    // Golden corpus: 30 documents with distinct, searchable content.
    let corpus: Vec<GoldenDoc> = vec![
        GoldenDoc { title: "rust-lang", content: "The Rust programming language was created by Mozilla Research. It is a systems programming language focused on memory safety and concurrency without garbage collection." },
        GoldenDoc { title: "sqlite-engine", content: "SQLite is an embedded SQL database engine. It is the most widely deployed database in the world, used in mobile phones, web browsers, and operating systems." },
        GoldenDoc { title: "vector-embeddings", content: "Vector embeddings represent text as numerical arrays. They enable semantic similarity search by comparing embedding vectors using cosine similarity." },
        GoldenDoc { title: "bm25-ranking", content: "BM25 is a ranking function used in information retrieval. It ranks documents based on term frequency and inverse document frequency." },
        GoldenDoc { title: "rrf-fusion", content: "Reciprocal Rank Fusion combines multiple ranked result lists into a single ranking. It is simple, parameter-light, and effective for hybrid search." },
        GoldenDoc { title: "tokio-runtime", content: "Tokio is an asynchronous runtime for the Rust programming language. It provides an I/O driver, timer, and multi-threaded task scheduler." },
        GoldenDoc { title: "ftss-full-text", content: "SQLite FTS5 is a full-text search module. It creates virtual tables that support fast text search using inverted indexes." },
        GoldenDoc { title: "provenance-semiring", content: "Semiring provenance tracks evidence confidence using algebraic structures. Each item carries a confidence value and a support chain of observations." },
        GoldenDoc { title: "temporal-decay", content: "Temporal weight decay reduces the relevance of old knowledge over time. It combines age, supersession, support, and contradiction factors." },
        GoldenDoc { title: "discord-search", content: "Discord search is a second-order retrieval method. It finds items related to direct search results through graph neighbors but not themselves direct hits." },
        GoldenDoc { title: "graph-edges", content: "Graph edges connect nodes in a knowledge graph. Typed edges include semantic similarity, temporal proximity, causal relationships, and entity co-occurrence." },
        GoldenDoc { title: "factor-graph", content: "Factor graphs model probabilistic relationships between variables. Belief propagation propagates confidence through factor connections until convergence." },
        GoldenDoc { title: "compression-governor", content: "The compression governor decides which vectors to quantize based on importance scores. It reduces memory usage while preserving recall for important items." },
        GoldenDoc { title: "adaptive-routing", content: "Adaptive routing profiles a query and selects which retrieval stages to activate. It avoids unnecessary computation for simple or complex queries." },
        GoldenDoc { title: "lawful-subtraction", content: "Lawful subtraction is a safe forgetting mechanism. It removes low-value items while preserving invariants and maintaining retrieval quality." },
        GoldenDoc { title: "usearch-backend", content: "usearch is a vector search library. It won the benchmark against hnsw_rs by 2-78x on key metrics for desktop RAG applications." },
        GoldenDoc { title: "episode-causal", content: "Episodes link operations into causal chains. An episode records what happened, why it happened, and what the outcome was." },
        GoldenDoc { title: "chunking-strategy", content: "Document chunking splits long texts into smaller pieces for embedding. Each chunk is independently embedded and indexed for search." },
        GoldenDoc { title: "namespace-organization", content: "Namespaces organize facts and documents into logical groups. They enable scoped search and prevent cross-domain contamination." },
        GoldenDoc { title: "mcp-protocol", content: "The Model Context Protocol is a standard for connecting AI models to data sources. It defines a JSON-RPC based protocol for tool discovery and invocation." },
        GoldenDoc { title: "blake3-digest", content: "Blake3 is a cryptographic hash function. It is used for content digests that ensure idempotent operations and tamper detection." },
        GoldenDoc { title: "wal-mode", content: "Write-Ahead Logging (WAL) mode improves SQLite write concurrency. It allows readers and writers to operate simultaneously without blocking." },
        GoldenDoc { title: "cosine-similarity", content: "Cosine similarity measures the angle between two vectors. It ranges from -1 to 1, with 1 meaning identical direction and 0 meaning orthogonal." },
        GoldenDoc { title: "belief-propagation", content: "Belief propagation is a message-passing algorithm for inference on factor graphs. It iteratively updates variable beliefs based on factor constraints." },
        GoldenDoc { title: "syndrome-detection", content: "Syndrome detection identifies contradictions in search results. It flags items that disagree with each other or with known constraints." },
        GoldenDoc { title: "hnsw-index", content: "Hierarchical Navigable Small World (HNSW) is a graph-based vector index. It provides fast approximate nearest neighbor search with tunable recall." },
        GoldenDoc { title: "turbo-quant-codec", content: "The turbo-quant codec compresses vector embeddings using product quantization. It reduces storage by 8x while maintaining recall above 0.99." },
        GoldenDoc { title: "append-only-doctrine", content: "The append-only doctrine forbids destructive updates to truth-bearing rows. Instead, new rows supersede old ones, preserving full history." },
        GoldenDoc { title: "poly-kv-pool", content: "The poly-KV pool stores compressed vector candidates in a generation-managed pool. It enables lazy decompression and codec family switching." },
        GoldenDoc { title: "matryoshka-routing", content: "Matryoshka routing selects embedding dimensions based on query complexity. Simple queries use fewer dimensions, complex queries use more." },
    ];

    // Ingest all golden documents.
    let mut title_to_id: HashMap<String, String> = HashMap::new();
    for doc in &corpus {
        let doc_id = store
            .ingest_document(doc.title, doc.content, "recall_gate", None, None)
            .await
            .unwrap();
        title_to_id.insert(doc.title.to_string(), doc_id);
    }

    // Ground truth: 12 queries with known relevant documents.
    let queries: Vec<GoldenQuery> = vec![
        GoldenQuery {
            query: "Rust programming language",
            relevant_titles: &["rust-lang"],
        },
        GoldenQuery {
            query: "embedded SQL database",
            relevant_titles: &["sqlite-engine"],
        },
        GoldenQuery {
            query: "vector embeddings semantic search",
            relevant_titles: &["vector-embeddings"],
        },
        GoldenQuery {
            query: "BM25 ranking function",
            relevant_titles: &["bm25-ranking"],
        },
        GoldenQuery {
            query: "reciprocal rank fusion hybrid",
            relevant_titles: &["rrf-fusion"],
        },
        GoldenQuery {
            query: "Tokio async runtime",
            relevant_titles: &["tokio-runtime"],
        },
        GoldenQuery {
            query: "full text search FTS5",
            relevant_titles: &["ftss-full-text"],
        },
        GoldenQuery {
            query: "provenance evidence confidence",
            relevant_titles: &["provenance-semiring"],
        },
        GoldenQuery {
            query: "temporal decay old knowledge",
            relevant_titles: &["temporal-decay"],
        },
        GoldenQuery {
            query: "factor graph belief propagation",
            relevant_titles: &["factor-graph", "belief-propagation"],
        },
        GoldenQuery {
            query: "cosine similarity vectors",
            relevant_titles: &["cosine-similarity"],
        },
        GoldenQuery {
            query: "append only supersession history",
            relevant_titles: &["append-only-doctrine"],
        },
    ];

    // Run search for each query, compute recall@10.
    let mut per_query: Vec<serde_json::Value> = Vec::new();
    let mut total_relevant = 0usize;
    let mut total_found = 0usize;

    for q in &queries {
        let results = store
            .search(q.query, Some(10), Some(&["recall_gate"]), None)
            .await
            .unwrap();

        // Build set of relevant doc IDs from titles.
        let relevant_ids: Vec<String> = q
            .relevant_titles
            .iter()
            .filter_map(|t| title_to_id.get(*t).cloned())
            .collect();

        // Check which results match relevant docs (by chunk's document_id).
        let mut found_relevant: Vec<String> = Vec::new();
        for r in &results {
            if let semantic_memory::SearchSource::Chunk { document_id, .. } = &r.source {
                if relevant_ids.contains(document_id) && !found_relevant.contains(document_id) {
                    found_relevant.push(document_id.clone());
                }
            }
        }

        let recall = if relevant_ids.is_empty() {
            1.0
        } else {
            found_relevant.len() as f64 / relevant_ids.len() as f64
        };

        total_relevant += relevant_ids.len();
        total_found += found_relevant.len();

        per_query.push(serde_json::json!({
            "query": q.query,
            "relevant_count": relevant_ids.len(),
            "found_count": found_relevant.len(),
            "recall_at_10": (recall * 1000.0).round() / 1000.0,
            "results_returned": results.len(),
        }));
    }

    let overall_recall = if total_relevant > 0 {
        total_found as f64 / total_relevant as f64
    } else {
        1.0
    };

    // Assert minimum recall threshold.
    // MockEmbedder produces deterministic embeddings from text hash, so
    // recall depends on BM25 + vector overlap. With 30 docs and 12 queries,
    // a 0.5 threshold is achievable.
    let threshold = 0.5;
    let pass = overall_recall >= threshold;

    let receipt = serde_json::json!({
        "gate": "hybrid_retrieval_recall_gate",
        "corpus_size": corpus.len(),
        "query_count": queries.len(),
        "overall_recall_at_10": (overall_recall * 1000.0).round() / 1000.0,
        "threshold": threshold,
        "pass": pass,
        "total_relevant": total_relevant,
        "total_found": total_found,
        "per_query": per_query,
    });

    println!("{}", serde_json::to_string_pretty(&receipt).unwrap());

    assert!(
        pass,
        "Hybrid recall@10 = {overall_recall:.3} is below threshold {threshold}"
    );

    // Clean up.
    let _ = std::fs::remove_dir_all(&tmp);
}
