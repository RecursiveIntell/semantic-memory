#![allow(clippy::expect_used)]

#[tokio::main]
async fn main() {
    use semantic_memory::config::EmbeddingConfig;
    use semantic_memory::{MemoryConfig, MemoryStore};
    use std::path::PathBuf;

    let db_dir = "/home/sikmindz/.hermes/semantic-memory.db";

    println!("Opening MemoryStore — pending_index_ops has 22420 entries...");
    println!("This should trigger recover_hnsw_sidecar_sync on startup.");

    let config = MemoryConfig {
        base_dir: PathBuf::from(db_dir),
        embedding: EmbeddingConfig {
            ollama_url: "http://192.168.50.69:11434".to_string(),
            model: "nomic-embed-text:latest".to_string(),
            dimensions: 768,
            batch_size: 32,
            timeout_secs: 60,
        },
        ..Default::default()
    };

    let store = MemoryStore::open(config).expect("Failed to open store");
    println!("Store opened.");

    // Check the HNSW index state by running a search
    println!("\nRunning test search...");
    let results = store
        .search(
            "turbo-quant compression codec kv-cache",
            Some(5),
            None,
            None,
        )
        .await
        .expect("search");
    println!("Search returned {} results", results.len());
    for (i, r) in results.iter().enumerate() {
        println!(
            "  [{}] vector_rank={:?} cos_sim={:?} content={:.60}",
            i, r.vector_rank, r.cosine_similarity, r.content
        );
    }

    let stats = store.stats().await.expect("stats");
    println!(
        "\nStats: facts={}, chunks={}, docs={}, db={:.1}MB",
        stats.total_facts,
        stats.total_chunks,
        stats.total_documents,
        stats.database_size_bytes as f64 / 1024.0 / 1024.0
    );

    println!("\nDone. Check hnsw_keymap and sidecar files.");
}
