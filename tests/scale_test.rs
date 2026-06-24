//! Production scale validation tests.
//!
//! These tests verify behavior at scale: bulk ingestion throughput,
//! search latency at various data sizes, and concurrent access safety.
//!
//! Run with: cargo test --all-features -- scale_test -- --ignored

#![cfg(all(
    feature = "usearch-backend",
    feature = "candle-embedder"
))]

use semantic_memory::{
    EmbeddingConfig, MemoryConfig, MemoryStore, SearchConfig,
    embedder::MockEmbedder,
};
use std::sync::Arc;
use tempfile::TempDir;

fn open_store(tmp: &TempDir) -> MemoryStore {
    let config = MemoryConfig {
        base_dir: tmp.path().to_path_buf(),
        embedding: EmbeddingConfig {
            model: "test".to_string(),
            dimensions: 768,
            batch_size: 32,
            timeout_secs: 60,
            ollama_url: "http://localhost:11434".to_string(),
        },
        search: SearchConfig::default(),
        ..Default::default()
    };
    let embedder: Box<dyn semantic_memory::embedder::Embedder> =
        Box::new(MockEmbedder::new(768));
    MemoryStore::open_with_embedder(config, embedder).expect("open store")
}

#[tokio::test]
#[ignore]
async fn bulk_ingestion_100_facts() {
    let tmp = TempDir::new().expect("tempdir");
    let store = open_store(&tmp);

    let start = std::time::Instant::now();

    for i in 0..100 {
        let content = format!(
            "Fact number {} about topic {} with keywords alpha-{} beta-{} gamma-{}",
            i, i % 10, i, i * 2, i * 3
        );
        store
            .add_fact("scale_test", &content, None, None)
            .await
            .expect("add fact");
    }

    let elapsed = start.elapsed();
    println!("100 facts ingested in {:?}", elapsed);
    println!("Throughput: {:.1} facts/second", 100.0 / elapsed.as_secs_f64());

    // Verify search still works
    let results = store
        .search("alpha keywords", Some(5), None, None)
        .await
        .expect("search");
    assert!(!results.is_empty(), "search should return results after bulk ingestion");

    // Check stats
    let stats = store.stats().await.expect("stats");
    println!("Stats: {} facts, {} chunks", stats.total_facts, stats.total_chunks);
    assert!(stats.total_facts >= 100, "should have at least 100 facts");
}

#[tokio::test]
#[ignore]
async fn concurrent_search_and_add_4_threads() {
    let tmp = TempDir::new().expect("tempdir");
    let store = Arc::new(open_store(&tmp));

    // Pre-populate with some data
    for i in 0..20 {
        let content = format!("Pre-existing fact {} about concurrent testing", i);
        store
            .add_fact("concurrent_test", &content, None, None)
            .await
            .expect("pre-populate");
    }

    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let store_clone = store.clone();
        let handle = tokio::spawn(async move {
            let mut errors = 0;
            for i in 0..25 {
                // Alternate between search and add
                if i % 2 == 0 {
                    let content = format!(
                        "Thread {} fact {} about concurrency and parallel access",
                        thread_id, i
                    );
                    if store_clone
                        .add_fact("concurrent_test", &content, None, None)
                        .await
                        .is_err()
                    {
                        errors += 1;
                    }
                } else {
                    if store_clone
                        .search("concurrent", Some(5), None, None)
                        .await
                        .is_err()
                    {
                        errors += 1;
                    }
                }
            }
            errors
        });
        handles.push(handle);
    }

    let mut total_errors = 0;
    for handle in handles {
        let errors = handle.await.expect("join");
        total_errors += errors;
    }

    assert_eq!(total_errors, 0, "no errors should occur during concurrent access");

    // Verify data integrity
    let stats = store.stats().await.expect("stats");
    println!(
        "After concurrent test: {} facts (expected >= 20 + 50 = 70)",
        stats.total_facts
    );
    assert!(stats.total_facts >= 70, "should have at least 70 facts after concurrent adds");
}