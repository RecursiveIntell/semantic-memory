use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder};
use tempfile::TempDir;

fn test_store() -> (MemoryStore, TempDir) {
    let tmp = TempDir::new().unwrap();
    let config = MemoryConfig {
        base_dir: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let embedder = Box::new(MockEmbedder::new(768));
    let store = MemoryStore::open_with_embedder(config, embedder).unwrap();
    (store, tmp)
}

#[tokio::test]
async fn add_fact_and_get() {
    let (store, _tmp) = test_store();
    let fact_id = store
        .add_fact("general", "The sky is blue", None, None)
        .await
        .unwrap();

    let fact = store
        .get_fact(&fact_id)
        .await
        .unwrap()
        .expect("fact should exist");
    assert_eq!(fact.content, "The sky is blue");
    assert_eq!(fact.namespace, "general");
}

#[tokio::test]
async fn add_fact_with_source_and_metadata() {
    let (store, _tmp) = test_store();
    let metadata = serde_json::json!({"confidence": 0.9});
    let fact_id = store
        .add_fact(
            "user",
            "Josh likes Rust",
            Some("conversation:abc"),
            Some(metadata),
        )
        .await
        .unwrap();

    let fact = store.get_fact(&fact_id).await.unwrap().unwrap();
    assert_eq!(fact.namespace, "user");
    assert_eq!(fact.source.as_deref(), Some("conversation:abc"));
    assert!(fact.metadata.is_some());
}

#[tokio::test]
async fn add_fact_with_embedding_sync() {
    let (store, _tmp) = test_store();
    let embedding = vec![0.1f32; 768];
    let fact_id = store
        .add_fact_with_embedding("test", "Pre-embedded fact", &embedding, None, None)
        .await
        .unwrap();

    let fact = store.get_fact(&fact_id).await.unwrap().unwrap();
    assert_eq!(fact.content, "Pre-embedded fact");
}

#[tokio::test]
async fn fts_finds_inserted_fact() {
    let (store, _tmp) = test_store();
    store
        .add_fact("general", "Rust programming language", None, None)
        .await
        .unwrap();

    let results = store
        .search_fts_only("Rust programming", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty(), "FTS should find the fact");
    assert!(results[0].content.contains("Rust"));
}

#[tokio::test]
async fn update_fact_fts_reflects_new_content() {
    let (store, _tmp) = test_store();
    let fact_id = store
        .add_fact("general", "Old content about cats", None, None)
        .await
        .unwrap();

    // Should find old content
    let results = store
        .search_fts_only("cats", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty());

    // Update
    store
        .update_fact(&fact_id, "New content about dogs")
        .await
        .unwrap();

    // Should find new content
    let results = store
        .search_fts_only("dogs", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty());

    // Should NOT find old content
    let results = store
        .search_fts_only("cats", None, None, None)
        .await
        .unwrap();
    assert!(
        results.is_empty(),
        "FTS should not find old content after update"
    );
}

#[tokio::test]
async fn delete_fact_removes_from_fts() {
    let (store, _tmp) = test_store();
    let fact_id = store
        .add_fact("general", "Temporary fact about whales", None, None)
        .await
        .unwrap();

    let results = store
        .search_fts_only("whales", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty());

    store.delete_fact(&fact_id).await.unwrap();

    let results = store
        .search_fts_only("whales", None, None, None)
        .await
        .unwrap();
    assert!(results.is_empty(), "FTS should return nothing after delete");
}

#[tokio::test]
async fn bulk_insert_delete_fts_consistency() {
    let (store, _tmp) = test_store();

    // Insert 20 facts
    let mut ids = Vec::new();
    for i in 0..20 {
        let id = store
            .add_fact("bulk", &format!("Bulk fact number {}", i), None, None)
            .await
            .unwrap();
        ids.push(id);
    }

    // Delete first 10
    for id in &ids[..10] {
        store.delete_fact(id).await.unwrap();
    }

    // FTS should only find 10
    let results = store
        .search_fts_only("Bulk fact", Some(30), None, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 10, "Should have exactly 10 remaining facts");
}

#[tokio::test]
async fn namespace_filtering_on_list_facts() {
    let (store, _tmp) = test_store();
    store
        .add_fact("ns_a", "Fact in namespace A", None, None)
        .await
        .unwrap();
    store
        .add_fact("ns_b", "Fact in namespace B", None, None)
        .await
        .unwrap();
    store
        .add_fact("ns_a", "Another fact in namespace A", None, None)
        .await
        .unwrap();

    let facts_a = store.list_facts("ns_a", 100, 0).await.unwrap();
    assert_eq!(facts_a.len(), 2);

    let facts_b = store.list_facts("ns_b", 100, 0).await.unwrap();
    assert_eq!(facts_b.len(), 1);
}

#[tokio::test]
async fn delete_namespace() {
    let (store, _tmp) = test_store();
    store
        .add_fact("deleteme", "Fact 1", None, None)
        .await
        .unwrap();
    store
        .add_fact("deleteme", "Fact 2", None, None)
        .await
        .unwrap();
    store
        .add_fact("keepme", "Fact 3", None, None)
        .await
        .unwrap();

    let count = store.delete_namespace("deleteme").await.unwrap();
    assert_eq!(count, 2);

    let remaining = store.list_facts("deleteme", 100, 0).await.unwrap();
    assert!(remaining.is_empty());

    let kept = store.list_facts("keepme", 100, 0).await.unwrap();
    assert_eq!(kept.len(), 1);
}

#[tokio::test]
async fn embedding_blob_roundtrip() {
    let (store, _tmp) = test_store();
    let original: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001).collect();
    let fact_id = store
        .add_fact_with_embedding("test", "Roundtrip test", &original, None, None)
        .await
        .unwrap();

    // Verify the fact exists and content matches
    let fact = store.get_fact(&fact_id).await.unwrap().unwrap();
    assert_eq!(fact.content, "Roundtrip test");

    // Verify embedding roundtrip via embed helper
    let stored = store
        .get_fact_embedding(&fact_id)
        .await
        .unwrap()
        .expect("embedding should exist");
    assert_eq!(stored.len(), original.len());
    for (a, b) in stored.iter().zip(original.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Embedding values should match: {} vs {}",
            a,
            b
        );
    }
}

#[tokio::test]
async fn get_nonexistent_fact_returns_none() {
    let (store, _tmp) = test_store();
    let result = store.get_fact("nonexistent-uuid").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn delete_nonexistent_fact_returns_error() {
    let (store, _tmp) = test_store();
    let result = store.delete_fact("nonexistent-uuid").await;
    assert!(result.is_err());
}
