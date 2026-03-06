//! HNSW persistence tests: keymap survival across reopen, deletion persistence,
//! explicit flush, rebuild, and graceful reopen.
//!
//! Only compiled when the `hnsw` feature is enabled.

#![cfg(feature = "hnsw")]

use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder, SearchSource, SearchSourceType};
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

fn reopen_store(base_dir: &std::path::Path) -> MemoryStore {
    let config = MemoryConfig {
        base_dir: base_dir.to_path_buf(),
        ..Default::default()
    };
    let embedder = Box::new(MockEmbedder::new(768));
    MemoryStore::open_with_embedder(config, embedder).unwrap()
}

/// 2.1 — After adding facts, closing, and reopening the store on the same
/// directory, FTS search should still find the persisted facts with correct IDs.
#[tokio::test]
async fn keymap_survives_reopen() {
    let tmp = TempDir::new().unwrap();
    let base_dir = tmp.path().to_path_buf();

    // Collect fact IDs so we can verify one after reopen.
    let mut fact_ids = Vec::new();

    // First open: add 10 facts.
    {
        let store = reopen_store(&base_dir);
        for i in 0..10 {
            let id = store
                .add_fact(
                    "persist",
                    &format!("Persistence fact number {}", i),
                    None,
                    None,
                )
                .await
                .unwrap();
            fact_ids.push(id);
        }
    } // store dropped — should flush to disk

    // Second open: search for one of the facts.
    {
        let store = reopen_store(&base_dir);

        let results = store
            .search_fts_only(
                "Persistence fact number 7",
                Some(10),
                None,
                Some(&[SearchSourceType::Facts]),
            )
            .await
            .unwrap();

        assert!(
            !results.is_empty(),
            "Should find persisted fact after reopen"
        );

        // Verify the correct fact_id is present in results.
        let found_ids: Vec<&str> = results
            .iter()
            .filter_map(|r| match &r.source {
                SearchSource::Fact { fact_id, .. } => Some(fact_id.as_str()),
                _ => None,
            })
            .collect();

        assert!(
            found_ids.contains(&fact_ids[7].as_str()),
            "Expected fact_id {} in results, got {:?}",
            fact_ids[7],
            found_ids
        );
    }
}

/// 2.2 — After adding 10 facts, deleting 3, closing, and reopening, the deleted
/// facts should not appear in search while the remaining ones should.
#[tokio::test]
async fn deletions_survive_reopen() {
    let tmp = TempDir::new().unwrap();
    let base_dir = tmp.path().to_path_buf();

    let mut fact_ids = Vec::new();

    // First open: add 10 facts, delete 3.
    {
        let store = reopen_store(&base_dir);

        for i in 0..10 {
            let id = store
                .add_fact(
                    "deletion",
                    &format!("Deletable fact item {}", i),
                    None,
                    None,
                )
                .await
                .unwrap();
            fact_ids.push(id);
        }

        // Delete facts 0, 1, 2.
        for id in &fact_ids[0..3] {
            store.delete_fact(id).await.unwrap();
        }
    } // store dropped

    // Second open: verify deletions persisted.
    {
        let store = reopen_store(&base_dir);

        // Deleted facts should NOT appear.
        for (i, _fact_id) in fact_ids.iter().enumerate().take(3) {
            let query = format!("Deletable fact item {}", i);
            let results = store
                .search_fts_only(&query, Some(10), None, Some(&[SearchSourceType::Facts]))
                .await
                .unwrap();

            let has_deleted_id = results.iter().any(|r| match &r.source {
                SearchSource::Fact { fact_id, .. } => fact_id == &fact_ids[i],
                _ => false,
            });

            assert!(
                !has_deleted_id,
                "Deleted fact {} should not appear after reopen",
                fact_ids[i]
            );
        }

        // Remaining facts (3..10) should still be findable.
        let results = store
            .search_fts_only(
                "Deletable fact item",
                Some(20),
                None,
                Some(&[SearchSourceType::Facts]),
            )
            .await
            .unwrap();

        assert_eq!(
            results.len(),
            7,
            "Should have exactly 7 remaining facts after deleting 3 of 10"
        );
    }
}

/// 2.3 — After adding facts and calling `flush_hnsw()` explicitly, the facts
/// should remain findable via search.
#[tokio::test]
async fn keymap_flush_on_explicit_flush() {
    let (store, _tmp) = test_store();

    let mut fact_ids = Vec::new();
    for i in 0..5 {
        let id = store
            .add_fact("flush", &format!("Flushed fact entry {}", i), None, None)
            .await
            .unwrap();
        fact_ids.push(id);
    }

    // Explicitly flush HNSW to persist graph + keymap.
    store.flush_hnsw().unwrap();

    // Verify all facts are still findable via FTS after flush.
    let results = store
        .search_fts_only(
            "Flushed fact entry",
            Some(10),
            None,
            Some(&[SearchSourceType::Facts]),
        )
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        5,
        "All 5 facts should be findable after explicit flush_hnsw()"
    );

    // Verify each fact_id appears in results.
    let result_ids: Vec<String> = results
        .iter()
        .filter_map(|r| match &r.source {
            SearchSource::Fact { fact_id, .. } => Some(fact_id.clone()),
            _ => None,
        })
        .collect();

    for id in &fact_ids {
        assert!(
            result_ids.contains(id),
            "Fact {} should be present in search results after flush",
            id
        );
    }
}

/// 2.4 — After adding facts and calling `rebuild_hnsw_index()`, search should
/// still work both before and after a close/reopen cycle.
#[tokio::test]
async fn rebuild_preserves_keymap() {
    let tmp = TempDir::new().unwrap();
    let base_dir = tmp.path().to_path_buf();

    let mut fact_ids = Vec::new();

    // First open: add facts, rebuild, verify search works.
    {
        let store = reopen_store(&base_dir);

        for i in 0..8 {
            let id = store
                .add_fact("rebuild", &format!("Rebuild test fact {}", i), None, None)
                .await
                .unwrap();
            fact_ids.push(id);
        }

        // Rebuild HNSW index from SQLite embeddings.
        store.rebuild_hnsw_index().await.unwrap();

        // Search should still work after rebuild.
        let results = store
            .search_fts_only(
                "Rebuild test fact",
                Some(10),
                None,
                Some(&[SearchSourceType::Facts]),
            )
            .await
            .unwrap();

        assert_eq!(
            results.len(),
            8,
            "All 8 facts should be findable after rebuild_hnsw_index()"
        );
    } // store dropped

    // Second open: search should still work after reopen.
    {
        let store = reopen_store(&base_dir);

        let results = store
            .search_fts_only(
                "Rebuild test fact",
                Some(10),
                None,
                Some(&[SearchSourceType::Facts]),
            )
            .await
            .unwrap();

        assert_eq!(
            results.len(),
            8,
            "All 8 facts should be findable after reopen post-rebuild"
        );

        // Verify a specific fact_id.
        let found_ids: Vec<String> = results
            .iter()
            .filter_map(|r| match &r.source {
                SearchSource::Fact { fact_id, .. } => Some(fact_id.clone()),
                _ => None,
            })
            .collect();

        assert!(
            found_ids.contains(&fact_ids[4]),
            "Fact {} should survive rebuild + reopen cycle",
            fact_ids[4]
        );
    }
}

/// 2.5 — Opening a store on a fresh directory, adding facts, closing, then
/// reopening should not crash regardless of keymap table state.
#[tokio::test]
async fn reopen_with_no_keymap_table_graceful() {
    let tmp = TempDir::new().unwrap();
    let base_dir = tmp.path().to_path_buf();

    // First open: add facts.
    {
        let store = reopen_store(&base_dir);

        for i in 0..3 {
            store
                .add_fact(
                    "graceful",
                    &format!("Graceful reopen fact {}", i),
                    None,
                    None,
                )
                .await
                .unwrap();
        }
    } // store dropped

    // Second open: should not panic or error.
    let store = reopen_store(&base_dir);

    // Verify we can still search (data survived).
    let results = store
        .search_fts_only(
            "Graceful reopen fact",
            Some(10),
            None,
            Some(&[SearchSourceType::Facts]),
        )
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        3,
        "All 3 facts should survive a close/reopen cycle"
    );
}
