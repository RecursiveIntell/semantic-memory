use semantic_memory::search::{cosine_similarity, sanitize_fts_query};
use semantic_memory::{
    MemoryConfig, MemoryStore, MockEmbedder, SearchConfig, SearchSource, SearchSourceType,
};
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

// ─── Cosine Similarity ─────────────────────────────────────────

#[test]
fn cosine_identical_vectors() {
    let v = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&v, &v);
    assert!(
        (sim - 1.0).abs() < 0.001,
        "Identical vectors should have similarity ~1.0, got {}",
        sim
    );
}

#[test]
fn cosine_orthogonal_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(
        sim.abs() < 0.001,
        "Orthogonal vectors should have similarity ~0.0, got {}",
        sim
    );
}

#[test]
fn cosine_opposite_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    let sim = cosine_similarity(&a, &b);
    assert!(
        (sim + 1.0).abs() < 0.001,
        "Opposite vectors should have similarity ~-1.0, got {}",
        sim
    );
}

#[test]
fn cosine_zero_vector() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0, "Zero vector should return 0.0 similarity");
}

// ─── FTS Query Sanitization ────────────────────────────────────

#[test]
fn sanitize_strips_fts_operators() {
    let result = sanitize_fts_query("hello \"world\" + test");
    assert_eq!(result, Some("hello world test".to_string()));
}

#[test]
fn sanitize_empty_after_stripping() {
    let result = sanitize_fts_query("\"*+-()^{}~:");
    assert_eq!(result, None);
}

#[test]
fn sanitize_normal_query_unchanged() {
    let result = sanitize_fts_query("hello world");
    assert_eq!(result, Some("hello world".to_string()));
}

#[test]
fn sanitize_unicode_preserved() {
    let result = sanitize_fts_query("中文 搜索");
    assert_eq!(result, Some("中文 搜索".to_string()));
}

#[test]
fn sanitize_empty_string() {
    assert_eq!(sanitize_fts_query(""), None);
}

#[test]
fn sanitize_only_whitespace() {
    assert_eq!(sanitize_fts_query("   "), None);
}

// ─── RRF Fusion ────────────────────────────────────────────────

#[test]
fn rrf_fusion_order() {
    // From SPEC.md §13:
    // BM25 results: [A(rank 1), B(rank 2), C(rank 3)]
    // Vector results: [B(rank 1), D(rank 2), A(rank 3)]
    // With k=60, weights=1.0:
    //   A: 1/61 + 1/63 = 0.01639 + 0.01587 = 0.03226
    //   B: 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252  <-- highest
    //   C: 1/63 + 0    = 0.01587
    //   D: 0    + 1/62 = 0.01613
    // Expected order: B, A, D, C

    use semantic_memory::search::{rrf_fuse, Bm25Hit, VectorHit};
    use semantic_memory::{SearchConfig, SearchSource};

    let make_fact_source = |id: &str| SearchSource::Fact {
        fact_id: id.to_string(),
        namespace: "test".to_string(),
    };

    let bm25_hits = vec![
        Bm25Hit {
            id: "A".to_string(),
            content: "content A".to_string(),
            source: make_fact_source("A"),
            updated_at: None,
        },
        Bm25Hit {
            id: "B".to_string(),
            content: "content B".to_string(),
            source: make_fact_source("B"),
            updated_at: None,
        },
        Bm25Hit {
            id: "C".to_string(),
            content: "content C".to_string(),
            source: make_fact_source("C"),
            updated_at: None,
        },
    ];

    let vector_hits = vec![
        VectorHit {
            id: "B".to_string(),
            content: "content B".to_string(),
            source: make_fact_source("B"),
            similarity: 0.9,
            updated_at: None,
        },
        VectorHit {
            id: "D".to_string(),
            content: "content D".to_string(),
            source: make_fact_source("D"),
            similarity: 0.8,
            updated_at: None,
        },
        VectorHit {
            id: "A".to_string(),
            content: "content A".to_string(),
            source: make_fact_source("A"),
            similarity: 0.7,
            updated_at: None,
        },
    ];

    let config = SearchConfig::default();
    let results = rrf_fuse(&bm25_hits, &vector_hits, &config, 10);

    assert_eq!(results.len(), 4);

    // Extract IDs in order
    let ids: Vec<String> = results
        .iter()
        .map(|r| match &r.source {
            SearchSource::Fact { fact_id, .. } => fact_id.clone(),
            SearchSource::Chunk { chunk_id, .. } => chunk_id.clone(),
            SearchSource::Message { message_id, .. } => message_id.to_string(),
        })
        .collect();

    assert_eq!(ids, vec!["B", "A", "D", "C"]);

    // Verify B has the highest score
    assert!(results[0].score > results[1].score);
}

// ─── Full Integration ──────────────────────────────────────────

#[tokio::test]
async fn hybrid_search_finds_facts() {
    let (store, _tmp) = test_store();

    store
        .add_fact(
            "general",
            "Rust is a systems programming language",
            None,
            None,
        )
        .await
        .unwrap();
    store
        .add_fact("general", "Python is great for data science", None, None)
        .await
        .unwrap();
    store
        .add_fact("general", "JavaScript runs in browsers", None, None)
        .await
        .unwrap();

    let results = store
        .search("systems programming", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty(), "Hybrid search should return results");
}

#[tokio::test]
async fn fts_only_search() {
    let (store, _tmp) = test_store();

    store
        .add_fact(
            "general",
            "Rust is a systems programming language",
            None,
            None,
        )
        .await
        .unwrap();
    store
        .add_fact("general", "Python is great for data science", None, None)
        .await
        .unwrap();

    let results = store
        .search_fts_only("Rust systems", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].content.contains("Rust"));
}

#[tokio::test]
async fn search_with_namespace_filter() {
    let (store, _tmp) = test_store();

    store
        .add_fact("ns_a", "Fact in namespace A about dogs", None, None)
        .await
        .unwrap();
    store
        .add_fact("ns_b", "Fact in namespace B about dogs", None, None)
        .await
        .unwrap();

    let results = store
        .search_fts_only("dogs", None, Some(&["ns_a"]), None)
        .await
        .unwrap();
    assert_eq!(results.len(), 1, "Should only find fact in namespace A");
}

#[tokio::test]
async fn search_with_source_type_filter() {
    let (store, _tmp) = test_store();

    store
        .add_fact(
            "general",
            "This is a fact about quantum physics",
            None,
            None,
        )
        .await
        .unwrap();

    // Search only facts
    let results = store
        .search_fts_only(
            "quantum physics",
            None,
            None,
            Some(&[SearchSourceType::Facts]),
        )
        .await
        .unwrap();
    assert!(!results.is_empty());

    // Search only chunks (should be empty since we only have facts)
    let results = store
        .search_fts_only(
            "quantum physics",
            None,
            None,
            Some(&[SearchSourceType::Chunks]),
        )
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn empty_query_returns_empty_results() {
    let (store, _tmp) = test_store();
    store
        .add_fact("general", "Some content", None, None)
        .await
        .unwrap();

    let results = store.search_fts_only("", None, None, None).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn special_chars_only_query_returns_empty() {
    let (store, _tmp) = test_store();
    store
        .add_fact("general", "Some content", None, None)
        .await
        .unwrap();

    let results = store
        .search_fts_only("\"*+-()^{}~:", None, None, None)
        .await
        .unwrap();
    assert!(results.is_empty());
}

// ─── Parameterized Namespace Filtering (Fix 1) ───────────────

#[tokio::test]
async fn parameterized_namespace_adversarial() {
    let (store, _tmp) = test_store();

    store
        .add_fact("safe", "Safe fact about cats", None, None)
        .await
        .unwrap();
    store
        .add_fact("also-safe", "Also safe fact about cats", None, None)
        .await
        .unwrap();
    // Adversarial namespace with a single quote
    store
        .add_fact(
            "it's-a-test",
            "Adversarial namespace fact about cats",
            None,
            None,
        )
        .await
        .unwrap();

    // Search with adversarial namespace — should find it, not crash
    let results = store
        .search_fts_only("cats", None, Some(&["it's-a-test"]), None)
        .await
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "Should find fact in adversarial namespace"
    );
    assert!(results[0].content.contains("Adversarial"));

    // Search with safe namespace — should only find safe fact
    let results = store
        .search_fts_only("cats", None, Some(&["safe"]), None)
        .await
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "Should only find fact in 'safe' namespace"
    );
    assert!(results[0].content.contains("Safe fact"));
}

// ─── Content Deduplication (Fix 6) ───────────────────────────

#[tokio::test]
async fn dedup_removes_duplicate_content() {
    let (store, _tmp) = test_store();

    // Add a fact with specific content
    store
        .add_fact("general", "Rust was released in 2015", None, None)
        .await
        .unwrap();

    // Ingest a document with a chunk containing the exact same text
    store
        .ingest_document(
            "Rust History",
            "Rust was released in 2015",
            "general",
            None,
            None,
        )
        .await
        .unwrap();

    // After provenance-based dedup, both the fact and chunk are kept
    // because they come from different source types (Fact vs Chunk).
    let results = store
        .search("Rust released", None, None, None)
        .await
        .unwrap();
    assert_eq!(
        results.len(),
        2,
        "Should keep results from different source types even with identical content"
    );
}

#[tokio::test]
async fn dedup_keeps_different_content() {
    let (store, _tmp) = test_store();

    store
        .add_fact(
            "general",
            "Rust was released as a language in 2015",
            None,
            None,
        )
        .await
        .unwrap();
    store
        .add_fact(
            "general",
            "Go was released as a language in 2009",
            None,
            None,
        )
        .await
        .unwrap();

    // FTS will find both since they share the word "released" and "language"
    let results = store
        .search_fts_only("released language", None, None, None)
        .await
        .unwrap();
    assert_eq!(
        results.len(),
        2,
        "Should keep both results since content is different"
    );
}

// ─── Recency Weighting (Fix 3) ──────────────────────────────

fn test_store_with_recency(
    half_life: Option<f64>,
    recency_weight: f64,
) -> (MemoryStore, TempDir) {
    let tmp = TempDir::new().unwrap();
    let config = MemoryConfig {
        base_dir: tmp.path().to_path_buf(),
        search: SearchConfig {
            recency_half_life_days: half_life,
            recency_weight,
            ..Default::default()
        },
        ..Default::default()
    };
    let embedder = Box::new(MockEmbedder::new(768));
    let store = MemoryStore::open_with_embedder(config, embedder).unwrap();
    (store, tmp)
}

#[tokio::test]
async fn recency_disabled_no_effect() {
    // recency_half_life_days: None → same behavior as V1
    let (store, _tmp) = test_store_with_recency(None, 0.5);

    store
        .add_fact("general", "Recency test fact alpha", None, None)
        .await
        .unwrap();

    let results = store
        .search_fts_only("Recency test fact", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty());
    // Score should be purely BM25-based with no recency component
    let expected_score = 1.0 / (60.0 + 1.0); // bm25_weight / (rrf_k + rank)
    assert!(
        (results[0].score - expected_score).abs() < 0.0001,
        "Score should be pure BM25 RRF score without recency, got {} expected {}",
        results[0].score,
        expected_score
    );
}

#[cfg(feature = "testing")]
#[tokio::test]
async fn recency_boosts_recent_facts() {
    let (store, _tmp) = test_store_with_recency(Some(30.0), 0.5);

    // Add two facts with the same content relevance
    let fact_a_id = store
        .add_fact(
            "general",
            "Recency quantum computing breakthrough",
            None,
            None,
        )
        .await
        .unwrap();
    let fact_b_id = store
        .add_fact("general", "Recency quantum computing discovery", None, None)
        .await
        .unwrap();

    // Set fact B to 60 days ago
    let sixty_days_ago = (chrono::Utc::now() - chrono::Duration::days(60))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    store
        .raw_execute(
            "UPDATE facts SET updated_at = ?1 WHERE id = ?2",
            vec![sixty_days_ago, fact_b_id.clone()],
        )
        .await
        .unwrap();

    let results = store
        .search("quantum computing", None, None, None)
        .await
        .unwrap();
    assert!(results.len() >= 2, "Should find both facts");

    // Find scores for each fact
    let score_a = results
        .iter()
        .find(|r| match &r.source {
            SearchSource::Fact { fact_id, .. } => fact_id == &fact_a_id,
            _ => false,
        })
        .map(|r| r.score);
    let score_b = results
        .iter()
        .find(|r| match &r.source {
            SearchSource::Fact { fact_id, .. } => fact_id == &fact_b_id,
            _ => false,
        })
        .map(|r| r.score);

    assert!(
        score_a.unwrap() > score_b.unwrap(),
        "Recent fact A ({}) should score higher than old fact B ({})",
        score_a.unwrap(),
        score_b.unwrap()
    );
}

#[tokio::test]
async fn recency_zero_half_life_no_panic() {
    let (store, _tmp) = test_store_with_recency(Some(0.0), 0.5);

    store
        .add_fact("general", "Zero half life test fact", None, None)
        .await
        .unwrap();

    // Should not panic or produce NaN
    let results = store
        .search("half life test", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(
        results[0].score.is_finite(),
        "Score should be finite with zero half-life, got {}",
        results[0].score
    );
}

// ─── V2: Buffer Reuse Correctness (Fix 6 regression) ────────

#[tokio::test]
async fn test_vector_search_buffer_reuse_correctness() {
    let (store, _tmp) = test_store();

    // Insert 100 facts with known embeddings
    for i in 0..100 {
        store
            .add_fact(
                "general",
                &format!("Buffer reuse test fact number {}", i),
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Search and verify results are returned correctly
    let results = store
        .search("Buffer reuse test fact", None, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty(), "Should find facts with buffer reuse");

    // Verify scores are valid (not NaN, not infinite)
    for result in &results {
        assert!(
            result.score.is_finite(),
            "Score should be finite, got {}",
            result.score
        );
        assert!(
            result.score >= 0.0,
            "Score should be non-negative, got {}",
            result.score
        );
    }

    // Verify results are ordered by score descending
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be ordered by score descending: {} < {}",
            results[i - 1].score,
            results[i].score
        );
    }
}

// ─── V2: Large Row Count Warning (Fix 9) ────────────────────

#[tokio::test]
async fn test_vector_search_completes_with_many_rows() {
    let (store, _tmp) = test_store();

    // Insert 100 facts (can't easily test 50K in unit tests)
    for i in 0..100 {
        store
            .add_fact(
                "general",
                &format!("Row count test fact number {}", i),
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Search should succeed — the warning threshold is about logging, not blocking
    let results = store
        .search("Row count test fact", None, None, None)
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "Search should complete successfully with many rows"
    );
}
