use semantic_memory::embedder::{EmbedBatchFuture, EmbedFuture};
#[cfg(feature = "admin-ops")]
use semantic_memory::StoragePaths;
use semantic_memory::{
    Embedder, MemoryConfig, MemoryStore, MultiFunctionEmbedding, MultiVectorEmbedding,
    OptionalMultiEmbedFuture, ReceiptMode, Role, SearchContext, SearchSource, SearchSourceType,
    SparseWeights,
};
use tempfile::TempDir;

#[derive(Clone, Copy)]
struct DeterministicSparseEmbedder;

fn representations(text: &str) -> (Vec<f32>, SparseWeights) {
    if text.ends_with("common") || text.contains("durable query") {
        (vec![1.0, 0.0], SparseWeights::from_entries(vec![(42, 1.0)]))
    } else if text.contains("dense winner") {
        (vec![1.0, 0.0], SparseWeights::from_entries(vec![(7, 1.0)]))
    } else if text.contains("sparse winner") || text.contains("durable target") {
        (vec![0.8, 0.6], SparseWeights::from_entries(vec![(42, 1.0)]))
    } else {
        (vec![0.0, 1.0], SparseWeights::from_entries(vec![(99, 1.0)]))
    }
}

impl Embedder for DeterministicSparseEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> EmbedFuture<'a> {
        let dense = representations(text).0;
        Box::pin(async move { Ok(dense) })
    }

    fn embed_batch<'a>(&'a self, texts: Vec<String>) -> EmbedBatchFuture<'a> {
        Box::pin(async move { Ok(texts.iter().map(|text| representations(text).0).collect()) })
    }

    fn model_name(&self) -> &str {
        "test-native-sparse"
    }

    fn dimensions(&self) -> usize {
        2
    }

    fn embed_multi_optional<'a>(&'a self, text: &'a str) -> OptionalMultiEmbedFuture<'a> {
        let (dense, sparse) = representations(text);
        Box::pin(async move {
            Ok(Some(MultiFunctionEmbedding {
                dense,
                sparse,
                multi_vec: MultiVectorEmbedding::from_token_vectors(Vec::new()),
            }))
        })
    }
}

#[derive(Clone, Copy)]
struct DeterministicDenseOnlyEmbedder;

impl Embedder for DeterministicDenseOnlyEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> EmbedFuture<'a> {
        let dense = representations(text).0;
        Box::pin(async move { Ok(dense) })
    }

    fn embed_batch<'a>(&'a self, texts: Vec<String>) -> EmbedBatchFuture<'a> {
        Box::pin(async move { Ok(texts.iter().map(|text| representations(text).0).collect()) })
    }

    fn model_name(&self) -> &str {
        "test-dense-only"
    }

    fn dimensions(&self) -> usize {
        2
    }
}

fn config(temp: &TempDir, sparse_weight: f64) -> MemoryConfig {
    let mut config = MemoryConfig {
        base_dir: temp.path().to_path_buf(),
        ..MemoryConfig::default()
    };
    config.embedding.dimensions = 2;
    config.search.sparse_weight = sparse_weight;
    config.search.sparse_top_k = 10;
    config.search.sparse_min_score = 0.0;
    config.search.min_similarity = -1.0;
    config.search.late_interaction_weight = 0.0;
    config
}

async fn seed_rank_pair(store: &MemoryStore) {
    store
        .add_fact("sparse-tests", "common dense winner", None, None)
        .await
        .unwrap();
    store
        .add_fact("sparse-tests", "common sparse winner", None, None)
        .await
        .unwrap();
}

#[tokio::test]
async fn sparse_third_signal_changes_ranking_and_is_inspectable() {
    let disabled_temp = TempDir::new().unwrap();
    let disabled = MemoryStore::open_with_embedder(
        config(&disabled_temp, 0.0),
        Box::new(DeterministicSparseEmbedder),
    )
    .unwrap();
    seed_rank_pair(&disabled).await;
    let disabled_results = disabled
        .search_explained("common", Some(2), Some(&["sparse-tests"]), None)
        .await
        .unwrap();
    assert_eq!(disabled_results[0].result.content, "common dense winner");
    assert!(disabled_results
        .iter()
        .all(|result| result.breakdown.sparse_rank.is_none()));

    let enabled_temp = TempDir::new().unwrap();
    let enabled = MemoryStore::open_with_embedder(
        config(&enabled_temp, 10.0),
        Box::new(DeterministicSparseEmbedder),
    )
    .unwrap();
    seed_rank_pair(&enabled).await;
    let mut context = SearchContext::default_now();
    context.receipt_mode = ReceiptMode::ReturnReceipt;
    let response = enabled
        .search_explained_with_context(
            "common",
            Some(2),
            Some(&["sparse-tests"]),
            Some(&[SearchSourceType::Facts]),
            context,
        )
        .await
        .unwrap();

    assert_eq!(response.results[0].result.content, "common sparse winner");
    assert_eq!(response.results[0].breakdown.sparse_rank, Some(1));
    assert!(response.results[0]
        .breakdown
        .sparse_contribution
        .is_some_and(|value| value > 0.0));
    let receipt = response.receipt.unwrap();
    assert!(receipt.sparse_enabled);
    assert_eq!(receipt.sparse_candidate_count, Some(1));
    assert_eq!(receipt.sparse_query_nonzero_count, Some(1));
    assert_eq!(receipt.sparse_representations, vec!["native_sparse"]);
    assert_eq!(receipt.sparse_result_ranks[0].rank, 1);
}

#[tokio::test]
async fn sparse_message_search_respects_session_filtering() {
    let temp = TempDir::new().unwrap();
    let store =
        MemoryStore::open_with_embedder(config(&temp, 5.0), Box::new(DeterministicSparseEmbedder))
            .unwrap();
    let first = store.create_session("test").await.unwrap();
    let second = store.create_session("test").await.unwrap();
    store
        .add_message_embedded(&first, Role::User, "first durable target", None, None)
        .await
        .unwrap();
    store
        .add_message_embedded(&second, Role::User, "second durable target", None, None)
        .await
        .unwrap();

    let results = store
        .search_conversations("durable query", Some(5), Some(&[second.as_str()]))
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "second durable target");
    assert!(matches!(
        &results[0].source,
        SearchSource::Message { session_id, .. } if session_id == &second
    ));
}

#[tokio::test]
async fn sparse_disabled_is_exactly_dense_only_compatible() {
    let sparse_temp = TempDir::new().unwrap();
    let sparse_capable = MemoryStore::open_with_embedder(
        config(&sparse_temp, 0.0),
        Box::new(DeterministicSparseEmbedder),
    )
    .unwrap();
    seed_rank_pair(&sparse_capable).await;

    let dense_temp = TempDir::new().unwrap();
    let dense_only = MemoryStore::open_with_embedder(
        config(&dense_temp, 0.0),
        Box::new(DeterministicDenseOnlyEmbedder),
    )
    .unwrap();
    seed_rank_pair(&dense_only).await;

    let sparse_results = sparse_capable
        .search_explained("common", Some(2), Some(&["sparse-tests"]), None)
        .await
        .unwrap();
    let dense_results = dense_only
        .search_explained("common", Some(2), Some(&["sparse-tests"]), None)
        .await
        .unwrap();
    let sparse_projection: Vec<_> = sparse_results
        .iter()
        .map(|result| (&result.result.content, result.result.score))
        .collect();
    let dense_projection: Vec<_> = dense_results
        .iter()
        .map(|result| (&result.result.content, result.result.score))
        .collect();
    assert_eq!(sparse_projection, dense_projection);
}

#[cfg(feature = "admin-ops")]
#[tokio::test]
async fn sparse_persistence_survives_reopen_and_delete_cleans_it() {
    let temp = TempDir::new().unwrap();
    let config = config(&temp, 5.0);
    let store =
        MemoryStore::open_with_embedder(config.clone(), Box::new(DeterministicSparseEmbedder))
            .unwrap();
    let fact_id = store
        .add_fact("sparse-tests", "durable target", None, None)
        .await
        .unwrap();
    drop(store);

    let reopened =
        MemoryStore::open_with_embedder(config.clone(), Box::new(DeterministicSparseEmbedder))
            .unwrap();
    let results = reopened
        .search_explained("durable query", Some(1), Some(&["sparse-tests"]), None)
        .await
        .unwrap();
    assert_eq!(results[0].result.content, "durable target");
    assert_eq!(results[0].breakdown.sparse_rank, Some(1));

    let sqlite_path = StoragePaths::new(&config.base_dir).sqlite_path;
    let connection = rusqlite::Connection::open(&sqlite_path).unwrap();
    let before: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sparse_vectors WHERE item_key = ?1",
            [format!("fact:{fact_id}")],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(before, 1);
    drop(connection);

    reopened.delete_fact(&fact_id).await.unwrap();
    drop(reopened);
    let connection = rusqlite::Connection::open(sqlite_path).unwrap();
    let after: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sparse_vectors WHERE item_key = ?1",
            [format!("fact:{fact_id}")],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(after, 0);
}
