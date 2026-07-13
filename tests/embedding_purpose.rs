use semantic_memory::{EmbedBatchFuture, EmbedFuture, Embedder, MemoryConfig, MemoryStore};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

#[derive(Clone)]
struct AsymmetricEmbedder {
    calls: Arc<Mutex<Vec<String>>>,
}

impl Embedder for AsymmetricEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> EmbedFuture<'a> {
        let calls = Arc::clone(&self.calls);
        Box::pin(async move {
            calls.lock().unwrap().push(text.to_string());
            Ok(if text.starts_with("search_query:") {
                vec![1.0, 0.0, 0.0]
            } else {
                vec![0.0, 1.0, 0.0]
            })
        })
    }

    fn embed_batch<'a>(&'a self, texts: Vec<String>) -> EmbedBatchFuture<'a> {
        Box::pin(async move {
            let mut output = Vec::with_capacity(texts.len());
            for text in &texts {
                output.push(self.embed(text).await?);
            }
            Ok(output)
        })
    }

    fn model_name(&self) -> &str {
        "asymmetric-purpose-test"
    }

    fn dimensions(&self) -> usize {
        3
    }
}

#[tokio::test]
async fn writes_use_document_queries_use_query_and_cache_keys_do_not_alias() {
    let temp = TempDir::new().unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut config = MemoryConfig {
        base_dir: temp.path().to_path_buf(),
        ..Default::default()
    };
    config.embedding.dimensions = 3;
    config.embedding.model = "asymmetric-purpose-test".into();
    let store = MemoryStore::open_with_embedder(
        config,
        Box::new(AsymmetricEmbedder {
            calls: Arc::clone(&calls),
        }),
    )
    .unwrap();

    store
        .add_fact("purpose", "same text", None, None)
        .await
        .unwrap();
    let query = store.embed_query("same text").await.unwrap();
    let document = store.embed_document("same text").await.unwrap();

    assert_ne!(query, document);
    let calls = calls.lock().unwrap();
    assert!(calls
        .iter()
        .any(|call| call == "search_document: same text"));
    assert!(calls.iter().any(|call| call == "search_query: same text"));
}
