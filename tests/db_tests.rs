use semantic_memory::db::{bytes_to_embedding, embedding_to_bytes};
use semantic_memory::embedder::parse_embedding_response;
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder};
use tempfile::TempDir;

// ─── bytes_to_embedding (Fix 1: stable Rust compat) ─────────

#[test]
fn test_bytes_to_embedding_valid() {
    let original = vec![1.0f32, 2.0, 3.0];
    let bytes = embedding_to_bytes(&original);
    let decoded = bytes_to_embedding(&bytes).unwrap();
    assert_eq!(original, decoded);
}

#[test]
fn test_bytes_to_embedding_invalid_length() {
    let bytes = vec![0u8; 5]; // Not divisible by 4
    let result = bytes_to_embedding(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_bytes_to_embedding_empty() {
    let bytes: Vec<u8> = vec![];
    let decoded = bytes_to_embedding(&bytes).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn test_bytes_to_embedding_roundtrip_large() {
    let original: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001).collect();
    let bytes = embedding_to_bytes(&original);
    let decoded = bytes_to_embedding(&bytes).unwrap();
    assert_eq!(original.len(), decoded.len());
    for (a, b) in original.iter().zip(decoded.iter()) {
        assert!((a - b).abs() < 1e-7, "Values should match: {} vs {}", a, b);
    }
}

// ─── embeddings_dirty default is false ──────────────────────

#[tokio::test]
async fn test_fresh_db_not_dirty() {
    let tmp = TempDir::new().unwrap();
    let config = MemoryConfig {
        base_dir: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let embedder = Box::new(MockEmbedder::new(768));
    let store = MemoryStore::open_with_embedder(config, embedder).unwrap();
    assert!(!store.embeddings_are_dirty().await.unwrap());
}

// ─── parse_embedding_response (Fix 3) ───────────────────────

#[test]
fn test_parse_rejects_non_numeric() {
    let body = serde_json::json!({
        "embeddings": [[1.0, "bad", 3.0]]
    });
    let result = parse_embedding_response(&body, 3);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("non-numeric"),
        "Error should mention non-numeric value"
    );
}

#[test]
fn test_parse_valid_embedding() {
    let body = serde_json::json!({
        "embeddings": [[1.0, 2.0, 3.0]]
    });
    let result = parse_embedding_response(&body, 3);
    assert!(result.is_ok());
    assert_eq!(result.unwrap()[0], vec![1.0f32, 2.0, 3.0]);
}

#[test]
fn test_parse_wrong_dimensions() {
    let body = serde_json::json!({
        "embeddings": [[1.0, 2.0, 3.0]]
    });
    // Expect 5 dims but got 3
    let result = parse_embedding_response(&body, 5);
    assert!(result.is_err());
}

#[test]
fn test_parse_multiple_embeddings() {
    let body = serde_json::json!({
        "embeddings": [[1.0, 2.0], [3.0, 4.0]]
    });
    let result = parse_embedding_response(&body, 2);
    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0], vec![1.0f32, 2.0]);
    assert_eq!(embeddings[1], vec![3.0f32, 4.0]);
}

// ─── Role trait impls (Fix 7) ───────────────────────────────

#[test]
fn test_role_display() {
    use semantic_memory::Role;
    assert_eq!(format!("{}", Role::User), "user");
    assert_eq!(format!("{}", Role::Assistant), "assistant");
    assert_eq!(format!("{}", Role::System), "system");
    assert_eq!(format!("{}", Role::Tool), "tool");
}

#[test]
fn test_role_from_str() {
    use semantic_memory::Role;
    assert_eq!("user".parse::<Role>().unwrap(), Role::User);
    assert_eq!("assistant".parse::<Role>().unwrap(), Role::Assistant);
    assert_eq!("system".parse::<Role>().unwrap(), Role::System);
    assert_eq!("tool".parse::<Role>().unwrap(), Role::Tool);
    assert!("invalid".parse::<Role>().is_err());
}
