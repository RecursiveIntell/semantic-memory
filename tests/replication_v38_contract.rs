use rusqlite::Connection;
use semantic_memory::journal::{
    export_verified_contiguous, ExportStatus, FactCreatePayloadV1, FACT_CREATE_OPERATION,
    FACT_CREATE_PAYLOAD_SCHEMA, GENESIS_PREDECESSOR,
};
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder, ReplicationMode};
use tempfile::TempDir;

fn enabled_store() -> (MemoryStore, TempDir) {
    let temp = TempDir::new().unwrap();
    let store = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: temp.path().to_path_buf(),
            journal_device_id: Some("device-1".to_string()),
            journal_store_id: Some("store-1".to_string()),
            replication_mode: ReplicationMode::FactCreateRequired,
            replication_stream_epoch: 9,
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )
    .unwrap();
    (store, temp)
}

#[tokio::test]
async fn real_fact_paths_commit_exact_verified_outbox_and_dedup_emits_once() {
    let (store, temp) = enabled_store();
    let metadata = serde_json::json!({"origin": "contract-test"});
    let first_id = store
        .add_fact(
            "replication-contract",
            "canonical fact-create content",
            Some("test-source"),
            Some(metadata.clone()),
        )
        .await
        .unwrap();
    let duplicate_id = store
        .add_fact(
            "replication-contract",
            "canonical fact-create content",
            Some("test-source"),
            Some(metadata.clone()),
        )
        .await
        .unwrap();
    assert_eq!(duplicate_id, first_id);

    let embedded_id = store
        .add_fact_with_embedding(
            "replication-contract",
            "precomputed fact-create content",
            &vec![0.0; 768],
            Some("test-source"),
            Some(metadata),
        )
        .await
        .unwrap();

    let conn = Connection::open(temp.path().join("memory.db")).unwrap();
    let fact_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM facts WHERE namespace = 'replication-contract'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(fact_count, 2);

    let batch = export_verified_contiguous(&conn, "device-1", "store-1", 9, 1, 10).unwrap();
    assert_eq!(batch.status, ExportStatus::End);
    assert_eq!(batch.entries.len(), 2);
    assert_eq!(batch.entries[0].sequence, 1);
    assert_eq!(batch.entries[1].sequence, 2);
    assert_eq!(batch.entries[0].predecessor_digest, GENESIS_PREDECESSOR);
    assert_eq!(
        batch.entries[1].predecessor_digest,
        batch.entries[0].envelope_digest
    );
    for entry in &batch.entries {
        assert_eq!(entry.operation_kind, FACT_CREATE_OPERATION);
        assert_eq!(entry.payload_schema, FACT_CREATE_PAYLOAD_SCHEMA);
    }

    let first_payload: FactCreatePayloadV1 =
        serde_json::from_slice(&batch.entries[0].payload).unwrap();
    assert_eq!(first_payload.fact_id, first_id);
    assert_eq!(first_payload.namespace, "replication-contract");
    assert_eq!(first_payload.content, "canonical fact-create content");
    assert_eq!(first_payload.source.as_deref(), Some("test-source"));

    let second_payload: FactCreatePayloadV1 =
        serde_json::from_slice(&batch.entries[1].payload).unwrap();
    assert_eq!(second_payload.fact_id, embedded_id);
}

#[tokio::test]
async fn disabled_mode_commits_fact_without_outbox() {
    let temp = TempDir::new().unwrap();
    let store = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: temp.path().to_path_buf(),
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )
    .unwrap();
    store
        .add_fact("local-only", "not replicated", None, None)
        .await
        .unwrap();

    let conn = Connection::open(temp.path().join("memory.db")).unwrap();
    let facts: i64 = conn
        .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
        .unwrap();
    let outbox: i64 = conn
        .query_row("SELECT COUNT(*) FROM mutation_journal", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(facts, 1);
    assert_eq!(outbox, 0);
}

#[tokio::test]
async fn outbox_failure_rolls_back_fact_and_does_not_consume_sequence() {
    let (store, temp) = enabled_store();
    let conn = Connection::open(temp.path().join("memory.db")).unwrap();
    conn.execute_batch(
        "CREATE TRIGGER reject_verified_outbox
         BEFORE INSERT ON mutation_journal
         BEGIN
             SELECT RAISE(ABORT, 'injected outbox failure');
         END;",
    )
    .unwrap();

    let result = store
        .add_fact(
            "replication-contract",
            "must roll back with outbox",
            None,
            None,
        )
        .await;
    assert!(result.is_err());

    let facts: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM facts WHERE content = 'must roll back with outbox'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let next_stream: Option<i64> = conn
        .query_row(
            "SELECT next_sequence FROM replication_streams
             WHERE home_device_id = 'device-1' AND store_id = 'store-1' AND stream_epoch = 9",
            [],
            |row| row.get(0),
        )
        .ok();
    assert_eq!(facts, 0);
    assert!(next_stream.is_none() || next_stream == Some(1));
}
