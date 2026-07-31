use rusqlite::Connection;
use semantic_memory::journal::{
    encode_fact_create_payload, envelope_digest, export_verified_contiguous, payload_digest,
    ExportStatus, FactCreatePayloadV1, FactCreateReplicaEnvelopeV1, JournalEntry,
    ReplicaApplyOutcome, FACT_CREATE_OPERATION, FACT_CREATE_PAYLOAD_SCHEMA, GENESIS_PREDECESSOR,
    MIGRATION_V37, MIGRATION_V38, MIGRATION_V39,
};
use semantic_memory::{
    EmbedBatchFuture, EmbedFuture, Embedder, MemoryConfig, MemoryError, MemoryStore, MockEmbedder,
    ReplicationMode,
};
use tempfile::TempDir;

struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn embed<'a>(&'a self, _text: &'a str) -> EmbedFuture<'a> {
        Box::pin(async {
            Err(MemoryError::EmbedderUnavailable(
                "injected embedding outage".to_string(),
            ))
        })
    }

    fn embed_batch<'a>(&'a self, _texts: Vec<String>) -> EmbedBatchFuture<'a> {
        Box::pin(async {
            Err(MemoryError::EmbedderUnavailable(
                "injected embedding outage".to_string(),
            ))
        })
    }

    fn model_name(&self) -> &str {
        "mock-embedder"
    }

    fn dimensions(&self) -> usize {
        768
    }
}

fn envelope_at_epoch(
    stream_epoch: u64,
    sequence: i64,
    predecessor_digest: [u8; 32],
    fact_id: &str,
    content: &str,
) -> FactCreateReplicaEnvelopeV1 {
    let payload = encode_fact_create_payload(&FactCreatePayloadV1 {
        fact_id: fact_id.to_string(),
        namespace: "replica-contract".to_string(),
        content: content.to_string(),
        source: Some("device-primary".to_string()),
        metadata: Some(serde_json::json!({"contract": "v39"})),
    })
    .unwrap();
    let payload_hash = payload_digest(&payload);
    let envelope_hash = envelope_digest(
        "device-1",
        "store-1",
        stream_epoch,
        sequence,
        FACT_CREATE_OPERATION,
        FACT_CREATE_PAYLOAD_SCHEMA,
        &predecessor_digest,
        &payload_hash,
    );
    FactCreateReplicaEnvelopeV1 {
        home_device_id: "device-1".to_string(),
        store_id: "store-1".to_string(),
        stream_epoch,
        sequence,
        operation_kind: FACT_CREATE_OPERATION.to_string(),
        payload_schema: FACT_CREATE_PAYLOAD_SCHEMA.to_string(),
        payload,
        payload_digest: payload_hash,
        predecessor_digest,
        envelope_digest: envelope_hash,
    }
}

fn envelope(
    sequence: i64,
    predecessor_digest: [u8; 32],
    fact_id: &str,
    content: &str,
) -> FactCreateReplicaEnvelopeV1 {
    envelope_at_epoch(7, sequence, predecessor_digest, fact_id, content)
}

fn envelope_from_verified_outbox(entry: &JournalEntry) -> FactCreateReplicaEnvelopeV1 {
    FactCreateReplicaEnvelopeV1 {
        home_device_id: entry.home_device_id.clone(),
        store_id: entry.store_id.clone(),
        stream_epoch: entry.stream_epoch,
        sequence: entry.sequence,
        operation_kind: entry.operation_kind.clone(),
        payload_schema: entry.payload_schema.clone(),
        payload: entry.payload.clone(),
        payload_digest: entry.payload_digest,
        predecessor_digest: entry.predecessor_digest,
        envelope_digest: entry.envelope_digest,
    }
}

fn open_store(temp: &TempDir) -> MemoryStore {
    MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: temp.path().to_path_buf(),
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )
    .unwrap()
}

#[test]
fn fresh_verified_outbox_is_explicitly_empty() {
    let temp = TempDir::new().unwrap();
    let _store = open_store(&temp);
    let conn = Connection::open(temp.path().join("memory.db")).unwrap();
    let batch = export_verified_contiguous(&conn, "device-1", "store-1", 7, 1, 10).unwrap();
    assert_eq!(batch.status, ExportStatus::Empty);
    assert!(batch.entries.is_empty());
}

#[tokio::test]
async fn closed_fact_create_apply_is_atomic_idempotent_and_persistent() {
    let temp = TempDir::new().unwrap();
    let store = open_store(&temp);

    let first = envelope(
        1,
        GENESIS_PREDECESSOR,
        "11111111-1111-4111-8111-111111111111",
        "first replicated fact",
    );
    assert_eq!(
        store
            .apply_verified_fact_create(first.clone())
            .await
            .unwrap(),
        ReplicaApplyOutcome::Applied {
            sequence: 1,
            fact_id: "11111111-1111-4111-8111-111111111111".to_string(),
        }
    );
    assert_eq!(
        store
            .apply_verified_fact_create(first.clone())
            .await
            .unwrap(),
        ReplicaApplyOutcome::Duplicate { sequence: 1 }
    );

    let fork = envelope(
        1,
        GENESIS_PREDECESSOR,
        "22222222-2222-4222-8222-222222222222",
        "changed same sequence",
    );
    assert_eq!(
        store.apply_verified_fact_create(fork).await.unwrap(),
        ReplicaApplyOutcome::Fork { sequence: 1 }
    );

    let gap = envelope(
        3,
        first.envelope_digest,
        "33333333-3333-4333-8333-333333333333",
        "sequence gap",
    );
    assert_eq!(
        store.apply_verified_fact_create(gap).await.unwrap(),
        ReplicaApplyOutcome::Gap {
            expected: 2,
            received: 3,
        }
    );

    let wrong_predecessor = envelope(
        2,
        [9; 32],
        "44444444-4444-4444-8444-444444444444",
        "wrong predecessor",
    );
    assert_eq!(
        store
            .apply_verified_fact_create(wrong_predecessor)
            .await
            .unwrap(),
        ReplicaApplyOutcome::Fork { sequence: 2 }
    );

    let second = envelope(
        2,
        first.envelope_digest,
        "55555555-5555-4555-8555-555555555555",
        "second replicated fact",
    );
    assert_eq!(
        store
            .apply_verified_fact_create(second.clone())
            .await
            .unwrap(),
        ReplicaApplyOutcome::Applied {
            sequence: 2,
            fact_id: "55555555-5555-4555-8555-555555555555".to_string(),
        }
    );

    drop(store);
    let reopened = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: temp.path().to_path_buf(),
            ..Default::default()
        },
        Box::new(FailingEmbedder),
    )
    .unwrap();
    assert_eq!(
        reopened.apply_verified_fact_create(second).await.unwrap(),
        ReplicaApplyOutcome::Duplicate { sequence: 2 }
    );

    let conn = Connection::open(temp.path().join("memory.db")).unwrap();
    let facts: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM facts WHERE namespace = 'replica-contract'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let inbox: i64 = conn
        .query_row("SELECT COUNT(*) FROM replication_inbox", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(facts, 2);
    assert_eq!(inbox, 2);
}

#[tokio::test]
async fn inbox_failure_rolls_back_fact_and_stream_head() {
    let temp = TempDir::new().unwrap();
    let store = open_store(&temp);
    let conn = Connection::open(temp.path().join("memory.db")).unwrap();
    conn.execute_batch(
        "CREATE TRIGGER reject_replication_inbox
         BEFORE INSERT ON replication_inbox
         BEGIN
             SELECT RAISE(ABORT, 'injected replica inbox failure');
         END;",
    )
    .unwrap();

    let candidate = envelope(
        1,
        GENESIS_PREDECESSOR,
        "66666666-6666-4666-8666-666666666666",
        "must roll back",
    );
    assert!(store.apply_verified_fact_create(candidate).await.is_err());

    let facts: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM facts WHERE id = '66666666-6666-4666-8666-666666666666'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let streams: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM replication_inbox_streams",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(facts, 0);
    assert_eq!(streams, 0);
}

#[tokio::test]
async fn verified_v38_outbox_entry_applies_to_v39_receiver_and_replays() {
    let primary_temp = TempDir::new().unwrap();
    let primary = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: primary_temp.path().to_path_buf(),
            journal_device_id: Some("device-1".to_string()),
            journal_store_id: Some("store-1".to_string()),
            replication_mode: ReplicationMode::FactCreateRequired,
            replication_stream_epoch: 7,
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )
    .unwrap();
    let fact_id = primary
        .add_fact(
            "replica-contract",
            "outbox entry applied by receiver",
            Some("primary-store"),
            Some(serde_json::json!({"contract": "v38-to-v39"})),
        )
        .await
        .unwrap();
    let entry = {
        let conn = Connection::open(primary_temp.path().join("memory.db")).unwrap();
        let batch = export_verified_contiguous(&conn, "device-1", "store-1", 7, 1, 10).unwrap();
        assert_eq!(batch.status, ExportStatus::End);
        assert_eq!(batch.entries.len(), 1);
        batch.entries.into_iter().next().unwrap()
    };

    let receiver_temp = TempDir::new().unwrap();
    let receiver = open_store(&receiver_temp);
    let envelope = envelope_from_verified_outbox(&entry);
    assert_eq!(
        receiver
            .apply_verified_fact_create(envelope.clone())
            .await
            .unwrap(),
        ReplicaApplyOutcome::Applied {
            sequence: 1,
            fact_id: fact_id.clone(),
        }
    );
    assert_eq!(
        receiver.apply_verified_fact_create(envelope).await.unwrap(),
        ReplicaApplyOutcome::Duplicate { sequence: 1 }
    );

    let conn = Connection::open(receiver_temp.path().join("memory.db")).unwrap();
    let replicated: (String, String) = conn
        .query_row(
            "SELECT content, source FROM facts WHERE id = ?1",
            [&fact_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(replicated.0, "outbox entry applied by receiver");
    assert_eq!(replicated.1, "primary-store");
}

#[tokio::test]
async fn replica_envelope_digest_tampering_is_rejected_without_a_write() {
    let temp = TempDir::new().unwrap();
    let store = open_store(&temp);
    let mut tampered = envelope(
        1,
        GENESIS_PREDECESSOR,
        "77777777-7777-4777-8777-777777777777",
        "tampered digest must not apply",
    );
    tampered.payload_digest[0] ^= 0xFF;
    assert!(store.apply_verified_fact_create(tampered).await.is_err());

    let conn = Connection::open(temp.path().join("memory.db")).unwrap();
    let facts: i64 = conn
        .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
        .unwrap();
    let inbox: i64 = conn
        .query_row("SELECT COUNT(*) FROM replication_inbox", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(facts, 0);
    assert_eq!(inbox, 0);
}

#[tokio::test]
async fn replica_epoch_rotation_is_explicitly_rejected() {
    let temp = TempDir::new().unwrap();
    let store = open_store(&temp);
    let first = envelope(
        1,
        GENESIS_PREDECESSOR,
        "88888888-8888-4888-8888-888888888888",
        "first receiver epoch",
    );
    assert!(matches!(
        store.apply_verified_fact_create(first).await.unwrap(),
        ReplicaApplyOutcome::Applied { .. }
    ));

    let rotated = envelope_at_epoch(
        8,
        1,
        GENESIS_PREDECESSOR,
        "99999999-9999-4999-8999-999999999999",
        "new epoch must not replace receiver stream",
    );
    assert_eq!(
        store.apply_verified_fact_create(rotated).await.unwrap(),
        ReplicaApplyOutcome::EpochConflict {
            active: 7,
            received: 8,
        }
    );
}

#[test]
fn v39_migration_is_idempotent_for_a_v38_database() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(MIGRATION_V37).unwrap();
    conn.execute_batch(MIGRATION_V38).unwrap();
    conn.execute_batch(MIGRATION_V39).unwrap();
    conn.execute_batch(MIGRATION_V39).unwrap();

    let tables: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table'
               AND name IN ('replication_inbox', 'replication_inbox_streams')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tables, 2);
    conn.execute(
        "INSERT INTO replication_inbox_streams
         (home_device_id, store_id, stream_epoch, next_sequence, head_digest)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![
            "device-1",
            "store-1",
            7_i64,
            1_i64,
            GENESIS_PREDECESSOR.as_slice()
        ],
    )
    .unwrap();
}
