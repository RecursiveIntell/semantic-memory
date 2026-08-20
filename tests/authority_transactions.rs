use semantic_memory::{
    AuthorityFaultStage, AuthorityOperationKind, AuthorityPermit, MemoryConfig, MemoryError,
    MemoryStore, MockEmbedder, ReplicationMode, RetrievalEpoch, StateView,
};
use tempfile::TempDir;

fn test_store() -> (MemoryStore, TempDir) {
    let tmp = TempDir::new().unwrap();
    let store = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: tmp.path().to_path_buf(),
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )
    .unwrap();
    (store, tmp)
}

fn permit(capability: &str) -> AuthorityPermit {
    AuthorityPermit::operator_system("principal:test", "caller:test", capability)
}

#[tokio::test]
async fn append_rejects_model_permit_without_evidence_and_persists_nothing() {
    let (store, _tmp) = test_store();
    let result = store
        .authority()
        .append(
            AuthorityPermit::new(
                "model:unsupported",
                "hostile-benchmark",
                AuthorityPermit::APPEND_CAPABILITY,
            ),
            "unsupported-model".into(),
            "general".into(),
            "unsupported model claim".into(),
            Some("model-proposal".into()),
        )
        .await;
    assert!(matches!(
        result,
        Err(MemoryError::AuthorityAdmissionRejected { .. })
    ));
    assert!(store.list_facts("general", 10, 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn append_rejects_unresolved_evidence_strings() {
    let (store, _tmp) = test_store();
    let permit = AuthorityPermit::with_evidence(
        "principal:test",
        "caller:test",
        AuthorityPermit::APPEND_CAPABILITY,
        vec!["evidence:caller-controlled".into()],
    )
    .with_origin(semantic_memory::OriginAuthorityLabelV1::operator_system(
        "principal:test",
        "caller:test",
    ));
    let result = store
        .authority()
        .append(
            permit,
            "unresolved-evidence".into(),
            "general".into(),
            "must not persist".into(),
            None,
        )
        .await;
    assert!(matches!(
        result,
        Err(MemoryError::AuthorityAdmissionRejected { .. })
    ));
}

#[tokio::test]
async fn append_persists_fact_and_receipt() {
    let (store, _tmp) = test_store();
    let authority = store.authority();

    let receipt = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "append-1".into(),
            "general".into(),
            "The sky is blue".into(),
            Some("test".into()),
        )
        .await
        .unwrap();

    assert_eq!(receipt.operation_kind, AuthorityOperationKind::Append);
    assert_eq!(receipt.before_epoch, RetrievalEpoch(0));
    assert_eq!(receipt.after_epoch, RetrievalEpoch(1));
    assert_eq!(receipt.affected_ids.len(), 2);
    assert_eq!(
        authority
            .get_receipt_by_idempotency_key("append-1")
            .await
            .unwrap(),
        Some(receipt.clone())
    );
    assert_eq!(
        store.list_facts("general", 10, 0).await.unwrap()[0].content,
        "The sky is blue"
    );
}

#[tokio::test]
async fn supersede_and_redact_are_append_only_and_keep_one_head() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    let first = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "append-1".into(),
            "general".into(),
            "old assertion".into(),
            None,
        )
        .await
        .unwrap();
    let old_id = first.affected_ids[0].clone();

    let second = authority
        .supersede(
            permit(AuthorityPermit::SUPERSEDE_CAPABILITY),
            "supersede-1".into(),
            old_id.clone(),
            "new assertion".into(),
            None,
        )
        .await
        .unwrap();
    let new_id = second.affected_ids[0].clone();
    assert_ne!(old_id, new_id);
    assert_eq!(store.list_facts("general", 10, 0).await.unwrap().len(), 1);
    assert_eq!(
        store
            .list_facts_with_view("general", 10, 0, StateView::IncludeSuperseded)
            .await
            .unwrap()
            .len(),
        2
    );

    let redacted = authority
        .redact(
            permit(AuthorityPermit::REDACT_CAPABILITY),
            "redact-1".into(),
            new_id,
            "privacy request".into(),
        )
        .await
        .unwrap();
    assert_eq!(redacted.operation_kind, AuthorityOperationKind::Redact);
    assert_eq!(store.list_facts("general", 10, 0).await.unwrap().len(), 1);
    assert_eq!(
        store.list_facts("general", 10, 0).await.unwrap()[0].content,
        "[REDACTED]"
    );
}

#[tokio::test]
async fn duplicate_retry_is_same_receipt_and_conflicting_payload_fails() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    let first = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "same-key".into(),
            "general".into(),
            "same content".into(),
            None,
        )
        .await
        .unwrap();
    let retry = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "same-key".into(),
            "general".into(),
            "same content".into(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&retry).unwrap()
    );
    assert_eq!(first.before_epoch, retry.before_epoch);
    assert_eq!(first.after_epoch, retry.after_epoch);
    assert_eq!(
        authority
            .get_receipt_by_operation_id(&first.operation_id)
            .await
            .unwrap(),
        Some(first.clone())
    );
    assert_eq!(store.list_facts("general", 10, 0).await.unwrap().len(), 1);

    let conflict = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "same-key".into(),
            "general".into(),
            "different content".into(),
            None,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        conflict,
        MemoryError::AuthorityIdempotencyConflict { .. }
    ));
}

#[tokio::test]
async fn unauthorized_permit_is_rejected() {
    let (store, _tmp) = test_store();
    let error = store
        .authority()
        .append(
            permit("not-authorized"),
            "unauthorized".into(),
            "general".into(),
            "must not persist".into(),
            None,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, MemoryError::AuthorityUnauthorized { .. }));
    assert_eq!(store.stats().await.unwrap().total_facts, 0);
}

#[tokio::test]
async fn every_fault_gate_rolls_back_the_whole_mutation() {
    let stages = [
        AuthorityFaultStage::BeforeAppend,
        AuthorityFaultStage::AfterAppend,
        AuthorityFaultStage::BeforeLineage,
        AuthorityFaultStage::AfterLineage,
        AuthorityFaultStage::BeforeJournal,
        AuthorityFaultStage::AfterJournal,
        AuthorityFaultStage::BeforeEpoch,
        AuthorityFaultStage::AfterEpoch,
        AuthorityFaultStage::BeforeReceipt,
        AuthorityFaultStage::AfterReceipt,
    ];

    for (index, stage) in stages.into_iter().enumerate() {
        let (store, _tmp) = test_store();
        let authority = store.authority();
        authority.set_fault(Some(stage));
        let error = authority
            .append(
                permit(AuthorityPermit::APPEND_CAPABILITY),
                format!("fault-{index}"),
                "general".into(),
                "must roll back".into(),
                None,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::AuthorityFaultInjected { stage: s } if s == stage));
        assert_eq!(
            store.stats().await.unwrap().total_facts,
            0,
            "stage {stage:?}"
        );
        assert!(store.list_all_graph_edges().await.unwrap().is_empty());
        assert!(authority
            .get_receipt_by_idempotency_key(&format!("fault-{index}"))
            .await
            .unwrap()
            .is_none());
    }
}

// ── Governed authority → verified mutation_journal (replication outbox) ──────
//
// Canary contract: a governed append_with_metadata must write the exact
// canonical payload into mutation_journal via append_verified_in_tx in the
// same SQLite transaction as the fact row, so mnemes-sync-client can export
// it. Idempotent retry must not journal twice; the stream allocator must
// continue across operations; supersede/redact have no admitted replication
// contract and must stay out of the outbox; a store without replication
// identity must remain local-only.

fn replication_store() -> (MemoryStore, TempDir) {
    let tmp = TempDir::new().unwrap();
    let store = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: tmp.path().to_path_buf(),
            journal_device_id: Some("canary-device".into()),
            journal_store_id: Some("canary-store".into()),
            replication_mode: ReplicationMode::FactCreateRequired,
            replication_stream_epoch: 1,
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )
    .unwrap();
    (store, tmp)
}

fn journal_conn(tmp: &TempDir) -> rusqlite::Connection {
    rusqlite::Connection::open(tmp.path().join("memory.db")).unwrap()
}

#[tokio::test]
async fn governed_append_writes_verified_fact_create_outbox_row() {
    let (store, tmp) = replication_store();
    let authority = store.authority();

    let metadata = serde_json::json!({"channel": "canary", "importance": 3});
    let receipt = authority
        .append_with_metadata(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "governed-canary-1".into(),
            "general".into(),
            "Canary fact for governed journaling".into(),
            Some("canary-source".into()),
            Some(metadata.clone()),
        )
        .await
        .unwrap();
    let fact_id = receipt.affected_ids[0].clone();

    let conn = journal_conn(&tmp);
    let (count, seq, op, schema, state): (i64, i64, String, String, String) = conn
        .query_row(
            "SELECT COUNT(*), MIN(sequence), MIN(operation_kind), MIN(payload_schema),
                    MIN(record_state)
             FROM mutation_journal",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .unwrap();
    assert_eq!((count, seq), (1, 1), "one verified row at sequence 1");
    assert_eq!(op, "fact.create");
    assert_eq!(schema, "semantic_memory.fact.create.v1");
    assert_eq!(state, "verified_v1");

    // The outbox payload is the exact canonical payload a replica replays.
    let payload: Vec<u8> = conn
        .query_row(
            "SELECT payload FROM mutation_journal WHERE sequence = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let decoded: serde_json::Value = serde_json::from_slice(&payload).unwrap();
    assert_eq!(decoded["fact_id"], serde_json::json!(fact_id));
    assert_eq!(decoded["namespace"], serde_json::json!("general"));
    assert_eq!(
        decoded["content"],
        serde_json::json!("Canary fact for governed journaling")
    );
    assert_eq!(decoded["source"], serde_json::json!("canary-source"));
    assert_eq!(decoded["metadata"], metadata);

    // The stream allocator advanced and the first record chains to genesis.
    let (next_sequence, head_len): (i64, i64) = conn
        .query_row(
            "SELECT next_sequence, length(head_digest) FROM replication_streams
             WHERE home_device_id = 'canary-device' AND store_id = 'canary-store'
               AND stream_epoch = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(next_sequence, 2);
    assert_eq!(head_len, 32);
    let predecessor: Vec<u8> = conn
        .query_row(
            "SELECT predecessor_digest FROM mutation_journal WHERE sequence = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(predecessor, vec![0u8; 32], "first record chains to genesis");
}

#[tokio::test]
async fn governed_append_idempotent_retry_does_not_duplicate_and_stream_continues() {
    let (store, tmp) = replication_store();
    let authority = store.authority();

    let first = authority
        .append_with_metadata(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "canary-dup-key".into(),
            "general".into(),
            "first governed fact".into(),
            None,
            None,
        )
        .await
        .unwrap();

    // Idempotent retry: same receipt, no second journal row.
    let retry = authority
        .append_with_metadata(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "canary-dup-key".into(),
            "general".into(),
            "first governed fact".into(),
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(first.after_epoch, retry.after_epoch);
    let conn = journal_conn(&tmp);
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM mutation_journal", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1, "idempotent retry must not journal twice");

    // A second distinct append continues the stream at sequence 2 and chains
    // to the first envelope digest — the allocator owns continuity, so a
    // restart can never re-derive a sequence from journal contents.
    let second = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "canary-second".into(),
            "general".into(),
            "second governed fact".into(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(second.after_epoch, RetrievalEpoch(2));
    let (seq2_pred, seq1_envelope): (Vec<u8>, Vec<u8>) = conn
        .query_row(
            "SELECT (SELECT predecessor_digest FROM mutation_journal WHERE sequence = 2),
                    (SELECT envelope_digest FROM mutation_journal WHERE sequence = 1)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        seq2_pred, seq1_envelope,
        "sequence 2 must chain to sequence 1"
    );
    let next_sequence: i64 = conn
        .query_row(
            "SELECT next_sequence FROM replication_streams
             WHERE home_device_id = 'canary-device' AND store_id = 'canary-store'
               AND stream_epoch = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(next_sequence, 3);

    // Supersede/redact have no admitted replication operation yet: the outbox
    // still holds exactly the two fact-create rows.
    authority
        .supersede(
            permit(AuthorityPermit::SUPERSEDE_CAPABILITY),
            "canary-supersede".into(),
            first.affected_ids[0].clone(),
            "superseded content".into(),
            None,
        )
        .await
        .unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM mutation_journal", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 2, "supersede must not emit a replication outbox row");
}

#[tokio::test]
async fn governed_append_without_replication_identity_emits_no_outbox() {
    let (store, tmp) = test_store(); // default MemoryConfig: replication disabled
    store
        .authority()
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "local-only".into(),
            "general".into(),
            "stays local".into(),
            None,
        )
        .await
        .unwrap();
    let conn = journal_conn(&tmp);
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM mutation_journal", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 0, "local-only store must emit no outbox rows");
}
