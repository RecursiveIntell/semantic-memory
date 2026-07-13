use semantic_memory::{
    AuthorityFaultStage, AuthorityOperationKind, AuthorityPermit, MemoryConfig, MemoryError,
    MemoryStore, MockEmbedder, RetrievalEpoch, StateView,
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
