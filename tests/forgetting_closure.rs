use semantic_memory::{
    AuthorityFaultStage, AuthorityPermit, ForgettingClosureRequestV1, ForgettingDispositionV1,
    GovernedAccessPurposeV1, GovernedAccessRequestV1, MemoryConfig, MemoryStore, MockEmbedder,
    ProjectionQuery, ReceiptMode, SearchContext, StateDependencyEdgeV1, StateView,
};
use stack_ids::ScopeKey;
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
    AuthorityPermit::operator_system("principal:test", "forgetting-test", capability)
}

fn access() -> GovernedAccessRequestV1 {
    GovernedAccessRequestV1::new(
        "principal:test",
        "principal:test",
        GovernedAccessPurposeV1::Recall,
        "private",
    )
}

async fn append(store: &MemoryStore, key: &str, namespace: &str, content: &str) -> String {
    store
        .authority()
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            key.into(),
            namespace.into(),
            content.into(),
            Some("forgetting-fixture".into()),
        )
        .await
        .unwrap()
        .affected_ids[0]
        .clone()
}

#[tokio::test]
async fn forget_closes_canonical_and_derived_access_paths() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    let ancestor = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "forget-append-ancestor".into(),
            "private".into(),
            "forbidden canary alpha-7391".into(),
            Some("subject-request".into()),
        )
        .await
        .unwrap()
        .affected_ids[0]
        .clone();
    let derived = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "forget-append-derived".into(),
            "private".into(),
            "summary laundering alpha-7391".into(),
            Some("derived-summary".into()),
        )
        .await
        .unwrap()
        .affected_ids[0]
        .clone();
    store
        .add_state_dependency_edge(
            StateDependencyEdgeV1::derived_from_state(
                format!("fact:{derived}"),
                format!("fact:{ancestor}"),
            ),
            1.0,
        )
        .await
        .unwrap();

    assert!(store.reembed_all().await.unwrap() >= 2);
    assert!(store.get_fact_embedding(&ancestor).await.unwrap().is_some());

    let mut context = SearchContext::default_now();
    context.receipt_mode = ReceiptMode::ReturnReceipt;
    context.request_id = Some("forget-replay-source".into());
    let replay_source = store
        .search_with_context("alpha-7391", Some(10), Some(&["private"]), None, context)
        .await
        .unwrap()
        .receipt
        .unwrap()
        .receipt_id;

    // Populate the ordinary in-process search cache before forgetting.
    assert!(!store
        .search("alpha-7391", Some(10), None, None)
        .await
        .unwrap()
        .is_empty());

    let receipt = authority
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "forget-closure-1".into(),
            ForgettingClosureRequestV1::new(
                vec![ancestor.clone()],
                "private",
                "subject erasure request",
                128,
            ),
        )
        .await
        .unwrap();

    assert_eq!(receipt.schema_version, "forgetting_closure_receipt_v1");
    assert_eq!(receipt.disposition, ForgettingDispositionV1::Applied);
    assert!(receipt.affected_canonical_ids.contains(&ancestor));
    assert!(receipt.affected_canonical_ids.contains(&derived));
    assert!(receipt.deferred_surfaces.is_empty());
    assert!(receipt.not_tested_surfaces.is_empty());
    assert!(receipt.verification.iter().all(|check| check.passed));
    assert_eq!(receipt.after_epoch.0, receipt.before_epoch.0 + 1);
    assert_eq!(
        authority
            .get_forgetting_receipt_by_idempotency_key("forget-closure-1")
            .await
            .unwrap(),
        Some(receipt.clone())
    );
    let receipt_json = serde_json::to_string(&receipt).unwrap();
    assert!(!receipt_json.contains("alpha-7391"));
    assert!(!receipt_json.contains("subject erasure request"));

    for id in [&ancestor, &derived] {
        let raw = store.get_fact_raw_compat(id).await.unwrap().unwrap();
        assert_eq!(raw.content, "[FORGOTTEN]");
        assert!(store.get_fact_embedding(id).await.unwrap().is_none());
        assert!(authority
            .get_fact_governed(id, access())
            .await
            .unwrap()
            .fact
            .is_none());
        assert!(authority
            .export_fact_governed(id, access())
            .await
            .unwrap()
            .fact
            .is_none());
        assert!(store
            .list_graph_edges_for_node(&format!("fact:{id}"))
            .await
            .unwrap()
            .is_empty());
    }

    for view in [
        StateView::Current,
        StateView::IncludeSuperseded,
        StateView::HistoricalAt("2999-01-01T00:00:00Z".into()),
    ] {
        let results = store
            .search_with_view("alpha-7391", Some(10), Some(&["private"]), None, view)
            .await
            .unwrap();
        assert!(results
            .iter()
            .all(|result| !result.content.contains("alpha-7391")));
    }
    assert!(store
        .search("alpha-7391", Some(10), None, None)
        .await
        .unwrap()
        .iter()
        .all(|result| !result.content.contains("alpha-7391")));
    assert!(matches!(
        store
            .replay_search_receipt(
                &replay_source,
                "alpha-7391",
                Some(10),
                Some(&["private"]),
                None,
            )
            .await,
        Err(semantic_memory::MemoryError::ForgettingClosureIncomplete { .. })
    ));
}

#[tokio::test]
async fn cycles_and_shared_derivations_close_once_without_collateral_deletion() {
    let (store, _tmp) = test_store();
    let root = append(&store, "cycle-root", "private", "cycle root canary").await;
    let shared = append(&store, "cycle-shared", "private", "shared derived canary").await;
    let cycle = append(&store, "cycle-node", "private", "cycle derived canary").await;
    let unrelated = append(&store, "cycle-unrelated", "private", "unrelated survives").await;
    for edge in [
        StateDependencyEdgeV1::derived_from_state(format!("fact:{shared}"), format!("fact:{root}")),
        StateDependencyEdgeV1::derived_from_state(format!("fact:{cycle}"), format!("fact:{root}")),
        StateDependencyEdgeV1::derived_from_state(
            format!("fact:{shared}"),
            format!("fact:{cycle}"),
        ),
        StateDependencyEdgeV1::derived_from_state(
            format!("fact:{cycle}"),
            format!("fact:{shared}"),
        ),
    ] {
        store.add_state_dependency_edge(edge, 1.0).await.unwrap();
    }

    let receipt = store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "cycle-forget".into(),
            ForgettingClosureRequestV1::new(vec![root.clone()], "private", "cycle request", 32),
        )
        .await
        .unwrap();
    let affected = receipt
        .affected_canonical_ids
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        affected,
        [cycle, root, shared]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
    );
    assert_eq!(
        store
            .get_fact_raw_compat(&unrelated)
            .await
            .unwrap()
            .unwrap()
            .content,
        "unrelated survives"
    );
}

#[tokio::test]
async fn budget_and_scope_boundaries_fail_closed_before_mutation() {
    let (store, _tmp) = test_store();
    let root = append(&store, "bounded-root", "private", "bounded root").await;
    let derived = append(&store, "bounded-derived", "private", "bounded derived").await;
    store
        .add_state_dependency_edge(
            StateDependencyEdgeV1::derived_from_state(
                format!("fact:{derived}"),
                format!("fact:{root}"),
            ),
            1.0,
        )
        .await
        .unwrap();

    let exhausted = store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "bounded-budget".into(),
            ForgettingClosureRequestV1::new(vec![root.clone()], "private", "bounded request", 1),
        )
        .await;
    assert!(matches!(
        exhausted,
        Err(semantic_memory::MemoryError::ForgettingBudgetExceeded { .. })
    ));
    assert_eq!(
        store
            .get_fact_raw_compat(&root)
            .await
            .unwrap()
            .unwrap()
            .content,
        "bounded root"
    );

    let wrong_namespace = store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "bounded-scope".into(),
            ForgettingClosureRequestV1::new(vec![root.clone()], "other", "wrong scope", 8),
        )
        .await;
    assert!(matches!(
        wrong_namespace,
        Err(semantic_memory::MemoryError::ForgettingClosureIncomplete { .. })
    ));
    assert_eq!(
        store
            .get_fact_raw_compat(&derived)
            .await
            .unwrap()
            .unwrap()
            .content,
        "bounded derived"
    );

    let other_principal = store
        .authority()
        .append(
            AuthorityPermit::operator_system(
                "principal:other",
                "forgetting-test-other",
                AuthorityPermit::APPEND_CAPABILITY,
            ),
            "bounded-other-principal".into(),
            "private".into(),
            "other principal survives".into(),
            None,
        )
        .await
        .unwrap()
        .affected_ids[0]
        .clone();
    store
        .add_state_dependency_edge(
            StateDependencyEdgeV1::derived_from_state(
                format!("fact:{other_principal}"),
                format!("fact:{root}"),
            ),
            1.0,
        )
        .await
        .unwrap();
    let cross_principal = store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "bounded-principal".into(),
            ForgettingClosureRequestV1::new(vec![root.clone()], "private", "principal boundary", 8),
        )
        .await;
    assert!(matches!(
        cross_principal,
        Err(semantic_memory::MemoryError::ForgettingClosureIncomplete { .. })
    ));
    assert_eq!(
        store
            .get_fact_raw_compat(&other_principal)
            .await
            .unwrap()
            .unwrap()
            .content,
        "other principal survives"
    );
}

#[tokio::test]
async fn idempotency_is_exact_and_conflicting_retries_fail_closed() {
    let (store, _tmp) = test_store();
    let root = append(&store, "idem-root", "private", "idempotent canary").await;
    let request =
        ForgettingClosureRequestV1::new(vec![root.clone()], "private", "idempotent request", 8);
    let first = store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "idem-forget".into(),
            request.clone(),
        )
        .await
        .unwrap();
    let replay = store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "idem-forget".into(),
            request,
        )
        .await
        .unwrap();
    assert_eq!(first, replay);
    let conflict = store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "idem-forget".into(),
            ForgettingClosureRequestV1::new(vec![root], "private", "different reason", 8),
        )
        .await;
    assert!(matches!(
        conflict,
        Err(semantic_memory::MemoryError::AuthorityIdempotencyConflict { .. })
    ));
}

#[tokio::test]
async fn injected_fault_rolls_back_scrubbing_invalidations_epochs_and_receipt() {
    let (store, _tmp) = test_store();
    let root = append(&store, "rollback-root", "private", "rollback canary").await;
    store
        .authority()
        .set_fault(Some(AuthorityFaultStage::AfterForgettingMutation));
    let result = store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "rollback-forget".into(),
            ForgettingClosureRequestV1::new(vec![root.clone()], "private", "rollback request", 8),
        )
        .await;
    assert!(matches!(
        result,
        Err(semantic_memory::MemoryError::AuthorityFaultInjected {
            stage: AuthorityFaultStage::AfterForgettingMutation
        })
    ));
    assert_eq!(
        store
            .get_fact_raw_compat(&root)
            .await
            .unwrap()
            .unwrap()
            .content,
        "rollback canary"
    );
    assert!(
        store.get_fact_embedding(&root).await.unwrap().is_some(),
        "failed forgetting must preserve the governed fact embedding"
    );
    assert!(!store
        .search("rollback canary", Some(4), Some(&["private"]), None)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn projection_derivations_are_hidden_after_ancestor_forgetting() {
    let (store, _tmp) = test_store();
    let root = append(&store, "projection-root", "private", "projection ancestor").await;
    store
        .raw_execute(
            "INSERT INTO claim_versions
             (claim_version_id, claim_id, projection_family, subject_entity_id, predicate,
              object_anchor, scope_namespace, source_envelope_id, source_authority, content)
             VALUES (?1, ?2, 'test', 'entity-1', 'contains', ?3, 'private', 'env-1',
                     'test', ?4)",
            vec![
                "claim-v1".into(),
                "claim-1".into(),
                "\"projection ancestor\"".into(),
                "derived projection forbidden canary".into(),
            ],
        )
        .await
        .unwrap();
    store
        .raw_execute(
            "INSERT INTO derivation_edges
             (source_kind, source_id, target_kind, target_id, derivation_type)
             VALUES ('fact', ?1, 'claim_version', 'claim-v1', 'derived_from_fact')",
            vec![root.clone()],
        )
        .await
        .unwrap();
    assert_eq!(
        store
            .query_claim_versions(ProjectionQuery::new(ScopeKey::namespace_only("private")))
            .await
            .unwrap()
            .len(),
        1
    );

    store
        .authority()
        .forget(
            permit(AuthorityPermit::FORGET_CAPABILITY),
            "projection-forget".into(),
            ForgettingClosureRequestV1::new(vec![root], "private", "projection request", 16),
        )
        .await
        .unwrap();
    assert!(store
        .query_claim_versions(ProjectionQuery::new(ScopeKey::namespace_only("private")))
        .await
        .unwrap()
        .is_empty());
}
