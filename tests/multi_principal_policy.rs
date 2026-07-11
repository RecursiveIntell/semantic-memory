use semantic_memory::{
    evaluate_governed_access_v1, AuthorityScopeV1, AuthorityScopesV1, CallerPrincipalV1,
    DelegationElevationLeaseV1, ElevationRequirementV1, GovernedAccessPurposeV1,
    GovernedAccessRequestV1, MemoryConfig, MemoryStore, MockEmbedder, NamespaceScopeV1,
    OriginAuthorityLabelV1, OriginClassV1, OriginRiskV1, PolicyDecisionV1, ProjectionQuery,
    RevocationStatusV1, StateResolutionMode, StateView, SubjectPrincipalV1,
};
use stack_ids::ScopeKey;
use tempfile::TempDir;

fn label() -> OriginAuthorityLabelV1 {
    OriginAuthorityLabelV1::new(
        OriginClassV1::UserStatement,
        "principal:writer",
        "test",
        "blake3:source",
        OriginRiskV1::Low,
        AuthorityScopesV1 {
            recall: AuthorityScopeV1::Audience,
            assertion: AuthorityScopeV1::Audience,
            action: AuthorityScopeV1::Denied,
        },
        ElevationRequirementV1::ExplicitOperatorApproval,
        None,
        RevocationStatusV1::Active,
        vec!["principal:alice".into(), "team:medical".into()],
    )
    .unwrap()
    .with_subject_principal(SubjectPrincipalV1::new("principal:patient").unwrap())
    .with_resource_scope(NamespaceScopeV1::exact("medical"))
}

fn request() -> GovernedAccessRequestV1 {
    GovernedAccessRequestV1::for_principals(
        CallerPrincipalV1::new("principal:alice").unwrap(),
        SubjectPrincipalV1::new("principal:patient").unwrap(),
        vec!["principal:alice".into(), "team:medical".into()],
        GovernedAccessPurposeV1::Recall,
        NamespaceScopeV1::exact("medical"),
    )
}

fn store() -> (MemoryStore, TempDir) {
    let temp = TempDir::new().unwrap();
    let store = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: temp.path().to_path_buf(),
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )
    .unwrap();
    (store, temp)
}

#[test]
fn caller_subject_confusion_and_confused_deputy_fail_closed_without_a_live_lease() {
    let decision = evaluate_governed_access_v1(
        "fact:patient",
        Some("medical"),
        Some(&label()),
        None,
        &request(),
    );
    assert!(!decision.allowed);
    assert_eq!(decision.outcome, PolicyDecisionV1::Deny);
    assert!(decision
        .reasons
        .iter()
        .any(|reason| reason == "caller_subject_delegation_required"));
    assert!(!decision.policy_digest.is_empty());

    let mut delegated = request();
    delegated = delegated.with_delegation_or_elevation(DelegationElevationLeaseV1::delegation(
        "lease:care-team",
        "principal:patient",
        "principal:alice",
        vec![GovernedAccessPurposeV1::Recall],
        NamespaceScopeV1::exact("medical"),
        vec!["team:medical".into()],
        "2999-01-01T00:00:00Z",
    ));
    let decision = evaluate_governed_access_v1(
        "fact:patient",
        Some("medical"),
        Some(&label()),
        None,
        &delegated,
    );
    assert!(decision.allowed, "{:?}", decision.reasons);
}

#[test]
fn audience_namespace_expiry_revocation_and_admin_elevation_are_receipted() {
    let mut audience_mismatch = request();
    audience_mismatch = audience_mismatch.with_audiences(vec!["team:other".into()]);
    let denied = evaluate_governed_access_v1(
        "fact:patient",
        Some("medical"),
        Some(&label()),
        None,
        &audience_mismatch,
    );
    assert!(!denied.allowed);
    assert!(denied
        .reasons
        .iter()
        .any(|reason| reason == "audience_intersection_empty"));

    let mut widened = request();
    widened.scope = NamespaceScopeV1::exact("other");
    let denied = evaluate_governed_access_v1(
        "fact:patient",
        Some("medical"),
        Some(&label()),
        None,
        &widened,
    );
    assert!(!denied.allowed);
    assert!(denied
        .reasons
        .iter()
        .any(|reason| reason == "namespace_scope_mismatch"));

    let mut expired = request();
    expired = expired.with_delegation_or_elevation(DelegationElevationLeaseV1::delegation(
        "lease:expired",
        "principal:patient",
        "principal:alice",
        vec![GovernedAccessPurposeV1::Recall],
        NamespaceScopeV1::exact("medical"),
        vec!["team:medical".into()],
        "2000-01-01T00:00:00Z",
    ));
    let denied = evaluate_governed_access_v1(
        "fact:patient",
        Some("medical"),
        Some(&label()),
        None,
        &expired,
    );
    assert!(!denied.allowed);
    assert!(denied
        .reasons
        .iter()
        .any(|reason| reason == "delegation_expired"));

    let revoked = request()
        .with_delegation_or_elevation(DelegationElevationLeaseV1::delegation(
            "lease:revoked",
            "principal:patient",
            "principal:alice",
            vec![GovernedAccessPurposeV1::Recall],
            NamespaceScopeV1::exact("medical"),
            vec!["team:medical".into()],
            "2999-01-01T00:00:00Z",
        ))
        .with_lease_revoked();
    let denied = evaluate_governed_access_v1(
        "fact:patient",
        Some("medical"),
        Some(&label()),
        None,
        &revoked,
    );
    assert!(!denied.allowed);
    assert!(denied
        .reasons
        .iter()
        .any(|reason| reason == "delegation_revoked"));

    let mut admin = request();
    admin = admin
        .with_purpose(GovernedAccessPurposeV1::Admin)
        .with_delegation_or_elevation(DelegationElevationLeaseV1::elevation(
            "lease:admin",
            "principal:patient",
            "principal:alice",
            NamespaceScopeV1::exact("medical"),
            "2999-01-01T00:00:00Z",
        ));
    let allowed = evaluate_governed_access_v1(
        "fact:patient",
        Some("medical"),
        Some(&label()),
        None,
        &admin,
    );
    assert!(allowed.allowed, "{:?}", allowed.reasons);
}

#[test]
fn cross_principal_derived_state_cannot_pick_a_subject_to_gain_access() {
    let left = label()
        .with_subject_principal(SubjectPrincipalV1::new("principal:alice").unwrap())
        .with_resource_scope(NamespaceScopeV1::exact("medical"));
    let right = OriginAuthorityLabelV1::new(
        OriginClassV1::UserStatement,
        "principal:other",
        "test",
        "blake3:other",
        OriginRiskV1::Low,
        left.scopes.clone(),
        ElevationRequirementV1::ExplicitOperatorApproval,
        None,
        RevocationStatusV1::Active,
        vec!["principal:alice".into()],
    )
    .unwrap()
    .with_subject_principal(SubjectPrincipalV1::new("principal:other-subject").unwrap())
    .with_resource_scope(NamespaceScopeV1::exact("medical"));
    let derived = OriginAuthorityLabelV1::derive(
        &[left, right],
        semantic_memory::OriginDerivationKindV1::Summary,
        "blake3:derived-cross",
    )
    .unwrap();
    let request = GovernedAccessRequestV1::new(
        "principal:alice",
        "principal:alice",
        GovernedAccessPurposeV1::Recall,
        "medical",
    );
    let denied = evaluate_governed_access_v1(
        "fact:derived",
        Some("medical"),
        Some(&derived),
        None,
        &request,
    );
    assert!(!denied.allowed);
    assert!(denied
        .reasons
        .iter()
        .any(|reason| reason == "cross_principal_derived_subject_ambiguous"));
}

#[tokio::test]
async fn direct_cache_graph_export_replay_state_historical_and_projection_paths_do_not_bypass_policy(
) {
    let (store, _temp) = store();
    let write_label = label()
        .with_subject_principal(SubjectPrincipalV1::new("principal:alice").unwrap())
        .with_resource_scope(NamespaceScopeV1::exact("medical"));
    let receipt = store
        .authority()
        .append(
            semantic_memory::AuthorityPermit::with_evidence(
                "principal:writer",
                "test",
                semantic_memory::AuthorityPermit::APPEND_CAPABILITY,
                vec!["evidence:test".into()],
            )
            .with_origin(write_label),
            "multi-principal-paths".into(),
            "medical".into(),
            "confidential marker".into(),
            None,
        )
        .await
        .unwrap();
    let fact_id = receipt.affected_ids[0].clone();
    let denied = GovernedAccessRequestV1::new(
        "principal:bob",
        "principal:bob",
        GovernedAccessPurposeV1::Recall,
        "medical",
    );

    assert!(store
        .authority()
        .get_fact_governed(&fact_id, denied.clone())
        .await
        .unwrap()
        .fact
        .is_none());
    assert!(store
        .authority()
        .search_governed("confidential marker", Some(8), denied.clone())
        .await
        .unwrap()
        .results
        .is_empty());
    assert!(
        store
            .authority()
            .search_governed("confidential marker", Some(8), denied.clone())
            .await
            .unwrap()
            .results
            .is_empty(),
        "cache cannot leak"
    );
    assert!(store
        .authority()
        .list_facts_governed(denied.clone(), 8, 0, StateView::Current)
        .await
        .unwrap()
        .facts
        .is_empty());
    assert!(store
        .authority()
        .export_fact_governed(&fact_id, denied.clone())
        .await
        .unwrap()
        .fact
        .is_none());

    let node_id = format!("fact:{fact_id}");
    store
        .add_graph_edge(
            &node_id,
            "fact:missing",
            semantic_memory::GraphEdgeType::Entity {
                relation: "related".into(),
            },
            1.0,
            None,
        )
        .await
        .unwrap();
    assert!(store
        .authority()
        .list_graph_edges_for_node_governed(&format!("fact:{fact_id}"), denied.clone())
        .await
        .unwrap()
        .edges
        .is_empty());
    assert!(store
        .authority()
        .search_governed_with_view(
            "confidential marker",
            Some(8),
            denied.clone(),
            StateView::HistoricalAt("2999-01-01T00:00:00Z".into())
        )
        .await
        .unwrap()
        .results
        .is_empty());
    assert!(store
        .authority()
        .resolve_memory_governed(
            "confidential marker",
            Some(8),
            denied.clone(),
            StateResolutionMode::Current,
            8
        )
        .await
        .unwrap()
        .response
        .answer
        .is_none());
    let projection = store
        .query_claim_versions_governed(
            ProjectionQuery::new(ScopeKey::namespace_only("other")),
            denied,
        )
        .await
        .unwrap();
    assert!(projection.items.is_empty());
    assert!(projection
        .decisions
        .iter()
        .all(|decision| !decision.allowed));
}
