use semantic_memory::{
    rerank_state_aware, AuthorityPermit, AuthorityScopeV1, AuthorityScopesV1,
    ElevationRequirementV1, EvidenceGapRequestV1, EvidencePacketV1, EvidenceTerminalOutcomeV1,
    GovernedAccessPurposeV1, GovernedAccessRequestV1, MemoryConfig, MemoryStore, MockEmbedder,
    OriginAuthorityLabelV1, OriginClassV1, OriginRiskV1, PremiseStatus, SearchResult, SearchSource,
    StateRerankCandidateV1, StateRerankWeightsV1, StateView,
};
use tempfile::TempDir;

fn store() -> (MemoryStore, TempDir) {
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

fn origin(principal: &str, scope: AuthorityScopeV1) -> OriginAuthorityLabelV1 {
    OriginAuthorityLabelV1::new(
        OriginClassV1::ExternalEvidence,
        principal,
        "evidence-gap-test",
        format!("blake3:{principal}:evidence"),
        OriginRiskV1::Low,
        AuthorityScopesV1 {
            recall: scope,
            assertion: scope,
            action: scope,
        },
        ElevationRequirementV1::ExplicitOperatorApproval,
        None,
        semantic_memory::RevocationStatusV1::Active,
        vec![principal.to_string()],
    )
    .unwrap()
}

fn permit(principal: &str) -> AuthorityPermit {
    AuthorityPermit::with_evidence(
        principal,
        "evidence-gap-test",
        AuthorityPermit::APPEND_CAPABILITY,
        vec!["fixture:evidence-gap".into()],
    )
    .with_origin(origin(principal, AuthorityScopeV1::PrincipalOnly))
}

fn access() -> GovernedAccessRequestV1 {
    GovernedAccessRequestV1::new(
        "principal:alice",
        "principal:alice",
        GovernedAccessPurposeV1::Recall,
        "general",
    )
}

#[test]
fn state_critical_candidate_moves_above_lexical_stale_candidate_without_losing_scores() {
    let stale = SearchResult {
        content: "service color was blue".into(),
        source: SearchSource::Fact {
            fact_id: "stale".into(),
            namespace: "general".into(),
        },
        score: 0.99,
        bm25_rank: Some(1),
        vector_rank: None,
        cosine_similarity: None,
    };
    let current = SearchResult {
        content: "service color is green".into(),
        source: SearchSource::Fact {
            fact_id: "current".into(),
            namespace: "general".into(),
        },
        score: 0.10,
        bm25_rank: Some(2),
        vector_rank: Some(1),
        cosine_similarity: Some(0.8),
    };
    let ranked = rerank_state_aware(
        vec![
            StateRerankCandidateV1::from_result(stale, PremiseStatus::Stale, 1.0),
            StateRerankCandidateV1::from_result(current, PremiseStatus::Supported, 1.0),
        ],
        StateRerankWeightsV1::default(),
    );
    assert_eq!(ranked[0].result.content, "service color is green");
    assert_eq!(ranked[1].result.score, 0.99);
    assert_eq!(ranked[1].result.bm25_rank, Some(1));
}

#[tokio::test]
async fn sufficient_packet_has_answer_level_receipt_and_exact_noop_ablation() {
    let (store, _tmp) = store();
    store
        .authority()
        .append(
            permit("principal:alice"),
            "evidence-sufficient".into(),
            "general".into(),
            "service color is green".into(),
            Some("fixture:current".into()),
        )
        .await
        .unwrap();

    let mut request = EvidenceGapRequestV1::new("service color green", StateView::Current);
    request.required_evidence = vec!["green".into()];
    request.access_request = Some(access());
    request.access_request = Some(GovernedAccessRequestV1::new(
        "principal:alice",
        "principal:alice",
        GovernedAccessPurposeV1::Recall,
        "general",
    ));
    request.budget = 4;
    let packet: EvidencePacketV1 = store.retrieve_evidence(request).await.unwrap();

    assert_eq!(packet.outcome, EvidenceTerminalOutcomeV1::Sufficient);
    assert!(packet.answer.is_some());
    assert_eq!(
        packet.state_resolution_receipt.state_view,
        StateView::Current
    );
    assert_eq!(
        packet.retrieval_witness.ordered_result_ids,
        packet
            .items
            .iter()
            .map(|item| item.result.source.result_id())
            .collect::<Vec<_>>()
    );
    assert!(!packet.packet_digest.is_empty());
    assert!(!packet.ablation.packet_digest_without_gap_loop.is_empty());
    assert!(!packet.ablation.packet_digest_with_gap_loop.is_empty());
    assert!(!packet.ablation.improved);
}

#[tokio::test]
async fn missing_evidence_gets_bounded_follow_up_and_reports_improvement() {
    let (store, _tmp) = store();
    store
        .authority()
        .append(
            permit("principal:alice"),
            "evidence-missing".into(),
            "general".into(),
            "service color is green".into(),
            None,
        )
        .await
        .unwrap();

    let mut request = EvidenceGapRequestV1::new("service status", StateView::Current);
    request.required_evidence = vec!["green".into()];
    request.budget = 8;
    request.follow_up_limit = 4;
    request.access_request = Some(access());
    let packet = store.retrieve_evidence(request).await.unwrap();

    assert_eq!(packet.outcome, EvidenceTerminalOutcomeV1::Sufficient);
    assert!(packet.ablation.improved);
    assert!(packet.ablation.answer_changed || packet.ablation.order_changed);
    assert!(packet.routes.len() <= 5);
    assert!(packet.gaps.is_empty());
}

#[tokio::test]
async fn budget_conflict_and_invalid_lineage_are_terminal_and_not_negative() {
    let (store, _tmp) = store();
    let authority = store.authority();
    let old = authority
        .append(
            permit("principal:alice"),
            "terminal-old".into(),
            "general".into(),
            "service mode was safe".into(),
            None,
        )
        .await
        .unwrap();
    let new = authority
        .append(
            permit("principal:alice"),
            "terminal-new".into(),
            "general".into(),
            "service mode is safe".into(),
            None,
        )
        .await
        .unwrap();
    let second_new = authority
        .append(
            permit("principal:alice"),
            "terminal-second-new".into(),
            "general".into(),
            "service mode is safe now".into(),
            None,
        )
        .await
        .unwrap();

    let mut budget = EvidenceGapRequestV1::new("never-present", StateView::Current);
    budget.budget = 0;
    budget.access_request = Some(access());
    let budget_packet = store.retrieve_evidence(budget).await.unwrap();
    assert_eq!(
        budget_packet.outcome,
        EvidenceTerminalOutcomeV1::BudgetExceeded
    );
    assert!(budget_packet.answer.is_none());
    assert!(!budget_packet.answer_policy.confident_negative);

    let mut insufficient = EvidenceGapRequestV1::new("never-present", StateView::Current);
    insufficient.top_k = 0;
    insufficient.budget = 1;
    insufficient.access_request = Some(access());
    let insufficient_packet = store.retrieve_evidence(insufficient).await.unwrap();
    assert_eq!(
        insufficient_packet.outcome,
        EvidenceTerminalOutcomeV1::EvidenceInsufficient
    );
    assert!(insufficient_packet.answer.is_none());

    store
        .add_state_dependency_edge(
            semantic_memory::StateDependencyEdgeV1::invalidates(
                format!("fact:{}", new.affected_ids[0]),
                format!("fact:{}", old.affected_ids[0]),
            ),
            1.0,
        )
        .await
        .unwrap();
    store
        .add_state_dependency_edge(
            semantic_memory::StateDependencyEdgeV1::invalidates(
                format!("fact:{}", second_new.affected_ids[0]),
                format!("fact:{}", old.affected_ids[0]),
            ),
            1.0,
        )
        .await
        .unwrap();
    let mut conflict = EvidenceGapRequestV1::new("service mode safe", StateView::Current);
    conflict.budget = 8;
    conflict.access_request = Some(access());
    let conflict_packet = store.retrieve_evidence(conflict).await.unwrap();
    assert_ne!(
        conflict_packet.outcome,
        EvidenceTerminalOutcomeV1::Sufficient
    );
    assert!(!conflict_packet.answer_policy.confident_negative);

    store
        .add_state_dependency_edge(
            semantic_memory::StateDependencyEdgeV1::derived_from_state(
                format!("fact:{}", old.affected_ids[0]),
                format!("fact:{}", new.affected_ids[0]),
            ),
            1.0,
        )
        .await
        .unwrap();
    let mut invalid = EvidenceGapRequestV1::new("service mode safe", StateView::Current);
    invalid.budget = 8;
    invalid.access_request = Some(access());
    let invalid_packet = store.retrieve_evidence(invalid).await.unwrap();
    assert_eq!(
        invalid_packet.outcome,
        EvidenceTerminalOutcomeV1::InvalidLineage
    );
    assert!(invalid_packet.answer.is_none());
}

#[tokio::test]
async fn direct_ids_and_cached_search_remain_origin_contained() {
    let (store, _tmp) = store();
    let receipt = store
        .authority()
        .append(
            permit("principal:alice"),
            "origin-contained".into(),
            "general".into(),
            "private evidence marker".into(),
            None,
        )
        .await
        .unwrap();
    let fact_id = receipt.affected_ids[0].clone();

    let mut direct = EvidenceGapRequestV1::new("does not matter", StateView::Current);
    direct.direct_ids = vec![format!("fact:{fact_id}")];
    direct.access_request = Some(GovernedAccessRequestV1::new(
        "principal:bob",
        "principal:bob",
        GovernedAccessPurposeV1::Recall,
        "general",
    ));
    direct.budget = 4;
    let denied = store.retrieve_evidence(direct).await.unwrap();
    assert!(denied.items.is_empty());
    assert!(denied
        .gaps
        .iter()
        .any(|gap| gap.missing_items.iter().any(|item| item.contains("origin"))));

    let mut first = EvidenceGapRequestV1::new("private evidence marker", StateView::Current);
    first.access_request = Some(GovernedAccessRequestV1::new(
        "principal:alice",
        "principal:alice",
        GovernedAccessPurposeV1::Recall,
        "general",
    ));
    first.budget = 2;
    let _ = store.retrieve_evidence(first).await.unwrap();

    let mut cached_denied =
        EvidenceGapRequestV1::new("private evidence marker", StateView::Current);
    cached_denied.access_request = Some(GovernedAccessRequestV1::new(
        "principal:bob",
        "principal:bob",
        GovernedAccessPurposeV1::Recall,
        "general",
    ));
    cached_denied.budget = 2;
    let denied_cached = store.retrieve_evidence(cached_denied).await.unwrap();
    assert!(denied_cached.items.is_empty());
    assert!(denied_cached
        .retrieval_witness
        .ordered_result_ids
        .is_empty());
}
