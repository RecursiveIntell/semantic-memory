#![cfg(feature = "rl-routing")]
#![allow(clippy::expect_used)]

use semantic_memory::rl_routing::{
    is_trained, record_routing_outcome, route_with_policy, RoutingOutcome, RoutingPolicy,
};
use semantic_memory::routing::{QueryProfile, RetrievalRouter, RoutingDecision};
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder};

fn open_store(config: MemoryConfig) -> MemoryStore {
    let dimensions = config.embedding.dimensions;
    MemoryStore::open_with_embedder(config, Box::new(MockEmbedder::new(dimensions)))
        .expect("open routing-policy test store")
}

fn same_stages(left: &RoutingDecision, right: &RoutingDecision) -> bool {
    left.bm25_coarse == right.bm25_coarse
        && left.vector_medium == right.vector_medium
        && left.rerank_fine == right.rerank_fine
        && left.graph_expansion == right.graph_expansion
        && left.decoder == right.decoder
        && left.discord == right.discord
        && left.no_retrieval == right.no_retrieval
}

#[tokio::test]
async fn trained_policy_persists_and_routes_identically_after_restart() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = MemoryConfig {
        base_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    let store = open_store(config.clone());

    let profile = QueryProfile::from_query("compare rust vs python performance");
    let heuristic = RetrievalRouter::default().route(&profile);
    assert!(heuristic.rerank_fine);

    let mut policy = RoutingPolicy::default();
    for _ in 0..11 {
        record_routing_outcome(&mut policy, &profile, &heuristic, RoutingOutcome::Bad);
    }
    assert!(is_trained(&policy));
    assert!(policy.last_updated.is_some());

    let before_restart = route_with_policy(&policy, &profile);
    assert!(
        !same_stages(&before_restart, &heuristic),
        "training should change at least one routing stage"
    );
    store
        .save_routing_policy(&policy)
        .await
        .expect("persist routing policy");
    drop(store);

    let restarted = open_store(config);
    let loaded = restarted
        .load_routing_policy()
        .await
        .expect("load routing policy")
        .expect("persisted routing policy");
    assert_eq!(loaded.trained_examples, policy.trained_examples);
    for (stage, expected) in &policy.weights {
        let actual = loaded.weights.get(stage).expect("persisted stage weight");
        assert!(
            (actual - expected).abs() < 1e-12,
            "weight changed for {stage}: expected {expected}, loaded {actual}"
        );
    }
    assert_eq!(loaded.last_updated, policy.last_updated);

    let after_restart = route_with_policy(&loaded, &profile);
    assert!(same_stages(&before_restart, &after_restart));
}
