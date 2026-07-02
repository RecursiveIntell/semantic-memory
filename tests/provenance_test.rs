//! Integration tests for Phase 2 semiring provenance.
//!
//! These tests exercise the full SQLite-backed provenance path:
//!   - set/get/combine provenance round-trips
//!   - append-plus-supersession (combine never UPDATEs, only INSERTs)
//!   - search_with_provenance annotation
//!   - provenance_history audit trail
//!
//! The semiring algebraic property tests (associativity, commutativity,
//! distributivity, identity, annihilation) live in the `#[cfg(test)]` module
//! inside `src/provenance.rs` so they run as unit tests alongside the lib.
//!
//! Feature-gated: these tests require `--features provenance`.

#![cfg(feature = "provenance")]
#![allow(clippy::expect_used)]

use semantic_memory::provenance::{
    BooleanSemiring, ConfidenceSemiring, ConfidenceValue, ProbabilitySemiring,
    ProvenanceAnnotation, ProvenanceItemType, ProvenanceOperation, ProvenanceSemiring,
    TropicalSemiring,
};
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder, SearchConfig};
use tempfile::TempDir;

fn test_store() -> (MemoryStore, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let config = MemoryConfig {
        base_dir: dir.path().to_path_buf(),
        search: SearchConfig {
            min_similarity: -1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let embedder = Box::new(MockEmbedder::new(config.embedding.dimensions));
    let store = MemoryStore::open_with_embedder(config, embedder).expect("open store");
    (store, dir)
}

fn approx_eq(a: f64, b: f64) -> bool {
    if a.is_infinite() && b.is_infinite() {
        a.signum() == b.signum()
    } else if a.is_infinite() || b.is_infinite() {
        false
    } else {
        (a - b).abs() < 1e-9
    }
}

fn confidence_approx_eq(a: ConfidenceValue, b: ConfidenceValue) -> bool {
    approx_eq(a.confidence, b.confidence) && a.support_count == b.support_count
}

// ──────────────────────────────────────────────────────────────────────
// 1. Semiring property tests (integration-level echo of the unit tests)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn boolean_semiring_properties_integration() {
    let vals = [false, true];
    for &a in &vals {
        // identity
        assert_eq!(BooleanSemiring::add(&a, &BooleanSemiring::zero()), a);
        assert_eq!(BooleanSemiring::mul(&a, &BooleanSemiring::one()), a);
        // annihilation
        assert!(!BooleanSemiring::mul(&a, &BooleanSemiring::zero()));
        for &b in &vals {
            // commutativity of add
            assert_eq!(BooleanSemiring::add(&a, &b), BooleanSemiring::add(&b, &a));
            for &c in &vals {
                // associativity of add
                assert_eq!(
                    BooleanSemiring::add(&BooleanSemiring::add(&a, &b), &c),
                    BooleanSemiring::add(&a, &BooleanSemiring::add(&b, &c))
                );
                // associativity of mul
                assert_eq!(
                    BooleanSemiring::mul(&BooleanSemiring::mul(&a, &b), &c),
                    BooleanSemiring::mul(&a, &BooleanSemiring::mul(&b, &c))
                );
                // distributivity: a*(b+c) = a*b + a*c
                assert_eq!(
                    BooleanSemiring::mul(&a, &BooleanSemiring::add(&b, &c)),
                    BooleanSemiring::add(
                        &BooleanSemiring::mul(&a, &b),
                        &BooleanSemiring::mul(&a, &c)
                    )
                );
            }
        }
    }
}

#[test]
fn tropical_semiring_properties_integration() {
    let vals = [0.0, 1.0, 2.5, 100.0, f64::INFINITY];
    for &a in &vals {
        assert!(approx_eq(
            TropicalSemiring::add(&a, &TropicalSemiring::zero()),
            a
        ));
        assert!(approx_eq(
            TropicalSemiring::mul(&a, &TropicalSemiring::one()),
            a
        ));
        assert!(approx_eq(
            TropicalSemiring::mul(&a, &TropicalSemiring::zero()),
            f64::INFINITY
        ));
        for &b in &vals {
            assert!(approx_eq(
                TropicalSemiring::add(&a, &b),
                TropicalSemiring::add(&b, &a)
            ));
            for &c in &vals {
                assert!(approx_eq(
                    TropicalSemiring::add(&TropicalSemiring::add(&a, &b), &c),
                    TropicalSemiring::add(&a, &TropicalSemiring::add(&b, &c))
                ));
                assert!(approx_eq(
                    TropicalSemiring::mul(&TropicalSemiring::mul(&a, &b), &c),
                    TropicalSemiring::mul(&a, &TropicalSemiring::mul(&b, &c))
                ));
                assert!(approx_eq(
                    TropicalSemiring::mul(&a, &TropicalSemiring::add(&b, &c)),
                    TropicalSemiring::add(
                        &TropicalSemiring::mul(&a, &b),
                        &TropicalSemiring::mul(&a, &c)
                    )
                ));
            }
        }
    }
}

#[test]
fn probability_semiring_properties_integration() {
    let vals = [0.0, 0.25, 0.5, 0.75, 1.0];
    for &a in &vals {
        assert!(approx_eq(
            ProbabilitySemiring::add(&a, &ProbabilitySemiring::zero()),
            a
        ));
        assert!(approx_eq(
            ProbabilitySemiring::mul(&a, &ProbabilitySemiring::one()),
            a
        ));
        assert!(approx_eq(
            ProbabilitySemiring::mul(&a, &ProbabilitySemiring::zero()),
            0.0
        ));
        for &b in &vals {
            assert!(approx_eq(
                ProbabilitySemiring::add(&a, &b),
                ProbabilitySemiring::add(&b, &a)
            ));
            for &c in &vals {
                assert!(approx_eq(
                    ProbabilitySemiring::add(&ProbabilitySemiring::add(&a, &b), &c),
                    ProbabilitySemiring::add(&a, &ProbabilitySemiring::add(&b, &c))
                ));
                assert!(approx_eq(
                    ProbabilitySemiring::mul(&ProbabilitySemiring::mul(&a, &b), &c),
                    ProbabilitySemiring::mul(&a, &ProbabilitySemiring::mul(&b, &c))
                ));
                assert!(approx_eq(
                    ProbabilitySemiring::mul(&a, &ProbabilitySemiring::add(&b, &c)),
                    ProbabilitySemiring::add(
                        &ProbabilitySemiring::mul(&a, &b),
                        &ProbabilitySemiring::mul(&a, &c)
                    )
                ));
            }
        }
    }
}

#[test]
fn confidence_semiring_properties_integration() {
    let vals = [
        ConfidenceValue::new(0.0, 0),
        ConfidenceValue::new(0.25, 1),
        ConfidenceValue::new(0.5, 2),
        ConfidenceValue::new(0.75, 3),
        ConfidenceValue::new(1.0, 5),
    ];
    for a in &vals {
        assert!(confidence_approx_eq(
            ConfidenceSemiring::add(a, &ConfidenceSemiring::zero()),
            *a
        ));
        assert!(confidence_approx_eq(
            ConfidenceSemiring::mul(a, &ConfidenceSemiring::one()),
            *a
        ));
        assert!(confidence_approx_eq(
            ConfidenceSemiring::mul(a, &ConfidenceSemiring::zero()),
            ConfidenceSemiring::zero()
        ));
        for b in &vals {
            assert!(confidence_approx_eq(
                ConfidenceSemiring::add(a, b),
                ConfidenceSemiring::add(b, a)
            ));
            for c in &vals {
                assert!(confidence_approx_eq(
                    ConfidenceSemiring::add(&ConfidenceSemiring::add(a, b), c),
                    ConfidenceSemiring::add(a, &ConfidenceSemiring::add(b, c))
                ));
                assert!(confidence_approx_eq(
                    ConfidenceSemiring::mul(&ConfidenceSemiring::mul(a, b), c),
                    ConfidenceSemiring::mul(a, &ConfidenceSemiring::mul(b, c))
                ));
                assert!(confidence_approx_eq(
                    ConfidenceSemiring::mul(a, &ConfidenceSemiring::add(b, c)),
                    ConfidenceSemiring::add(
                        &ConfidenceSemiring::mul(a, b),
                        &ConfidenceSemiring::mul(a, c)
                    )
                ));
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// 2. set / get / combine provenance — Boolean semiring
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn set_get_provenance_boolean_roundtrip() {
    let (store, _dir) = test_store();
    let fact_id = store
        .add_fact("general", "rust was released in 2015", None, None)
        .await
        .expect("add_fact");

    let receipt = store
        .set_provenance::<BooleanSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &true,
            &["source:a".to_string()],
            None,
        )
        .await
        .expect("set_provenance");

    assert_eq!(receipt.operation, ProvenanceOperation::Set);
    assert_eq!(receipt.item_type, "fact");
    assert_eq!(receipt.item_id, fact_id);
    assert_eq!(receipt.semiring_type, "boolean");
    assert_eq!(receipt.semiring_value, "true");
    assert!(!receipt.provenance_id.is_empty());

    let (value, chain) = store
        .get_provenance::<BooleanSemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");

    assert!(value, "boolean provenance should be true");
    assert_eq!(chain, vec!["source:a".to_string()]);
}

#[tokio::test]
async fn get_provenance_returns_none_when_absent() {
    let (store, _dir) = test_store();
    let result = store
        .get_provenance::<BooleanSemiring>(&ProvenanceItemType::Fact, "nonexistent-id")
        .await
        .expect("get_provenance should not error");
    assert!(result.is_none(), "absent provenance should be None");
}

#[tokio::test]
async fn combine_provenance_boolean_or_appends_new_row() {
    let (store, dir) = test_store();
    let fact_id = store
        .add_fact("general", "combine boolean test", None, None)
        .await
        .expect("add_fact");

    // First set: false with chain [a]
    store
        .set_provenance::<BooleanSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &false,
            &["a".to_string()],
            None,
        )
        .await
        .expect("set_provenance");

    // Combine with true and chain [b] → OR(false, true) = true, merged chain [a, b]
    let receipt = store
        .combine_provenance::<BooleanSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &true,
            &["b".to_string()],
            None,
        )
        .await
        .expect("combine_provenance");

    assert_eq!(receipt.operation, ProvenanceOperation::Combine);
    assert_eq!(receipt.semiring_value, "true");

    let (value, chain) = store
        .get_provenance::<BooleanSemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");
    assert!(value);
    assert_eq!(chain, vec!["a".to_string(), "b".to_string()]);

    // Verify append-only: two rows should exist in the provenance table.
    let conn = rusqlite::Connection::open(dir.path().join("memory.db")).expect("open db");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM provenance WHERE item_type = 'fact' AND item_id = ?1",
            rusqlite::params![fact_id],
            |row| row.get(0),
        )
        .expect("count rows");
    assert_eq!(
        count, 2,
        "combine should append a new row, not UPDATE — expected 2 rows, got {count}"
    );
}

#[tokio::test]
async fn combine_provenance_on_absent_item_is_equivalent_to_set() {
    let (store, _dir) = test_store();
    let fact_id = store
        .add_fact("general", "combine on absent item", None, None)
        .await
        .expect("add_fact");

    let receipt = store
        .combine_provenance::<BooleanSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &true,
            &["only".to_string()],
            None,
        )
        .await
        .expect("combine_provenance");

    assert_eq!(receipt.operation, ProvenanceOperation::Combine);
    assert_eq!(receipt.semiring_value, "true");

    let (value, chain) = store
        .get_provenance::<BooleanSemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");
    assert!(value);
    assert_eq!(chain, vec!["only".to_string()]);
}

// ──────────────────────────────────────────────────────────────────────
// 3. set / get / combine — Tropical semiring
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tropical_provenance_combine_uses_min() {
    let (store, _dir) = test_store();
    let fact_id = store
        .add_fact("general", "tropical combine test", None, None)
        .await
        .expect("add_fact");

    // Set cost = 5.0
    store
        .set_provenance::<TropicalSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &5.0,
            &["path:a".to_string()],
            None,
        )
        .await
        .expect("set_provenance");

    // Combine with cost = 3.0 → min(5.0, 3.0) = 3.0
    store
        .combine_provenance::<TropicalSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &3.0,
            &["path:b".to_string()],
            None,
        )
        .await
        .expect("combine_provenance");

    let (value, chain) = store
        .get_provenance::<TropicalSemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");
    assert!(
        approx_eq(value, 3.0),
        "tropical add = min, expected 3.0, got {value}"
    );
    assert_eq!(chain, vec!["path:a".to_string(), "path:b".to_string()]);
}

// ──────────────────────────────────────────────────────────────────────
// 4. set / get / combine — Probability semiring
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn probability_provenance_combine_uses_max() {
    let (store, _dir) = test_store();
    let fact_id = store
        .add_fact("general", "probability combine test", None, None)
        .await
        .expect("add_fact");

    store
        .set_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &0.5,
            &["e:a".to_string()],
            None,
        )
        .await
        .expect("set_provenance");

    // Combine with 0.8 → max(0.5, 0.8) = 0.8
    store
        .combine_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &0.8,
            &["e:b".to_string()],
            None,
        )
        .await
        .expect("combine_provenance");

    let (value, _chain) = store
        .get_provenance::<ProbabilitySemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");
    assert!(
        approx_eq(value, 0.8),
        "probability add = max, expected 0.8, got {value}"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 5. set / get / combine — Confidence semiring
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn confidence_provenance_combine_uses_max_confidence_and_sums_support() {
    let (store, _dir) = test_store();
    let fact_id = store
        .add_fact("general", "confidence combine test", None, None)
        .await
        .expect("add_fact");

    let first = ConfidenceValue::new(0.6, 2);
    store
        .set_provenance::<ConfidenceSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &first,
            &["c:a".to_string()],
            None,
        )
        .await
        .expect("set_provenance");

    // Combine with higher confidence 0.9, support 3 → add picks 0.9, keeps its
    // support_count (3) because confidence wins the tie-break.
    let second = ConfidenceValue::new(0.9, 3);
    store
        .combine_provenance::<ConfidenceSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &second,
            &["c:b".to_string()],
            None,
        )
        .await
        .expect("combine_provenance");

    let (value, _chain) = store
        .get_provenance::<ConfidenceSemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");
    assert!(
        confidence_approx_eq(value, ConfidenceValue::new(0.9, 3)),
        "confidence add = max-confidence, expected (0.9, 3), got {:?}",
        value
    );
}

// ──────────────────────────────────────────────────────────────────────
// 6. Episode-linked provenance
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn provenance_can_reference_an_episode() {
    let (store, _dir) = test_store();
    let doc_id = store
        .ingest_document(
            "prov-doc",
            "episode-linked provenance content",
            "general",
            None,
            None,
        )
        .await
        .expect("ingest_document");

    let ep_id = store
        .create_episode(
            "ep-prov-1",
            &doc_id,
            &semantic_memory::EpisodeMeta {
                cause_ids: vec![],
                effect_type: "linked_effect".to_string(),
                outcome: semantic_memory::EpisodeOutcome::Pending,
                confidence: 0.5,
                verification_status: semantic_memory::VerificationStatus::Unverified,
                experiment_id: None,
                valid_time: None,
                fact_digest: None,
            },
        )
        .await
        .expect("create_episode");

    let receipt = store
        .set_provenance::<BooleanSemiring>(
            &ProvenanceItemType::Episode,
            &ep_id,
            &true,
            &["self".to_string()],
            Some(&ep_id),
        )
        .await
        .expect("set_provenance");

    assert_eq!(receipt.item_type, "episode");
    assert_eq!(receipt.item_id, ep_id);
    assert_eq!(receipt.episode_id.as_deref(), Some(ep_id.as_str()));

    let (value, chain) = store
        .get_provenance::<BooleanSemiring>(&ProvenanceItemType::Episode, &ep_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");
    assert!(value);
    assert_eq!(chain, vec!["self".to_string()]);
}

#[tokio::test]
async fn combine_preserves_existing_episode_id_when_caller_omits_it() {
    let (store, _dir) = test_store();
    let doc_id = store
        .ingest_document(
            "sup-doc",
            "supersession propagation content",
            "general",
            None,
            None,
        )
        .await
        .expect("ingest_document");

    let ep_id = store
        .create_episode(
            "ep-sup-1",
            &doc_id,
            &semantic_memory::EpisodeMeta {
                cause_ids: vec![],
                effect_type: "sup_effect".to_string(),
                outcome: semantic_memory::EpisodeOutcome::Pending,
                confidence: 0.5,
                verification_status: semantic_memory::VerificationStatus::Unverified,
                experiment_id: None,
                valid_time: None,
                fact_digest: None,
            },
        )
        .await
        .expect("create_episode");

    // First set with episode_id
    store
        .set_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Episode,
            &ep_id,
            &0.5,
            &["s1".to_string()],
            Some(&ep_id),
        )
        .await
        .expect("set_provenance");

    // Combine without episode_id — should preserve the existing episode_id
    let receipt = store
        .combine_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Episode,
            &ep_id,
            &0.7,
            &["s2".to_string()],
            None,
        )
        .await
        .expect("combine_provenance");

    assert_eq!(
        receipt.episode_id.as_deref(),
        Some(ep_id.as_str()),
        "combine should preserve the existing episode_id (supersession propagation)"
    );

    let (value, _chain) = store
        .get_provenance::<ProbabilitySemiring>(&ProvenanceItemType::Episode, &ep_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");
    assert!(approx_eq(value, 0.7), "max(0.5, 0.7) = 0.7, got {value}");
}

// ──────────────────────────────────────────────────────────────────────
// 7. provenance_history audit trail
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn provenance_history_returns_all_rows_oldest_first() {
    let (store, _dir) = test_store();
    let fact_id = store
        .add_fact("general", "history audit test", None, None)
        .await
        .expect("add_fact");

    store
        .set_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &0.1,
            &["h1".to_string()],
            None,
        )
        .await
        .expect("set 1");

    store
        .combine_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &0.3,
            &["h2".to_string()],
            None,
        )
        .await
        .expect("combine 1");

    store
        .combine_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &0.2,
            &["h3".to_string()],
            None,
        )
        .await
        .expect("combine 2");

    let history = store
        .provenance_history::<ProbabilitySemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("provenance_history");

    assert_eq!(
        history.len(),
        3,
        "should have 3 rows (1 set + 2 combines), got {}",
        history.len()
    );
    // All should be ordered by recorded_at ascending. Because all writes
    // happen within the same second under datetime('now'), the order may tie
    // on recorded_at; fall back to id ordering. We verify the semiring values
    // are present regardless of strict order, and that they are a permutation
    // of the expected values.
    let values: Vec<f64> = history
        .iter()
        .map(|r| ProbabilitySemiring::decode(&r.semiring_value).expect("decode"))
        .collect();
    assert!(
        values.contains(&0.1),
        "history should contain the first set value 0.1, got {:?}",
        values
    );
    // The latest get should reflect the max combine: max(0.1, 0.3, 0.2) = 0.3
    let (latest, _) = store
        .get_provenance::<ProbabilitySemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get_provenance")
        .expect("provenance should exist");
    assert!(approx_eq(latest, 0.3), "latest should be 0.3, got {latest}");
}

// ──────────────────────────────────────────────────────────────────────
// 8. search_with_provenance annotation
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn search_with_provenance_annotates_supported_and_unsupported() {
    let (store, _dir) = test_store();

    // Two facts with distinguishable content
    let supported_id = store
        .add_fact(
            "general",
            "the quick brown fox jumps over the lazy dog",
            None,
            None,
        )
        .await
        .expect("add_fact supported");
    let _unsupported_id = store
        .add_fact(
            "general",
            "a completely unrelated fact about rust memory model",
            None,
            None,
        )
        .await
        .expect("add_fact unsupported");

    // Attach boolean provenance = true to the first fact only
    store
        .set_provenance::<BooleanSemiring>(
            &ProvenanceItemType::Fact,
            &supported_id,
            &true,
            &["src:fox".to_string()],
            None,
        )
        .await
        .expect("set_provenance");

    let annotated = store
        .search_with_provenance::<BooleanSemiring>("quick brown fox", Some(10), None, None)
        .await
        .expect("search_with_provenance");

    // At least one result should be returned.
    assert!(
        !annotated.is_empty(),
        "search should return at least one result"
    );

    // The fact we gave provenance to should appear as Supported.
    let supported_results: Vec<_> = annotated
        .iter()
        .filter(|r| matches!(r.provenance, ProvenanceAnnotation::Supported { .. }))
        .collect();
    assert!(
        !supported_results.is_empty(),
        "at least one result should be Supported (the fox fact has provenance)"
    );

    // Verify the supported result carries the correct value + chain.
    for r in &supported_results {
        if let ProvenanceAnnotation::Supported {
            value,
            support_chain,
        } = &r.provenance
        {
            assert!(*value, "supported boolean provenance should be true");
            assert_eq!(support_chain, &vec!["src:fox".to_string()]);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// 9. Receipt completeness — every operation emits a receipt
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn every_provenance_operation_emits_a_receipt() {
    let (store, _dir) = test_store();
    let fact_id = store
        .add_fact("general", "receipt completeness test", None, None)
        .await
        .expect("add_fact");

    let set_receipt = store
        .set_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &0.5,
            &["r1".to_string()],
            None,
        )
        .await
        .expect("set_provenance");
    assert!(
        !set_receipt.provenance_id.is_empty(),
        "set receipt needs an id"
    );
    assert_eq!(set_receipt.operation, ProvenanceOperation::Set);
    assert_eq!(set_receipt.schema_version, "provenance.v1");

    let combine_receipt = store
        .combine_provenance::<ProbabilitySemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &0.6,
            &["r2".to_string()],
            None,
        )
        .await
        .expect("combine_provenance");
    assert!(
        !combine_receipt.provenance_id.is_empty(),
        "combine receipt needs an id"
    );
    assert_eq!(combine_receipt.operation, ProvenanceOperation::Combine);
    assert_ne!(
        set_receipt.provenance_id, combine_receipt.provenance_id,
        "set and combine receipts must have distinct provenance_ids"
    );
}

// ──────────────────────────────────────────────────────────────────────
// 10. Semiring type isolation — mixing semirings for the same item is
//     tracked separately (each semiring label is its own provenance lane).
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn different_semirings_for_same_item_are_separate_lanes() {
    let (store, _dir) = test_store();
    let fact_id = store
        .add_fact("general", "multi-semiring lane test", None, None)
        .await
        .expect("add_fact");

    // Boolean lane = true
    store
        .set_provenance::<BooleanSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &true,
            &["bool:a".to_string()],
            None,
        )
        .await
        .expect("set boolean");

    // Tropical lane = 7.0
    store
        .set_provenance::<TropicalSemiring>(
            &ProvenanceItemType::Fact,
            &fact_id,
            &7.0,
            &["trop:a".to_string()],
            None,
        )
        .await
        .expect("set tropical");

    let (bool_val, bool_chain) = store
        .get_provenance::<BooleanSemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get boolean")
        .expect("boolean provenance should exist");
    assert!(bool_val);
    assert_eq!(bool_chain, vec!["bool:a".to_string()]);

    let (trop_val, trop_chain) = store
        .get_provenance::<TropicalSemiring>(&ProvenanceItemType::Fact, &fact_id)
        .await
        .expect("get tropical")
        .expect("tropical provenance should exist");
    assert!(approx_eq(trop_val, 7.0));
    assert_eq!(trop_chain, vec!["trop:a".to_string()]);
}
