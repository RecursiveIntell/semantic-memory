use semantic_memory::whole_path_receipt::{
    BuildMetadata, ByteAccounting, DecodeMetrics, LatencyMetrics, QualityMetrics,
    WholePathReceiptV1, Workload,
};

fn valid() -> WholePathReceiptV1 {
    WholePathReceiptV1 {
        schema_version: 1,
        workload: Workload {
            dimensions: 4,
            corpus_size: 8,
            top_k: 2,
            candidate_k: 4,
            seed: 7,
            iterations: 1,
        },
        build: BuildMetadata {
            profile: "release".into(),
            cpu: "test".into(),
            kernel: "test".into(),
            os: "test".into(),
            rustc: "test".into(),
            cargo: "test".into(),
            target: "test".into(),
            source_head: "abc".into(),
            source_status_digest: "def".into(),
        },
        bytes: ByteAccounting {
            authoritative_raw_f32_bytes: 128,
            compressed_payload_bytes: 64,
            manifest_bytes: 1,
            receipt_bytes: 1,
            index_bytes: 1,
            codebook_bytes: 1,
            fallback_bytes: 128,
            reader_scratch_bytes: None,
            total_resident_derived_bytes: 196,
        },
        quality: QualityMetrics {
            candidate_recall_at_k: 1.0,
            candidate_precision_at_candidate_k: 1.0,
            exact_rerank_ordered_parity: true,
            exact_rerank_tie_policy: "score_desc_token_asc".into(),
            exact_rerank_overlap_at_k: 1.0,
        },
        latency: LatencyMetrics {
            exact_baseline_ns: Vec::new(),
            scoring_ns: vec![1],
            prepare_ns: vec![1],
            rerank_ns: vec![1],
            whole_path_ns: Vec::new(),
        },
        decoding: DecodeMetrics {
            modeled_selected_values: 2,
            modeled_full_decode_values: 8,
            observed_decode_calls: None,
            observed_decoded_values: None,
        },
        semantic_path: None,
        fallback_disposition: "not_observed".into(),
        evidence_limitations: vec!["synthetic schema fixture".into()],
    }
}
#[test]
fn schema_rejects_invalid_k_and_decode_counts() {
    let mut r = valid();
    r.workload.candidate_k = 1;
    assert!(r.validate().is_err());
    let mut r = valid();
    r.decoding.modeled_selected_values = 9;
    assert!(r.validate().is_err());
}
#[test]
fn schema_rejects_non_finite_quality_and_wrong_latency_lengths() {
    let mut r = valid();
    r.quality.candidate_recall_at_k = f64::NAN;
    assert!(r.validate().is_err());
    let mut r = valid();
    r.latency.rerank_ns.clear();
    assert!(r.validate().is_err());
}
#[test]
fn schema_round_trips_json() {
    let r = valid();
    r.validate().unwrap();
    let encoded = serde_json::to_vec(&r).unwrap();
    let decoded: WholePathReceiptV1 = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, r);
}
