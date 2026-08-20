use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WholePathReceiptV1 {
    pub schema_version: u16,
    pub workload: Workload,
    pub build: BuildMetadata,
    pub bytes: ByteAccounting,
    pub quality: QualityMetrics,
    pub latency: LatencyMetrics,
    pub decoding: DecodeMetrics,
    /// Canonical semantic-memory persisted-generation execution evidence (schema v2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_path: Option<SemanticPathEvidence>,
    pub fallback_disposition: String,
    pub evidence_limitations: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Workload {
    pub dimensions: usize,
    pub corpus_size: usize,
    pub top_k: usize,
    pub candidate_k: usize,
    pub seed: u64,
    pub iterations: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BuildMetadata {
    pub profile: String,
    pub cpu: String,
    pub kernel: String,
    pub os: String,
    pub rustc: String,
    pub cargo: String,
    pub target: String,
    pub source_head: String,
    pub source_status_digest: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ByteAccounting {
    pub authoritative_raw_f32_bytes: u64,
    pub compressed_payload_bytes: u64,
    pub manifest_bytes: u64,
    pub receipt_bytes: u64,
    pub index_bytes: u64,
    pub codebook_bytes: u64,
    pub fallback_bytes: u64,
    pub reader_scratch_bytes: Option<u64>,
    pub total_resident_derived_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityMetrics {
    pub candidate_recall_at_k: f64,
    pub candidate_precision_at_candidate_k: f64,
    pub exact_rerank_ordered_parity: bool,
    pub exact_rerank_tie_policy: String,
    pub exact_rerank_overlap_at_k: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatencyMetrics {
    /// Exact authoritative f32 top-k baseline over the complete corpus.
    #[serde(default)]
    pub exact_baseline_ns: Vec<u128>,
    pub scoring_ns: Vec<u128>,
    pub prepare_ns: Vec<u128>,
    pub rerank_ns: Vec<u128>,
    /// Public semantic-memory search API latency including admission, compressed scoring,
    /// filtering, exact rerank, result materialization, and receipt persistence.
    #[serde(default)]
    pub whole_path_ns: Vec<u128>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecodeMetrics {
    pub modeled_selected_values: u64,
    pub modeled_full_decode_values: u64,
    pub observed_decode_calls: Option<u64>,
    pub observed_decoded_values: Option<u64>,
}

/// Evidence emitted by the canonical semantic-memory persisted-generation path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticPathEvidence {
    pub artifact_generation_id: String,
    pub source_snapshot_digest: String,
    pub source_digest: String,
    pub pool_manifest_digest: String,
    pub codec_profile_digest: String,
    pub artifact_digest: String,
    pub candidate_backend: String,
    pub fresh_process_reload: bool,
    pub fresh_process_probe_digest: String,
    pub exact_f32_rerank: bool,
    pub build_ns: u128,
    pub persisted_bundle_bytes: u64,
    pub generation_row_canonical_bytes: u64,
    pub item_map_canonical_bytes: u64,
    pub sqlite_main_bytes: u64,
    pub sqlite_wal_bytes: u64,
    pub sqlite_shm_bytes: u64,
    pub durable_search_receipt_ids: Vec<String>,
}
impl WholePathReceiptV1 {
    pub fn validate(&self) -> Result<(), String> {
        let w = &self.workload;
        if !matches!(self.schema_version, 1 | 2)
            || w.dimensions == 0
            || w.corpus_size == 0
            || w.iterations == 0
            || w.top_k == 0
            || w.top_k > w.corpus_size
            || w.candidate_k < w.top_k
            || w.candidate_k > w.corpus_size
        {
            return Err("invalid schema/workload bounds".into());
        }
        for (name, v) in [
            ("candidate recall", self.quality.candidate_recall_at_k),
            (
                "candidate precision",
                self.quality.candidate_precision_at_candidate_k,
            ),
            ("exact overlap", self.quality.exact_rerank_overlap_at_k),
        ] {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(format!("{name} must be finite and in [0,1]"));
            }
        }
        if self.quality.exact_rerank_tie_policy.is_empty()
            || self.build.profile.is_empty()
            || self.build.cpu.is_empty()
            || self.build.kernel.is_empty()
            || self.build.rustc.is_empty()
            || self.build.cargo.is_empty()
            || self.build.target.is_empty()
            || self.build.source_head.is_empty()
            || self.build.source_status_digest.is_empty()
        {
            return Err("build identity and tie policy are required".into());
        }
        if (self.schema_version == 2 && self.latency.exact_baseline_ns.len() != w.iterations)
            || (self.schema_version == 1 && !self.latency.exact_baseline_ns.is_empty())
            || self.latency.scoring_ns.len() != w.iterations
            || self.latency.prepare_ns.len() != 1
            || self.latency.rerank_ns.len() != w.iterations
            || (self.schema_version == 2 && self.latency.whole_path_ns.len() != w.iterations)
            || (self.schema_version == 1 && !self.latency.whole_path_ns.is_empty())
        {
            return Err(
                "latency lengths must match iterations (prepare exactly once; whole-path for v2)"
                    .into(),
            );
        }
        if self.decoding.modeled_selected_values > self.decoding.modeled_full_decode_values
            || self.fallback_disposition.is_empty()
            || self.evidence_limitations.is_empty()
        {
            return Err("invalid decode accounting or missing evidence limits".into());
        }
        match (self.schema_version, self.semantic_path.as_ref()) {
            (1, None) => {}
            (2, Some(path))
                if path.fresh_process_reload
                    && !path.fresh_process_probe_digest.is_empty()
                    && path.exact_f32_rerank
                    && path.build_ns > 0
                    && path.persisted_bundle_bytes > 0
                    && path.sqlite_main_bytes + path.sqlite_wal_bytes > 0
                    && !path.artifact_generation_id.is_empty()
                    && !path.source_snapshot_digest.is_empty()
                    && !path.source_digest.is_empty()
                    && !path.pool_manifest_digest.is_empty()
                    && !path.codec_profile_digest.is_empty()
                    && !path.artifact_digest.is_empty()
                    && path.candidate_backend == "poly_kv_fibquant_persisted_generation"
                    && path.durable_search_receipt_ids.len() == w.iterations => {}
            _ => return Err("schema v2 requires complete canonical semantic-path evidence".into()),
        }
        let resident = self.bytes.compressed_payload_bytes
            + self.bytes.manifest_bytes
            + self.bytes.receipt_bytes
            + self.bytes.index_bytes
            + self.bytes.codebook_bytes
            + self.bytes.fallback_bytes
            + self.bytes.reader_scratch_bytes.unwrap_or(0);
        if self.bytes.authoritative_raw_f32_bytes == 0
            || self.bytes.total_resident_derived_bytes < resident
        {
            return Err("byte accounting is inconsistent".into());
        }
        Ok(())
    }
}
