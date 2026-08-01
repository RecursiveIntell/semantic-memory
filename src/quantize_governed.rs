//! Governed compression pipeline integration for semantic-memory.
//!
//! This module evaluates `quant-governor` policy and produces an artifact only
//! when the selected runtime codec has a real encoder. It never relabels raw
//! bytes as a compressed representation.

#[cfg(feature = "turbo-quant-codec")]
pub mod governed {
    use quant_governor::{
        AdmissibilityClass, CodecProfile, ContentType, GovernancePolicy, GovernanceRequest,
    };

    /// Result of governed encoding — encoded bytes plus policy metadata.
    #[derive(Debug, Clone)]
    pub struct GovernedEncodeResult {
        /// Bytes in the representation selected by `codec_profile`.
        pub compressed_bytes: Vec<u8>,
        /// Codec profile selected by the governance policy.
        pub codec_profile: CodecProfile,
        /// Governance receipt ID for the audit trail.
        pub governance_receipt_id: String,
        /// Allowed degradation budget from the policy decision.
        pub degradation_budget: f64,
    }

    /// Encode only profiles backed by a concrete runtime encoder.
    ///
    /// `scr-runtime-compression` currently provides strict decode adapters but
    /// no registered TurboQuant/FibQuant encoder. Returning an error for those
    /// profiles preserves artifact truth: callers cannot mistake raw f32 bytes
    /// for a compressed representation.
    pub(super) fn encode_selected_profile(
        embedding: &[f32],
        profile: &CodecProfile,
    ) -> Result<Vec<u8>, String> {
        match profile {
            CodecProfile::Raw => Ok(bytemuck::cast_slice(embedding).to_vec()),
            CodecProfile::Q8 => Err("governed q8 encoding is not implemented".to_string()),
            CodecProfile::Q4 => Err("governed q4 encoding is not implemented".to_string()),
            CodecProfile::Turbo => Err(
                "governed turbo-quant encoding is unavailable: no runtime encoder is registered"
                    .to_string(),
            ),
            CodecProfile::Fib => Err(
                "governed fib-quant encoding is unavailable: no runtime encoder is registered"
                    .to_string(),
            ),
        }
    }

    /// Encode an embedding vector through the governed compression pipeline.
    ///
    /// The caller receives bytes only when their representation is truthful.
    /// Unsupported compression decisions fail explicitly instead of silently
    /// substituting the uncompressed representation.
    pub fn encode_governed(
        embedding: &[f32],
        policy: &GovernancePolicy,
    ) -> Result<GovernedEncodeResult, String> {
        let request = GovernanceRequest {
            content_type: ContentType::Structured,
            size_bytes: (embedding.len() * std::mem::size_of::<f32>()) as u64,
            accuracy_requirement: 0.99,
            latency_tolerance_ms: 500,
            admissibility: AdmissibilityClass::Standard,
        };

        let decision = policy.evaluate(request).map_err(|e| e.to_string())?;
        let encoded = encode_selected_profile(embedding, &decision.codec)?;

        Ok(GovernedEncodeResult {
            compressed_bytes: encoded,
            codec_profile: decision.codec,
            governance_receipt_id: format!("gr-{}", uuid::Uuid::new_v4()),
            degradation_budget: decision.degradation_budget,
        })
    }

    /// Encode with governance using the default policy.
    pub fn encode_governed_default(embedding: &[f32]) -> Result<GovernedEncodeResult, String> {
        encode_governed(embedding, &GovernancePolicy::default())
    }
}

#[cfg(not(feature = "turbo-quant-codec"))]
pub mod governed {
    //! Stub module when `turbo-quant-codec` is not enabled.

    /// Stub result type — codec_profile is a string when feature is disabled.
    #[derive(Debug, Clone)]
    pub struct GovernedEncodeResult {
        pub compressed_bytes: Vec<u8>,
        pub codec_profile: String,
        pub governance_receipt_id: String,
        pub degradation_budget: f64,
    }

    /// Returns an error indicating the turbo-quant-codec feature is not enabled.
    pub fn encode_governed(
        _embedding: &[f32],
        _policy: (),
    ) -> Result<GovernedEncodeResult, String> {
        Err("turbo-quant-codec feature is not enabled".to_string())
    }

    /// Returns an error indicating the turbo-quant-codec feature is not enabled.
    pub fn encode_governed_default(_embedding: &[f32]) -> Result<GovernedEncodeResult, String> {
        Err("turbo-quant-codec feature is not enabled".to_string())
    }
}

pub use governed::{encode_governed, encode_governed_default, GovernedEncodeResult};

#[cfg(all(test, feature = "turbo-quant-codec"))]
mod tests {
    use super::governed::{encode_governed_default, encode_selected_profile};
    use quant_governor::CodecProfile;

    #[test]
    fn raw_governed_encoding_preserves_f32_wire_bytes() {
        let embedding = vec![1.25_f32, -2.5, 0.0, 4.75];
        let result = encode_governed_default(&embedding).expect("default policy selects raw");
        assert_eq!(result.codec_profile, CodecProfile::Raw);
        assert_eq!(
            result.compressed_bytes,
            bytemuck::cast_slice::<f32, u8>(&embedding)
        );
        assert!(result.governance_receipt_id.starts_with("gr-"));
    }

    #[test]
    fn unavailable_compression_profiles_fail_closed() {
        let embedding = vec![0.25_f32; 16];
        for profile in [
            CodecProfile::Q8,
            CodecProfile::Q4,
            CodecProfile::Turbo,
            CodecProfile::Fib,
        ] {
            let error = encode_selected_profile(&embedding, &profile)
                .expect_err("unsupported profile must not be relabeled as raw bytes");
            assert!(error.contains("encoding"));
        }
    }
}
