# Static grep evidence
## current run
# Current Codex Run

Current run: `P30`
Updated UTC: `2026-05-13T02:36:02Z`

Historical run material in `docs/codex-runs/archive/` is evidence, not active instruction.
## zero accuracy
turbo-quant/README.md:5:Compress high-dimensional vectors (embeddings, KV cache entries) to 3-8 bits per value with **zero accuracy loss** and **no dataset-specific calibration**. Based on the algorithms from Google Research published at ICLR 2026, AISTATS 2026, and AAAI 2025.
semantic-memory/docs/TURBOQUANT_READINESS.md:9:- TurboQuant is not the default retrieval backend.
semantic-memory/docs/audits/codex-giga-pass-20260511.md:14:Address the first gated slices of the Giga-Pass prompt: Layer 0 ledger setup, Layer 1 HNSW/vector-index truth hardening, Layer 2 deterministic search context plus receipt scaffolding, a focused Layer 3 semantic-boundary defaulting cleanup, Layer 4 codec abstraction, Layer 5 TurboQuant optional-backend prototype, Layer 6 practical receipt/explanation APIs, Layer 7 public framing, and the continuation pass for durable replay-addressable search receipts plus replay verification. TurboQuant remains derived-only and is not default retrieval.
semantic-memory/docs/audits/codex-giga-pass-20260511.md:149:Context-aware search APIs can return `VectorSearchReceiptV1` when `receipt_mode` is `ExplainOnly` or `ReturnReceipt`. Receipts include evaluation time, query embedding digest, search profile, candidate backend, optional codec family/profile digest, approximate status, requested/returned/post-filter candidate counts, fallback, exact rerank, result IDs, and degradations. `VectorSearchReceiptV1::answers()` turns those fields into product-facing replay/source/approximation/rebuild answers. `ExplainedResult::answer()` answers why an individual result appeared and which authoritative source row it came from. When a receipt is produced, it is now persisted in `search_receipts` as versioned metadata and can be loaded by `MemoryStore::get_search_receipt(receipt_id)`. `MemoryStore::replay_search_receipt()` reruns the recorded hybrid/vector-only search family with the stored evaluation time, a fresh replay receipt ID, and caller-supplied query/filter inputs; it reports query embedding digest match, result-ID order match, missing IDs, and added IDs. The stored JSON uses fixed-width count conversion rather than persisting public `usize` fields. Filtered HNSW under-return records `hnsw_filtered_underreturn_fallback`. TurboQuant artifacts expose profile digests and encoded digests, but TurboQuant is not yet wired into live retrieval receipts.
semantic-memory/docs/audits/codex-giga-pass-20260511.md:160:Prototype implemented; not default retrieval eligible.
semantic-memory/docs/audits/codex-giga-pass-20260511.md:177:Run the next pass to wire optional TurboQuant artifacts into retrieval receipts behind `VectorCodec` with exact raw fallback and accepted drift thresholds. Add receipt retention/pruning policy before treating durable receipts as production storage. Do not make TurboQuant default retrieval.
semantic-memory/docs/codex-runs/P31_TURBOQUANT_READINESS_FINAL.md:5:`tq-live-feature-gated-exact-rerank-ready`
semantic-memory/docs/codex-runs/P31_TURBOQUANT_READINESS_FINAL.md:28:- Removed public "zero accuracy loss" wording in TurboQuant crate docs.
## fnv
## artifact digest
semantic-memory/src/db.rs:270:    codec_profile_digest    TEXT NOT NULL,
semantic-memory/src/db.rs:271:    source_embedding_digest TEXT NOT NULL,
semantic-memory/src/db.rs:273:    artifact_digest         TEXT NOT NULL,
semantic-memory/src/db.rs:279:    PRIMARY KEY (item_key, codec_family, codec_profile_digest)
semantic-memory/src/db.rs:283:ON derived_vector_artifacts(codec_family, codec_profile_digest, status);
semantic-memory/src/db.rs:286:ON derived_vector_artifacts(source_embedding_digest);
semantic-memory/src/db.rs:773:         SET encoded_digest = artifact_digest
semantic-memory/src/db.rs:797:         ON derived_vector_artifacts(codec_family, codec_profile_digest, status);
semantic-memory/src/db.rs:799:         ON derived_vector_artifacts(source_embedding_digest);",
semantic-memory/src/db.rs:822:    query_embedding_digest: Option<String>,
semantic-memory/src/db.rs:838:    codec_profile_digest: Option<String>,
semantic-memory/src/db.rs:840:    artifact_profile_digest: Option<String>,
semantic-memory/src/db.rs:885:    pub codec_profile_digest: String,
semantic-memory/src/db.rs:886:    pub source_embedding_digest: String,
semantic-memory/src/db.rs:896:pub(crate) fn source_embedding_digest(
semantic-memory/src/db.rs:918:             (item_key, codec_family, codec_profile_digest, source_embedding_digest,
semantic-memory/src/db.rs:919:              encoded_digest, artifact_digest, encoding, dim, encoded, created_at, status)
semantic-memory/src/db.rs:924:            row.codec_profile_digest,
semantic-memory/src/db.rs:925:            row.source_embedding_digest,
semantic-memory/src/db.rs:952:    codec_profile_digest: &str,
semantic-memory/src/db.rs:955:        "SELECT item_key, codec_family, codec_profile_digest, source_embedding_digest,
semantic-memory/src/db.rs:958:         WHERE codec_family = ?1 AND codec_profile_digest = ?2 AND status = 'active'",
semantic-memory/src/db.rs:960:    let rows = stmt.query_map(params![codec_family, codec_profile_digest], |row| {
semantic-memory/src/db.rs:965:            codec_profile_digest: row.get(2)?,
semantic-memory/src/db.rs:966:            source_embedding_digest: row.get(3)?,
semantic-memory/src/db.rs:991:    codec_profile_digest: &str,
semantic-memory/src/db.rs:995:         WHERE codec_family = ?1 AND codec_profile_digest = ?2 AND status = 'active'",
semantic-memory/src/db.rs:996:        params![codec_family, codec_profile_digest],
semantic-memory/src/db.rs:1015:    let codec_profile_digest = codec.profile().digest();
semantic-memory/src/db.rs:1059:            codec_profile_digest: codec_profile_digest.clone(),
semantic-memory/src/db.rs:1060:            source_embedding_digest: source_embedding_digest(&blob, dim)?,
semantic-memory/src/db.rs:1061:            encoded_digest: artifact.artifact_digest,
semantic-memory/src/db.rs:1073:             WHERE codec_family = ?1 AND codec_profile_digest = ?2",
semantic-memory/src/db.rs:1074:            params!["turbo_quant", &codec_profile_digest],
semantic-memory/src/db.rs:1086:        codec_profile_digest,
semantic-memory/src/db.rs:1128:        query_embedding_digest: receipt.query_embedding_digest.clone(),
semantic-memory/src/db.rs:1138:        codec_profile_digest: receipt.codec_profile_digest.clone(),
semantic-memory/src/db.rs:1139:        artifact_profile_digest: receipt.artifact_profile_digest.clone(),
semantic-memory/src/db.rs:1217:        query_embedding_digest: stored.query_embedding_digest,
semantic-memory/src/db.rs:1227:        codec_profile_digest: stored.codec_profile_digest,
semantic-memory/src/db.rs:1228:        artifact_profile_digest: stored.artifact_profile_digest,
semantic-memory/src/lib.rs:1496:        let query_embedding_digest_matches =
semantic-memory/src/lib.rs:1497:            original_receipt.query_embedding_digest == replay_receipt.query_embedding_digest;
semantic-memory/src/lib.rs:1517:            query_embedding_digest_matches,
semantic-memory/src/search.rs:902:    let profile_digest = profile.digest();
semantic-memory/src/search.rs:905:        codec_profile_digest: Some(profile_digest.clone()),
semantic-memory/src/search.rs:934:        crate::db::count_derived_vector_artifacts(conn, "turbo_quant", &profile_digest)?;
semantic-memory/src/search.rs:936:        crate::db::load_derived_vector_artifacts_by_profile(conn, "turbo_quant", &profile_digest)?;
semantic-memory/src/search.rs:1004:        let source_digest = crate::db::source_embedding_digest(&raw_row.blob, dim)?;
semantic-memory/src/search.rs:1005:        if source_digest != artifact_row.source_embedding_digest {
semantic-memory/src/search.rs:1010:        if artifact.profile_digest != artifact_row.codec_profile_digest
semantic-memory/src/search.rs:1011:            || artifact.artifact_digest != artifact_row.encoded_digest
semantic-memory/src/search.rs:1250:    codec_profile_digest: Option<String>,
semantic-memory/src/search.rs:1394:pub(crate) fn query_embedding_digest(query_embedding: &[f32]) -> String {
semantic-memory/src/search.rs:1414:                row.source_embedding_digest.as_str(),
semantic-memory/src/search.rs:1425:    for (item_key, source_embedding_digest, encoded_digest) in entries {
semantic-memory/src/search.rs:1429:            .update_str(source_embedding_digest)
semantic-memory/src/search.rs:1497:        query_embedding_digest: Some(query_embedding_digest(query_embedding)),
semantic-memory/src/search.rs:1507:        codec_profile_digest: metadata.codec_profile_digest.clone(),
semantic-memory/src/search.rs:1508:        artifact_profile_digest: metadata.codec_profile_digest.clone(),
semantic-memory/src/search.rs:2453:    use super::query_embedding_digest;
semantic-memory/src/search.rs:2456:    fn query_embedding_digest_includes_dimension_and_bytes() {
semantic-memory/src/search.rs:2457:        let two_dims = query_embedding_digest(&[1.0, 2.0]);
semantic-memory/src/search.rs:2458:        let three_dims = query_embedding_digest(&[1.0, 2.0, 0.0]);
semantic-memory/src/search.rs:2459:        let changed_byte = query_embedding_digest(&[1.0, 2.000_001]);
semantic-memory/src/search.rs:2465:        assert_eq!(two_dims, query_embedding_digest(&[1.0, 2.0]));
semantic-memory/src/types.rs:256:    pub query_embedding_digest: Option<String>,
semantic-memory/src/types.rs:282:    pub codec_profile_digest: Option<String>,
semantic-memory/src/types.rs:285:    pub artifact_profile_digest: Option<String>,
semantic-memory/src/types.rs:342:    pub codec_profile_digest: String,
semantic-memory/src/types.rs:377:    pub codec_profile_digest: Option<String>,
semantic-memory/src/types.rs:424:        if let Some(codec_profile_digest) = &self.codec_profile_digest {
semantic-memory/src/types.rs:427:                codec_profile_digest
semantic-memory/src/types.rs:432:        if let Some(query_embedding_digest) = &self.query_embedding_digest {
semantic-memory/src/types.rs:435:                query_embedding_digest
semantic-memory/src/types.rs:446:            codec_profile_digest: self.codec_profile_digest.clone(),
semantic-memory/src/types.rs:452:            replay_ready: self.query_embedding_digest.is_some(),
semantic-memory/src/types.rs:453:            rebuild_ready: self.query_embedding_digest.is_some()
semantic-memory/src/types.rs:501:    pub query_embedding_digest_matches: bool,
semantic-memory/src/vector_codec.rs:127:    pub profile_digest: String,
semantic-memory/src/vector_codec.rs:130:    pub artifact_digest: String,
semantic-memory/src/vector_codec.rs:138:        let profile_digest = profile.digest();
semantic-memory/src/vector_codec.rs:139:        let artifact_digest = b3_digest(&encoded);
semantic-memory/src/vector_codec.rs:143:            profile_digest,
semantic-memory/src/vector_codec.rs:144:            artifact_digest,
semantic-memory/src/vector_codec.rs:171:    let artifact_profile_digest = artifact.profile.digest();
semantic-memory/src/vector_codec.rs:172:    if artifact.profile_digest != artifact_profile_digest {
semantic-memory/src/vector_codec.rs:174:            expected_digest: artifact_profile_digest,
semantic-memory/src/vector_codec.rs:175:            actual_digest: artifact.profile_digest.clone(),
semantic-memory/src/vector_codec.rs:180:    if artifact.profile_digest != expected_digest {
semantic-memory/src/vector_codec.rs:183:            actual_digest: artifact.profile_digest.clone(),
semantic-memory/src/vector_codec.rs:188:    if !artifact.artifact_digest.is_empty() && artifact.artifact_digest != encoded_digest {
semantic-memory/src/vector_codec.rs:191:            row_id: artifact.profile_digest.clone(),
semantic-memory/src/vector_codec.rs:194:                artifact.artifact_digest, encoded_digest
semantic-memory/tests/db_tests.rs:100:        "codec_profile_digest",
semantic-memory/tests/db_tests.rs:101:        "source_embedding_digest",
semantic-memory/tests/search_tests.rs:980:                receipt.codec_profile_digest.as_deref(),
semantic-memory/tests/search_tests.rs:981:                Some(build.codec_profile_digest.as_str())
semantic-memory/tests/search_tests.rs:1089:            assert_eq!(first.codec_profile_digest, second.codec_profile_digest);
semantic-memory/tests/search_tests.rs:1274:    assert!(report.query_embedding_digest_matches);
semantic-memory/tests/search_tests.rs:1338:    assert!(!report.query_embedding_digest_matches);
semantic-memory/tests/vector_codec.rs:31:fn raw_profile_digest_is_stable_and_identity_sensitive() -> Result<(), MemoryError> {
semantic-memory/tests/vector_codec.rs:44:fn profile_digest_uses_blake3_and_changes_on_profile_fields() -> Result<(), MemoryError> {
semantic-memory/tests/vector_codec.rs:63:    assert_eq!(artifact.profile_digest, codec.profile().digest());
semantic-memory/tests/vector_codec.rs:64:    assert_eq!(artifact.artifact_digest, artifact.encoded_digest());
semantic-memory/tests/vector_codec.rs:65:    assert!(artifact.artifact_digest.starts_with("blake3:"));
semantic-memory/tests/vector_codec.rs:71:fn vector_artifact_digest_tampering_fails_closed() -> Result<(), MemoryError> {
semantic-memory/tests/vector_codec.rs:115:fn artifact_profile_digest_tampering_fails_closed() -> Result<(), MemoryError> {
semantic-memory/tests/vector_codec.rs:138:    assert_eq!(artifact_a.profile_digest, artifact_b.profile_digest);
semantic-memory/tests/vector_codec.rs:140:    assert_eq!(artifact_a.artifact_digest, artifact_a.encoded_digest());
semantic-memory/tests/vector_codec.rs:155:    assert_ne!(artifact_a.profile_digest, artifact_b.profile_digest);
semantic-memory/tests/vector_codec.rs:162:fn profile_digest_uses_blake3_and_changes_on_seed() -> Result<(), MemoryError> {
## wire seed
turbo-quant/src/wire.rs:53:        bytes.extend_from_slice(&profile.seed().to_le_bytes());
turbo-quant/src/wire.rs:132:        let _seed = cursor.read_u64()?;
turbo-quant/src/wire.rs:135:        let payload_len = cursor.read_u64()? as usize;
turbo-quant/src/wire.rs:282:    fn read_u64(&mut self) -> Result<u64> {
