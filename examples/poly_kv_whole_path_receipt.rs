#[cfg(not(feature = "fib-quant-codec"))]
fn main() {
    eprintln!("requires --features fib-quant-codec");
    std::process::exit(2);
}

#[cfg(feature = "fib-quant-codec")]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct FreshProcessProbe {
    process_id: u32,
    artifact_generation_id: String,
    candidate_backend: String,
    codec_profile_digest: String,
    approximate_candidate_count: usize,
    exact_rerank: bool,
    final_scores_approximate: bool,
    fallback_reason: Option<String>,
    receipt_id: String,
    receipt_readback: bool,
    result_count: usize,
}

#[cfg(feature = "fib-quant-codec")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use semantic_memory::whole_path_receipt::*;
    use semantic_memory::{
        DerivedVectorBackendPolicy, MemoryConfig, MemoryStore, MockEmbedder, ReceiptMode,
        SearchContext, SearchSource, SearchSourceType, StoragePaths,
    };
    use sha2::Digest;
    use std::{collections::HashMap, process::Command, time::Instant};

    const D: usize = 384;
    const N: usize = 1_000;
    const K: usize = 10;
    const C: usize = 50;
    const ITERS: usize = 50;
    const SEED: u64 = 0x5eed_2026;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(a, b)| a * b).sum()
    }

    fn top_indices(
        rows: &[(usize, String, String, Vec<f32>)],
        query: &[f32],
        candidates: impl IntoIterator<Item = usize>,
        k: usize,
    ) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|index| (index, dot(query, &rows[index].3)))
            .collect();
        scored.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| rows[*left_index].1.cmp(&rows[*right_index].1))
        });
        scored.truncate(k);
        scored.into_iter().map(|(index, _)| index).collect()
    }

    fn decode_f32_blob(blob: &[u8]) -> Result<Vec<f32>, String> {
        if blob.len() % 4 != 0 {
            return Err(format!(
                "embedding blob length {} is not divisible by 4",
                blob.len()
            ));
        }
        Ok(blob
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect())
    }

    fn file_len(path: &std::path::Path) -> u64 {
        std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }

    fn command_text(program: &str, args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
        Ok(
            String::from_utf8(Command::new(program).args(args).output()?.stdout)?
                .trim()
                .to_string(),
        )
    }

    fn benchmark_config(base_dir: &std::path::Path) -> MemoryConfig {
        let mut config = MemoryConfig {
            base_dir: base_dir.to_path_buf(),
            ..MemoryConfig::default()
        };
        config.embedding.dimensions = D;
        config.embedding.model = "mock-whole-path-v2".to_string();
        config.search.derived_vector_backend = DerivedVectorBackendPolicy::FibQuantCandidateOnly;
        config.search.default_top_k = K;
        config.search.candidate_pool_size = C;
        config.search.fib_quant_block_size = 4;
        config.search.fib_quant_codebook_size = 32;
        config.search.fib_quant_seed = 42;
        config.search.fib_quant_max_value_mse = 0.2;
        config.search.fib_quant_candidate_oversample = 5;
        config.search.min_similarity = -1.0;
        config
    }

    if let Some(base_dir) = std::env::var_os("SM_WHOLE_PATH_PROBE_BASE_DIR") {
        let expected_generation = std::env::var("SM_WHOLE_PATH_PROBE_GENERATION")?;
        let config = benchmark_config(std::path::Path::new(&base_dir));
        let store = MemoryStore::open_with_embedder(config, Box::new(MockEmbedder::new(D)))?;
        let mut context = SearchContext::default_now();
        context.receipt_mode = ReceiptMode::ReturnReceipt;
        context.request_id = Some("whole-path-fresh-process-probe".to_string());
        let response = store
            .search_vector_only_with_context(
                &format!("whole-path deterministic item {SEED:016x} 0000"),
                Some(K),
                None,
                Some(&[SearchSourceType::Facts]),
                context,
            )
            .await?;
        let receipt = response
            .receipt
            .ok_or("fresh-process probe receipt missing")?;
        let receipt_readback = store
            .get_search_receipt(&receipt.receipt_id)
            .await?
            .is_some();
        let probe = FreshProcessProbe {
            process_id: std::process::id(),
            artifact_generation_id: receipt
                .artifact_generation_id
                .ok_or("fresh-process generation identity missing")?,
            candidate_backend: receipt.candidate_backend,
            codec_profile_digest: receipt
                .codec_profile_digest
                .ok_or("fresh-process codec profile missing")?,
            approximate_candidate_count: receipt.approximate_candidate_count.unwrap_or(0),
            exact_rerank: receipt.exact_rerank,
            final_scores_approximate: receipt.approximate,
            fallback_reason: receipt.fallback_reason,
            receipt_id: receipt.receipt_id,
            receipt_readback,
            result_count: response.results.len(),
        };
        if probe.artifact_generation_id != expected_generation
            || probe.candidate_backend != "poly_kv_fibquant_persisted_generation"
            || probe.approximate_candidate_count != C
            || !probe.exact_rerank
            || probe.final_scores_approximate
            || probe.fallback_reason.is_some()
            || !probe.receipt_readback
            || probe.result_count != K
        {
            return Err(format!("inadmissible fresh-process probe: {probe:?}").into());
        }
        println!("{}", serde_json::to_string(&probe)?);
        return Ok(());
    }

    let temp_dir = tempfile::tempdir()?;
    let config = benchmark_config(temp_dir.path());

    let store = MemoryStore::open_with_embedder(config.clone(), Box::new(MockEmbedder::new(D)))?;
    let mut inserted = Vec::with_capacity(N);
    for index in 0..N {
        let text = format!("whole-path deterministic item {SEED:016x} {index:04}");
        let embedding = store.embed_document(&text).await?;
        let fact_id = store
            .add_fact_with_embedding(
                "whole-path-benchmark",
                &text,
                &embedding,
                Some("local-cpu-receipt"),
                None,
            )
            .await?;
        inserted.push((fact_id, text, embedding));
    }

    let build_started = Instant::now();
    let generation_receipt = store.rebuild_fibquant_pool_generation().await?;
    let build_ns = build_started.elapsed().as_nanos();
    let generation_id = generation_receipt.generation_id.clone();
    drop(store);

    let probe_output = Command::new(std::env::current_exe()?)
        .env("SM_WHOLE_PATH_PROBE_BASE_DIR", temp_dir.path())
        .env("SM_WHOLE_PATH_PROBE_GENERATION", &generation_id)
        .output()?;
    if !probe_output.status.success() {
        return Err(format!(
            "fresh-process probe failed with {:?}: {}",
            probe_output.status.code(),
            String::from_utf8_lossy(&probe_output.stderr)
        )
        .into());
    }
    let fresh_process_probe: FreshProcessProbe = serde_json::from_slice(&probe_output.stdout)?;
    if fresh_process_probe.process_id == std::process::id()
        || fresh_process_probe.artifact_generation_id != generation_id
        || fresh_process_probe.codec_profile_digest != generation_receipt.codec_profile
    {
        return Err(
            format!("fresh-process probe identity mismatch: {fresh_process_probe:?}").into(),
        );
    }
    let fresh_process_probe_digest =
        format!("blake3:{}", blake3::hash(&probe_output.stdout).to_hex());

    // Fresh process reload above proves durable admission independently. This second reload
    let store = MemoryStore::open_with_embedder(config.clone(), Box::new(MockEmbedder::new(D)))?;
    let paths = StoragePaths::new(temp_dir.path());
    let conn = rusqlite::Connection::open(&paths.sqlite_path)?;
    let payload: Vec<u8> = conn.query_row(
        "SELECT payload FROM provekv_pool_generations WHERE generation_id = ?1 AND status = 'ready'",
        [&generation_id],
        |row| row.get(0),
    )?;

    let mut statement = conn.prepare(
        "SELECT m.pool_index, m.item_id, m.embedding_digest, f.embedding
         FROM provekv_pool_item_map m
         JOIN facts f ON m.item_id = ('fact:' || f.id)
         WHERE m.generation_id = ?1
         ORDER BY m.pool_index ASC",
    )?;
    let mapped = statement.query_map([&generation_id], |row| {
        Ok((
            row.get::<_, usize>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Vec<u8>>(3)?,
        ))
    })?;
    let mut rows = Vec::with_capacity(N);
    for row in mapped {
        let (pool_index, item_id, embedding_digest, blob) = row?;
        if pool_index != rows.len() {
            return Err(format!("non-contiguous pool index {pool_index}").into());
        }
        rows.push((
            pool_index,
            item_id,
            embedding_digest,
            decode_f32_blob(&blob)?,
        ));
    }
    if rows.len() != N {
        return Err(format!("expected {N} item-map rows, found {}", rows.len()).into());
    }
    drop(statement);
    drop(conn);

    let item_map_canonical_bytes = serde_json::to_vec(
        &rows
            .iter()
            .map(|(index, item_id, digest, _)| (index, item_id, digest))
            .collect::<Vec<_>>(),
    )?
    .len() as u64;
    let generation_row_canonical_bytes = serde_json::to_vec(&generation_receipt)?.len() as u64;

    let prepare_started = Instant::now();
    let pool =
        poly_kv::decode_fibquant_pool_bundle(&payload, config.search.fib_quant_max_value_mse)?;
    let prepared = pool.prepare_compressed_index(0, 0)?;
    let prepare_ns = prepare_started.elapsed().as_nanos();
    if prepared.fib_profile_digest.as_str() != generation_receipt.codec_profile.as_str() {
        return Err("reloaded owner profile does not match semantic generation".into());
    }

    let index_by_fact_id: HashMap<String, usize> = rows
        .iter()
        .filter_map(|(index, item_id, _, _)| {
            item_id
                .strip_prefix("fact:")
                .map(|fact_id| (fact_id.to_string(), *index))
        })
        .collect();
    if index_by_fact_id.len() != rows.len() {
        return Err("non-fact item present in facts-only generation".into());
    }

    // One unreported warmup exercises bundle admission, prepared scoring, exact rerank,
    // result materialization, and SQLite receipt persistence before measurement.
    let warmup_query = &inserted[0].1;
    let mut warmup_context = SearchContext::default_now();
    warmup_context.receipt_mode = ReceiptMode::ReturnReceipt;
    let _ = store
        .search_vector_only_with_context(
            warmup_query,
            Some(K),
            None,
            Some(&[SearchSourceType::Facts]),
            warmup_context,
        )
        .await?;
    let _ = pool.attention_topk_compressed_prepared(&prepared, &inserted[0].2, C)?;

    let mut exact_baseline_ns = Vec::with_capacity(ITERS);
    let mut scoring_ns = Vec::with_capacity(ITERS);
    let mut rerank_ns = Vec::with_capacity(ITERS);
    let mut whole_path_ns = Vec::with_capacity(ITERS);
    let mut durable_receipt_ids = Vec::with_capacity(ITERS);
    let mut candidate_hits = 0_u64;
    let mut rerank_hits = 0_u64;
    let mut semantic_ordered_parity = true;
    let mut observed_backend = None;

    for iteration in 0..ITERS {
        let inserted_index = (iteration * 17) % N;
        let query_text = &inserted[inserted_index].1;
        let query = store.embed_query(query_text).await?;
        let exact_started = Instant::now();
        let exact = top_indices(&rows, &query, 0..N, K);
        exact_baseline_ns.push(exact_started.elapsed().as_nanos());

        let scoring_started = Instant::now();
        let selection = pool.attention_topk_compressed_prepared(&prepared, &query, C)?;
        scoring_ns.push(scoring_started.elapsed().as_nanos());
        candidate_hits += selection
            .hits
            .iter()
            .filter(|hit| exact.contains(&hit.token_index))
            .count() as u64;

        let rerank_started = Instant::now();
        let reranked = top_indices(
            &rows,
            &query,
            selection.hits.iter().map(|hit| hit.token_index),
            K,
        );
        rerank_ns.push(rerank_started.elapsed().as_nanos());
        rerank_hits += reranked
            .iter()
            .filter(|index| exact.contains(index))
            .count() as u64;
        let mut context = SearchContext::default_now();
        context.receipt_mode = ReceiptMode::ReturnReceipt;
        context.request_id = Some(format!("whole-path-v2-{iteration:04}"));
        let whole_path_started = Instant::now();
        let response = store
            .search_vector_only_with_context(
                query_text,
                Some(K),
                None,
                Some(&[SearchSourceType::Facts]),
                context,
            )
            .await?;
        whole_path_ns.push(whole_path_started.elapsed().as_nanos());
        let receipt = response.receipt.ok_or("missing semantic search receipt")?;
        if receipt.artifact_generation_id.as_deref() != Some(generation_id.as_str())
            || receipt.codec_profile_digest.as_deref()
                != Some(generation_receipt.codec_profile.as_str())
            || receipt.approximate
            || !receipt.exact_rerank
            || receipt.approximate_candidate_count != Some(C)
            || receipt.fallback_reason.is_some()
        {
            return Err(format!("inadmissible semantic receipt: {receipt:?}").into());
        }
        observed_backend.get_or_insert_with(|| receipt.candidate_backend.clone());
        if observed_backend.as_deref() != Some(receipt.candidate_backend.as_str()) {
            return Err("candidate backend changed during benchmark".into());
        }
        durable_receipt_ids.push(receipt.receipt_id.clone());
        let returned: Vec<String> = response
            .results
            .iter()
            .map(|result| match &result.source {
                SearchSource::Fact { fact_id, .. } => Ok(fact_id.clone()),
                other => Err(format!("unexpected search source: {other:?}")),
            })
            .collect::<Result<_, _>>()?;
        let expected: Vec<String> = reranked
            .iter()
            .map(|index| {
                rows[*index]
                    .1
                    .strip_prefix("fact:")
                    .map(str::to_string)
                    .ok_or_else(|| "non-fact item present in facts-only generation".to_string())
            })
            .collect::<Result<_, _>>()?;
        semantic_ordered_parity &= returned == expected;
        for fact_id in returned {
            if !index_by_fact_id.contains_key(&fact_id) {
                return Err(format!("semantic result {fact_id} absent from item map").into());
            }
        }
    }

    let backend = observed_backend.ok_or("no observed semantic backend")?;
    if backend != "poly_kv_fibquant_persisted_generation" {
        return Err(format!("unexpected semantic candidate backend {backend}").into());
    }
    for receipt_id in &durable_receipt_ids {
        if store.get_search_receipt(receipt_id).await?.is_none() {
            return Err(format!("durable search receipt {receipt_id} was not readable").into());
        }
    }

    let status = command_text("git", &["status", "--short"])?;
    let head = command_text("git", &["rev-parse", "HEAD"])?;
    let status_digest = format!("{:x}", sha2::Sha256::digest(status.as_bytes()));
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("model name\t: ").map(str::to_string))
        })
        .unwrap_or_else(|| command_text("uname", &["-m"]).unwrap_or_else(|_| "unknown".into()));
    let sqlite_main_bytes = file_len(&paths.sqlite_path);
    let sqlite_wal_bytes = file_len(&paths.sqlite_path.with_extension("db-wal"));
    let sqlite_shm_bytes = file_len(&paths.sqlite_path.with_extension("db-shm"));
    let owner_build = pool.build_receipt();
    let owner_receipt_bytes = serde_json::to_vec(owner_build)?.len() as u64;
    let semantic_index_bytes = item_map_canonical_bytes + generation_row_canonical_bytes;

    let receipt = WholePathReceiptV1 {
        schema_version: 2,
        workload: Workload {
            dimensions: D,
            corpus_size: N,
            top_k: K,
            candidate_k: C,
            seed: SEED,
            iterations: ITERS,
        },
        build: BuildMetadata {
            profile: if cfg!(debug_assertions) { "debug" } else { "release" }.into(),
            cpu,
            kernel: command_text("uname", &["-sr"])? ,
            os: std::env::consts::OS.into(),
            rustc: command_text("rustc", &["--version"])? ,
            cargo: command_text("cargo", &["--version"])? ,
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            source_head: head,
            source_status_digest: status_digest,
        },
        bytes: ByteAccounting {
            authoritative_raw_f32_bytes: (N * D * std::mem::size_of::<f32>()) as u64,
            compressed_payload_bytes: owner_build.encoded_bytes,
            manifest_bytes: pool.manifest().canonical_serialized_len(),
            receipt_bytes: owner_receipt_bytes,
            index_bytes: semantic_index_bytes,
            codebook_bytes: 0,
            fallback_bytes: owner_build.exact_fallback_bytes,
            reader_scratch_bytes: Some(pool.memory_accounting().per_reader_scratch_bytes),
            total_resident_derived_bytes: pool
                .memory_accounting()
                .total_bytes()
                .saturating_add(semantic_index_bytes)
                .saturating_add(owner_receipt_bytes),
        },
        quality: QualityMetrics {
            candidate_recall_at_k: candidate_hits as f64 / (ITERS * K) as f64,
            candidate_precision_at_candidate_k: candidate_hits as f64 / (ITERS * C) as f64,
            exact_rerank_ordered_parity: semantic_ordered_parity,
            exact_rerank_tie_policy: "score_desc_item_id_asc".into(),
            exact_rerank_overlap_at_k: rerank_hits as f64 / (ITERS * K) as f64,
        },
        latency: LatencyMetrics {
            exact_baseline_ns,
            scoring_ns,
            prepare_ns: vec![prepare_ns],
            rerank_ns,
            whole_path_ns,
        },
        decoding: DecodeMetrics {
            modeled_selected_values: (ITERS * C) as u64,
            modeled_full_decode_values: (ITERS * N) as u64,
            observed_decode_calls: None,
            observed_decoded_values: None,
        },
        semantic_path: Some(SemanticPathEvidence {
            artifact_generation_id: generation_id,
            source_snapshot_digest: generation_receipt.embedding_snapshot_digest.clone(),
            source_digest: generation_receipt.source_digest.clone(),
            pool_manifest_digest: generation_receipt.pool_manifest_digest.clone(),
            codec_profile_digest: generation_receipt.codec_profile.clone(),
            artifact_digest: format!("blake3:{}", blake3::hash(&payload).to_hex()),
            candidate_backend: backend,
            fresh_process_reload: true,
            fresh_process_probe_digest,
            exact_f32_rerank: true,
            build_ns,
            persisted_bundle_bytes: generation_receipt.payload_bytes,
            generation_row_canonical_bytes,
            item_map_canonical_bytes,
            sqlite_main_bytes,
            sqlite_wal_bytes,
            sqlite_shm_bytes,
            durable_search_receipt_ids: durable_receipt_ids,
        }),
        fallback_disposition: "benchmark fallback not triggered; hostile missing/corrupt/stale/profile/item-map tests passed separately".into(),
        evidence_limitations: vec![
            "locally generated deterministic MockEmbedder corpus; not a production workload or external benchmark".into(),
            format!("one unreported warmup preceded {ITERS} measured iterations; no allocator, energy, or CPU-affinity controls"),
            "candidate scoring and rerank component timings come from the authenticated owner bundle read back from SQLite; whole_path_ns comes from the public semantic-memory API".into(),
            "decode counts are modeled from selected/full cardinalities because the owner codec does not expose decode counters".into(),
            "codebook bytes are included inside encoded payload bytes and are not separately exposed by the owner API".into(),
            "item-map and generation-row bytes are canonical JSON content lengths; SQLite main/WAL/SHM file sizes separately capture allocated physical storage including unrelated page/index overhead".into(),
            "no competitor comparison, GPU claim, throughput SLA, or production-readiness claim is supported by this receipt".into(),
        ],
    };
    receipt
        .validate()
        .map_err(|error| format!("receipt validation failed: {error}"))?;

    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/whole-path-receipt-current.json".into());
    let output_path = std::path::Path::new(&output);
    std::fs::create_dir_all(
        output_path
            .parent()
            .unwrap_or(std::path::Path::new("target")),
    )?;
    std::fs::write(output_path, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", output_path.display());
    Ok(())
}
