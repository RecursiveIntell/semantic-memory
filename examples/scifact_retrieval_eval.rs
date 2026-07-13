//! Receipt-bearing official BEIR SciFact evaluation over production retrieval APIs.
//!
//! This runner consumes precomputed document and query embeddings, persists one
//! semantic-memory fact store, and emits independent raw/aggregate artifacts for
//! FTS-only, exact f32 vector-only, and baseline hybrid retrieval. It deliberately
//! contains no tuning loop and does not branch on held-out relevance outcomes.

use chrono::Utc;
use semantic_memory::{
    Embedder, EmbeddingConfig, ExactnessProfile, MemoryConfig, MemoryError, MemoryStore,
    ReceiptMode, ReplayMode, SearchContext, SearchResult, SearchSource, SearchSourceType,
    VectorSearchReceiptV1,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::future::Future;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

const ROW_SCHEMA: &str = "semantic-memory-scifact-query-v1";
const AGGREGATE_SCHEMA: &str = "semantic-memory-scifact-aggregate-v1";
const MAPPING_SCHEMA: &str = "semantic-memory-scifact-doc-map-v1";
const NAMESPACE: &str = "beir-scifact";
const TOP_K: usize = 10;
const DEFAULT_CALIBRATION_COUNT: usize = 100;
const SCHEMA_DESCRIPTOR: &str = "semantic-memory-scifact-eval-v1|split=lowest-sha256-query-id|metrics=ndcg10,recall1,5,10,mrr10,map10,success1,5,10|latency=mean,p50,p95,max|rows=jsonl";

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type SearchExecution = (
    Vec<SearchResult>,
    Option<Vec<semantic_memory::ExplainedResult>>,
    Option<VectorSearchReceiptV1>,
);

#[derive(Debug, Deserialize)]
struct CorpusFile {
    schema: String,
    corpus_id: String,
    source: Value,
    source_hashes: Value,
    payload_hashes: PayloadHashes,
    embedding: EmbeddingMetadata,
    truncation: Value,
    counts: CorpusCounts,
    documents: Vec<CorpusDocument>,
    queries: Vec<CorpusQuery>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PayloadHashes {
    corpus_sha256: String,
    query_sha256: String,
    qrels_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct EmbeddingMetadata {
    model: String,
    dimensions: usize,
    normalized: bool,
}

#[derive(Debug, Deserialize)]
struct CorpusCounts {
    documents: usize,
    test_queries: usize,
}

#[derive(Debug, Deserialize)]
struct CorpusDocument {
    doc_id: String,
    title: String,
    text: String,
    semantic_text: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct CorpusQuery {
    query_id: String,
    text: String,
    embedding: Vec<f32>,
    qrels: BTreeMap<String, i32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoreMapping {
    schema: String,
    corpus_file_sha256: String,
    corpus_payload_sha256: String,
    embedding_model: String,
    embedding_dimensions: usize,
    namespace: String,
    documents: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    FtsOnly,
    VectorOnly,
    Hybrid,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::FtsOnly => "fts_only",
            Self::VectorOnly => "vector_only",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    Calibration,
    Heldout,
    All,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::Heldout => "heldout",
            Self::All => "all",
        }
    }
}

#[derive(Debug)]
struct Args {
    corpus: PathBuf,
    output_dir: PathBuf,
    store_dir: PathBuf,
    modes: Vec<Mode>,
    split: Split,
    calibration_count: usize,
}

#[derive(Clone)]
struct FixtureEmbedder {
    model: String,
    dimensions: usize,
    queries: Arc<HashMap<String, Vec<f32>>>,
}

impl Embedder for FixtureEmbedder {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, MemoryError>> + Send + 'a>> {
        Box::pin(async move {
            self.queries.get(text).cloned().ok_or_else(|| {
                MemoryError::Other(
                    "SciFact evaluator refused to re-embed an unknown query text".to_string(),
                )
            })
        })
    }

    fn embed_batch<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, MemoryError>> + Send + 'a>> {
        Box::pin(async move {
            let mut output = Vec::with_capacity(texts.len());
            for text in texts {
                output.push(self.queries.get(&text).cloned().ok_or_else(|| {
                    MemoryError::Other(
                        "SciFact evaluator refused to re-embed an unknown query text".to_string(),
                    )
                })?);
            }
            Ok(output)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryMetrics {
    ndcg_at_10: f64,
    recall_at_1: f64,
    recall_at_5: f64,
    recall_at_10: f64,
    mrr_at_10: f64,
    map_at_10: f64,
    success_at_1: f64,
    success_at_5: f64,
    success_at_10: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryRow {
    schema: String,
    mode: String,
    split: String,
    query_id: String,
    query_sha256: String,
    qrels: BTreeMap<String, i32>,
    ranked_doc_ids: Vec<String>,
    results: Vec<Value>,
    latency_ms: f64,
    error: Option<String>,
    metrics: QueryMetrics,
    search_receipt: Option<VectorSearchReceiptV1>,
}

fn parse_args() -> Result<Args, BoxError> {
    let mut corpus = PathBuf::from("target/scifact-eval/scifact-all-minilm-corpus.json");
    let mut output_dir = PathBuf::from("target/scifact-eval/results");
    let mut store_dir = PathBuf::from("target/scifact-eval/store");
    let mut mode = "all".to_string();
    let mut split = Split::Calibration;
    let mut calibration_count = DEFAULT_CALIBRATION_COUNT;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, name: &str| {
            args.next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match argument.as_str() {
            "--corpus" => corpus = PathBuf::from(value(&mut args, "--corpus")?),
            "--output-dir" => output_dir = PathBuf::from(value(&mut args, "--output-dir")?),
            "--store-dir" => store_dir = PathBuf::from(value(&mut args, "--store-dir")?),
            "--mode" => mode = value(&mut args, "--mode")?,
            "--split" => {
                split = match value(&mut args, "--split")?.as_str() {
                    "calibration" => Split::Calibration,
                    "heldout" => Split::Heldout,
                    "all" => Split::All,
                    other => return Err(format!("unknown split '{other}'").into()),
                }
            }
            "--calibration-count" => {
                calibration_count = value(&mut args, "--calibration-count")?.parse()?;
            }
            "-h" | "--help" => {
                println!(
                    "scifact_retrieval_eval [--corpus PATH] [--output-dir DIR] \
                     [--store-dir DIR] [--mode fts_only|vector_only|hybrid|all] \
                     [--split calibration|heldout|all] [--calibration-count N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}'").into()),
        }
    }
    let modes = match mode.as_str() {
        "fts_only" => vec![Mode::FtsOnly],
        "vector_only" => vec![Mode::VectorOnly],
        "hybrid" => vec![Mode::Hybrid],
        "all" => vec![Mode::FtsOnly, Mode::VectorOnly, Mode::Hybrid],
        other => return Err(format!("unknown mode '{other}'").into()),
    };
    Ok(Args {
        corpus,
        output_dir,
        store_dir,
        modes,
        split,
        calibration_count,
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, BoxError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn validate_corpus(corpus: &CorpusFile) -> Result<(), BoxError> {
    if corpus.schema != "semantic-memory-scifact-corpus-v1" {
        return Err(format!("unsupported corpus schema '{}'", corpus.schema).into());
    }
    if corpus.documents.len() != corpus.counts.documents
        || corpus.queries.len() != corpus.counts.test_queries
        || corpus.documents.len() != 5_183
        || corpus.queries.len() != 300
    {
        return Err(
            "official SciFact corpus must contain exactly 5,183 docs and 300 test queries".into(),
        );
    }
    if corpus.embedding.dimensions == 0 || !corpus.embedding.normalized {
        return Err("corpus embeddings must be non-empty and normalized".into());
    }
    let mut doc_ids = HashSet::new();
    for document in &corpus.documents {
        if !doc_ids.insert(document.doc_id.as_str()) {
            return Err(format!("duplicate document ID '{}'", document.doc_id).into());
        }
        if document.embedding.len() != corpus.embedding.dimensions
            || document.embedding.iter().any(|value| !value.is_finite())
            || document.semantic_text.is_empty()
        {
            return Err(format!("invalid document '{}'", document.doc_id).into());
        }
        let _ = (&document.title, &document.text);
    }
    let mut query_ids = HashSet::new();
    let mut query_texts = HashSet::new();
    for query in &corpus.queries {
        if !query_ids.insert(query.query_id.as_str()) {
            return Err(format!("duplicate query ID '{}'", query.query_id).into());
        }
        if !query_texts.insert(query.text.as_str()) {
            return Err(format!(
                "duplicate query text prevents exact fixture embedding lookup: '{}'",
                query.query_id
            )
            .into());
        }
        if query.embedding.len() != corpus.embedding.dimensions
            || query.embedding.iter().any(|value| !value.is_finite())
            || query.qrels.values().all(|grade| *grade <= 0)
        {
            return Err(format!("invalid query '{}'", query.query_id).into());
        }
        if let Some(missing) = query
            .qrels
            .keys()
            .find(|doc_id| !doc_ids.contains(doc_id.as_str()))
        {
            return Err(format!(
                "query '{}' qrels reference missing doc '{missing}'",
                query.query_id
            )
            .into());
        }
    }
    Ok(())
}

fn split_membership(query_ids: &[String], calibration_count: usize) -> BTreeSet<String> {
    let mut ordered = query_ids
        .iter()
        .map(|query_id| (sha256_bytes(query_id.as_bytes()), query_id.clone()))
        .collect::<Vec<_>>();
    ordered.sort();
    ordered
        .into_iter()
        .take(calibration_count.min(query_ids.len()))
        .map(|(_, query_id)| query_id)
        .collect()
}

fn selected_queries(
    queries: &[CorpusQuery],
    split: Split,
    calibration_count: usize,
) -> Vec<&CorpusQuery> {
    let ids = queries
        .iter()
        .map(|query| query.query_id.clone())
        .collect::<Vec<_>>();
    let calibration = split_membership(&ids, calibration_count);
    queries
        .iter()
        .filter(|query| match split {
            Split::Calibration => calibration.contains(&query.query_id),
            Split::Heldout => !calibration.contains(&query.query_id),
            Split::All => true,
        })
        .collect()
}

async fn open_and_prepare_store(
    args: &Args,
    corpus: &CorpusFile,
    corpus_file_sha256: &str,
) -> Result<(MemoryStore, StoreMapping, PathBuf), BoxError> {
    fs::create_dir_all(&args.store_dir)?;
    let mapping_path = args.store_dir.join("scifact-doc-map.json");
    let query_vectors = corpus
        .queries
        .iter()
        .map(|query| {
            (
                format!("search_query: {}", query.text),
                query.embedding.clone(),
            )
        })
        .collect::<HashMap<_, _>>();
    let embedder = FixtureEmbedder {
        model: corpus.embedding.model.clone(),
        dimensions: corpus.embedding.dimensions,
        queries: Arc::new(query_vectors),
    };
    let mut config = MemoryConfig {
        base_dir: args.store_dir.clone(),
        embedding: EmbeddingConfig {
            model: corpus.embedding.model.clone(),
            dimensions: corpus.embedding.dimensions,
            ..EmbeddingConfig::default()
        },
        ..MemoryConfig::default()
    };
    config.search.sparse_weight = 0.0;
    config.search.derive_sparse_from_dense = false;
    config.search.late_interaction_weight = 0.0;
    config.search.candidate_dims = None;
    config.search.recency_half_life_days = None;
    config.search.compress_results = false;
    config.search.derived_vector_backend = semantic_memory::DerivedVectorBackendPolicy::Disabled;
    config.search.default_top_k = TOP_K;
    let store = MemoryStore::open_with_embedder(config, Box::new(embedder))?;

    let mut mapping = if mapping_path.exists() {
        let mapping: StoreMapping = serde_json::from_slice(&fs::read(&mapping_path)?)?;
        if mapping.schema != MAPPING_SCHEMA
            || mapping.corpus_file_sha256 != corpus_file_sha256
            || mapping.corpus_payload_sha256 != corpus.payload_hashes.corpus_sha256
            || mapping.embedding_model != corpus.embedding.model
            || mapping.embedding_dimensions != corpus.embedding.dimensions
            || mapping.namespace != NAMESPACE
            || mapping.documents.len() > corpus.documents.len()
        {
            return Err(
                "persisted SciFact mapping does not match this corpus/model; use a new --store-dir"
                    .into(),
            );
        }
        mapping
    } else {
        StoreMapping {
            schema: MAPPING_SCHEMA.to_string(),
            corpus_file_sha256: corpus_file_sha256.to_string(),
            corpus_payload_sha256: corpus.payload_hashes.corpus_sha256.clone(),
            embedding_model: corpus.embedding.model.clone(),
            embedding_dimensions: corpus.embedding.dimensions,
            namespace: NAMESPACE.to_string(),
            documents: BTreeMap::new(),
        }
    };

    let corpus_doc_ids = corpus
        .documents
        .iter()
        .map(|document| document.doc_id.as_str())
        .collect::<HashSet<_>>();
    let existing_facts = store
        .list_facts(NAMESPACE, corpus.documents.len() + 1, 0)
        .await?;
    for fact in &existing_facts {
        let doc_id = fact
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("beir_doc_id"))
            .and_then(Value::as_str)
            .ok_or("SciFact namespace contains a fact without beir_doc_id metadata")?;
        if !corpus_doc_ids.contains(doc_id) {
            return Err(format!("SciFact store contains unknown BEIR doc ID '{doc_id}'").into());
        }
        if let Some(previous) = mapping
            .documents
            .insert(doc_id.to_string(), fact.id.clone())
        {
            if previous != fact.id {
                return Err(format!("multiple stored facts map to BEIR doc ID '{doc_id}'").into());
            }
        }
    }
    if mapping.documents.len() < corpus.documents.len() {
        eprintln!(
            "ingesting/resuming SciFact documents: {} already present, {} total",
            mapping.documents.len(),
            corpus.documents.len()
        );
        for (index, document) in corpus.documents.iter().enumerate() {
            if mapping.documents.contains_key(&document.doc_id) {
                continue;
            }
            let fact_id = store
                .add_fact_with_embedding(
                    NAMESPACE,
                    &document.semantic_text,
                    &document.embedding,
                    Some("beir-scifact"),
                    Some(json!({
                        "benchmark": "beir-scifact-test-v1",
                        "beir_doc_id": document.doc_id,
                        "semantic_text_sha256": sha256_bytes(document.semantic_text.as_bytes()),
                    })),
                )
                .await?;
            mapping.documents.insert(document.doc_id.clone(), fact_id);
            if index == 0 || (index + 1) % 100 == 0 || index + 1 == corpus.documents.len() {
                eprintln!("ingest: {}/{}", index + 1, corpus.documents.len());
                write_json_atomic(&mapping_path, &mapping)?;
            }
        }
    }
    write_json_atomic(&mapping_path, &mapping)?;

    let facts = store
        .list_facts(NAMESPACE, corpus.documents.len() + 1, 0)
        .await?;
    let stored_ids = facts
        .iter()
        .map(|fact| fact.id.as_str())
        .collect::<HashSet<_>>();
    let mapped_ids = mapping
        .documents
        .values()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    if facts.len() != corpus.documents.len() || stored_ids != mapped_ids {
        return Err("persisted store contents do not exactly match the mapping sidecar".into());
    }
    Ok((store, mapping, mapping_path))
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), BoxError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    let mut file = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn search_context(mode: Mode, split: Split, query: &CorpusQuery) -> SearchContext {
    SearchContext {
        evaluation_time: Utc::now(),
        receipt_mode: ReceiptMode::ReturnReceipt,
        replay_mode: ReplayMode::NoReplay,
        exactness_profile: if mode == Mode::VectorOnly {
            ExactnessProfile::PreferExact
        } else {
            ExactnessProfile::Default
        },
        request_id: Some(format!(
            "scifact:{}:{}:{}",
            mode.as_str(),
            split.as_str(),
            query.query_id
        )),
        query_text_digest: Some(sha256_bytes(query.text.as_bytes())),
        query_input_digest: Some(sha256_bytes(query.text.as_bytes())),
        filter_digest: Some(sha256_bytes(b"namespace=beir-scifact;source=facts")),
        redaction_state: Some("query_text_hashed_in_benchmark_receipt".to_string()),
        ..SearchContext::default_now()
    }
}

fn map_results(
    results: &[SearchResult],
    breakdowns: Option<&[semantic_memory::ExplainedResult]>,
    reverse_mapping: &HashMap<String, String>,
) -> Result<(Vec<String>, Vec<Value>), String> {
    let mut ranked = Vec::with_capacity(results.len());
    let mut rows = Vec::with_capacity(results.len());
    let mut seen_docs = HashSet::new();
    for (index, result) in results.iter().enumerate() {
        let fact_id = match &result.source {
            SearchSource::Fact { fact_id, .. } => fact_id,
            other => return Err(format!("unexpected non-fact result: {other:?}")),
        };
        let doc_id = reverse_mapping
            .get(fact_id)
            .ok_or_else(|| format!("retrieved fact '{fact_id}' is absent from mapping"))?;
        if !seen_docs.insert(doc_id.clone()) {
            return Err(format!(
                "retrieval returned duplicate BEIR doc ID '{doc_id}'"
            ));
        }
        ranked.push(doc_id.clone());
        let mut row = json!({
            "rank": index + 1,
            "doc_id": doc_id,
            "fact_id": fact_id,
            "score": result.score,
            "bm25_rank": result.bm25_rank,
            "vector_rank": result.vector_rank,
            "cosine_similarity": result.cosine_similarity,
        });
        if let Some(explained) = breakdowns.and_then(|values| values.get(index)) {
            row["breakdown"] =
                serde_json::to_value(&explained.breakdown).map_err(|error| error.to_string())?;
        }
        rows.push(row);
    }
    Ok((ranked, rows))
}

async fn execute_query(
    store: &MemoryStore,
    mode: Mode,
    split: Split,
    query: &CorpusQuery,
    reverse_mapping: &HashMap<String, String>,
) -> QueryRow {
    let started = Instant::now();
    let context = search_context(mode, split, query);
    let namespace = [NAMESPACE];
    let source_types = [SearchSourceType::Facts];
    let response: Result<SearchExecution, MemoryError> = match mode {
        Mode::FtsOnly => store
            .search_fts_only_with_context(
                &query.text,
                Some(TOP_K),
                Some(&namespace),
                Some(&source_types),
                context,
            )
            .await
            .map(|response| (response.results, None, response.receipt)),
        Mode::VectorOnly => store
            .search_vector_only_with_context(
                &query.text,
                Some(TOP_K),
                Some(&namespace),
                Some(&source_types),
                context,
            )
            .await
            .map(|response| (response.results, None, response.receipt)),
        Mode::Hybrid => store
            .search_explained_with_context(
                &query.text,
                Some(TOP_K),
                Some(&namespace),
                Some(&source_types),
                context,
            )
            .await
            .map(|response| {
                let results = response
                    .results
                    .iter()
                    .map(|result| result.result.clone())
                    .collect();
                (results, Some(response.results), response.receipt)
            }),
    };
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (ranked_doc_ids, results, receipt, error) = match response {
        Ok((search_results, breakdowns, receipt)) => {
            match map_results(&search_results, breakdowns.as_deref(), reverse_mapping) {
                Ok((ranked, rows)) => (ranked, rows, receipt, None),
                Err(error) => (Vec::new(), Vec::new(), receipt, Some(error)),
            }
        }
        Err(error) => (Vec::new(), Vec::new(), None, Some(error.to_string())),
    };
    let metrics = query_metrics(&ranked_doc_ids, &query.qrels);
    QueryRow {
        schema: ROW_SCHEMA.to_string(),
        mode: mode.as_str().to_string(),
        split: split.as_str().to_string(),
        query_id: query.query_id.clone(),
        query_sha256: sha256_bytes(query.text.as_bytes()),
        qrels: query.qrels.clone(),
        ranked_doc_ids,
        results,
        latency_ms,
        error,
        metrics,
        search_receipt: receipt,
    }
}

fn positive_relevant(qrels: &BTreeMap<String, i32>) -> HashSet<&str> {
    qrels
        .iter()
        .filter(|(_, grade)| **grade > 0)
        .map(|(doc_id, _)| doc_id.as_str())
        .collect()
}

fn query_metrics(ranked: &[String], qrels: &BTreeMap<String, i32>) -> QueryMetrics {
    let relevant = positive_relevant(qrels);
    let recall = |k: usize| {
        ranked
            .iter()
            .take(k)
            .filter(|doc_id| relevant.contains(doc_id.as_str()))
            .count() as f64
            / relevant.len() as f64
    };
    let success = |k: usize| {
        if ranked
            .iter()
            .take(k)
            .any(|doc_id| relevant.contains(doc_id.as_str()))
        {
            1.0
        } else {
            0.0
        }
    };
    let dcg = ranked
        .iter()
        .take(TOP_K)
        .enumerate()
        .map(|(index, doc_id)| {
            let grade = f64::from(*qrels.get(doc_id).unwrap_or(&0));
            (2_f64.powf(grade) - 1.0) / ((index + 2) as f64).log2()
        })
        .sum::<f64>();
    let mut ideal = qrels
        .values()
        .copied()
        .filter(|grade| *grade > 0)
        .collect::<Vec<_>>();
    ideal.sort_by(|left, right| right.cmp(left));
    let idcg = ideal
        .iter()
        .take(TOP_K)
        .enumerate()
        .map(|(index, grade)| (2_f64.powi(*grade) - 1.0) / ((index + 2) as f64).log2())
        .sum::<f64>();
    let mrr = ranked
        .iter()
        .take(TOP_K)
        .position(|doc_id| relevant.contains(doc_id.as_str()))
        .map(|index| 1.0 / (index + 1) as f64)
        .unwrap_or(0.0);
    let mut hits = 0_usize;
    let mut precision_sum = 0.0;
    for (index, doc_id) in ranked.iter().take(TOP_K).enumerate() {
        if relevant.contains(doc_id.as_str()) {
            hits += 1;
            precision_sum += hits as f64 / (index + 1) as f64;
        }
    }
    QueryMetrics {
        ndcg_at_10: if idcg > 0.0 { dcg / idcg } else { 0.0 },
        recall_at_1: recall(1),
        recall_at_5: recall(5),
        recall_at_10: recall(10),
        mrr_at_10: mrr,
        map_at_10: precision_sum / relevant.len().min(TOP_K) as f64,
        success_at_1: success(1),
        success_at_5: success(5),
        success_at_10: success(10),
    }
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let index = ((quantile * ordered.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(ordered.len() - 1);
    ordered[index]
}

fn aggregate_metrics(rows: &[QueryRow]) -> Value {
    let count = rows.len() as f64;
    let mean = |field: fn(&QueryMetrics) -> f64| {
        rows.iter().map(|row| field(&row.metrics)).sum::<f64>() / count
    };
    let latencies = rows.iter().map(|row| row.latency_ms).collect::<Vec<_>>();
    let mut result_counts = BTreeMap::<String, usize>::new();
    let mut unique_docs = HashSet::new();
    let mut top_ones = HashMap::<String, usize>::new();
    for row in rows {
        *result_counts
            .entry(row.ranked_doc_ids.len().to_string())
            .or_default() += 1;
        unique_docs.extend(row.ranked_doc_ids.iter().cloned());
        if let Some(doc_id) = row.ranked_doc_ids.first() {
            *top_ones.entry(doc_id.clone()).or_default() += 1;
        }
    }
    let repeated_top1 = top_ones
        .iter()
        .max_by(|left, right| left.1.cmp(right.1).then_with(|| right.0.cmp(left.0)))
        .map(|(doc_id, count)| (Some(doc_id.clone()), *count))
        .unwrap_or((None, 0));
    let nonempty = top_ones.values().sum::<usize>();
    json!({
        "ndcg_at_10": mean(|metrics| metrics.ndcg_at_10),
        "recall_at_1": mean(|metrics| metrics.recall_at_1),
        "recall_at_5": mean(|metrics| metrics.recall_at_5),
        "recall_at_10": mean(|metrics| metrics.recall_at_10),
        "mrr_at_10": mean(|metrics| metrics.mrr_at_10),
        "map_at_10": mean(|metrics| metrics.map_at_10),
        "success_at_1": mean(|metrics| metrics.success_at_1),
        "success_at_5": mean(|metrics| metrics.success_at_5),
        "success_at_10": mean(|metrics| metrics.success_at_10),
        "latency_ms": {
            "mean": latencies.iter().sum::<f64>() / count,
            "p50": percentile(&latencies, 0.50),
            "p95": percentile(&latencies, 0.95),
            "max": latencies.iter().copied().fold(0.0_f64, f64::max),
        },
        "failures": rows.iter().filter(|row| row.error.is_some()).count(),
        "result_count_distribution": result_counts,
        "unique_retrieved_docs": unique_docs.len(),
        "repeated_top1": {
            "doc_id": repeated_top1.0,
            "count": repeated_top1.1,
            "frequency": if nonempty == 0 { 0.0 } else { repeated_top1.1 as f64 / nonempty as f64 },
        },
    })
}

fn command_output(args: &[&str]) -> Option<String> {
    let output = Command::new(args[0]).args(&args[1..]).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || "-._/:=".contains(character))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn current_command() -> String {
    env::args()
        .map(|argument| shell_quote(&argument))
        .collect::<Vec<_>>()
        .join(" ")
}

fn receipt_evidence(rows: &[QueryRow]) -> Value {
    let receipts = rows
        .iter()
        .filter_map(|row| row.search_receipt.as_ref())
        .collect::<Vec<_>>();
    let backends = receipts
        .iter()
        .map(|receipt| receipt.candidate_backend.clone())
        .collect::<BTreeSet<_>>();
    let profiles = receipts
        .iter()
        .map(|receipt| receipt.search_profile.clone())
        .collect::<BTreeSet<_>>();
    let exact_rerank_values = receipts
        .iter()
        .map(|receipt| receipt.exact_rerank)
        .collect::<BTreeSet<_>>();
    let degradations = receipts
        .iter()
        .flat_map(|receipt| receipt.degradations.clone())
        .collect::<BTreeSet<_>>();
    json!({
        "receipt_count": receipts.len(),
        "candidate_backends": backends,
        "search_profiles": profiles,
        "exact_rerank_values": exact_rerank_values,
        "degradations": degradations,
    })
}

#[allow(clippy::too_many_arguments)]
fn aggregate_receipt(
    args: &Args,
    corpus: &CorpusFile,
    corpus_path: &Path,
    corpus_file_sha256: &str,
    mapping_path: &Path,
    mapping_sha256: &str,
    mode: Mode,
    rows: &[QueryRow],
    rows_path: &Path,
    rows_sha256: &str,
) -> Result<Value, BoxError> {
    let executable = fs::canonicalize(env::current_exe()?)?;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runner_source = manifest_dir.join("examples/scifact_retrieval_eval.rs");
    let store_path = fs::canonicalize(&args.store_dir)?;
    let database_path = store_path.join("memory.db");
    let crate_git_commit =
        command_output(&["git", "-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"]);
    let crate_git_status = command_output(&[
        "git",
        "-C",
        env!("CARGO_MANIFEST_DIR"),
        "status",
        "--porcelain=v1",
    ])
    .unwrap_or_default();
    let workspace_dir = manifest_dir
        .parent()
        .ok_or("semantic-memory manifest has no workspace parent")?;
    let workspace_git_commit = command_output(&[
        "git",
        "-C",
        workspace_dir
            .to_str()
            .ok_or("workspace path is not UTF-8")?,
        "rev-parse",
        "HEAD",
    ]);
    let workspace_git_status = command_output(&[
        "git",
        "-C",
        workspace_dir
            .to_str()
            .ok_or("workspace path is not UTF-8")?,
        "status",
        "--porcelain=v1",
    ])
    .unwrap_or_default();
    let split_ids = rows
        .iter()
        .map(|row| row.query_id.clone())
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": AGGREGATE_SCHEMA,
        "mode": mode.as_str(),
        "split": args.split.as_str(),
        "row_count": rows.len(),
        "per_query_path": fs::canonicalize(rows_path)?.to_string_lossy(),
        "per_query_sha256": rows_sha256,
        "metrics": aggregate_metrics(rows),
        "receipt": {
            "schema_sha256": sha256_bytes(SCHEMA_DESCRIPTOR.as_bytes()),
            "executable": {"path": executable.to_string_lossy(), "sha256": sha256_file(&executable)?},
            "launcher": {
                "reported_command": env::var("SCIFACT_EVAL_LAUNCHER").ok(),
                "cargo_executable": env::var("CARGO").ok(),
                "note": "set SCIFACT_EVAL_LAUNCHER to the exact outer cargo/shell command when launcher provenance is required; final_command always records the resolved executable invocation",
            },
            "source": {
                "runner_path": runner_source.to_string_lossy(),
                "runner_sha256": sha256_file(&runner_source)?,
                "crate_repository": {
                    "root": manifest_dir.to_string_lossy(),
                    "git_commit": crate_git_commit,
                    "git_dirty": !crate_git_status.is_empty(),
                    "git_status_sha256": sha256_bytes(crate_git_status.as_bytes()),
                },
                "workspace_repository": {
                    "root": workspace_dir.to_string_lossy(),
                    "git_commit": workspace_git_commit,
                    "git_dirty": !workspace_git_status.is_empty(),
                    "git_status_sha256": sha256_bytes(workspace_git_status.as_bytes()),
                },
            },
            "corpus": {
                "id": corpus.corpus_id,
                "path": corpus_path.to_string_lossy(),
                "file_sha256": corpus_file_sha256,
                "source": corpus.source,
                "source_hashes": corpus.source_hashes,
                "payload_hashes": corpus.payload_hashes,
            },
            "embedding": corpus.embedding,
            "truncation": corpus.truncation,
            "store": {
                "path": store_path.to_string_lossy(),
                "database_path": database_path.to_string_lossy(),
                "database_sha256_after_run": sha256_file(&database_path)?,
                "namespace": NAMESPACE,
                "mapping_path": fs::canonicalize(mapping_path)?.to_string_lossy(),
                "mapping_sha256": mapping_sha256,
                "document_count": corpus.documents.len(),
            },
            "retrieval": {
                "mode": mode.as_str(),
                "top_k": TOP_K,
                "source_types": ["facts"],
                "config": {
                    "sparse_weight": 0.0,
                    "derive_sparse_from_dense": false,
                    "late_interaction_weight": 0.0,
                    "candidate_dims": null,
                    "recency_half_life_days": null,
                    "derived_vector_backend": "disabled",
                    "vector_exactness": if mode == Mode::VectorOnly { "prefer_exact" } else { "default" },
                },
                "evidence": receipt_evidence(rows),
            },
            "split_definition": {
                "algorithm": "sort all test query IDs by (sha256(utf8(query_id)), query_id); first calibration_count are calibration; remainder heldout",
                "calibration_count": args.calibration_count,
                "total_query_count": corpus.queries.len(),
                "selected_query_ids_sha256": sha256_bytes(split_ids.join("\n").as_bytes()),
            },
            "capabilities": {
                "native_sparse": {"available": false, "reason": "the all-minilm fixture supplies dense vectors only; dense-derived sparse is disabled and is not called native sparse or SPLADE"},
                "factor_graph": {"available": false, "reason": "SciFact ingestion creates no legitimate graph edges"},
                "hybrid_matryoshka": {"available": false, "reason": "no independently materialized compatible reduced-dimension candidate index is part of this frozen baseline"},
                "hybrid_late_interaction": {"available": false, "reason": "the corpus contains one dense vector per document/query, not genuine token-level late-interaction vectors"},
            },
            "final_command": current_command(),
            "thresholds": {
                "quality_gate": null,
                "policy": "no quality threshold is selected or tuned by this runner; calibration may diagnose a frozen configuration, heldout is evaluation-only",
            },
            "claim_boundary": "official BEIR SciFact retrieval quality and local latency for the named semantic-memory production APIs/configuration only; no superiority, general-domain, graph, native-sparse/SPLADE, late-interaction, or matryoshka claim",
        },
    }))
}

#[allow(clippy::too_many_arguments)]
async fn run_mode(
    args: &Args,
    corpus: &CorpusFile,
    corpus_path: &Path,
    corpus_file_sha256: &str,
    store: &MemoryStore,
    mapping: &StoreMapping,
    mapping_path: &Path,
    mode: Mode,
) -> Result<(), BoxError> {
    fs::create_dir_all(&args.output_dir)?;
    let prefix = format!("scifact-{}-{}", mode.as_str(), args.split.as_str());
    let rows_path = args.output_dir.join(format!("{prefix}.jsonl"));
    let aggregate_path = args.output_dir.join(format!("{prefix}.aggregate.json"));
    let reverse_mapping = mapping
        .documents
        .iter()
        .map(|(doc_id, fact_id)| (fact_id.clone(), doc_id.clone()))
        .collect::<HashMap<_, _>>();
    let queries = selected_queries(&corpus.queries, args.split, args.calibration_count);
    let file = File::create(&rows_path)?;
    let mut writer = BufWriter::new(file);
    let mut rows = Vec::with_capacity(queries.len());
    for (index, query) in queries.iter().enumerate() {
        let row = execute_query(store, mode, args.split, query, &reverse_mapping).await;
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        rows.push(row);
        if index == 0 || (index + 1) % 25 == 0 || index + 1 == queries.len() {
            eprintln!(
                "{} {}: {}/{}",
                mode.as_str(),
                args.split.as_str(),
                index + 1,
                queries.len()
            );
        }
    }
    writer.get_ref().sync_all()?;
    drop(writer);
    let rows_path = fs::canonicalize(rows_path)?;
    let rows_sha256 = sha256_file(&rows_path)?;
    let mapping_sha256 = sha256_file(mapping_path)?;
    let aggregate = aggregate_receipt(
        args,
        corpus,
        corpus_path,
        corpus_file_sha256,
        mapping_path,
        &mapping_sha256,
        mode,
        &rows,
        &rows_path,
        &rows_sha256,
    )?;
    write_json_atomic(&aggregate_path, &aggregate)?;
    eprintln!(
        "wrote {} and {}",
        rows_path.display(),
        aggregate_path.display()
    );
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), BoxError> {
    let args = parse_args()?;
    if args.calibration_count == 0 {
        return Err("--calibration-count must be positive".into());
    }
    let corpus_path = fs::canonicalize(&args.corpus)?;
    let corpus_file_sha256 = sha256_file(&corpus_path)?;
    let corpus: CorpusFile = serde_json::from_slice(&fs::read(&corpus_path)?)?;
    validate_corpus(&corpus)?;
    if args.calibration_count >= corpus.queries.len() {
        return Err("--calibration-count must be smaller than the test query count".into());
    }
    let (store, mapping, mapping_path) =
        open_and_prepare_store(&args, &corpus, &corpus_file_sha256).await?;
    for mode in &args.modes {
        run_mode(
            &args,
            &corpus,
            &corpus_path,
            &corpus_file_sha256,
            &store,
            &mapping,
            &mapping_path,
            *mode,
        )
        .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qrels(ids: &[(&str, i32)]) -> BTreeMap<String, i32> {
        ids.iter()
            .map(|(id, grade)| ((*id).to_string(), *grade))
            .collect()
    }

    #[test]
    fn metrics_match_tiny_fixture() {
        let ranked = vec!["x".to_string(), "b".to_string(), "a".to_string()];
        let metrics = query_metrics(&ranked, &qrels(&[("a", 1), ("b", 1)]));
        assert!((metrics.recall_at_1 - 0.0).abs() < 1e-12);
        assert!((metrics.recall_at_5 - 1.0).abs() < 1e-12);
        assert!((metrics.mrr_at_10 - 0.5).abs() < 1e-12);
        assert!((metrics.map_at_10 - ((0.5 + 2.0 / 3.0) / 2.0)).abs() < 1e-12);
        assert!(metrics.ndcg_at_10 > 0.0 && metrics.ndcg_at_10 < 1.0);
        assert_eq!(metrics.success_at_1, 0.0);
        assert_eq!(metrics.success_at_5, 1.0);
    }

    #[test]
    fn graded_ndcg_rewards_ideal_order() {
        let labels = qrels(&[("high", 3), ("low", 1)]);
        let ideal = query_metrics(&["high".into(), "low".into()], &labels);
        let reversed = query_metrics(&["low".into(), "high".into()], &labels);
        assert!((ideal.ndcg_at_10 - 1.0).abs() < 1e-12);
        assert!(reversed.ndcg_at_10 < ideal.ndcg_at_10);
    }

    #[test]
    fn split_is_deterministic_and_disjoint() {
        let ids = (0..300)
            .map(|index| format!("q{index}"))
            .collect::<Vec<_>>();
        let first = split_membership(&ids, 100);
        let mut reversed = ids.clone();
        reversed.reverse();
        let second = split_membership(&reversed, 100);
        assert_eq!(first, second);
        assert_eq!(first.len(), 100);
        let heldout = ids
            .iter()
            .filter(|id| !first.contains(*id))
            .collect::<HashSet<_>>();
        assert_eq!(heldout.len(), 200);
        assert!(first.iter().all(|id| !heldout.contains(id)));
    }

    #[test]
    fn nearest_rank_percentiles_are_stable() {
        assert_eq!(percentile(&[3.0, 1.0, 2.0, 4.0], 0.50), 2.0);
        assert_eq!(percentile(&[3.0, 1.0, 2.0, 4.0], 0.95), 4.0);
    }
}
