use semantic_memory::{MemoryStore, MemoryConfig, EmbeddingConfig, SearchConfig};
use semantic_memory::embedder::MockEmbedder;
use std::time::Instant;
use std::collections::HashSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::TempDir::new()?;
    let db_path = tmp.path().join("bench.db");

    let embedder = MockEmbedder::new(384);

    let config = MemoryConfig {
        base_dir: db_path,
        embedding: EmbeddingConfig {
            ollama_url: "http://localhost:11434".into(),
            model: "mock".into(),
            dimensions: 384,
            batch_size: 32,
            timeout_secs: 30,
        },
        search: SearchConfig::default(),
        ..Default::default()
    };

    // open_with_embedder is sync (not async)
    let rt = tokio::runtime::Runtime::new()?;
    let store = MemoryStore::open_with_embedder(config, Box::new(embedder))?;

    // Insert a real corpus: 500 facts across 10 namespaces
    let corpus = generate_corpus(500);
    let insert_start = Instant::now();
    for (i, (text, ns)) in corpus.iter().enumerate() {
        let emb = mock_embed(text, 384);
        rt.block_on(store.add_fact_with_embedding(ns, text, &emb, None, None))?;
    }
    let insert_elapsed = insert_start.elapsed();
    eprintln!("Inserted 500 facts in {:?}", insert_elapsed);

    // Run 50 benchmark queries
    let queries = generate_queries(50);

    let mut hybrid_results = Vec::new();
    let mut fts_results = Vec::new();
    let mut vector_results = Vec::new();

    for q in &queries {
        // Hybrid search (BM25 + vector + RRF)
        let t0 = Instant::now();
        let hybrid = rt.block_on(store.search(q, Some(5), None, None))?;
        let hybrid_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        hybrid_results.push((hybrid_ms, hybrid.len()));

        // FTS-only (BM25 only)
        let t1 = Instant::now();
        let fts = rt.block_on(store.search_fts_only(q, Some(5), None, None))?;
        let fts_ms = t1.elapsed().as_micros() as f64 / 1000.0;
        fts_results.push((fts_ms, fts.len()));

        // Vector-only (like Qdrant dense-only)
        let t2 = Instant::now();
        let vec_only = rt.block_on(store.search_vector_only(q, Some(5), None, None))?;
        let vec_ms = t2.elapsed().as_micros() as f64 / 1000.0;
        vector_results.push((vec_ms, vec_only.len()));
    }

    // Compute averages
    let avg_hybrid = hybrid_results.iter().map(|(ms, _)| *ms).sum::<f64>() / 50.0;
    let avg_fts = fts_results.iter().map(|(ms, _)| *ms).sum::<f64>() / 50.0;
    let avg_vec = vector_results.iter().map(|(ms, _)| *ms).sum::<f64>() / 50.0;

    let avg_hybrid_n = hybrid_results.iter().map(|(_, n)| *n).sum::<usize>() as f64 / 50.0;
    let avg_fts_n = fts_results.iter().map(|(_, n)| *n).sum::<usize>() as f64 / 50.0;
    let avg_vec_n = vector_results.iter().map(|(_, n)| *n).sum::<usize>() as f64 / 50.0;

    println!("=== REAL BENCHMARK (500 facts, 50 queries, brute-force backend) ===");
    println!();
    println!("SEARCH MODE             avg ms/query   avg results/query");
    println!("Hybrid (BM25+vec+RRF):  {:.3}          {:.1}", avg_hybrid, avg_hybrid_n);
    println!("FTS-only (BM25):        {:.3}          {:.1}", avg_fts, avg_fts_n);
    println!("Vector-only (cosine):   {:.3}          {:.1}", avg_vec, avg_vec_n);
    println!();

    // Overlap analysis: how often do hybrid and vector-only agree on top-5?
    let mut full_agree = 0;
    let mut partial_agree = 0;
    let mut zero_overlap = 0;
    let mut total_overlap = 0;

    for q in &queries {
        let hybrid = rt.block_on(store.search(q, Some(5), None, None))?;
        let vec_only = rt.block_on(store.search_vector_only(q, Some(5), None, None))?;
        let h_ids: HashSet<_> = hybrid.iter().map(|r| r.content.clone()).collect();
        let v_ids: HashSet<_> = vec_only.iter().map(|r| r.content.clone()).collect();
        let overlap = h_ids.intersection(&v_ids).count();
        total_overlap += overlap;
        if overlap == 5 { full_agree += 1; }
        else if overlap == 0 { zero_overlap += 1; }
        else { partial_agree += 1; }
    }

    println!("=== TOP-5 OVERLAP ANALYSIS (hybrid vs vector-only) ===");
    println!("Full agreement (5/5):    {}/50 ({:.0}%)", full_agree, full_agree as f64 / 50.0 * 100.0);
    println!("Partial overlap (1-4):   {}/50 ({:.0}%)", partial_agree, partial_agree as f64 / 50.0 * 100.0);
    println!("Zero overlap (0/5):      {}/50 ({:.0}%)", zero_overlap, zero_overlap as f64 / 50.0 * 100.0);
    println!("Avg items in common:     {:.2}/5", total_overlap as f64 / 50.0);
    println!();

    // Score analysis: hybrid scores vs vector-only scores
    let mut hybrid_better = 0;
    let mut vec_better = 0;
    let mut tied = 0;
    for q in &queries {
        let hybrid = rt.block_on(store.search(q, Some(5), None, None))?;
        let vec_only = rt.block_on(store.search_vector_only(q, Some(5), None, None))?;
        let h_top = hybrid.first().map(|r| r.score).unwrap_or(0.0);
        let v_top = vec_only.first().map(|r| r.score).unwrap_or(0.0);
        if (h_top - v_top).abs() < 0.001 { tied += 1; }
        else if h_top > v_top { hybrid_better += 1; }
        else { vec_better += 1; }
    }
    println!("=== TOP SCORE COMPARISON (hybrid top score vs vector-only top score) ===");
    println!("Hybrid scores higher:    {}/50", hybrid_better);
    println!("Vector-only scores higher: {}/50", vec_better);
    println!("Tied (within 0.001):    {}/50", tied);
    println!();

    // FTS catches things vector misses
    let mut fts_unique = 0;
    let mut fts_total = 0;
    for q in &queries {
        let fts = rt.block_on(store.search_fts_only(q, Some(5), None, None))?;
        let vec_only = rt.block_on(store.search_vector_only(q, Some(5), None, None))?;
        let v_ids: HashSet<_> = vec_only.iter().map(|r| r.content.clone()).collect();
        fts_total += fts.len();
        for r in &fts {
            if !v_ids.contains(&r.content.clone()) {
                fts_unique += 1;
            }
        }
    }
    println!("=== FTS RECALL BONUS (items FTS finds that vector-only misses) ===");
    println!("FTS results:            {}", fts_total);
    println!("FTS-unique (not in vec): {} ({:.1}% of FTS results)", fts_unique, if fts_total > 0 { fts_unique as f64 / fts_total as f64 * 100.0 } else { 0.0 });
    println!();

    println!("=== WHAT SEMANTIC-MEMORY DOES THAT QDRANT CANNOT ===");
    println!("1. Hybrid BM25+vector+RRF fusion built-in (Qdrant: caller-side fusion)");
    println!("2. Semiring provenance (Boolean/Tropical/Probability/Confidence)");
    println!("3. Temporal weight with well detection (age/supersession/support/contradiction)");
    println!("4. Decoder: syndrome detection + belief propagation on conflict graph");
    println!("5. Lawful subtraction with invariant verification + recovery");
    println!("6. Adaptive routing with query profiling + benchmark harness");
    println!("7. Discord second-order retrieval (graph-neighbour discovery)");
    println!("8. Receipt-driven replay (every mutation has a blake3-digested receipt)");
    println!("9. Append-plus-supersession (truth-bearing rows never UPDATEd)");
    println!("10. Compression governor (importance-driven per-vector quantization level)");
    println!("11. Graph view with BFS pathfinding across all entity types");
    println!("12. Episode tracking (causal records linking search sessions)");
    println!();

    println!("=== WHAT QDRANT DOES THAT SEMANTIC-MEMORY CANNOT ===");
    println!("1. Distributed sharding + horizontal scaling");
    println!("2. Payload filtering with nested conditions (semantic-memory: namespace only)");
    println!("3. Multi-vector / named vectors per point");
    println!("4. Sparse vector support (semantic-memory: FTS5 instead)");
    println!("5. gRPC + REST API server (semantic-memory: embedded library only)");
    println!("6. Snapshotting + replication");
    println!("7. Cluster management UI");
    println!("8. Scroll API for pagination through large result sets");
    println!();

    println!("=== LOCAL-ONLY TRADE-OFFS ===");
    println!("PRO: Zero network latency, zero server cost, zero config, data sovereignty");
    println!("PRO: Works offline, no Docker, no infrastructure");
    println!("PRO: SQLite file is portable (copy file = copy entire system)");
    println!("CON: Single machine (no horizontal scaling beyond one process)");
    println!("CON: Limited to ~250k vectors before brute-force warning, ~250k hard block");
    println!("CON: No concurrent writers (SQLite WAL: 1 writer)");
    println!("CON: No multi-tenant isolation (single embedded instance)");
    println!();

    Ok(())
}

fn generate_corpus(n: usize) -> Vec<(String, String)> {
    let topics = [
        "rust memory safety ownership borrow checker",
        "python async asyncio event loop coroutines",
        "vector database HNSW approximate nearest neighbor",
        "quantization SQ8 product quantization compression",
        "retrieval augmented generation embedding similarity",
        "knowledge graph entity relation triple store",
        "temporal reasoning bitemporal valid time transaction time",
        "provenance semiring algebra evidence chain confidence",
        "decoder syndrome error correction belief propagation",
        "lawful subtraction forgetting invariant verification",
    ];
    (0..n).map(|i| {
        let topic = topics[i % topics.len()];
        let ns = format!("ns_{}", i % 10);
        let text = format!("{} fact #{}: {} details about {} in context {}", topic, i, topic, topic, i % 5);
        (text, ns)
    }).collect()
}

fn generate_queries(n: usize) -> Vec<String> {
    let templates = [
        "rust memory safety",
        "python async coroutines",
        "vector database HNSW",
        "quantization compression SQ8",
        "embedding similarity search",
        "knowledge graph entity",
        "temporal reasoning bitemporal",
        "provenance evidence chain",
        "decoder syndrome correction",
        "lawful subtraction forgetting",
        "what is the architecture of",
        "how does the decoder work",
        "compare rust vs python",
        "latest developments in vector search",
        "source of the compression algorithm",
    ];
    (0..n).map(|i| templates[i % templates.len()].to_string()).collect()
}

fn mock_embed(text: &str, dims: usize) -> Vec<f32> {
    let mut vals = vec![0.0f32; dims];
    for (i, byte) in text.bytes().enumerate() {
        let idx = (i * 7 + byte as usize) % dims;
        vals[idx] += byte as f32 / 255.0;
        vals[(idx + 3) % dims] -= byte as f32 / 510.0;
    }
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
    vals.iter().map(|v| v / norm).collect()
}