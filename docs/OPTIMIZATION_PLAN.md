# semantic-memory Optimization Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Wire up all existing-but-unused modules into the search pipeline, add proven optimizations, close all gaps vs competitors, and achieve production-grade retrieval quality and latency.

**Architecture:** All changes are to the semantic-memory crate (`/home/sikmindz/Coding/Libraries/semantic-memory/`) and its MCP server (`/home/sikmindz/Coding/Libraries/semantic-memory-mcp/`). The workspace root is `/home/sikmindz/Coding/Libraries/`. Build with `cargo test --all-features` from the workspace root.

**Tech Stack:** Rust 2021, MSRV 1.75, SQLite + FTS5 + usearch 2.25, Candle (CPU) / Ollama (GPU-accelerated), rmcp 1.8, 708 tests currently passing.

**Current state (verified 2026-06-23):**
- 708 tests passing, 0 failures
- Warm HTTP server at 127.0.0.1:1738, ~138ms avg query latency
- FTS5 already uses `porter unicode61` tokenizer (stemming is active)
- Embedding LRU cache exists (256 entries) for query embeddings
- Asymmetric prefix active (`search_query:` for queries, `search_document:` for docs)
- SQLite PRAGMAs: WAL, cache_size=25MB, mmap_size=256MB
- Search dedup by source-type + first-30-words fingerprint
- Subgraph neighborhood loading for discord/factor_graph
- Ingestion dedup for facts and documents

---

## Phase 1: Wire temporal_weight into search scoring (EASY, HIGH ROI)

**Problem:** `temporal_weight` is computed per fact/chunk (temporal feature is enabled) but never multiplied into the final RRF score. Stale facts rank the same as fresh ones.

### Task 1.1: Add temporal_weight to CandidateRow

**Files:**
- Modify: `src/search.rs` — the `CandidateRow` struct and the SQL query that populates it

**Step 1:** Read the CandidateRow struct and the SQL query that builds candidate rows.

```bash
grep -n "struct CandidateRow" src/search.rs
grep -n "SELECT.*FROM.*facts\|SELECT.*FROM.*chunks" src/search.rs | head -10
```

**Step 2:** Add `temporal_weight: Option<f64>` to `CandidateRow`.

**Step 3:** Add `temporal_weight` to the SELECT columns in the BM25 and vector candidate SQL queries. The column already exists in facts, chunks, and messages tables (migration V26).

**Step 4:** Run `cargo check --all-features`. Fix any type errors.

**Step 5:** Commit.

### Task 1.2: Apply temporal_weight in RRF score

**Files:**
- Modify: `src/search.rs` — the `finalize_score` or equivalent method where `rrf_score` is computed

**Step 1:** Find the line `let rrf_score = bm25_contribution.unwrap_or(0.0) + vector_contribution...`

**Step 2:** After computing `rrf_score`, multiply by temporal_weight:
```rust
let temporal_factor = self.temporal_weight.unwrap_or(1.0);
let rrf_score = rrf_score * temporal_factor;
```

**Step 3:** Write a test: ingest a fact, wait, ingest another with same content, verify the older one scores lower.

**Step 4:** Run `cargo test --all-features -- temporal`. Expected: PASS.

**Step 5:** Commit.

**Phase 1 Gate:** `cargo test --all-features` — 708+ tests, 0 failures.

---

## Phase 2: Enable late-interaction RRF signal (EASY, HIGH ROI)

**Problem:** `late_interaction_weight` defaults to 0.0 and the `late-interaction` feature is not in the MCP build. The proxy MaxSim code exists and computes scores but they have zero effect.

### Task 2.1: Add late-interaction to MCP features

**Files:**
- Modify: `semantic-memory-mcp/Cargo.toml` — add `"semantic-memory/late-interaction"` to the `full` feature list

**Step 1:** Add `"semantic-memory/late-interaction"` to the `full = [...]` array in `semantic-memory-mcp/Cargo.toml`.

**Step 2:** Run `cd /home/sikmindz/Coding/Libraries/semantic-memory-mcp && cargo check`. Expected: compiles.

**Step 3:** Commit.

### Task 2.2: Set late_interaction_weight default to 0.15

**Files:**
- Modify: `src/config.rs` — change default from 0.0 to 0.15

**Step 1:** Find `late_interaction_weight: 0.0` in `src/config.rs` and change to `late_interaction_weight: 0.15`.

**Step 2:** Run `cargo test --all-features -- late_interaction`. Expected: PASS.

**Step 3:** Commit.

**Phase 2 Gate:** `cargo test --all-features` — all pass. `cd semantic-memory-mcp && cargo check` — compiles.

---

## Phase 3: Document embedding cache (EASY, HIGH ROI)

**Problem:** `embed_batch_internal` (document ingestion path) doesn't use the LRU cache. Re-ingesting the same content re-embeds it.

### Task 3.1: Add cache to embed_batch_internal

**Files:**
- Modify: `src/lib.rs` — `embed_batch_internal` method

**Step 1:** In `embed_batch_internal`, before calling `self.inner.embedder.embed_batch(texts)`, check each text against the cache. Collect hits and only embed misses.

```rust
async fn embed_batch_internal(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, MemoryError> {
    let requested = texts.len();
    
    // Check cache for each text
    let mut results: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
    let mut misses: Vec<String> = Vec::new();
    let mut miss_indices: Vec<usize> = Vec::new();
    
    for (i, text) in texts.iter().enumerate() {
        let mut cache = self.inner.embedding_cache.lock().expect("cache lock poisoned");
        if let Some(cached) = cache.get(text).cloned() {
            results.push(Some(cached));
        } else {
            results.push(None);
            miss_indices.push(i);
            misses.push(text.clone());
        }
    }
    
    // Only embed misses
    let _permit = self.with_embedding_permit().await?;
    let miss_embeddings = if misses.is_empty() {
        Vec::new()
    } else {
        let embeddings = self.inner.embedder.embed_batch(misses.clone()).await?;
        // Cache the new embeddings
        let mut cache = self.inner.embedding_cache.lock().expect("cache lock poisoned");
        for (text, emb) in misses.iter().zip(embeddings.iter()) {
            cache.put(text.clone(), emb.clone());
        }
        embeddings
    };
    
    // Assemble results in order
    let mut final_results = Vec::with_capacity(requested);
    let mut miss_idx = 0;
    for i in 0..requested {
        if let Some(emb) = &results[i] {
            final_results.push(emb.clone());
        } else {
            final_results.push(miss_embeddings[miss_idx].clone());
            miss_idx += 1;
        }
    }
    
    db::validate_embedding_batch(&final_results, requested, self.inner.config.embedding.dimensions)?;
    Ok(final_results)
}
```

**Step 2:** Run `cargo test --all-features`. Expected: all pass.

**Step 3:** Commit.

**Phase 3 Gate:** `cargo test --all-features` — all pass.

---

## Phase 4: Query expansion for hyphens and case (EASY, HIGH ROI)

**Problem:** BM25 with porter stemming handles case and basic morphology, but hyphenated variants ("turbo-quant" vs "turboquant" vs "TurboQuant") don't match.

### Task 4.1: Add query normalization function

**Files:**
- Modify: `src/search.rs` — add a `normalize_query` function

**Step 1:** Add a function that generates FTS5 OR variants for hyphenated terms:
```rust
fn expand_query_for_fts(query: &str) -> String {
    // For each term with a hyphen, also add the de-hyphenated variant
    // e.g. "turbo-quant" -> "turbo-quant OR turboquant"
    let terms: Vec<&str> = query.split_whitespace().collect();
    let expanded: Vec<String> = terms.iter().map(|term| {
        if term.contains('-') {
            let no_hyphen = term.replace('-', "");
            format!("{term} OR {no_hyphen}")
        } else {
            term.to_string()
        }
    }).collect();
    expanded.join(" ")
}
```

**Step 2:** Apply `expand_query_for_fts` to the query string before it's passed to the FTS5 MATCH clause.

**Step 3:** Write a test: search "turboquant" (no hyphen) against DB with "turbo-quant" content, verify results returned.

**Step 4:** Run `cargo test --all-features -- query_expansion`. Expected: PASS.

**Step 5:** Commit.

**Phase 4 Gate:** `cargo test --all-features` — all pass.

---

## Phase 5: Result diversity — same-document dedup (EASY, HIGH ROI)

**Problem:** If the top 5 results are all chunks from the same document, the user gets a narrow view. No diversity reordering exists.

### Task 5.1: Add document diversity to dedup_by_content

**Files:**
- Modify: `src/lib.rs` — `dedup_by_content` function

**Step 1:** Extend `dedup_by_content` to also limit results per document_id. After the existing source-type+content fingerprint dedup, add a second pass that limits to max 2 results per `document_id` (for Chunk sources):

```rust
fn dedup_by_content(results: Vec<types::SearchResult>) -> Vec<types::SearchResult> {
    use std::collections::{HashSet, HashMap};
    
    // Pass 1: content fingerprint dedup (existing)
    let mut seen: HashSet<String> = HashSet::new();
    let mut deduped: Vec<types::SearchResult> = results
        .into_iter()
        .filter(|r| {
            let fingerprint: String = r.content.split_whitespace().take(30)
                .collect::<Vec<_>>().join(" ").to_lowercase();
            let source_type = match &r.source {
                types::SearchSource::Fact { .. } => "fact",
                types::SearchSource::Chunk { .. } => "chunk",
                types::SearchSource::Message { .. } => "message",
                types::SearchSource::Episode { .. } => "episode",
                types::SearchSource::Projection { .. } => "projection",
            };
            let key = format!("{}:{}", source_type, fingerprint);
            seen.insert(key)
        })
        .collect();
    
    // Pass 2: document diversity — max 2 chunks per document_id
    let mut doc_counts: HashMap<String, usize> = HashMap::new();
    deduped.retain(|r| {
        if let types::SearchSource::Chunk { document_id, .. } = &r.source {
            let count = doc_counts.entry(document_id.clone()).or_insert(0);
            if *count >= 2 {
                return false;
            }
            *count += 1;
        }
        true
    });
    
    deduped
}
```

**Step 2:** Run `cargo test --all-features -- dedup`. Expected: PASS (existing dedup tests should still pass since they use different documents).

**Step 3:** Commit.

**Phase 5 Gate:** `cargo test --all-features` — all pass.

---

## Phase 6: Wire routing to actually execute tools (MEDIUM, HIGH ROI)

**Problem:** `sm_search_with_routing` classifies the query and generates a plan, but then calls plain `store.search()` regardless of the plan. The decoder, discord, and graph tools are never invoked.

### Task 6.1: Execute discord when routing plan calls for it

**Files:**
- Modify: `semantic-memory-mcp/src/server.rs` — `sm_search_with_routing`

**Step 1:** After the plain search returns results, check `decision.discord_enabled`. If true, call `sm_discord_search` with the result IDs and merge the discord scores into the result ranking.

**Step 2:** If `decision.decoder_enabled` and contradictions are provided, call the decoder on the results.

**Step 3:** Update the response JSON to accurately report `decoder_executed` and `discord_executed` based on what actually ran.

**Step 4:** Run `cargo test --all-features`. Expected: all pass.

**Step 5:** Commit.

### Task 6.2: Update the SM-AUD-007 comment

**Files:**
- Modify: `semantic-memory-mcp/src/server.rs`

**Step 1:** Remove the `SM-AUD-007` comment since the routing now actually executes.

**Step 2:** Commit.

**Phase 6 Gate:** `cargo test --all-features` — all pass. `cd semantic-memory-mcp && cargo check` — compiles.

---

## Phase 7: Search result caching (MEDIUM, HIGH ROI)

**Problem:** If the same query is searched twice (e.g., recall hook + primer hook), the full search pipeline runs both times including embedding.

### Task 7.1: Add search result LRU cache

**Files:**
- Modify: `src/lib.rs` — add a `search_cache` to `MemoryStoreInner` and check it in `search_with_context`

**Step 1:** Add a second LRU cache for search results:
```rust
search_cache: std::sync::Mutex<lru::LruCache<String, Vec<SearchResult>>>,
```
Initialize with `LruCache::new(NonZeroUsize::new(64).unwrap())` (64 entries, ~64KB).

**Step 2:** In `search_with_context`, before running the search, check the cache. Key = `format!("{}:{}:{}", query, top_k, namespaces_serialized)`. If hit, return cached results.

**Step 3:** After search completes, store results in cache.

**Step 4:** Write a test: call search twice with same query, verify second call returns same results (can check via timing or by mocking the embedder).

**Step 5:** Run `cargo test --all-features`. Expected: all pass.

**Step 6:** Commit.

**Phase 7 Gate:** `cargo test --all-features` — all pass.

---

## Phase 8: Embedding similarity dedup (MEDIUM, HIGH ROI)

**Problem:** The first-30-words fingerprint catches exact and near-exact duplicates but misses semantic near-duplicates (same passage with minor rewording from different document copies).

### Task 8.1: Add cosine similarity dedup after RRF

**Files:**
- Modify: `src/lib.rs` — `dedup_by_content` or a new `dedup_by_embedding_similarity` function

**Step 1:** Add a function that computes pairwise cosine similarity between result embeddings. If two results from the same source type have cosine > 0.95, drop the lower-scoring one.

**Step 2:** This requires loading embeddings for the top results. Add a batch embedding fetch in the dedup path.

**Step 3:** Write a test: two chunks with different first-30-words but same semantic content (cosine > 0.95), verify one is dropped.

**Step 4:** Run `cargo test --all-features`. Expected: all pass.

**Step 5:** Commit.

**Phase 8 Gate:** `cargo test --all-features` — all pass.

---

## Phase 9: Self-RAG retrieve-or-not gate (MEDIUM, HIGH ROI)

**Problem:** Every user message triggers the recall hook which runs a full search (~138ms). But simple greetings, confirmations, and single-word responses don't need retrieval.

### Task 9.1: Add retrieve-or-not classifier to recall hook

**Files:**
- Modify: `~/.hermes/agent-hooks/sm-recall.py`

**Step 1:** Add a pre-classification step that skips search for:
- Messages < 12 chars (already exists)
- Messages that are pure greetings/confirmations: "ok", "yes", "no", "thanks", "done", "sure", "yeah", "right", "correct", "agreed", "ok thanks", "got it"
- Messages starting with "can you", "could you", "would you" (these are instructions, not queries)
- Slash commands (already exists)

**Step 2:** For messages that pass the gate, proceed with the existing search flow.

**Step 3:** Test: send "ok" to the recall hook, verify it returns exit 0 without querying the HTTP server (check timing < 5ms).

**Step 4:** Commit.

### Task 9.2: Add retrieve-or-not gate to library search

**Files:**
- Modify: `src/search.rs` or `src/lib.rs`

**Step 1:** Add a `should_retrieve(query: &str) -> bool` function that classifies whether a query needs retrieval. Use simple heuristics: word count, presence of technical terms, question structure.

**Step 2:** Call this in `search_with_context` before the embedding step. If false, return empty results immediately.

**Step 3:** Write tests for the classifier.

**Step 4:** Run `cargo test --all-features`. Expected: all pass.

**Step 5:** Commit.

**Phase 9 Gate:** `cargo test --all-features` — all pass. Hook timing for "ok" < 5ms.

---

## Phase 10: Provenance-weighted scoring (MEDIUM, HIGH ROI)

**Problem:** Provenance confidence values exist (provenance feature is enabled) but are not used in search ranking. High-confidence facts should rank higher.

### Task 10.1: Add provenance confidence to search score

**Files:**
- Modify: `src/search.rs` — add provenance confidence to CandidateRow and RRF score
- Modify: `src/search.rs` — SQL query to join provenance table

**Step 1:** Add `provenance_confidence: Option<f64>` to `CandidateRow`.

**Step 2:** LEFT JOIN the provenance table in the BM25/vector candidate SQL queries to get confidence.

**Step 3:** In the RRF score computation, multiply by `1.0 + (confidence.unwrap_or(0.5) - 0.5) * 0.2` — this gives a ±10% score adjustment based on confidence.

**Step 4:** Write a test: add two facts with same content but different provenance, verify the higher-confidence one ranks higher.

**Step 5:** Run `cargo test --all-features`. Expected: all pass.

**Step 6:** Commit.

**Phase 10 Gate:** `cargo test --all-features` — all pass.

---

## Phase 11: Namespace-weighted scoring (MEDIUM, MEDIUM ROI)

**Problem:** All namespaces are scored equally. But "projects" namespace facts are often more relevant than "general" for coding queries.

### Task 11.1: Add namespace weights to SearchConfig

**Files:**
- Modify: `src/config.rs` — add `namespace_weights: HashMap<String, f64>` to `SearchConfig`
- Modify: `src/search.rs` — apply namespace weight in score computation

**Step 1:** Add `namespace_weights: HashMap<String, f64>` to `SearchConfig` with default empty (no weighting).

**Step 2:** In the RRF score, multiply by the namespace weight if one is configured for the result's namespace.

**Step 3:** Write a test: search with namespace_weights = {"projects": 1.5}, verify projects results rank higher.

**Step 4:** Run `cargo test --all-features`. Expected: all pass.

**Step 5:** Commit.

**Phase 11 Gate:** `cargo test --all-features` — all pass.

---

## Phase 12: Matryoshka multi-resolution search (HIGHER EFFORT, HIGH ROI)

**Problem:** The matryoshka module exists but isn't wired into search. Multi-resolution (64d candidate → 768d rerank) would cut vector search time significantly.

### Task 12.1: Wire matryoshka into vector search

**Files:**
- Modify: `src/search.rs` — vector search path
- Modify: `src/usearch_backend.rs` — add 64d truncated index option

**Step 1:** When matryoshka feature is enabled, create a second usearch index with 64d truncated embeddings (first 64 dims of each 768d vector).

**Step 2:** In search, first query the 64d index for candidates (fast, ~0.1ms), then rerank the top-N candidates using full 768d cosine similarity.

**Step 3:** Write a test: verify 64d candidate + 768d rerank produces same top-k as direct 768d search (within tolerance).

**Step 4:** Run `cargo test --all-features`. Expected: all pass.

**Step 5:** Commit.

**Phase 12 Gate:** `cargo test --all-features` — all pass. Benchmark: verify latency improvement.

---

## Phase 13: FTS5 BM25 parameter tuning (EASY, MEDIUM ROI)

**Problem:** FTS5 uses default BM25 parameters (k1=1.2, b=0.75). Technical content with repeated terms may benefit from lower k1.

### Task 13.1: Make BM25 parameters configurable

**Files:**
- Modify: `src/config.rs` — add `bm25_k1: f64` and `bm25_b: f64` to `SearchConfig`
- Modify: `src/search.rs` — use configured values in BM25 score computation

**Step 1:** Add `bm25_k1: f64` (default 1.2) and `bm25_b: f64` (default 0.75) to `SearchConfig`.

**Step 2:** Find where FTS5 bm25() is called in search.rs. FTS5's `bm25()` function accepts k1 and b as optional parameters. Pass the configured values.

**Step 3:** Write a test: verify different k1 values produce different rankings for a query with repeated terms.

**Step 4:** Run `cargo test --all-features`. Expected: all pass.

**Step 5:** Commit.

**Phase 13 Gate:** `cargo test --all-features` — all pass.

---

## Phase 14: Community-aware result grouping (HIGHER EFFORT, MEDIUM ROI)

**Problem:** Search results are a flat list. For synthesis queries, grouping by community helps the agent understand relationships.

### Task 14.1: Add optional community grouping to search response

**Files:**
- Modify: `semantic-memory-mcp/src/server.rs` — `sm_search` response

**Step 1:** After search results are assembled, optionally run community detection on the result set. Group results by community ID.

**Step 2:** Add a `grouped_results` field to the search response JSON when communities are detected.

**Step 3:** This is opt-in — only runs when a `group_by_community: true` parameter is passed.

**Step 4:** Commit.

**Phase 14 Gate:** `cargo test --all-features` — all pass.

---

## Phase 15: Switch to Ollama embedder for GPU acceleration (EASY, CRITICAL ROI)

**Problem:** Candle CPU embedder takes 138ms per query. Ollama with the same model (nomic-embed-text) does it in 33ms -- 4x faster. Ollama has ROCm support built in and can use the AMD Barcelo APU. This is a config change, not a code change.

**Verified:** Ollama `/api/embed` with nomic-embed-text returns 768d embeddings in 33ms (single) and 64ms (batch of 5). Same model, same dimensions, 4x faster.

### Task 15.1: Switch MCP config to Ollama embedder

**Files:**
- Modify: `~/.hermes/config.yaml` — change `--embedder candle` to `--embedder ollama`
- Modify: `~/.hermes/config.yaml` — add `--embedding-url http://127.0.0.1:11434`

**Step 1:** Change the MCP server args:
```yaml
mcp_servers:
  semantic_memory:
    args:
    - --memory-dir
    - /home/sikmindz/.hermes/semantic-memory.db
    - --embedder
    - ollama
    - --embedding-model
    - nomic-embed-text
    - --embedding-dims
    - '768'
    - --embedding-url
    - http://127.0.0.1:11434
    - --http-port
    - '1738'
    command: semantic-memory-mcp
    enabled: true
```

**Step 2:** Restart the MCP server (`/reload-mcp` or `/reset`).

**Step 3:** Verify embeddings work: `curl -s -X POST http://127.0.0.1:1738/search -d '{"query":"turbo-quant","top_k":5}'`

**Step 4:** Run benchmark: `python3 ~/.hermes/agent-hooks/run_benchmark_http.py ollama_gpu 1738`

Expected: avg latency < 50ms (down from 138ms).

**Step 5:** IMPORTANT: After switching embedder, existing embeddings in the DB were generated with Candle using `search_document:` prefix. Ollama's nomic-embed-text model adds its own prefix. Run `store.reembed_all()` to regenerate all embeddings with the Ollama backend. This is a one-time cost.

**Step 6:** Commit config change.

**Phase 15 Gate:** Benchmark avg latency < 50ms. Search results still return relevant content.

---

## Phase 16: Cross-encoder reranking via LLM (MEDIUM, HIGH ROI)

**Problem:** No cross-encoder reranker exists. RRF fusion gives approximate ranking. A reranker re-scores the top-N results against the query for precision.

**Research:** bge-reranker-v2-m3 is not available on Ollama registry. Two approaches:
- **LLM-based reranking:** Use a small LLM to score (query, document) pairs 1-5. Works with any Ollama model. ~50ms per pair.
- **Multi-model cosine reranking:** Embed both query and top-N documents with a second model (e.g., bge-m3 at 1024d) and rerank by cosine similarity. Faster but less precise than LLM reranking.

The LLM-based approach is more flexible and doesn't need a dedicated reranker model. Use it.

### Task 16.1: Add LLM reranker to HTTP server

**Files:**
- Modify: `semantic-memory-mcp/src/http_server.rs` — add `/rerank` endpoint
- Modify: `semantic-memory-mcp/src/http_server.rs` — add reranking to `/search` when `rerank: true` is passed

**Step 1:** Add a `/rerank` endpoint that takes a query and a list of (id, content) pairs, sends them to Ollama for scoring, and returns sorted results.

The Ollama prompt:
```
Rate the relevance of this document to the query on a scale of 1-5.
Query: {query}
Document: {content[:500]}
Rating (1-5):
```

**Step 2:** Add optional `rerank: true` parameter to `/search`. When set, after RRF returns top-N*2 candidates, rerank the top-N with the LLM.

**Step 3:** Use a small fast model for reranking (e.g., `qwen3.5:0.8b` or `granite4.1:3b`).

**Step 4:** Write a test: search with rerank=true, verify top results are more relevant than without rerank.

**Step 5:** Commit.

**Phase 16 Gate:** `cargo test --all-features` — all pass. Search with rerank returns better top-3 (higher cosine or more relevant content).

---

## Phase 17: Self-editing memory via MCP tools (MEDIUM, HIGH ROI)

**Problem:** MemGPT lets the LLM manage its own memory (edit, delete, promote). semantic-memory has `sm_supersede_fact` and `sm_delete_fact` but no `sm_update_fact` and no self-management guidance.

**Research:** The LLM needs three capabilities for self-editing memory:
1. **Update** a fact's content (already exists in library: `update_fact`, missing from MCP)
2. **Consolidate** -- merge two near-duplicate facts into one
3. **Self-management prompt** -- system prompt guidance telling the LLM when to edit its own memory

### Task 17.1: Add sm_update_fact MCP tool

**Files:**
- Modify: `semantic-memory-mcp/src/server.rs` — add `sm_update_fact` tool
- Modify: `semantic-memory-mcp/src/tools.rs` — add `UpdateFactParams`

**Step 1:** Add an `sm_update_fact` tool that calls `store.update_fact(fact_id, new_content)`. This re-embeds the fact and updates FTS.

**Step 2:** Add `sm_consolidate_facts` tool that takes two fact IDs, merges their content into one, supersedes the other. Calls `sm_supersede_fact` with merged content.

**Step 3:** Write tests for both tools.

**Step 4:** Run `cargo test --all-features`. Expected: all pass.

**Step 5:** Commit.

### Task 17.2: Add self-management system prompt guidance

**Files:**
- Modify: `~/.hermes/agent-hooks/sm-primer.py` — add self-editing instructions to the session-start context

**Step 1:** Add to the primer context:
```
- SELF-EDIT: You can manage your own memory. Use sm_update_fact to correct
  outdated facts, sm_supersede_fact to replace facts with newer versions,
  sm_consolidate_facts to merge near-duplicates, and sm_delete_fact for
  removal. Manage memory proactively when you notice stale or incorrect info.
```

**Step 2:** Test the hook output includes the new guidance.

**Step 3:** Commit.

**Phase 17 Gate:** `cargo test --all-features` — all pass. `hermes hooks doctor` — all pass.

---

## Phase 18: Entity extraction pipeline (MEDIUM, HIGH ROI)

**Problem:** semantic-memory stores facts but doesn't auto-extract entities from raw text. GraphRAG and Zep both do entity extraction. The `entity_aliases` table exists but is unused.

**Research:** Two approaches:
- **LLM-based extraction:** Send text to Ollama, ask for entities (name, type, relationships). Flexible, works with any model. ~100ms per text.
- **Rule-based extraction:** Regex/NER patterns for common entity types (paths, version numbers, project names, crate names). Faster but less flexible.

Use LLM-based for flexibility, with rule-based fallback for speed.

### Task 18.1: Add entity extraction on fact ingestion

**Files:**
- Modify: `semantic-memory-mcp/src/server.rs` — add entity extraction to `sm_add_fact`
- Modify: `semantic-memory-mcp/src/http_server.rs` — add entity extraction to `/add` endpoint

**Step 1:** After a fact is added, send its content to Ollama with a prompt:
```
Extract entities from this text as JSON. Format: {"entities": [{"name": "...", "type": "person|project|concept|tool|version|path", "aliases": ["..."]}]}
Text: {content}
```

**Step 2:** Store extracted entities in the `entity_aliases` table.

**Step 3:** Auto-create graph edges from the fact to extracted entities (Entity relation "mentions").

**Step 4:** Make this opt-in with an `extract_entities: true` parameter on `sm_add_fact`.

**Step 5:** Write tests with mock extraction.

**Step 6:** Commit.

### Task 18.2: Add entity-based search boost

**Files:**
- Modify: `src/search.rs` — boost results that match query entities

**Step 1:** Before search, extract entities from the query (rule-based: capitalized words, known project names, paths).

**Step 2:** Boost results whose content mentions those entities by 1.1x.

**Step 3:** Commit.

**Phase 18 Gate:** `cargo test --all-features` — all pass.

---

## Phase 19: Hierarchical community summaries (MEDIUM, MEDIUM ROI)

**Problem:** Community detection exists but doesn't produce summaries. GraphRAG generates hierarchical summaries at each community level. This enables "summarize what I know about topic X" queries.

### Task 19.1: Add community summary generation

**Files:**
- Modify: `semantic-memory-mcp/src/server.rs` — extend `sm_community` to optionally generate summaries
- Modify: `src/community.rs` — add `CommunitySummary` struct

**Step 1:** After community detection, for each community, collect all fact content in that community.

**Step 2:** Send the collected content to Ollama with a prompt:
```
Summarize the key knowledge in these facts as a concise paragraph.
Facts: {all_content[:2000]}
Summary:
```

**Step 3:** Store the summary in the community result. Add a `summary` field to the `Community` struct.

**Step 4:** Make this opt-in with a `summarize: true` parameter on `sm_community`.

**Step 5:** Write tests.

**Step 6:** Commit.

### Task 19.2: Add community summary search

**Files:**
- Modify: `semantic-memory-mcp/src/http_server.rs` — add `/community-search` endpoint

**Step 1:** Add an endpoint that takes a query, detects communities, and returns matching community summaries.

**Step 2:** This enables "what do I know about X?" queries that return summarized knowledge groups rather than individual facts.

**Step 3:** Commit.

**Phase 19 Gate:** `cargo test --all-features` — all pass.

---

## Phase 20: Production scale validation (HIGHER EFFORT, CRITICAL FOR TRUST)

**Problem:** Only tested with 6815 facts. Need to validate behavior at 100K+ facts and document limits.

### Task 20.1: Bulk ingestion stress test

**Files:**
- Create: `semantic-memory/tests/scale_test.rs`

**Step 1:** Write a test that ingests 10,000 synthetic facts with varied content, measuring:
- Ingestion throughput (facts/second)
- Search latency at 10K, 50K, 100K facts
- DB size growth
- Memory usage (RSS)

**Step 2:** Run the test: `cargo test --all-features -- scale_test -- --ignored`

**Step 3:** Record results in `docs/SCALE_TEST_RESULTS.md`.

**Step 4:** Commit.

### Task 20.2: Concurrent access stress test

**Files:**
- Create: `semantic-memory/tests/concurrent_test.rs`

**Step 1:** Write a test that spawns 10 threads, each doing search + add_fact concurrently for 100 iterations.

**Step 2:** Verify no "database is locked" errors, no data corruption.

**Step 3:** Record results.

**Step 4:** Commit.

### Task 20.3: Document production limits

**Files:**
- Create: `docs/PRODUCTION_LIMITS.md`

**Step 1:** Based on stress test results, document:
- Max recommended facts/documents/chunks
- Search latency at various scales
- Memory usage
- Concurrent writer limits
- Recommended SQLite PRAGMA tuning at scale

**Step 2:** Commit.

**Phase 20 Gate:** All stress tests pass. Production limits documented with evidence.

---

## Phase 21: Final validation and benchmarking

### Task 21.1: Run full test suite

```bash
cd /home/sikmindz/Coding/Libraries/semantic-memory && cargo test --all-features
```
Expected: all pass, 0 failures.

### Task 21.2: Rebuild and install MCP binary

```bash
cd /home/sikmindz/Coding/Libraries/semantic-memory-mcp && cargo build --release
cp target/release/semantic-memory-mcp ~/.cargo/bin/semantic-memory-mcp
cp target/release/semantic-memory-mcp /tmp/semantic-memory-mcp-bench
```

### Task 21.3: Run benchmark comparison

```bash
# Kill old server, start new one
kill $(lsof -t -i:1738) 2>/dev/null
/tmp/semantic-memory-mcp-bench --memory-dir ~/.hermes/semantic-memory.db --embedder candle --http-port 1738 --http-only &
sleep 3

# Run benchmark
python3 ~/.hermes/agent-hooks/run_benchmark_http.py final_optimized 1738
```

Expected: latency < 50ms avg (with Ollama GPU), avg top cosine > 0.70, duplicates = 0.

### Task 21.4: Commit final state

```bash
cd /home/sikmindz/Coding/Libraries/semantic-memory
git add -A
git commit -m "feat: wire temporal_weight, late-interaction, routing execution, result caching, query expansion, diversity dedup, provenance scoring, namespace weights, matryoshka, BM25 tuning"
```

---

## Summary

| Phase | Item | Effort | ROI | Dependencies |
|-------|------|--------|-----|--------------|
| 1 | temporal_weight in search | Easy | High | None |
| 2 | late-interaction weight + feature | Easy | High | None |
| 3 | Document embedding cache | Easy | High | None |
| 4 | Query expansion for hyphens | Easy | High | None |
| 5 | Result diversity (document dedup) | Easy | High | None |
| 6 | Wire routing to execute tools | Medium | High | None |
| 7 | Search result caching | Medium | High | Phase 3 (cache struct) |
| 8 | Embedding similarity dedup | Medium | High | None |
| 9 | Self-RAG retrieve-or-not gate | Medium | High | None |
| 10 | Provenance-weighted scoring | Medium | High | None |
| 11 | Namespace-weighted scoring | Medium | Medium | None |
| 12 | Matryoshka multi-resolution | Higher | High | None |
| 13 | FTS5 BM25 parameter tuning | Easy | Medium | None |
| 14 | Community-aware grouping | Higher | Medium | None |
| 15 | Ollama GPU embedder switch | Easy | Critical | None |
| 16 | LLM cross-encoder reranking | Medium | High | Phase 15 (Ollama) |
| 17 | Self-editing memory (MCP tools) | Medium | High | None |
| 18 | Entity extraction pipeline | Medium | High | Phase 15 (Ollama) |
| 19 | Hierarchical community summaries | Medium | Medium | Phase 15 (Ollama) |
| 20 | Production scale validation | Higher | Critical | All |
| 21 | Final validation + benchmark | Easy | — | All |

**Total estimated time:** 12-18 hours with AI assist.

**Execution order:**
- Phases 1-5: independent, any order (easy high-ROI)
- Phase 15: do early -- Ollama switch is a config change that makes everything faster
- Phase 7: depends on Phase 3
- Phases 16, 18, 19: depend on Phase 15 (need Ollama for LLM calls)
- Phases 6, 8-14, 17, 20: independent
- Phase 21: last

**Recommended sequence:** 15 → 1 → 2 → 3 → 4 → 5 → 7 → 6 → 8 → 9 → 10 → 11 → 13 → 12 → 16 → 17 → 18 → 19 → 14 → 20 → 21

**After completion, semantic-memory will have NO gaps vs any competitor:**
- GPU embedding (Ollama) -- matches all competitors
- Cross-encoder reranking (LLM-based) -- matches/exceeds RAG systems
- Self-editing memory (MCP tools) -- matches MemGPT
- Entity extraction (LLM-based) -- matches GraphRAG/Zep
- Hierarchical community summaries -- matches GraphRAG
- Production scale validation -- matches enterprise deployments
- Plus all existing advantages: provenance semiring, factor graph BP, persistent homology, contradiction detection, subgraph neighborhood loading, lawful subtraction, compression governor, 30 MCP tools, local-first