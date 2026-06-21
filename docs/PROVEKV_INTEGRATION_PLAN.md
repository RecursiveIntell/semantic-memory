# proveKV × semantic-memory integration plan

Status: planning artifact only; no implementation changes in this pass.
Created: 2026-06-05

## Executive decision

The next best path is NOT to force proveKV through semantic-memory's per-vector `VectorCodec` abstraction.

The identity boundary for that path is now explicit in `docs/PROVEKV_IDENTITY_MODEL.md`: semantic IDs, embedding IDs, derived vector generation IDs, rendered prompt segment IDs, token/KV IDs, and compression artifact IDs are separate layers. The code must not use semantic similarity as proof of KV reuse.

Implement proveKV as a generation-level derived candidate backend:

- semantic-memory remains authoritative for text, metadata, f32 embeddings, bitemporal state, and receipts.
- proveKV / poly-kv supplies a rebuildable compressed pool artifact over a deterministic snapshot of those embeddings.
- search uses the proveKV pool only for approximate/candidate generation, then exact-reranks from SQLite f32 embeddings before returning results.
- AgentShell / hot-shell support is deferred until static pool integration is stable.

Reason: proveKV's real efficiency win is batched, content-addressed, generation-level pooling (FB2/TQB1), while `VectorCodec::encode(vector)` is per-vector and loses the shared-pool compression model.

## Current repository state observed

### proveKV

Path: `/home/sikmindz/proveKV`
Branch: `main`
Head: `d6e08d281103c6283546e6233c51efd7d27c5c7d`
Remote: `https://github.com/RecursiveIntell/proveKV`
Status note: clean except pre-existing untracked `docs/INTEGRATION_TIER1_STACK_IDS_BOUNDARY_COMPILER.md`.

Relevant claim state:

- lossless: 36.00× vs f32 raw KV, 18.00× vs fp16-equivalent KV
- lossy: 68.04× vs f32 raw KV, 34.02× vs fp16-equivalent KV
- raw denominator: 2,315,255,808 B
- PPL-neutral validated N=8 default b=4 run

### semantic-memory

Path: `/home/sikmindz/Coding/Libraries/semantic-memory`
Branch: `master`
Head observed: `e1f06b28dade30b817b8a7cc23db256d7e7dec37`
Status note: dirty workspace with broader `/home/sikmindz/Coding/Libraries` changes. Do not assume existing untracked/modified pool-codec work belongs to this task.

Observed relevant files:

- `Cargo.toml`
- `src/config.rs`
- `src/db.rs`
- `src/lib.rs`
- `src/vector_codec.rs`
- `src/quantize_governed.rs`
- `src/pool_codec.rs` (untracked at observation time)
- `tests/pool_codec.rs` (untracked at observation time)

Verification already run by inspection worker:

- `cargo check --features poly-kv-pool --no-default-features`: failed because no search backend was enabled.
- `cargo check --features 'poly-kv-pool brute-force' --no-default-features`: passed with warnings only.

## Local API facts

### proveKV / poly-kv pool seam

Files:

- `/home/sikmindz/proveKV/proveKV/src/pool.rs`
- `/home/sikmindz/proveKV/proveKV/src/shell.rs`
- `/home/sikmindz/proveKV/proveKV/src/manifest.rs`
- `/home/sikmindz/proveKV/proveKV/src/receipt.rs`
- `/home/sikmindz/proveKV/proveKV/src/policy.rs`

Relevant APIs:

- `SharedKVPool::build(corpus, shape, seed)`
- `SharedKVPool::build_with_policy(corpus, shape, seed, policy)`
- `SharedKVPool::decompress_layer(layer_idx)`
- `SharedKVPool::decompress_all_layers_with_seed(seed)`
- `CompressionPolicy::default_two_tier()`
- `CompressionPolicy::default_two_tier_lossy()`
- `CODEC_FIB_K4_N32_BATCHED`
- `turbo_batched_codec_id(bits, lossy)`
- `PoolBuildReceipt`, `PoolManifest`, `ShellManifest`, `ShellMaterializeReceipt`

proveKV input contract:

- `corpus: &[(String, Vec<f32>)]`
- each vector is flattened KV data with length:
  `num_layers * num_kv_heads * head_dim * 2`
- semantic-memory embedding mapping should be:
  - `num_layers = 1`
  - `num_heads = 1`
  - `num_kv_heads = 1`
  - `head_dim = embedding_dim`
  - `hidden_size = embedding_dim`
  - `kv = embedding || embedding`

### semantic-memory current seam

Files:

- `src/vector_codec.rs`: object-safe per-vector `VectorCodec` trait and `VectorArtifactV1`.
- `src/config.rs`: `DerivedVectorBackendPolicy` currently has only `Disabled` and `TurboQuantCandidateOnly`.
- `src/db.rs`: V19/V21/V23 derived vector artifact tables and generation manifests.
- `src/lib.rs`: `rebuild_vector_artifacts()` is currently TurboQuant-specific.
- `src/pool_codec.rs`: existing pool adapter builds a `SharedKVPool` from embeddings but should be treated as experimental/pre-existing.

Important mismatch:

- semantic-memory currently imports `poly_kv_core::{...}` from package `poly-kv` at `../poly-kv`.
- proveKV crate path is `/home/sikmindz/proveKV/proveKV`, package/lib name `provekv`.
- Therefore semantic-memory's current `poly-kv-pool` code is not directly wired to the proveKV repository unless dependency naming/pathing is intentionally aligned.

## Architecture

### Ownership model

Authoritative:

- SQLite rows in semantic-memory: facts, chunks, messages, episodes, embeddings, temporal/scope metadata.
- f32 embedding bytes remain the scoring source of truth.

Derived/rebuildable:

- usearch/HNSW sidecars.
- TurboQuant derived vector artifacts.
- proveKV shared pool artifacts.
- future shells / prompt KV artifacts.

Rule: compressed artifacts must never become the only copy of semantic-memory embeddings.

### Candidate backend model

New backend name:

- policy enum variant: `ProveKvPoolCandidateOnly` or `SharedKvPoolCandidateOnly`
- receipt candidate backend string: `provekv_shared_kv_pool`
- codec family: `provekv_shared_kv_pool`
- encoding: `provekv_fb2_pool_v1`

Search flow:

1. Embed query using semantic-memory's configured embedder.
2. Load active proveKV pool generation for embedding dimension/profile.
3. Decode pool vectors in generation order.
4. Score approximate candidates against decoded pool vectors.
5. Select `candidate_pool_size` candidates.
6. Load authoritative f32 embeddings for those item keys from SQLite.
7. Exact-rerank with f32 cosine/IP.
8. Fuse with BM25/RRF as existing pipeline does.
9. Persist receipt with generation ID, pool manifest digest, profile digest, approximate counts, exact rerank count, and fallback status.

Fallback policy:

- If no active pool generation exists: fall back to raw f32/usearch/brute-force path and record `fallback = "missing_provekv_pool_generation"`.
- If pool decode fails: fall back to raw f32 and record decode error count, not silent success.
- If generation is stale/invalidated: fall back or force rebuild depending on explicit caller action; search should not rebuild implicitly unless that is already semantic-memory policy.

### Storage model

Do not store the full pool once per item.

Add generation-level storage plus item mapping:

```sql
CREATE TABLE derived_vector_pool_generations (
    generation_id              TEXT PRIMARY KEY,
    schema_version             TEXT NOT NULL,
    codec_family               TEXT NOT NULL,
    codec_profile_digest       TEXT NOT NULL,
    source_snapshot_digest     TEXT NOT NULL,
    source_row_count           INTEGER NOT NULL,
    artifact_count             INTEGER NOT NULL,
    dim                        INTEGER NOT NULL,
    encoding                   TEXT NOT NULL,
    pool_manifest_json         TEXT NOT NULL,
    pool_build_receipt_json    TEXT NOT NULL,
    pool_envelope              BLOB NOT NULL,
    pool_envelope_digest       TEXT NOT NULL,
    artifact_manifest_digest   TEXT NOT NULL,
    created_at                 TEXT NOT NULL,
    status                     TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'invalidated', 'failed')),
    degradations_json          TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE derived_vector_pool_items (
    generation_id              TEXT NOT NULL REFERENCES derived_vector_pool_generations(generation_id) ON DELETE CASCADE,
    item_key                   TEXT NOT NULL,
    token_index                INTEGER NOT NULL,
    source_embedding_digest    TEXT NOT NULL,
    status                     TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (generation_id, item_key),
    UNIQUE (generation_id, token_index)
);
```

Alternative if minimizing migrations:

- use `derived_vector_artifact_generations` for generation metadata;
- add only one new `derived_vector_pool_payloads` table keyed by `generation_id`;
- store item mapping in a new `derived_vector_pool_items` table.

Avoid:

- `item_key = "__generation_pool__:<id>"` sentinel rows inside per-item table unless short-term spike only.

### Pool envelope

proveKV `SharedKVPool` / `PoolLayer` do not currently expose a clean whole-pool Serialize/Deserialize seam. Add one of these before production integration:

Option A: expose serde derives on proveKV pool structs.

Option B: define a stable `ProveKvPoolEnvelopeV1` in semantic-memory or adapter crate:

```rust
struct ProveKvPoolEnvelopeV1 {
    schema_version: String,
    manifest: PoolManifest,
    build_receipt: PoolBuildReceipt,
    token_order: Vec<String>,
    layers: Vec<PoolLayerEnvelopeV1>,
}
```

Option B is safer for compatibility because semantic-memory controls what it persists.

### Dependency model

Decide before coding:

1. If `../poly-kv` is canonical for Libraries, integrate that and document that proveKV repo is upstream/proof repo.
2. If `/home/sikmindz/proveKV/proveKV` is canonical, change semantic-memory dependency to `provekv` or create a compatibility crate/export that exposes the expected `poly_kv_core` surface.

Do not leave semantic-memory importing `poly_kv_core` while claiming direct proveKV integration unless the dependency path proves it.

## Implementation phases

### Phase 0 — dependency and API identity decision

Goal: remove ambiguity between `poly-kv` and `provekv`.

Tasks:

1. Inspect `../poly-kv` vs `/home/sikmindz/proveKV/proveKV` API drift.
2. Decide canonical crate for semantic-memory integration.
3. Update/record Cargo dependency identity.
4. Add a short `docs/PROVEKV_DEPENDENCY_DECISION.md` or section in this plan.

Files likely touched:

- `semantic-memory/Cargo.toml`
- workspace `Cargo.toml` if using a workspace path dependency
- possible compatibility shim crate if required

Gate:

- `cargo check -p semantic-memory --no-default-features --features 'brute-force poly-kv-pool'`
- grep proves the integration path points to the chosen crate.

### Phase 1 — proveKV pool envelope spike

Goal: build, persist, reload, and decode a generation-level pool from semantic-memory-style embeddings without touching search.

Tasks:

1. Add adapter module, e.g. `src/provekv_pool.rs` or rename `src/pool_codec.rs` into a generation-level module.
2. Implement:
   - `build_pool_from_embeddings(dim, rows, seed, policy)`
   - `decode_all_embeddings(envelope)`
   - `decode_embedding_at(envelope, token_index)`
   - `profile_digest(policy, shape, seed, dim)`
3. Use deterministic ordering by `item_key ASC`.
4. Store token order and source embedding digests.
5. Add round-trip tests with dimensions 64 and 384.

Files likely touched:

- `src/provekv_pool.rs` or `src/pool_codec.rs`
- `src/lib.rs`
- `tests/provekv_pool.rs`

Gates:

- decoded vector count equals source row count.
- min cosine threshold is honest and measured; do not overclaim bit-exactness.
- envelope digest stable across repeated builds with same input/order/seed.

### Phase 2 — database migrations for generation-level pools

Goal: persist pool envelope and token mapping as first-class derived artifacts.

Tasks:

1. Add migration V24 (or next available) for `derived_vector_pool_generations` and `derived_vector_pool_items`.
2. Implement upsert/load/invalidate helpers.
3. Generalize existing derived artifact gating so pool persistence does not depend on `turbo-quant-codec`.
4. Invalidate active pool generation when an authoritative embedding row changes.

Files likely touched:

- `src/db.rs`
- `src/types.rs`
- `src/lib.rs`
- tests under `tests/`

Gates:

- fresh DB migrates to new schema.
- old DB migrates idempotently.
- insert/update/delete of facts/chunks/messages/episodes invalidates active pool generation.
- integrity check catches missing mapping rows, stale source digests, corrupt envelope digest.

### Phase 3 — rebuild API and receipts

Goal: expose a public rebuild path that produces receipt-backed pool generations.

Tasks:

1. Add config:
   - `DerivedVectorBackendPolicy::ProveKvPoolCandidateOnly`
   - seed/policy fields if not using defaults.
2. Add public API:
   - `MemoryStore::rebuild_provekv_pool_artifacts()` or extend `rebuild_vector_artifacts()` to dispatch by policy.
3. Emit `VectorArtifactBuildReceiptV1` plus pool-specific receipt fields or a new `PoolArtifactBuildReceiptV1`.
4. Store proveKV `PoolBuildReceipt` inside the pool generation row.

Files likely touched:

- `src/config.rs`
- `src/types.rs`
- `src/lib.rs`
- `src/db.rs`
- README feature docs

Gates:

- rebuild from empty corpus returns a controlled no-op/degradation, not a panic.
- rebuild from invalid embedding skips row and records degradation.
- build receipt includes generation ID, profile digest, source snapshot digest, pool envelope digest, source row count, artifact count, elapsed ms.

### Phase 4 — search candidate backend

Goal: use proveKV decoded pool vectors for candidate generation with exact f32 rerank.

Tasks:

1. Add search dispatch for `ProveKvPoolCandidateOnly`.
2. Load active generation.
3. Decode pool once per search initially; later cache decoded generation in memory with generation ID guard.
4. Score query against decoded vectors.
5. Return candidate item keys to existing exact-rerank path.
6. Populate `VectorSearchReceiptV1` fields:
   - `candidate_backend = "provekv_shared_kv_pool"`
   - `codec_family = Some("provekv_shared_kv_pool")`
   - `codec_profile_digest`
   - `artifact_profile_digest`
   - `vector_artifact_manifest_digest`
   - `artifact_generation_id`
   - `approximate_scanned_count`
   - `approximate_returned_count`
   - `raw_rows_loaded_count`
   - `exact_rerank = true`
   - `fallback` / `degradations` when applicable.

Files likely touched:

- `src/search.rs`
- `src/lib.rs`
- `src/types.rs`
- `src/db.rs`
- search tests

Gates:

- exact f32 rerank is mandatory; config validation rejects disabling it for proveKV pool backend.
- search with no pool falls back with receipt.
- search with corrupt pool falls back with receipt.
- result ordering matches f32 rerank for returned top-k.

### Phase 5 — benchmark and evidence

Goal: measure whether proveKV pool helps semantic-memory in practice.

Benchmarks:

1. Build time by N and dim: 384, 768, 1024.
2. Pool envelope bytes vs raw f32 embedding bytes.
3. Decode-all latency.
4. Search latency p50/p95/p99 with:
   - brute-force f32
   - usearch sidecar
   - TurboQuant derived artifacts
   - proveKV pool decoded candidates
5. Retrieval quality:
   - recall@10 against f32 exact baseline
   - MRR/NDCG if labeled data exists
   - top-k overlap vs f32

Expected early outcome:

- proveKV may win on persisted/cold artifact size, receiptability, and cross-agent prompt/KV path alignment.
- It may not beat usearch p99 until decoded generation caching or direct compressed scoring exists.
- Do not claim speedups until measured.

Gates:

- benchmark receipts checked in or generated under `target/` with exact commands.
- README claims are derived from receipts, not hand-written.

### Phase 6 — optional hot overlays / AgentShell

Only after Phases 1–5 pass.

Possible use cases:

- per-session memory overlays that contain just newly-added messages since the last pool rebuild.
- multi-agent shared memory: common long-term pool + per-agent short-term shell.
- prompt/KV path: semantic-memory selects memory, renderer creates deterministic segments, proveKV materializes exact model/token KV blocks.

Do not start here. Shells are a lifecycle problem, not a first integration primitive.

## External research implications

### KV compression

- KIVI: asymmetric 2-bit KV quantization; keys and values need separate treatment.
  URL: https://arxiv.org/abs/2402.02750
- KVQuant: outlier-aware KV cache quantization for long context.
  URL: https://arxiv.org/abs/2401.18079
- CacheGen: compress/stream KV cache to improve TTFT and serving reuse.
  URL: https://arxiv.org/abs/2310.07240
- H2O: keep heavy-hitter/recent tokens.
  URL: https://arxiv.org/abs/2306.14048
- StreamingLLM: attention sinks + recent window.
  URL: https://arxiv.org/abs/2309.17453
- SnapKV: prompt KV selection/compression.
  URL: https://arxiv.org/abs/2404.14469

Implication: keep KV artifact receipts model/token/layout-specific and separate from semantic vector artifact receipts.

### Prefix/KV reuse

- vLLM PagedAttention and automatic prefix caching.
  URL: https://arxiv.org/abs/2309.06180
  URL: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- SGLang RadixAttention.
  URL: https://www.lmsys.org/blog/2024-01-17-sglang/
- Provider prompt caching:
  - OpenAI: https://platform.openai.com/docs/guides/prompt-caching
  - Anthropic: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
  - Vertex: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview

Implication: semantic identity and KV identity must be separate. Semantic similarity can propose context, but KV reuse requires exact token/model/config identity.

### Semantic cache / memory hierarchy

- GPTCache: semantic response cache.
  URL: https://github.com/zilliztech/GPTCache
- RedisVL semantic cache.
  URL: https://redis.io/docs/latest/develop/ai/redisvl/0.9.1/user_guide/llmcache/
- LangChain LLM caching.
  URL: https://python.langchain.com/docs/integrations/llm_caching/
- MemGPT / Letta.
  URL: https://arxiv.org/abs/2310.08560
  URL: https://github.com/letta-ai/letta
- Mem0 memory layer.
  URL: https://mem0.ai/research

Implication: semantic-memory should own extraction/retrieval/consolidation; proveKV should own exact compressed artifacts after deterministic rendering/tokenization.

### Compressed vector retrieval

- FAISS indexes / IVF-PQ.
  URL: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- ScaNN anisotropic vector quantization.
  URL: https://research.google/blog/announcing-scann-efficient-vector-similarity-search/
- DiskANN.
  URL: https://github.com/microsoft/DiskANN
- USearch.
  URL: https://github.com/unum-cloud/usearch

Implication: semantic-memory's authoritative f32 + rebuildable derived artifact model is correct. proveKV pool should be one more derived candidate backend, not a replacement for f32 or usearch.

## Risks and mitigations

1. Dependency identity drift
   - Risk: semantic-memory claims proveKV integration while using `../poly-kv`.
   - Mitigation: Phase 0 dependency decision and grep gate.

2. Wrong abstraction
   - Risk: using per-vector `VectorCodec` makes proveKV look weak or wire-mismatched.
   - Mitigation: generation-level pool backend only.

3. Pool persistence gap
   - Risk: `SharedKVPool` lacks stable serde envelope.
   - Mitigation: define `ProveKvPoolEnvelopeV1` and digest it.

4. Search latency regression
   - Risk: decoding all pool vectors per query is slower than usearch.
   - Mitigation: first measure; then add generation decode cache or direct compressed scoring.

5. Quality regression
   - Risk: decoded pool vectors have lower semantic retrieval recall.
   - Mitigation: exact f32 rerank mandatory; measure recall@k/top-k overlap.

6. Invalid stale artifacts
   - Risk: authoritative embedding changes but pool generation remains active.
   - Mitigation: source digest mapping + invalidation on all embedding mutations + integrity gate.

7. Claim drift
   - Risk: proveKV headline ratios leak into semantic-memory without semantic-memory evidence.
   - Mitigation: semantic-memory README only claims measured semantic-memory ratios/latencies.

## Best immediate next step

Run Phase 0 and Phase 1 as a spike:

1. Decide whether semantic-memory should depend on `../poly-kv` or `/home/sikmindz/proveKV/proveKV`.
2. Implement a tiny generation-level adapter around `SharedKVPool` with deterministic `item_key` ordering.
3. Add a test that builds a pool from 384-dim mock embeddings, decodes all vectors, verifies item order, reports bytes/raw ratio, and checks cosine similarity.
4. Do not touch search until the pool envelope and persistence story is clean.

Target first PR/commit title:

`spike(semantic-memory): add proveKV pool generation adapter`

Non-goal for first PR:

- no AgentShell
- no README performance claims
- no default feature changes
- no replacing usearch
- no direct prompt/KV runtime integration
