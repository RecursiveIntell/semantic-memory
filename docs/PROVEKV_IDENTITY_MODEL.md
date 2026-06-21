# proveKV × semantic-memory identity model

Status: implementation guardrail for the proveKV / poly-kv candidate-backend integration.
Created: 2026-06-05

## Executive rule

Do not collapse semantic similarity, embedding identity, rendered prompt identity, token/KV identity, and compression-artifact identity into one ID.

semantic-memory may use semantic similarity to find memories. proveKV may reuse or compare compressed KV/pool artifacts only when the exact source identity for that artifact matches. Same meaning is not enough for KV reuse. Same text is not enough unless the renderer, tokenizer, model, position layout, and codec profile also match.

## Identity layers

### 1. Semantic identity

Purpose: durable user/domain memory.

Inputs:
- namespace
- fact/document/chunk/message/episode primary key
- normalized content used by semantic-memory
- bitemporal valid/transaction times
- metadata and source type

Allowed use:
- BM25 / FTS retrieval
- vector retrieval candidate generation
- graph traversal
- deciding which memory records should enter a prompt

Forbidden use:
- proving exact prompt/KV reuse
- proving codec compatibility
- proving token-position compatibility

### 2. Embedding identity

Purpose: identify the authoritative f32 embedding stored by semantic-memory.

Inputs:
- semantic record ID
- embedding model name
- embedding dimensions
- embedding provider/config digest
- normalized embedding bytes digest, or row-generation digest when the bytes are loaded from SQLite

Allowed use:
- exact f32 rerank
- deterministic generation snapshots
- rebuilding derived candidate artifacts

Forbidden use:
- claiming prompt/KV cache equivalence

### 3. Derived vector generation identity

Purpose: identify a rebuildable candidate-search artifact over an embedding snapshot.

Inputs:
- ordered list of `(item_id, embedding_digest)` pairs
- embedding dimension
- candidate backend family (`turbo_quant`, `provekv_pool`)
- codec profile/version/seed
- source DB snapshot marker or generation counter

Allowed use:
- approximate candidate generation
- receipt-backed fallback/degradation reporting
- artifact rebuild / stale-generation detection

Required guard:
- exact f32 rerank remains mandatory. A derived vector generation is never authoritative.

### 4. Rendered prompt segment identity

Purpose: identify exact bytes sent to a tokenizer for prompt construction.

Inputs:
- renderer version
- segment type (`system`, `tool_schema`, `memory_prelude`, `retrieved_doc`, `user_turn`, `agent_tail`)
- ordered semantic-memory source IDs
- rendered UTF-8 bytes digest
- separator/template policy digest

Allowed use:
- prompt assembly receipts
- segment-level prefix-cache lookup after tokenization

Forbidden use:
- semantic equivalence. Two segments with the same meaning but different bytes are different prompt segments.

### 5. Token/KV identity

Purpose: prove exact compatibility for prompt-prefix/KV reuse.

Inputs:
- rendered prompt segment digest
- tokenizer name/version/digest
- model ID/revision/config digest
- RoPE/position policy digest
- layer count, KV heads, head dimension, attention layout
- token IDs and token range
- dtype/layout used to materialize the cache

Allowed use:
- exact prefix/KV reuse
- proving a compressed KV artifact can be applied to this prompt/model position

Forbidden use:
- matching by semantic similarity

### 6. Compression artifact identity

Purpose: identify bytes emitted by proveKV / poly-kv / fib-quant / turbo-quant.

Inputs:
- token/KV identity for prompt KV artifacts, or derived vector generation identity for embedding-pool artifacts
- codec family/version
- key codec and value codec where applicable
- bits/group/profile (`fib_k4_n32`, `turbo_4bit_batched`, etc.)
- radii codec / outlier policy / sink-token or recent-window policy when applicable
- compressed byte digest
- manifest digest
- receipt digest

Allowed use:
- decompression/rebuild checks
- benchmark and claim receipts
- candidate artifact staleness checks

## Current semantic-memory implementation boundary

Implemented first slice:
- `DerivedVectorBackendPolicy::ProveKvPoolCandidateOnly` is a configuration-visible derived backend.
- Config validation requires exact f32 rerank for every derived backend, including proveKV pool.
- Search receipts can label a requested proveKV pool candidate path and fall back to authoritative f32 until a materialized pool generation is present.
- `PoolCodec` builds a single generation-level `SharedKVPool` over an ordered embedding corpus for adapter tests and future materialization work.

Intentionally not implemented in this slice:
- exact prompt/KV prefix reuse
- AgentShell runtime hot overlays
- using semantic similarity as evidence for KV reuse
- replacing SQLite f32 embeddings or usearch with proveKV

## Required receipt chain

A full future prompt run should chain IDs like this:

```text
semantic_memory_record_id
  -> embedding_identity
  -> derived_vector_generation_identity
  -> candidate_search_receipt
  -> exact_f32_rerank_receipt
  -> rendered_prompt_segment_identity
  -> token_kv_identity
  -> provekv_compression_artifact_identity
```

For the current embedding-pool slice, the chain stops at exact f32 rerank. That is correct. Prompt/KV reuse starts only after deterministic rendering and tokenization exist.

## Research consequences baked into the model

- vLLM/SGLang/prompt-cache systems imply exact token/model/position identity for KV reuse.
- semantic cache systems imply semantic similarity is useful for retrieval, not for KV equivalence.
- FAISS/ScaNN/DiskANN/usearch imply compressed/index artifacts are derived and should rerank from exact vectors when quality matters.
- KIVI/KVQuant/CacheGen/H2O/StreamingLLM/SnapKV imply KV compression receipts must record key/value codec details, outliers, sink/recent policies, and token ranges when those features are added.

## Implementation gates

1. Any derived vector candidate backend must fail config validation if exact f32 rerank is disabled.
2. A proveKV pool generation must record item ordering; otherwise decoded vectors cannot be mapped back to source rows safely.
3. Prompt/KV reuse code must require rendered bytes + tokenizer + model/config + position/layout identity, not semantic IDs alone.
4. Public compression-ratio claims for semantic-memory must be measured on semantic-memory vector artifacts, not copied from proveKV PPL/KV-cache receipts.
