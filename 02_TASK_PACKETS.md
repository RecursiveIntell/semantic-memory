# Codex Task Packets

Each packet is designed to be pasted into a separate Codex task or PR. Run them in order unless the dependency graph says they can be parallelized.

---

## Packet 00 — Baseline, AGENTS, and audit ledger

### Prompt

You are preparing the semantic-memory workspace for a gated stabilization and TurboQuant-readiness program.

Do not change product behavior in this pass except for repo hygiene needed to support later passes.

Tasks:

1. Inspect workspace layout and existing `AGENTS.md` files.
2. Add/update a root or package-local `AGENTS.md` using the rules in `05_DROP_IN_AGENTS.md`.
3. Create:

```text
docs/audits/codex-giga-pass-20260511.md
docs/audits/codex-giga-pass-20260511-status.json
```

4. Record starting commit, Rust/Cargo versions, commands run, and baseline failures.
5. Do not mark anything fixed unless you add evidence.

### Likely files

```text
AGENTS.md
docs/audits/codex-giga-pass-20260511.md
docs/audits/codex-giga-pass-20260511-status.json
```

### Commands

```bash
cargo --version || true
rustc --version || true
cargo fmt --all --check || true
cargo check --workspace --all-targets --all-features || true
cargo test --workspace --all-features || true
```

### Acceptance

- Ledger exists.
- Baseline command output is recorded honestly.
- `AGENTS.md` contains non-negotiable invariants.
- No unrelated code changes.

---

## Packet 01 — Central vector validation

### Prompt

Implement central vector/embedding validation so invalid vectors cannot reach SQLite, q8, HNSW, projection imports, or future codec paths.

Tasks:

1. Find all embedding/vector write/index paths.
2. Add or consolidate validation functions:

```rust
validate_embedding(values: &[f32], expected_dim: usize)
validate_embedding_batch(...)
validate_vector_blob_len(bytes: &[u8], expected_dim: usize)
validate_f32_blob_le(bytes: &[u8], expected_dim: usize)
```

3. Validation must reject:
   - wrong dimension,
   - zero dimension when invalid,
   - NaN,
   - +/- infinity,
   - malformed byte lengths,
   - native-endian/alignment-dependent blob decode.
4. Route all storage/indexing paths through the validation layer.
5. Add regression tests.

### Likely files

```text
semantic-memory/src/types.rs
semantic-memory/src/db.rs
semantic-memory/src/lib.rs
semantic-memory/src/storage.rs
semantic-memory/src/hnsw.rs
semantic-memory/src/hnsw_ops.rs
semantic-memory/src/quantize.rs
semantic-memory/src/projection_import.rs
semantic-memory/tests/vector_invariants.rs
semantic-memory/tests/hardening_semantics.rs
```

### Tests to add

- NaN embedding rejected before storage.
- Infinity embedding rejected before HNSW insert.
- Wrong dimension rejected before q8.
- Malformed little-endian blob rejected.
- Valid vector roundtrips exactly through LE bytes.

### Acceptance

- All vector write/index paths call central validation or justify why not.
- Tests fail before fix and pass after fix.
- No `bytemuck`/native-cast persistence for f32 vectors unless explicitly proven safe and endian-stable.

---

## Packet 02 — HNSW key-level parity

### Prompt

Replace count-only HNSW integrity checks with key-level parity between SQLite live embedding rows, HNSW keymap rows, and typed retrieval keys.

Tasks:

1. Define a structured key codec for HNSW keys.
2. For every live embedded row, compute the expected key.
3. Verify expected key exists in active keymap.
4. For every active keymap row, parse key and verify the corresponding live SQLite row exists with a valid embedding.
5. Treat malformed keys, wrong domain prefixes, stale IDs, duplicate active keys, and swapped IDs as integrity failures.
6. Add full/incremental integrity modes if useful.

### Likely files

```text
semantic-memory/src/hnsw.rs
semantic-memory/src/hnsw_ops.rs
semantic-memory/src/lib.rs
semantic-memory/tests/hnsw_integration.rs
semantic-memory/tests/vector_only_hnsw.rs
semantic-memory/tests/episode_identity.rs
```

### Tests to add

- Missing keymap row detected.
- Stale active key detected after source row deletion.
- Swapped key IDs detected even when counts match.
- Wrong domain prefix detected.
- Duplicate active key detected.
- Episode key maps to episode row, not document container.

### Acceptance

- Count equality alone is never the final integrity proof.
- Integrity failures produce clear diagnostics.
- HNSW remains rebuildable from SQLite.

---

## Packet 03 — HNSW fixed-width sidecar header and rebuild-on-suspicion

### Prompt

Make HNSW sidecar persistence explicit, portable, and safe. Remove persisted `usize` from sidecar formats. Add versioned header validation or a conservative rebuild-on-suspicion policy.

Tasks:

1. Define a fixed-width sidecar header.
2. Store dimension and vector count as `u32`/`u64`, not `usize`.
3. Include magic/version/header length.
4. Include enough profile information to reject incompatible sidecars.
5. If old sidecar format is detected, rebuild safely or migrate explicitly.
6. If graph/data/keymap validation fails, mark HNSW degraded and rebuild before trusting search.

### Suggested header

```text
magic:        u32
version:      u16
header_len:   u16
dim:          u32
vector_count: u64
profile_hash: [u8; 32] or optional digest
```

### Likely files

```text
semantic-memory/src/hnsw.rs
semantic-memory/src/hnsw_ops.rs
semantic-memory/tests/hnsw_persistence.rs
semantic-memory/tests/hnsw_hotswap.rs
```

### Tests to add

- Current sidecar roundtrip.
- Old/unsupported header rejected or rebuilt.
- Wrong dimension sidecar rejected.
- Truncated header rejected.
- Empty graph sidecar does not pass as valid.
- Unsupported version fails cleanly.

### Acceptance

- No persisted `usize` in HNSW sidecar format.
- Sidecar validation is semantic, not just non-empty-file checking.
- Suspicious sidecar cannot silently participate in search.

---

## Packet 04 — Pending HNSW mutation failure recovery

### Prompt

Harden HNSW pending-op journaling so failed sidecar/keymap flushes cannot leave the live in-memory index trusted in an ambiguous state.

Tasks:

1. Inspect current pending operation lifecycle.
2. If an HNSW mutation is applied in memory before durable save succeeds, define safe recovery.
3. Prefer simplest safe behavior: on save/keymap failure after mutation, mark HNSW dirty/degraded and force rebuild from SQLite before future HNSW search.
4. Add generation tracking if needed.
5. Ensure errors are observable.

### Likely files

```text
semantic-memory/src/hnsw_ops.rs
semantic-memory/src/hnsw.rs
semantic-memory/src/lib.rs
semantic-memory/tests/hnsw_hotswap.rs
semantic-memory/tests/hnsw_persistence.rs
```

### Tests to add

- Simulated sidecar save failure marks index degraded.
- Future search does not trust degraded in-memory HNSW.
- Rebuild clears degraded state.
- Pending ops replay idempotently.

### Acceptance

- No path silently trusts HNSW after a failed durable save.
- Rebuild-from-SQLite remains the escape hatch.

---

## Packet 05 — Filtered HNSW under-return fallback

### Prompt

Fix HNSW filtered search so sparse namespace/source/session filters cannot under-return while valid rows exist outside the global candidate pool.

Tasks:

1. Inspect search flow for HNSW candidate generation followed by filters.
2. Detect when post-filter vector hits are less than requested `top_k` while scope has enough valid rows.
3. Add deterministic fallback:
   - adaptive overfetch, or
   - brute-force exact vector search within scope.
4. Record fallback/degradation in explained search output and future receipt skeleton.
5. Add sparse namespace/source tests.

### Likely files

```text
semantic-memory/src/search.rs
semantic-memory/src/lib.rs
semantic-memory/tests/search_tests.rs
semantic-memory/tests/brute_force_parity.rs
semantic-memory/tests/vector_only_hnsw.rs
```

### Tests to add

- Namespace-sparse corpus where global HNSW top candidates are filtered out.
- Fallback returns same top-k as brute force within namespace.
- Fallback/degradation flag appears in explained output.
- Source/session filters covered if supported.

### Acceptance

- Filtered HNSW never silently under-returns when exact scoped fallback can find valid results.
- The user/debug output can tell fallback occurred.

---

## Packet 06 — SearchContext and replay-clean recency

### Prompt

Make search deterministic under an explicit `SearchContext`. Remove hidden wall-clock dependence from internal ranking.

Tasks:

1. Add `SearchContext` or equivalent:

```rust
pub struct SearchContext {
    pub evaluation_time: DateTime<Utc>,
    pub receipt_mode: ReceiptMode,
    pub exactness_profile: ExactnessProfile,
}
```

2. Capture `Utc::now()` only at public API boundaries.
3. Internal scoring uses `context.evaluation_time`.
4. Tests freeze time and verify stable rankings.
5. Existing APIs may keep convenience methods by creating a default context at boundary.

### Likely files

```text
semantic-memory/src/search.rs
semantic-memory/src/lib.rs
semantic-memory/tests/search_tests.rs
```

### Tests to add

- Same query/context produces same order.
- Different evaluation time changes recency only predictably.
- No internal repeated `Utc::now()` calls in ranking.

### Acceptance

- Replay can reproduce search ranking given same context.
- Wall clock is not sampled deep inside ranking logic.

---

## Packet 07 — Search receipt skeleton

### Prompt

Add a minimal search receipt/explanation object that records backend, candidate counts, filters, fallback, approximate/exact scoring, rerank state, result IDs, and evaluation time.

Tasks:

1. Inspect existing `ExplainedResult`/score breakdown types.
2. Add a durable or serializable receipt-like struct, e.g. `VectorSearchReceiptV1`.
3. Receipt must not become authoritative truth.
4. Add optional receipt mode so normal search need not always persist receipts.
5. Ensure fallback/degradation is included.

### Minimum fields

```text
receipt_id or transient id
evaluation_time
query embedding digest if available
backend: brute_force | hnsw | hybrid | future_turbo
requested_candidates
returned_candidates
post_filter_candidates
filters applied
fallback/degradation list
exact_rerank bool
result IDs and scores
```

### Likely files

```text
semantic-memory/src/search.rs
semantic-memory/src/types.rs
semantic-memory/src/lib.rs
semantic-memory/tests/search_tests.rs
```

### Tests to add

- Receipt emitted in receipt mode.
- Receipt captures HNSW fallback.
- Receipt captures exact rerank flag.
- Receipt result IDs match returned results.

### Acceptance

- Search can answer “why this result?” at least at backend/fallback/score-breakdown level.

---

## Packet 08 — Boundary/defaulting audit

### Prompt

Audit semantic boundary paths for silent defaulting and convert dangerous defaults into typed errors, explicit policy defaults, or degradation records.

Tasks:

1. Search for `unwrap_or_default`, `unwrap_or`, permissive JSON fallback, and default constructors in:
   - `semantic-memory-forge/src/envelope.rs`
   - `forge-memory-bridge/src/transform.rs`
   - `semantic-memory/src/projection_*`
   - `semantic-memory/src/json_compat_import.rs`
   - `semantic-memory/src/embedder.rs`
2. Categorize each default:
   - harmless presentation/helper default,
   - explicit compatibility default,
   - semantic-boundary bug.
3. Fix semantic-boundary bugs.
4. Add tests for missing/malformed fields.

### Acceptance

- Bridge/import paths do not silently invent required semantic fields.
- Legacy compatibility defaults are documented and tested.
- Dangerous defaulting has typed error/degradation behavior.

---

## Packet 09 — Vector codec abstraction

### Prompt

Add a vector codec abstraction that supports raw reference scoring and current SQ8 without adding TurboQuant yet.

Tasks:

1. Add codec profile type:

```text
VectorCodecProfileV1
```

2. Add vector artifact type/table if persistence is in scope:

```text
VectorArtifactV1
```

3. Use an object-safe byte-oriented trait for persistence compatibility:

```rust
pub trait VectorCodec: Send + Sync {
    fn profile(&self) -> &VectorCodecProfile;
    fn encode_to_bytes(&self, raw: &[f32]) -> Result<Vec<u8>>;
    fn score_inner_product_from_bytes(&self, encoded: &[u8], query: &[f32]) -> Result<f32>;
    fn score_l2_from_bytes(&self, encoded: &[u8], query: &[f32]) -> Result<f32>;
    fn decode_approx_from_bytes(&self, encoded: &[u8]) -> Result<Option<Vec<f32>>>;
}
```

4. Implement `RawF32Codec`.
5. Implement `Sq8Codec` using existing quantize module if feasible.
6. Add profile digest and mismatch validation.

### Likely files

```text
semantic-memory/src/quantize.rs
semantic-memory/src/types.rs
semantic-memory/src/db.rs
semantic-memory/src/lib.rs
semantic-memory/tests/quantization.rs
semantic-memory/tests/quantization_pipeline.rs
```

### Tests to add

- Raw codec deterministic bytes.
- SQ8 codec deterministic bytes.
- Stable profile digest.
- Wrong profile/dimension fails closed.
- Raw reference scoring remains available.

### Acceptance

- No TurboQuant dependency yet.
- Current q8 path is not overloaded with TurboQuant semantics.
- Future codecs have a lawful admission seam.

---

## Packet 10 — Vector artifact persistence

### Prompt

Add minimal persistence for codec profiles and vector artifacts without making compressed vectors authoritative.

Tasks:

1. Add migration/table(s):

```sql
vector_codec_profiles
vector_artifacts
```

2. Store profile canonical JSON/digest.
3. Store encoded bytes/digest.
4. Store source row kind/id and source embedding digest.
5. Preserve raw embedding/reference path.
6. Add rebuild/delete lifecycle tests.

### Suggested schema

```sql
CREATE TABLE vector_codec_profiles (
  profile_id TEXT PRIMARY KEY,
  codec_family TEXT NOT NULL,
  codec_version TEXT NOT NULL,
  dim INTEGER NOT NULL,
  bits INTEGER,
  projections INTEGER,
  seed TEXT,
  score_semantics TEXT NOT NULL,
  canonical_json TEXT NOT NULL,
  profile_digest TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE vector_artifacts (
  artifact_id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  source_embedding_digest TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  encoded_bytes BLOB NOT NULL,
  encoded_digest TEXT NOT NULL,
  created_at TEXT NOT NULL,
  generation INTEGER NOT NULL,
  FOREIGN KEY(profile_id) REFERENCES vector_codec_profiles(profile_id)
);
```

### Acceptance

- Vector artifacts are deleted/retired with source lifecycle or marked stale.
- Rebuild can recreate artifacts from raw embeddings and profiles.
- Compressed vectors are never the only truth.

---

## Packet 11 — TurboQuant optional backend

### Prompt

Add TurboQuant as an optional vector codec backend only after the codec abstraction and integrity gates are green.

Tasks:

1. Add optional dependency/feature:

```toml
[features]
turbo-quant = ["dep:turbo-quant"]
```

2. Implement `TurboQuantCodec` behind feature flag.
3. Profile identity must include:
   - family = `turbo_quant`,
   - codec version,
   - dim,
   - bits,
   - projections,
   - seed,
   - score semantics,
   - normalization.
4. Serialize/deserialize `TurboCode` to canonical bytes.
5. Add raw reference comparison harness.
6. Add receipt integration.

### Tests to add

- Same profile/vector gives same encoded digest.
- Different seed gives different digest.
- Wrong dim/profile fails closed.
- Inner product estimate deterministic.
- Raw-vs-TurboQuant rank drift fixture emits metrics.

### Acceptance

- Feature compiles independently.
- Default build works without TurboQuant.
- TurboQuant codes are derived artifacts only.
- Search receipts disclose approximate scoring and exact rerank state.

---

## Packet 12 — Rank drift and benchmark harness

### Prompt

Add a deterministic benchmark/conformance harness for raw vs HNSW vs SQ8 vs TurboQuant retrieval profiles.

Tasks:

1. Fixed-seed synthetic corpus generator.
2. Corpus profiles:
   - 1k vectors / 100 queries,
   - namespace-sparse corpus,
   - deleted/tombstoned rows,
   - mixed row kinds if supported.
3. Metrics:
   - recall@1, recall@5, recall@10,
   - MRR or NDCG@10,
   - mean rank drift,
   - max top-k loss,
   - mean absolute score error,
   - p95 latency,
   - storage bytes/vector,
   - fallback count.
4. Output JSON/Markdown benchmark report.

### Acceptance

- Harness is deterministic.
- Raw reference is the oracle.
- Approximate profiles cannot claim exactness.
- Metrics can be attached to release notes.

---

## Packet 13 — Product-facing “Why this result?” API

### Prompt

Expose a practical explanation surface for search results.

Tasks:

1. Add API or method to retrieve explanation/receipt for a result.
2. Include:
   - source trail,
   - search profile,
   - vector/index backend,
   - approximate/exact status,
   - fallback/degradation,
   - result score breakdown,
   - receipt/replay handle.
3. Keep wording product-facing.

### Example output

```text
Result came from chunk:123.
It matched by vector search using HNSW.
HNSW under-returned after namespace filtering, so exact scoped fallback ran.
Raw rerank was applied.
No TurboQuant codec was used.
Receipt: search:01J...
```

### Acceptance

- Users/developers can answer “why did this appear?”
- Does not expose internal doctrine jargon by default.

---

## Packet 14 — Docs and public story

### Prompt

Update README/docs to reflect the product/career thesis without overselling unfinished internals.

Tasks:

1. README pitch:

```text
semantic-memory is a Rust substrate for local-first AI memory: SQLite + FTS + vector search, episode identity, source-grounded imports, and receipt-ready retrieval. It is designed to support compressed vector codecs such as TurboQuant without letting indexes or compressed vectors become the source of truth.
```

2. Add architecture diagram.
3. Add “Why this result?” example.
4. Add TurboQuant roadmap section, not default claim unless implemented.
5. Add release gates and benchmark expectations.

### Acceptance

- Public docs lead with visible value.
- Docs do not claim TurboQuant default support before implementation.
- Provenance is translated into receipts/source grounding/replay.

---

## Packet 15 — Final release audit

### Prompt

Run a final release audit after all implementation passes.

Tasks:

1. Run full acceptance commands.
2. Run benchmark/conformance harness.
3. Check no P0/P1 open without blocker.
4. Check no derived artifact is authoritative.
5. Check docs match implementation.
6. Write final report.

### Acceptance

- Final report is complete.
- Tests pass or failures are honestly documented.
- TurboQuant eligibility is explicitly marked:
   - not eligible,
   - prototype eligible,
   - default eligible.
