# Codex Pass: semantic-memory Stabilization / Fix-All Audit Pass

Created: 2026-05-11  
Target: `semantic-memory` plus included local dependency roots `stack-ids`, `semantic-memory-forge`, and `forge-memory-bridge`  
Mode: **stabilization before TurboQuant**  
Primary objective: **fix every issue identified in the previous audit responses, or explicitly close it as not-a-bug with evidence.**

This is a Codex-ready master prompt. Use it as the top-level instruction for a full repair pass.

---

## Start-here prompt for Codex

You are working on the `semantic-memory` Rust package and its included local dependency roots:

```text
semantic-memory/
stack-ids/
semantic-memory-forge/
forge-memory-bridge/
```

Your job is to perform a hard stabilization pass that closes the audit backlog before any TurboQuant integration. Do **not** add TurboQuant in this pass. Do **not** add broad new features. Fix correctness, reproducibility, durability, integrity, packaging, search, HNSW persistence, validation, and test coverage.

You must process the audit artifacts included with this pass:

```text
04_DEEP_AUDIT_REPORT.md
02_FINDINGS_INDEX.csv
03_FINDINGS_INDEX.json
```

The audit contains **608 findings**:

| Severity | Count |
|---|---:|
| P0 | 27 |
| P1 | 83 |
| P2 | 296 |
| P3 | 202 |

Confidence distribution:

| Confidence | Count |
|---|---:|
| confirmed | 196 |
| probable | 21 |
| static | 391 |


Area distribution:

| Area | Count |
|---|---:|
| numeric-cast | 110 |
| permissions | 107 |
| error-default | 95 |
| unwrap-expect | 80 |
| runtime-clock | 56 |
| docs | 25 |
| hnsw | 22 |
| hashmap-order | 20 |
| zip-truncate | 13 |
| dynamic-sql | 12 |
| search | 11 |
| embedding | 6 |
| insert-or-ignore | 5 |
| delete/integrity | 4 |
| db | 4 |
| packaging | 3 |
| conversation | 3 |
| validation | 3 |
| structure | 3 |
| integrity | 2 |
| api | 2 |
| q8 | 2 |
| episodes | 2 |
| keys | 2 |
| projection | 2 |
| import | 2 |
| security | 2 |
| bytemuck-storage | 2 |
| diagnostics | 1 |
| foreign-keys | 1 |
| sqlite | 1 |
| pool | 1 |
| bridge | 1 |
| ci | 1 |
| benchmarks | 1 |
| debug-assert | 1 |

### Non-negotiable rules

1. **No TurboQuant integration in this pass.** This pass is substrate stabilization only.
2. **Fix P0 and P1 findings, do not defer them.** A P0/P1 can only be closed as not-a-bug if you add a clear rationale and, where possible, a regression test proving safety.
3. **All P2/P3 findings must be processed.** They may be grouped, but none may be silently ignored.
4. **Do not remove features to make tests pass.** Preserve SQLite, FTS5, vector search, HNSW, q8, projection imports, Forge bridge integration, conversation memory, episode memory, and examples unless a feature is explicitly deprecated in a documented migration.
5. **Prefer small, reviewable patches.** Each patch should close a coherent cluster of findings.
6. **Every correctness fix requires a regression test.** No “trust me” patches.
7. **Every fallback/degraded mode must be observable.** Add status/health output rather than silent degradation.
8. **Every lossy operation must be explicit.** Especially q8/HNSW/compressed-vector paths.
9. **Every embedding write path must validate count, dimension, and finite values before storage/indexing.**
10. **The final package must build from a clean extraction without relying on the parent `/Coding/Libraries` workspace.**

---

## Required patch order

Implement in this order. Do not jump ahead unless the earlier gate is blocked and you record why.

### Pass 0 — Baseline and ledger

Create or update:

```text
docs/audits/semantic-memory-fix-pass-20260511.md
docs/audits/semantic-memory-fix-pass-20260511-findings-status.json
```

The Markdown ledger must include:

- audit source files used;
- starting commit/hash if available;
- Rust/Cargo version;
- commands run;
- P0/P1 closure table;
- grouped P2/P3 closure table;
- unresolved/deferred findings with rationale;
- final acceptance-gate output.

The JSON ledger must map each finding ID from `03_FINDINGS_INDEX.json` to one of:

```json
{
  "status": "fixed | not_a_bug | superseded | deferred",
  "evidence": "test/module/commit/rationale",
  "notes": "short explanation"
}
```

P0/P1 findings may not be left `deferred` unless the codebase literally cannot compile without a preceding environmental fix; in that case, document the blocker and continue with all unrelated fixes.

### Pass 1 — Hermetic workspace and packaging reproducibility

Goal: a cleanly extracted semantic-memory handoff builds without relying on a parent workspace.

Required actions:

1. Add an archive-root `Cargo.toml` workspace when packaging these four roots together:

```toml
[workspace]
resolver = "2"
members = [
  "semantic-memory",
  "stack-ids",
  "semantic-memory-forge",
  "forge-memory-bridge",
]
default-members = ["semantic-memory"]

[workspace.dependencies]
rusqlite = { version = "0.32", features = ["bundled", "blob"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["rt", "macros", "sync"] }
thiserror = "2"
tracing = "0.1"
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
schemars = "0.8"
tempfile = "3"
proptest = "1"

[workspace.lints.rust]
unsafe_code = "forbid"
missing_docs = "allow"

[workspace.lints.clippy]
unwrap_used = "warn"
expect_used = "warn"
panic = "warn"
```

Use the real parent workspace dependency versions if available. The template above is a fallback, not a substitute for correct local resolution.

2. Remove ambiguity from multiple lockfiles:
   - For the standalone package, prefer one root `Cargo.lock` generated from the packaged root workspace.
   - If retaining member lockfiles, document why and ensure `cargo test --workspace` still uses the root lock.

3. Update the zip/certifier flow if necessary so the generated archive includes the workspace root manifest and excludes stale sidecars, archive files, `target`, and `target-*` build-output directories.

4. If `z.py` is part of this package, add a `semantic-memory` profile and explicitly prune `target-*` directories.

Acceptance gate:

```bash
rm -rf /tmp/sm-hermetic
mkdir -p /tmp/sm-hermetic
unzip semantic-memory-*.zip -d /tmp/sm-hermetic
cd /tmp/sm-hermetic
cargo check --workspace --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-features -- -D warnings
cargo doc --workspace --all-features --no-deps
```

### Pass 2 — Central embedding and vector invariants

Goal: no invalid vector or malformed batch can reach SQLite, q8, HNSW, projection imports, or future compressed-vector code.

Required actions:

1. Add a central validation module/function, e.g.:

```rust
pub(crate) fn validate_embedding(values: &[f32], expected_dim: usize) -> Result<(), MemoryError>;
pub(crate) fn validate_embedding_batch(values: &[Vec<f32>], requested: usize, expected_dim: usize) -> Result<(), MemoryError>;
pub(crate) fn validate_vector_blob_len(bytes: &[u8], expected_dim: usize) -> Result<(), MemoryError>;
```

2. Validation must enforce:

```text
returned batch length == requested input length
vector length == configured dimension
every value is finite: no NaN, +Inf, -Inf
blob bytes length == dim * 4 for f32
q8/TurboQuant/etc. metadata dimensions match the configured dimension
```

3. Apply validation before writes in:

```text
facts
chunks/documents
messages/conversations
episodes
projection imports
reembed_all/reembed_* paths
HNSW insert/update
q8 encode/decode
search/rerank paths
```

4. Replace silent `zip()` truncation with explicit count checks.

5. Add `MemoryError` variants for:

```text
EmbeddingBatchCountMismatch { requested, returned }
EmbeddingDimensionMismatch { expected, actual }
NonFiniteEmbeddingValue { index }
VectorBlobLengthMismatch { expected_bytes, actual_bytes }
```

Required tests:

- fake embedder returns fewer vectors than requested: all ingest/reembed paths fail atomically;
- fake embedder returns more vectors than requested: fail loudly;
- fake embedder returns NaN/Inf/-Inf: fail before DB/HNSW/q8 write;
- wrong dimension through every public write path: fail;
- invalid f32 blob decode: fail without panic.

### Pass 3 — f32 binary codec correctness

Goal: vector byte encoding/decoding must be portable and validation-aware.

Required actions:

1. Replace `bytemuck::try_cast_slice::<u8, f32>` decoding for persisted SQLite blobs with explicit little-endian decode:

```rust
fn encode_f32_le(values: &[f32]) -> Vec<u8>;
fn decode_f32_le(bytes: &[u8], expected_dim: usize) -> Result<Vec<f32>, MemoryError>;
```

2. Reject odd/truncated lengths.
3. Reject dimension mismatches.
4. Reject non-finite decoded values.
5. Keep bytemuck only where alignment/endian assumptions are proven and not persisted-format critical.

Required tests:

- round-trip f32 values;
- reject length not multiple of 4;
- reject wrong dimension;
- reject encoded NaN/Inf;
- ensure no silent truncation.

### Pass 4 — Deletion, cascade, and stale-memory correctness

Goal: deleting facts/documents/namespaces/sessions removes or invalidates all derived memory surfaces.

Required actions:

1. Fix `delete_document` / `delete_document_with_chunks` so it explicitly handles:

```text
document row
document_chunks/chunks
chunks_fts
chunks_rowid_map
episodes linked to document
episodes_fts
episodes_rowid_map
episode_causes
projection derivations/imports linked to deleted rows
q8 rows/compressed sidecars
HNSW pending delete ops for all affected keys
```

2. Fix `delete_fact` so it removes/updates dependent `episode_causes` and any projection/search surfaces that reference the fact.

3. Fix `delete_namespace` to return a structured report, not just fact count:

```rust
pub struct NamespaceDeleteReport {
    pub facts: usize,
    pub documents: usize,
    pub chunks: usize,
    pub messages: usize,
    pub sessions: usize,
    pub episodes: usize,
    pub projection_rows: usize,
    pub hnsw_ops: usize,
}
```

4. Ensure deletes run transactionally or in a documented two-phase flow with repairable pending HNSW ops.

Required tests:

- document with chunks + episode + FTS + HNSW search hit, then delete document: no stale search result;
- fact referenced by episode cause, then delete fact: no dangling cause;
- namespace delete report counts every affected surface;
- delete failure rolls back DB state or leaves explicit repairable pending state;
- full integrity after deletion reports clean.

### Pass 5 — Integrity engine upgrade

Goal: integrity checks must prove key/content coherence, not just equal counts.

Required actions:

1. Replace/augment count-only checks with key-level checks:

```text
chunks ↔ chunks_rowid_map ↔ chunks_fts
episodes ↔ episodes_rowid_map ↔ episodes_fts
episode_causes ↔ live fact/document/session/message keys
HNSW keymap ↔ live embedded rows
q8/compressed rows ↔ live f32 embeddings if policy says mandatory
projection import/derivation rows ↔ live source keys
```

2. Add content-level checks where feasible:

```text
FTS row maps to the correct source key
stored text hash or snapshot matches source row, if available
HNSW sidecar metadata dimension/model/version matches DB config
```

3. Clarify q8 policy:

- If q8 is mandatory: writes fail when q8 fails; integrity errors are hard errors.
- If q8 is optional acceleration: integrity reports missing q8 as degraded/warning, and repair can backfill.

Recommended: q8/compressed vectors should be optional acceleration artifacts; f32 remains source of truth and rerank representation.

Required tests:

- equal counts but swapped rowid map: integrity fails;
- missing FTS row: integrity fails;
- stale HNSW key: integrity fails;
- missing q8 follows chosen policy;
- repair removes stale derived state and/or regenerates missing acceleration rows.

### Pass 6 — HNSW persistence, corruption, and concurrency hardening

Goal: HNSW sidecar handling must be bounded, versioned, atomic, and observable.

Required actions:

1. Add versioned sidecar headers with:

```text
magic bytes
format version
embedding dimension
model id / index config hash if available
entry count
checksum/hash for keymap/data/graph payloads
```

2. Replace `usize` persisted fields with fixed-width integer fields (`u32`/`u64`) and explicit little-endian encoding.
3. Bound all allocations before reading sidecar payloads:

```text
byte_len == dim * 4
entries <= configured max
file size <= configured cap
dim <= configured max dimension
```

4. Make sidecar saves atomic:

```text
write to .tmp
fsync if practical
rename into place
only mark pending ops flushed after save succeeds
```

5. Make `upsert` idempotent. Repeated upsert for the same logical key must replace or tombstone old entries without graph bloat causing stale duplicate hits.
6. Ensure save uses a real immutable snapshot or holds the correct lock until serialization is complete.
7. Add explicit startup policy:

```rust
pub enum HnswStartupPolicy {
    RebuildInlineSmallOnly { max_vectors: usize },
    RebuildBackground,
    DegradeToBruteForce,
    FailFast,
}
```

8. Expose HNSW health/status:

```text
ready | missing | stale | rebuilding | degraded | failed
```

Required tests:

- corrupt header: clean error/degraded rebuild, no panic;
- huge byte length: clean error, no large allocation;
- dimension mismatch: degraded/rebuild path;
- truncated file: degraded/rebuild path;
- graph exists but keymap missing: degraded/rebuild path;
- save failure preserves pending ops;
- repeated upsert does not produce stale duplicate search hits;
- concurrent write/save cannot corrupt sidecar.

### Pass 7 — Search correctness and async/blocking boundaries

Goal: search must be deterministic enough to test, safe under filters, and not block async runtime threads.

Required actions:

1. Make `cosine_similarity` return `Result<f32>` or keep it private behind dimension-validated callers. No release-build silent truncation.
2. Reject non-finite scores or treat them as errors before sorting.
3. Fix filtered HNSW search:
   - if global ANN candidates are filtered out by namespace/session/source, fall back to filtered brute-force or oversample ANN candidates with bounded retry;
   - report fallback/degraded mode in explain output/status.
4. Move conversation HNSW/vector search CPU work into the same blocking helper pattern as regular search.
5. Clamp `top_k`, candidate count, oversampling, and rerank limits with documented maxima.
6. Inject/test clock for recency scoring; avoid runtime-clock nondeterminism in tests.
7. Make tie-breaking deterministic where possible:

```text
score desc, then timestamp desc if intentional, then stable key/id asc
```

Required tests:

- mismatched vector lengths fail;
- NaN score cannot enter ranking;
- namespace-filtered HNSW search falls back and returns valid hits;
- async conversation search does not perform CPU-heavy work directly on async task;
- top_k cap enforced;
- deterministic tie order test.

### Pass 8 — Projection, episode, and bridge consistency

Goal: imported/projection memory must stay coherent with source keys and deletion/repair logic.

Required actions:

1. Verify projection import paths use central vector validation.
2. Ensure projection-derived rows reference live source IDs/scopes.
3. Ensure deletion of source facts/documents/episodes invalidates or removes projection rows as policy dictates.
4. Add namespace/scope validation to bridge imports.
5. Validate metadata size, source URI length, IDs, and JSON payload bounds.

Required tests:

- import invalid vector: rejected;
- import stale source reference: rejected or marked degraded;
- delete source row then projection query: no stale trusted hit;
- bridge import with invalid namespace/scope: rejected;
- oversized metadata/source URI rejected.

### Pass 9 — API validation and error hygiene

Goal: callers get precise errors and cannot store pathological keys/payloads.

Required actions:

1. Add validation for:

```text
namespace nonempty and max length
title max length
source/source_uri max length
metadata max serialized size
chunk text max size
session/conversation/message identifiers
model id / embedding dimension config
```

2. Replace silent defaults/swallowed errors where audit findings indicate correctness impact.
3. Audit `unwrap`, `expect`, `panic`, unchecked casts, dynamic SQL construction, runtime-clock use, and HashMap iteration determinism using the P2/P3 findings.
4. Dynamic SQL must whitelist identifiers. Values must use parameters.

Required tests:

- invalid namespace rejected;
- oversized metadata rejected;
- invalid source URI/title rejected if policy says so;
- dynamic SQL identifier injection attempt rejected;
- no panic on malformed caller input.

### Pass 10 — Documentation, public surface, and CI

Goal: make the project reviewable and stop internal prompt clutter from looking like architecture sprawl.

Required actions:

1. Move internal prompt/spec/addendum docs into:

```text
docs/internal/codex/
docs/design-history/
```

2. Keep public entry clean:

```text
README.md
ARCHITECTURE.md
API.md
MIGRATIONS.md
INTEGRITY.md
HNSW_PERSISTENCE.md
BENCHMARKS.md
TESTING.md
```

3. Add/update CI:

```yaml
cargo fmt --all --check
cargo check --workspace --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-features -- -D warnings
cargo doc --workspace --all-features --no-deps
cargo test -p semantic-memory --no-default-features --features brute-force
cargo test -p semantic-memory --no-default-features --features hnsw
```

4. Add benchmark harness skeleton but do not optimize yet. Benchmark harness should support:

```text
f32 brute force
q8
HNSW
future TurboQuant placeholder only, disabled/not implemented
hybrid FTS5 + vector
rerank
```

Required docs:

- `ARCHITECTURE.md`: SQLite source of truth; FTS/HNSW/q8 are derived/acceleration surfaces.
- `INTEGRITY.md`: derived-state invariants and repair strategy.
- `HNSW_PERSISTENCE.md`: sidecar format, startup policy, corruption behavior.
- `BENCHMARKS.md`: current baseline table, even if initially small.

### Pass 11 — Final acceptance and repackage

Run the acceptance script included in this pass:

```bash
bash 01_ACCEPTANCE_GATES.sh
```

Then regenerate a source/context archive and verify:

```text
certifier findings: 0
ZIP duplicate names: 0
ZIP unsafe names: 0
clean extraction cargo check/test/clippy/doc: pass
P0/P1 ledger: all fixed or not-a-bug with evidence
P2/P3 ledger: processed/grouped with evidence
```

---

## P0 findings that must be closed

### SM-AUD-0001 — Archive is not hermetic despite passing certifier

- Severity: `P0`
- Confidence: `confirmed`
- Area: `packaging`
- File: `semantic-memory-generic-rust-next-codex-context-20260511.report.md`:1
- Why it matters: Cargo metadata succeeded from parent /Coding/Libraries workspace with 326 packages/30 workspace members while the archive has zero workspace manifests. A clean extractor may fail or resolve differently.
- Required fix: Add an archive-root Cargo.toml workspace, or remove workspace-only dependency/lint reliance; validate from fresh extraction.

### SM-AUD-0002 — No packaged root workspace manifest for included local crates

- Severity: `P0`
- Confidence: `confirmed`
- Area: `packaging`
- File: `semantic-memory/Cargo.toml`:1
- Why it matters: The zip includes semantic-memory plus three path roots, but no top-level workspace manifest tying them together.
- Required fix: Generate a root Cargo.toml with members semantic-memory, stack-ids, semantic-memory-forge, forge-memory-bridge.

### SM-AUD-0003 — Multiple Cargo.lock files create ambiguous dependency source of truth

- Severity: `P0`
- Confidence: `confirmed`
- Area: `packaging`
- File: `semantic-memory/Cargo.lock`:1
- Why it matters: Each included package has its own lockfile; without a packaged root workspace, dependency resolution can differ between crates.
- Required fix: Use one workspace lockfile at archive root for review builds or document crate-by-crate build commands.

### SM-AUD-0004 — Document ingest silently truncates chunks on embedder batch-count mismatch

- Severity: `P0`
- Confidence: `confirmed`
- Area: `embedding`
- File: `semantic-memory/src/documents.rs`:324
- Why it matters: text_chunks.iter().zip(embeddings.iter()) drops chunks if the embedder returns fewer vectors, and ignores extra vectors.
- Required fix: Centralize embed_batch validation: returned len must equal requested len before any write.

### SM-AUD-0005 — Fact re-embedding silently truncates on batch-count mismatch

- Severity: `P0`
- Confidence: `confirmed`
- Area: `embedding`
- File: `semantic-memory/src/lib.rs`:1254
- Why it matters: Batch rows are zipped with embeddings; fewer embeddings still increments fact_count by batch.len().
- Required fix: Fail loudly on batch-count mismatch before constructing updates.

### SM-AUD-0006 — Chunk re-embedding silently truncates on batch-count mismatch

- Severity: `P0`
- Confidence: `confirmed`
- Area: `embedding`
- File: `semantic-memory/src/lib.rs`:1314
- Why it matters: Batch rows are zipped with embeddings; missing embeddings leave stale rows while progress counters report success.
- Required fix: Fail loudly on batch-count mismatch before update transaction.

### SM-AUD-0007 — Message re-embedding silently truncates on batch-count mismatch

- Severity: `P0`
- Confidence: `confirmed`
- Area: `embedding`
- File: `semantic-memory/src/lib.rs`:1374
- Why it matters: Batch rows are zipped with embeddings; message embeddings can be partially refreshed without being reported.
- Required fix: Fail loudly on batch-count mismatch before update transaction.

### SM-AUD-0008 — Episode re-embedding silently truncates on batch-count mismatch

- Severity: `P0`
- Confidence: `confirmed`
- Area: `embedding`
- File: `semantic-memory/src/lib.rs`:1434
- Why it matters: Batch rows are zipped with embeddings; missing episode embeddings can leave stale recall state.
- Required fix: Fail loudly on batch-count mismatch before update transaction.

### SM-AUD-0009 — Public embedding validation is dimension-only

- Severity: `P0`
- Confidence: `confirmed`
- Area: `embedding`
- File: `semantic-memory/src/lib.rs`:536
- Why it matters: validate_embedding_dimensions checks length but not NaN/Inf, while HNSW later rejects non-finite values; SQLite can still store bad f32 blobs.
- Required fix: Replace with validate_embedding that checks dimensions and all components finite.

### SM-AUD-0010 — delete_document does not explicitly clean episode derived state

- Severity: `P0`
- Confidence: `confirmed`
- Area: `delete/integrity`
- File: `semantic-memory/src/documents.rs`:109
- Why it matters: Document deletion removes chunks/docs but not episode_causes, episodes_fts, episodes_rowid_map, or episode HNSW ops before cascade/cleanup boundaries.
- Required fix: Collect episode_ids first and delete all episode derived surfaces plus queued HNSW deletes in one transaction.

### SM-AUD-0011 — delete_document can leave stale HNSW episode keys

- Severity: `P0`
- Confidence: `probable`
- Area: `delete/integrity`
- File: `semantic-memory/src/documents.rs`:109
- Why it matters: If episodes are cascaded from documents, HNSW sidecar does not know about episode deletes unless explicit pending ops are queued.
- Required fix: Queue Delete for every episode:{episode_id} before deleting the document.

### SM-AUD-0012 — Vector scan uses bytemuck::try_cast_slice on SQLite Vec<u8>

- Severity: `P0`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/search.rs`:272
- Why it matters: SQLite blobs are byte vectors; casting requires alignment and native endian assumptions. Valid blobs may fail or decode incorrectly on non-little-endian targets.
- Required fix: Use db::bytes_to_embedding for all blob decoding; avoid bytemuck on storage bytes.

### SM-AUD-0013 — HNSW sidecar loader allocates raw byte_len from file without cap

- Severity: `P0`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:501
- Why it matters: A corrupt sidecar can declare a huge byte length and trigger large allocation before validation.
- Required fix: Require byte_len == dimensions*4 and <= configured max before allocation.

### SM-AUD-0014 — HNSW data format stores dimensions using usize

- Severity: `P0`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:460
- Why it matters: usize serialization is platform-width dependent; sidecar created on 64-bit may not load on 32-bit and is not a stable portable format.
- Required fix: Use fixed-width u32/u64 little-endian fields with versioned header.

### SM-AUD-0015 — HNSW save is not atomic

- Severity: `P0`
- Confidence: `probable`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:142
- Why it matters: file_dump writes directly; crash or process kill can leave partial graph/data sidecars.
- Required fix: Write to temp files, fsync, then atomic rename graph/data/keymap as a set.

### SM-AUD-0016 — Pending HNSW mutations are applied before sidecar save succeeds

- Severity: `P0`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw_ops.rs`:184
- Why it matters: If save fails after mutating in-memory index, pending ops remain and can be applied again, creating duplicate tombstones/nodes.
- Required fix: Build/save a snapshot or roll back in-memory mutations on save failure.

### SM-AUD-0017 — Pending upsert calls insert instead of update

- Severity: `P0`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw_ops.rs`:192
- Why it matters: Upsert on an existing key allocates a new node and tombstones the old node, increasing graph bloat on repeated retries.
- Required fix: Use update() or replace semantics for existing keys; dedupe pending ops by key.

### SM-AUD-0018 — HNSW sidecar save clones Arc while graph can still mutate

- Severity: `P0`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/lib.rs`:669
- Why it matters: Cloning HnswIndex does not snapshot the underlying graph; concurrent writes can mutate while save is in progress.
- Required fix: Hold exclusive lock during save or introduce immutable snapshot serialization.

### SM-AUD-0019 — Graph sidecar validation only checks non-empty file

- Severity: `P0`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:429
- Why it matters: validate_graph_sidecar does not verify magic/version/consistency with data/keymap.
- Required fix: Validate graph/data/keymap together with checksums and stored dimensions/counts.

### SM-AUD-0020 — HNSW integrity is count-based, not key-level

- Severity: `P0`
- Confidence: `confirmed`
- Area: `integrity`
- File: `semantic-memory/src/lib.rs`:730
- Why it matters: Equal counts can hide wrong mappings, stale IDs, or wrong source types.
- Required fix: Verify each keymap key maps to a live row and each live embedded row has a matching key.

### SM-AUD-0021 — FTS integrity count checks use dynamic table names and count parity only

- Severity: `P0`
- Confidence: `confirmed`
- Area: `integrity`
- File: `semantic-memory/src/db.rs`:1400
- Why it matters: Count parity can pass despite rowid/content mismatches.
- Required fix: Perform key-level rowid_map/content checks for every FTS-backed table.

### SM-AUD-0022 — cosine_similarity truncates mismatched vectors in release builds

- Severity: `P0`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/search.rs`:61
- Why it matters: debug_assert_eq is disabled in release; zip then compares only the shorter length.
- Required fix: Return Result or validate dimensions before zip in all builds.

### SM-AUD-0023 — cosine similarity accepts non-finite stored/query vectors

- Severity: `P0`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/search.rs`:290
- Why it matters: NaN/Inf can produce NaN scores and unstable sorting.
- Required fix: Validate finite vectors before storage and skip/error on non-finite during reads.

### SM-AUD-0024 — HNSW filtered search can return empty results without brute-force fallback

- Severity: `P0`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/search.rs`:963
- Why it matters: HNSW gets global candidates before namespace/session/source-type filters; if filtered candidates are removed, valid rows outside the top candidate pool are missed.
- Required fix: Overfetch adaptively after filters or fallback to brute force when post-filter hits < k.

### SM-AUD-0025 — conversation HNSW search runs blocking CPU work on async thread

- Severity: `P0`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/conversation.rs`:683
- Why it matters: It directly holds the HNSW read lock and searches in async context, unlike main hnsw_search_blocking.
- Required fix: Route through spawn_blocking helper.

### SM-AUD-0026 — delete_fact does not clean episode_causes references

- Severity: `P0`
- Confidence: `confirmed`
- Area: `delete/integrity`
- File: `semantic-memory/src/knowledge.rs`:143
- Why it matters: Deleting a fact queues HNSW delete and removes FTS, but episodes may still cite the fact as a cause.
- Required fix: Delete or mark episode_causes rows referencing the fact and update affected episode search/provenance.

### SM-AUD-0027 — update_fact does not update dependent episode/projection search text

- Severity: `P0`
- Confidence: `probable`
- Area: `delete/integrity`
- File: `semantic-memory/src/knowledge.rs`:181
- Why it matters: Fact content changes can make derived episode/projection references semantically stale.
- Required fix: Record invalidation edges or recompute affected derived search surfaces.


---

## P1 findings that must be closed

### SM-AUD-0028 — delete_namespace returns only fact count despite deleting many entity types

- Severity: `P1`
- Confidence: `confirmed`
- Area: `api`
- File: `semantic-memory/src/knowledge.rs`:236
- Why it matters: The API return value underreports blast radius and can make receipts/logs false.
- Required fix: Return NamespaceDeleteReport with counts per entity/table/op.

### SM-AUD-0029 — Open-time HNSW rebuild/degrade policy is implicit

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/lib.rs`:369
- Why it matters: The open path may rebuild/clear/swap indexes based on metadata without an explicit user-visible startup policy.
- Required fix: Expose HnswStartupPolicy and health status.

### SM-AUD-0030 — SQL errors while counting embeddings are swallowed as zero

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/lib.rs`:403
- Why it matters: query_row(...).unwrap_or(0) can suppress a DB error and avoid needed rebuild.
- Required fix: Propagate DB errors during integrity decisions.

### SM-AUD-0031 — Orphan-count SQL errors are swallowed as zero

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/lib.rs`:455
- Why it matters: A failed orphan check can make a stale keymap appear clean.
- Required fix: Propagate the error or force degraded/rebuild state.

### SM-AUD-0032 — Missing hnsw_keymap table silently leaves loaded graph without keys

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:310
- Why it matters: load_keymap returns Ok with empty keymap; graph points become unresolvable.
- Required fix: Treat graph+missing keymap as degraded/rebuild, not clean load.

### SM-AUD-0033 — Malformed next_id metadata falls back silently

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:353
- Why it matters: Invalid next_id is parsed with ok/unwrap_or graph point count.
- Required fix: Report error or mark sidecar stale when metadata is malformed.

### SM-AUD-0034 — HNSW len can report nonzero even when keymap is empty

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:222
- Why it matters: len uses graph point count minus deleted IDs, not resolvable key count.
- Required fix: Expose separate graph_len and live_key_count; search should use resolvable key count.

### SM-AUD-0035 — Tombstone overfetch is too naive

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:185
- Why it matters: fetch_count = top_k + deleted_ids.len() still can miss valid neighbors if tombstones cluster near the query.
- Required fix: Iteratively overfetch until enough live hits or graph exhausted.

### SM-AUD-0036 — deleted_ratio may divide using graph count that includes unreachable/unmapped points

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:250
- Why it matters: Compaction threshold can be misleading if keymap drift exists.
- Required fix: Compute deleted/live ratios from verified keymap state.

### SM-AUD-0037 — u64 node id is cast to usize without range check

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:496
- Why it matters: Sidecar with node id > usize::MAX truncates on 32-bit and can corrupt mapping.
- Required fix: TryFrom<u64> with explicit error.

### SM-AUD-0038 — insert ignores return/status from hnsw_rs graph.insert

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw.rs`:383
- Why it matters: If insertion fails or panics internally, keymap may still be updated.
- Required fix: Wrap insert in catch_unwind if needed and use API result if available; update keymap only after success.

### SM-AUD-0039 — HNSW rebuild silently skips invalid fact embeddings

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw_ops.rs`:71
- Why it matters: db::bytes_to_embedding errors are ignored, producing an incomplete index.
- Required fix: Count skipped rows and return degraded integrity finding.

### SM-AUD-0040 — HNSW rebuild silently skips invalid chunk embeddings

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw_ops.rs`:89
- Why it matters: Invalid embeddings are dropped without a repair error.
- Required fix: Count skipped rows and expose rebuild diagnostics.

### SM-AUD-0041 — HNSW rebuild silently skips invalid message embeddings

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw_ops.rs`:107
- Why it matters: Invalid message embeddings are dropped without a repair error.
- Required fix: Count skipped rows and expose rebuild diagnostics.

### SM-AUD-0042 — HNSW rebuild silently skips invalid episode embeddings

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw_ops.rs`:125
- Why it matters: Invalid episode embeddings are dropped without a repair error.
- Required fix: Count skipped rows and expose rebuild diagnostics.

### SM-AUD-0043 — clear_pending_index_ops is all-or-nothing per processed key list after sidecar save

- Severity: `P1`
- Confidence: `confirmed`
- Area: `hnsw`
- File: `semantic-memory/src/hnsw_ops.rs`:205
- Why it matters: If keymap flush succeeds but clear fails, mutations may be replayed and duplicate/tombstone bloat occurs.
- Required fix: Use transactional state machine with op generation numbers and idempotent upsert.

### SM-AUD-0044 — q8 optionality conflicts with integrity expectations

- Severity: `P1`
- Confidence: `confirmed`
- Area: `q8`
- File: `semantic-memory/src/lib.rs`:1256
- Why it matters: Code comments say q8 is optional/non-fatal, but full integrity can treat missing q8 as an issue.
- Required fix: Define compressed vectors as mandatory or optional; align write, repair, and integrity.

### SM-AUD-0045 — q8 baseline lacks explicit versioned storage envelope

- Severity: `P1`
- Confidence: `confirmed`
- Area: `q8`
- File: `semantic-memory/src/quantize.rs`:1
- Why it matters: Packed q8 bytes need version/dims/scale metadata for future TurboQuant coexistence.
- Required fix: Add a vector-codec envelope with codec, version, dim, checksum, and params.

### SM-AUD-0046 — Invalid timestamp becomes maximally fresh

- Severity: `P1`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/search.rs`:102
- Why it matters: days_since(ts).unwrap_or(0.0) makes parse failures age 0 after a warning.
- Required fix: Treat invalid timestamps as no recency contribution or stale.

### SM-AUD-0047 — recency scoring uses wall-clock inside ranking

- Severity: `P1`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/search.rs`:85
- Why it matters: Utc::now makes ranking nondeterministic and hard to test/replay.
- Required fix: Inject clock into SearchConfig or query context.

### SM-AUD-0048 — candidate_pool_size.max(k * 3) can overflow

- Severity: `P1`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/lib.rs`:851
- Why it matters: usize multiplication can overflow for large top_k.
- Required fix: Use k.saturating_mul(3) and cap top_k.

### SM-AUD-0049 — Second candidate_pool_size.max(k * 3) overflow surface

- Severity: `P1`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/lib.rs`:954
- Why it matters: Same unbounded top_k multiplication appears in another search path.
- Required fix: Use saturating_mul and configured max_top_k.

### SM-AUD-0050 — Third candidate_pool_size.max(k * 3) overflow surface

- Severity: `P1`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/lib.rs`:1033
- Why it matters: Same unbounded top_k multiplication appears in another path.
- Required fix: Use saturating_mul and configured max_top_k.

### SM-AUD-0051 — conversation candidate_pool_size.max(k * 3) overflow surface

- Severity: `P1`
- Confidence: `confirmed`
- Area: `search`
- File: `semantic-memory/src/conversation.rs`:683
- Why it matters: Unbounded top_k multiplication in conversation search.
- Required fix: Use saturating_mul and configured max_top_k.

### SM-AUD-0052 — Unknown token counts are treated as zero in budget selection

- Severity: `P1`
- Confidence: `confirmed`
- Area: `conversation`
- File: `semantic-memory/src/conversation.rs`:172
- Why it matters: Messages with missing token_count can bypass max_tokens.
- Required fix: Recompute missing token_count or treat unknown as conservative upper bound.

### SM-AUD-0053 — Token budget addition can overflow u32

- Severity: `P1`
- Confidence: `confirmed`
- Area: `conversation`
- File: `semantic-memory/src/conversation.rs`:173
- Why it matters: total_tokens + msg_tokens may overflow before comparison.
- Required fix: Use checked_add/saturating_add and error or clamp.

### SM-AUD-0054 — session_token_count casts negative SQL sum to u64

- Severity: `P1`
- Confidence: `confirmed`
- Area: `conversation`
- File: `semantic-memory/src/conversation.rs`:185
- Why it matters: If corrupt rows contain negative token_count, cast wraps to huge positive.
- Required fix: Validate nonnegative aggregate before conversion.

### SM-AUD-0055 — Session/channel identifiers are not consistently length/whitespace validated

- Severity: `P1`
- Confidence: `probable`
- Area: `validation`
- File: `semantic-memory/src/conversation.rs`:63
- Why it matters: Malformed or huge identifiers can leak into storage and logs.
- Required fix: Centralize validation for session_id, channel, namespace, title, source URI.

### SM-AUD-0056 — Document title/source/metadata size validation is weaker than content validation

- Severity: `P1`
- Confidence: `probable`
- Area: `validation`
- File: `semantic-memory/src/documents.rs`:315
- Why it matters: Large metadata/title/source strings can bloat DB and docs.
- Required fix: Add max lengths and metadata byte caps.

### SM-AUD-0057 — Episode search limit is unbounded

- Severity: `P1`
- Confidence: `probable`
- Area: `validation`
- File: `semantic-memory/src/episodes.rs`:393
- Why it matters: A caller can request a huge limit and cause large result materialization.
- Required fix: Cap limit using config max_top_k/max_query_rows.

### SM-AUD-0058 — search_episodes drops episode_id and returns document_id

- Severity: `P1`
- Confidence: `confirmed`
- Area: `api`
- File: `semantic-memory/src/episodes.rs`:437
- Why it matters: Multiple episodes per document cannot be distinguished by caller.
- Required fix: Return episode_id plus document_id or a typed EpisodeSearchResult.

### SM-AUD-0059 — Episode parse errors report document_id instead of episode_id

- Severity: `P1`
- Confidence: `confirmed`
- Area: `diagnostics`
- File: `semantic-memory/src/episodes.rs`:452
- Why it matters: Diagnostic row id for cause_ids/outcome/status points to document_id, not the episode row.
- Required fix: Use episode_id in parse helpers.

### SM-AUD-0060 — INSERT OR IGNORE collapses duplicate cause IDs silently

- Severity: `P1`
- Confidence: `confirmed`
- Area: `episodes`
- File: `semantic-memory/src/episodes.rs`:285
- Why it matters: Duplicate causes with different ordinal positions are discarded.
- Required fix: Validate and reject duplicate cause_ids or preserve multiplicity intentionally.

### SM-AUD-0061 — update_episode_outcome cannot clear experiment_id

- Severity: `P1`
- Confidence: `confirmed`
- Area: `episodes`
- File: `semantic-memory/src/episodes.rs`:360
- Why it matters: COALESCE(?3, experiment_id) preserves old experiment_id when None is passed.
- Required fix: Add explicit clear operation or Option<Option<String>> semantics.

### SM-AUD-0062 — HNSW key parsing split_once(:) is fragile for IDs containing colon

- Severity: `P1`
- Confidence: `probable`
- Area: `keys`
- File: `semantic-memory/src/hnsw.rs`:58
- Why it matters: Episode/document/user-provided IDs may contain colon unless validated.
- Required fix: Use structured key encoding or reject colon in IDs.

### SM-AUD-0063 — Message dedup key uses session_id:message_id delimiter

- Severity: `P1`
- Confidence: `probable`
- Area: `keys`
- File: `semantic-memory/src/search.rs`:119
- Why it matters: If session_id contains colon, downstream parsing/dedup diagnostics can become ambiguous.
- Required fix: Use tuple type internally or escaped/keyed serialization.

### SM-AUD-0064 — Correctness depends on PRAGMA foreign_keys being enabled for every pooled connection

- Severity: `P1`
- Confidence: `probable`
- Area: `foreign-keys`
- File: `semantic-memory/src/db.rs`:1
- Why it matters: Cascades/derived cleanup assumptions fail if a connection misses the pragma.
- Required fix: Assert PRAGMA foreign_keys=ON after every connection checkout.

### SM-AUD-0065 — PRAGMA max_page_count computed with dynamic formatting

- Severity: `P1`
- Confidence: `probable`
- Area: `db`
- File: `semantic-memory/src/db.rs`:558
- Why it matters: Even though value is numeric, config-derived PRAGMA should be validated/capped.
- Required fix: Validate max_page_count range before execute.

### SM-AUD-0066 — Dynamic table_info table name formatting relies on internal callers only

- Severity: `P1`
- Confidence: `confirmed`
- Area: `db`
- File: `semantic-memory/src/db.rs`:705
- Why it matters: If any caller ever passes user input, PRAGMA table_info becomes injection-prone.
- Required fix: Make table an enum or whitelist.

### SM-AUD-0067 — Dynamic ALTER TABLE formatting relies on internal table/column whitelists

- Severity: `P1`
- Confidence: `confirmed`
- Area: `db`
- File: `semantic-memory/src/db.rs`:715
- Why it matters: Schema migration helpers must not accept arbitrary strings.
- Required fix: Make migration table/column identifiers enum-backed.

### SM-AUD-0068 — Dynamic SELECT COUNT table name relies on internal map table list

- Severity: `P1`
- Confidence: `confirmed`
- Area: `db`
- File: `semantic-memory/src/db.rs`:1400
- Why it matters: Future caller could turn table name into SQL injection.
- Required fix: Use enum/constant-only function signature.

### SM-AUD-0069 — SQLite WAL/checkpoint/backpressure policy not visible in archive-level docs

- Severity: `P1`
- Confidence: `probable`
- Area: `sqlite`
- File: `semantic-memory/src/db.rs`:1
- Why it matters: Long-running local stores need defined WAL/checkpoint/backup behavior.
- Required fix: Document and test WAL mode, busy timeout, checkpoint, and backup semantics.

### SM-AUD-0070 — Connection pool shutdown/poison behavior needs stress coverage

- Severity: `P1`
- Confidence: `probable`
- Area: `pool`
- File: `semantic-memory/src/pool.rs`:1
- Why it matters: spawn_blocking and pool handoff may hide panics as Other, but lifecycle correctness needs proof.
- Required fix: Add pool close/drop/concurrent open tests under load.

### SM-AUD-0071 — Projection storage integrity is likely separate from memory integrity

- Severity: `P1`
- Confidence: `probable`
- Area: `projection`
- File: `semantic-memory/src/projection_storage.rs`:1
- Why it matters: Projection tables are substantial but not obviously covered by full integrity parity.
- Required fix: Add projection-level integrity: rows, derivations, episodes, imports, claim versions.

### SM-AUD-0072 — Projection query uses unwrap_or_default for missing claim/source IDs

- Severity: `P1`
- Confidence: `probable`
- Area: `projection`
- File: `semantic-memory/src/projection_storage_query.rs`:271
- Why it matters: Missing IDs become empty strings, hiding malformed rows.
- Required fix: Return structured parse/error instead of default empty identifiers.

### SM-AUD-0073 — Bridge transform uses unwrap_or_default, potentially hiding malformed optional payloads

- Severity: `P1`
- Confidence: `confirmed`
- Area: `bridge`
- File: `forge-memory-bridge/src/transform.rs`:301
- Why it matters: Defaulting in bridge paths can mask corrupted envelope fields.
- Required fix: Emit explicit transform error or warning with field name.

### SM-AUD-0074 — JSON import begins with from_str(...).ok()

- Severity: `P1`
- Confidence: `probable`
- Area: `import`
- File: `semantic-memory/src/json_compat_import.rs`:25
- Why it matters: Malformed JSON is converted into None instead of a typed parse error at the earliest boundary.
- Required fix: Preserve parse error and source payload hash in import receipt.

### SM-AUD-0075 — Legacy compatibility serializes with unwrap_or_default

- Severity: `P1`
- Confidence: `probable`
- Area: `import`
- File: `semantic-memory/src/projection_legacy_compat.rs`:127
- Why it matters: Serialization failure becomes empty string, which can look like valid empty JSON.
- Required fix: Return error on serialization failure.

### SM-AUD-0076 — HTTP embedder response body uses unwrap_or_default on error

- Severity: `P1`
- Confidence: `probable`
- Area: `security`
- File: `semantic-memory/src/embedder.rs`:127
- Why it matters: Failed response body read can erase useful diagnostic details.
- Required fix: Propagate body-read error or preserve status + partial diagnostics.

### SM-AUD-0077 — External embedder failure modes need retry/backoff/rate-limit policy

- Severity: `P1`
- Confidence: `probable`
- Area: `security`
- File: `semantic-memory/src/embedder.rs`:1
- Why it matters: Embedding APIs are external/unreliable; partial failures are currently risky for batch workflows.
- Required fix: Add retry policy, per-batch timeout, and idempotent transaction boundaries.

### SM-AUD-0312 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/reference/chunk.rs`:485
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0348 — INSERT OR IGNORE can hide duplicate/constraint bugs

- Severity: `P1`
- Confidence: `static`
- Area: `insert-or-ignore`
- File: `semantic-memory/src/db.rs`:197
- Why it matters: Ignored inserts can silently discard data or ordinal metadata.
- Required fix: Validate duplicates before insert or assert affected row count where required.

### SM-AUD-0349 — INSERT OR IGNORE can hide duplicate/constraint bugs

- Severity: `P1`
- Confidence: `static`
- Area: `insert-or-ignore`
- File: `semantic-memory/src/db.rs`:208
- Why it matters: Ignored inserts can silently discard data or ordinal metadata.
- Required fix: Validate duplicates before insert or assert affected row count where required.

### SM-AUD-0352 — INSERT OR IGNORE can hide duplicate/constraint bugs

- Severity: `P1`
- Confidence: `static`
- Area: `insert-or-ignore`
- File: `semantic-memory/src/db.rs`:370
- Why it matters: Ignored inserts can silently discard data or ordinal metadata.
- Required fix: Validate duplicates before insert or assert affected row count where required.

### SM-AUD-0353 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/db.rs`:558
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0354 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/db.rs`:581
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0358 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/db.rs`:653
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0359 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/db.rs`:705
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0360 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/db.rs`:715
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0363 — Storage byte cast is alignment/endian fragile

- Severity: `P1`
- Confidence: `confirmed`
- Area: `bytemuck-storage`
- File: `semantic-memory/src/db.rs`:788
- Why it matters: Casting SQLite bytes into f32 assumes alignment/native endian and can fail or misdecode.
- Required fix: Decode storage bytes via from_le_bytes/db::bytes_to_embedding.

### SM-AUD-0375 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/db.rs`:1400
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0377 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/src/documents.rs`:70
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0383 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/documents.rs`:234
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0384 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/src/documents.rs`:324
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0395 — INSERT OR IGNORE can hide duplicate/constraint bugs

- Severity: `P1`
- Confidence: `static`
- Area: `insert-or-ignore`
- File: `semantic-memory/src/episodes.rs`:285
- Why it matters: Ignored inserts can silently discard data or ordinal metadata.
- Required fix: Validate duplicates before insert or assert affected row count where required.

### SM-AUD-0398 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/episodes.rs`:410
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0399 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/episodes.rs`:414
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0400 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/src/episodes.rs`:417
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0466 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/src/lib.rs`:1123
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0470 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/src/lib.rs`:1254
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0473 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/src/lib.rs`:1314
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0475 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/src/lib.rs`:1374
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0477 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/src/lib.rs`:1434
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0506 — INSERT OR IGNORE can hide duplicate/constraint bugs

- Severity: `P1`
- Confidence: `static`
- Area: `insert-or-ignore`
- File: `semantic-memory/src/projection_legacy_compat.rs`:177
- Why it matters: Ignored inserts can silently discard data or ordinal metadata.
- Required fix: Validate duplicates before insert or assert affected row count where required.

### SM-AUD-0566 — debug_assert is not a release invariant

- Severity: `P1`
- Confidence: `confirmed`
- Area: `debug-assert`
- File: `semantic-memory/src/search.rs`:61
- Why it matters: debug_asserts disappear in release builds.
- Required fix: Use a normal check for correctness invariants.

### SM-AUD-0567 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/src/search.rs`:62
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0573 — Storage byte cast is alignment/endian fragile

- Severity: `P1`
- Confidence: `confirmed`
- Area: `bytemuck-storage`
- File: `semantic-memory/src/search.rs`:272
- Why it matters: Casting SQLite bytes into f32 assumes alignment/native endian and can fail or misdecode.
- Required fix: Decode storage bytes via from_le_bytes/db::bytes_to_embedding.

### SM-AUD-0595 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/tests/db_tests.rs`:36
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0596 — Dynamic SQL construction should be whitelisted

- Severity: `P1`
- Confidence: `static`
- Area: `dynamic-sql`
- File: `semantic-memory/tests/import_ugly_cases.rs`:33
- Why it matters: Identifier or limit formatting is safe only if all inputs are trusted and bounded.
- Required fix: Use whitelisted enums for identifiers and bind parameters for values.

### SM-AUD-0598 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/tests/knowledge_tests.rs`:555
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0599 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/tests/quantization.rs`:9
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0602 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/tests/quantization.rs`:46
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.

### SM-AUD-0603 — zip iteration can silently truncate mismatched collections

- Severity: `P1`
- Confidence: `static`
- Area: `zip-truncate`
- File: `semantic-memory/tests/quantization.rs`:68
- Why it matters: zip stops at the shorter collection, which is dangerous for batch write/update paths.
- Required fix: Pre-check lengths before zip unless truncation is intentional and documented.


---

## P2/P3 processing requirement

All P2/P3 rows in `02_FINDINGS_INDEX.csv` and `03_FINDINGS_INDEX.json` must be processed. Do not paste all of them into code comments. Instead, group them into repair buckets, patch them, and record closure in the JSON ledger.

Recommended buckets:

```text
unchecked casts
unwrap/expect/panic review
swallowed/defaulted errors
runtime-clock determinism
HashMap iteration determinism
executable-bit normalization
dynamic SQL identifier whitelist
root-doc cleanup
HNSW/keymap/pending-op robustness
zip/certifier hygiene
metadata/input bounds
projection/bridge consistency
search ranking stability
```

Each bucket must list:

```text
finding IDs covered
files touched
tests added/updated
acceptance command output
remaining risk, if any
```

---

## Definition of done

The pass is done only when all are true:

1. `cargo fmt --all --check` passes.
2. `cargo check --workspace --all-features` passes.
3. `cargo test --workspace --all-features` passes.
4. `cargo clippy --workspace --all-features -- -D warnings` passes.
5. `cargo doc --workspace --all-features --no-deps` passes.
6. `cargo test -p semantic-memory --no-default-features --features brute-force` passes.
7. `cargo test -p semantic-memory --no-default-features --features hnsw` passes.
8. Clean extracted archive builds/tests without parent workspace.
9. All P0/P1 findings are fixed or explicitly closed as not-a-bug with evidence.
10. All P2/P3 findings are processed and mapped in the JSON ledger.
11. New regression tests cover embedding count mismatch, non-finite embeddings, deletion stale state, HNSW sidecar corruption, filtered HNSW fallback, q8 policy, endian-safe vector decode, and namespace delete reports.
12. README/ARCHITECTURE/INTEGRITY/HNSW_PERSISTENCE/BENCHMARKS explain the stabilized system accurately.

---

## Final response format for Codex

When finished, report:

```text
Summary:
- What changed
- What invariants were added
- What tests were added
- What findings were closed
- What remains unresolved, if any

Commands run:
- <command>: pass/fail + key output

Finding closure:
- P0: X/Y fixed, Z not-a-bug
- P1: X/Y fixed, Z not-a-bug
- P2/P3: grouped closure summary

Files changed:
- path: reason

Known limitations:
- only honest remaining limitations
```

Do not claim success unless the gates ran and passed.
