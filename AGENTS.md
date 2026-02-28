# AGENTS.md — semantic-memory v0.2.0 Upgrade

## Agent System Overview

This upgrade is executed in 6 phases, each handled by a specialized agent role. Agents operate sequentially — each phase must compile and pass tests before the next begins. State is tracked in `.agent_context/state.json` and agent outputs are written to `.agent_context/{agent}.md`.

## Agent Role Detection

1. Check if `.agent_context/state.json` exists
   - If yes: Read it and activate the `current_agent`
   - Load context from previous agent's output in `.agent_context/`

2. If no state file, check for explicit agent prefix in prompt:
   - `@researcher:` → Dependency verification
   - `@foundation:` → Phase 1 foundation work
   - `@hnsw:` → Phase 2 HNSW module
   - `@integrator:` → Phase 3 wiring into MemoryStore
   - `@bugfixer:` → Phase 4 known bugs
   - `@tester:` → Phase 5 integration tests
   - `@reviewer:` → Phase 6 polish and verification

3. If no prefix, start with `@researcher`

## State Management

### State File: `.agent_context/state.json`

```json
{
  "schema_version": "1.0",
  "task": "semantic-memory-v0.2.0-upgrade",
  "current_agent": "@researcher",
  "agents_completed": [],
  "halted": false,
  "halt_reason": null,
  "notes": {}
}
```

### After Each Agent Completes

1. Write findings/summary to `.agent_context/{agent_name}.md`
2. Update `state.json`: move agent to `agents_completed`, set next `current_agent`
3. State your handoff explicitly:
```
## Handoff to @{next_agent}
{What they need to know from your work}
```

### Emergency Halt

Any agent can halt the workflow if they discover a blocking issue:
- Set `halted: true` in state.json
- Set `halt_reason` with clear description
- Create `.agent_context/HALT.md` with details
- Report to user immediately

Common halt triggers:
- `hnswlib-rs` not on crates.io and fallback also fails
- Existing crate source has breaking changes not covered in spec
- Feature flag combination causes irreconcilable compile conflicts

---

## @researcher — Dependency Verification

**Purpose:** Verify all external dependencies exist and match expected APIs before writing any code.

**Runs:** First, before any code changes.

**Permissions:**
- ✅ Read all project files
- ✅ Run `cargo search`, `cargo add` (dry-run)
- ✅ Read crate documentation online
- ✅ Write to `.agent_context/researcher.md`
- ❌ Do NOT modify any source files

**Tasks:**

1. **Verify `hnswlib-rs` on crates.io:**
   ```bash
   cargo search hnswlib-rs
   # Also try: cargo search hnswlib
   ```

2. **If found:** Add it as a dependency and check that these types exist:
   ```bash
   # Add to Cargo.toml temporarily
   cargo add hnswlib-rs --optional
   # Try to compile a minimal usage
   ```
   Verify these types/traits are exported:
   - `Hnsw<K, M>` — the graph
   - `HnswConfig` — configuration builder
   - `InMemoryQi8VectorStore` — quantized vector storage
   - `CosineQi8` (or `Cosine` that works with Qi8) — distance metric
   - `Qi8Ref` — borrowed quantized vector with `data`, `scale`, `zero_point`
   - `Hit<K>` — search result with `key` and `distance`
   - `save_to()` / `load_from()` on both graph and store

3. **If NOT found (or types don't match):**
   - Try `hnsw_rs` as fallback: `cargo search hnsw_rs`
   - Document which API is available in `.agent_context/researcher.md`
   - Note deviations from `HNSWLIB_RS_REFERENCE.md`
   - Do NOT halt unless both crates are unusable

4. **Check other dependencies:**
   - Verify `rusqlite` version supports everything we need
   - Verify no version conflicts between existing and new deps

5. **Write findings** to `.agent_context/researcher.md`:
   - Exact crate name and version to use
   - Any type name differences from reference
   - Any API differences that require spec adaptation
   - Recommended Cargo.toml dependency line

**Handoff to:** `@foundation`

**Success criteria:** We know exactly which crate to use, what its real API looks like, and the exact `Cargo.toml` line to add.

---

## @foundation — Phase 1: Foundation

**Purpose:** Create the new modules and types that don't depend on HNSW behavior. Everything compiles, all existing tests pass.

**Runs:** After @researcher confirms dependencies.

**Permissions:**
- ✅ Modify `Cargo.toml`
- ✅ Create new files: `src/storage.rs`, `src/quantize.rs`
- ✅ Modify `src/error.rs` (or wherever error types live) — add new variants only
- ✅ Modify `src/lib.rs` — add `mod storage; mod quantize;` declarations
- ✅ Create `tests/quantization.rs`
- ❌ Do NOT modify search.rs, knowledge.rs, documents.rs, conversation.rs yet
- ❌ Do NOT change any existing behavior

**Tasks:**

1. **Update `Cargo.toml`:**
   - Version → `0.2.0`
   - Add `[features]` section: `default = ["hnsw"]`, `hnsw = ["dep:hnswlib-rs"]`, `brute-force = []`
   - Add `hnswlib-rs` dependency (use exact line from @researcher findings)
   - Add compile_error for no-backend case

2. **Create `src/storage.rs`:**
   - `StoragePaths` struct with `base_dir`, `sqlite_path`, `hnsw_path`, `vectors_path`
   - `StoragePaths::new(base_dir)` constructor
   - See UPGRADE_SPEC.md Section 2

3. **Create `src/quantize.rs`:**
   - `QuantizedVector` struct: `data: Vec<i8>`, `scale: f32`, `zero_point: i8`
   - `Quantizer` struct with `dimensions: usize`
   - `Quantizer::quantize(&self, vector: &[f32]) -> Result<QuantizedVector>`
   - `Quantizer::dequantize(&self, qv: &QuantizedVector) -> Vec<f32>`
   - Handle edge case: constant vector (all same values)
   - See UPGRADE_SPEC.md Section 6

4. **Add error variants:**
   - `HnswError(String)`
   - `InvalidKey(String)`
   - `QuantizationError(String)`
   - `StorageError(String)`
   - `IntegrityError { in_sqlite_not_hnsw, in_hnsw_not_sqlite }`
   - See UPGRADE_SPEC.md Appendix C

5. **Write quantization tests** (`tests/quantization.rs`):
   - Round-trip accuracy (max error < 0.01 for normalized vectors)
   - Cosine similarity ranking preservation
   - Constant vector edge case
   - Extreme values edge case
   - See UPGRADE_SPEC.md Section 15 for exact test implementations

6. **Verify:**
   ```bash
   cargo build
   cargo build --no-default-features --features "brute-force"
   cargo test
   cargo clippy
   ```

**Handoff to:** `@hnsw`

**Success criteria:** Both feature flag combinations compile. All existing tests pass. Quantization tests pass. No clippy warnings.

---

## @hnsw — Phase 2: HNSW Module

**Purpose:** Create the `src/hnsw.rs` wrapper around hnswlib-rs. Test HNSW operations in isolation, independent of MemoryStore.

**Runs:** After @foundation confirms compilation.

**Permissions:**
- ✅ Create `src/hnsw.rs`
- ✅ Modify `src/lib.rs` — add `mod hnsw;` (gated on feature)
- ✅ Write HNSW-specific unit tests (in hnsw.rs or tests/)
- ❌ Do NOT modify MemoryStore, search.rs, or any existing modules yet

**Context from @researcher:** Read `.agent_context/researcher.md` for the actual crate API. The types in `HNSWLIB_RS_REFERENCE.md` may differ from reality. Adapt to whatever @researcher found.

**Tasks:**

1. **Create `src/hnsw.rs`** (gated: `#[cfg(feature = "hnsw")]`):

   Types to implement:
   - `HnswConfig` — m, ef_construction, ef_search, dimensions, max_elements with defaults
   - `HnswIndex` — wrapper struct (thread-safe, cheaply cloneable via Arc)
   - `HnswHit` — key: String, distance: f32, with `similarity()` and `parse_key()` methods

   Methods to implement:
   - `HnswIndex::new(config) -> Result<Self>`
   - `HnswIndex::load(hnsw_path, vectors_path, config) -> Result<Self>`
   - `HnswIndex::save(hnsw_path, vectors_path) -> Result<()>`
   - `HnswIndex::insert(key: String, vector: &[f32]) -> Result<()>` — quantizes internally
   - `HnswIndex::delete(key: &str) -> Result<()>`
   - `HnswIndex::update(key: String, vector: &[f32]) -> Result<()>`
   - `HnswIndex::search(query: &[f32], top_k: usize) -> Result<Vec<HnswHit>>`
   - `HnswIndex::len() -> usize`
   - `HnswIndex::is_empty() -> bool`

   Key format: `"{domain}:{id}"` — e.g., `"fact:42"`, `"chunk:17"`, `"msg:99"`

   See UPGRADE_SPEC.md Section 5 for full type definitions and design rationale.

2. **Handle API adaptation:**
   - If hnswlib-rs uses different names → adapt internally, preserve our interface
   - If hnswlib-rs doesn't have Qi8 → use f32 vectors with our Quantizer for manual SQ8
   - If using hnsw_rs fallback → keys are usize, need internal HashMap<String, usize> mapping
   - **The external interface (HnswIndex, HnswHit, HnswConfig) must match the spec regardless of backend**

3. **Test concurrency:**
   Write a test that inserts from one thread while searching from another. This verifies hnswlib-rs's claimed concurrent safety. If it deadlocks or panics → wrap inner types in RwLock and note this in handoff.

4. **Test persistence:**
   - Create index, insert 100 items, save to temp files
   - Load from files, verify search returns same results
   - Delete sidecar files, verify clean error or empty index

5. **Verify:**
   ```bash
   cargo test --features hnsw
   cargo build --no-default-features --features "brute-force"  # Must still compile without HNSW
   ```

**Handoff to:** `@integrator`

**Success criteria:** HnswIndex works in isolation. Insert → search → delete → persistence all pass. Concurrency verified. Both feature flag combinations still compile.

---

## @integrator — Phase 3: Wire Into MemoryStore

**Purpose:** Connect the HNSW module to MemoryStore. This is the largest phase — every write path and the search pipeline must be modified.

**Runs:** After @hnsw confirms HNSW module works in isolation.

**Permissions:**
- ✅ Modify `src/lib.rs` — MemoryStore struct, constructors, MemoryConfig, Drop
- ✅ Modify `src/db.rs` — add migration v3 (hnsw_metadata table)
- ✅ Modify `src/search.rs` — search dispatch, SearchConfig, reranking
- ✅ Modify `src/knowledge.rs` — HNSW insert/delete in write paths
- ✅ Modify `src/documents.rs` — HNSW insert/delete in write paths
- ✅ Modify `src/conversation.rs` — HNSW insert/delete in write paths
- ✅ Modify `src/embedder.rs` — add dimensions() if missing

**Tasks:**

Work through UPGRADE_SPEC.md Sections 7-13 systematically:

1. **`src/lib.rs` — MemoryStore changes (Section 9):**
   - Add `paths: StoragePaths` field
   - Add `#[cfg(feature = "hnsw")] hnsw_index: HnswIndex` field
   - Add `#[cfg(feature = "hnsw")] quantizer: Quantizer` field
   - Change `open()` signature: takes directory path, not file path
   - Add `open_with_embedder()` for framework integration
   - Add `open_with_paths()` for custom layouts
   - Implement initialization logic: create dir, open SQLite, load or create HNSW
   - Add `rebuild_hnsw_index()` — loads all f32 from SQLite, bulk inserts into fresh HNSW
   - Add `flush_hnsw()` — explicit persistence
   - Add `verify_index_integrity()` — cross-check SQLite vs HNSW
   - Implement `Drop` — synchronous `save()` call (NOT async)
   - Add HnswConfig to MemoryConfig

2. **`src/db.rs` — Migration (Section 8):**
   - Add `hnsw_metadata` table in next migration version
   - Columns: `key TEXT PRIMARY KEY, value TEXT NOT NULL`

3. **`src/search.rs` — Search Pipeline (Section 7):**
   - Add new fields to SearchConfig: `hnsw_candidates`, `rerank_with_f32`, `force_brute_force`, `ef_search_override`
   - Add SearchDomain enum if not already present
   - Implement vector_search dispatch: HNSW by default, brute-force if force_brute_force or no HNSW feature
   - Move existing brute-force code behind `#[cfg(feature = "brute-force")]`
   - Implement domain filtering on HNSW results (parse key prefix)
   - Implement optional f32 reranking (load from SQLite, exact cosine)
   - RRF fusion algorithm is UNCHANGED — just receives hits from different source

4. **`src/knowledge.rs` — Write Paths (Section 11):**
   - `add_fact()`: after SQLite insert + FTS bridge, insert into HNSW with key `"fact:{id}"`
   - `delete_fact()`: after SQLite delete, delete from HNSW
   - `update_fact()` (if exists): after SQLite update, update HNSW
   - `delete_namespace()`: delete all matching keys from HNSW
   - `reembed_all()`: after completion, call `rebuild_hnsw_index()`

5. **`src/documents.rs` — Write Paths (Section 12):**
   - `ingest_document()`: for each chunk, insert into HNSW with key `"chunk:{id}"`
   - Document deletion: iterate chunks, delete each from HNSW

6. **`src/conversation.rs` — Write Paths (Section 13):**
   - `add_message_embedded()`: insert into HNSW with key `"msg:{id}"`
   - Only for messages that actually get embedded (respect existing threshold logic)

7. **`src/embedder.rs` — dimensions() (Section 10):**
   - Add `fn dimensions(&self) -> usize` to Embedder trait if not present
   - Implement for OllamaEmbedder, MockEmbedder
   - Add dimension validation in MemoryStore constructor

8. **Verify everything compiles in all feature combinations:**
   ```bash
   cargo check --features "hnsw"
   cargo check --features "brute-force"
   cargo check --features "hnsw,brute-force"
   cargo check --no-default-features --features "brute-force"
   cargo test
   ```

**Handoff to:** `@bugfixer`

**Success criteria:** All feature flag combinations compile. Existing tests pass (may need minor updates for API change from file path → directory path). The search pipeline dispatches to HNSW. Write paths insert into both SQLite and HNSW.

---

## @bugfixer — Phase 4: Bug Fixes

**Purpose:** Fix the 4 bugs identified during the v0.1.0 review.

**Runs:** After @integrator confirms integration compiles.

**Permissions:**
- ✅ Modify `src/knowledge.rs` (delete_namespace fix)
- ✅ Modify `src/db.rs` or `src/lib.rs` (raw_execute fix)
- ✅ Modify `src/search.rs` (decode_buf removal, FTS sanitization)
- ❌ Do NOT change any HNSW-related code

**Tasks:**

See UPGRADE_SPEC.md Section 14 for exact fixes:

1. **Bug 1: `delete_namespace` atomicity (MEDIUM)**
   - Collect all fact_ids first
   - Wrap all deletes (FTS bridge, FTS content, facts) in single unchecked_transaction
   - HNSW deletes happen AFTER SQLite transaction commits

2. **Bug 2: `raw_execute` exposure (MINOR)**
   - Change from `pub #[doc(hidden)]` to `#[cfg(any(test, feature = "testing"))]`

3. **Bug 3: Vestigial `decode_buf` (MINOR)**
   - Remove the `let mut decode_buf = Vec::new();` and `decode_buf.clear();` lines
   - Only relevant if brute-force feature is enabled

4. **Bug 4: FTS sanitization edge cases (MINOR)**
   - Improve `sanitize_fts_query()` to strip bare AND/OR/NOT/NEAR tokens
   - Remove FTS5 special characters
   - Handle empty result after sanitization

5. **Verify:**
   ```bash
   cargo test --all-features
   cargo clippy --all-features
   ```

**Handoff to:** `@tester`

**Success criteria:** All 4 bugs fixed. All existing tests pass. No new clippy warnings.

---

## @tester — Phase 5: Integration Tests

**Purpose:** Write comprehensive integration tests that verify the entire upgrade works end-to-end.

**Runs:** After @bugfixer confirms fixes.

**Permissions:**
- ✅ Create test files in `tests/` directory
- ✅ Modify existing test helpers if needed for new API (directory path)
- ❌ Do NOT modify source code — if a test reveals a bug, document it in handoff and let @reviewer decide

**Tasks:**

Create these test files. See UPGRADE_SPEC.md Section 15 for exact test implementations:

1. **`tests/hnsw_integration.rs`** — Full pipeline tests (require `hnsw` feature):
   - Insert → search → verify results contain expected content
   - Multi-domain search (facts + chunks + messages)
   - Namespace filtering
   - Search empty store returns empty vec
   - Delete then search — deleted item absent from results

2. **`tests/storage_lifecycle.rs`** — Persistence tests (require `hnsw` feature):
   - Open → populate → flush → close → reopen → search → results match
   - Delete sidecar files → reopen → auto-rebuild → search works
   - Verify hnsw_metadata table is populated after save

3. **`tests/concurrent_access.rs`** — Multi-task safety:
   - Spawn inserter task + searcher task on shared Arc<MemoryStore>
   - Both complete without panic or deadlock
   - Use tokio::time::timeout to detect deadlocks (fail after 30s)

4. **`tests/brute_force_parity.rs`** — Backend comparison (require BOTH features):
   - Populate store with 50 facts
   - Search with HNSW, search with force_brute_force
   - Top-1 result should match
   - At least 3/5 of top-5 should overlap
   - Gate entire file: `#![cfg(all(feature = "hnsw", feature = "brute-force"))]`

5. **`tests/quantization.rs`** should already exist from @foundation — verify it still passes.

6. **Run everything:**
   ```bash
   cargo test --all-features
   cargo test --features "hnsw"
   cargo test --no-default-features --features "brute-force"
   ```

**Handoff to:** `@reviewer`

**Success criteria:** All tests pass in all feature flag combinations. No panics, no deadlocks, no timeouts. Test count increased significantly from v0.1.0.

**IMPORTANT:** Tests are LOCKED after creation. If a test fails, it means the source code has a bug, not the test. Document failing tests in your handoff — do not modify source to make tests pass.

---

## @reviewer — Phase 6: Polish & Verification

**Purpose:** Final review, documentation, and verification that everything meets the spec.

**Runs:** Last.

**Permissions:**
- ✅ Modify `README.md`
- ✅ Fix any clippy warnings
- ✅ Fix minor issues found during review
- ✅ Run all verification commands

**Tasks:**

1. **Update README.md:**
   - Document the directory-based storage convention
   - Show new `open()` API with directory path
   - Document feature flags: `hnsw` (default), `brute-force`
   - Show `open_with_embedder()` for framework integration
   - Add performance characteristics section
   - Add migration note (old file-based → new directory-based)

2. **Run full lint suite:**
   ```bash
   cargo clippy --all-features -- -D warnings
   ```

3. **Verify all feature combinations compile:**
   ```bash
   cargo check --features "hnsw"
   cargo check --features "brute-force"
   cargo check --features "hnsw,brute-force"
   cargo check --no-default-features --features "brute-force"
   ```
   And the compile_error case:
   ```bash
   # This should FAIL to compile with clear error message:
   cargo check --no-default-features 2>&1 | grep "At least one search backend"
   ```

4. **Run full test suite:**
   ```bash
   cargo test --all-features
   ```

5. **Verify docs build:**
   ```bash
   cargo doc --all-features --no-deps
   ```

6. **Checklist verification** against UPGRADE_SPEC.md Appendix B:
   - [ ] All new files created
   - [ ] All modified files updated
   - [ ] All test files written
   - [ ] Version in Cargo.toml is 0.2.0
   - [ ] README is current

7. **Write final summary** to `.agent_context/reviewer.md`:
   - Files changed (count + list)
   - Tests added (count)
   - Lines of code added/removed
   - Any spec deviations and reasons
   - Any known issues or TODOs for v0.3.0

**Success criteria:** Zero clippy warnings. All feature combos compile. All tests pass. README is accurate. Spec checklist complete.

---

## Recovery Procedures

### Compile Error After Modification

If a phase introduces a compile error that the agent can't resolve:
1. `git stash` the changes
2. Document the error in `.agent_context/{agent}.md`
3. Set `halted: true` in state.json
4. The error likely means the spec assumed something about the existing code that isn't true — the spec may need updating

### Test Failure

If an integration test fails:
1. Run the test in isolation with verbose output: `cargo test test_name -- --nocapture`
2. Document the failure in `.agent_context/{agent}.md`
3. If the fix is obvious (e.g., wrong assertion value), apply it
4. If the fix requires design changes, halt and document

### Dependency Not Found

If neither `hnswlib-rs` nor `hnsw_rs` works:
1. @researcher should document what was tried
2. Consider implementing a minimal HNSW using the spec's `HnswIndex` interface as the contract
3. The `instant-distance` crate is another pure-Rust HNSW option
4. As absolute last resort, keep brute-force as default and implement HNSW later
