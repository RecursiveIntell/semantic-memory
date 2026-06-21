# semantic-memory Deep Static Audit — 2026-05-11

## Scope and method

This is an adversarial static audit of the uploaded `semantic-memory-generic-rust-next-codex-context-20260511.zip` package. It combines manual source review with generated static-finding passes. I could not run `cargo check`, `cargo test`, or `cargo clippy` in this container because Cargo is not installed here; treat this as a source-level defect hunt, not an executed test report.

The package-level certifier reported 134 included files, 85 Rust files, 33 Markdown files, 4 Cargo manifests, 0 workspace manifests, 3 external Cargo roots, successful cargo metadata from the parent `/home/sikmindz/Coding/Libraries` workspace, and 0 certifier findings.

## Read this before patching

- Not every row below is a confirmed runtime bug. I classify rows as `confirmed`, `probable`, `static`, or `heuristic`.
- The P0/P1 manual rows are the ones I would patch before TurboQuant integration.
- The generated rows are intentionally noisy. Their purpose is to feed broad Codex/audit passes, not to assert that every line is broken.

## Counts

- Total findings: **608**
- By severity: `P0`=27, `P1`=83, `P2`=296, `P3`=202
- By confidence: `confirmed`=196, `probable`=21, `static`=391
- Top areas: `numeric-cast`=110, `permissions`=107, `error-default`=95, `unwrap-expect`=80, `runtime-clock`=56, `docs`=25, `hnsw`=22, `hashmap-order`=20, `zip-truncate`=13, `dynamic-sql`=12, `search`=11, `embedding`=6, `insert-or-ignore`=5, `delete/integrity`=4, `db`=4

## Stop-ship findings

### SM-AUD-0001 [P0 / confirmed] Archive is not hermetic despite passing certifier
- Location: `semantic-memory-generic-rust-next-codex-context-20260511.report.md:1`
- Why: Cargo metadata succeeded from parent /Coding/Libraries workspace with 326 packages/30 workspace members while the archive has zero workspace manifests. A clean extractor may fail or resolve differently.
- Fix: Add an archive-root Cargo.toml workspace, or remove workspace-only dependency/lint reliance; validate from fresh extraction.

### SM-AUD-0002 [P0 / confirmed] No packaged root workspace manifest for included local crates
- Location: `semantic-memory/Cargo.toml:1`
- Why: The zip includes semantic-memory plus three path roots, but no top-level workspace manifest tying them together.
- Fix: Generate a root Cargo.toml with members semantic-memory, stack-ids, semantic-memory-forge, forge-memory-bridge.

### SM-AUD-0003 [P0 / confirmed] Multiple Cargo.lock files create ambiguous dependency source of truth
- Location: `semantic-memory/Cargo.lock:1`
- Why: Each included package has its own lockfile; without a packaged root workspace, dependency resolution can differ between crates.
- Fix: Use one workspace lockfile at archive root for review builds or document crate-by-crate build commands.

### SM-AUD-0004 [P0 / confirmed] Document ingest silently truncates chunks on embedder batch-count mismatch
- Location: `semantic-memory/src/documents.rs:324`
- Why: text_chunks.iter().zip(embeddings.iter()) drops chunks if the embedder returns fewer vectors, and ignores extra vectors.
- Fix: Centralize embed_batch validation: returned len must equal requested len before any write.

### SM-AUD-0005 [P0 / confirmed] Fact re-embedding silently truncates on batch-count mismatch
- Location: `semantic-memory/src/lib.rs:1254`
- Why: Batch rows are zipped with embeddings; fewer embeddings still increments fact_count by batch.len().
- Fix: Fail loudly on batch-count mismatch before constructing updates.

### SM-AUD-0006 [P0 / confirmed] Chunk re-embedding silently truncates on batch-count mismatch
- Location: `semantic-memory/src/lib.rs:1314`
- Why: Batch rows are zipped with embeddings; missing embeddings leave stale rows while progress counters report success.
- Fix: Fail loudly on batch-count mismatch before update transaction.

### SM-AUD-0007 [P0 / confirmed] Message re-embedding silently truncates on batch-count mismatch
- Location: `semantic-memory/src/lib.rs:1374`
- Why: Batch rows are zipped with embeddings; message embeddings can be partially refreshed without being reported.
- Fix: Fail loudly on batch-count mismatch before update transaction.

### SM-AUD-0008 [P0 / confirmed] Episode re-embedding silently truncates on batch-count mismatch
- Location: `semantic-memory/src/lib.rs:1434`
- Why: Batch rows are zipped with embeddings; missing episode embeddings can leave stale recall state.
- Fix: Fail loudly on batch-count mismatch before update transaction.

### SM-AUD-0009 [P0 / confirmed] Public embedding validation is dimension-only
- Location: `semantic-memory/src/lib.rs:536`
- Why: validate_embedding_dimensions checks length but not NaN/Inf, while HNSW later rejects non-finite values; SQLite can still store bad f32 blobs.
- Fix: Replace with validate_embedding that checks dimensions and all components finite.

### SM-AUD-0010 [P0 / confirmed] delete_document does not explicitly clean episode derived state
- Location: `semantic-memory/src/documents.rs:109`
- Why: Document deletion removes chunks/docs but not episode_causes, episodes_fts, episodes_rowid_map, or episode HNSW ops before cascade/cleanup boundaries.
- Fix: Collect episode_ids first and delete all episode derived surfaces plus queued HNSW deletes in one transaction.

### SM-AUD-0011 [P0 / probable] delete_document can leave stale HNSW episode keys
- Location: `semantic-memory/src/documents.rs:109`
- Why: If episodes are cascaded from documents, HNSW sidecar does not know about episode deletes unless explicit pending ops are queued.
- Fix: Queue Delete for every episode:{episode_id} before deleting the document.

### SM-AUD-0012 [P0 / confirmed] Vector scan uses bytemuck::try_cast_slice on SQLite Vec<u8>
- Location: `semantic-memory/src/search.rs:272`
- Why: SQLite blobs are byte vectors; casting requires alignment and native endian assumptions. Valid blobs may fail or decode incorrectly on non-little-endian targets.
- Fix: Use db::bytes_to_embedding for all blob decoding; avoid bytemuck on storage bytes.

### SM-AUD-0013 [P0 / confirmed] HNSW sidecar loader allocates raw byte_len from file without cap
- Location: `semantic-memory/src/hnsw.rs:501`
- Why: A corrupt sidecar can declare a huge byte length and trigger large allocation before validation.
- Fix: Require byte_len == dimensions*4 and <= configured max before allocation.

### SM-AUD-0014 [P0 / confirmed] HNSW data format stores dimensions using usize
- Location: `semantic-memory/src/hnsw.rs:460`
- Why: usize serialization is platform-width dependent; sidecar created on 64-bit may not load on 32-bit and is not a stable portable format.
- Fix: Use fixed-width u32/u64 little-endian fields with versioned header.

### SM-AUD-0015 [P0 / probable] HNSW save is not atomic
- Location: `semantic-memory/src/hnsw.rs:142`
- Why: file_dump writes directly; crash or process kill can leave partial graph/data sidecars.
- Fix: Write to temp files, fsync, then atomic rename graph/data/keymap as a set.

### SM-AUD-0016 [P0 / confirmed] Pending HNSW mutations are applied before sidecar save succeeds
- Location: `semantic-memory/src/hnsw_ops.rs:184`
- Why: If save fails after mutating in-memory index, pending ops remain and can be applied again, creating duplicate tombstones/nodes.
- Fix: Build/save a snapshot or roll back in-memory mutations on save failure.

### SM-AUD-0017 [P0 / confirmed] Pending upsert calls insert instead of update
- Location: `semantic-memory/src/hnsw_ops.rs:192`
- Why: Upsert on an existing key allocates a new node and tombstones the old node, increasing graph bloat on repeated retries.
- Fix: Use update() or replace semantics for existing keys; dedupe pending ops by key.

### SM-AUD-0018 [P0 / confirmed] HNSW sidecar save clones Arc while graph can still mutate
- Location: `semantic-memory/src/lib.rs:669`
- Why: Cloning HnswIndex does not snapshot the underlying graph; concurrent writes can mutate while save is in progress.
- Fix: Hold exclusive lock during save or introduce immutable snapshot serialization.

### SM-AUD-0019 [P0 / confirmed] Graph sidecar validation only checks non-empty file
- Location: `semantic-memory/src/hnsw.rs:429`
- Why: validate_graph_sidecar does not verify magic/version/consistency with data/keymap.
- Fix: Validate graph/data/keymap together with checksums and stored dimensions/counts.

### SM-AUD-0020 [P0 / confirmed] HNSW integrity is count-based, not key-level
- Location: `semantic-memory/src/lib.rs:730`
- Why: Equal counts can hide wrong mappings, stale IDs, or wrong source types.
- Fix: Verify each keymap key maps to a live row and each live embedded row has a matching key.

### SM-AUD-0021 [P0 / confirmed] FTS integrity count checks use dynamic table names and count parity only
- Location: `semantic-memory/src/db.rs:1400`
- Why: Count parity can pass despite rowid/content mismatches.
- Fix: Perform key-level rowid_map/content checks for every FTS-backed table.

### SM-AUD-0022 [P0 / confirmed] cosine_similarity truncates mismatched vectors in release builds
- Location: `semantic-memory/src/search.rs:61`
- Why: debug_assert_eq is disabled in release; zip then compares only the shorter length.
- Fix: Return Result or validate dimensions before zip in all builds.

### SM-AUD-0023 [P0 / confirmed] cosine similarity accepts non-finite stored/query vectors
- Location: `semantic-memory/src/search.rs:290`
- Why: NaN/Inf can produce NaN scores and unstable sorting.
- Fix: Validate finite vectors before storage and skip/error on non-finite during reads.

### SM-AUD-0024 [P0 / confirmed] HNSW filtered search can return empty results without brute-force fallback
- Location: `semantic-memory/src/search.rs:963`
- Why: HNSW gets global candidates before namespace/session/source-type filters; if filtered candidates are removed, valid rows outside the top candidate pool are missed.
- Fix: Overfetch adaptively after filters or fallback to brute force when post-filter hits < k.

### SM-AUD-0025 [P0 / confirmed] conversation HNSW search runs blocking CPU work on async thread
- Location: `semantic-memory/src/conversation.rs:683`
- Why: It directly holds the HNSW read lock and searches in async context, unlike main hnsw_search_blocking.
- Fix: Route through spawn_blocking helper.

### SM-AUD-0026 [P0 / confirmed] delete_fact does not clean episode_causes references
- Location: `semantic-memory/src/knowledge.rs:143`
- Why: Deleting a fact queues HNSW delete and removes FTS, but episodes may still cite the fact as a cause.
- Fix: Delete or mark episode_causes rows referencing the fact and update affected episode search/provenance.

### SM-AUD-0027 [P0 / probable] update_fact does not update dependent episode/projection search text
- Location: `semantic-memory/src/knowledge.rs:181`
- Why: Fact content changes can make derived episode/projection references semantically stale.
- Fix: Record invalidation edges or recompute affected derived search surfaces.

## High-priority P1 findings

- **SM-AUD-0028** `api` `confirmed` `semantic-memory/src/knowledge.rs:236` — delete_namespace returns only fact count despite deleting many entity types Fix: Return NamespaceDeleteReport with counts per entity/table/op.
- **SM-AUD-0029** `hnsw` `confirmed` `semantic-memory/src/lib.rs:369` — Open-time HNSW rebuild/degrade policy is implicit Fix: Expose HnswStartupPolicy and health status.
- **SM-AUD-0030** `hnsw` `confirmed` `semantic-memory/src/lib.rs:403` — SQL errors while counting embeddings are swallowed as zero Fix: Propagate DB errors during integrity decisions.
- **SM-AUD-0031** `hnsw` `confirmed` `semantic-memory/src/lib.rs:455` — Orphan-count SQL errors are swallowed as zero Fix: Propagate the error or force degraded/rebuild state.
- **SM-AUD-0032** `hnsw` `confirmed` `semantic-memory/src/hnsw.rs:310` — Missing hnsw_keymap table silently leaves loaded graph without keys Fix: Treat graph+missing keymap as degraded/rebuild, not clean load.
- **SM-AUD-0033** `hnsw` `confirmed` `semantic-memory/src/hnsw.rs:353` — Malformed next_id metadata falls back silently Fix: Report error or mark sidecar stale when metadata is malformed.
- **SM-AUD-0034** `hnsw` `confirmed` `semantic-memory/src/hnsw.rs:222` — HNSW len can report nonzero even when keymap is empty Fix: Expose separate graph_len and live_key_count; search should use resolvable key count.
- **SM-AUD-0035** `hnsw` `confirmed` `semantic-memory/src/hnsw.rs:185` — Tombstone overfetch is too naive Fix: Iteratively overfetch until enough live hits or graph exhausted.
- **SM-AUD-0036** `hnsw` `confirmed` `semantic-memory/src/hnsw.rs:250` — deleted_ratio may divide using graph count that includes unreachable/unmapped points Fix: Compute deleted/live ratios from verified keymap state.
- **SM-AUD-0037** `hnsw` `confirmed` `semantic-memory/src/hnsw.rs:496` — u64 node id is cast to usize without range check Fix: TryFrom<u64> with explicit error.
- **SM-AUD-0038** `hnsw` `confirmed` `semantic-memory/src/hnsw.rs:383` — insert ignores return/status from hnsw_rs graph.insert Fix: Wrap insert in catch_unwind if needed and use API result if available; update keymap only after success.
- **SM-AUD-0039** `hnsw` `confirmed` `semantic-memory/src/hnsw_ops.rs:71` — HNSW rebuild silently skips invalid fact embeddings Fix: Count skipped rows and return degraded integrity finding.
- **SM-AUD-0040** `hnsw` `confirmed` `semantic-memory/src/hnsw_ops.rs:89` — HNSW rebuild silently skips invalid chunk embeddings Fix: Count skipped rows and expose rebuild diagnostics.
- **SM-AUD-0041** `hnsw` `confirmed` `semantic-memory/src/hnsw_ops.rs:107` — HNSW rebuild silently skips invalid message embeddings Fix: Count skipped rows and expose rebuild diagnostics.
- **SM-AUD-0042** `hnsw` `confirmed` `semantic-memory/src/hnsw_ops.rs:125` — HNSW rebuild silently skips invalid episode embeddings Fix: Count skipped rows and expose rebuild diagnostics.
- **SM-AUD-0043** `hnsw` `confirmed` `semantic-memory/src/hnsw_ops.rs:205` — clear_pending_index_ops is all-or-nothing per processed key list after sidecar save Fix: Use transactional state machine with op generation numbers and idempotent upsert.
- **SM-AUD-0044** `q8` `confirmed` `semantic-memory/src/lib.rs:1256` — q8 optionality conflicts with integrity expectations Fix: Define compressed vectors as mandatory or optional; align write, repair, and integrity.
- **SM-AUD-0045** `q8` `confirmed` `semantic-memory/src/quantize.rs:1` — q8 baseline lacks explicit versioned storage envelope Fix: Add a vector-codec envelope with codec, version, dim, checksum, and params.
- **SM-AUD-0046** `search` `confirmed` `semantic-memory/src/search.rs:102` — Invalid timestamp becomes maximally fresh Fix: Treat invalid timestamps as no recency contribution or stale.
- **SM-AUD-0047** `search` `confirmed` `semantic-memory/src/search.rs:85` — recency scoring uses wall-clock inside ranking Fix: Inject clock into SearchConfig or query context.
- **SM-AUD-0048** `search` `confirmed` `semantic-memory/src/lib.rs:851` — candidate_pool_size.max(k * 3) can overflow Fix: Use k.saturating_mul(3) and cap top_k.
- **SM-AUD-0049** `search` `confirmed` `semantic-memory/src/lib.rs:954` — Second candidate_pool_size.max(k * 3) overflow surface Fix: Use saturating_mul and configured max_top_k.
- **SM-AUD-0050** `search` `confirmed` `semantic-memory/src/lib.rs:1033` — Third candidate_pool_size.max(k * 3) overflow surface Fix: Use saturating_mul and configured max_top_k.
- **SM-AUD-0051** `search` `confirmed` `semantic-memory/src/conversation.rs:683` — conversation candidate_pool_size.max(k * 3) overflow surface Fix: Use saturating_mul and configured max_top_k.
- **SM-AUD-0052** `conversation` `confirmed` `semantic-memory/src/conversation.rs:172` — Unknown token counts are treated as zero in budget selection Fix: Recompute missing token_count or treat unknown as conservative upper bound.
- **SM-AUD-0053** `conversation` `confirmed` `semantic-memory/src/conversation.rs:173` — Token budget addition can overflow u32 Fix: Use checked_add/saturating_add and error or clamp.
- **SM-AUD-0054** `conversation` `confirmed` `semantic-memory/src/conversation.rs:185` — session_token_count casts negative SQL sum to u64 Fix: Validate nonnegative aggregate before conversion.
- **SM-AUD-0055** `validation` `probable` `semantic-memory/src/conversation.rs:63` — Session/channel identifiers are not consistently length/whitespace validated Fix: Centralize validation for session_id, channel, namespace, title, source URI.
- **SM-AUD-0056** `validation` `probable` `semantic-memory/src/documents.rs:315` — Document title/source/metadata size validation is weaker than content validation Fix: Add max lengths and metadata byte caps.
- **SM-AUD-0057** `validation` `probable` `semantic-memory/src/episodes.rs:393` — Episode search limit is unbounded Fix: Cap limit using config max_top_k/max_query_rows.
- **SM-AUD-0058** `api` `confirmed` `semantic-memory/src/episodes.rs:437` — search_episodes drops episode_id and returns document_id Fix: Return episode_id plus document_id or a typed EpisodeSearchResult.
- **SM-AUD-0059** `diagnostics` `confirmed` `semantic-memory/src/episodes.rs:452` — Episode parse errors report document_id instead of episode_id Fix: Use episode_id in parse helpers.
- **SM-AUD-0060** `episodes` `confirmed` `semantic-memory/src/episodes.rs:285` — INSERT OR IGNORE collapses duplicate cause IDs silently Fix: Validate and reject duplicate cause_ids or preserve multiplicity intentionally.
- **SM-AUD-0061** `episodes` `confirmed` `semantic-memory/src/episodes.rs:360` — update_episode_outcome cannot clear experiment_id Fix: Add explicit clear operation or Option<Option<String>> semantics.
- **SM-AUD-0062** `keys` `probable` `semantic-memory/src/hnsw.rs:58` — HNSW key parsing split_once(:) is fragile for IDs containing colon Fix: Use structured key encoding or reject colon in IDs.
- **SM-AUD-0063** `keys` `probable` `semantic-memory/src/search.rs:119` — Message dedup key uses session_id:message_id delimiter Fix: Use tuple type internally or escaped/keyed serialization.
- **SM-AUD-0064** `foreign-keys` `probable` `semantic-memory/src/db.rs:1` — Correctness depends on PRAGMA foreign_keys being enabled for every pooled connection Fix: Assert PRAGMA foreign_keys=ON after every connection checkout.
- **SM-AUD-0065** `db` `probable` `semantic-memory/src/db.rs:558` — PRAGMA max_page_count computed with dynamic formatting Fix: Validate max_page_count range before execute.
- **SM-AUD-0066** `db` `confirmed` `semantic-memory/src/db.rs:705` — Dynamic table_info table name formatting relies on internal callers only Fix: Make table an enum or whitelist.
- **SM-AUD-0067** `db` `confirmed` `semantic-memory/src/db.rs:715` — Dynamic ALTER TABLE formatting relies on internal table/column whitelists Fix: Make migration table/column identifiers enum-backed.
- **SM-AUD-0068** `db` `confirmed` `semantic-memory/src/db.rs:1400` — Dynamic SELECT COUNT table name relies on internal map table list Fix: Use enum/constant-only function signature.
- **SM-AUD-0069** `sqlite` `probable` `semantic-memory/src/db.rs:1` — SQLite WAL/checkpoint/backpressure policy not visible in archive-level docs Fix: Document and test WAL mode, busy timeout, checkpoint, and backup semantics.
- **SM-AUD-0070** `pool` `probable` `semantic-memory/src/pool.rs:1` — Connection pool shutdown/poison behavior needs stress coverage Fix: Add pool close/drop/concurrent open tests under load.
- **SM-AUD-0071** `projection` `probable` `semantic-memory/src/projection_storage.rs:1` — Projection storage integrity is likely separate from memory integrity Fix: Add projection-level integrity: rows, derivations, episodes, imports, claim versions.
- **SM-AUD-0072** `projection` `probable` `semantic-memory/src/projection_storage_query.rs:271` — Projection query uses unwrap_or_default for missing claim/source IDs Fix: Return structured parse/error instead of default empty identifiers.
- **SM-AUD-0073** `bridge` `confirmed` `forge-memory-bridge/src/transform.rs:301` — Bridge transform uses unwrap_or_default, potentially hiding malformed optional payloads Fix: Emit explicit transform error or warning with field name.
- **SM-AUD-0074** `import` `probable` `semantic-memory/src/json_compat_import.rs:25` — JSON import begins with from_str(...).ok() Fix: Preserve parse error and source payload hash in import receipt.
- **SM-AUD-0075** `import` `probable` `semantic-memory/src/projection_legacy_compat.rs:127` — Legacy compatibility serializes with unwrap_or_default Fix: Return error on serialization failure.
- **SM-AUD-0076** `security` `probable` `semantic-memory/src/embedder.rs:127` — HTTP embedder response body uses unwrap_or_default on error Fix: Propagate body-read error or preserve status + partial diagnostics.
- **SM-AUD-0077** `security` `probable` `semantic-memory/src/embedder.rs:1` — External embedder failure modes need retry/backoff/rate-limit policy Fix: Add retry policy, per-batch timeout, and idempotent transaction boundaries.
- **SM-AUD-0312** `dynamic-sql` `static` `semantic-memory/reference/chunk.rs:485` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0348** `insert-or-ignore` `static` `semantic-memory/src/db.rs:197` — INSERT OR IGNORE can hide duplicate/constraint bugs Fix: Validate duplicates before insert or assert affected row count where required.
- **SM-AUD-0349** `insert-or-ignore` `static` `semantic-memory/src/db.rs:208` — INSERT OR IGNORE can hide duplicate/constraint bugs Fix: Validate duplicates before insert or assert affected row count where required.
- **SM-AUD-0352** `insert-or-ignore` `static` `semantic-memory/src/db.rs:370` — INSERT OR IGNORE can hide duplicate/constraint bugs Fix: Validate duplicates before insert or assert affected row count where required.
- **SM-AUD-0353** `dynamic-sql` `static` `semantic-memory/src/db.rs:558` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0354** `dynamic-sql` `static` `semantic-memory/src/db.rs:581` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0358** `dynamic-sql` `static` `semantic-memory/src/db.rs:653` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0359** `dynamic-sql` `static` `semantic-memory/src/db.rs:705` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0360** `dynamic-sql` `static` `semantic-memory/src/db.rs:715` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0363** `bytemuck-storage` `confirmed` `semantic-memory/src/db.rs:788` — Storage byte cast is alignment/endian fragile Fix: Decode storage bytes via from_le_bytes/db::bytes_to_embedding.
- **SM-AUD-0375** `dynamic-sql` `static` `semantic-memory/src/db.rs:1400` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0377** `zip-truncate` `static` `semantic-memory/src/documents.rs:70` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0383** `dynamic-sql` `static` `semantic-memory/src/documents.rs:234` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0384** `zip-truncate` `static` `semantic-memory/src/documents.rs:324` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0395** `insert-or-ignore` `static` `semantic-memory/src/episodes.rs:285` — INSERT OR IGNORE can hide duplicate/constraint bugs Fix: Validate duplicates before insert or assert affected row count where required.
- **SM-AUD-0398** `dynamic-sql` `static` `semantic-memory/src/episodes.rs:410` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0399** `dynamic-sql` `static` `semantic-memory/src/episodes.rs:414` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0400** `dynamic-sql` `static` `semantic-memory/src/episodes.rs:417` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0466** `zip-truncate` `static` `semantic-memory/src/lib.rs:1123` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0470** `zip-truncate` `static` `semantic-memory/src/lib.rs:1254` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0473** `zip-truncate` `static` `semantic-memory/src/lib.rs:1314` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0475** `zip-truncate` `static` `semantic-memory/src/lib.rs:1374` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0477** `zip-truncate` `static` `semantic-memory/src/lib.rs:1434` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0506** `insert-or-ignore` `static` `semantic-memory/src/projection_legacy_compat.rs:177` — INSERT OR IGNORE can hide duplicate/constraint bugs Fix: Validate duplicates before insert or assert affected row count where required.
- **SM-AUD-0566** `debug-assert` `confirmed` `semantic-memory/src/search.rs:61` — debug_assert is not a release invariant Fix: Use a normal check for correctness invariants.
- **SM-AUD-0567** `zip-truncate` `static` `semantic-memory/src/search.rs:62` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0573** `bytemuck-storage` `confirmed` `semantic-memory/src/search.rs:272` — Storage byte cast is alignment/endian fragile Fix: Decode storage bytes via from_le_bytes/db::bytes_to_embedding.
- **SM-AUD-0595** `zip-truncate` `static` `semantic-memory/tests/db_tests.rs:36` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.
- **SM-AUD-0596** `dynamic-sql` `static` `semantic-memory/tests/import_ugly_cases.rs:33` — Dynamic SQL construction should be whitelisted Fix: Use whitelisted enums for identifiers and bind parameters for values.
- **SM-AUD-0598** `zip-truncate` `static` `semantic-memory/tests/knowledge_tests.rs:555` — zip iteration can silently truncate mismatched collections Fix: Pre-check lengths before zip unless truncation is intentional and documented.

## Full finding table

| ID | Sev | Confidence | Area | Location | Finding | Fix |
|---|---|---|---|---|---|---|
| SM-AUD-0001 | P0 | confirmed | packaging | `semantic-memory-generic-rust-next-codex-context-20260511.report.md:1` | Archive is not hermetic despite passing certifier | Add an archive-root Cargo.toml workspace, or remove workspace-only dependency/lint reliance; validate from fresh extraction. |
| SM-AUD-0002 | P0 | confirmed | packaging | `semantic-memory/Cargo.toml:1` | No packaged root workspace manifest for included local crates | Generate a root Cargo.toml with members semantic-memory, stack-ids, semantic-memory-forge, forge-memory-bridge. |
| SM-AUD-0003 | P0 | confirmed | packaging | `semantic-memory/Cargo.lock:1` | Multiple Cargo.lock files create ambiguous dependency source of truth | Use one workspace lockfile at archive root for review builds or document crate-by-crate build commands. |
| SM-AUD-0004 | P0 | confirmed | embedding | `semantic-memory/src/documents.rs:324` | Document ingest silently truncates chunks on embedder batch-count mismatch | Centralize embed_batch validation: returned len must equal requested len before any write. |
| SM-AUD-0005 | P0 | confirmed | embedding | `semantic-memory/src/lib.rs:1254` | Fact re-embedding silently truncates on batch-count mismatch | Fail loudly on batch-count mismatch before constructing updates. |
| SM-AUD-0006 | P0 | confirmed | embedding | `semantic-memory/src/lib.rs:1314` | Chunk re-embedding silently truncates on batch-count mismatch | Fail loudly on batch-count mismatch before update transaction. |
| SM-AUD-0007 | P0 | confirmed | embedding | `semantic-memory/src/lib.rs:1374` | Message re-embedding silently truncates on batch-count mismatch | Fail loudly on batch-count mismatch before update transaction. |
| SM-AUD-0008 | P0 | confirmed | embedding | `semantic-memory/src/lib.rs:1434` | Episode re-embedding silently truncates on batch-count mismatch | Fail loudly on batch-count mismatch before update transaction. |
| SM-AUD-0009 | P0 | confirmed | embedding | `semantic-memory/src/lib.rs:536` | Public embedding validation is dimension-only | Replace with validate_embedding that checks dimensions and all components finite. |
| SM-AUD-0010 | P0 | confirmed | delete/integrity | `semantic-memory/src/documents.rs:109` | delete_document does not explicitly clean episode derived state | Collect episode_ids first and delete all episode derived surfaces plus queued HNSW deletes in one transaction. |
| SM-AUD-0011 | P0 | probable | delete/integrity | `semantic-memory/src/documents.rs:109` | delete_document can leave stale HNSW episode keys | Queue Delete for every episode:{episode_id} before deleting the document. |
| SM-AUD-0012 | P0 | confirmed | search | `semantic-memory/src/search.rs:272` | Vector scan uses bytemuck::try_cast_slice on SQLite Vec<u8> | Use db::bytes_to_embedding for all blob decoding; avoid bytemuck on storage bytes. |
| SM-AUD-0013 | P0 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:501` | HNSW sidecar loader allocates raw byte_len from file without cap | Require byte_len == dimensions*4 and <= configured max before allocation. |
| SM-AUD-0014 | P0 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:460` | HNSW data format stores dimensions using usize | Use fixed-width u32/u64 little-endian fields with versioned header. |
| SM-AUD-0015 | P0 | probable | hnsw | `semantic-memory/src/hnsw.rs:142` | HNSW save is not atomic | Write to temp files, fsync, then atomic rename graph/data/keymap as a set. |
| SM-AUD-0016 | P0 | confirmed | hnsw | `semantic-memory/src/hnsw_ops.rs:184` | Pending HNSW mutations are applied before sidecar save succeeds | Build/save a snapshot or roll back in-memory mutations on save failure. |
| SM-AUD-0017 | P0 | confirmed | hnsw | `semantic-memory/src/hnsw_ops.rs:192` | Pending upsert calls insert instead of update | Use update() or replace semantics for existing keys; dedupe pending ops by key. |
| SM-AUD-0018 | P0 | confirmed | hnsw | `semantic-memory/src/lib.rs:669` | HNSW sidecar save clones Arc while graph can still mutate | Hold exclusive lock during save or introduce immutable snapshot serialization. |
| SM-AUD-0019 | P0 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:429` | Graph sidecar validation only checks non-empty file | Validate graph/data/keymap together with checksums and stored dimensions/counts. |
| SM-AUD-0020 | P0 | confirmed | integrity | `semantic-memory/src/lib.rs:730` | HNSW integrity is count-based, not key-level | Verify each keymap key maps to a live row and each live embedded row has a matching key. |
| SM-AUD-0021 | P0 | confirmed | integrity | `semantic-memory/src/db.rs:1400` | FTS integrity count checks use dynamic table names and count parity only | Perform key-level rowid_map/content checks for every FTS-backed table. |
| SM-AUD-0022 | P0 | confirmed | search | `semantic-memory/src/search.rs:61` | cosine_similarity truncates mismatched vectors in release builds | Return Result or validate dimensions before zip in all builds. |
| SM-AUD-0023 | P0 | confirmed | search | `semantic-memory/src/search.rs:290` | cosine similarity accepts non-finite stored/query vectors | Validate finite vectors before storage and skip/error on non-finite during reads. |
| SM-AUD-0024 | P0 | confirmed | search | `semantic-memory/src/search.rs:963` | HNSW filtered search can return empty results without brute-force fallback | Overfetch adaptively after filters or fallback to brute force when post-filter hits < k. |
| SM-AUD-0025 | P0 | confirmed | search | `semantic-memory/src/conversation.rs:683` | conversation HNSW search runs blocking CPU work on async thread | Route through spawn_blocking helper. |
| SM-AUD-0026 | P0 | confirmed | delete/integrity | `semantic-memory/src/knowledge.rs:143` | delete_fact does not clean episode_causes references | Delete or mark episode_causes rows referencing the fact and update affected episode search/provenance. |
| SM-AUD-0027 | P0 | probable | delete/integrity | `semantic-memory/src/knowledge.rs:181` | update_fact does not update dependent episode/projection search text | Record invalidation edges or recompute affected derived search surfaces. |
| SM-AUD-0028 | P1 | confirmed | api | `semantic-memory/src/knowledge.rs:236` | delete_namespace returns only fact count despite deleting many entity types | Return NamespaceDeleteReport with counts per entity/table/op. |
| SM-AUD-0029 | P1 | confirmed | hnsw | `semantic-memory/src/lib.rs:369` | Open-time HNSW rebuild/degrade policy is implicit | Expose HnswStartupPolicy and health status. |
| SM-AUD-0030 | P1 | confirmed | hnsw | `semantic-memory/src/lib.rs:403` | SQL errors while counting embeddings are swallowed as zero | Propagate DB errors during integrity decisions. |
| SM-AUD-0031 | P1 | confirmed | hnsw | `semantic-memory/src/lib.rs:455` | Orphan-count SQL errors are swallowed as zero | Propagate the error or force degraded/rebuild state. |
| SM-AUD-0032 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:310` | Missing hnsw_keymap table silently leaves loaded graph without keys | Treat graph+missing keymap as degraded/rebuild, not clean load. |
| SM-AUD-0033 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:353` | Malformed next_id metadata falls back silently | Report error or mark sidecar stale when metadata is malformed. |
| SM-AUD-0034 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:222` | HNSW len can report nonzero even when keymap is empty | Expose separate graph_len and live_key_count; search should use resolvable key count. |
| SM-AUD-0035 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:185` | Tombstone overfetch is too naive | Iteratively overfetch until enough live hits or graph exhausted. |
| SM-AUD-0036 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:250` | deleted_ratio may divide using graph count that includes unreachable/unmapped points | Compute deleted/live ratios from verified keymap state. |
| SM-AUD-0037 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:496` | u64 node id is cast to usize without range check | TryFrom<u64> with explicit error. |
| SM-AUD-0038 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw.rs:383` | insert ignores return/status from hnsw_rs graph.insert | Wrap insert in catch_unwind if needed and use API result if available; update keymap only after success. |
| SM-AUD-0039 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw_ops.rs:71` | HNSW rebuild silently skips invalid fact embeddings | Count skipped rows and return degraded integrity finding. |
| SM-AUD-0040 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw_ops.rs:89` | HNSW rebuild silently skips invalid chunk embeddings | Count skipped rows and expose rebuild diagnostics. |
| SM-AUD-0041 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw_ops.rs:107` | HNSW rebuild silently skips invalid message embeddings | Count skipped rows and expose rebuild diagnostics. |
| SM-AUD-0042 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw_ops.rs:125` | HNSW rebuild silently skips invalid episode embeddings | Count skipped rows and expose rebuild diagnostics. |
| SM-AUD-0043 | P1 | confirmed | hnsw | `semantic-memory/src/hnsw_ops.rs:205` | clear_pending_index_ops is all-or-nothing per processed key list after sidecar save | Use transactional state machine with op generation numbers and idempotent upsert. |
| SM-AUD-0044 | P1 | confirmed | q8 | `semantic-memory/src/lib.rs:1256` | q8 optionality conflicts with integrity expectations | Define compressed vectors as mandatory or optional; align write, repair, and integrity. |
| SM-AUD-0045 | P1 | confirmed | q8 | `semantic-memory/src/quantize.rs:1` | q8 baseline lacks explicit versioned storage envelope | Add a vector-codec envelope with codec, version, dim, checksum, and params. |
| SM-AUD-0046 | P1 | confirmed | search | `semantic-memory/src/search.rs:102` | Invalid timestamp becomes maximally fresh | Treat invalid timestamps as no recency contribution or stale. |
| SM-AUD-0047 | P1 | confirmed | search | `semantic-memory/src/search.rs:85` | recency scoring uses wall-clock inside ranking | Inject clock into SearchConfig or query context. |
| SM-AUD-0048 | P1 | confirmed | search | `semantic-memory/src/lib.rs:851` | candidate_pool_size.max(k * 3) can overflow | Use k.saturating_mul(3) and cap top_k. |
| SM-AUD-0049 | P1 | confirmed | search | `semantic-memory/src/lib.rs:954` | Second candidate_pool_size.max(k * 3) overflow surface | Use saturating_mul and configured max_top_k. |
| SM-AUD-0050 | P1 | confirmed | search | `semantic-memory/src/lib.rs:1033` | Third candidate_pool_size.max(k * 3) overflow surface | Use saturating_mul and configured max_top_k. |
| SM-AUD-0051 | P1 | confirmed | search | `semantic-memory/src/conversation.rs:683` | conversation candidate_pool_size.max(k * 3) overflow surface | Use saturating_mul and configured max_top_k. |
| SM-AUD-0052 | P1 | confirmed | conversation | `semantic-memory/src/conversation.rs:172` | Unknown token counts are treated as zero in budget selection | Recompute missing token_count or treat unknown as conservative upper bound. |
| SM-AUD-0053 | P1 | confirmed | conversation | `semantic-memory/src/conversation.rs:173` | Token budget addition can overflow u32 | Use checked_add/saturating_add and error or clamp. |
| SM-AUD-0054 | P1 | confirmed | conversation | `semantic-memory/src/conversation.rs:185` | session_token_count casts negative SQL sum to u64 | Validate nonnegative aggregate before conversion. |
| SM-AUD-0055 | P1 | probable | validation | `semantic-memory/src/conversation.rs:63` | Session/channel identifiers are not consistently length/whitespace validated | Centralize validation for session_id, channel, namespace, title, source URI. |
| SM-AUD-0056 | P1 | probable | validation | `semantic-memory/src/documents.rs:315` | Document title/source/metadata size validation is weaker than content validation | Add max lengths and metadata byte caps. |
| SM-AUD-0057 | P1 | probable | validation | `semantic-memory/src/episodes.rs:393` | Episode search limit is unbounded | Cap limit using config max_top_k/max_query_rows. |
| SM-AUD-0058 | P1 | confirmed | api | `semantic-memory/src/episodes.rs:437` | search_episodes drops episode_id and returns document_id | Return episode_id plus document_id or a typed EpisodeSearchResult. |
| SM-AUD-0059 | P1 | confirmed | diagnostics | `semantic-memory/src/episodes.rs:452` | Episode parse errors report document_id instead of episode_id | Use episode_id in parse helpers. |
| SM-AUD-0060 | P1 | confirmed | episodes | `semantic-memory/src/episodes.rs:285` | INSERT OR IGNORE collapses duplicate cause IDs silently | Validate and reject duplicate cause_ids or preserve multiplicity intentionally. |
| SM-AUD-0061 | P1 | confirmed | episodes | `semantic-memory/src/episodes.rs:360` | update_episode_outcome cannot clear experiment_id | Add explicit clear operation or Option<Option<String>> semantics. |
| SM-AUD-0062 | P1 | probable | keys | `semantic-memory/src/hnsw.rs:58` | HNSW key parsing split_once(:) is fragile for IDs containing colon | Use structured key encoding or reject colon in IDs. |
| SM-AUD-0063 | P1 | probable | keys | `semantic-memory/src/search.rs:119` | Message dedup key uses session_id:message_id delimiter | Use tuple type internally or escaped/keyed serialization. |
| SM-AUD-0064 | P1 | probable | foreign-keys | `semantic-memory/src/db.rs:1` | Correctness depends on PRAGMA foreign_keys being enabled for every pooled connection | Assert PRAGMA foreign_keys=ON after every connection checkout. |
| SM-AUD-0065 | P1 | probable | db | `semantic-memory/src/db.rs:558` | PRAGMA max_page_count computed with dynamic formatting | Validate max_page_count range before execute. |
| SM-AUD-0066 | P1 | confirmed | db | `semantic-memory/src/db.rs:705` | Dynamic table_info table name formatting relies on internal callers only | Make table an enum or whitelist. |
| SM-AUD-0067 | P1 | confirmed | db | `semantic-memory/src/db.rs:715` | Dynamic ALTER TABLE formatting relies on internal table/column whitelists | Make migration table/column identifiers enum-backed. |
| SM-AUD-0068 | P1 | confirmed | db | `semantic-memory/src/db.rs:1400` | Dynamic SELECT COUNT table name relies on internal map table list | Use enum/constant-only function signature. |
| SM-AUD-0069 | P1 | probable | sqlite | `semantic-memory/src/db.rs:1` | SQLite WAL/checkpoint/backpressure policy not visible in archive-level docs | Document and test WAL mode, busy timeout, checkpoint, and backup semantics. |
| SM-AUD-0070 | P1 | probable | pool | `semantic-memory/src/pool.rs:1` | Connection pool shutdown/poison behavior needs stress coverage | Add pool close/drop/concurrent open tests under load. |
| SM-AUD-0071 | P1 | probable | projection | `semantic-memory/src/projection_storage.rs:1` | Projection storage integrity is likely separate from memory integrity | Add projection-level integrity: rows, derivations, episodes, imports, claim versions. |
| SM-AUD-0072 | P1 | probable | projection | `semantic-memory/src/projection_storage_query.rs:271` | Projection query uses unwrap_or_default for missing claim/source IDs | Return structured parse/error instead of default empty identifiers. |
| SM-AUD-0073 | P1 | confirmed | bridge | `forge-memory-bridge/src/transform.rs:301` | Bridge transform uses unwrap_or_default, potentially hiding malformed optional payloads | Emit explicit transform error or warning with field name. |
| SM-AUD-0074 | P1 | probable | import | `semantic-memory/src/json_compat_import.rs:25` | JSON import begins with from_str(...).ok() | Preserve parse error and source payload hash in import receipt. |
| SM-AUD-0075 | P1 | probable | import | `semantic-memory/src/projection_legacy_compat.rs:127` | Legacy compatibility serializes with unwrap_or_default | Return error on serialization failure. |
| SM-AUD-0076 | P1 | probable | security | `semantic-memory/src/embedder.rs:127` | HTTP embedder response body uses unwrap_or_default on error | Propagate body-read error or preserve status + partial diagnostics. |
| SM-AUD-0077 | P1 | probable | security | `semantic-memory/src/embedder.rs:1` | External embedder failure modes need retry/backoff/rate-limit policy | Add retry policy, per-batch timeout, and idempotent transaction boundaries. |
| SM-AUD-0078 | P2 | confirmed | docs | `semantic-memory/CLAUDE_CODE_PROMPT.md:1` | Prompt docs are packaged at project root | Move prompt files under docs/internal/codex/ or exclude public packages. |
| SM-AUD-0079 | P2 | confirmed | docs | `semantic-memory/IMPLEMENTATION_PROMPT.md:1` | Implementation prompt is packaged at project root | Archive or relocate prompt files. |
| SM-AUD-0080 | P2 | confirmed | docs | `semantic-memory/PATCH_PROMPT.md:1` | Patch prompt is packaged at project root | Archive or relocate prompt files. |
| SM-AUD-0081 | P2 | confirmed | docs | `semantic-memory/V2_PATCH_PROMPT.md:1` | V2 patch prompt is packaged at project root | Archive or relocate prompt files. |
| SM-AUD-0082 | P2 | confirmed | docs | `semantic-memory/semantic-memory-generic-rust-next-codex-context-20260507.codex-archive.json:1` | Prior codex sidecar is included in current source package | Exclude generated/codex sidecars consistently. |
| SM-AUD-0083 | P2 | confirmed | permissions | `manifest:1` | Most source/docs are marked executable | Normalize permissions: 0644 for non-scripts, 0755 only for actual executables. |
| SM-AUD-0084 | P2 | confirmed | ci | `manifest:1` | No packaged CI workflow surface | Add CI for all features, no-default-features variants, and clean extraction build. |
| SM-AUD-0085 | P2 | confirmed | benchmarks | `manifest:1` | No benchmark suite/result artifact included | Add benches and BENCHMARKS.md with f32/q8/HNSW/TurboQuant matrix. |
| SM-AUD-0086 | P2 | probable | structure | `semantic-memory/src/lib.rs:1` | lib.rs is very large and high blast-radius | Split store open/search/reembed/delete/integrity modules after P0 fixes. |
| SM-AUD-0087 | P2 | probable | structure | `semantic-memory/src/db.rs:1` | db.rs is very large and high blast-radius | Split schema/migrations/integrity/hnsw metadata helpers. |
| SM-AUD-0088 | P2 | probable | structure | `semantic-memory/src/search.rs:1` | search.rs combines lexical, vector, HNSW resolution, fusion, scoring | Split search/fts.rs vector.rs hnsw.rs fusion.rs explain.rs. |
| SM-AUD-0089 | P3 | confirmed | permissions | `forge-memory-bridge/AGENTS.md` | Non-script file is executable: forge-memory-bridge/AGENTS.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0090 | P3 | confirmed | permissions | `forge-memory-bridge/Cargo.lock` | Non-script file is executable: forge-memory-bridge/Cargo.lock | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0091 | P3 | confirmed | permissions | `forge-memory-bridge/Cargo.toml` | Non-script file is executable: forge-memory-bridge/Cargo.toml | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0092 | P3 | confirmed | permissions | `forge-memory-bridge/src/batch.rs` | Non-script file is executable: forge-memory-bridge/src/batch.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0093 | P3 | confirmed | permissions | `forge-memory-bridge/src/error.rs` | Non-script file is executable: forge-memory-bridge/src/error.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0094 | P3 | confirmed | permissions | `forge-memory-bridge/src/legacy.rs` | Non-script file is executable: forge-memory-bridge/src/legacy.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0095 | P3 | confirmed | permissions | `forge-memory-bridge/src/lib.rs` | Non-script file is executable: forge-memory-bridge/src/lib.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0096 | P3 | confirmed | permissions | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs` | Non-script file is executable: forge-memory-bridge/tests/forge_bridge_memory_proof.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0097 | P3 | confirmed | permissions | `semantic-memory-forge/Cargo.toml` | Non-script file is executable: semantic-memory-forge/Cargo.toml | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0098 | P3 | confirmed | permissions | `semantic-memory-forge/src/bundle.rs` | Non-script file is executable: semantic-memory-forge/src/bundle.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0099 | P3 | confirmed | permissions | `semantic-memory-forge/src/estimator.rs` | Non-script file is executable: semantic-memory-forge/src/estimator.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0100 | P3 | confirmed | permissions | `semantic-memory-forge/src/lib.rs` | Non-script file is executable: semantic-memory-forge/src/lib.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0101 | P3 | confirmed | permissions | `semantic-memory-forge/src/tool_receipt.rs` | Non-script file is executable: semantic-memory-forge/src/tool_receipt.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0102 | P3 | confirmed | permissions | `semantic-memory-forge/src/v11.rs` | Non-script file is executable: semantic-memory-forge/src/v11.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0103 | P3 | confirmed | permissions | `semantic-memory-forge/src/v13.rs` | Non-script file is executable: semantic-memory-forge/src/v13.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0104 | P3 | confirmed | permissions | `semantic-memory-forge/src/v14.rs` | Non-script file is executable: semantic-memory-forge/src/v14.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0105 | P3 | confirmed | permissions | `semantic-memory-forge/src/v9.rs` | Non-script file is executable: semantic-memory-forge/src/v9.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0106 | P3 | confirmed | permissions | `semantic-memory/.gitignore` | Non-script file is executable: semantic-memory/.gitignore | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0107 | P3 | confirmed | permissions | `semantic-memory/AGENTS.md` | Non-script file is executable: semantic-memory/AGENTS.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0108 | P3 | confirmed | permissions | `semantic-memory/CLAUDE_CODE_PROMPT.md` | Non-script file is executable: semantic-memory/CLAUDE_CODE_PROMPT.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0109 | P3 | confirmed | permissions | `semantic-memory/Cargo.lock` | Non-script file is executable: semantic-memory/Cargo.lock | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0110 | P3 | confirmed | permissions | `semantic-memory/Cargo.toml` | Non-script file is executable: semantic-memory/Cargo.toml | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0111 | P3 | confirmed | permissions | `semantic-memory/HNSWLIB_RS_REFERENCE.md` | Non-script file is executable: semantic-memory/HNSWLIB_RS_REFERENCE.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0112 | P3 | confirmed | permissions | `semantic-memory/IMPLEMENTATION_PROMPT.md` | Non-script file is executable: semantic-memory/IMPLEMENTATION_PROMPT.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0113 | P3 | confirmed | permissions | `semantic-memory/PATCH_PROMPT.md` | Non-script file is executable: semantic-memory/PATCH_PROMPT.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0114 | P3 | confirmed | permissions | `semantic-memory/SPEC.md` | Non-script file is executable: semantic-memory/SPEC.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0115 | P3 | confirmed | permissions | `semantic-memory/TESTING.md` | Non-script file is executable: semantic-memory/TESTING.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0116 | P3 | confirmed | permissions | `semantic-memory/UPGRADE_SPEC.md` | Non-script file is executable: semantic-memory/UPGRADE_SPEC.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0117 | P3 | confirmed | permissions | `semantic-memory/V1_1_AGENTS_ADDENDUM.md` | Non-script file is executable: semantic-memory/V1_1_AGENTS_ADDENDUM.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0118 | P3 | confirmed | permissions | `semantic-memory/V1_1_SPEC_ADDENDUM.md` | Non-script file is executable: semantic-memory/V1_1_SPEC_ADDENDUM.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0119 | P3 | confirmed | permissions | `semantic-memory/V1_1_TESTING_ADDENDUM.md` | Non-script file is executable: semantic-memory/V1_1_TESTING_ADDENDUM.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0120 | P3 | confirmed | permissions | `semantic-memory/V2_AGENTS_ADDENDUM.md` | Non-script file is executable: semantic-memory/V2_AGENTS_ADDENDUM.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0121 | P3 | confirmed | permissions | `semantic-memory/V2_PATCH_PROMPT.md` | Non-script file is executable: semantic-memory/V2_PATCH_PROMPT.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0122 | P3 | confirmed | permissions | `semantic-memory/V2_SPEC_ADDENDUM.md` | Non-script file is executable: semantic-memory/V2_SPEC_ADDENDUM.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0123 | P3 | confirmed | permissions | `semantic-memory/V2_TESTING_ADDENDUM.md` | Non-script file is executable: semantic-memory/V2_TESTING_ADDENDUM.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0124 | P3 | confirmed | permissions | `semantic-memory/V3_AGENTS_ADDENDUM.md` | Non-script file is executable: semantic-memory/V3_AGENTS_ADDENDUM.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0125 | P3 | confirmed | permissions | `semantic-memory/V3_CHANGE_MAP.md` | Non-script file is executable: semantic-memory/V3_CHANGE_MAP.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0126 | P3 | confirmed | permissions | `semantic-memory/V3_SPEC.md` | Non-script file is executable: semantic-memory/V3_SPEC.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0127 | P3 | confirmed | permissions | `semantic-memory/V3_TESTING.md` | Non-script file is executable: semantic-memory/V3_TESTING.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0128 | P3 | confirmed | permissions | `semantic-memory/examples/basic_search.rs` | Non-script file is executable: semantic-memory/examples/basic_search.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0129 | P3 | confirmed | permissions | `semantic-memory/examples/conversation_memory.rs` | Non-script file is executable: semantic-memory/examples/conversation_memory.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0130 | P3 | confirmed | permissions | `semantic-memory/reference/chunk.rs` | Non-script file is executable: semantic-memory/reference/chunk.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0131 | P3 | confirmed | permissions | `semantic-memory/reference/hybrid_search.rs` | Non-script file is executable: semantic-memory/reference/hybrid_search.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0132 | P3 | confirmed | permissions | `semantic-memory/semantic-memory-spec.md` | Non-script file is executable: semantic-memory/semantic-memory-spec.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0133 | P3 | confirmed | permissions | `semantic-memory/src/chunker.rs` | Non-script file is executable: semantic-memory/src/chunker.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0134 | P3 | confirmed | permissions | `semantic-memory/src/config.rs` | Non-script file is executable: semantic-memory/src/config.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0135 | P3 | confirmed | permissions | `semantic-memory/src/conversation.rs` | Non-script file is executable: semantic-memory/src/conversation.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0136 | P3 | confirmed | permissions | `semantic-memory/src/db.rs` | Non-script file is executable: semantic-memory/src/db.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0137 | P3 | confirmed | permissions | `semantic-memory/src/documents.rs` | Non-script file is executable: semantic-memory/src/documents.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0138 | P3 | confirmed | permissions | `semantic-memory/src/embedder.rs` | Non-script file is executable: semantic-memory/src/embedder.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0139 | P3 | confirmed | permissions | `semantic-memory/src/episodes.rs` | Non-script file is executable: semantic-memory/src/episodes.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0140 | P3 | confirmed | permissions | `semantic-memory/src/error.rs` | Non-script file is executable: semantic-memory/src/error.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0141 | P3 | confirmed | permissions | `semantic-memory/src/graph.rs` | Non-script file is executable: semantic-memory/src/graph.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0142 | P3 | confirmed | permissions | `semantic-memory/src/hnsw.rs` | Non-script file is executable: semantic-memory/src/hnsw.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0143 | P3 | confirmed | permissions | `semantic-memory/src/hnsw_ops.rs` | Non-script file is executable: semantic-memory/src/hnsw_ops.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0144 | P3 | confirmed | permissions | `semantic-memory/src/json_compat_import.rs` | Non-script file is executable: semantic-memory/src/json_compat_import.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0145 | P3 | confirmed | permissions | `semantic-memory/src/knowledge.rs` | Non-script file is executable: semantic-memory/src/knowledge.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0146 | P3 | confirmed | permissions | `semantic-memory/src/lib.rs` | Non-script file is executable: semantic-memory/src/lib.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0147 | P3 | confirmed | permissions | `semantic-memory/src/pool.rs` | Non-script file is executable: semantic-memory/src/pool.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0148 | P3 | confirmed | permissions | `semantic-memory/src/projection_batch.rs` | Non-script file is executable: semantic-memory/src/projection_batch.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0149 | P3 | confirmed | permissions | `semantic-memory/src/projection_derivation.rs` | Non-script file is executable: semantic-memory/src/projection_derivation.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0150 | P3 | confirmed | permissions | `semantic-memory/src/projection_import.rs` | Non-script file is executable: semantic-memory/src/projection_import.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0151 | P3 | confirmed | permissions | `semantic-memory/src/projection_lane.rs` | Non-script file is executable: semantic-memory/src/projection_lane.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0152 | P3 | confirmed | permissions | `semantic-memory/src/projection_legacy_compat.rs` | Non-script file is executable: semantic-memory/src/projection_legacy_compat.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0153 | P3 | confirmed | permissions | `semantic-memory/src/projection_storage.rs` | Non-script file is executable: semantic-memory/src/projection_storage.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0154 | P3 | confirmed | permissions | `semantic-memory/src/quantize.rs` | Non-script file is executable: semantic-memory/src/quantize.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0155 | P3 | confirmed | permissions | `semantic-memory/src/search.rs` | Non-script file is executable: semantic-memory/src/search.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0156 | P3 | confirmed | permissions | `semantic-memory/src/storage.rs` | Non-script file is executable: semantic-memory/src/storage.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0157 | P3 | confirmed | permissions | `semantic-memory/src/store_support.rs` | Non-script file is executable: semantic-memory/src/store_support.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0158 | P3 | confirmed | permissions | `semantic-memory/src/tokenizer.rs` | Non-script file is executable: semantic-memory/src/tokenizer.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0159 | P3 | confirmed | permissions | `semantic-memory/src/types.rs` | Non-script file is executable: semantic-memory/src/types.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0160 | P3 | confirmed | permissions | `semantic-memory/tests/brute_force_parity.rs` | Non-script file is executable: semantic-memory/tests/brute_force_parity.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0161 | P3 | confirmed | permissions | `semantic-memory/tests/chunker_tests.rs` | Non-script file is executable: semantic-memory/tests/chunker_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0162 | P3 | confirmed | permissions | `semantic-memory/tests/compaction.rs` | Non-script file is executable: semantic-memory/tests/compaction.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0163 | P3 | confirmed | permissions | `semantic-memory/tests/concurrent_access.rs` | Non-script file is executable: semantic-memory/tests/concurrent_access.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0164 | P3 | confirmed | permissions | `semantic-memory/tests/conversation_search_tests.rs` | Non-script file is executable: semantic-memory/tests/conversation_search_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0165 | P3 | confirmed | permissions | `semantic-memory/tests/conversation_tests.rs` | Non-script file is executable: semantic-memory/tests/conversation_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0166 | P3 | confirmed | permissions | `semantic-memory/tests/db_tests.rs` | Non-script file is executable: semantic-memory/tests/db_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0167 | P3 | confirmed | permissions | `semantic-memory/tests/episode_identity.rs` | Non-script file is executable: semantic-memory/tests/episode_identity.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0168 | P3 | confirmed | permissions | `semantic-memory/tests/hardening_semantics.rs` | Non-script file is executable: semantic-memory/tests/hardening_semantics.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0169 | P3 | confirmed | permissions | `semantic-memory/tests/hardening_v5.rs` | Non-script file is executable: semantic-memory/tests/hardening_v5.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0170 | P3 | confirmed | permissions | `semantic-memory/tests/hnsw_hotswap.rs` | Non-script file is executable: semantic-memory/tests/hnsw_hotswap.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0171 | P3 | confirmed | permissions | `semantic-memory/tests/hnsw_integration.rs` | Non-script file is executable: semantic-memory/tests/hnsw_integration.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0172 | P3 | confirmed | permissions | `semantic-memory/tests/hnsw_persistence.rs` | Non-script file is executable: semantic-memory/tests/hnsw_persistence.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0173 | P3 | confirmed | permissions | `semantic-memory/tests/import_boundary_tests.rs` | Non-script file is executable: semantic-memory/tests/import_boundary_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0174 | P3 | confirmed | permissions | `semantic-memory/tests/import_ugly_cases.rs` | Non-script file is executable: semantic-memory/tests/import_ugly_cases.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0175 | P3 | confirmed | permissions | `semantic-memory/tests/integration_tests.rs` | Non-script file is executable: semantic-memory/tests/integration_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0176 | P3 | confirmed | permissions | `semantic-memory/tests/knowledge_tests.rs` | Non-script file is executable: semantic-memory/tests/knowledge_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0177 | P3 | confirmed | permissions | `semantic-memory/tests/migration_v5.rs` | Non-script file is executable: semantic-memory/tests/migration_v5.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0178 | P3 | confirmed | permissions | `semantic-memory/tests/projection_v11_tests.rs` | Non-script file is executable: semantic-memory/tests/projection_v11_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0179 | P3 | confirmed | permissions | `semantic-memory/tests/quantization.rs` | Non-script file is executable: semantic-memory/tests/quantization.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0180 | P3 | confirmed | permissions | `semantic-memory/tests/quantization_pipeline.rs` | Non-script file is executable: semantic-memory/tests/quantization_pipeline.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0181 | P3 | confirmed | permissions | `semantic-memory/tests/search_tests.rs` | Non-script file is executable: semantic-memory/tests/search_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0182 | P3 | confirmed | permissions | `semantic-memory/tests/step3_verification.rs` | Non-script file is executable: semantic-memory/tests/step3_verification.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0183 | P3 | confirmed | permissions | `semantic-memory/tests/step4_verification.rs` | Non-script file is executable: semantic-memory/tests/step4_verification.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0184 | P3 | confirmed | permissions | `semantic-memory/tests/storage_lifecycle.rs` | Non-script file is executable: semantic-memory/tests/storage_lifecycle.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0185 | P3 | confirmed | permissions | `semantic-memory/tests/tokenizer_tests.rs` | Non-script file is executable: semantic-memory/tests/tokenizer_tests.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0186 | P3 | confirmed | permissions | `semantic-memory/tests/trace_id_write_seam.rs` | Non-script file is executable: semantic-memory/tests/trace_id_write_seam.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0187 | P3 | confirmed | permissions | `semantic-memory/tests/vector_only_hnsw.rs` | Non-script file is executable: semantic-memory/tests/vector_only_hnsw.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0188 | P3 | confirmed | permissions | `stack-ids/AGENTS.md` | Non-script file is executable: stack-ids/AGENTS.md | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0189 | P3 | confirmed | permissions | `stack-ids/Cargo.lock` | Non-script file is executable: stack-ids/Cargo.lock | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0190 | P3 | confirmed | permissions | `stack-ids/Cargo.toml` | Non-script file is executable: stack-ids/Cargo.toml | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0191 | P3 | confirmed | permissions | `stack-ids/src/digest.rs` | Non-script file is executable: stack-ids/src/digest.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0192 | P3 | confirmed | permissions | `stack-ids/src/lib.rs` | Non-script file is executable: stack-ids/src/lib.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0193 | P3 | confirmed | permissions | `stack-ids/src/scope.rs` | Non-script file is executable: stack-ids/src/scope.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0194 | P3 | confirmed | permissions | `stack-ids/src/trace.rs` | Non-script file is executable: stack-ids/src/trace.rs | Normalize to 0644 unless this file is intentionally runnable. |
| SM-AUD-0195 | P2 | confirmed | docs | `semantic-memory/CLAUDE_CODE_PROMPT.md` | Root markdown prompt candidate should be archived: CLAUDE_CODE_PROMPT.md | Move under docs/internal/codex or exclude from public handoff packages. |
| SM-AUD-0196 | P2 | confirmed | docs | `semantic-memory/IMPLEMENTATION_PROMPT.md` | Root markdown prompt candidate should be archived: IMPLEMENTATION_PROMPT.md | Move under docs/internal/codex or exclude from public handoff packages. |
| SM-AUD-0197 | P2 | confirmed | docs | `semantic-memory/PATCH_PROMPT.md` | Root markdown prompt candidate should be archived: PATCH_PROMPT.md | Move under docs/internal/codex or exclude from public handoff packages. |
| SM-AUD-0198 | P2 | confirmed | docs | `semantic-memory/V2_PATCH_PROMPT.md` | Root markdown prompt candidate should be archived: V2_PATCH_PROMPT.md | Move under docs/internal/codex or exclude from public handoff packages. |
| SM-AUD-0199 | P3 | confirmed | docs | `semantic-memory/HNSWLIB_RS_REFERENCE.md` | Ambiguous root markdown file needs source-of-truth decision: HNSWLIB_RS_REFERENCE.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0200 | P3 | confirmed | docs | `semantic-memory/SPEC.md` | Ambiguous root markdown file needs source-of-truth decision: SPEC.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0201 | P3 | confirmed | docs | `semantic-memory/TESTING.md` | Ambiguous root markdown file needs source-of-truth decision: TESTING.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0202 | P3 | confirmed | docs | `semantic-memory/UPGRADE_SPEC.md` | Ambiguous root markdown file needs source-of-truth decision: UPGRADE_SPEC.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0203 | P3 | confirmed | docs | `semantic-memory/V1_1_AGENTS_ADDENDUM.md` | Ambiguous root markdown file needs source-of-truth decision: V1_1_AGENTS_ADDENDUM.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0204 | P3 | confirmed | docs | `semantic-memory/V1_1_SPEC_ADDENDUM.md` | Ambiguous root markdown file needs source-of-truth decision: V1_1_SPEC_ADDENDUM.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0205 | P3 | confirmed | docs | `semantic-memory/V1_1_TESTING_ADDENDUM.md` | Ambiguous root markdown file needs source-of-truth decision: V1_1_TESTING_ADDENDUM.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0206 | P3 | confirmed | docs | `semantic-memory/V2_AGENTS_ADDENDUM.md` | Ambiguous root markdown file needs source-of-truth decision: V2_AGENTS_ADDENDUM.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0207 | P3 | confirmed | docs | `semantic-memory/V2_SPEC_ADDENDUM.md` | Ambiguous root markdown file needs source-of-truth decision: V2_SPEC_ADDENDUM.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0208 | P3 | confirmed | docs | `semantic-memory/V2_TESTING_ADDENDUM.md` | Ambiguous root markdown file needs source-of-truth decision: V2_TESTING_ADDENDUM.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0209 | P3 | confirmed | docs | `semantic-memory/V3_AGENTS_ADDENDUM.md` | Ambiguous root markdown file needs source-of-truth decision: V3_AGENTS_ADDENDUM.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0210 | P3 | confirmed | docs | `semantic-memory/V3_CHANGE_MAP.md` | Ambiguous root markdown file needs source-of-truth decision: V3_CHANGE_MAP.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0211 | P3 | confirmed | docs | `semantic-memory/V3_SPEC.md` | Ambiguous root markdown file needs source-of-truth decision: V3_SPEC.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0212 | P3 | confirmed | docs | `semantic-memory/V3_TESTING.md` | Ambiguous root markdown file needs source-of-truth decision: V3_TESTING.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0213 | P3 | confirmed | docs | `semantic-memory/semantic-memory-generic-rust-next-codex-context-20260507.report.md` | Ambiguous root markdown file needs source-of-truth decision: semantic-memory-generic-rust-next-codex-context-20260507.report.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0214 | P3 | confirmed | docs | `semantic-memory/semantic-memory-spec.md` | Ambiguous root markdown file needs source-of-truth decision: semantic-memory-spec.md | Mark as protected, archive it, or move to docs/design-history. |
| SM-AUD-0215 | P2 | static | runtime-clock | `forge-memory-bridge/src/error.rs:95` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0216 | P2 | static | error-default | `forge-memory-bridge/src/legacy.rs:121` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0217 | P2 | static | runtime-clock | `forge-memory-bridge/src/legacy.rs:158` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0218 | P3 | static | unwrap-expect | `forge-memory-bridge/src/legacy.rs:215` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0219 | P3 | static | unwrap-expect | `forge-memory-bridge/src/legacy.rs:224` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0220 | P3 | static | unwrap-expect | `forge-memory-bridge/src/legacy.rs:230` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0221 | P3 | static | unwrap-expect | `forge-memory-bridge/src/legacy.rs:236` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0222 | P3 | static | unwrap-expect | `forge-memory-bridge/src/legacy.rs:246` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0223 | P3 | static | unwrap-expect | `forge-memory-bridge/src/legacy.rs:262` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0224 | P3 | static | unwrap-expect | `forge-memory-bridge/src/legacy.rs:287` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0225 | P2 | static | runtime-clock | `forge-memory-bridge/src/transform.rs:51` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0226 | P2 | static | runtime-clock | `forge-memory-bridge/src/transform.rs:95` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0227 | P2 | static | runtime-clock | `forge-memory-bridge/src/transform.rs:133` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0228 | P2 | static | error-default | `forge-memory-bridge/src/transform.rs:301` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0229 | P2 | static | error-default | `forge-memory-bridge/src/transform.rs:336` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0230 | P2 | static | error-default | `forge-memory-bridge/src/transform.rs:505` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0231 | P2 | static | error-default | `forge-memory-bridge/src/transform.rs:658` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0232 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:22` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0233 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:53` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0234 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:70` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0235 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:93` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0236 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:96` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0237 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:97` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0238 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:104` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0239 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:171` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0240 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:184` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0241 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:221` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0242 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:234` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0243 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:266` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0244 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:290` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0245 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:303` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0246 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:335` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0247 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:348` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0248 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:364` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0249 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:379` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0250 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:391` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0251 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:418` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0252 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:431` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0253 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:471` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0254 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:530` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0255 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:557` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0256 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:583` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0257 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:596` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0258 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:666` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0259 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:678` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0260 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:736` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0261 | P3 | static | unwrap-expect | `forge-memory-bridge/src/transform_tests.rs:750` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0262 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:115` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0263 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:184` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0264 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:248` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0265 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:331` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0266 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:370` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0267 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:418` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0268 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:463` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0269 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:476` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0270 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:518` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0271 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:532` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0272 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:557` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0273 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:563` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0274 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:592` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0275 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:601` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0276 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:615` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0277 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:617` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0278 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:618` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0279 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:619` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0280 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:650` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0281 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:661` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0282 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:670` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0283 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:707` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0284 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:724` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0285 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:741` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0286 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:772` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0287 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:773` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0288 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:781` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0289 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:803` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0290 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:805` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0291 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:817` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0292 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:850` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0293 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:871` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0294 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:893` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0295 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:895` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0296 | P3 | static | unwrap-expect | `forge-memory-bridge/tests/forge_bridge_memory_proof.rs:898` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0297 | P2 | static | error-default | `semantic-memory/examples/basic_search.rs:125` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0298 | P2 | static | error-default | `semantic-memory/examples/conversation_memory.rs:72` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0299 | P2 | static | error-default | `semantic-memory/examples/conversation_memory.rs:106` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0300 | P2 | static | error-default | `semantic-memory/reference/chunk.rs:54` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0301 | P2 | static | error-default | `semantic-memory/reference/chunk.rs:76` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0302 | P2 | static | numeric-cast | `semantic-memory/reference/chunk.rs:85` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0303 | P2 | static | numeric-cast | `semantic-memory/reference/chunk.rs:87` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0304 | P2 | static | numeric-cast | `semantic-memory/reference/chunk.rs:88` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0305 | P2 | static | numeric-cast | `semantic-memory/reference/chunk.rs:89` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0306 | P2 | static | numeric-cast | `semantic-memory/reference/chunk.rs:105` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0307 | P2 | static | numeric-cast | `semantic-memory/reference/chunk.rs:107` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0308 | P2 | static | error-default | `semantic-memory/reference/chunk.rs:293` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0309 | P2 | static | error-default | `semantic-memory/reference/chunk.rs:305` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0310 | P2 | static | error-default | `semantic-memory/reference/chunk.rs:316` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0311 | P2 | static | error-default | `semantic-memory/reference/chunk.rs:402` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0312 | P1 | static | dynamic-sql | `semantic-memory/reference/chunk.rs:485` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0313 | P2 | static | error-default | `semantic-memory/reference/chunk.rs:496` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0314 | P2 | static | hashmap-order | `semantic-memory/reference/hybrid_search.rs:4` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0315 | P2 | static | numeric-cast | `semantic-memory/reference/hybrid_search.rs:42` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0316 | P2 | static | hashmap-order | `semantic-memory/reference/hybrid_search.rs:81` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0317 | P2 | static | numeric-cast | `semantic-memory/reference/hybrid_search.rs:84` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0318 | P2 | static | numeric-cast | `semantic-memory/reference/hybrid_search.rs:92` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0319 | P2 | static | error-default | `semantic-memory/reference/hybrid_search.rs:104` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0320 | P2 | static | numeric-cast | `semantic-memory/reference/hybrid_search.rs:125` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0321 | P2 | static | error-default | `semantic-memory/src/chunker.rs:173` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0322 | P2 | static | error-default | `semantic-memory/src/chunker.rs:195` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0323 | P2 | static | error-default | `semantic-memory/src/chunker.rs:204` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0324 | P2 | static | runtime-clock | `semantic-memory/src/conversation.rs:63` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0325 | P2 | static | numeric-cast | `semantic-memory/src/conversation.rs:85` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0326 | P2 | static | error-default | `semantic-memory/src/conversation.rs:172` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0327 | P2 | static | numeric-cast | `semantic-memory/src/conversation.rs:191` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0328 | P2 | static | runtime-clock | `semantic-memory/src/conversation.rs:251` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0329 | P2 | static | runtime-clock | `semantic-memory/src/conversation.rs:319` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0330 | P2 | static | numeric-cast | `semantic-memory/src/conversation.rs:392` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0331 | P2 | static | numeric-cast | `semantic-memory/src/conversation.rs:399` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0332 | P2 | static | runtime-clock | `semantic-memory/src/conversation.rs:433` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0333 | P2 | static | numeric-cast | `semantic-memory/src/conversation.rs:564` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0334 | P2 | static | numeric-cast | `semantic-memory/src/conversation.rs:629` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0335 | P2 | static | error-default | `semantic-memory/src/conversation.rs:638` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0336 | P2 | static | error-default | `semantic-memory/src/conversation.rs:672` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0337 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:16` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0338 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:17` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0339 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:29` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0340 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:43` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0341 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:44` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0342 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:69` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0343 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:80` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0344 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:102` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0345 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:161` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0346 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:171` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0347 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:194` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0348 | P1 | static | insert-or-ignore | `semantic-memory/src/db.rs:197` | INSERT OR IGNORE can hide duplicate/constraint bugs | Validate duplicates before insert or assert affected row count where required. |
| SM-AUD-0349 | P1 | static | insert-or-ignore | `semantic-memory/src/db.rs:208` | INSERT OR IGNORE can hide duplicate/constraint bugs | Validate duplicates before insert or assert affected row count where required. |
| SM-AUD-0350 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:297` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0351 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:298` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0352 | P1 | static | insert-or-ignore | `semantic-memory/src/db.rs:370` | INSERT OR IGNORE can hide duplicate/constraint bugs | Validate duplicates before insert or assert affected row count where required. |
| SM-AUD-0353 | P1 | static | dynamic-sql | `semantic-memory/src/db.rs:558` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0354 | P1 | static | dynamic-sql | `semantic-memory/src/db.rs:581` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0355 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:594` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0356 | P2 | static | error-default | `semantic-memory/src/db.rs:605` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0357 | P2 | static | error-default | `semantic-memory/src/db.rs:652` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0358 | P1 | static | dynamic-sql | `semantic-memory/src/db.rs:653` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0359 | P1 | static | dynamic-sql | `semantic-memory/src/db.rs:705` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0360 | P1 | static | dynamic-sql | `semantic-memory/src/db.rs:715` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0361 | P2 | static | error-default | `semantic-memory/src/db.rs:735` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0362 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:752` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0363 | P1 | confirmed | bytemuck-storage | `semantic-memory/src/db.rs:788` | Storage byte cast is alignment/endian fragile | Decode storage bytes via from_le_bytes/db::bytes_to_embedding. |
| SM-AUD-0364 | P2 | static | error-default | `semantic-memory/src/db.rs:807` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0365 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:828` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0366 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:834` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0367 | P2 | static | error-default | `semantic-memory/src/db.rs:863` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0368 | P2 | static | numeric-cast | `semantic-memory/src/db.rs:887` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0369 | P2 | static | error-default | `semantic-memory/src/db.rs:903` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0370 | P2 | static | numeric-cast | `semantic-memory/src/db.rs:911` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0371 | P2 | static | runtime-clock | `semantic-memory/src/db.rs:926` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0372 | P2 | static | error-default | `semantic-memory/src/db.rs:1023` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0373 | P2 | static | error-default | `semantic-memory/src/db.rs:1197` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0374 | P2 | static | error-default | `semantic-memory/src/db.rs:1392` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0375 | P1 | static | dynamic-sql | `semantic-memory/src/db.rs:1400` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0376 | P2 | static | error-default | `semantic-memory/src/db.rs:1401` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0377 | P1 | static | zip-truncate | `semantic-memory/src/documents.rs:70` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0378 | P2 | static | numeric-cast | `semantic-memory/src/documents.rs:78` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0379 | P2 | static | numeric-cast | `semantic-memory/src/documents.rs:80` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0380 | P2 | static | numeric-cast | `semantic-memory/src/documents.rs:168` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0381 | P2 | static | numeric-cast | `semantic-memory/src/documents.rs:187` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0382 | P2 | static | numeric-cast | `semantic-memory/src/documents.rs:195` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0383 | P1 | static | dynamic-sql | `semantic-memory/src/documents.rs:234` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0384 | P1 | static | zip-truncate | `semantic-memory/src/documents.rs:324` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0385 | P2 | static | error-default | `semantic-memory/src/documents.rs:330` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0386 | P2 | static | error-default | `semantic-memory/src/documents.rs:426` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0387 | P2 | static | error-default | `semantic-memory/src/embedder.rs:127` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0388 | P2 | static | numeric-cast | `semantic-memory/src/embedder.rs:179` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0389 | P2 | static | numeric-cast | `semantic-memory/src/embedder.rs:250` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0390 | P2 | static | numeric-cast | `semantic-memory/src/embedder.rs:251` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0391 | P2 | static | runtime-clock | `semantic-memory/src/episodes.rs:90` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0392 | P2 | static | error-default | `semantic-memory/src/episodes.rs:167` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0393 | P2 | static | runtime-clock | `semantic-memory/src/episodes.rs:191` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0394 | P2 | static | runtime-clock | `semantic-memory/src/episodes.rs:231` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0395 | P1 | static | insert-or-ignore | `semantic-memory/src/episodes.rs:285` | INSERT OR IGNORE can hide duplicate/constraint bugs | Validate duplicates before insert or assert affected row count where required. |
| SM-AUD-0396 | P2 | static | numeric-cast | `semantic-memory/src/episodes.rs:287` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0397 | P2 | static | runtime-clock | `semantic-memory/src/episodes.rs:369` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0398 | P1 | static | dynamic-sql | `semantic-memory/src/episodes.rs:410` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0399 | P1 | static | dynamic-sql | `semantic-memory/src/episodes.rs:414` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0400 | P1 | static | dynamic-sql | `semantic-memory/src/episodes.rs:417` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0401 | P2 | static | error-default | `semantic-memory/src/episodes.rs:641` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0402 | P2 | static | error-default | `semantic-memory/src/episodes.rs:700` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0403 | P2 | static | error-default | `semantic-memory/src/episodes.rs:779` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0404 | P2 | static | error-default | `semantic-memory/src/episodes.rs:843` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0405 | P2 | static | numeric-cast | `semantic-memory/src/graph.rs:34` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0406 | P2 | static | numeric-cast | `semantic-memory/src/graph.rs:55` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0407 | P2 | static | error-default | `semantic-memory/src/graph.rs:99` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0408 | P2 | static | error-default | `semantic-memory/src/graph.rs:469` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0409 | P2 | static | error-default | `semantic-memory/src/graph.rs:597` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0410 | P2 | static | numeric-cast | `semantic-memory/src/graph.rs:607` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0411 | P2 | static | error-default | `semantic-memory/src/graph.rs:680` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0412 | P2 | static | error-default | `semantic-memory/src/graph.rs:691` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0413 | P2 | static | error-default | `semantic-memory/src/graph.rs:702` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0414 | P2 | static | error-default | `semantic-memory/src/graph.rs:714` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0415 | P2 | static | error-default | `semantic-memory/src/graph.rs:726` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0416 | P2 | static | error-default | `semantic-memory/src/graph.rs:737` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0417 | P2 | static | numeric-cast | `semantic-memory/src/graph.rs:767` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0418 | P2 | static | error-default | `semantic-memory/src/graph.rs:819` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0419 | P2 | static | numeric-cast | `semantic-memory/src/graph.rs:832` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0420 | P2 | static | numeric-cast | `semantic-memory/src/graph.rs:893` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0421 | P2 | static | error-default | `semantic-memory/src/graph.rs:920` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0422 | P2 | static | hashmap-order | `semantic-memory/src/hnsw.rs:10` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0423 | P2 | static | hashmap-order | `semantic-memory/src/hnsw.rs:77` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0424 | P2 | static | hashmap-order | `semantic-memory/src/hnsw.rs:79` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0425 | P2 | static | runtime-clock | `semantic-memory/src/hnsw.rs:84` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0426 | P2 | static | error-default | `semantic-memory/src/hnsw.rs:86` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0427 | P2 | static | numeric-cast | `semantic-memory/src/hnsw.rs:250` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0428 | P2 | static | numeric-cast | `semantic-memory/src/hnsw.rs:291` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0429 | P2 | static | numeric-cast | `semantic-memory/src/hnsw.rs:294` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0430 | P2 | static | error-default | `semantic-memory/src/hnsw.rs:317` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0431 | P2 | static | hashmap-order | `semantic-memory/src/hnsw.rs:324` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0432 | P2 | static | hashmap-order | `semantic-memory/src/hnsw.rs:326` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0433 | P2 | static | numeric-cast | `semantic-memory/src/hnsw.rs:332` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0434 | P2 | static | error-default | `semantic-memory/src/hnsw.rs:359` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0435 | P2 | static | error-default | `semantic-memory/src/hnsw.rs:360` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0436 | P2 | static | numeric-cast | `semantic-memory/src/hnsw.rs:496` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0437 | P2 | static | numeric-cast | `semantic-memory/src/hnsw.rs:501` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0438 | P3 | static | unwrap-expect | `semantic-memory/src/hnsw.rs:530` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0439 | P2 | static | numeric-cast | `semantic-memory/src/hnsw.rs:533` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0440 | P3 | static | unwrap-expect | `semantic-memory/src/hnsw.rs:534` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0441 | P3 | static | unwrap-expect | `semantic-memory/src/hnsw.rs:543` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0442 | P3 | static | unwrap-expect | `semantic-memory/src/hnsw.rs:552` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0443 | P3 | static | unwrap-expect | `semantic-memory/src/hnsw.rs:564` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0444 | P3 | static | unwrap-expect | `semantic-memory/src/hnsw.rs:566` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0445 | P3 | static | unwrap-expect | `semantic-memory/src/hnsw.rs:575` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0446 | P3 | static | unwrap-expect | `semantic-memory/src/hnsw.rs:576` | unwrap/expect occurrence should be reviewed | Replace in production paths; leave in tests only when failure message is useful. |
| SM-AUD-0447 | P2 | static | error-default | `semantic-memory/src/json_compat_import.rs:25` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0448 | P2 | static | error-default | `semantic-memory/src/json_compat_import.rs:82` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0449 | P2 | static | error-default | `semantic-memory/src/json_compat_import.rs:91` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0450 | P2 | static | error-default | `semantic-memory/src/json_compat_import.rs:118` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0451 | P2 | static | error-default | `semantic-memory/src/json_compat_import.rs:145` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0452 | P2 | static | runtime-clock | `semantic-memory/src/knowledge.rs:213` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0453 | P2 | static | numeric-cast | `semantic-memory/src/knowledge.rs:621` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0454 | P2 | static | error-default | `semantic-memory/src/knowledge.rs:692` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0455 | P2 | static | error-default | `semantic-memory/src/knowledge.rs:767` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0456 | P2 | static | error-default | `semantic-memory/src/knowledge.rs:817` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0457 | P2 | static | error-default | `semantic-memory/src/lib.rs:413` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0458 | P2 | static | numeric-cast | `semantic-memory/src/lib.rs:416` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0459 | P2 | static | error-default | `semantic-memory/src/lib.rs:465` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0460 | P2 | static | error-default | `semantic-memory/src/lib.rs:771` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0461 | P2 | static | error-default | `semantic-memory/src/lib.rs:782` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0462 | P2 | static | error-default | `semantic-memory/src/lib.rs:845` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0463 | P2 | static | error-default | `semantic-memory/src/lib.rs:927` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0464 | P2 | static | error-default | `semantic-memory/src/lib.rs:949` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0465 | P2 | static | error-default | `semantic-memory/src/lib.rs:1028` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0466 | P1 | static | zip-truncate | `semantic-memory/src/lib.rs:1123` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0467 | P2 | static | error-default | `semantic-memory/src/lib.rs:1176` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0468 | P2 | static | error-default | `semantic-memory/src/lib.rs:1184` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0469 | P2 | static | error-default | `semantic-memory/src/lib.rs:1214` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0470 | P1 | static | zip-truncate | `semantic-memory/src/lib.rs:1254` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0471 | P2 | static | error-default | `semantic-memory/src/lib.rs:1260` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0472 | P2 | static | runtime-clock | `semantic-memory/src/lib.rs:1269` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0473 | P1 | static | zip-truncate | `semantic-memory/src/lib.rs:1314` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0474 | P2 | static | error-default | `semantic-memory/src/lib.rs:1320` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0475 | P1 | static | zip-truncate | `semantic-memory/src/lib.rs:1374` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0476 | P2 | static | error-default | `semantic-memory/src/lib.rs:1380` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0477 | P1 | static | zip-truncate | `semantic-memory/src/lib.rs:1434` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0478 | P2 | static | error-default | `semantic-memory/src/lib.rs:1440` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0479 | P2 | static | runtime-clock | `semantic-memory/src/lib.rs:1452` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0480 | P2 | static | numeric-cast | `semantic-memory/src/pool.rs:289` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0481 | P2 | static | numeric-cast | `semantic-memory/src/pool.rs:295` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0482 | P2 | static | numeric-cast | `semantic-memory/src/pool.rs:305` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0483 | P2 | static | error-default | `semantic-memory/src/projection_batch.rs:71` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0484 | P2 | static | error-default | `semantic-memory/src/projection_import.rs:206` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0485 | P2 | static | runtime-clock | `semantic-memory/src/projection_import.rs:278` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0486 | P2 | static | error-default | `semantic-memory/src/projection_import.rs:301` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0487 | P2 | static | numeric-cast | `semantic-memory/src/projection_import.rs:324` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0488 | P2 | static | numeric-cast | `semantic-memory/src/projection_import.rs:347` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0489 | P2 | static | numeric-cast | `semantic-memory/src/projection_import.rs:357` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0490 | P2 | static | numeric-cast | `semantic-memory/src/projection_import.rs:396` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0491 | P2 | static | error-default | `semantic-memory/src/projection_import.rs:422` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0492 | P2 | static | error-default | `semantic-memory/src/projection_import.rs:433` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0493 | P2 | static | error-default | `semantic-memory/src/projection_lane.rs:130` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0494 | P2 | static | error-default | `semantic-memory/src/projection_lane.rs:644` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0495 | P2 | static | runtime-clock | `semantic-memory/src/projection_lane.rs:674` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0496 | P2 | static | runtime-clock | `semantic-memory/src/projection_lane.rs:952` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0497 | P2 | static | runtime-clock | `semantic-memory/src/projection_lane.rs:1024` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0498 | P2 | static | runtime-clock | `semantic-memory/src/projection_lane.rs:1046` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0499 | P2 | static | runtime-clock | `semantic-memory/src/projection_lane.rs:1090` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0500 | P2 | static | runtime-clock | `semantic-memory/src/projection_lane.rs:1412` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0501 | P2 | static | error-default | `semantic-memory/src/projection_legacy_compat.rs:50` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0502 | P2 | static | error-default | `semantic-memory/src/projection_legacy_compat.rs:92` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0503 | P2 | static | error-default | `semantic-memory/src/projection_legacy_compat.rs:127` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0504 | P2 | static | error-default | `semantic-memory/src/projection_legacy_compat.rs:130` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0505 | P2 | static | error-default | `semantic-memory/src/projection_legacy_compat.rs:135` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0506 | P1 | static | insert-or-ignore | `semantic-memory/src/projection_legacy_compat.rs:177` | INSERT OR IGNORE can hide duplicate/constraint bugs | Validate duplicates before insert or assert affected row count where required. |
| SM-AUD-0507 | P2 | static | numeric-cast | `semantic-memory/src/projection_legacy_compat.rs:179` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0508 | P2 | static | runtime-clock | `semantic-memory/src/projection_legacy_compat.rs:207` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0509 | P2 | static | numeric-cast | `semantic-memory/src/projection_legacy_compat.rs:264` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0510 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:58` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0511 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:99` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0512 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:153` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0513 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:171` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0514 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:201` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0515 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:226` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0516 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:253` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0517 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:309` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0518 | P2 | static | error-default | `semantic-memory/src/projection_storage.rs:362` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0519 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:430` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0520 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:477` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0521 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:520` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0522 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:521` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0523 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:615` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0524 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:616` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0525 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:617` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0526 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:618` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0527 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:619` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0528 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:620` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0529 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:627` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0530 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:701` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0531 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:702` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0532 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:703` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0533 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:704` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0534 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:705` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0535 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:706` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0536 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:713` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0537 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:756` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0538 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:764` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0539 | P2 | static | runtime-clock | `semantic-memory/src/projection_storage.rs:894` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0540 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage.rs:922` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0541 | P2 | static | error-default | `semantic-memory/src/projection_storage_query.rs:17` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0542 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:134` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0543 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:256` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0544 | P2 | static | error-default | `semantic-memory/src/projection_storage_query.rs:271` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0545 | P2 | static | error-default | `semantic-memory/src/projection_storage_query.rs:272` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0546 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:370` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0547 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:474` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0548 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:575` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0549 | P2 | static | error-default | `semantic-memory/src/projection_storage_query.rs:586` | Error/default swallowing should be audited | Preserve typed error or add explicit comment/test proving default is safe. |
| SM-AUD-0550 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:651` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0551 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:668` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0552 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:727` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0553 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:743` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0554 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:761` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0555 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:762` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0556 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:763` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0557 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:764` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0558 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:765` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0559 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:766` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0560 | P2 | static | numeric-cast | `semantic-memory/src/projection_storage_query.rs:797` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0561 | P2 | static | numeric-cast | `semantic-memory/src/quantize.rs:71` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0562 | P2 | static | numeric-cast | `semantic-memory/src/quantize.rs:76` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0563 | P2 | static | numeric-cast | `semantic-memory/src/quantize.rs:77` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0564 | P2 | static | numeric-cast | `semantic-memory/src/quantize.rs:92` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0565 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:9` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0566 | P1 | confirmed | debug-assert | `semantic-memory/src/search.rs:61` | debug_assert is not a release invariant | Use a normal check for correctness invariants. |
| SM-AUD-0567 | P1 | static | zip-truncate | `semantic-memory/src/search.rs:62` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0568 | P2 | static | runtime-clock | `semantic-memory/src/search.rs:73` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0569 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:75` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0570 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:104` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0571 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:186` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0572 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:189` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0573 | P1 | confirmed | bytemuck-storage | `semantic-memory/src/search.rs:272` | Storage byte cast is alignment/endian fragile | Decode storage bytes via from_le_bytes/db::bytes_to_embedding. |
| SM-AUD-0574 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:290` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0575 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:367` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0576 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:409` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0577 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:429` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0578 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:457` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0579 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:504` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0580 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:633` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0581 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:755` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0582 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:962` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0583 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:964` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0584 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:966` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0585 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:968` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0586 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:971` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0587 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:1061` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0588 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:1109` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0589 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:1173` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0590 | P2 | static | numeric-cast | `semantic-memory/src/search.rs:1232` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0591 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:1255` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0592 | P2 | static | hashmap-order | `semantic-memory/src/search.rs:1321` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0593 | P2 | static | runtime-clock | `semantic-memory/src/store_support.rs:102` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0594 | P2 | static | numeric-cast | `semantic-memory/tests/db_tests.rs:32` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0595 | P1 | static | zip-truncate | `semantic-memory/tests/db_tests.rs:36` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0596 | P1 | static | dynamic-sql | `semantic-memory/tests/import_ugly_cases.rs:33` | Dynamic SQL construction should be whitelisted | Use whitelisted enums for identifiers and bind parameters for values. |
| SM-AUD-0597 | P2 | static | numeric-cast | `semantic-memory/tests/knowledge_tests.rs:538` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0598 | P1 | static | zip-truncate | `semantic-memory/tests/knowledge_tests.rs:555` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0599 | P1 | static | zip-truncate | `semantic-memory/tests/quantization.rs:9` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0600 | P2 | static | numeric-cast | `semantic-memory/tests/quantization.rs:29` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0601 | P2 | static | numeric-cast | `semantic-memory/tests/quantization.rs:30` | Unchecked numeric cast should be audited | Prefer TryFrom, checked conversion, or explicit range assertion. |
| SM-AUD-0602 | P1 | static | zip-truncate | `semantic-memory/tests/quantization.rs:46` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0603 | P1 | static | zip-truncate | `semantic-memory/tests/quantization.rs:68` | zip iteration can silently truncate mismatched collections | Pre-check lengths before zip unless truncation is intentional and documented. |
| SM-AUD-0604 | P2 | static | runtime-clock | `semantic-memory/tests/search_tests.rs:672` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0605 | P2 | static | runtime-clock | `semantic-memory-forge/src/bundle.rs:351` | Runtime clock use affects replay/determinism | Inject a clock or document that this path is intentionally real-time. |
| SM-AUD-0606 | P2 | static | hashmap-order | `stack-ids/src/digest.rs:234` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0607 | P2 | static | hashmap-order | `stack-ids/src/digest.rs:275` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |
| SM-AUD-0608 | P2 | static | hashmap-order | `stack-ids/src/digest.rs:279` | HashMap order should be audited in deterministic surfaces | Sort before output or use BTreeMap where deterministic ordering matters. |