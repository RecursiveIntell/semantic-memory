//! Projection storage tables and operations for V11+ schema.
//!
//! This module provides the storage layer for claim projection versions,
//! relation versions, entity aliases, evidence refs, and import tracking
//! introduced by the canonical stack v4 architecture.
//!
//! ## Authority
//!
//! `semantic-memory` is authoritative for queryable knowledge state. These
//! projection tables store the imported, queryable representation of data
//! that originated from Forge or other sources via the bridge.
//!
//! ## Validation (SM-004)
//!
//! Entity alias `review_state` and `merge_decision` values are validated at the
//! application level before insertion (see `validate_review_state()` and
//! `validate_merge_decision()` in `lib.rs`). DB-level CHECK constraints exist for
//! claim_state and freshness columns; review_state/merge_decision use application
//! validation because they can contain JSON variant objects.
//!
//! ## Phase status: current / implemented now

use crate::error::MemoryError;
use crate::types::{
    ProjectionClaimVersion, ProjectionEntityAlias, ProjectionEpisode, ProjectionEvidenceRef,
    ProjectionQuery, ProjectionRelationVersion,
};
use stack_ids::{
    ClaimId, ClaimVersionId, EntityId, EnvelopeId, EpisodeId, RelationVersionId, ScopeKey,
};

/// V11 migration: projection storage tables for canonical stack v4.
///
/// Adds tables for:
/// - `claim_versions` — versioned knowledge assertions with temporal validity
/// - `relation_versions` — versioned entity relations with audit metadata
/// - `entity_aliases` — entity alias/merge state with review durability
/// - `evidence_refs` — opaque evidence references for audit dereference
/// - `projection_import_log` — enhanced import tracking with scope keys
/// - `derivation_edges` — lineage tracking between projections
pub(crate) const MIGRATION_V11: &str = r#"
-- CLAIM PROJECTION VERSIONS
-- Each row is a specific version of a claim. The claim_id is stable across
-- versions; claim_version_id is unique per mutation.
CREATE TABLE IF NOT EXISTS claim_versions (
    claim_version_id        TEXT PRIMARY KEY,
    claim_id                TEXT NOT NULL,
    claim_state             TEXT NOT NULL DEFAULT 'active'
                            CHECK (claim_state IN ('active', 'superseded', 'retracted', 'archived', 'pending_review', 'disputed')),
    projection_family       TEXT NOT NULL,
    subject_entity_id       TEXT NOT NULL,
    predicate               TEXT NOT NULL,
    object_anchor           TEXT NOT NULL,
    scope_namespace         TEXT NOT NULL,
    scope_domain            TEXT,
    scope_workspace_id      TEXT,
    scope_repo_id           TEXT,
    valid_from              TEXT,
    valid_to                TEXT,
    recorded_at             TEXT NOT NULL DEFAULT (datetime('now')),
    preferred_open          INTEGER NOT NULL DEFAULT 0,
    source_envelope_id      TEXT NOT NULL,
    source_authority        TEXT NOT NULL,
    trace_id                TEXT,
    freshness               TEXT NOT NULL DEFAULT 'current'
                            CHECK (freshness IN ('current', 'stale', 'superseded', 'import_failed', 'never_imported', 'import_lagging')),
    contradiction_status    TEXT NOT NULL DEFAULT 'none',
    supersedes_claim_version_id TEXT,
    content                 TEXT NOT NULL,
    confidence              REAL NOT NULL DEFAULT 1.0,
    content_digest          TEXT,
    metadata                TEXT
);

CREATE INDEX IF NOT EXISTS idx_cv_claim_id ON claim_versions(claim_id);
CREATE INDEX IF NOT EXISTS idx_cv_subject ON claim_versions(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_cv_scope ON claim_versions(scope_namespace, scope_domain);
CREATE INDEX IF NOT EXISTS idx_cv_predicate ON claim_versions(predicate);
CREATE INDEX IF NOT EXISTS idx_cv_state ON claim_versions(claim_state);
-- Enforce at most one preferred_open row per logical claim (I014).
CREATE UNIQUE INDEX IF NOT EXISTS idx_cv_preferred ON claim_versions(claim_id) WHERE preferred_open = 1;
CREATE INDEX IF NOT EXISTS idx_cv_recorded ON claim_versions(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_cv_envelope ON claim_versions(source_envelope_id);
CREATE INDEX IF NOT EXISTS idx_cv_freshness ON claim_versions(freshness);

-- RELATION VERSIONS
-- Preserves audit-grade metadata parity with claim versions.
CREATE TABLE IF NOT EXISTS relation_versions (
    relation_version_id     TEXT PRIMARY KEY,
    subject_entity_id       TEXT NOT NULL,
    predicate               TEXT NOT NULL,
    object_anchor           TEXT NOT NULL,
    scope_namespace         TEXT NOT NULL,
    scope_domain            TEXT,
    scope_workspace_id      TEXT,
    scope_repo_id           TEXT,
    claim_id                TEXT,
    source_episode_id       TEXT,
    valid_from              TEXT,
    valid_to                TEXT,
    recorded_at             TEXT NOT NULL DEFAULT (datetime('now')),
    preferred_open          INTEGER NOT NULL DEFAULT 0,
    supersedes_relation_version_id TEXT,
    contradiction_status    TEXT NOT NULL DEFAULT 'none',
    source_confidence       REAL NOT NULL DEFAULT 1.0,
    projection_family       TEXT NOT NULL,
    source_envelope_id      TEXT NOT NULL,
    source_authority        TEXT NOT NULL,
    trace_id                TEXT,
    freshness               TEXT NOT NULL DEFAULT 'current'
                            CHECK (freshness IN ('current', 'stale', 'superseded', 'import_failed', 'never_imported', 'import_lagging')),
    metadata                TEXT
);

CREATE INDEX IF NOT EXISTS idx_rv_subject ON relation_versions(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_rv_predicate ON relation_versions(predicate);
CREATE INDEX IF NOT EXISTS idx_rv_scope ON relation_versions(scope_namespace, scope_domain);
CREATE INDEX IF NOT EXISTS idx_rv_claim ON relation_versions(claim_id);
CREATE INDEX IF NOT EXISTS idx_rv_episode ON relation_versions(source_episode_id);
CREATE INDEX IF NOT EXISTS idx_rv_envelope ON relation_versions(source_envelope_id);
CREATE INDEX IF NOT EXISTS idx_rv_recorded ON relation_versions(recorded_at DESC);
-- Enforce at most one preferred_open row per logical relation key.
-- V5 spec §11.5: logical key is (subject_entity_id, predicate, object_anchor, scope_key, projection_family).
-- scope_key decomposes to (scope_namespace, scope_domain, scope_workspace_id, scope_repo_id).
-- COALESCE maps NULLs to '' so the uniqueness constraint works correctly.
CREATE UNIQUE INDEX IF NOT EXISTS idx_rv_preferred
    ON relation_versions(
        subject_entity_id, predicate, object_anchor,
        scope_namespace, COALESCE(scope_domain, ''),
        COALESCE(scope_workspace_id, ''), COALESCE(scope_repo_id, ''),
        projection_family
    )
    WHERE preferred_open = 1;

-- ENTITY ALIASES
-- Includes explicit scope semantics and durable review state.
CREATE TABLE IF NOT EXISTS entity_aliases (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_entity_id     TEXT NOT NULL,
    alias_text              TEXT NOT NULL,
    alias_source            TEXT NOT NULL,
    match_evidence          TEXT,
    confidence              REAL NOT NULL DEFAULT 0.0,
    merge_decision          TEXT NOT NULL DEFAULT 'pending_review',
    scope_namespace         TEXT NOT NULL,
    scope_domain            TEXT,
    scope_workspace_id      TEXT,
    scope_repo_id           TEXT,
    review_state            TEXT NOT NULL DEFAULT 'unreviewed',
    is_human_confirmed      INTEGER NOT NULL DEFAULT 0,
    is_human_confirmed_final INTEGER NOT NULL DEFAULT 0,
    superseded_by_entity_id TEXT,
    split_from_entity_id    TEXT,
    source_envelope_id      TEXT NOT NULL,
    recorded_at             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ea_canonical ON entity_aliases(canonical_entity_id);
CREATE INDEX IF NOT EXISTS idx_ea_alias ON entity_aliases(alias_text);
CREATE INDEX IF NOT EXISTS idx_ea_scope ON entity_aliases(scope_namespace, scope_domain);
CREATE INDEX IF NOT EXISTS idx_ea_review ON entity_aliases(review_state) WHERE review_state = 'pending_review';
CREATE INDEX IF NOT EXISTS idx_ea_envelope ON entity_aliases(source_envelope_id);

-- EVIDENCE REFS
-- Opaque by default with explicit audit-only dereference.
CREATE TABLE IF NOT EXISTS evidence_refs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id                TEXT NOT NULL,
    claim_version_id        TEXT,
    fetch_handle            TEXT NOT NULL,
    source_authority        TEXT NOT NULL,
    source_envelope_id      TEXT NOT NULL,
    recorded_at             TEXT NOT NULL DEFAULT (datetime('now')),
    metadata                TEXT
);

CREATE INDEX IF NOT EXISTS idx_er_claim ON evidence_refs(claim_id);
CREATE INDEX IF NOT EXISTS idx_er_version ON evidence_refs(claim_version_id);
CREATE INDEX IF NOT EXISTS idx_er_envelope ON evidence_refs(source_envelope_id);

-- PROJECTION IMPORT LOG (enhanced)
-- Extends V10 import_log with scope key columns.
CREATE TABLE IF NOT EXISTS projection_import_log (
    batch_id                TEXT PRIMARY KEY,
    source_envelope_id      TEXT NOT NULL,
    schema_version          TEXT NOT NULL,
    content_digest          TEXT NOT NULL,
    source_authority        TEXT NOT NULL,
    scope_namespace         TEXT NOT NULL,
    scope_domain            TEXT,
    scope_workspace_id      TEXT,
    scope_repo_id           TEXT,
    trace_id                TEXT,
    record_count            INTEGER NOT NULL,
    claim_count             INTEGER NOT NULL DEFAULT 0,
    relation_count          INTEGER NOT NULL DEFAULT 0,
    episode_count           INTEGER NOT NULL DEFAULT 0,
    alias_count             INTEGER NOT NULL DEFAULT 0,
    evidence_count          INTEGER NOT NULL DEFAULT 0,
    status                  TEXT NOT NULL DEFAULT 'complete',
    source_exported_at      TEXT,
    transformed_at          TEXT,
    imported_at             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pil_envelope ON projection_import_log(source_envelope_id);
CREATE INDEX IF NOT EXISTS idx_pil_scope ON projection_import_log(scope_namespace);
CREATE INDEX IF NOT EXISTS idx_pil_imported ON projection_import_log(imported_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pil_dedupe
    ON projection_import_log(source_envelope_id, schema_version, content_digest);

-- EPISODE LINKS
-- Projection-level episode tracking for imported episodes.
-- Stores the episode projection record as received from the bridge,
-- separate from the searchable episodes table (which includes embeddings).
CREATE TABLE IF NOT EXISTS episode_links (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id              TEXT NOT NULL,
    document_id             TEXT NOT NULL,
    cause_ids               TEXT NOT NULL DEFAULT '[]',
    effect_type             TEXT NOT NULL,
    outcome                 TEXT NOT NULL,
    confidence              REAL NOT NULL DEFAULT 0.0,
    experiment_id           TEXT,
    source_envelope_id      TEXT NOT NULL,
    source_authority        TEXT NOT NULL,
    trace_id                TEXT,
    recorded_at             TEXT NOT NULL DEFAULT (datetime('now')),
    metadata                TEXT
);

CREATE INDEX IF NOT EXISTS idx_el_episode ON episode_links(episode_id);
CREATE INDEX IF NOT EXISTS idx_el_document ON episode_links(document_id);
CREATE INDEX IF NOT EXISTS idx_el_envelope ON episode_links(source_envelope_id);

-- DERIVATION EDGES
-- Tracks lineage between projection records.
-- Each derived artifact declares an invalidation_mode.
CREATE TABLE IF NOT EXISTS derivation_edges (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_kind             TEXT NOT NULL,
    source_id               TEXT NOT NULL,
    target_kind             TEXT NOT NULL,
    target_id               TEXT NOT NULL,
    derivation_type         TEXT NOT NULL,
    invalidation_mode       TEXT NOT NULL DEFAULT 'on_source_change'
                            CHECK (invalidation_mode IN (
                                'on_source_change', 'on_contradiction', 'on_refutation',
                                'on_alias_split', 'on_supersession', 'on_estimator_change',
                                'on_policy_change', 'manual_only'
                            )),
    is_invalidated          INTEGER NOT NULL DEFAULT 0,
    invalidated_at          TEXT,
    invalidation_reason     TEXT,
    recorded_at             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_de_source ON derivation_edges(source_kind, source_id);
CREATE INDEX IF NOT EXISTS idx_de_target ON derivation_edges(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_de_invalidated ON derivation_edges(is_invalidated) WHERE is_invalidated = 1;
"#;

/// V12 migration: fix relation_versions preferred_open unique index (I011).
///
/// The V11 index keyed only on (subject_entity_id, predicate, object_anchor, scope_namespace),
/// missing scope_domain/workspace_id/repo_id/projection_family. This caused distinct scoped
/// relations to collide or be blocked incorrectly.
pub(crate) const MIGRATION_V12: &str = r#"
-- Drop the under-scoped V11 preferred_open index.
DROP INDEX IF EXISTS idx_rv_preferred;
-- Recreate with the full logical scope key per V5 spec §11.5.
CREATE UNIQUE INDEX IF NOT EXISTS idx_rv_preferred
    ON relation_versions(
        subject_entity_id, predicate, object_anchor,
        scope_namespace, COALESCE(scope_domain, ''),
        COALESCE(scope_workspace_id, ''), COALESCE(scope_repo_id, ''),
        projection_family
    )
    WHERE preferred_open = 1;
"#;

/// V13 migration: persist export_schema_version separately from import schema.
pub(crate) const MIGRATION_V13: &str = r#"
ALTER TABLE projection_import_log ADD COLUMN export_schema_version TEXT;
"#;

/// V14 migration: preserve V2 export metadata and durable failure receipts.
pub(crate) const MIGRATION_V14: &str = r#"
ALTER TABLE projection_import_log ADD COLUMN source_run_id TEXT;
ALTER TABLE projection_import_log ADD COLUMN comparability_snapshot_version TEXT;
ALTER TABLE projection_import_log ADD COLUMN direct_write INTEGER NOT NULL DEFAULT 0;
ALTER TABLE projection_import_log ADD COLUMN failure_reason TEXT;

CREATE TABLE IF NOT EXISTS projection_import_failures (
    failure_id               TEXT PRIMARY KEY,
    source_envelope_id       TEXT NOT NULL,
    schema_version           TEXT NOT NULL,
    export_schema_version    TEXT,
    content_digest           TEXT NOT NULL,
    source_authority         TEXT NOT NULL,
    scope_namespace          TEXT NOT NULL,
    scope_domain             TEXT,
    scope_workspace_id       TEXT,
    scope_repo_id            TEXT,
    trace_id                 TEXT,
    record_count             INTEGER NOT NULL,
    error_kind               TEXT NOT NULL,
    error_message            TEXT NOT NULL,
    source_exported_at       TEXT,
    transformed_at           TEXT,
    failed_at                TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pif_envelope ON projection_import_failures(source_envelope_id);
CREATE INDEX IF NOT EXISTS idx_pif_scope ON projection_import_failures(scope_namespace);
CREATE INDEX IF NOT EXISTS idx_pif_failed_at ON projection_import_failures(failed_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pif_dedupe
    ON projection_import_failures(source_envelope_id, schema_version, content_digest);
"#;

/// V15 migration: preserve canonical evidence-bundle receipts on import log rows.
pub(crate) const MIGRATION_V15: &str = r#"
ALTER TABLE projection_import_log ADD COLUMN evidence_bundle_id TEXT;
ALTER TABLE projection_import_log ADD COLUMN evidence_bundle_json TEXT;

ALTER TABLE projection_import_failures ADD COLUMN source_run_id TEXT;
ALTER TABLE projection_import_failures ADD COLUMN comparability_snapshot_version TEXT;
ALTER TABLE projection_import_failures ADD COLUMN direct_write INTEGER NOT NULL DEFAULT 0;
ALTER TABLE projection_import_failures ADD COLUMN evidence_bundle_id TEXT;
ALTER TABLE projection_import_failures ADD COLUMN evidence_bundle_json TEXT;
"#;

/// V16 migration: preserve rebuildable kernel payloads on import receipts.
pub(crate) const MIGRATION_V16: &str = r#"
ALTER TABLE projection_import_log ADD COLUMN kernel_payload_json TEXT;
ALTER TABLE projection_import_failures ADD COLUMN kernel_payload_json TEXT;
"#;

/// V17 migration: preserve v9 episode-bundle and execution-context proof surfaces.
pub(crate) const MIGRATION_V17: &str = r#"
ALTER TABLE projection_import_log ADD COLUMN episode_bundle_id TEXT;
ALTER TABLE projection_import_log ADD COLUMN episode_bundle_json TEXT;
ALTER TABLE projection_import_log ADD COLUMN execution_context_json TEXT;
ALTER TABLE projection_import_failures ADD COLUMN episode_bundle_id TEXT;
ALTER TABLE projection_import_failures ADD COLUMN episode_bundle_json TEXT;
ALTER TABLE projection_import_failures ADD COLUMN execution_context_json TEXT;
"#;

/// Check if a projection import batch has already been ingested.
pub(crate) fn check_projection_import_exists(
    conn: &rusqlite::Connection,
    source_envelope_id: &str,
    schema_version: &str,
    content_digest: &str,
) -> Result<bool, MemoryError> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM projection_import_log
             WHERE source_envelope_id = ?1 AND schema_version = ?2 AND content_digest = ?3
               AND status = 'complete'",
            rusqlite::params![source_envelope_id, schema_version, content_digest],
            |row| row.get(0),
        )
        .unwrap_or(0);
    Ok(count > 0)
}

pub(crate) fn claim_version_source_envelope(
    conn: &rusqlite::Connection,
    claim_version_id: &str,
) -> Result<Option<String>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT source_envelope_id FROM claim_versions WHERE claim_version_id = ?1 LIMIT 1",
    )?;
    let mut rows = stmt.query(rusqlite::params![claim_version_id])?;
    if let Some(row) = rows.next()? {
        Ok(Some(row.get(0)?))
    } else {
        Ok(None)
    }
}

pub(crate) fn relation_version_source_envelope(
    conn: &rusqlite::Connection,
    relation_version_id: &str,
) -> Result<Option<String>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT source_envelope_id FROM relation_versions WHERE relation_version_id = ?1 LIMIT 1",
    )?;
    let mut rows = stmt.query(rusqlite::params![relation_version_id])?;
    if let Some(row) = rows.next()? {
        Ok(Some(row.get(0)?))
    } else {
        Ok(None)
    }
}

/// Insert a claim version record.
pub(crate) fn insert_claim_version(
    tx: &rusqlite::Transaction<'_>,
    cv: &ClaimVersionRow,
) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO claim_versions (
            claim_version_id, claim_id, claim_state, projection_family,
            subject_entity_id, predicate, object_anchor,
            scope_namespace, scope_domain, scope_workspace_id, scope_repo_id,
            valid_from, valid_to, recorded_at, preferred_open,
            source_envelope_id, source_authority, trace_id,
            freshness, contradiction_status, supersedes_claim_version_id,
            content, confidence, content_digest, metadata
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
            ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
            ?22, ?23, ?24, ?25
        )",
        rusqlite::params![
            cv.claim_version_id,
            cv.claim_id,
            cv.claim_state,
            cv.projection_family,
            cv.subject_entity_id,
            cv.predicate,
            cv.object_anchor,
            cv.scope_namespace,
            cv.scope_domain,
            cv.scope_workspace_id,
            cv.scope_repo_id,
            cv.valid_from,
            cv.valid_to,
            cv.recorded_at,
            cv.preferred_open as i32,
            cv.source_envelope_id,
            cv.source_authority,
            cv.trace_id,
            cv.freshness,
            cv.contradiction_status,
            cv.supersedes_claim_version_id,
            cv.content,
            cv.confidence,
            cv.content_digest,
            cv.metadata,
        ],
    )?;
    Ok(())
}

/// Insert a relation version record.
pub(crate) fn insert_relation_version(
    tx: &rusqlite::Transaction<'_>,
    rv: &RelationVersionRow,
) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO relation_versions (
            relation_version_id, subject_entity_id, predicate, object_anchor,
            scope_namespace, scope_domain, scope_workspace_id, scope_repo_id,
            claim_id, source_episode_id, valid_from, valid_to,
            recorded_at, preferred_open, supersedes_relation_version_id,
            contradiction_status, source_confidence, projection_family,
            source_envelope_id, source_authority, trace_id, freshness, metadata
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
            ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23
        )",
        rusqlite::params![
            rv.relation_version_id,
            rv.subject_entity_id,
            rv.predicate,
            rv.object_anchor,
            rv.scope_namespace,
            rv.scope_domain,
            rv.scope_workspace_id,
            rv.scope_repo_id,
            rv.claim_id,
            rv.source_episode_id,
            rv.valid_from,
            rv.valid_to,
            rv.recorded_at,
            rv.preferred_open as i32,
            rv.supersedes_relation_version_id,
            rv.contradiction_status,
            rv.source_confidence,
            rv.projection_family,
            rv.source_envelope_id,
            rv.source_authority,
            rv.trace_id,
            rv.freshness,
            rv.metadata,
        ],
    )?;
    Ok(())
}

/// Insert an entity alias record.
pub(crate) fn insert_entity_alias(
    tx: &rusqlite::Transaction<'_>,
    ea: &EntityAliasRow,
) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO entity_aliases (
            canonical_entity_id, alias_text, alias_source, match_evidence,
            confidence, merge_decision,
            scope_namespace, scope_domain, scope_workspace_id, scope_repo_id,
            review_state, is_human_confirmed, is_human_confirmed_final,
            superseded_by_entity_id, split_from_entity_id,
            source_envelope_id, recorded_at
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17
        )",
        rusqlite::params![
            ea.canonical_entity_id,
            ea.alias_text,
            ea.alias_source,
            ea.match_evidence,
            ea.confidence,
            ea.merge_decision,
            ea.scope_namespace,
            ea.scope_domain,
            ea.scope_workspace_id,
            ea.scope_repo_id,
            ea.review_state,
            ea.is_human_confirmed as i32,
            ea.is_human_confirmed_final as i32,
            ea.superseded_by_entity_id,
            ea.split_from_entity_id,
            ea.source_envelope_id,
            ea.recorded_at,
        ],
    )?;
    Ok(())
}

/// Insert an evidence ref record.
pub(crate) fn insert_evidence_ref(
    tx: &rusqlite::Transaction<'_>,
    er: &EvidenceRefRow,
) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO evidence_refs (
            claim_id, claim_version_id, fetch_handle,
            source_authority, source_envelope_id, recorded_at, metadata
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            er.claim_id,
            er.claim_version_id,
            er.fetch_handle,
            er.source_authority,
            er.source_envelope_id,
            er.recorded_at,
            er.metadata,
        ],
    )?;
    Ok(())
}

/// Insert a projection import log entry.
pub(crate) fn insert_projection_import_log(
    tx: &rusqlite::Transaction<'_>,
    log: &ProjectionImportLogRow,
) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO projection_import_log (
            batch_id, source_envelope_id, schema_version, export_schema_version,
            content_digest, source_authority, scope_namespace, scope_domain,
            scope_workspace_id, scope_repo_id, trace_id, record_count,
            claim_count, relation_count, episode_count, alias_count,
            evidence_count, status, source_exported_at, transformed_at, imported_at,
            source_run_id, comparability_snapshot_version, direct_write, failure_reason,
            evidence_bundle_id, evidence_bundle_json, episode_bundle_id, episode_bundle_json,
            execution_context_json, kernel_payload_json
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
            ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
            ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31
        )
        ON CONFLICT(source_envelope_id, schema_version, content_digest) DO UPDATE SET
            batch_id = excluded.batch_id,
            export_schema_version = excluded.export_schema_version,
            source_authority = excluded.source_authority,
            scope_namespace = excluded.scope_namespace,
            scope_domain = excluded.scope_domain,
            scope_workspace_id = excluded.scope_workspace_id,
            scope_repo_id = excluded.scope_repo_id,
            trace_id = excluded.trace_id,
            record_count = excluded.record_count,
            claim_count = excluded.claim_count,
            relation_count = excluded.relation_count,
            episode_count = excluded.episode_count,
            alias_count = excluded.alias_count,
            evidence_count = excluded.evidence_count,
            status = excluded.status,
            source_exported_at = excluded.source_exported_at,
            transformed_at = excluded.transformed_at,
            imported_at = excluded.imported_at,
            source_run_id = excluded.source_run_id,
            comparability_snapshot_version = excluded.comparability_snapshot_version,
            direct_write = excluded.direct_write,
            failure_reason = excluded.failure_reason,
            evidence_bundle_id = excluded.evidence_bundle_id,
            evidence_bundle_json = excluded.evidence_bundle_json,
            episode_bundle_id = excluded.episode_bundle_id,
            episode_bundle_json = excluded.episode_bundle_json,
            execution_context_json = excluded.execution_context_json,
            kernel_payload_json = excluded.kernel_payload_json",
        rusqlite::params![
            log.batch_id,
            log.source_envelope_id,
            log.schema_version,
            log.export_schema_version,
            log.content_digest,
            log.source_authority,
            log.scope_namespace,
            log.scope_domain,
            log.scope_workspace_id,
            log.scope_repo_id,
            log.trace_id,
            log.record_count as i64,
            log.claim_count as i64,
            log.relation_count as i64,
            log.episode_count as i64,
            log.alias_count as i64,
            log.evidence_count as i64,
            log.status,
            log.source_exported_at,
            log.transformed_at,
            log.imported_at,
            log.source_run_id,
            log.comparability_snapshot_version,
            log.direct_write as i32,
            log.failure_reason,
            log.evidence_bundle_id,
            log.evidence_bundle_json,
            log.episode_bundle_id,
            log.episode_bundle_json,
            log.execution_context_json,
            log.kernel_payload_json,
        ],
    )?;
    Ok(())
}

/// Upsert a projection import log entry outside the main import transaction.
pub(crate) fn upsert_projection_import_log_conn(
    conn: &rusqlite::Connection,
    log: &ProjectionImportLogRow,
) -> Result<(), MemoryError> {
    conn.execute(
        "INSERT INTO projection_import_log (
            batch_id, source_envelope_id, schema_version, export_schema_version,
            content_digest, source_authority, scope_namespace, scope_domain,
            scope_workspace_id, scope_repo_id, trace_id, record_count,
            claim_count, relation_count, episode_count, alias_count,
            evidence_count, status, source_exported_at, transformed_at, imported_at,
            source_run_id, comparability_snapshot_version, direct_write, failure_reason,
            evidence_bundle_id, evidence_bundle_json, episode_bundle_id, episode_bundle_json,
            execution_context_json, kernel_payload_json
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
            ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
            ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31
        )
        ON CONFLICT(source_envelope_id, schema_version, content_digest) DO UPDATE SET
            batch_id = excluded.batch_id,
            export_schema_version = excluded.export_schema_version,
            source_authority = excluded.source_authority,
            scope_namespace = excluded.scope_namespace,
            scope_domain = excluded.scope_domain,
            scope_workspace_id = excluded.scope_workspace_id,
            scope_repo_id = excluded.scope_repo_id,
            trace_id = excluded.trace_id,
            record_count = excluded.record_count,
            claim_count = excluded.claim_count,
            relation_count = excluded.relation_count,
            episode_count = excluded.episode_count,
            alias_count = excluded.alias_count,
            evidence_count = excluded.evidence_count,
            status = excluded.status,
            source_exported_at = excluded.source_exported_at,
            transformed_at = excluded.transformed_at,
            imported_at = excluded.imported_at,
            source_run_id = excluded.source_run_id,
            comparability_snapshot_version = excluded.comparability_snapshot_version,
            direct_write = excluded.direct_write,
            failure_reason = excluded.failure_reason,
            evidence_bundle_id = excluded.evidence_bundle_id,
            evidence_bundle_json = excluded.evidence_bundle_json,
            episode_bundle_id = excluded.episode_bundle_id,
            episode_bundle_json = excluded.episode_bundle_json,
            execution_context_json = excluded.execution_context_json,
            kernel_payload_json = excluded.kernel_payload_json",
        rusqlite::params![
            log.batch_id,
            log.source_envelope_id,
            log.schema_version,
            log.export_schema_version,
            log.content_digest,
            log.source_authority,
            log.scope_namespace,
            log.scope_domain,
            log.scope_workspace_id,
            log.scope_repo_id,
            log.trace_id,
            log.record_count as i64,
            log.claim_count as i64,
            log.relation_count as i64,
            log.episode_count as i64,
            log.alias_count as i64,
            log.evidence_count as i64,
            log.status,
            log.source_exported_at,
            log.transformed_at,
            log.imported_at,
            log.source_run_id,
            log.comparability_snapshot_version,
            log.direct_write as i32,
            log.failure_reason,
            log.evidence_bundle_id,
            log.evidence_bundle_json,
            log.episode_bundle_id,
            log.episode_bundle_json,
            log.execution_context_json,
            log.kernel_payload_json,
        ],
    )?;
    Ok(())
}

/// Insert or update a durable failed import receipt.
pub(crate) fn insert_projection_import_failure(
    conn: &rusqlite::Connection,
    row: &ProjectionImportFailureRow,
) -> Result<(), MemoryError> {
    conn.execute(
        "INSERT OR REPLACE INTO projection_import_failures (
            failure_id, source_envelope_id, schema_version, export_schema_version,
            content_digest, source_authority, scope_namespace, scope_domain,
            scope_workspace_id, scope_repo_id, trace_id, record_count,
            error_kind, error_message, source_exported_at, transformed_at, failed_at,
            source_run_id, comparability_snapshot_version, direct_write,
            evidence_bundle_id, evidence_bundle_json, episode_bundle_id, episode_bundle_json,
            execution_context_json, kernel_payload_json
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
            ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26
        )",
        rusqlite::params![
            row.failure_id,
            row.source_envelope_id,
            row.schema_version,
            row.export_schema_version,
            row.content_digest,
            row.source_authority,
            row.scope_namespace,
            row.scope_domain,
            row.scope_workspace_id,
            row.scope_repo_id,
            row.trace_id,
            row.record_count as i64,
            row.error_kind,
            row.error_message,
            row.source_exported_at,
            row.transformed_at,
            row.failed_at,
            row.source_run_id,
            row.comparability_snapshot_version,
            row.direct_write as i32,
            row.evidence_bundle_id,
            row.evidence_bundle_json,
            row.episode_bundle_id,
            row.episode_bundle_json,
            row.execution_context_json,
            row.kernel_payload_json,
        ],
    )?;
    Ok(())
}

/// Insert an episode link record.
pub(crate) fn insert_episode_link(
    tx: &rusqlite::Transaction<'_>,
    el: &EpisodeLinkRow,
) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO episode_links (
            episode_id, document_id, cause_ids, effect_type, outcome,
            confidence, experiment_id, source_envelope_id, source_authority,
            trace_id, recorded_at, metadata
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        rusqlite::params![
            el.episode_id,
            el.document_id,
            el.cause_ids,
            el.effect_type,
            el.outcome,
            el.confidence,
            el.experiment_id,
            el.source_envelope_id,
            el.source_authority,
            el.trace_id,
            el.recorded_at,
            el.metadata,
        ],
    )?;
    Ok(())
}

/// Insert a derivation edge with invalidation mode.
pub(crate) fn insert_derivation_edge(
    tx: &rusqlite::Transaction<'_>,
    source_kind: &str,
    source_id: &str,
    target_kind: &str,
    target_id: &str,
    derivation_type: &str,
    invalidation_mode: &str,
) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO derivation_edges (
            source_kind, source_id, target_kind, target_id,
            derivation_type, invalidation_mode
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            source_kind,
            source_id,
            target_kind,
            target_id,
            derivation_type,
            invalidation_mode
        ],
    )?;
    Ok(())
}

/// Query derivation edges by source.
#[allow(dead_code)]
pub(crate) fn query_derivation_edges_by_source(
    conn: &rusqlite::Connection,
    source_kind: &str,
    source_id: &str,
) -> Result<Vec<DerivationEdgeRow>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, source_kind, source_id, target_kind, target_id,
                derivation_type, invalidation_mode, is_invalidated,
                invalidated_at, invalidation_reason, recorded_at
         FROM derivation_edges
         WHERE source_kind = ?1 AND source_id = ?2
         ORDER BY recorded_at ASC",
    )?;
    let rows = stmt
        .query_map(rusqlite::params![source_kind, source_id], |row| {
            Ok(DerivationEdgeRow {
                id: row.get(0)?,
                source_kind: row.get(1)?,
                source_id: row.get(2)?,
                target_kind: row.get(3)?,
                target_id: row.get(4)?,
                derivation_type: row.get(5)?,
                invalidation_mode: row.get(6)?,
                is_invalidated: row.get(7)?,
                invalidated_at: row.get(8)?,
                invalidation_reason: row.get(9)?,
                recorded_at: row.get(10)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Query derivation edges by target.
#[allow(dead_code)]
pub(crate) fn query_derivation_edges_by_target(
    conn: &rusqlite::Connection,
    target_kind: &str,
    target_id: &str,
) -> Result<Vec<DerivationEdgeRow>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, source_kind, source_id, target_kind, target_id,
                derivation_type, invalidation_mode, is_invalidated,
                invalidated_at, invalidation_reason, recorded_at
         FROM derivation_edges
         WHERE target_kind = ?1 AND target_id = ?2
         ORDER BY recorded_at ASC",
    )?;
    let rows = stmt
        .query_map(rusqlite::params![target_kind, target_id], |row| {
            Ok(DerivationEdgeRow {
                id: row.get(0)?,
                source_kind: row.get(1)?,
                source_id: row.get(2)?,
                target_kind: row.get(3)?,
                target_id: row.get(4)?,
                derivation_type: row.get(5)?,
                invalidation_mode: row.get(6)?,
                is_invalidated: row.get(7)?,
                invalidated_at: row.get(8)?,
                invalidation_reason: row.get(9)?,
                recorded_at: row.get(10)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Invalidate derivation edges matching a trigger mode, bounded by source.
///
/// Returns the number of edges invalidated.
pub(crate) fn invalidate_derivation_edges(
    conn: &rusqlite::Connection,
    source_kind: &str,
    source_id: &str,
    trigger_mode: &str,
    reason: &str,
) -> Result<usize, MemoryError> {
    let now = chrono::Utc::now().to_rfc3339();
    let count = conn.execute(
        "UPDATE derivation_edges
         SET is_invalidated = 1, invalidated_at = ?1, invalidation_reason = ?2
         WHERE source_kind = ?3 AND source_id = ?4
           AND invalidation_mode = ?5
           AND is_invalidated = 0",
        rusqlite::params![now, reason, source_kind, source_id, trigger_mode],
    )?;
    Ok(count)
}

/// Get all invalidated derivation edge targets for bounded recomputation.
#[allow(dead_code)]
pub(crate) fn list_invalidated_targets(
    conn: &rusqlite::Connection,
    limit: usize,
) -> Result<Vec<DerivationEdgeRow>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, source_kind, source_id, target_kind, target_id,
                derivation_type, invalidation_mode, is_invalidated,
                invalidated_at, invalidation_reason, recorded_at
         FROM derivation_edges
         WHERE is_invalidated = 1
         ORDER BY invalidated_at ASC
         LIMIT ?1",
    )?;
    let rows = stmt
        .query_map(rusqlite::params![limit as i64], |row| {
            Ok(DerivationEdgeRow {
                id: row.get(0)?,
                source_kind: row.get(1)?,
                source_id: row.get(2)?,
                target_kind: row.get(3)?,
                target_id: row.get(4)?,
                derivation_type: row.get(5)?,
                invalidation_mode: row.get(6)?,
                is_invalidated: row.get(7)?,
                invalidated_at: row.get(8)?,
                invalidation_reason: row.get(9)?,
                recorded_at: row.get(10)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Clear invalidation flag after recomputation (bounded rebuild).
#[allow(dead_code)]
pub(crate) fn clear_invalidation(
    conn: &rusqlite::Connection,
    edge_id: i64,
) -> Result<(), MemoryError> {
    conn.execute(
        "UPDATE derivation_edges
         SET is_invalidated = 0, invalidated_at = NULL, invalidation_reason = NULL
         WHERE id = ?1",
        rusqlite::params![edge_id],
    )?;
    Ok(())
}

/// List preferred claim versions for the same logical key with interval metadata.
///
/// This is used to enforce temporal integrity invariants for preferred versions
/// in the canonical projection import path.
#[allow(clippy::type_complexity)]
pub(crate) fn query_preferred_claim_intervals(
    conn: &rusqlite::Connection,
    claim_id: &str,
) -> Result<Vec<(String, Option<String>, Option<String>)>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT claim_version_id, valid_from, valid_to
           FROM claim_versions
          WHERE claim_id = ?1
            AND preferred_open = 1",
    )?;
    let rows = stmt
        .query_map(rusqlite::params![claim_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// List preferred relation versions for the same logical key with interval metadata.
///
/// Logical key is `(subject_entity_id, predicate, object_anchor, scope_key, projection_family)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn query_preferred_relation_intervals(
    conn: &rusqlite::Connection,
    subject_entity_id: &str,
    predicate: &str,
    object_anchor: &str,
    scope_namespace: &str,
    scope_domain: Option<&str>,
    scope_workspace_id: Option<&str>,
    scope_repo_id: Option<&str>,
    projection_family: &str,
) -> Result<Vec<(String, Option<String>, Option<String>)>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT relation_version_id, valid_from, valid_to
           FROM relation_versions
          WHERE subject_entity_id = ?1
            AND predicate = ?2
            AND object_anchor = ?3
            AND scope_namespace = ?4
            AND (?5 IS NULL AND scope_domain IS NULL OR scope_domain = ?5)
            AND (?6 IS NULL AND scope_workspace_id IS NULL OR scope_workspace_id = ?6)
            AND (?7 IS NULL AND scope_repo_id IS NULL OR scope_repo_id = ?7)
            AND projection_family = ?8
            AND preferred_open = 1",
    )?;
    let rows = stmt
        .query_map(
            rusqlite::params![
                subject_entity_id,
                predicate,
                object_anchor,
                scope_namespace,
                scope_domain,
                scope_workspace_id,
                scope_repo_id,
                projection_family,
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn projection_scan_limit(query: &ProjectionQuery) -> usize {
    let base = query.limit.max(1);
    base.saturating_mul(8).min(256).max(base)
}

fn query_terms(text_query: Option<&str>) -> Vec<String> {
    text_query
        .unwrap_or_default()
        .split_whitespace()
        .map(|term| term.trim().to_lowercase())
        .filter(|term| !term.is_empty())
        .collect()
}

fn text_score(terms: &[String], searchable: &str) -> usize {
    if terms.is_empty() {
        return 1;
    }
    let haystack = searchable.to_lowercase();
    terms
        .iter()
        .filter(|term| haystack.contains(term.as_str()))
        .count()
}

fn decode_scope_key(
    namespace: String,
    domain: Option<String>,
    workspace_id: Option<String>,
    repo_id: Option<String>,
) -> ScopeKey {
    ScopeKey {
        namespace,
        domain,
        workspace_id,
        repo_id,
    }
}

fn decode_json_value(
    table: &'static str,
    row_id: &str,
    column: &'static str,
    raw: &str,
) -> Result<serde_json::Value, MemoryError> {
    serde_json::from_str(raw).map_err(|err| MemoryError::CorruptData {
        table,
        row_id: row_id.to_string(),
        detail: format!("invalid {column} JSON: {err}"),
    })
}

fn decode_optional_json_value(
    table: &'static str,
    row_id: &str,
    column: &'static str,
    raw: Option<String>,
) -> Result<Option<serde_json::Value>, MemoryError> {
    raw.map(|value| decode_json_value(table, row_id, column, &value))
        .transpose()
}

fn decode_string_vec(
    table: &'static str,
    row_id: &str,
    column: &'static str,
    raw: &str,
) -> Result<Vec<String>, MemoryError> {
    serde_json::from_str(raw).map_err(|err| MemoryError::CorruptData {
        table,
        row_id: row_id.to_string(),
        detail: format!("invalid {column} JSON: {err}"),
    })
}

/// Query imported claim versions with full scope and temporal filters.
pub(crate) fn query_claim_versions(
    conn: &rusqlite::Connection,
    query: &ProjectionQuery,
) -> Result<Vec<ProjectionClaimVersion>, MemoryError> {
    if query.limit == 0 {
        return Ok(Vec::new());
    }

    let scan_limit = projection_scan_limit(query);
    let terms = query_terms(query.text_query.as_deref());
    let mut stmt = conn.prepare(
        "SELECT cv.claim_version_id, cv.claim_id, cv.claim_state, cv.projection_family,
                cv.subject_entity_id, cv.predicate, cv.object_anchor,
                cv.scope_namespace, cv.scope_domain, cv.scope_workspace_id, cv.scope_repo_id,
                cv.valid_from, cv.valid_to, cv.recorded_at, cv.preferred_open,
                cv.source_envelope_id, cv.source_authority, cv.trace_id,
                cv.freshness, cv.contradiction_status, cv.supersedes_claim_version_id,
                cv.content, cv.confidence, cv.metadata,
                pil.source_exported_at, pil.transformed_at
         FROM claim_versions cv
         LEFT JOIN projection_import_log pil
           ON pil.source_envelope_id = cv.source_envelope_id
          AND pil.imported_at = cv.recorded_at
         WHERE cv.scope_namespace = ?1
           AND ((?2 IS NULL AND cv.scope_domain IS NULL) OR cv.scope_domain = ?2)
           AND ((?3 IS NULL AND cv.scope_workspace_id IS NULL) OR cv.scope_workspace_id = ?3)
           AND ((?4 IS NULL AND cv.scope_repo_id IS NULL) OR cv.scope_repo_id = ?4)
           AND (?5 IS NULL OR cv.subject_entity_id = ?5)
           AND (?6 IS NULL OR cv.claim_state = ?6)
           AND (?7 IS NULL OR cv.claim_id = ?7)
           AND (?8 IS NULL OR cv.claim_version_id = ?8)
           AND (?9 IS NULL OR cv.recorded_at <= ?9)
           AND (?10 IS NULL OR ((cv.valid_from IS NULL OR cv.valid_from <= ?10)
                            AND (cv.valid_to IS NULL OR cv.valid_to > ?10)))
         ORDER BY cv.preferred_open DESC, cv.recorded_at DESC
         LIMIT ?11",
    )?;
    let mut rows = stmt.query(rusqlite::params![
        query.scope.namespace.as_str(),
        query.scope.domain.as_deref(),
        query.scope.workspace_id.as_deref(),
        query.scope.repo_id.as_deref(),
        query.subject_entity_id.as_ref().map(|id| id.as_str()),
        query.claim_state.as_deref(),
        query.claim_id.as_ref().map(|id| id.as_str()),
        query.claim_version_id.as_ref().map(|id| id.as_str()),
        query.recorded_at_or_before.as_deref(),
        query.valid_at.as_deref(),
        scan_limit as i64,
    ])?;

    let mut results = Vec::new();
    while let Some(row) = rows.next()? {
        let claim_version_id = row.get::<_, String>(0)?;
        let claim_id = row.get::<_, String>(1)?;
        let subject_entity_id = row.get::<_, String>(4)?;
        let object_anchor_raw = row.get::<_, String>(6)?;
        let searchable = format!(
            "{} {} {} {} {}",
            row.get::<_, String>(21)?,
            row.get::<_, String>(5)?,
            subject_entity_id,
            object_anchor_raw,
            claim_id
        );
        let score = text_score(&terms, &searchable);
        if !terms.is_empty() && score == 0 {
            continue;
        }

        results.push((
            score,
            ProjectionClaimVersion {
                claim_version_id: ClaimVersionId::new(claim_version_id.clone()),
                claim_id: ClaimId::new(claim_id.clone()),
                claim_state: row.get(2)?,
                projection_family: row.get(3)?,
                subject_entity_id: EntityId::new(subject_entity_id.clone()),
                predicate: row.get(5)?,
                object_anchor: decode_json_value(
                    "claim_versions",
                    &claim_version_id,
                    "object_anchor",
                    &object_anchor_raw,
                )?,
                scope_key: decode_scope_key(row.get(7)?, row.get(8)?, row.get(9)?, row.get(10)?),
                valid_from: row.get(11)?,
                valid_to: row.get(12)?,
                recorded_at: row.get(13)?,
                preferred_open: row.get::<_, i32>(14)? != 0,
                source_envelope_id: EnvelopeId::new(row.get::<_, String>(15)?),
                source_authority: row.get(16)?,
                trace_id: row.get(17)?,
                freshness: row.get(18)?,
                contradiction_status: row.get(19)?,
                supersedes_claim_version_id: row
                    .get::<_, Option<String>>(20)?
                    .map(ClaimVersionId::new),
                content: row.get(21)?,
                confidence: row.get(22)?,
                metadata: decode_optional_json_value(
                    "claim_versions",
                    &claim_version_id,
                    "metadata",
                    row.get(23)?,
                )?,
                source_exported_at: row.get(24)?,
                transformed_at: row.get(25)?,
            },
        ));
    }

    if !terms.is_empty() {
        results.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.preferred_open.cmp(&a.1.preferred_open))
                .then_with(|| b.1.recorded_at.cmp(&a.1.recorded_at))
        });
    }

    Ok(results
        .into_iter()
        .map(|(_, item)| item)
        .take(query.limit)
        .collect())
}

/// Query imported relation versions with full scope and temporal filters.
pub(crate) fn query_relation_versions(
    conn: &rusqlite::Connection,
    query: &ProjectionQuery,
) -> Result<Vec<ProjectionRelationVersion>, MemoryError> {
    if query.limit == 0 {
        return Ok(Vec::new());
    }

    let scan_limit = projection_scan_limit(query);
    let terms = query_terms(query.text_query.as_deref());
    let mut stmt = conn.prepare(
        "SELECT rv.relation_version_id, rv.subject_entity_id, rv.predicate, rv.object_anchor,
                rv.scope_namespace, rv.scope_domain, rv.scope_workspace_id, rv.scope_repo_id,
                rv.claim_id, rv.source_episode_id, rv.valid_from, rv.valid_to,
                rv.recorded_at, rv.preferred_open, rv.supersedes_relation_version_id,
                rv.contradiction_status, rv.source_confidence, rv.projection_family,
                rv.source_envelope_id, rv.source_authority, rv.trace_id,
                rv.freshness, rv.metadata,
                pil.source_exported_at, pil.transformed_at
         FROM relation_versions rv
         LEFT JOIN projection_import_log pil
           ON pil.source_envelope_id = rv.source_envelope_id
          AND pil.imported_at = rv.recorded_at
         WHERE rv.scope_namespace = ?1
           AND ((?2 IS NULL AND rv.scope_domain IS NULL) OR rv.scope_domain = ?2)
           AND ((?3 IS NULL AND rv.scope_workspace_id IS NULL) OR rv.scope_workspace_id = ?3)
           AND ((?4 IS NULL AND rv.scope_repo_id IS NULL) OR rv.scope_repo_id = ?4)
           AND (?5 IS NULL OR rv.subject_entity_id = ?5)
           AND (?6 IS NULL OR rv.recorded_at <= ?6)
           AND (?7 IS NULL OR ((rv.valid_from IS NULL OR rv.valid_from <= ?7)
                           AND (rv.valid_to IS NULL OR rv.valid_to > ?7)))
         ORDER BY rv.preferred_open DESC, rv.recorded_at DESC
         LIMIT ?8",
    )?;
    let mut rows = stmt.query(rusqlite::params![
        query.scope.namespace.as_str(),
        query.scope.domain.as_deref(),
        query.scope.workspace_id.as_deref(),
        query.scope.repo_id.as_deref(),
        query.subject_entity_id.as_ref().map(|id| id.as_str()),
        query.recorded_at_or_before.as_deref(),
        query.valid_at.as_deref(),
        scan_limit as i64,
    ])?;

    let mut results = Vec::new();
    while let Some(row) = rows.next()? {
        let relation_version_id = row.get::<_, String>(0)?;
        let subject_entity_id = row.get::<_, String>(1)?;
        let object_anchor_raw = row.get::<_, String>(3)?;
        let claim_id = row.get::<_, Option<String>>(8)?;
        let source_episode_id = row.get::<_, Option<String>>(9)?;
        let searchable = format!(
            "{} {} {} {} {}",
            row.get::<_, String>(2)?,
            subject_entity_id,
            object_anchor_raw,
            claim_id.clone().unwrap_or_default(),
            source_episode_id.clone().unwrap_or_default()
        );
        let score = text_score(&terms, &searchable);
        if !terms.is_empty() && score == 0 {
            continue;
        }

        results.push((
            score,
            ProjectionRelationVersion {
                relation_version_id: RelationVersionId::new(relation_version_id.clone()),
                subject_entity_id: EntityId::new(subject_entity_id),
                predicate: row.get(2)?,
                object_anchor: decode_json_value(
                    "relation_versions",
                    &relation_version_id,
                    "object_anchor",
                    &object_anchor_raw,
                )?,
                scope_key: decode_scope_key(row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?),
                claim_id: claim_id.map(ClaimId::new),
                source_episode_id: source_episode_id.map(EpisodeId::new),
                valid_from: row.get(10)?,
                valid_to: row.get(11)?,
                recorded_at: row.get(12)?,
                preferred_open: row.get::<_, i32>(13)? != 0,
                supersedes_relation_version_id: row
                    .get::<_, Option<String>>(14)?
                    .map(RelationVersionId::new),
                contradiction_status: row.get(15)?,
                source_confidence: row.get(16)?,
                projection_family: row.get(17)?,
                source_envelope_id: EnvelopeId::new(row.get::<_, String>(18)?),
                source_authority: row.get(19)?,
                trace_id: row.get(20)?,
                freshness: row.get(21)?,
                metadata: decode_optional_json_value(
                    "relation_versions",
                    &relation_version_id,
                    "metadata",
                    row.get(22)?,
                )?,
                source_exported_at: row.get(23)?,
                transformed_at: row.get(24)?,
            },
        ));
    }

    if !terms.is_empty() {
        results.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.preferred_open.cmp(&a.1.preferred_open))
                .then_with(|| b.1.recorded_at.cmp(&a.1.recorded_at))
        });
    }

    Ok(results
        .into_iter()
        .map(|(_, item)| item)
        .take(query.limit)
        .collect())
}

/// Query imported episode rows with full scope and recorded-time filters.
pub(crate) fn query_episode_rows(
    conn: &rusqlite::Connection,
    query: &ProjectionQuery,
) -> Result<Vec<ProjectionEpisode>, MemoryError> {
    if query.limit == 0 {
        return Ok(Vec::new());
    }

    let scan_limit = projection_scan_limit(query);
    let terms = query_terms(query.text_query.as_deref());
    let mut stmt = conn.prepare(
        "SELECT el.episode_id, el.document_id, el.cause_ids, el.effect_type, el.outcome,
                el.confidence, el.experiment_id, el.source_envelope_id, el.source_authority,
                el.trace_id, el.recorded_at, el.metadata,
                pil.scope_namespace, pil.scope_domain, pil.scope_workspace_id, pil.scope_repo_id,
                pil.source_exported_at, pil.transformed_at
         FROM episode_links el
         JOIN projection_import_log pil
           ON pil.source_envelope_id = el.source_envelope_id
          AND pil.imported_at = el.recorded_at
         WHERE pil.scope_namespace = ?1
           AND ((?2 IS NULL AND pil.scope_domain IS NULL) OR pil.scope_domain = ?2)
           AND ((?3 IS NULL AND pil.scope_workspace_id IS NULL) OR pil.scope_workspace_id = ?3)
           AND ((?4 IS NULL AND pil.scope_repo_id IS NULL) OR pil.scope_repo_id = ?4)
           AND (?5 IS NULL OR el.recorded_at <= ?5)
         ORDER BY el.recorded_at DESC
         LIMIT ?6",
    )?;
    let mut rows = stmt.query(rusqlite::params![
        query.scope.namespace.as_str(),
        query.scope.domain.as_deref(),
        query.scope.workspace_id.as_deref(),
        query.scope.repo_id.as_deref(),
        query.recorded_at_or_before.as_deref(),
        scan_limit as i64,
    ])?;

    let mut results = Vec::new();
    while let Some(row) = rows.next()? {
        let episode_id = row.get::<_, String>(0)?;
        let cause_ids_raw = row.get::<_, String>(2)?;
        let searchable = format!(
            "{} {} {} {}",
            row.get::<_, String>(1)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            cause_ids_raw
        );
        let score = text_score(&terms, &searchable);
        if !terms.is_empty() && score == 0 {
            continue;
        }

        results.push((
            score,
            ProjectionEpisode {
                episode_id: EpisodeId::new(episode_id.clone()),
                document_id: row.get(1)?,
                cause_ids: decode_string_vec(
                    "episode_links",
                    &episode_id,
                    "cause_ids",
                    &cause_ids_raw,
                )?,
                effect_type: row.get(3)?,
                outcome: row.get(4)?,
                confidence: row.get(5)?,
                experiment_id: row.get(6)?,
                scope_key: decode_scope_key(row.get(12)?, row.get(13)?, row.get(14)?, row.get(15)?),
                source_envelope_id: EnvelopeId::new(row.get::<_, String>(7)?),
                source_authority: row.get(8)?,
                trace_id: row.get(9)?,
                recorded_at: row.get(10)?,
                metadata: decode_optional_json_value(
                    "episode_links",
                    &episode_id,
                    "metadata",
                    row.get(11)?,
                )?,
                source_exported_at: row.get(16)?,
                transformed_at: row.get(17)?,
            },
        ));
    }

    if !terms.is_empty() {
        results.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.recorded_at.cmp(&a.1.recorded_at))
        });
    }

    Ok(results
        .into_iter()
        .map(|(_, item)| item)
        .take(query.limit)
        .collect())
}

/// Query imported alias rows with full scope and recorded-time filters.
pub(crate) fn query_entity_aliases(
    conn: &rusqlite::Connection,
    query: &ProjectionQuery,
) -> Result<Vec<ProjectionEntityAlias>, MemoryError> {
    if query.limit == 0 {
        return Ok(Vec::new());
    }

    let scan_limit = projection_scan_limit(query);
    let terms = query_terms(query.text_query.as_deref());
    let mut stmt = conn.prepare(
        "SELECT ea.canonical_entity_id, ea.alias_text, ea.alias_source, ea.match_evidence,
                ea.confidence, ea.merge_decision, ea.scope_namespace, ea.scope_domain,
                ea.scope_workspace_id, ea.scope_repo_id, ea.review_state,
                ea.is_human_confirmed, ea.is_human_confirmed_final,
                ea.superseded_by_entity_id, ea.split_from_entity_id,
                ea.source_envelope_id, ea.recorded_at,
                pil.source_exported_at, pil.transformed_at
         FROM entity_aliases ea
         LEFT JOIN projection_import_log pil
           ON pil.source_envelope_id = ea.source_envelope_id
          AND pil.imported_at = ea.recorded_at
         WHERE ea.scope_namespace = ?1
           AND ((?2 IS NULL AND ea.scope_domain IS NULL) OR ea.scope_domain = ?2)
           AND ((?3 IS NULL AND ea.scope_workspace_id IS NULL) OR ea.scope_workspace_id = ?3)
           AND ((?4 IS NULL AND ea.scope_repo_id IS NULL) OR ea.scope_repo_id = ?4)
           AND (?5 IS NULL OR ea.canonical_entity_id = ?5)
           AND (?6 IS NULL OR ea.recorded_at <= ?6)
         ORDER BY ea.is_human_confirmed_final DESC, ea.recorded_at DESC
         LIMIT ?7",
    )?;
    let mut rows = stmt.query(rusqlite::params![
        query.scope.namespace.as_str(),
        query.scope.domain.as_deref(),
        query.scope.workspace_id.as_deref(),
        query.scope.repo_id.as_deref(),
        query.canonical_entity_id.as_ref().map(|id| id.as_str()),
        query.recorded_at_or_before.as_deref(),
        scan_limit as i64,
    ])?;

    let mut results = Vec::new();
    while let Some(row) = rows.next()? {
        let canonical_entity_id = row.get::<_, String>(0)?;
        let searchable = format!(
            "{} {} {}",
            row.get::<_, String>(1)?,
            canonical_entity_id,
            row.get::<_, String>(2)?
        );
        let score = text_score(&terms, &searchable);
        if !terms.is_empty() && score == 0 {
            continue;
        }

        results.push((
            score,
            ProjectionEntityAlias {
                canonical_entity_id: EntityId::new(canonical_entity_id.clone()),
                alias_text: row.get(1)?,
                alias_source: row.get(2)?,
                match_evidence: decode_optional_json_value(
                    "entity_aliases",
                    &canonical_entity_id,
                    "match_evidence",
                    row.get(3)?,
                )?,
                confidence: row.get(4)?,
                merge_decision: row.get(5)?,
                scope_key: decode_scope_key(row.get(6)?, row.get(7)?, row.get(8)?, row.get(9)?),
                review_state: row.get(10)?,
                is_human_confirmed: row.get::<_, i32>(11)? != 0,
                is_human_confirmed_final: row.get::<_, i32>(12)? != 0,
                superseded_by_entity_id: row.get::<_, Option<String>>(13)?.map(EntityId::new),
                split_from_entity_id: row.get::<_, Option<String>>(14)?.map(EntityId::new),
                source_envelope_id: EnvelopeId::new(row.get::<_, String>(15)?),
                recorded_at: row.get(16)?,
                source_exported_at: row.get(17)?,
                transformed_at: row.get(18)?,
            },
        ));
    }

    if !terms.is_empty() {
        results.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| {
                    b.1.is_human_confirmed_final
                        .cmp(&a.1.is_human_confirmed_final)
                })
                .then_with(|| b.1.recorded_at.cmp(&a.1.recorded_at))
        });
    }

    Ok(results
        .into_iter()
        .map(|(_, item)| item)
        .take(query.limit)
        .collect())
}

/// Query imported evidence-reference rows with full scope and recorded-time filters.
pub(crate) fn query_evidence_refs(
    conn: &rusqlite::Connection,
    query: &ProjectionQuery,
) -> Result<Vec<ProjectionEvidenceRef>, MemoryError> {
    if query.limit == 0 {
        return Ok(Vec::new());
    }

    let scan_limit = projection_scan_limit(query);
    let terms = query_terms(query.text_query.as_deref());
    let mut stmt = conn.prepare(
        "SELECT er.claim_id, er.claim_version_id, er.fetch_handle, er.source_authority,
                er.source_envelope_id, er.recorded_at, er.metadata,
                pil.scope_namespace, pil.scope_domain, pil.scope_workspace_id, pil.scope_repo_id,
                pil.source_exported_at, pil.transformed_at
         FROM evidence_refs er
         JOIN projection_import_log pil
           ON pil.source_envelope_id = er.source_envelope_id
          AND pil.imported_at = er.recorded_at
         WHERE pil.scope_namespace = ?1
           AND ((?2 IS NULL AND pil.scope_domain IS NULL) OR pil.scope_domain = ?2)
           AND ((?3 IS NULL AND pil.scope_workspace_id IS NULL) OR pil.scope_workspace_id = ?3)
           AND ((?4 IS NULL AND pil.scope_repo_id IS NULL) OR pil.scope_repo_id = ?4)
           AND (?5 IS NULL OR er.claim_id = ?5)
           AND (?6 IS NULL OR er.claim_version_id = ?6)
           AND (?7 IS NULL OR er.recorded_at <= ?7)
         ORDER BY er.recorded_at DESC
         LIMIT ?8",
    )?;
    let mut rows = stmt.query(rusqlite::params![
        query.scope.namespace.as_str(),
        query.scope.domain.as_deref(),
        query.scope.workspace_id.as_deref(),
        query.scope.repo_id.as_deref(),
        query.claim_id.as_ref().map(|id| id.as_str()),
        query.claim_version_id.as_ref().map(|id| id.as_str()),
        query.recorded_at_or_before.as_deref(),
        scan_limit as i64,
    ])?;

    let mut results = Vec::new();
    while let Some(row) = rows.next()? {
        let claim_id = row.get::<_, String>(0)?;
        let claim_version_id = row.get::<_, Option<String>>(1)?;
        let fetch_handle = row.get::<_, String>(2)?;
        let searchable = format!(
            "{} {} {}",
            claim_id,
            claim_version_id.clone().unwrap_or_default(),
            fetch_handle
        );
        let score = text_score(&terms, &searchable);
        if !terms.is_empty() && score == 0 {
            continue;
        }

        results.push((
            score,
            ProjectionEvidenceRef {
                claim_id: ClaimId::new(claim_id),
                claim_version_id: claim_version_id.map(ClaimVersionId::new),
                fetch_handle,
                source_authority: row.get(3)?,
                source_envelope_id: EnvelopeId::new(row.get::<_, String>(4)?),
                scope_key: decode_scope_key(row.get(7)?, row.get(8)?, row.get(9)?, row.get(10)?),
                recorded_at: row.get(5)?,
                metadata: decode_optional_json_value(
                    "evidence_refs",
                    &row.get::<_, String>(4)?,
                    "metadata",
                    row.get(6)?,
                )?,
                source_exported_at: row.get(11)?,
                transformed_at: row.get(12)?,
            },
        ));
    }

    if !terms.is_empty() {
        results.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.recorded_at.cmp(&a.1.recorded_at))
        });
    }

    Ok(results
        .into_iter()
        .map(|(_, item)| item)
        .take(query.limit)
        .collect())
}

/// Query projection import log entries.
pub(crate) fn query_projection_import_log(
    conn: &rusqlite::Connection,
    scope_namespace: Option<&str>,
    limit: usize,
) -> Result<Vec<ProjectionImportLogRow>, MemoryError> {
    let sql = if let Some(ns) = scope_namespace {
        let mut stmt = conn.prepare(
            "SELECT batch_id, source_envelope_id, schema_version, export_schema_version,
                    content_digest, source_authority, scope_namespace, scope_domain,
                    scope_workspace_id, scope_repo_id, trace_id, record_count,
                    claim_count, relation_count, episode_count, alias_count, evidence_count,
                    status, source_exported_at, transformed_at, imported_at,
                    source_run_id, comparability_snapshot_version, direct_write, failure_reason,
                    evidence_bundle_id, evidence_bundle_json, episode_bundle_id, episode_bundle_json,
                    execution_context_json, kernel_payload_json
             FROM projection_import_log
             WHERE scope_namespace = ?1
             ORDER BY imported_at DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(rusqlite::params![ns, limit as i64], map_import_log_row)?
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(rows);
    } else {
        "SELECT batch_id, source_envelope_id, schema_version, export_schema_version,
                content_digest, source_authority, scope_namespace, scope_domain,
                scope_workspace_id, scope_repo_id, trace_id, record_count,
                claim_count, relation_count, episode_count, alias_count, evidence_count,
                status, source_exported_at, transformed_at, imported_at,
                source_run_id, comparability_snapshot_version, direct_write, failure_reason,
                evidence_bundle_id, evidence_bundle_json, episode_bundle_id, episode_bundle_json,
                execution_context_json, kernel_payload_json
         FROM projection_import_log
         ORDER BY imported_at DESC LIMIT ?1"
    };
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt
        .query_map(rusqlite::params![limit as i64], map_import_log_row)?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub(crate) fn latest_rebuildable_kernel_projection_import(
    conn: &rusqlite::Connection,
    scope: &ScopeKey,
) -> Result<Option<ProjectionImportLogRow>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT batch_id, source_envelope_id, schema_version, export_schema_version,
                content_digest, source_authority, scope_namespace, scope_domain,
                scope_workspace_id, scope_repo_id, trace_id, record_count,
                claim_count, relation_count, episode_count, alias_count, evidence_count,
                status, source_exported_at, transformed_at, imported_at,
                source_run_id, comparability_snapshot_version, direct_write, failure_reason,
                evidence_bundle_id, evidence_bundle_json, episode_bundle_id, episode_bundle_json,
                execution_context_json, kernel_payload_json
         FROM projection_import_log
         WHERE scope_namespace = ?1
           AND ((?2 IS NULL AND scope_domain IS NULL) OR scope_domain = ?2)
           AND ((?3 IS NULL AND scope_workspace_id IS NULL) OR scope_workspace_id = ?3)
           AND ((?4 IS NULL AND scope_repo_id IS NULL) OR scope_repo_id = ?4)
           AND status = 'complete'
           AND kernel_payload_json IS NOT NULL
         ORDER BY imported_at DESC
         LIMIT 1",
    )?;
    let mut rows = stmt.query(rusqlite::params![
        scope.namespace.as_str(),
        scope.domain.as_deref(),
        scope.workspace_id.as_deref(),
        scope.repo_id.as_deref(),
    ])?;
    match rows.next()? {
        Some(row) => Ok(Some(map_import_log_row(row)?)),
        None => Ok(None),
    }
}

pub(crate) fn query_projection_import_failures(
    conn: &rusqlite::Connection,
    scope_namespace: Option<&str>,
    limit: usize,
) -> Result<Vec<ProjectionImportFailureRow>, MemoryError> {
    let sql = if let Some(ns) = scope_namespace {
        let mut stmt = conn.prepare(
            "SELECT failure_id, source_envelope_id, schema_version, export_schema_version,
                    content_digest, source_authority, scope_namespace, scope_domain,
                    scope_workspace_id, scope_repo_id, trace_id, record_count,
                    error_kind, error_message, source_exported_at, transformed_at, failed_at,
                    source_run_id, comparability_snapshot_version, direct_write,
                    evidence_bundle_id, evidence_bundle_json, episode_bundle_id, episode_bundle_json,
                    execution_context_json, kernel_payload_json
             FROM projection_import_failures
             WHERE scope_namespace = ?1
             ORDER BY failed_at DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(rusqlite::params![ns, limit as i64], map_import_failure_row)?
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(rows);
    } else {
        "SELECT failure_id, source_envelope_id, schema_version, export_schema_version,
                content_digest, source_authority, scope_namespace, scope_domain,
                scope_workspace_id, scope_repo_id, trace_id, record_count,
                error_kind, error_message, source_exported_at, transformed_at, failed_at,
                source_run_id, comparability_snapshot_version, direct_write,
                evidence_bundle_id, evidence_bundle_json, episode_bundle_id, episode_bundle_json,
                execution_context_json, kernel_payload_json
         FROM projection_import_failures
         ORDER BY failed_at DESC LIMIT ?1"
    };
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt
        .query_map(rusqlite::params![limit as i64], map_import_failure_row)?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn map_import_log_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProjectionImportLogRow> {
    Ok(ProjectionImportLogRow {
        batch_id: row.get(0)?,
        source_envelope_id: row.get(1)?,
        schema_version: row.get(2)?,
        export_schema_version: row.get(3)?,
        content_digest: row.get(4)?,
        source_authority: row.get(5)?,
        scope_namespace: row.get(6)?,
        scope_domain: row.get(7)?,
        scope_workspace_id: row.get(8)?,
        scope_repo_id: row.get(9)?,
        trace_id: row.get(10)?,
        record_count: row.get::<_, i64>(11)? as usize,
        claim_count: row.get::<_, i64>(12)? as usize,
        relation_count: row.get::<_, i64>(13)? as usize,
        episode_count: row.get::<_, i64>(14)? as usize,
        alias_count: row.get::<_, i64>(15)? as usize,
        evidence_count: row.get::<_, i64>(16)? as usize,
        status: row.get(17)?,
        source_exported_at: row.get(18)?,
        transformed_at: row.get(19)?,
        imported_at: row.get(20)?,
        source_run_id: row.get(21)?,
        comparability_snapshot_version: row.get(22)?,
        direct_write: row.get::<_, i64>(23)? != 0,
        failure_reason: row.get(24)?,
        evidence_bundle_id: row.get(25)?,
        evidence_bundle_json: row.get(26)?,
        episode_bundle_id: row.get(27)?,
        episode_bundle_json: row.get(28)?,
        execution_context_json: row.get(29)?,
        kernel_payload_json: row.get(30)?,
    })
}

fn map_import_failure_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProjectionImportFailureRow> {
    Ok(ProjectionImportFailureRow {
        failure_id: row.get(0)?,
        source_envelope_id: row.get(1)?,
        schema_version: row.get(2)?,
        export_schema_version: row.get(3)?,
        content_digest: row.get(4)?,
        source_authority: row.get(5)?,
        scope_namespace: row.get(6)?,
        scope_domain: row.get(7)?,
        scope_workspace_id: row.get(8)?,
        scope_repo_id: row.get(9)?,
        trace_id: row.get(10)?,
        record_count: row.get::<_, i64>(11)? as usize,
        error_kind: row.get(12)?,
        error_message: row.get(13)?,
        source_exported_at: row.get(14)?,
        transformed_at: row.get(15)?,
        failed_at: row.get(16)?,
        source_run_id: row.get(17)?,
        comparability_snapshot_version: row.get(18)?,
        direct_write: row.get::<_, i64>(19)? != 0,
        evidence_bundle_id: row.get(20)?,
        evidence_bundle_json: row.get(21)?,
        episode_bundle_id: row.get(22)?,
        episode_bundle_json: row.get(23)?,
        execution_context_json: row.get(24)?,
        kernel_payload_json: row.get(25)?,
    })
}

/// Update import log status to "aborted" for a failed batch.
#[allow(dead_code)]
pub(crate) fn abort_projection_import(
    conn: &rusqlite::Connection,
    batch_id: &str,
    reason: &str,
) -> Result<(), MemoryError> {
    conn.execute(
        "UPDATE projection_import_log SET status = ?1 WHERE batch_id = ?2",
        rusqlite::params![format!("aborted: {reason}"), batch_id],
    )?;
    Ok(())
}

// ─── Row structs for DB operations ────────────────────────────

/// Flat row struct for inserting claim versions.
pub(crate) struct ClaimVersionRow {
    pub claim_version_id: String,
    pub claim_id: String,
    pub claim_state: String,
    pub projection_family: String,
    pub subject_entity_id: String,
    pub predicate: String,
    pub object_anchor: String,
    pub scope_namespace: String,
    pub scope_domain: Option<String>,
    pub scope_workspace_id: Option<String>,
    pub scope_repo_id: Option<String>,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub recorded_at: String,
    pub preferred_open: bool,
    pub source_envelope_id: String,
    pub source_authority: String,
    pub trace_id: Option<String>,
    pub freshness: String,
    pub contradiction_status: String,
    pub supersedes_claim_version_id: Option<String>,
    pub content: String,
    pub confidence: f32,
    pub content_digest: Option<String>,
    pub metadata: Option<String>,
}

/// Flat row struct for inserting relation versions.
pub(crate) struct RelationVersionRow {
    pub relation_version_id: String,
    pub subject_entity_id: String,
    pub predicate: String,
    pub object_anchor: String,
    pub scope_namespace: String,
    pub scope_domain: Option<String>,
    pub scope_workspace_id: Option<String>,
    pub scope_repo_id: Option<String>,
    pub claim_id: Option<String>,
    pub source_episode_id: Option<String>,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub recorded_at: String,
    pub preferred_open: bool,
    pub supersedes_relation_version_id: Option<String>,
    pub contradiction_status: String,
    pub source_confidence: f32,
    pub projection_family: String,
    pub source_envelope_id: String,
    pub source_authority: String,
    pub trace_id: Option<String>,
    pub freshness: String,
    pub metadata: Option<String>,
}

/// Flat row struct for inserting entity aliases.
pub(crate) struct EntityAliasRow {
    pub canonical_entity_id: String,
    pub alias_text: String,
    pub alias_source: String,
    pub match_evidence: Option<String>,
    pub confidence: f32,
    pub merge_decision: String,
    pub scope_namespace: String,
    pub scope_domain: Option<String>,
    pub scope_workspace_id: Option<String>,
    pub scope_repo_id: Option<String>,
    pub review_state: String,
    pub is_human_confirmed: bool,
    pub is_human_confirmed_final: bool,
    pub superseded_by_entity_id: Option<String>,
    pub split_from_entity_id: Option<String>,
    pub source_envelope_id: String,
    pub recorded_at: String,
}

/// Flat row struct for inserting evidence refs.
pub(crate) struct EvidenceRefRow {
    pub claim_id: String,
    pub claim_version_id: Option<String>,
    pub fetch_handle: String,
    pub source_authority: String,
    pub source_envelope_id: String,
    pub recorded_at: String,
    pub metadata: Option<String>,
}

/// Flat row struct for episode links.
pub(crate) struct EpisodeLinkRow {
    pub episode_id: String,
    pub document_id: String,
    pub cause_ids: String,
    pub effect_type: String,
    pub outcome: String,
    pub confidence: f32,
    pub experiment_id: Option<String>,
    pub source_envelope_id: String,
    pub source_authority: String,
    pub trace_id: Option<String>,
    pub recorded_at: String,
    pub metadata: Option<String>,
}

/// Row representation of a derivation edge.
#[allow(dead_code)]
pub(crate) struct DerivationEdgeRow {
    pub id: i64,
    pub source_kind: String,
    pub source_id: String,
    pub target_kind: String,
    pub target_id: String,
    pub derivation_type: String,
    pub invalidation_mode: String,
    pub is_invalidated: bool,
    pub invalidated_at: Option<String>,
    pub invalidation_reason: Option<String>,
    pub recorded_at: String,
}

/// Flat row struct for projection import log.
#[derive(Clone)]
pub(crate) struct ProjectionImportLogRow {
    pub batch_id: String,
    pub source_envelope_id: String,
    pub schema_version: String,
    pub export_schema_version: Option<String>,
    pub content_digest: String,
    pub source_authority: String,
    pub scope_namespace: String,
    pub scope_domain: Option<String>,
    pub scope_workspace_id: Option<String>,
    pub scope_repo_id: Option<String>,
    pub trace_id: Option<String>,
    pub record_count: usize,
    pub claim_count: usize,
    pub relation_count: usize,
    pub episode_count: usize,
    pub alias_count: usize,
    pub evidence_count: usize,
    pub status: String,
    pub source_exported_at: Option<String>,
    pub transformed_at: Option<String>,
    pub imported_at: String,
    pub source_run_id: Option<String>,
    pub comparability_snapshot_version: Option<String>,
    pub direct_write: bool,
    pub failure_reason: Option<String>,
    pub evidence_bundle_id: Option<String>,
    pub evidence_bundle_json: Option<String>,
    pub episode_bundle_id: Option<String>,
    pub episode_bundle_json: Option<String>,
    pub execution_context_json: Option<String>,
    pub kernel_payload_json: Option<String>,
}

/// Durable failed projection import receipt.
pub(crate) struct ProjectionImportFailureRow {
    pub failure_id: String,
    pub source_envelope_id: String,
    pub schema_version: String,
    pub export_schema_version: Option<String>,
    pub content_digest: String,
    pub source_authority: String,
    pub scope_namespace: String,
    pub scope_domain: Option<String>,
    pub scope_workspace_id: Option<String>,
    pub scope_repo_id: Option<String>,
    pub trace_id: Option<String>,
    pub record_count: usize,
    pub error_kind: String,
    pub error_message: String,
    pub source_exported_at: Option<String>,
    pub transformed_at: Option<String>,
    pub failed_at: String,
    pub source_run_id: Option<String>,
    pub comparability_snapshot_version: Option<String>,
    pub direct_write: bool,
    pub evidence_bundle_id: Option<String>,
    pub evidence_bundle_json: Option<String>,
    pub episode_bundle_id: Option<String>,
    pub episode_bundle_json: Option<String>,
    pub execution_context_json: Option<String>,
    pub kernel_payload_json: Option<String>,
}
