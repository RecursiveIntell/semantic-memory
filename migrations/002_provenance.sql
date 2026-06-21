-- 002_provenance.sql — Phase 2 Semiring Provenance migration.
--
-- Idempotent. Mirrors MIGRATION_V25 embedded in src/db.rs.
-- SQLite is the only truth store; this table is append-only truth-bearing
-- provenance state keyed by (item_type, item_id) and referencing episodes.
--
-- Hard invariants (AGENTS.md §0):
--   * No shadow truth — provenance lives in SQLite, not a sidecar.
--   * Append-plus-supersession — never UPDATE a provenance row; INSERT a new
--     row and let get_provenance() return the latest by recorded_at.
--   * Receipts for every operation — callers emit ProvenanceReceiptV1 for each
--     set/combine call.
--   * Feature-gated — the Rust API is behind `#[cfg(feature = "provenance")]`.

CREATE TABLE IF NOT EXISTS provenance (
    id                 TEXT PRIMARY KEY,
    item_type          TEXT NOT NULL,
    item_id            TEXT NOT NULL,
    semiring_type      TEXT NOT NULL,
    semiring_value     TEXT NOT NULL,
    support_chain_json TEXT NOT NULL DEFAULT '[]',
    recorded_at        TEXT NOT NULL DEFAULT (datetime('now')),
    episode_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_provenance_item
    ON provenance(item_type, item_id);

CREATE INDEX IF NOT EXISTS idx_provenance_episode
    ON provenance(episode_id);