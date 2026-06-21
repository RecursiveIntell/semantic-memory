-- 003_temporal_weight.sql — Phase 3 Temporal Field Provenance migration.
--
-- Idempotent. Mirrors MIGRATION_V26 embedded in src/db.rs.
-- temporal_weight is a COMPUTED SCORE (not truth): it is the only column
-- in this schema that callers are allowed to UPDATE directly. All other
-- truth-bearing columns remain append-plus-supersession.
--
-- Hard invariants (AGENTS.md §0):
--   * Computed, not truth — temporal_weight reflects a decay/support/contradiction
--     score and may be freely rewritten by recompute_temporal_weights().
--   * Receipts for every recomputation — callers emit RecomputationReceipt.
--   * Feature-gated — the Rust API is behind `#[cfg(feature = "temporal")]`.
--   * Opt-in search boost — apply_temporal_boost() is never called by the
--     default search() path; it is an explicit caller method.

ALTER TABLE facts ADD COLUMN temporal_weight REAL NOT NULL DEFAULT 1.0;
ALTER TABLE chunks ADD COLUMN temporal_weight REAL NOT NULL DEFAULT 1.0;
ALTER TABLE messages ADD COLUMN temporal_weight REAL NOT NULL DEFAULT 1.0;

CREATE INDEX IF NOT EXISTS idx_facts_temporal ON facts(temporal_weight);
CREATE INDEX IF NOT EXISTS idx_chunks_temporal ON chunks(temporal_weight);