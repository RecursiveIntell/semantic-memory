//! Transactional fact-create outbox for device-primary replication.
//!
//! V37 journal rows are retained as `legacy_unverified`. V38 introduces an
//! explicit stream epoch, transaction-owned sequence/head state, and a
//! domain-separated digest chain. Only exact typed payload bytes are canonical;
//! derived embeddings and indexes are rebuilt by the replica.

use crate::error::MemoryError;
use rusqlite::{Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FACT_CREATE_OPERATION: &str = "fact.create";
pub const FACT_CREATE_PAYLOAD_SCHEMA: &str = "semantic_memory.fact.create.v1";
pub const VERIFIED_RECORD_STATE: &str = "verified_v1";
pub const LEGACY_RECORD_STATE: &str = "legacy_unverified";
pub const GENESIS_PREDECESSOR: [u8; 32] = [0; 32];
const COMPAT_STREAM_EPOCH: u64 = 1;

/// Canonical fact-create authority payload. Field order is fixed by this struct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactCreatePayloadV1 {
    pub fact_id: String,
    pub namespace: String,
    pub content: String,
    pub source: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

pub fn encode_fact_create_payload(payload: &FactCreatePayloadV1) -> Result<Vec<u8>, MemoryError> {
    serde_json::to_vec(payload)
        .map_err(|error| MemoryError::DigestError(format!("fact-create payload encoding: {error}")))
}

/// Domain-separated SHA-256 over length-prefixed exact fields.
pub fn digest_fields(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}

pub fn payload_digest(payload: &[u8]) -> [u8; 32] {
    digest_fields(b"semantic-memory.payload.v1", &[payload])
}

#[allow(clippy::too_many_arguments)]
pub fn envelope_digest(
    home_device_id: &str,
    store_id: &str,
    stream_epoch: u64,
    sequence: i64,
    operation_kind: &str,
    payload_schema: &str,
    predecessor_digest: &[u8; 32],
    payload_digest: &[u8; 32],
) -> [u8; 32] {
    digest_fields(
        b"semantic-memory.envelope.v1",
        &[
            home_device_id.as_bytes(),
            store_id.as_bytes(),
            &stream_epoch.to_be_bytes(),
            &sequence.to_be_bytes(),
            operation_kind.as_bytes(),
            payload_schema.as_bytes(),
            predecessor_digest,
            payload_digest,
        ],
    )
}

/// A verified outbox entry. V37 rows can still be inspected directly in SQLite
/// but are never returned by the verified export API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JournalEntry {
    pub journal_id: i64,
    pub home_device_id: String,
    pub store_id: String,
    pub stream_epoch: u64,
    pub sequence: i64,
    pub operation_kind: String,
    pub payload_schema: String,
    pub payload: Vec<u8>,
    pub payload_digest: [u8; 32],
    pub predecessor_digest: [u8; 32],
    pub envelope_digest: [u8; 32],
    pub record_state: String,
    pub created_at: String,
}

/// V37 migration retained for compatibility. These rows have no cryptographic
/// chain and become explicitly `legacy_unverified` under V38.
pub const MIGRATION_V37: &str = "\
CREATE TABLE IF NOT EXISTS mutation_journal (
    journal_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    home_device_id  TEXT NOT NULL,
    store_id        TEXT NOT NULL,
    sequence        INTEGER NOT NULL,
    operation_kind  TEXT NOT NULL,
    payload         BLOB NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_journal_sequence
    ON mutation_journal(home_device_id, store_id, sequence);
";

/// V38 verified stream state. SQLite ALTER defaults are literal constants so
/// existing V37 databases migrate without fabricating verified metadata.
pub const MIGRATION_V38: &str = r#"
CREATE TABLE IF NOT EXISTS replication_streams (
    home_device_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    stream_epoch INTEGER NOT NULL CHECK(stream_epoch > 0),
    next_sequence INTEGER NOT NULL DEFAULT 1 CHECK(next_sequence > 0),
    head_digest BLOB NOT NULL CHECK(length(head_digest) = 32),
    PRIMARY KEY(home_device_id, store_id, stream_epoch)
);
ALTER TABLE mutation_journal ADD COLUMN stream_epoch INTEGER NOT NULL DEFAULT 0;
ALTER TABLE mutation_journal ADD COLUMN payload_schema TEXT NOT NULL DEFAULT 'legacy.unverified';
ALTER TABLE mutation_journal ADD COLUMN payload_digest BLOB NOT NULL DEFAULT X'0000000000000000000000000000000000000000000000000000000000000000';
ALTER TABLE mutation_journal ADD COLUMN predecessor_digest BLOB NOT NULL DEFAULT X'0000000000000000000000000000000000000000000000000000000000000000';
ALTER TABLE mutation_journal ADD COLUMN envelope_digest BLOB NOT NULL DEFAULT X'0000000000000000000000000000000000000000000000000000000000000000';
ALTER TABLE mutation_journal ADD COLUMN record_state TEXT NOT NULL DEFAULT 'legacy_unverified';
DROP INDEX IF EXISTS idx_journal_sequence;
CREATE UNIQUE INDEX IF NOT EXISTS idx_journal_sequence_v38
    ON mutation_journal(home_device_id, store_id, stream_epoch, sequence);
"#;

fn validate_stream_identity(
    home_device_id: &str,
    store_id: &str,
    stream_epoch: u64,
) -> Result<(), MemoryError> {
    if stream_epoch == 0 {
        return Err(MemoryError::InvalidConfig {
            field: "replication_stream_epoch",
            reason: "must be positive".to_string(),
        });
    }
    for (field, value) in [
        ("journal_device_id", home_device_id),
        ("journal_store_id", store_id),
    ] {
        if value.is_empty() || value.trim() != value || value.chars().any(char::is_whitespace) {
            return Err(MemoryError::InvalidConfig {
                field,
                reason: "must be non-empty, trimmed, and contain no whitespace".to_string(),
            });
        }
    }
    Ok(())
}

fn digest_from_blob(column: usize, bytes: Vec<u8>) -> Result<[u8; 32], rusqlite::Error> {
    bytes.try_into().map_err(|bytes: Vec<u8>| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            rusqlite::types::Type::Blob,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected 32-byte digest, got {} bytes", bytes.len()),
            )),
        )
    })
}

fn row_to_entry(row: &rusqlite::Row<'_>) -> Result<JournalEntry, rusqlite::Error> {
    let epoch: i64 = row.get(3)?;
    let stream_epoch =
        u64::try_from(epoch).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(3, epoch))?;
    Ok(JournalEntry {
        journal_id: row.get(0)?,
        home_device_id: row.get(1)?,
        store_id: row.get(2)?,
        stream_epoch,
        sequence: row.get(4)?,
        operation_kind: row.get(5)?,
        payload_schema: row.get(6)?,
        payload: row.get(7)?,
        payload_digest: digest_from_blob(8, row.get(8)?)?,
        predecessor_digest: digest_from_blob(9, row.get(9)?)?,
        envelope_digest: digest_from_blob(10, row.get(10)?)?,
        record_state: row.get(11)?,
        created_at: row.get(12)?,
    })
}

const ENTRY_SELECT: &str = "journal_id, home_device_id, store_id, stream_epoch, sequence, \
operation_kind, payload_schema, payload, payload_digest, predecessor_digest, envelope_digest, \
record_state, created_at";

/// Append a verified record while the caller's semantic mutation transaction is active.
/// The stream row is the allocator; no sequence is derived from journal contents.
#[allow(clippy::too_many_arguments)]
pub fn append_verified_in_tx(
    tx: &Transaction<'_>,
    home_device_id: &str,
    store_id: &str,
    stream_epoch: u64,
    operation_kind: &str,
    payload_schema: &str,
    payload: &[u8],
) -> Result<JournalEntry, MemoryError> {
    validate_stream_identity(home_device_id, store_id, stream_epoch)?;
    if operation_kind != FACT_CREATE_OPERATION || payload_schema != FACT_CREATE_PAYLOAD_SCHEMA {
        return Err(MemoryError::NotImplemented(format!(
            "replication operation/schema not admitted: {operation_kind}/{payload_schema}"
        )));
    }
    let epoch = i64::try_from(stream_epoch).map_err(|_| MemoryError::InvalidConfig {
        field: "replication_stream_epoch",
        reason: "does not fit SQLite INTEGER".to_string(),
    })?;

    tx.execute(
        "INSERT INTO replication_streams
         (home_device_id, store_id, stream_epoch, next_sequence, head_digest)
         VALUES (?1, ?2, ?3, 1, ?4)
         ON CONFLICT(home_device_id, store_id, stream_epoch) DO NOTHING",
        rusqlite::params![
            home_device_id,
            store_id,
            epoch,
            GENESIS_PREDECESSOR.as_slice()
        ],
    )?;

    let (sequence, predecessor_bytes): (i64, Vec<u8>) = tx.query_row(
        "SELECT next_sequence, head_digest FROM replication_streams
         WHERE home_device_id = ?1 AND store_id = ?2 AND stream_epoch = ?3",
        rusqlite::params![home_device_id, store_id, epoch],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let predecessor = digest_from_blob(1, predecessor_bytes)?;
    let payload_hash = payload_digest(payload);
    let envelope_hash = envelope_digest(
        home_device_id,
        store_id,
        stream_epoch,
        sequence,
        operation_kind,
        payload_schema,
        &predecessor,
        &payload_hash,
    );

    tx.execute(
        "INSERT INTO mutation_journal
         (home_device_id, store_id, stream_epoch, sequence, operation_kind,
          payload_schema, payload, payload_digest, predecessor_digest,
          envelope_digest, record_state)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        rusqlite::params![
            home_device_id,
            store_id,
            epoch,
            sequence,
            operation_kind,
            payload_schema,
            payload,
            payload_hash.as_slice(),
            predecessor.as_slice(),
            envelope_hash.as_slice(),
            VERIFIED_RECORD_STATE,
        ],
    )?;
    let journal_id = tx.last_insert_rowid();

    let advanced = tx.execute(
        "UPDATE replication_streams
         SET next_sequence = ?4, head_digest = ?5
         WHERE home_device_id = ?1 AND store_id = ?2 AND stream_epoch = ?3
           AND next_sequence = ?6 AND head_digest = ?7",
        rusqlite::params![
            home_device_id,
            store_id,
            epoch,
            sequence + 1,
            envelope_hash.as_slice(),
            sequence,
            predecessor.as_slice(),
        ],
    )?;
    if advanced != 1 {
        return Err(MemoryError::Other(
            "replication stream allocator lost ownership".to_string(),
        ));
    }

    tx.query_row(
        &format!("SELECT {ENTRY_SELECT} FROM mutation_journal WHERE journal_id = ?1"),
        [journal_id],
        row_to_entry,
    )
    .map_err(MemoryError::Database)
}

/// Compatibility helper for fact-create tests and offline tools. It owns a
/// transaction and therefore remains atomic, but callers should use the real
/// MemoryStore fact mutation path instead.
#[deprecated(note = "use the MemoryStore fact-create path or append_verified_in_tx")]
pub fn append_journal_entry(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
    operation_kind: &str,
    payload: &[u8],
) -> Result<i64, MemoryError> {
    if operation_kind != "add_fact" && operation_kind != FACT_CREATE_OPERATION {
        return Err(MemoryError::NotImplemented(format!(
            "legacy journal operation not admitted: {operation_kind}"
        )));
    }
    let tx = conn.unchecked_transaction()?;
    let entry = append_verified_in_tx(
        &tx,
        home_device_id,
        store_id,
        COMPAT_STREAM_EPOCH,
        FACT_CREATE_OPERATION,
        FACT_CREATE_PAYLOAD_SCHEMA,
        payload,
    )?;
    tx.commit()?;
    Ok(entry.sequence)
}

/// Compatibility mutation wrapper. The semantic mutation and outbox record are
/// committed together, and a failed closure consumes no sequence.
#[deprecated(note = "use the typed MemoryStore mutation path")]
pub fn mutate_and_journal<F, T>(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
    operation_kind: &str,
    payload: &[u8],
    f: F,
) -> Result<(i64, i64, T), MemoryError>
where
    F: FnOnce(&Connection) -> Result<T, MemoryError>,
{
    if operation_kind != "add_fact" && operation_kind != FACT_CREATE_OPERATION {
        return Err(MemoryError::NotImplemented(format!(
            "legacy journal operation not admitted: {operation_kind}"
        )));
    }
    let tx = conn.unchecked_transaction()?;
    let result = f(&tx)?;
    let entry = append_verified_in_tx(
        &tx,
        home_device_id,
        store_id,
        COMPAT_STREAM_EPOCH,
        FACT_CREATE_OPERATION,
        FACT_CREATE_PAYLOAD_SCHEMA,
        payload,
    )?;
    tx.commit()?;
    Ok((entry.journal_id, entry.sequence, result))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportStatus {
    End,
    More,
    Gap { expected: i64, found: Option<i64> },
    Corrupt { sequence: i64, reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedExportBatch {
    pub entries: Vec<JournalEntry>,
    pub next_sequence: i64,
    pub status: ExportStatus,
}

fn corrupt(
    entries: Vec<JournalEntry>,
    next_sequence: i64,
    sequence: i64,
    reason: impl Into<String>,
) -> VerifiedExportBatch {
    VerifiedExportBatch {
        entries,
        next_sequence,
        status: ExportStatus::Corrupt {
            sequence,
            reason: reason.into(),
        },
    }
}

/// Export and verify a contiguous prefix. Legacy V37 rows are never promoted
/// into this API and any stored digest/chain mismatch is typed as corruption.
pub fn export_verified_contiguous(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
    stream_epoch: u64,
    start_sequence: i64,
    limit: usize,
) -> Result<VerifiedExportBatch, MemoryError> {
    validate_stream_identity(home_device_id, store_id, stream_epoch)?;
    if start_sequence < 1 || limit == 0 {
        return Err(MemoryError::InvalidConfig {
            field: "journal_export",
            reason: "start_sequence and limit must be positive".to_string(),
        });
    }
    let epoch = i64::try_from(stream_epoch).map_err(|_| MemoryError::InvalidConfig {
        field: "replication_stream_epoch",
        reason: "does not fit SQLite INTEGER".to_string(),
    })?;
    let stream_next: Option<i64> = conn
        .query_row(
            "SELECT next_sequence FROM replication_streams
             WHERE home_device_id = ?1 AND store_id = ?2 AND stream_epoch = ?3",
            rusqlite::params![home_device_id, store_id, epoch],
            |row| row.get(0),
        )
        .optional()?;
    let Some(stream_next) = stream_next else {
        return Ok(VerifiedExportBatch {
            entries: Vec::new(),
            next_sequence: start_sequence,
            status: ExportStatus::End,
        });
    };

    let mut expected_predecessor = if start_sequence == 1 {
        GENESIS_PREDECESSOR
    } else {
        let previous: Option<Vec<u8>> = conn
            .query_row(
                "SELECT envelope_digest FROM mutation_journal
                 WHERE home_device_id = ?1 AND store_id = ?2 AND stream_epoch = ?3
                   AND sequence = ?4 AND record_state = ?5",
                rusqlite::params![
                    home_device_id,
                    store_id,
                    epoch,
                    start_sequence - 1,
                    VERIFIED_RECORD_STATE,
                ],
                |row| row.get(0),
            )
            .optional()?;
        let Some(previous) = previous else {
            return Ok(VerifiedExportBatch {
                entries: Vec::new(),
                next_sequence: start_sequence,
                status: ExportStatus::Gap {
                    expected: start_sequence - 1,
                    found: None,
                },
            });
        };
        digest_from_blob(0, previous)?
    };

    let mut stmt = conn.prepare(&format!(
        "SELECT {ENTRY_SELECT} FROM mutation_journal
         WHERE home_device_id = ?1 AND store_id = ?2 AND stream_epoch = ?3
           AND sequence >= ?4 AND record_state = ?5
         ORDER BY sequence ASC LIMIT ?6"
    ))?;
    let rows = stmt.query_map(
        rusqlite::params![
            home_device_id,
            store_id,
            epoch,
            start_sequence,
            VERIFIED_RECORD_STATE,
            (limit + 1) as i64,
        ],
        row_to_entry,
    )?;

    let mut entries = Vec::new();
    let mut expected = start_sequence;
    let mut has_extra = false;
    for row in rows {
        let entry = row?;
        if entries.len() == limit {
            has_extra = true;
            break;
        }
        if entry.sequence != expected {
            return Ok(VerifiedExportBatch {
                entries,
                next_sequence: expected,
                status: ExportStatus::Gap {
                    expected,
                    found: Some(entry.sequence),
                },
            });
        }
        if entry.operation_kind != FACT_CREATE_OPERATION
            || entry.payload_schema != FACT_CREATE_PAYLOAD_SCHEMA
            || entry.record_state != VERIFIED_RECORD_STATE
        {
            return Ok(corrupt(
                entries,
                expected,
                entry.sequence,
                "unadmitted operation, schema, or record state",
            ));
        }
        let expected_payload = payload_digest(&entry.payload);
        if entry.payload_digest != expected_payload {
            return Ok(corrupt(
                entries,
                expected,
                entry.sequence,
                "payload digest mismatch",
            ));
        }
        if entry.predecessor_digest != expected_predecessor {
            return Ok(corrupt(
                entries,
                expected,
                entry.sequence,
                "predecessor digest mismatch",
            ));
        }
        let expected_envelope = envelope_digest(
            home_device_id,
            store_id,
            stream_epoch,
            entry.sequence,
            &entry.operation_kind,
            &entry.payload_schema,
            &entry.predecessor_digest,
            &entry.payload_digest,
        );
        if entry.envelope_digest != expected_envelope {
            return Ok(corrupt(
                entries,
                expected,
                entry.sequence,
                "envelope digest mismatch",
            ));
        }
        expected_predecessor = entry.envelope_digest;
        expected += 1;
        entries.push(entry);
    }

    let status = if has_extra || expected < stream_next {
        if has_extra {
            ExportStatus::More
        } else {
            ExportStatus::Gap {
                expected,
                found: None,
            }
        }
    } else if expected == stream_next {
        ExportStatus::End
    } else {
        ExportStatus::Corrupt {
            sequence: expected,
            reason: format!(
                "export advanced beyond stream allocator: export next {expected}, stream next {stream_next}"
            ),
        }
    };
    Ok(VerifiedExportBatch {
        entries,
        next_sequence: expected,
        status,
    })
}

/// Legacy shape retained for current callers. It exports only verified epoch-1
/// records and does not hide corruption as successful completion.
#[derive(Debug, Clone)]
pub struct ExportedBatch {
    pub entries: Vec<JournalEntry>,
    pub next_seq: i64,
    pub has_more: bool,
}

#[deprecated(note = "use export_verified_contiguous with an explicit epoch")]
pub fn export_contiguous(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
    start_seq: i64,
    limit: usize,
) -> Result<ExportedBatch, MemoryError> {
    let batch = export_verified_contiguous(
        conn,
        home_device_id,
        store_id,
        COMPAT_STREAM_EPOCH,
        start_seq,
        limit,
    )?;
    if let ExportStatus::Corrupt { sequence, reason } = &batch.status {
        return Err(MemoryError::CorruptData {
            table: "mutation_journal",
            row_id: sequence.to_string(),
            detail: reason.clone(),
        });
    }
    Ok(ExportedBatch {
        entries: batch.entries,
        next_seq: batch.next_sequence,
        has_more: matches!(batch.status, ExportStatus::More),
    })
}

/// Compatibility-only replay result. New remote admission must use the closed
/// typed fact-create dispatcher, not this closure-based adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayOutcome {
    Applied { sequence: i64 },
    AlreadyApplied { sequence: i64 },
    Conflict { sequence: i64 },
    Gap { expected: i64, received: i64 },
}

#[deprecated(note = "closure-based replay is compatibility-only; use closed typed admission")]
pub fn replay_journal_entry<F>(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
    sequence: i64,
    operation_kind: &str,
    payload: &[u8],
    replay_fn: F,
) -> Result<ReplayOutcome, MemoryError>
where
    F: FnOnce(&Connection) -> Result<(), MemoryError>,
{
    if operation_kind != "add_fact" && operation_kind != FACT_CREATE_OPERATION {
        return Ok(ReplayOutcome::Conflict { sequence });
    }
    let tx = conn.unchecked_transaction()?;
    let existing: Option<(String, Vec<u8>)> = tx
        .query_row(
            "SELECT operation_kind, payload FROM mutation_journal
             WHERE home_device_id = ?1 AND store_id = ?2 AND stream_epoch = ?3 AND sequence = ?4",
            rusqlite::params![
                home_device_id,
                store_id,
                COMPAT_STREAM_EPOCH as i64,
                sequence
            ],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((stored_operation, stored_payload)) = existing {
        if stored_operation == FACT_CREATE_OPERATION && stored_payload == payload {
            return Ok(ReplayOutcome::AlreadyApplied { sequence });
        }
        return Ok(ReplayOutcome::Conflict { sequence });
    }

    let expected =
        next_expected_sequence_for_epoch(&tx, home_device_id, store_id, COMPAT_STREAM_EPOCH)?;
    if sequence != expected {
        return Ok(ReplayOutcome::Gap {
            expected,
            received: sequence,
        });
    }
    replay_fn(&tx)?;
    let entry = append_verified_in_tx(
        &tx,
        home_device_id,
        store_id,
        COMPAT_STREAM_EPOCH,
        FACT_CREATE_OPERATION,
        FACT_CREATE_PAYLOAD_SCHEMA,
        payload,
    )?;
    debug_assert_eq!(entry.sequence, sequence);
    tx.commit()?;
    Ok(ReplayOutcome::Applied { sequence })
}

pub fn next_expected_sequence_for_epoch(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
    stream_epoch: u64,
) -> Result<i64, MemoryError> {
    validate_stream_identity(home_device_id, store_id, stream_epoch)?;
    let epoch = i64::try_from(stream_epoch).map_err(|_| MemoryError::InvalidConfig {
        field: "replication_stream_epoch",
        reason: "does not fit SQLite INTEGER".to_string(),
    })?;
    Ok(conn
        .query_row(
            "SELECT next_sequence FROM replication_streams
             WHERE home_device_id = ?1 AND store_id = ?2 AND stream_epoch = ?3",
            rusqlite::params![home_device_id, store_id, epoch],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(1))
}

#[deprecated(note = "use next_expected_sequence_for_epoch")]
pub fn next_expected_sequence(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
) -> Result<i64, MemoryError> {
    next_expected_sequence_for_epoch(conn, home_device_id, store_id, COMPAT_STREAM_EPOCH)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    fn test_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(MIGRATION_V37).unwrap();
        conn.execute_batch(MIGRATION_V38).unwrap();
        conn
    }

    #[test]
    #[allow(deprecated)]
    fn first_record_uses_genesis_and_export_verifies_chain() {
        let conn = test_conn();
        append_journal_entry(&conn, "device-1", "store-1", "add_fact", b"payload-1").unwrap();
        append_journal_entry(&conn, "device-1", "store-1", "add_fact", b"payload-2").unwrap();
        let batch = export_verified_contiguous(&conn, "device-1", "store-1", 1, 1, 10).unwrap();
        assert_eq!(batch.status, ExportStatus::End);
        assert_eq!(batch.entries.len(), 2);
        assert_eq!(batch.entries[0].predecessor_digest, GENESIS_PREDECESSOR);
        assert_eq!(
            batch.entries[1].predecessor_digest,
            batch.entries[0].envelope_digest
        );
        assert_eq!(batch.next_sequence, 3);
    }

    #[test]
    #[allow(deprecated)]
    fn failed_mutation_consumes_no_sequence() {
        let conn = test_conn();
        let result = mutate_and_journal(
            &conn,
            "device-1",
            "store-1",
            "add_fact",
            b"payload",
            |_conn| Err::<(), _>(MemoryError::Database(rusqlite::Error::InvalidQuery)),
        );
        assert!(result.is_err());
        assert_eq!(
            next_expected_sequence_for_epoch(&conn, "device-1", "store-1", 1).unwrap(),
            1
        );
    }

    #[test]
    #[allow(deprecated)]
    fn same_sequence_changed_payload_is_conflict() {
        let conn = test_conn();
        conn.execute("CREATE TABLE replayed(value TEXT)", [])
            .unwrap();
        let first =
            replay_journal_entry(&conn, "device-1", "store-1", 1, "add_fact", b"a", |conn| {
                conn.execute("INSERT INTO replayed(value) VALUES ('a')", [])?;
                Ok(())
            })
            .unwrap();
        assert_eq!(first, ReplayOutcome::Applied { sequence: 1 });
        let second =
            replay_journal_entry(&conn, "device-1", "store-1", 1, "add_fact", b"b", |_conn| {
                panic!("conflicting replay must not run")
            })
            .unwrap();
        assert_eq!(second, ReplayOutcome::Conflict { sequence: 1 });
        let value: String = conn
            .query_row("SELECT value FROM replayed", [], |row| row.get(0))
            .unwrap();
        assert_eq!(value, "a");
    }

    #[test]
    #[allow(deprecated)]
    fn separate_connections_allocate_one_contiguous_stream() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("journal.db");
        let setup = Connection::open(&path).unwrap();
        setup.execute_batch(MIGRATION_V37).unwrap();
        setup.execute_batch(MIGRATION_V38).unwrap();
        drop(setup);

        let workers = 8;
        let barrier = Arc::new(Barrier::new(workers));
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let conn = Connection::open(path).unwrap();
                    conn.busy_timeout(Duration::from_secs(10)).unwrap();
                    barrier.wait();
                    append_journal_entry(
                        &conn,
                        "device-1",
                        "store-1",
                        "add_fact",
                        format!("payload-{worker}").as_bytes(),
                    )
                    .unwrap()
                })
            })
            .collect();
        let mut sequences: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        sequences.sort_unstable();
        assert_eq!(sequences, (1..=workers as i64).collect::<Vec<_>>());
    }

    #[test]
    #[allow(deprecated)]
    fn tampered_chain_is_reported_as_corrupt() {
        let conn = test_conn();
        append_journal_entry(&conn, "device-1", "store-1", "add_fact", b"payload").unwrap();
        conn.execute(
            "UPDATE mutation_journal SET predecessor_digest = ?1 WHERE sequence = 1",
            [vec![7_u8; 32]],
        )
        .unwrap();
        let batch = export_verified_contiguous(&conn, "device-1", "store-1", 1, 1, 10).unwrap();
        assert!(matches!(
            batch.status,
            ExportStatus::Corrupt { sequence: 1, .. }
        ));
    }
}
