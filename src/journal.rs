//! Mutation journal for device-owned replication.
//!
//! Provides a durable, contiguous, append-only journal that wraps existing
//! semantic-memory mutation paths. Each journal entry records:
//! - device/store identity
//! - monotonic sequence number
//! - operation kind (add_fact, add_message, etc.)
//! - canonical payload for replication
//!
//! The journal is committed atomically with the semantic mutation in the
//! same SQLite transaction. Export and replay operations provide the
//! foundation for device-primary → server-replica synchronization.

use crate::error::MemoryError;
use rusqlite::Connection;

/// A single journal entry representing one semantic mutation.
#[derive(Debug, Clone)]
pub struct JournalEntry {
    pub journal_id: i64,
    pub home_device_id: String,
    pub store_id: String,
    pub sequence: i64,
    pub operation_kind: String,
    pub payload: Vec<u8>,
    pub created_at: String,
}

/// V37 migration: mutation journal for device-owned replication.
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

/// Append a journal entry within an active transaction.
///
/// The caller is responsible for opening the transaction and committing
/// the semantic mutation together. This function only appends the journal
/// row and returns the assigned journal_id.
pub fn append_journal_entry(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
    operation_kind: &str,
    payload: &[u8],
) -> Result<i64, MemoryError> {
    let next_seq: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(sequence), 0) + 1
             FROM mutation_journal
             WHERE home_device_id = ?1 AND store_id = ?2",
            rusqlite::params![home_device_id, store_id],
            |row| row.get(0),
        )
        .map_err(|e| MemoryError::Database(e))?;

    conn.execute(
        "INSERT INTO mutation_journal (home_device_id, store_id, sequence, operation_kind, payload)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![home_device_id, store_id, next_seq, operation_kind, payload],
    )
    .map_err(MemoryError::Database)?;

    Ok(next_seq)
}

/// Execute a mutation closure within a transaction that also journals it.
///
/// The closure receives `&Connection` and must perform the semantic mutation.
/// After the closure succeeds, a journal entry is appended. If any step fails,
/// the entire transaction is rolled back.
///
/// Returns `(journal_id, sequence, closure_result)`.
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
    let tx = conn
        .unchecked_transaction()
        .map_err(MemoryError::Database)?;

    // SAFETY: closure runs inside the transaction; any error aborts the tx.
    let result = f(conn)?;

    let seq = append_journal_entry(conn, home_device_id, store_id, operation_kind, payload)?;

    let journal_id: i64 = conn.last_insert_rowid();

    tx.commit().map_err(MemoryError::Database)?;

    Ok((journal_id, seq, result))
}

/// Export a contiguous range of journal entries.
///
/// Returns entries with sequence numbers in `[start_seq, start_seq + limit)`.
/// If a gap is detected (missing sequence), returns only the entries before the
/// gap and sets `has_more` to false with `next_seq` pointing to the gap.
pub fn export_contiguous(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
    start_seq: i64,
    limit: usize,
) -> Result<ExportedBatch, MemoryError> {
    let mut stmt = conn
        .prepare(
            "SELECT journal_id, home_device_id, store_id, sequence, operation_kind, payload, created_at
             FROM mutation_journal
             WHERE home_device_id = ?1 AND store_id = ?2 AND sequence >= ?3
             ORDER BY sequence ASC
             LIMIT ?4",
        )
        .map_err(MemoryError::Database)?;

    let rows = stmt
        .query_map(
            rusqlite::params![home_device_id, store_id, start_seq, limit as i64],
            |row| {
                Ok(JournalEntry {
                    journal_id: row.get(0)?,
                    home_device_id: row.get(1)?,
                    store_id: row.get(2)?,
                    sequence: row.get(3)?,
                    operation_kind: row.get(4)?,
                    payload: row.get(5)?,
                    created_at: row.get(6)?,
                })
            },
        )
        .map_err(MemoryError::Database)?;

    let mut entries: Vec<JournalEntry> = Vec::new();
    let mut expected = start_seq;
    let mut has_gap = false;
    for row in rows {
        let entry = row.map_err(MemoryError::Database)?;
        if entry.sequence != expected {
            has_gap = true;
            break;
        }
        expected = entry.sequence + 1;
        entries.push(entry);
    }

    let has_more = if has_gap {
        false
    } else {
        entries.len() >= limit
    };

    Ok(ExportedBatch {
        entries,
        next_seq: expected,
        has_more,
    })
}

/// A batch of journal entries ready for replication.
#[derive(Debug, Clone)]
pub struct ExportedBatch {
    pub entries: Vec<JournalEntry>,
    pub next_seq: i64,
    pub has_more: bool,
}

/// Apply a single journal entry on a replica store.
///
/// The `replay_fn` closure receives the operation_kind and payload and must
/// perform the equivalent semantic mutation on the replica. This function
/// first checks whether the sequence has already been applied (idempotent replay),
/// atomically applies the mutation and records the sequence, or quits silently
/// if the sequence was already seen.
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
    // Check if already applied
    let already: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM mutation_journal
             WHERE home_device_id = ?1 AND store_id = ?2 AND sequence = ?3",
            rusqlite::params![home_device_id, store_id, sequence],
            |row| row.get(0),
        )
        .map_err(MemoryError::Database)?;

    if already {
        return Ok(ReplayOutcome::AlreadyApplied { sequence });
    }

    let tx = conn
        .unchecked_transaction()
        .map_err(MemoryError::Database)?;

    // SAFETY: closure runs inside transaction; any error aborts
    replay_fn(conn)?;

    conn.execute(
        "INSERT INTO mutation_journal (home_device_id, store_id, sequence, operation_kind, payload)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![home_device_id, store_id, sequence, operation_kind, payload],
    )
    .map_err(MemoryError::Database)?;

    tx.commit().map_err(MemoryError::Database)?;

    Ok(ReplayOutcome::Applied { sequence })
}

/// Outcome of replaying a journal entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayOutcome {
    Applied { sequence: i64 },
    AlreadyApplied { sequence: i64 },
}

/// Get the next expected sequence number for a device/store pair.
pub fn next_expected_sequence(
    conn: &Connection,
    home_device_id: &str,
    store_id: &str,
) -> Result<i64, MemoryError> {
    let seq: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(sequence), 0) + 1
             FROM mutation_journal
             WHERE home_device_id = ?1 AND store_id = ?2",
            rusqlite::params![home_device_id, store_id],
            |row| row.get(0),
        )
        .map_err(MemoryError::Database)?;
    Ok(seq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn test_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(MIGRATION_V37).unwrap();
        conn
    }

    #[test]
    fn append_and_export() {
        let conn = test_conn();
        let seq =
            append_journal_entry(&conn, "device-1", "store-1", "add_fact", b"payload-1").unwrap();
        assert_eq!(seq, 1);

        let batch = export_contiguous(&conn, "device-1", "store-1", 1, 10).unwrap();
        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.next_seq, 2);
        assert!(!batch.has_more);
    }

    #[test]
    fn contiguous_gap_detection() {
        let conn = test_conn();
        append_journal_entry(&conn, "device-1", "store-1", "add_fact", b"p1").unwrap();
        append_journal_entry(&conn, "device-1", "store-1", "add_fact", b"p2").unwrap();
        // Delete sequence 2 to create a gap
        conn.execute("DELETE FROM mutation_journal WHERE sequence = 2", [])
            .unwrap();

        // Now export from seq 1: should only get seq 1, detecting gap at seq 2
        let batch = export_contiguous(&conn, "device-1", "store-1", 1, 10).unwrap();
        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.entries[0].sequence, 1);
        assert_eq!(batch.next_seq, 2);
        assert!(!batch.has_more, "gap must set has_more=false");
    }

    #[test]
    fn replay_is_idempotent() {
        let conn = test_conn();

        use std::cell::Cell;
        let call_count = Cell::new(0);
        let replay = |c: &Connection| -> Result<(), MemoryError> {
            call_count.set(call_count.get() + 1);
            c.execute("CREATE TABLE IF NOT EXISTS replayed (id INTEGER)", [])
                .map_err(MemoryError::Database)?;
            Ok(())
        };

        let result =
            replay_journal_entry(&conn, "device-1", "store-1", 1, "add_fact", b"p1", &replay)
                .unwrap();
        assert_eq!(result, ReplayOutcome::Applied { sequence: 1 });
        assert_eq!(call_count.get(), 1);

        // Replay same sequence — must be AlreadyApplied
        let result =
            replay_journal_entry(&conn, "device-1", "store-1", 1, "add_fact", b"p1", &replay)
                .unwrap();
        assert_eq!(result, ReplayOutcome::AlreadyApplied { sequence: 1 });
        assert_eq!(call_count.get(), 1, "replay fn must not be called again");
    }

    #[test]
    fn mutate_and_journal_is_atomic() {
        let conn = test_conn();

        // Test: mutation fails, journal must not be written
        let result = mutate_and_journal(
            &conn,
            "device-1",
            "store-1",
            "add_fact",
            b"payload",
            |_c| -> Result<(), MemoryError> {
                Err(MemoryError::Database(rusqlite::Error::InvalidQuery))
            },
        );
        assert!(result.is_err());

        // Journal must be empty
        let batch = export_contiguous(&conn, "device-1", "store-1", 1, 10).unwrap();
        assert!(batch.entries.is_empty());
    }

    #[test]
    fn mutate_and_journal_success() {
        let conn = test_conn();

        let (jid, seq, val) = mutate_and_journal(
            &conn,
            "device-1",
            "store-1",
            "add_fact",
            b"payload",
            |c| -> Result<_, MemoryError> {
                c.execute("CREATE TABLE IF NOT EXISTS test_table (x INTEGER)", [])
                    .map_err(MemoryError::Database)?;
                Ok(42)
            },
        )
        .unwrap();

        assert_eq!(seq, 1);
        assert_eq!(val, 42);
        assert!(jid > 0);

        let batch = export_contiguous(&conn, "device-1", "store-1", 1, 10).unwrap();
        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.entries[0].operation_kind, "add_fact");
    }
}
