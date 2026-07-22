//! End-to-end journal replication integration test.
//!
//! Proves: real fact mutation → mutation_journal payload → export →
//! fresh replica apply → query → idempotent replay → conflict → gap.
//!
//! This test uses only the public journal API and raw SQLite — no
//! MockEmbedder or MemoryStore needed, because the journal operates
//! at the connection level and the payload is canonical JSON.

use rusqlite::Connection;
use semantic_memory::journal::{
    append_journal_entry, export_contiguous, replay_journal_entry, ReplayOutcome, MIGRATION_V37,
};

/// Create a fresh in-memory DB with the mutation_journal table.
fn fresh_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(MIGRATION_V37).unwrap();
    conn
}

/// Create a minimal facts table so replay can insert into it.
fn create_facts_table(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS facts (
            id          TEXT PRIMARY KEY,
            namespace   TEXT NOT NULL,
            content     TEXT NOT NULL
        );",
    )
    .unwrap();
}

/// Canonical payload format matching knowledge.rs:insert_fact_with_fts_q8.
fn fact_payload(fact_id: &str, namespace: &str, content: &str) -> Vec<u8> {
    serde_json::json!({
        "fact_id": fact_id,
        "namespace": namespace,
        "content": content,
        "source": null,
        "metadata": null,
    })
    .to_string()
    .into_bytes()
}

/// Replay closure: insert a fact from a journal payload.
fn replay_insert_fact(
    conn: &Connection,
    payload: &[u8],
) -> Result<(), semantic_memory::error::MemoryError> {
    let payload_json: serde_json::Value = serde_json::from_slice(payload).unwrap();
    let fact_id = payload_json["fact_id"].as_str().unwrap();
    let namespace = payload_json["namespace"].as_str().unwrap();
    let content = payload_json["content"].as_str().unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO facts (id, namespace, content) VALUES (?1, ?2, ?3)",
        rusqlite::params![fact_id, namespace, content],
    )?;
    Ok(())
}

#[test]
fn full_replication_cycle() {
    let device = "uno-q";
    let store = "primary";

    // ── Primary: add 3 facts via journal ──────────────────────────
    let primary = fresh_db();
    create_facts_table(&primary);

    let facts = [
        ("fact-001", "general", "The sky is blue."),
        ("fact-002", "general", "Water boils at 100C."),
        (
            "fact-003",
            "science",
            "E=mc^2 describes mass-energy equivalence.",
        ),
    ];

    for (fid, ns, content) in &facts {
        // Insert fact (simulates what insert_fact_with_fts_q8 does)
        primary
            .execute(
                "INSERT INTO facts (id, namespace, content) VALUES (?1, ?2, ?3)",
                rusqlite::params![fid, ns, content],
            )
            .unwrap();
        // Append journal entry (same transaction in real code)
        let payload = fact_payload(fid, ns, content);
        let seq = append_journal_entry(&primary, device, store, "add_fact", &payload).unwrap();
        assert!(seq > 0);
    }

    // Verify primary has 3 facts
    let count: i64 = primary
        .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 3);

    // Verify primary journal has 3 entries
    let batch = export_contiguous(&primary, device, store, 1, 10).unwrap();
    assert_eq!(batch.entries.len(), 3);
    assert_eq!(batch.next_seq, 4);
    assert!(!batch.has_more);

    // ── Replica: apply journal entries ────────────────────────────
    let replica = fresh_db();
    create_facts_table(&replica);

    for entry in &batch.entries {
        let outcome = replay_journal_entry(
            &replica,
            device,
            store,
            entry.sequence,
            &entry.operation_kind,
            &entry.payload,
            |conn| replay_insert_fact(conn, &entry.payload),
        )
        .unwrap();
        assert_eq!(
            outcome,
            ReplayOutcome::Applied {
                sequence: entry.sequence
            }
        );
    }

    // Verify replica has same 3 facts with same content
    let replica_count: i64 = replica
        .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
        .unwrap();
    assert_eq!(replica_count, 3);

    // Verify content matches
    for (fid, ns, content) in &facts {
        let replica_content: String = replica
            .query_row(
                "SELECT content FROM facts WHERE id = ?1",
                rusqlite::params![fid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(&replica_content, content);
    }
}

#[test]
fn idempotent_replay_does_not_duplicate() {
    let device = "uno-q";
    let store = "primary";

    let primary = fresh_db();
    create_facts_table(&primary);
    let payload = fact_payload("fact-001", "general", "Test fact.");
    append_journal_entry(&primary, device, store, "add_fact", &payload).unwrap();

    let batch = export_contiguous(&primary, device, store, 1, 10).unwrap();
    let entry = &batch.entries[0];

    let replica = fresh_db();
    create_facts_table(&replica);

    // First apply
    let outcome1 = replay_journal_entry(
        &replica,
        device,
        store,
        entry.sequence,
        &entry.operation_kind,
        &entry.payload,
        |conn| replay_insert_fact(conn, &entry.payload),
    )
    .unwrap();
    assert_eq!(outcome1, ReplayOutcome::Applied { sequence: 1 });

    // Second apply — must be AlreadyApplied
    let outcome2 = replay_journal_entry(
        &replica,
        device,
        store,
        entry.sequence,
        &entry.operation_kind,
        &entry.payload,
        |conn| replay_insert_fact(conn, &entry.payload),
    )
    .unwrap();
    assert_eq!(outcome2, ReplayOutcome::AlreadyApplied { sequence: 1 });

    // Replica must still have exactly 1 fact
    let count: i64 = replica
        .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn conflicting_payload_is_rejected_by_idempotency() {
    let device = "uno-q";
    let store = "primary";

    let primary = fresh_db();
    create_facts_table(&primary);

    // Write fact-001 with content A
    let payload_a = fact_payload("fact-001", "general", "Content A");
    append_journal_entry(&primary, device, store, "add_fact", &payload_a).unwrap();

    let batch = export_contiguous(&primary, device, store, 1, 10).unwrap();
    let entry = &batch.entries[0];

    let replica = fresh_db();
    create_facts_table(&replica);

    // Apply with original payload
    let outcome = replay_journal_entry(
        &replica,
        device,
        store,
        entry.sequence,
        &entry.operation_kind,
        &entry.payload,
        |conn| replay_insert_fact(conn, &entry.payload),
    )
    .unwrap();
    assert_eq!(outcome, ReplayOutcome::Applied { sequence: 1 });

    // Try to replay same sequence with DIFFERENT payload
    let payload_b = fact_payload("fact-001", "general", "Content B - TAMPERED");
    let outcome_tamper = replay_journal_entry(
        &replica,
        device,
        store,
        entry.sequence,
        &entry.operation_kind,
        &payload_b,
        |conn| replay_insert_fact(conn, &payload_b),
    )
    .unwrap();

    // Must be AlreadyApplied — the sequence was already seen.
    // The tampered payload is NOT applied because the journal already
    // has this sequence number.
    assert_eq!(
        outcome_tamper,
        ReplayOutcome::AlreadyApplied { sequence: 1 }
    );

    // Content must still be A, not B
    let content: String = replica
        .query_row(
            "SELECT content FROM facts WHERE id = ?1",
            rusqlite::params!["fact-001"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(content, "Content A");
}

#[test]
fn gap_detection_during_export() {
    let device = "uno-q";
    let store = "primary";

    let primary = fresh_db();
    create_facts_table(&primary);

    // Write sequences 1, 2, 3
    for i in 1..=3 {
        let payload = fact_payload(&format!("fact-{i:03}"), "general", &format!("Fact {i}"));
        append_journal_entry(&primary, device, store, "add_fact", &payload).unwrap();
    }

    // Delete sequence 2 to create a gap
    primary
        .execute("DELETE FROM mutation_journal WHERE sequence = 2", [])
        .unwrap();

    // Export from seq 1 — should get only seq 1, detect gap at 2
    let batch = export_contiguous(&primary, device, store, 1, 10).unwrap();
    assert_eq!(batch.entries.len(), 1);
    assert_eq!(batch.entries[0].sequence, 1);
    assert_eq!(batch.next_seq, 2);
    assert!(!batch.has_more, "gap must set has_more=false");
}

#[test]
fn replica_query_returns_replicated_facts() {
    let device = "uno-q";
    let store = "primary";

    let primary = fresh_db();
    create_facts_table(&primary);

    let payload = fact_payload("fact-q1", "general", "Quantum entanglement is non-local.");
    append_journal_entry(&primary, device, store, "add_fact", &payload).unwrap();

    let batch = export_contiguous(&primary, device, store, 1, 10).unwrap();
    let entry = &batch.entries[0];

    let replica = fresh_db();
    create_facts_table(&replica);

    replay_journal_entry(
        &replica,
        device,
        store,
        entry.sequence,
        &entry.operation_kind,
        &entry.payload,
        |conn| replay_insert_fact(conn, &entry.payload),
    )
    .unwrap();

    // Query replica for the fact
    let content: String = replica
        .query_row(
            "SELECT content FROM facts WHERE namespace = 'general' AND content LIKE '%Quantum%'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(content, "Quantum entanglement is non-local.");
}
