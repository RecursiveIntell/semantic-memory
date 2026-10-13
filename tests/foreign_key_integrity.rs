//! Full owner diagnostics must expose FK corruption without repairing it.
//! Raw SQL below deliberately corrupts only test-owned temporary databases.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rusqlite::Connection;
use semantic_memory::{MemoryConfig, MemoryError, MemoryStore, MockEmbedder, VerifyMode};
use tempfile::TempDir;

fn fixture() -> (TempDir, MemoryStore, Connection) {
    let dir = TempDir::new().unwrap();
    let config = MemoryConfig {
        base_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    // Explicit unused test embedder: no provider/network request is needed to
    // test real SQLite schema and the canonical integrity API.
    let embedder = Box::new(MockEmbedder::new(config.embedding.dimensions));
    let store = MemoryStore::open_with_embedder(config, embedder).unwrap();
    let conn = Connection::open(dir.path().join("memory.db")).unwrap();
    conn.execute_batch("PRAGMA foreign_keys = OFF;").unwrap();
    (dir, store, conn)
}

const CORRUPT_AUTHORITY: &str = "
INSERT INTO authority_lineages VALUES ('lineage:fixture', 'missing-private-id', 0);
INSERT INTO authority_versions VALUES
    ('missing-private-id', 'lineage:fixture', 1, 'append', 1, 0, 'private-digest');
INSERT INTO origin_authority_labels VALUES
    ('missing-private-id', 'private-label-canary', 'private-digest', '2026-01-01');
INSERT INTO origin_authority_revocations VALUES
    ('revocation:fixture', 'missing-private-id', 'fixture-key', 'private-principal',
     'private-revocation-canary', '2026-01-01');
INSERT INTO forgotten_facts VALUES
    ('missing-private-id', 'receipt:fixture', 'private', 'private-digest', '2026-01-01');
";

fn fk_count(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
        row.get(0)
    })
    .unwrap()
}

fn authority_snapshot(conn: &Connection) -> Vec<String> {
    let mut snapshot = Vec::new();
    for table in [
        "authority_lineages",
        "authority_versions",
        "origin_authority_labels",
        "origin_authority_revocations",
        "forgotten_facts",
        "authority_state",
    ] {
        let mut stmt = conn
            .prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))
            .unwrap();
        let columns = stmt.column_count();
        let rows = stmt
            .query_map([], |row| {
                (0..columns)
                    .map(|index| row.get::<_, rusqlite::types::Value>(index))
                    .collect::<Result<Vec<_>, _>>()
            })
            .unwrap();
        snapshot.push(format!(
            "{table}: {:?}",
            rows.collect::<Result<Vec<_>, _>>().unwrap()
        ));
    }
    snapshot
}

#[tokio::test]
async fn full_reports_all_five_authority_families_without_mutation_or_content_echo() {
    let (_dir, store, conn) = fixture();
    assert!(store.verify_integrity(VerifyMode::Full).await.unwrap().ok);
    conn.execute_batch(CORRUPT_AUTHORITY).unwrap();
    conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
    assert_eq!(fk_count(&conn), 5);
    let sqlite_integrity: String = conn
        .query_row("PRAGMA integrity_check", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        sqlite_integrity, "ok",
        "SQLite structural check alone misses FK corruption"
    );
    let before = authority_snapshot(&conn);
    let report = store.verify_integrity(VerifyMode::Full).await.unwrap();
    assert!(
        !report.ok,
        "Full owner check must not report FK-corrupt authority as healthy"
    );
    let violations: Vec<_> = report
        .issues
        .iter()
        .filter(|issue| issue.starts_with("SQLite foreign_key_check:"))
        .collect();
    assert_eq!(violations.len(), 5, "{:?}", report.issues);
    for table in [
        "authority_lineages",
        "authority_versions",
        "origin_authority_labels",
        "origin_authority_revocations",
        "forgotten_facts",
    ] {
        assert!(
            violations
                .iter()
                .any(|issue| issue.contains(&format!("table={table},"))),
            "missing {table}: {:?}",
            report.issues
        );
    }
    for canary in [
        "missing-private-id",
        "private-label-canary",
        "private-digest",
        "private-revocation-canary",
        "private-principal",
    ] {
        assert!(!format!("{:?}", report.issues).contains(canary));
    }
    let repeat = store.verify_integrity(VerifyMode::Full).await.unwrap();
    assert_eq!(report.issues, repeat.issues);
    assert_eq!(
        authority_snapshot(&conn),
        before,
        "diagnostics cannot repair authority"
    );
    assert_eq!(fk_count(&conn), 5);
}

#[tokio::test]
async fn full_reports_each_violation_not_just_each_table() {
    let (_dir, store, conn) = fixture();
    conn.execute_batch(
        "INSERT INTO forgotten_facts VALUES
        ('missing-one', 'receipt:one', 'private', 'digest-one', '2026-01-01'),
        ('missing-two', 'receipt:two', 'private', 'digest-two', '2026-01-01');",
    )
    .unwrap();
    assert_eq!(fk_count(&conn), 2);
    let report = store.verify_integrity(VerifyMode::Full).await.unwrap();
    assert!(!report.ok);
    let issues: Vec<_> = report
        .issues
        .iter()
        .filter(|issue| issue.starts_with("SQLite foreign_key_check:"))
        .collect();
    assert_eq!(issues.len(), 2);
    assert_ne!(
        issues[0], issues[1],
        "row identity must distinguish violations"
    );
}

#[tokio::test]
async fn full_handles_without_rowid_child_without_echoing_primary_key() {
    let (_dir, store, conn) = fixture();
    conn.execute_batch(
        "CREATE TABLE diagnostic_child (
        id TEXT PRIMARY KEY, fact_id TEXT REFERENCES facts(id)) WITHOUT ROWID;
        INSERT INTO diagnostic_child VALUES ('private-primary-key', 'private-missing-parent');",
    )
    .unwrap();
    assert_eq!(fk_count(&conn), 1);
    let report = store.verify_integrity(VerifyMode::Full).await.unwrap();
    assert!(!report.ok);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.contains("table=diagnostic_child,")
            && issue.contains("rowid=without-rowid")));
    assert!(!format!("{:?}", report.issues).contains("private-"));
}

#[tokio::test]
async fn full_returns_typed_error_when_fk_check_cannot_run() {
    let (_dir, store, conn) = fixture();
    conn.execute_batch(
        "CREATE TABLE invalid_fk_parent (id TEXT);
        CREATE TABLE invalid_fk_child (id TEXT REFERENCES invalid_fk_parent(id));
        INSERT INTO invalid_fk_child VALUES ('missing');",
    )
    .unwrap();
    assert!(conn.prepare("PRAGMA foreign_key_check").is_err());
    let result = store.verify_integrity(VerifyMode::Full).await;
    assert!(
        matches!(result, Err(MemoryError::Database(_))),
        "{result:?}"
    );
}

#[tokio::test]
async fn clean_full_passes_and_quick_keeps_count_only_scope() {
    let (_dir, store, conn) = fixture();
    assert!(store.verify_integrity(VerifyMode::Full).await.unwrap().ok);
    assert!(store.verify_integrity(VerifyMode::Quick).await.unwrap().ok);
    conn.execute_batch(CORRUPT_AUTHORITY).unwrap();
    assert_eq!(fk_count(&conn), 5);
    // Quick is not a full structural or relational integrity certificate.
    assert!(store.verify_integrity(VerifyMode::Quick).await.unwrap().ok);
}
