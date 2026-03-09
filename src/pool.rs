//! Small SQLite connection pool tailored to rusqlite.
//!
//! SQLite still serializes writers at the database level, so the pool keeps a
//! single writer connection plus a bounded set of independent reader
//! connections. Under WAL mode, those readers can proceed concurrently while
//! the writer is idle or between write transactions.

use crate::config::{MemoryLimits, PoolConfig};
use crate::db;
use crate::error::MemoryError;
use rusqlite::Connection;
use std::path::Path;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Default reader acquisition timeout (30 seconds).
const DEFAULT_READER_TIMEOUT: Duration = Duration::from_secs(30);

/// Internal SQLite pool with one writer connection and N reader connections.
pub(crate) struct SqlitePool {
    writer: Mutex<Connection>,
    readers: Vec<Mutex<Connection>>,
    available_readers: Mutex<Vec<usize>>,
    available_cv: Condvar,
    reader_count: usize,
    reader_timeout: Duration,
}

/// RAII guard that returns a reader index to the pool on drop.
///
/// This ensures the reader slot is always returned even if the closure panics.
struct ReaderGuard<'a> {
    pool: &'a SqlitePool,
    idx: Option<usize>,
}

impl<'a> ReaderGuard<'a> {
    fn new(pool: &'a SqlitePool, idx: usize) -> Self {
        Self {
            pool,
            idx: Some(idx),
        }
    }
}

impl Drop for ReaderGuard<'_> {
    fn drop(&mut self) {
        if let Some(idx) = self.idx.take() {
            let mut available = self
                .pool
                .available_readers
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            available.push(idx);
            self.pool.available_cv.notify_one();
        }
    }
}

impl SqlitePool {
    /// Open a writer connection, run migrations once, and populate the reader pool.
    pub(crate) fn open(
        path: &Path,
        pool: &PoolConfig,
        limits: &MemoryLimits,
    ) -> Result<Self, MemoryError> {
        let writer = db::open_database(path, pool, limits)?;
        let mut readers = Vec::with_capacity(pool.max_read_connections);
        for _ in 0..pool.max_read_connections {
            readers.push(Mutex::new(db::open_pool_member_connection(
                path, pool, limits, true,
            )?));
        }

        let reader_count = pool.max_read_connections;
        let reader_timeout = if pool.reader_timeout_secs > 0 {
            Duration::from_secs(pool.reader_timeout_secs)
        } else {
            DEFAULT_READER_TIMEOUT
        };
        Ok(Self {
            writer: Mutex::new(writer),
            readers,
            available_readers: Mutex::new((0..reader_count).rev().collect()),
            available_cv: Condvar::new(),
            reader_count,
            reader_timeout,
        })
    }

    /// Run work against the single writer connection.
    pub(crate) fn with_write_conn<F, T>(&self, f: F) -> Result<T, MemoryError>
    where
        F: FnOnce(&Connection) -> Result<T, MemoryError>,
    {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        f(&conn)
    }

    /// Run work against one reader connection with bounded wait time.
    ///
    /// Reader slots are returned via an RAII guard, so even if `f` panics
    /// the slot is returned to the available pool (preventing permanent leaks).
    pub(crate) fn with_read_conn<F, T>(&self, f: F) -> Result<T, MemoryError>
    where
        F: FnOnce(&Connection) -> Result<T, MemoryError>,
    {
        let start = Instant::now();
        let reader_idx = {
            let mut available = self
                .available_readers
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(idx) = available.pop() {
                    break idx;
                }
                let (new_available, wait_result) = self
                    .available_cv
                    .wait_timeout(available, self.reader_timeout)
                    .unwrap_or_else(|e| e.into_inner());
                available = new_available;
                if wait_result.timed_out() {
                    if let Some(idx) = available.pop() {
                        break idx;
                    }
                    let elapsed = start.elapsed();
                    tracing::error!(
                        elapsed_ms = elapsed.as_millis() as u64,
                        pool_size = self.reader_count,
                        "Reader pool acquisition timed out"
                    );
                    return Err(MemoryError::PoolTimeout {
                        elapsed_ms: elapsed.as_millis() as u64,
                        pool_size: self.reader_count,
                    });
                }
            }
        };

        let wait_duration = start.elapsed();
        if wait_duration > Duration::from_millis(100) {
            tracing::warn!(
                wait_ms = wait_duration.as_millis() as u64,
                pool_size = self.reader_count,
                "Reader pool contention: waited {}ms for a reader slot",
                wait_duration.as_millis()
            );
        }

        // RAII guard ensures the reader index is returned even on panic
        let _guard = ReaderGuard::new(self, reader_idx);
        let conn = self.readers[reader_idx]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        f(&conn)
    }
}
