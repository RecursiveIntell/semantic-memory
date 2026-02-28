//! HNSW approximate nearest neighbor index wrapper.
//!
//! Wraps `hnsw_rs` to provide a string-keyed HNSW index with cosine distance.
//! The internal index uses `usize` keys, with bidirectional hash maps translating
//! between string keys (e.g., `"fact:42"`) and numeric IDs.
//!
//! Gated behind `#[cfg(feature = "hnsw")]`.

use crate::error::MemoryError;

use hnsw_rs::prelude::*;
use rusqlite::params;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Configuration for the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max connections per node per layer. Higher = better recall, more memory.
    pub m: usize,
    /// Width of search during index construction.
    pub ef_construction: usize,
    /// Width of search during queries. Must be >= top_k.
    pub ef_search: usize,
    /// Embedding dimensionality. Must match the embedder output.
    pub dimensions: usize,
    /// Maximum number of elements the index can hold.
    pub max_elements: usize,
    /// Ratio of deleted/total above which compaction is recommended.
    /// Default: 0.3 (30%)
    pub compaction_threshold: f32,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            dimensions: 768,
            max_elements: 100_000,
            compaction_threshold: 0.3,
        }
    }
}

/// A single hit from HNSW search.
#[derive(Debug, Clone)]
pub struct HnswHit {
    /// The key that was inserted (e.g., "fact:42").
    pub key: String,
    /// Distance from query (lower = more similar for cosine distance).
    pub distance: f32,
}

impl HnswHit {
    /// Convert distance to similarity score in [0, 1] range.
    pub fn similarity(&self) -> f32 {
        (1.0 - self.distance).max(0.0)
    }

    /// Parse the domain and numeric ID from the key.
    /// "fact:42" → ("fact", 42)
    pub fn parse_key(&self) -> Result<(&str, i64), MemoryError> {
        let (domain, id_str) = self
            .key
            .split_once(':')
            .ok_or_else(|| MemoryError::InvalidKey(self.key.clone()))?;
        let id = id_str
            .parse::<i64>()
            .map_err(|_| MemoryError::InvalidKey(self.key.clone()))?;
        Ok((domain, id))
    }
}

/// Thread-safe inner state of the HNSW index.
struct HnswIndexInner {
    graph: Hnsw<'static, f32, DistCosine>,
    /// Kept alive so the leaked Box from load is reclaimed on drop.
    /// In practice, hnsw_rs copies all data during load, so this is
    /// belt-and-suspenders — the graph doesn't borrow from the reloader.
    _reloader_keepalive: Option<Box<HnswIo>>,
    /// String key → usize ID mapping.
    key_to_id: RwLock<HashMap<String, usize>>,
    /// usize ID → String key mapping.
    id_to_key: RwLock<HashMap<usize, String>>,
    /// Monotonically increasing ID counter.
    next_id: AtomicUsize,
    /// Set of deleted IDs (hnsw_rs doesn't support native delete).
    deleted_ids: RwLock<std::collections::HashSet<usize>>,
    /// Whether the keymap has been modified since last flush.
    keymap_dirty: AtomicBool,
    config: HnswConfig,
}

/// Wrapper around hnsw_rs providing semantic-memory's HNSW operations.
///
/// Thread-safe: can be shared across async tasks via Arc.
/// Clone is cheap (Arc internals).
#[derive(Clone)]
pub struct HnswIndex {
    inner: Arc<HnswIndexInner>,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(config: HnswConfig) -> Result<Self, MemoryError> {
        let graph: Hnsw<'static, f32, DistCosine> = Hnsw::new(
            config.m,
            config.max_elements,
            16, // max layers
            config.ef_construction,
            DistCosine {},
        );

        Ok(Self {
            inner: Arc::new(HnswIndexInner {
                graph,
                _reloader_keepalive: None,
                key_to_id: RwLock::new(HashMap::new()),
                id_to_key: RwLock::new(HashMap::new()),
                next_id: AtomicUsize::new(0),
                deleted_ids: RwLock::new(std::collections::HashSet::new()),
                keymap_dirty: AtomicBool::new(false),
                config,
            }),
        })
    }

    /// Load an existing HNSW index from disk.
    ///
    /// The `HnswIo` reloader is kept alive in `_reloader_keepalive` to prevent
    /// memory leaks. The `'static` lifetime on `Hnsw` is safe because hnsw_rs
    /// copies all data into memory during load (no mmap by default), so the
    /// graph doesn't actually borrow from the reloader post-construction.
    ///
    /// Key mappings are NOT persisted by hnsw_rs. After loading, the caller
    /// must load mappings from SQLite (via `load_keymap`).
    pub fn load(
        dir: &Path,
        basename: &str,
        config: HnswConfig,
    ) -> Result<Self, MemoryError> {
        let mut reloader = Box::new(HnswIo::new(dir, basename));

        // SAFETY: hnsw_rs copies all data during load_hnsw(), so the graph
        // does not hold references to reloader memory after construction.
        // The 'static lifetime is a lie we tell the compiler; the real
        // invariant is enforced by keeping _reloader_keepalive alive.
        let reloader_ref: &'static mut HnswIo =
            unsafe { &mut *(reloader.as_mut() as *mut HnswIo) };
        let graph: Hnsw<'static, f32, DistCosine> = reloader_ref
            .load_hnsw()
            .map_err(|e| MemoryError::HnswError(format!("Failed to load HNSW index: {}", e)))?;

        let nb_point = graph.get_nb_point();

        Ok(Self {
            inner: Arc::new(HnswIndexInner {
                graph,
                _reloader_keepalive: Some(reloader),
                key_to_id: RwLock::new(HashMap::new()),
                id_to_key: RwLock::new(HashMap::new()),
                next_id: AtomicUsize::new(nb_point),
                deleted_ids: RwLock::new(std::collections::HashSet::new()),
                keymap_dirty: AtomicBool::new(false),
                config,
            }),
        })
    }

    /// Save the HNSW index to disk.
    pub fn save(&self, dir: &Path, basename: &str) -> Result<(), MemoryError> {
        self.inner
            .graph
            .file_dump(dir, basename)
            .map_err(|e| MemoryError::HnswError(format!("Failed to save HNSW index: {}", e)))?;
        Ok(())
    }

    /// Insert a vector with a string key.
    ///
    /// Key format: `"{domain}:{id}"` e.g., `"fact:42"`, `"chunk:17"`, `"msg:99"`.
    pub fn insert(&self, key: String, vector: &[f32]) -> Result<(), MemoryError> {
        if vector.len() != self.inner.config.dimensions {
            return Err(MemoryError::HnswError(format!(
                "expected {} dimensions, got {}",
                self.inner.config.dimensions,
                vector.len()
            )));
        }

        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);

        // Update mappings
        {
            let mut k2i = self.inner.key_to_id.write().unwrap();
            let mut i2k = self.inner.id_to_key.write().unwrap();
            // If key already exists, mark old id as deleted
            if let Some(old_id) = k2i.insert(key.clone(), id) {
                i2k.remove(&old_id);
                self.inner.deleted_ids.write().unwrap().insert(old_id);
            }
            i2k.insert(id, key);
        }

        // Insert into HNSW graph
        self.inner.graph.insert((vector, id));
        self.inner.keymap_dirty.store(true, Ordering::Release);

        Ok(())
    }

    /// Remove a vector by key.
    ///
    /// Since hnsw_rs doesn't support native deletion, we mark the ID as deleted
    /// and filter it out during search.
    pub fn delete(&self, key: &str) -> Result<(), MemoryError> {
        let mut k2i = self.inner.key_to_id.write().unwrap();
        let mut i2k = self.inner.id_to_key.write().unwrap();

        if let Some(id) = k2i.remove(key) {
            i2k.remove(&id);
            self.inner.deleted_ids.write().unwrap().insert(id);
            self.inner.keymap_dirty.store(true, Ordering::Release);
        }
        // Not an error if key doesn't exist — idempotent delete
        Ok(())
    }

    /// Update a vector (delete old + insert new).
    pub fn update(&self, key: String, vector: &[f32]) -> Result<(), MemoryError> {
        self.delete(&key)?;
        self.insert(key, vector)
    }

    /// Search for nearest neighbors.
    ///
    /// The query vector is f32. Returns keys and distances.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<HnswHit>, MemoryError> {
        if query.len() != self.inner.config.dimensions {
            return Err(MemoryError::HnswError(format!(
                "query has {} dimensions, expected {}",
                query.len(),
                self.inner.config.dimensions
            )));
        }

        if self.is_empty() {
            return Ok(Vec::new());
        }

        // Over-fetch to account for deleted items
        let deleted = self.inner.deleted_ids.read().unwrap();
        let fetch_count = top_k + deleted.len();
        drop(deleted);

        let neighbors = self.inner.graph.search(
            query,
            fetch_count,
            self.inner.config.ef_search,
        );

        let deleted = self.inner.deleted_ids.read().unwrap();
        let i2k = self.inner.id_to_key.read().unwrap();

        let mut hits: Vec<HnswHit> = neighbors
            .into_iter()
            .filter(|n| !deleted.contains(&n.d_id))
            .filter_map(|n| {
                i2k.get(&n.d_id).map(|key| HnswHit {
                    key: key.clone(),
                    distance: n.distance,
                })
            })
            .take(top_k)
            .collect();

        // Sort by distance ascending (closest first)
        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        Ok(hits)
    }

    /// Number of active (non-deleted) vectors in the index.
    pub fn len(&self) -> usize {
        let total = self.inner.graph.get_nb_point();
        let deleted = self.inner.deleted_ids.read().unwrap().len();
        total.saturating_sub(deleted)
    }

    /// Whether the index has no active vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ratio of deleted nodes to total nodes in the graph.
    pub fn deleted_ratio(&self) -> f32 {
        let total = self.inner.graph.get_nb_point();
        if total == 0 {
            return 0.0;
        }
        let deleted = self.inner.deleted_ids.read().unwrap().len();
        deleted as f32 / total as f32
    }

    /// Returns true if compaction is recommended (deleted ratio exceeds threshold).
    pub fn needs_compaction(&self) -> bool {
        self.deleted_ratio() > self.inner.config.compaction_threshold
    }

    /// Get the config.
    pub fn config(&self) -> &HnswConfig {
        &self.inner.config
    }

    /// Whether the keymap has been modified since last flush.
    pub fn is_keymap_dirty(&self) -> bool {
        self.inner.keymap_dirty.load(Ordering::Acquire)
    }

    /// Flush key mappings to SQLite. Called during MemoryStore::drop and flush_hnsw.
    pub fn flush_keymap(&self, conn: &rusqlite::Connection) -> Result<(), MemoryError> {
        if !self.is_keymap_dirty() {
            return Ok(());
        }

        let k2i = self.inner.key_to_id.read().unwrap();
        let deleted = self.inner.deleted_ids.read().unwrap();
        let next_id = self.inner.next_id.load(Ordering::SeqCst);

        crate::db::with_transaction(conn, |tx| {
            // Clear and rewrite all keymap entries
            tx.execute("DELETE FROM hnsw_keymap", [])?;

            let mut insert_stmt = tx.prepare(
                "INSERT INTO hnsw_keymap (node_id, item_key, deleted) VALUES (?1, ?2, ?3)",
            )?;

            // Write active mappings
            for (key, &id) in k2i.iter() {
                insert_stmt.execute(params![id as i64, key, 0])?;
            }

            // Write deleted entries
            for &id in deleted.iter() {
                // Deleted entries don't have a key in k2i, use a placeholder
                insert_stmt.execute(params![id as i64, format!("_deleted:{}", id), 1])?;
            }

            drop(insert_stmt);

            // Update next_id
            tx.execute(
                "INSERT OR REPLACE INTO hnsw_metadata (key, value) VALUES ('next_id', ?1)",
                params![next_id.to_string()],
            )?;

            Ok(())
        })?;

        self.inner.keymap_dirty.store(false, Ordering::Release);
        Ok(())
    }

    /// Load key mappings from SQLite into in-memory maps.
    pub fn load_keymap(&self, conn: &rusqlite::Connection) -> Result<(), MemoryError> {
        let mut key_to_id = HashMap::new();
        let mut id_to_key = HashMap::new();
        let mut deleted_ids = std::collections::HashSet::new();

        // Check if hnsw_keymap table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='hnsw_keymap'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !table_exists {
            tracing::warn!("hnsw_keymap table not found — key mappings will be empty until rebuild");
            return Ok(());
        }

        let mut stmt = conn.prepare(
            "SELECT node_id, item_key, deleted FROM hnsw_keymap",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)? as usize,
                row.get::<_, String>(1)?,
                row.get::<_, bool>(2)?,
            ))
        })?;

        for row in rows {
            let (node_id, key, is_deleted) = row?;
            if is_deleted {
                deleted_ids.insert(node_id);
            } else {
                key_to_id.insert(key.clone(), node_id);
                id_to_key.insert(node_id, key);
            }
        }

        // Load next_id from metadata
        let next_id: usize = conn
            .query_row(
                "SELECT value FROM hnsw_metadata WHERE key = 'next_id'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| self.inner.graph.get_nb_point());

        // Apply loaded data
        *self.inner.key_to_id.write().unwrap() = key_to_id;
        *self.inner.id_to_key.write().unwrap() = id_to_key;
        *self.inner.deleted_ids.write().unwrap() = deleted_ids;
        self.inner.next_id.store(next_id, Ordering::SeqCst);
        self.inner.keymap_dirty.store(false, Ordering::Release);

        tracing::info!(
            active = self.inner.key_to_id.read().unwrap().len(),
            deleted = self.inner.deleted_ids.read().unwrap().len(),
            next_id = next_id,
            "Loaded HNSW key mappings from SQLite"
        );

        Ok(())
    }
}
