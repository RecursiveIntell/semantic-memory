//! HNSW approximate nearest-neighbor index wrapper.
//!
//! SQLite remains the source of truth. The on-disk HNSW files are a recoverable
//! acceleration sidecar that can be rebuilt from SQLite whenever needed.

use crate::db;
use crate::error::MemoryError;
use hnsw_rs::prelude::*;
use rusqlite::params;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

const HNSW_DATA_MAGIC: u32 = 0xa67f0000;

/// Configuration for the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub dimensions: usize,
    pub max_elements: usize,
    pub compaction_threshold: f32,
    pub flush_interval_secs: Option<u64>,
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
            flush_interval_secs: None,
        }
    }
}

/// A single hit from HNSW search.
#[derive(Debug, Clone)]
pub struct HnswHit {
    pub key: String,
    pub distance: f32,
}

impl HnswHit {
    pub fn similarity(&self) -> f32 {
        (1.0 - self.distance).max(0.0)
    }

    /// Split the sidecar key into `(domain, identifier)`.
    pub fn parse_key(&self) -> Result<(&str, &str), MemoryError> {
        self.key
            .split_once(':')
            .ok_or_else(|| MemoryError::InvalidKey(self.key.clone()))
    }
}

struct HnswIndexInner {
    graph: Hnsw<'static, f32, DistCosine>,
    key_to_id: RwLock<HashMap<String, usize>>,
    id_to_key: RwLock<HashMap<usize, String>>,
    next_id: AtomicUsize,
    deleted_ids: RwLock<HashSet<usize>>,
    keymap_dirty: AtomicBool,
    last_flush_epoch: AtomicU64,
    config: HnswConfig,
}

fn current_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Clone)]
pub struct HnswIndex {
    inner: Arc<HnswIndexInner>,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Result<Self, MemoryError> {
        let graph: Hnsw<'static, f32, DistCosine> = Hnsw::new(
            config.m,
            config.max_elements,
            16,
            config.ef_construction,
            DistCosine {},
        );

        Ok(Self {
            inner: Arc::new(HnswIndexInner {
                graph,
                key_to_id: RwLock::new(HashMap::new()),
                id_to_key: RwLock::new(HashMap::new()),
                next_id: AtomicUsize::new(0),
                deleted_ids: RwLock::new(HashSet::new()),
                keymap_dirty: AtomicBool::new(false),
                last_flush_epoch: AtomicU64::new(current_epoch_secs()),
                config,
            }),
        })
    }

    /// Load a previously flushed HNSW sidecar by replaying the dumped vectors.
    ///
    /// This avoids relying on `hnsw_rs`'s borrowing reload API and keeps the safety
    /// boundary purely in safe Rust. Node IDs are preserved so the SQLite keymap can
    /// be loaded afterward.
    pub fn load(dir: &Path, basename: &str, config: HnswConfig) -> Result<Self, MemoryError> {
        let data_path = dir.join(format!("{}.hnsw.data", basename));
        let graph_path = dir.join(format!("{}.hnsw.graph", basename));
        if !data_path.exists() || !graph_path.exists() {
            return Err(MemoryError::HnswError(format!(
                "missing HNSW sidecar files under {}",
                dir.display()
            )));
        }

        let index = Self::new(config)?;
        let max_id = load_vectors_from_sidecar(&index, &data_path)?;
        index
            .inner
            .next_id
            .store(max_id.saturating_add(1), Ordering::SeqCst);
        Ok(index)
    }

    pub fn save(&self, dir: &Path, basename: &str) -> Result<(), MemoryError> {
        self.inner
            .graph
            .file_dump(dir, basename)
            .map_err(|e| MemoryError::HnswError(format!("failed to save HNSW index: {}", e)))?;
        Ok(())
    }

    pub fn insert(&self, key: String, vector: &[f32]) -> Result<(), MemoryError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        self.insert_with_id(Some(key), id, vector)
    }

    pub fn delete(&self, key: &str) -> Result<(), MemoryError> {
        let mut key_to_id = self
            .inner
            .key_to_id
            .write()
            .unwrap_or_else(|e| e.into_inner());
        let mut id_to_key = self
            .inner
            .id_to_key
            .write()
            .unwrap_or_else(|e| e.into_inner());

        if let Some(id) = key_to_id.remove(key) {
            id_to_key.remove(&id);
            self.inner
                .deleted_ids
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(id);
            self.inner.keymap_dirty.store(true, Ordering::Release);
        }
        Ok(())
    }

    pub fn update(&self, key: String, vector: &[f32]) -> Result<(), MemoryError> {
        self.delete(&key)?;
        self.insert(key, vector)
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<HnswHit>, MemoryError> {
        validate_dimensions(query, self.inner.config.dimensions)?;

        if self.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let deleted = self
            .inner
            .deleted_ids
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let fetch_count = top_k.saturating_add(deleted.len());
        drop(deleted);

        let neighbors = self
            .inner
            .graph
            .search(query, fetch_count, self.inner.config.ef_search);

        let deleted = self
            .inner
            .deleted_ids
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let id_to_key = self
            .inner
            .id_to_key
            .read()
            .unwrap_or_else(|e| e.into_inner());

        let mut hits: Vec<HnswHit> = neighbors
            .into_iter()
            .filter(|neighbor| !deleted.contains(&neighbor.d_id))
            .filter_map(|neighbor| {
                id_to_key.get(&neighbor.d_id).map(|key| HnswHit {
                    key: key.clone(),
                    distance: neighbor.distance,
                })
            })
            .take(top_k)
            .collect();

        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    pub fn len(&self) -> usize {
        let total = self.inner.graph.get_nb_point();
        let deleted = self
            .inner
            .deleted_ids
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        total.saturating_sub(deleted)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn deleted_ratio(&self) -> f32 {
        let total = self.inner.graph.get_nb_point();
        if total == 0 {
            return 0.0;
        }
        let deleted = self
            .inner
            .deleted_ids
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        deleted as f32 / total as f32
    }

    pub fn needs_compaction(&self) -> bool {
        self.deleted_ratio() > self.inner.config.compaction_threshold
    }

    pub fn config(&self) -> &HnswConfig {
        &self.inner.config
    }

    pub fn is_keymap_dirty(&self) -> bool {
        self.inner.keymap_dirty.load(Ordering::Acquire)
    }

    pub fn should_flush(&self, interval_secs: u64) -> bool {
        let last = self.inner.last_flush_epoch.load(Ordering::Relaxed);
        current_epoch_secs().saturating_sub(last) >= interval_secs
    }

    pub fn update_last_flush_epoch(&self) {
        self.inner
            .last_flush_epoch
            .store(current_epoch_secs(), Ordering::Relaxed);
    }

    pub fn flush_keymap(&self, conn: &rusqlite::Connection) -> Result<(), MemoryError> {
        if !self.is_keymap_dirty() {
            return Ok(());
        }

        let key_to_id = self
            .inner
            .key_to_id
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let deleted = self
            .inner
            .deleted_ids
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let next_id = self.inner.next_id.load(Ordering::SeqCst);

        db::with_transaction(conn, |tx| {
            tx.execute("DELETE FROM hnsw_keymap", [])?;
            let mut insert_stmt = tx.prepare(
                "INSERT INTO hnsw_keymap (node_id, item_key, deleted) VALUES (?1, ?2, ?3)",
            )?;

            for (key, id) in key_to_id.iter() {
                insert_stmt.execute(params![*id as i64, key, 0])?;
            }
            for id in deleted.iter() {
                insert_stmt.execute(params![*id as i64, format!("_deleted:{}", id), 1])?;
            }
            drop(insert_stmt);

            tx.execute(
                "INSERT INTO hnsw_metadata (key, value) VALUES ('next_id', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![next_id.to_string()],
            )?;
            Ok(())
        })?;

        self.inner.keymap_dirty.store(false, Ordering::Release);
        Ok(())
    }

    pub fn load_keymap(&self, conn: &rusqlite::Connection) -> Result<(), MemoryError> {
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='hnsw_keymap'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);
        if !table_exists {
            tracing::warn!("hnsw_keymap table missing; HNSW keys will remain empty until rebuild");
            return Ok(());
        }

        let mut key_to_id = HashMap::new();
        let mut id_to_key = HashMap::new();
        let mut deleted_ids = HashSet::new();

        let mut stmt = conn.prepare("SELECT node_id, item_key, deleted FROM hnsw_keymap")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)? as usize,
                row.get::<_, String>(1)?,
                row.get::<_, bool>(2)?,
            ))
        })?;

        for row in rows {
            let (node_id, item_key, deleted) = row?;
            if deleted {
                deleted_ids.insert(node_id);
            } else {
                key_to_id.insert(item_key.clone(), node_id);
                id_to_key.insert(node_id, item_key);
            }
        }

        let next_id = conn
            .query_row(
                "SELECT value FROM hnsw_metadata WHERE key = 'next_id'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| self.inner.graph.get_nb_point());

        *self
            .inner
            .key_to_id
            .write()
            .unwrap_or_else(|e| e.into_inner()) = key_to_id;
        *self
            .inner
            .id_to_key
            .write()
            .unwrap_or_else(|e| e.into_inner()) = id_to_key;
        *self
            .inner
            .deleted_ids
            .write()
            .unwrap_or_else(|e| e.into_inner()) = deleted_ids;
        self.inner.next_id.store(next_id, Ordering::SeqCst);
        self.inner.keymap_dirty.store(false, Ordering::Release);

        Ok(())
    }

    fn insert_with_id(
        &self,
        key: Option<String>,
        id: usize,
        vector: &[f32],
    ) -> Result<(), MemoryError> {
        validate_dimensions(vector, self.inner.config.dimensions)?;

        if let Some(key) = key {
            let mut key_to_id = self
                .inner
                .key_to_id
                .write()
                .unwrap_or_else(|e| e.into_inner());
            let mut id_to_key = self
                .inner
                .id_to_key
                .write()
                .unwrap_or_else(|e| e.into_inner());

            if let Some(old_id) = key_to_id.insert(key.clone(), id) {
                id_to_key.remove(&old_id);
                self.inner
                    .deleted_ids
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(old_id);
            }
            id_to_key.insert(id, key);
            self.inner.keymap_dirty.store(true, Ordering::Release);
        }

        self.inner.graph.insert((vector, id));
        Ok(())
    }
}

fn validate_dimensions(vector: &[f32], expected: usize) -> Result<(), MemoryError> {
    if vector.len() != expected {
        return Err(MemoryError::HnswError(format!(
            "expected {} dimensions, got {}",
            expected,
            vector.len()
        )));
    }
    Ok(())
}

fn load_vectors_from_sidecar(index: &HnswIndex, data_path: &Path) -> Result<usize, MemoryError> {
    let mut file = File::open(data_path).map_err(|e| {
        MemoryError::HnswError(format!("failed to open {}: {}", data_path.display(), e))
    })?;

    let mut u32_buf = [0u8; 4];
    file.read_exact(&mut u32_buf).map_err(|e| {
        MemoryError::HnswError(format!("failed to read HNSW sidecar header: {}", e))
    })?;
    if u32::from_ne_bytes(u32_buf) != HNSW_DATA_MAGIC {
        return Err(MemoryError::HnswError(
            "invalid HNSW data file magic header".to_string(),
        ));
    }

    let mut usize_buf = [0u8; std::mem::size_of::<usize>()];
    file.read_exact(&mut usize_buf).map_err(|e| {
        MemoryError::HnswError(format!("failed to read HNSW sidecar dimensions: {}", e))
    })?;
    let dims = usize::from_ne_bytes(usize_buf);
    if dims != index.inner.config.dimensions {
        return Err(MemoryError::HnswError(format!(
            "HNSW sidecar dimensions {} do not match configured {}",
            dims, index.inner.config.dimensions
        )));
    }

    let mut max_id = 0usize;

    loop {
        match file.read_exact(&mut u32_buf) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => {
                return Err(MemoryError::HnswError(format!(
                    "failed while reading HNSW sidecar entry header: {}",
                    err
                )))
            }
        }

        if u32::from_ne_bytes(u32_buf) != HNSW_DATA_MAGIC {
            return Err(MemoryError::HnswError(
                "invalid per-vector HNSW data magic".to_string(),
            ));
        }

        let mut u64_buf = [0u8; 8];
        file.read_exact(&mut u64_buf).map_err(|e| {
            MemoryError::HnswError(format!("failed to read HNSW sidecar node id: {}", e))
        })?;
        let id = u64::from_ne_bytes(u64_buf) as usize;

        file.read_exact(&mut u64_buf).map_err(|e| {
            MemoryError::HnswError(format!("failed to read HNSW sidecar vector length: {}", e))
        })?;
        let byte_len = u64::from_ne_bytes(u64_buf) as usize;
        let mut raw = vec![0u8; byte_len];
        file.read_exact(&mut raw).map_err(|e| {
            MemoryError::HnswError(format!("failed to read HNSW sidecar payload: {}", e))
        })?;

        let vector = db::bytes_to_embedding(&raw)?;
        index.insert_with_id(None, id, &vector)?;
        max_id = max_id.max(id);
    }

    Ok(max_id)
}
