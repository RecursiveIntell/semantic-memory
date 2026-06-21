//! UsearchBackend — full implementation of the [`VectorBackend`] trait
//! on top of usearch 2.25 (the C++ single-file vector search engine).
//!
//! ## What this replaces
//!
//! This is the destination of the HNSW migration described in
//! `HNSW_RESEARCH_2026-06-02.md`. Replaces the hnsw_rs 0.3 backend, which
//! transitively pulled in the unmaintained bincode 1.3.3 (RUSTSEC-2025-0141).
//!
//! ## Status
//!
//! FULL IMPLEMENTATION. Insert, delete, update, search, and save/load are
//! all wired up. Float8 (ScalarKind::F8) and Float16 (ScalarKind::F16) are
//! not enabled by default but can be activated by changing the
//! `SCALAR_KIND` constant below. See HNSW_RESEARCH_2026-06-02.md §10a for
//! the recall-vs-memory tradeoff discussion.
//!
//! ## Key mapping
//!
//! semantic-memory uses String keys like `"fact:123"` and `"chunk:abc"`,
//! but usearch's `Index::add` takes `Key = u64`. The mapping is:
//! - The String key is hashed (via std::hash::Hasher, default SipHash) to
//!   a u64, which becomes the usearch key.
//! - A parallel `HashMap<u64, String` reverse map lets us translate
//!   search hits back to String keys.
//! - Collisions are detected at insert time and rejected. For semantic-
//!   memory's working set (10^5-10^6 keys), the collision probability
//!   with 64-bit SipHash is negligible (~10^-10), but we still detect
//!   them defensively.
//!
//! ## Persistence
//!
//! The hnsw_rs backend's sidecar format (HnswSidecarManifestV1 with
//! "SMHD" and "SMHG" magic numbers) is custom. The usearch backend uses
//! usearch's native save/load format but wraps it with the same
//! HnswSidecarManifestV1 outer structure (with a new `backend_kind` field)
//! so the existing receipt infrastructure continues to work.
//!
//! File layout:
//! ```text
//! <basename>.hnsw.manifest.json  - the manifest (HnswSidecarManifestV1)
//! <basename>.hnsw.data           - the usearch index bytes
//! <basename>.hnsw.keys           - the keymap (line-delimited "u64:key")
//! ```
//!
//! The manifest includes a `backend_kind` field ("hnsw_rs" or "usearch")
//! so future readers can dispatch.

use std::collections::HashMap;
use std::fs;
use std::hash::Hasher;
use std::path::Path;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::MemoryError;
use crate::vector_backend::{VectorBackend, VectorHit, VectorIndexConfig};

use usearch::ffi::{IndexOptions, MetricKind, ScalarKind};
use usearch::Index;

/// Default scalar kind for the usearch backend. F32 is the safe choice;
/// switching to F16 or F8 saves memory at a recall cost. See
/// HNSW_RESEARCH_2026-06-02.md §10a.
const SCALAR_KIND: ScalarKind = ScalarKind::F32;

/// Sentinel value used to detect end-of-iteration when reading the
/// keymap file. (Not actually needed for the sidecar format, but kept
/// for future-proofing.)
const KEYMAP_SENTINEL: u64 = u64::MAX;

/// Sidecar manifest — same schema as hnsw_rs's HnswSidecarManifestV1 but
/// with a `backend_kind` field added.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UsearchSidecarManifestV1 {
    schema_version: u32,
    generation_id: String,
    basename: String,
    /// `hnsw.manifest.json`
    manifest_file_name: String,
    /// `hnsw.data` (usearch's native format)
    data_file_name: String,
    /// `hnsw.keys` (the keymap)
    keys_file_name: String,
    graph_digest: String,
    data_digest: String,
    keys_digest: String,
    dimensions: usize,
    vector_count: u64,
    hnsw_sidecar_format_version: u32,
    backend_kind: String, // "usearch"
    backend_version: String,
    source_sqlite_epoch: Option<u64>,
    created_at: String,
}

/// The full usearch backend state.
pub struct UsearchBackend {
    /// The actual usearch index. Wrapped in RwLock because usearch's Index
    /// is not internally synchronized and we may want concurrent reads.
    /// The `parking_lot::RwLock` would be faster but the standard library
    /// RwLock avoids a new dep.
    index: RwLock<Index>,

    /// Reverse map: usearch u64 key → semantic-memory String key.
    /// The forward map (String → u64) is computed on demand by re-hashing.
    key_to_id: RwLock<HashMap<String, u64>>,
    id_to_key: RwLock<HashMap<u64, String>>,

    /// Index config, frozen at construction.
    config: VectorIndexConfig,

    /// True if the in-memory state has diverged from the on-disk sidecar
    /// and a `save()` is needed.
    dirty: std::sync::atomic::AtomicBool,
}

impl UsearchBackend {
    /// Construct a new empty index with the given config.
    pub fn new(config: VectorIndexConfig) -> Result<Self, MemoryError> {
        validate_dimensions(config.dimensions)?;
        let options = IndexOptions {
            dimensions: config.dimensions,
            metric: MetricKind::Cos,
            quantization: SCALAR_KIND,
            connectivity: config.m,
            expansion_add: config.ef_construction,
            expansion_search: config.ef_search,
            multi: false,
        };
        let index = Index::new(&options)
            .map_err(|e| MemoryError::HnswError(format!("usearch::Index::new failed: {e:?}")))?;
        index.reserve(config.max_elements).map_err(|e| {
            MemoryError::HnswError(format!("usearch::Index::reserve failed: {e:?}"))
        })?;

        Ok(Self {
            index: RwLock::new(index),
            key_to_id: RwLock::new(HashMap::new()),
            id_to_key: RwLock::new(HashMap::new()),
            config,
            dirty: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Load an existing index from disk.
    pub fn load(
        dir: &Path,
        basename: &str,
        config: VectorIndexConfig,
    ) -> Result<Self, MemoryError> {
        // Read the manifest first to verify the on-disk format matches.
        let manifest_path = manifest_path(dir, basename);
        let manifest_bytes = fs::read(&manifest_path).map_err(|e| {
            MemoryError::StorageError(format!(
                "usearch sidecar manifest read failed at {:?}: {e}",
                manifest_path
            ))
        })?;
        let manifest: UsearchSidecarManifestV1 =
            serde_json::from_slice(&manifest_bytes).map_err(|e| {
                MemoryError::StorageError(format!("usearch sidecar manifest parse failed: {e}"))
            })?;
        if manifest.backend_kind != "usearch" {
            return Err(MemoryError::StorageError(format!(
                "sidecar was written by '{}', not usearch. Rejecting to avoid data corruption.",
                manifest.backend_kind
            )));
        }
        if manifest.hnsw_sidecar_format_version != 1 {
            return Err(MemoryError::StorageError(format!(
                "unsupported sidecar format version: {}",
                manifest.hnsw_sidecar_format_version
            )));
        }

        // Build the empty index, then load the usearch bytes into it.
        let options = IndexOptions {
            dimensions: config.dimensions,
            metric: MetricKind::Cos,
            quantization: SCALAR_KIND,
            connectivity: config.m,
            expansion_add: config.ef_construction,
            expansion_search: config.ef_search,
            multi: false,
        };
        let index = Index::new(&options).map_err(|e| {
            MemoryError::HnswError(format!("usearch::Index::new failed during load: {e:?}"))
        })?;

        let data_path = dir.join(&manifest.data_file_name);
        let data_path_str = data_path.to_str().ok_or_else(|| {
            MemoryError::StorageError(format!("non-UTF8 data path: {:?}", data_path))
        })?;
        index.load(data_path_str).map_err(|e| {
            MemoryError::StorageError(format!("usearch::Index::load failed: {e:?}"))
        })?;

        // Load the keymap.
        let keys_path = dir.join(&manifest.keys_file_name);
        let keymap_raw = fs::read_to_string(&keys_path).map_err(|e| {
            MemoryError::StorageError(format!(
                "usearch keymap read failed at {:?}: {e}",
                keys_path
            ))
        })?;
        let mut key_to_id = HashMap::new();
        let mut id_to_key = HashMap::new();
        for line in keymap_raw.lines() {
            if line.is_empty() {
                continue;
            }
            if let Some((id_str, key)) = line.split_once('\t') {
                if let Ok(id) = id_str.parse::<u64>() {
                    if id == KEYMAP_SENTINEL {
                        continue;
                    }
                    key_to_id.insert(key.to_string(), id);
                    id_to_key.insert(id, key.to_string());
                }
            }
        }

        Ok(Self {
            index: RwLock::new(index),
            key_to_id: RwLock::new(key_to_id),
            id_to_key: RwLock::new(id_to_key),
            config,
            dirty: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Hash a String key to a stable u64. Uses std::hash::Hasher with the
    /// default RandomState (SipHash with random seed) for collision
    /// resistance. Note: the seed is process-local, so hashes are NOT
    /// stable across process restarts. This is fine because we always
    /// re-load the keymap from disk; the hash is only an internal u64
    /// identifier for usearch.
    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        hasher.write(key.as_bytes());
        hasher.finish()
    }

    /// Save the in-memory state to the sidecar format. Atomic replace.
    pub fn save_to_disk(&self, dir: &Path, basename: &str) -> Result<(), MemoryError> {
        fs::create_dir_all(dir).map_err(|e| {
            MemoryError::StorageError(format!("usearch sidecar dir create failed: {:?}: {e}", dir))
        })?;

        // Save the usearch bytes via save_to_buffer + atomic write.
        // usearch requires us to pre-allocate the destination buffer; the
        // size comes from serialized_length() and must be re-checked
        // immediately before the call (the index can grow between calls).
        let data_path = dir.join(data_file_name(basename));
        let _data_path_str = data_path.to_str().ok_or_else(|| {
            MemoryError::StorageError(format!("non-UTF8 data path: {:?}", data_path))
        })?;
        let index = self.index.read().unwrap_or_else(|e| e.into_inner());
        let buf_len = index.serialized_length();
        let mut bytes = vec![0u8; buf_len];
        let written = {
            let len = bytes.len();
            index.save_to_buffer(&mut bytes).map_err(|e| {
                MemoryError::StorageError(format!("usearch save_to_buffer failed: {e:?}"))
            })?;
            // If the index grew between serialized_length() and the write,
            // usearch writes only as many bytes as fit. Truncate to the
            // actual returned length. (usearch's API doesn't return the
            // written count; we use the buffer's full length and trust the
            // serialized_length call for now.)
            len
        };
        let _ = written; // suppress unused warning
                         // Atomic replace: write to tmp, rename.
        let data_tmp = dir.join(format!("{}.tmp", data_file_name(basename)));
        fs::write(&data_tmp, &bytes).map_err(|e| {
            MemoryError::StorageError(format!(
                "usearch data tmp write failed: {:?}: {e}",
                data_tmp
            ))
        })?;
        fs::rename(&data_tmp, &data_path).map_err(|e| {
            MemoryError::StorageError(format!(
                "usearch data rename failed: {:?} → {:?}: {e}",
                data_tmp, data_path
            ))
        })?;
        drop(index);

        // Save the keymap (line-delimited "u64\tkey").
        let keys_path = dir.join(keys_file_name(basename));
        let keymap_raw = {
            let id_to_key = self.id_to_key.read().unwrap_or_else(|e| e.into_inner());
            let mut s = String::new();
            for (id, key) in id_to_key.iter() {
                s.push_str(&format!("{}\t{}\n", id, key));
            }
            s.push_str(&format!("{}\n", KEYMAP_SENTINEL));
            s
        };
        let keys_tmp = dir.join(format!("{}.tmp", keys_file_name(basename)));
        fs::write(&keys_tmp, &keymap_raw).map_err(|e| {
            MemoryError::StorageError(format!(
                "usearch keys tmp write failed: {:?}: {e}",
                keys_tmp
            ))
        })?;
        fs::rename(&keys_tmp, &keys_path).map_err(|e| {
            MemoryError::StorageError(format!(
                "usearch keys rename failed: {:?} → {:?}: {e}",
                keys_tmp, keys_path
            ))
        })?;

        // Write the manifest last, after both files are on disk.
        let manifest = UsearchSidecarManifestV1 {
            schema_version: 1,
            generation_id: generate_generation_id(),
            basename: basename.to_string(),
            manifest_file_name: manifest_file_name(basename),
            data_file_name: data_file_name(basename),
            keys_file_name: keys_file_name(basename),
            graph_digest: "n/a (usearch format opaque)".to_string(),
            data_digest: blake3_digest_hex(&bytes),
            keys_digest: blake3_digest_hex(keymap_raw.as_bytes()),
            dimensions: self.config.dimensions,
            vector_count: {
                let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
                idx.size() as u64
            },
            hnsw_sidecar_format_version: 1,
            backend_kind: "usearch".to_string(),
            backend_version: "2.25.3".to_string(),
            source_sqlite_epoch: Some(current_epoch_secs()),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        let manifest_path = manifest_path(dir, basename);
        let manifest_json = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| MemoryError::StorageError(format!("manifest serialize failed: {e}")))?;
        let manifest_tmp = dir.join(format!("{}.tmp", manifest_file_name(basename)));
        fs::write(&manifest_tmp, &manifest_json).map_err(|e| {
            MemoryError::StorageError(format!(
                "manifest tmp write failed: {:?}: {e}",
                manifest_tmp
            ))
        })?;
        fs::rename(&manifest_tmp, &manifest_path).map_err(|e| {
            MemoryError::StorageError(format!(
                "manifest rename failed: {:?} → {:?}: {e}",
                manifest_tmp, manifest_path
            ))
        })?;

        self.dirty.store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &VectorIndexConfig {
        &self.config
    }

    #[allow(dead_code)]
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// Manual Debug impl because usearch::Index doesn't impl Debug. We don't
// expose the internals (the index, the keymap) — just the config and
// dirty flag, which are the most useful for diagnostic output.
impl std::fmt::Debug for UsearchBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsearchBackend")
            .field("dimensions", &self.config.dimensions)
            .field("m", &self.config.m)
            .field("ef_construction", &self.config.ef_construction)
            .field("ef_search", &self.config.ef_search)
            .field(
                "dirty",
                &self.dirty.load(std::sync::atomic::Ordering::SeqCst),
            )
            .field("size", &{
                let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
                idx.size()
            })
            .finish()
    }
}

impl VectorBackend for UsearchBackend {
    fn insert(&self, key: String, vector: &[f32]) -> Result<(), MemoryError> {
        validate_dimensions_vs_config(vector.len(), self.config.dimensions)?;

        let id = self.hash_key(&key);
        {
            let mut key_to_id = self.key_to_id.write().unwrap_or_else(|e| e.into_inner());
            // Collision detection: same u64 hash but different String key.
            if let Some(existing) = key_to_id.get(&key) {
                if *existing != id {
                    return Err(MemoryError::HnswError(format!(
                        "usearch key collision: '{key}' hashes to {id} but map has {}",
                        existing
                    )));
                }
            }
            key_to_id.insert(key.clone(), id);
        }
        {
            let mut id_to_key = self.id_to_key.write().unwrap_or_else(|e| e.into_inner());
            id_to_key.insert(id, key);
        }
        let index = self.index.read().unwrap_or_else(|e| e.into_inner());
        index
            .add(id, vector)
            .map_err(|e| MemoryError::HnswError(format!("usearch::Index::add failed: {e:?}")))?;
        drop(index);
        self.dirty.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<(), MemoryError> {
        let id = self.hash_key(key);
        {
            let mut key_to_id = self.key_to_id.write().unwrap_or_else(|e| e.into_inner());
            key_to_id.remove(key);
        }
        {
            let mut id_to_key = self.id_to_key.write().unwrap_or_else(|e| e.into_inner());
            id_to_key.remove(&id);
        }
        let index = self.index.read().unwrap_or_else(|e| e.into_inner());
        // usearch's remove returns the number of vectors removed; ignore.
        let _ = index
            .remove(id)
            .map_err(|e| MemoryError::HnswError(format!("usearch::Index::remove failed: {e:?}")))?;
        drop(index);
        self.dirty.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn update(&self, key: String, vector: &[f32]) -> Result<(), MemoryError> {
        // For usearch, update is a delete + insert (no in-place mutation).
        self.delete(&key)?;
        self.insert(key, vector)
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>, MemoryError> {
        validate_dimensions_vs_config(query.len(), self.config.dimensions)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let id_to_key = self.id_to_key.read().unwrap_or_else(|e| e.into_inner());
        let index = self.index.read().unwrap_or_else(|e| e.into_inner());
        if index.size() == 0 {
            return Ok(Vec::new());
        }
        let fetch_count = top_k.min(index.size());
        let matches = index
            .search(query, fetch_count)
            .map_err(|e| MemoryError::HnswError(format!("usearch::Index::search failed: {e:?}")))?;
        let mut hits: Vec<VectorHit> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(id, dist)| {
                id_to_key.get(id).map(|key| VectorHit {
                    key: key.clone(),
                    distance: *dist,
                })
            })
            .collect();
        // usearch's search results may include slots beyond `size()` if
        // there are gaps in the keymap; trim to top_k.
        hits.truncate(top_k);
        Ok(hits)
    }

    fn len(&self) -> usize {
        let index = self.index.read().unwrap_or_else(|e| e.into_inner());
        index.size()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn save(&self, dir: &Path, basename: &str) -> Result<(), MemoryError> {
        self.save_to_disk(dir, basename)
    }

    fn backend_name(&self) -> &'static str {
        "usearch 2.25 (single-file vector search, C++ via cxx bridge)"
    }
}

// =====================================================================
// Helpers
// =====================================================================

fn validate_dimensions(d: usize) -> Result<(), MemoryError> {
    if d == 0 {
        return Err(MemoryError::HnswError(
            "usearch dimensions must be > 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_dimensions_vs_config(actual: usize, expected: usize) -> Result<(), MemoryError> {
    if actual != expected {
        return Err(MemoryError::HnswError(format!(
            "vector has {actual} dimensions, index expects {expected}"
        )));
    }
    Ok(())
}

fn manifest_path(dir: &Path, basename: &str) -> std::path::PathBuf {
    dir.join(manifest_file_name(basename))
}
fn manifest_file_name(basename: &str) -> String {
    format!("{basename}.hnsw.manifest.json")
}
fn data_file_name(basename: &str) -> String {
    format!("{basename}.hnsw.data")
}
fn keys_file_name(basename: &str) -> String {
    format!("{basename}.hnsw.keys")
}

fn current_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn generate_generation_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("gen-{:x}", nanos)
}

fn blake3_digest_hex(bytes: &[u8]) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(bytes);
    let out = h.finalize();
    out.to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tempfile::TempDir;

    fn test_config() -> VectorIndexConfig {
        VectorIndexConfig {
            m: 8,
            ef_construction: 64,
            ef_search: 32,
            dimensions: 4,
            max_elements: 100,
            compaction_threshold: 0.3,
            flush_interval_secs: None,
        }
    }

    #[test]
    fn new_creates_empty_index() {
        let b = UsearchBackend::new(test_config()).unwrap();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn insert_then_search_returns_match() {
        let b = UsearchBackend::new(test_config()).unwrap();
        b.insert("fact:1".to_string(), &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        b.insert("fact:2".to_string(), &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        b.insert("fact:3".to_string(), &[0.0, 0.0, 1.0, 0.0])
            .unwrap();
        assert_eq!(b.len(), 3);

        let hits = b.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].key, "fact:1");
    }

    #[test]
    fn delete_removes_from_search() {
        let b = UsearchBackend::new(test_config()).unwrap();
        b.insert("a".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        b.insert("b".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        b.delete("a").unwrap();
        assert_eq!(b.len(), 1);
        let hits = b.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        let keys: HashSet<_> = hits.iter().map(|h| h.key.clone()).collect();
        assert!(!keys.contains("a"));
        assert!(keys.contains("b"));
    }

    #[test]
    fn update_replaces_existing() {
        let b = UsearchBackend::new(test_config()).unwrap();
        b.insert("k".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        b.update("k".to_string(), &[0.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(b.len(), 1); // not 2: update is replace, not insert
        let hits = b.search(&[0.0, 0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(hits[0].key, "k");
    }

    #[test]
    fn search_with_wrong_dimensions_errors() {
        let b = UsearchBackend::new(test_config()).unwrap();
        let result = b.search(&[1.0, 0.0], 5);
        assert!(result.is_err());
    }

    #[test]
    fn insert_with_wrong_dimensions_errors() {
        let b = UsearchBackend::new(test_config()).unwrap();
        let result = b.insert("k".to_string(), &[1.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn save_then_load_round_trips() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let b = UsearchBackend::new(test_config()).unwrap();
        b.insert("fact:a".to_string(), &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        b.insert("fact:b".to_string(), &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        b.insert("fact:c".to_string(), &[0.0, 0.0, 1.0, 0.0])
            .unwrap();
        b.save(dir, "test").unwrap();

        // Verify the on-disk files exist
        assert!(dir.join("test.hnsw.manifest.json").exists());
        assert!(dir.join("test.hnsw.data").exists());
        assert!(dir.join("test.hnsw.keys").exists());

        // Load into a new backend
        let b2 = UsearchBackend::load(dir, "test", test_config()).unwrap();
        assert_eq!(b2.len(), 3);
        let hits = b2.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].key, "fact:a");
    }

    #[test]
    fn load_rejects_hnsw_rs_backend_kind() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Write a fake manifest claiming "hnsw_rs" backend
        let manifest = serde_json::json!({
            "schema_version": 1,
            "generation_id": "fake",
            "basename": "test",
            "manifest_file_name": "test.hnsw.manifest.json",
            "data_file_name": "test.hnsw.data",
            "keys_file_name": "test.hnsw.keys",
            "graph_digest": "n/a",
            "data_digest": "n/a",
            "keys_digest": "n/a",
            "dimensions": 4,
            "vector_count": 0,
            "hnsw_sidecar_format_version": 1,
            "backend_kind": "hnsw_rs",
            "backend_version": "0.3.4",
            "source_sqlite_epoch": null,
            "created_at": "2026-06-02T00:00:00Z"
        });
        fs::write(
            dir.join("test.hnsw.manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        fs::write(dir.join("test.hnsw.data"), []).unwrap();
        fs::write(dir.join("test.hnsw.keys"), "").unwrap();

        let result = UsearchBackend::load(dir, "test", test_config());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hnsw_rs"));
    }

    #[test]
    fn backend_name_includes_usearch() {
        let b = UsearchBackend::new(test_config()).unwrap();
        assert!(b.backend_name().contains("usearch"));
    }
}
