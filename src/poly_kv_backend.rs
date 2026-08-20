//! PolyKvBackend — VectorBackend backed by PolyKV compressed embedding pools.
//!
//! Behind `feature = "poly-kv-codec"`, this backend stores embeddings as
//! FibQuant-encoded FQKV wire bytes and searches via compressed candidate
//! scoring (`attention_topk_compressed`) without full-layer decode.
//!
//! Exact fallback is automatic for any quality breach.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::error::MemoryError;
use crate::vector_backend::{VectorBackend, VectorHit, VectorIndexConfig};

use poly_kv::{
    BranchConfig, CompressionPolicyV1, DType, ExactFallback, ExactKvBlock, KvLayout, KvPoolStore,
    KvRole, KvTensorShape, LayerId, ModelFingerprint, PoolBuilder, QualityGateResultV1,
    SharedKvPool, TokenizerFingerprint,
};

/// PolyKV-backed vector index for compressed embedding storage and search.
pub struct PolyKvBackend {
    /// The current pool (if built).
    pool: Mutex<Option<SharedKvPool>>,
    /// Accumulated exact blocks before pool build.
    pending: Mutex<Vec<ExactKvBlock>>,
    /// Key → token index mapping.
    key_index: Mutex<HashMap<String, usize>>,
    /// Vector dimension.
    dim: usize,
    /// Number of stored items.
    count: Mutex<usize>,
    #[allow(dead_code)]
    /// Persistence store.
    store: Mutex<Option<KvPoolStore>>,
    #[allow(dead_code)]
    /// Config for pool persistence.
    store_root: Mutex<Option<std::path::PathBuf>>,
}

impl PolyKvBackend {
    /// Build a new empty backend.
    pub fn new(config: VectorIndexConfig) -> Result<Self, MemoryError> {
        Ok(Self {
            pool: Mutex::new(None),
            pending: Mutex::new(Vec::new()),
            key_index: Mutex::new(HashMap::new()),
            dim: config.dimensions,
            count: Mutex::new(0),
            store: Mutex::new(None),
            store_root: Mutex::new(None),
        })
    }

    /// Load from a persisted pool.
    pub fn load(
        dir: &Path,
        basename: &str,
        config: VectorIndexConfig,
    ) -> Result<Self, MemoryError> {
        let store = KvPoolStore::open(dir)
            .map_err(|e| MemoryError::Other(format!("poly-kv store open: {e}")))?;
        let identity_path = dir.join(format!("{basename}.manifest.json"));
        let identity: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&identity_path)
                .map_err(|e| MemoryError::Other(format!("poly-kv manifest identity load: {e}")))?,
        )
        .map_err(|e| MemoryError::Other(format!("poly-kv manifest identity decode: {e}")))?;
        let digest: poly_kv::ArtifactDigest =
            serde_json::from_value(identity.get("manifest_digest").cloned().ok_or_else(|| {
                MemoryError::Other("poly-kv manifest identity missing digest".into())
            })?)
            .map_err(|e| MemoryError::Other(format!("poly-kv manifest digest: {e}")))?;
        let (manifest, _encoded_blocks) = store
            .load(&digest)
            .map_err(|e| MemoryError::Other(format!("poly-kv encoded pool validation: {e}")))?;
        if manifest.shape.head_dim as usize != config.dimensions {
            return Err(MemoryError::Other(format!(
                "poly-kv persisted dimension {} != configured dimension {}",
                manifest.shape.head_dim, config.dimensions
            )));
        }
        #[cfg(feature = "fib-quant-codec")]
        let pool = if manifest.policy.value_codec_id.as_str() == "poly-kv:value:fibquant" {
            store
                .load_fibquant_pool(&digest, manifest.policy.quality_gate.max_value_mse)
                .map_err(|e| MemoryError::Other(format!("poly-kv FibQuant pool reload: {e}")))?
        } else {
            store
                .load_pool_with_value_codec(&digest, poly_kv::RawExactValueCodec)
                .map_err(|e| MemoryError::Other(format!("poly-kv exact pool reload: {e}")))?
        };
        #[cfg(not(feature = "fib-quant-codec"))]
        let pool = if manifest.policy.value_codec_id.as_str() == "poly-kv:value:raw-exact" {
            store
                .load_pool_with_value_codec(&digest, poly_kv::RawExactValueCodec)
                .map_err(|e| MemoryError::Other(format!("poly-kv exact pool reload: {e}")))?
        } else {
            return Err(MemoryError::Other(format!(
                "poly-kv codec {} is unavailable in this build",
                manifest.policy.value_codec_id
            )));
        };

        let keys_path = dir.join(format!("{basename}.keys.json"));
        let key_index: HashMap<String, usize> = serde_json::from_slice(
            &std::fs::read(&keys_path)
                .map_err(|e| MemoryError::Other(format!("poly-kv key map load: {e}")))?,
        )
        .map_err(|e| MemoryError::Other(format!("poly-kv key map decode: {e}")))?;
        let count = key_index.len();
        if count != manifest.shape.seq_len as usize {
            return Err(MemoryError::Other(format!(
                "poly-kv key map count {} != persisted sequence length {}",
                count, manifest.shape.seq_len
            )));
        }
        let indices: std::collections::HashSet<usize> = key_index.values().copied().collect();
        if key_index.keys().any(|key| key.is_empty())
            || indices.len() != count
            || indices.iter().any(|index| *index >= count)
        {
            return Err(MemoryError::Other(
                "poly-kv key map is not a contiguous permutation".into(),
            ));
        }
        let value_data = &pool
            .exact_fallback_ref()
            .and_then(|fallback| {
                fallback
                    .blocks
                    .iter()
                    .find(|block| block.role == KvRole::Value && block.layer == LayerId(0))
            })
            .ok_or_else(|| {
                MemoryError::Other("poly-kv reload missing aggregate value fallback block".into())
            })?
            .data;
        let expected_values = count.checked_mul(config.dimensions).ok_or_else(|| {
            MemoryError::Other("poly-kv reload vector cardinality overflow".into())
        })?;
        if value_data.len() != expected_values {
            return Err(MemoryError::Other(format!(
                "poly-kv fallback value count {} != key count {} × dimension {}",
                value_data.len(),
                count,
                config.dimensions
            )));
        }
        let pending = value_data
            .chunks_exact(config.dimensions)
            .map(|vector| ExactKvBlock {
                role: KvRole::Value,
                layer: LayerId(0),
                shape: KvTensorShape {
                    layers: 1,
                    key_heads: 1,
                    value_heads: 1,
                    seq_len: 1,
                    head_dim: config.dimensions as u32,
                    layout: KvLayout::LayersHeadsTokensDim,
                    dtype: DType::F32,
                },
                data: vector.to_vec(),
            })
            .collect();
        Ok(Self {
            pool: Mutex::new(Some(pool)),
            pending: Mutex::new(pending),
            key_index: Mutex::new(key_index),
            dim: config.dimensions,
            count: Mutex::new(count),
            store: Mutex::new(Some(store)),
            store_root: Mutex::new(Some(dir.to_path_buf())),
        })
    }

    fn build_pool(&self) -> Result<SharedKvPool, MemoryError> {
        let pending = self.pending.lock().unwrap();
        if pending.is_empty() {
            return Err(MemoryError::Other("no embeddings to build pool".into()));
        }
        let count = pending.len();
        let shape = KvTensorShape {
            layers: 1,
            key_heads: 1,
            value_heads: 1,
            seq_len: count as u64,
            head_dim: self.dim as u32,
            layout: KvLayout::LayersHeadsTokensDim,
            dtype: DType::F32,
        };
        let data: Vec<f32> = pending
            .iter()
            .flat_map(|b| b.data.iter().copied())
            .collect();
        let blocks = vec![
            ExactKvBlock {
                role: KvRole::Key,
                layer: LayerId(0),
                shape: shape.clone(),
                data: data.clone(),
            },
            ExactKvBlock {
                role: KvRole::Value,
                layer: LayerId(0),
                shape: shape.clone(),
                data,
            },
        ];
        let fallback = ExactFallback::from_blocks(blocks.clone());
        let pool = PoolBuilder::default()
            .shape(shape.clone())
            .model_fingerprint(
                ModelFingerprint::new("semantic-memory-embeddings")
                    .map_err(|e| MemoryError::Other(format!("model fp: {e}")))?,
            )
            .tokenizer_fingerprint(
                TokenizerFingerprint::new("none")
                    .map_err(|e| MemoryError::Other(format!("tokenizer fp: {e}")))?,
            )
            .exact_fallback(fallback)
            .policy(CompressionPolicyV1 {
                quality_gate: QualityGateResultV1 {
                    max_key_mse: 0.01,
                    max_value_mse: 0.01,
                    ..QualityGateResultV1::alpha_reference()
                },
                ..CompressionPolicyV1::alpha_reference()
            })
            .value_codec({
                #[cfg(feature = "fib-quant-codec")]
                {
                    poly_kv::adapters::fibquant::FibQuantValueCodec::new(self.dim, 4, 32, 42)
                        .map_err(|e| MemoryError::Other(format!("fibquant codec: {e}")))?
                        .with_max_mse(0.01)
                        .map_err(|e| MemoryError::Other(format!("fibquant quality: {e}")))?
                }
                #[cfg(not(feature = "fib-quant-codec"))]
                {
                    poly_kv::RawExactValueCodec
                }
            })
            .build_from_blocks(blocks)
            .map_err(|e| MemoryError::Other(format!("pool build: {e}")))?;
        Ok(pool)
    }

    fn get_or_build_pool(&self) -> Result<SharedKvPool, MemoryError> {
        let mut guard = self.pool.lock().unwrap();
        if let Some(ref pool) = *guard {
            return Ok(pool.clone());
        }
        let pool = self.build_pool()?;
        *guard = Some(pool.clone());
        Ok(pool)
    }
}

impl VectorBackend for PolyKvBackend {
    fn insert(&self, key: String, vector: &[f32]) -> Result<(), MemoryError> {
        if vector.len() != self.dim {
            return Err(MemoryError::Other(format!(
                "dim mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        let mut pending = self.pending.lock().unwrap();
        let mut key_index = self.key_index.lock().unwrap();
        let mut count = self.count.lock().unwrap();

        // If key already exists, remove old block.
        if let Some(&old_idx) = key_index.get(&key) {
            if old_idx < pending.len() {
                pending.remove(old_idx);
                // Shift indices.
                for idx in key_index.values_mut() {
                    if *idx > old_idx {
                        *idx -= 1;
                    }
                }
                *count -= 1;
            }
        }

        let token_idx = *count;
        pending.push(ExactKvBlock {
            role: KvRole::Value,
            layer: LayerId(0),
            shape: KvTensorShape {
                layers: 1,
                key_heads: 1,
                value_heads: 1,
                seq_len: 1,
                head_dim: self.dim as u32,
                layout: KvLayout::LayersHeadsTokensDim,
                dtype: DType::F32,
            },
            data: vector.to_vec(),
        });
        key_index.insert(key, token_idx);
        *count += 1;
        // Invalidate cached pool.
        *self.pool.lock().unwrap() = None;
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<(), MemoryError> {
        let mut key_index = self.key_index.lock().unwrap();
        let mut pending = self.pending.lock().unwrap();
        let mut count = self.count.lock().unwrap();

        if let Some(&idx) = key_index.get(key) {
            if idx < pending.len() {
                pending.remove(idx);
            }
            key_index.remove(key);
            // Shift indices.
            for v in key_index.values_mut() {
                if *v > idx {
                    *v -= 1;
                }
            }
            *count -= 1;
            *self.pool.lock().unwrap() = None;
        }
        Ok(())
    }

    fn update(&self, key: String, vector: &[f32]) -> Result<(), MemoryError> {
        self.insert(key, vector)
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>, MemoryError> {
        if query.len() != self.dim {
            return Err(MemoryError::Other(format!(
                "query dim mismatch: expected {}, got {}",
                self.dim,
                query.len()
            )));
        }
        let count = *self.count.lock().unwrap();
        if count == 0 {
            return Ok(Vec::new());
        }

        let pool = self.get_or_build_pool()?;

        // Use compressed scoring for the first head.
        #[cfg(feature = "fib-quant-codec")]
        {
            let selection = pool
                .attention_topk_compressed(0, 0, query, top_k.min(count))
                .map_err(|e| MemoryError::Other(format!("poly-kv search: {e}")))?;

            let key_index = self.key_index.lock().unwrap();
            // Reverse lookup: token_index → key
            let idx_to_key: HashMap<usize, String> =
                key_index.iter().map(|(k, v)| (*v, k.clone())).collect();

            Ok(selection
                .hits
                .iter()
                .filter_map(|hit| {
                    idx_to_key.get(&hit.token_index).map(|key| VectorHit {
                        key: key.clone(),
                        distance: 1.0 - hit.score.clamp(-1.0, 1.0),
                    })
                })
                .collect())
        }
        #[cfg(not(feature = "fib-quant-codec"))]
        {
            // Fallback: decode full pool and compute cosine.
            let reader = pool
                .attach_reader(Default::default())
                .map_err(|e| MemoryError::Other(format!("reader: {e}")))?;
            let decoded = reader
                .decode_layer(LayerId(0))
                .map_err(|e| MemoryError::Other(format!("decode: {e}")))?;
            let all_values = decoded.value.data;
            let key_index = self.key_index.lock().unwrap();
            let idx_to_key: HashMap<usize, String> =
                key_index.iter().map(|(k, v)| (*v, k.clone())).collect();
            let mut scored: Vec<(String, f32)> = (0..count)
                .filter_map(|i| {
                    let start = i * self.dim;
                    let end = start + self.dim;
                    if end > all_values.len() {
                        return None;
                    }
                    let sim = cosine_similarity(query, &all_values[start..end]);
                    Some((idx_to_key.get(&i)?.clone(), sim))
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(top_k);
            Ok(scored
                .into_iter()
                .map(|(key, sim)| VectorHit {
                    key,
                    distance: 1.0 - sim,
                })
                .collect())
        }
    }

    fn len(&self) -> usize {
        *self.count.lock().unwrap()
    }

    fn save(&self, dir: &Path, _basename: &str) -> Result<(), MemoryError> {
        let pool = self.get_or_build_pool()?;
        let store = KvPoolStore::open(dir)
            .map_err(|e| MemoryError::Other(format!("poly-kv store open: {e}")))?;
        let persisted = store
            .persist(&pool)
            .map_err(|e| MemoryError::Other(format!("poly-kv save: {e}")))?;
        let key_index = self.key_index.lock().unwrap();
        let keys_path = dir.join(format!("{_basename}.keys.json"));
        let key_bytes = serde_json::to_vec(&*key_index)
            .map_err(|e| MemoryError::Other(format!("poly-kv key map encode: {e}")))?;
        let temp_path = keys_path.with_extension("keys.json.tmp");
        std::fs::write(&temp_path, key_bytes)
            .map_err(|e| MemoryError::Other(format!("poly-kv key map write: {e}")))?;
        std::fs::rename(&temp_path, &keys_path)
            .map_err(|e| MemoryError::Other(format!("poly-kv key map publish: {e}")))?;
        let identity = serde_json::json!({"manifest_digest": persisted.manifest.manifest_digest});
        std::fs::write(
            dir.join(format!("{_basename}.manifest.json")),
            serde_json::to_vec(&identity).unwrap(),
        )
        .map_err(|e| MemoryError::Other(format!("poly-kv manifest identity write: {e}")))?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "poly-kv fibquant compressed embedding pool"
    }
}

#[cfg(all(test, feature = "fib-quant-codec"))]
mod tests {
    use super::*;

    fn config(dimensions: usize) -> VectorIndexConfig {
        VectorIndexConfig {
            dimensions,
            ..VectorIndexConfig::default()
        }
    }

    fn axis(dimensions: usize, index: usize, value: f32) -> Vec<f32> {
        let mut vector = vec![0.0; dimensions];
        vector[index] = value;
        vector
    }

    #[test]
    fn multiple_items_persist_reload_and_score_with_stable_keys() {
        let dir = tempfile::tempdir().expect("temporary PolyKV directory");
        let backend = PolyKvBackend::new(config(32)).expect("new backend");
        backend
            .insert("fact:a".into(), &axis(32, 0, 1.0))
            .expect("insert a");
        backend
            .insert("fact:b".into(), &axis(32, 1, 1.0))
            .expect("insert b");
        backend
            .insert("fact:c".into(), &axis(32, 0, -1.0))
            .expect("insert c");

        let before = backend
            .search(&axis(32, 0, 1.0), 2)
            .expect("compressed search before save");
        assert_eq!(before.first().map(|hit| hit.key.as_str()), Some("fact:a"));
        backend.save(dir.path(), "semantic").expect("persist pool");

        let loaded =
            PolyKvBackend::load(dir.path(), "semantic", config(32)).expect("fresh backend reload");
        assert_eq!(loaded.len(), 3);
        let after = loaded
            .search(&axis(32, 0, 1.0), 2)
            .expect("compressed search after reload");
        assert_eq!(after.first().map(|hit| hit.key.as_str()), Some("fact:a"));
        assert_eq!(
            before.iter().map(|hit| &hit.key).collect::<Vec<_>>(),
            after.iter().map(|hit| &hit.key).collect::<Vec<_>>()
        );
    }

    #[test]
    fn reload_rejects_non_permutation_key_map() {
        let dir = tempfile::tempdir().expect("temporary PolyKV directory");
        let backend = PolyKvBackend::new(config(32)).expect("new backend");
        backend
            .insert("fact:a".into(), &axis(32, 0, 1.0))
            .expect("insert a");
        backend
            .insert("fact:b".into(), &axis(32, 1, 1.0))
            .expect("insert b");
        backend.save(dir.path(), "semantic").expect("persist pool");
        std::fs::write(
            dir.path().join("semantic.keys.json"),
            br#"{"fact:a":0,"fact:b":0}"#,
        )
        .expect("corrupt key map");
        let error = match PolyKvBackend::load(dir.path(), "semantic", config(32)) {
            Ok(_) => panic!("non-permutation key map must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("permutation"));
    }

    #[test]
    fn reload_rejects_corrupt_encoded_block_before_fallback_rebuild() {
        let dir = tempfile::tempdir().expect("temporary PolyKV directory");
        let backend = PolyKvBackend::new(config(32)).expect("new backend");
        backend
            .insert("fact:a".into(), &axis(32, 0, 1.0))
            .expect("insert a");
        backend.save(dir.path(), "semantic").expect("persist pool");

        let block_path = std::fs::read_dir(dir.path().join("blocks"))
            .expect("read persisted block directory")
            .next()
            .expect("at least one persisted block")
            .expect("read persisted block entry")
            .path();
        let mut bytes = std::fs::read(&block_path).expect("read persisted block");
        bytes[0] ^= 0x01;
        std::fs::write(&block_path, bytes).expect("corrupt persisted block");

        let error = match PolyKvBackend::load(dir.path(), "semantic", config(32)) {
            Ok(_) => panic!("corrupted content-addressed block must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("block digest mismatch"));
    }

    #[test]
    fn reload_rejects_incomplete_key_map() {
        let dir = tempfile::tempdir().expect("temporary PolyKV directory");
        let backend = PolyKvBackend::new(config(32)).expect("new backend");
        backend
            .insert("fact:a".into(), &axis(32, 0, 1.0))
            .expect("insert a");
        backend.save(dir.path(), "semantic").expect("persist pool");
        std::fs::write(dir.path().join("semantic.keys.json"), b"{}")
            .expect("truncate key map semantically");

        let error = match PolyKvBackend::load(dir.path(), "semantic", config(32)) {
            Ok(_) => panic!("incomplete key map must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("key map count"));
    }
}

#[cfg(not(feature = "fib-quant-codec"))]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (dot, na, nb) = a
        .iter()
        .zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(d, na, nb), (x, y)| {
            (d + x * y, na + x * x, nb + y * y)
        });
    let denom = (na * nb).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}
