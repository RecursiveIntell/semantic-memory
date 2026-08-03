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
    /// Persistence store.
    store: Mutex<Option<KvPoolStore>>,
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
        _basename: &str,
        config: VectorIndexConfig,
    ) -> Result<Self, MemoryError> {
        let store = KvPoolStore::open(dir)
            .map_err(|e| MemoryError::Other(format!("poly-kv store open: {e}")))?;
        let manifests = store
            .list_pools()
            .map_err(|e| MemoryError::Other(format!("poly-kv list: {e}")))?;
        if manifests.is_empty() {
            return Self::new(config);
        }
        // Load the first pool found.
        let digest = manifests[0].manifest_digest;
        let (_manifest, _blocks) = store
            .load(&digest)
            .map_err(|e| MemoryError::Other(format!("poly-kv load: {e}")))?;
        // TODO: rebuild pool from loaded blocks.
        Ok(Self {
            pool: Mutex::new(None),
            pending: Mutex::new(Vec::new()),
            key_index: Mutex::new(HashMap::new()),
            dim: config.dimensions,
            count: Mutex::new(0),
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
        let fallback = ExactFallback::from_blocks(pending.clone());
        let pool = PoolBuilder::default()
            .shape(shape)
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
            .build_from_blocks(pending.clone())
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
        // Build one key block and one value block for this token.
        // For embeddings, key and value are both the embedding vector.
        let layer_id = LayerId(0);
        let block_shape = KvTensorShape {
            layers: 1,
            key_heads: 1,
            value_heads: 1,
            seq_len: 1,
            head_dim: self.dim as u32,
            layout: KvLayout::LayersHeadsTokensDim,
            dtype: DType::F32,
        };
        let key_block = ExactKvBlock {
            role: KvRole::Key,
            layer: layer_id,
            shape: block_shape.clone(),
            data: vector.to_vec(),
        };
        let value_block = ExactKvBlock {
            role: KvRole::Value,
            layer: layer_id,
            shape: block_shape,
            data: vector.to_vec(),
        };
        pending.push(key_block);
        pending.push(value_block);
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
            // Remove both key and value blocks (2 blocks per token).
            let block_idx = idx * 2;
            if block_idx + 1 < pending.len() {
                pending.remove(block_idx + 1); // value
                pending.remove(block_idx); // key
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
        #[cfg(feature = "fibquant-adapter")]
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
                        distance: 1.0 - hit.score.max(-1.0).min(1.0),
                    })
                })
                .collect())
        }
        #[cfg(not(feature = "fibquant-adapter"))]
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
        store
            .persist(&pool)
            .map_err(|e| MemoryError::Other(format!("poly-kv save: {e}")))?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "poly-kv fibquant compressed embedding pool"
    }
}

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
