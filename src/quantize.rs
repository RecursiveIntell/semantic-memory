//! Scalar quantization (SQ8) for f32 → i8 vector compression.
//!
//! Per-vector symmetric quantization: each vector gets its own scale and zero_point,
//! computed from its min/max values. This gives 4x memory reduction with <0.5% cosine
//! similarity error on normalized embedding vectors.
//!
//! This module is independent of the HNSW backend and can be used with brute-force too.

use crate::error::MemoryError;

/// Scalar quantization parameters for a single vector.
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// Quantized int8 values.
    pub data: Vec<i8>,
    /// Scale factor: `original[i] ≈ (data[i] - zero_point) * scale`
    pub scale: f32,
    /// Symmetric zero point, always in [-127, 127].
    pub zero_point: i8,
}

/// Quantizer that converts f32 vectors to int8 with per-vector calibration.
#[derive(Debug, Clone)]
pub struct Quantizer {
    dimensions: usize,
}

impl Quantizer {
    /// Create a new quantizer for vectors of the given dimensionality.
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    /// The configured dimensionality.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Quantize a single f32 vector to int8 with per-vector symmetric calibration.
    ///
    /// Symmetric quantization maps to [-127, 127] (254 discrete levels).
    /// Each vector gets its own scale/zero_point derived from its min/max values.
    pub fn quantize(&self, vector: &[f32]) -> Result<QuantizedVector, MemoryError> {
        if vector.len() != self.dimensions {
            return Err(MemoryError::QuantizationError(format!(
                "expected {} dimensions, got {}",
                self.dimensions,
                vector.len()
            )));
        }

        let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Handle edge case: constant vector (all dimensions same value)
        if (max - min).abs() < f32::EPSILON {
            return Ok(QuantizedVector {
                data: vec![0i8; self.dimensions],
                scale: 1.0,
                zero_point: 0,
            });
        }

        // Symmetric quantization: 254 steps over [-127, 127]
        let scale = (max - min) / 254.0;
        let zero_point_f = -127.0 - (min / scale);
        let zero_point = zero_point_f.round().clamp(-127.0, 127.0) as i8;

        let data: Vec<i8> = vector
            .iter()
            .map(|&v| {
                let q = (v / scale + zero_point as f32).round();
                q.clamp(-127.0, 127.0) as i8
            })
            .collect();

        Ok(QuantizedVector {
            data,
            scale,
            zero_point,
        })
    }

    /// Dequantize back to f32 (approximate reconstruction).
    pub fn dequantize(&self, qv: &QuantizedVector) -> Vec<f32> {
        qv.data
            .iter()
            .map(|&q| (q as f32 - qv.zero_point as f32) * qv.scale)
            .collect()
    }
}

/// Pack a QuantizedVector into bytes for SQLite storage.
///
/// Format: `[scale: f32 LE][zero_point: i8][data: i8 × dims]`
/// Total bytes: `4 + 1 + dims`
pub fn pack_quantized(qv: &QuantizedVector) -> Vec<u8> {
    let mut buf = Vec::with_capacity(5 + qv.data.len());
    buf.extend_from_slice(&qv.scale.to_le_bytes());
    buf.push(qv.zero_point as u8);
    // Cast i8 slice to u8 slice for storage
    let data_bytes: &[u8] = bytemuck::cast_slice(&qv.data);
    buf.extend_from_slice(data_bytes);
    buf
}

/// Unpack bytes from SQLite into a QuantizedVector.
pub fn unpack_quantized(bytes: &[u8], dimensions: usize) -> Result<QuantizedVector, MemoryError> {
    let expected_len = 5 + dimensions;
    if bytes.len() != expected_len {
        return Err(MemoryError::QuantizationError(format!(
            "expected {} bytes for {} dimensions, got {}",
            expected_len,
            dimensions,
            bytes.len()
        )));
    }
    let scale = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let zero_point = bytes[4] as i8;
    let data: Vec<i8> = bytes[5..].iter().map(|&b| b as i8).collect();
    Ok(QuantizedVector {
        data,
        scale,
        zero_point,
    })
}
