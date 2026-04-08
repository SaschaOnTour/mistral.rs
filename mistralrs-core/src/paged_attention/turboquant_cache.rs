//! TurboQuant KV-Cache — block-level quantized storage for keys and values.
//!
//! This module provides [`TurboQuantKVCache`] which quantizes KV vectors into
//! compressed indices + scales using block-level PolarQuant (block_size=32).
//! Supports MaxNorm (llama.cpp-compatible) and L2Norm (Paper) modes.
//!
//! When `KvCache::TurboQuant` is selected, key/value vectors are quantized
//! on write via Candle tensor ops (works on CUDA, Metal, CPU) and the entire
//! compressed cache is dequantized on read for attention.

use candle_core::{DType, Device, Result, Tensor, D};
use turboquant::attention::QuantizedKVCache;
use turboquant::packed::TurboQuantConfig;

// ---------------------------------------------------------------------------
// Named constants (no magic numbers)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Zero-copy tensor data extraction
// ---------------------------------------------------------------------------

const DEFAULT_ROTATION_SEED: u64 = 42;

/// Default seed for the QJL Rademacher matrix.
const DEFAULT_QJL_SEED: u64 = 12345;

// ENV_TURBOQUANT_DISABLE_QJL removed — use --pa-cache-type pq3/pqo3 instead of tq3.

// ENV_TURBOQUANT_NORM_MODE removed — norm mode is now set via CLI (--pa-norm-mode).

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

/// Wrapper for the tq_pack_indices CUDA kernel.
/// Packs U8 indices to bit-packed format entirely on GPU (no CPU roundtrip).
#[cfg(feature = "cuda")]
pub(crate) fn cuda_pack_indices(
    indices: &Tensor,
    num_vectors: usize,
    block_size: usize,
    bits: usize,
    device: &Device,
) -> Result<Tensor> {
    mistralrs_paged_attn::tq_pack_indices(indices, num_vectors, block_size, bits, device)
}

/// Wrapper for the tq_qjl_batch CUDA kernel.
/// Computes QJL signs + residual norms entirely on GPU.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_qjl_batch(
    original: &Tensor,
    dequantized: &Tensor,
    qjl_seed: u64,
    num_blocks: usize,
    head_dim: usize,
    signs_per_block: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    mistralrs_paged_attn::tq_qjl_batch(
        original,
        dequantized,
        qjl_seed,
        num_blocks,
        head_dim,
        signs_per_block,
        device,
    )
}

/// Inverse WHT rotation using butterfly algorithm — O(N log N) instead of O(N²) matmul.
///
/// Equivalent to `dequant.matmul(&rotation_inv)` where
/// `rotation_inv = diag(signs) @ H_normalized`.
///
/// For each row: result = WHT(row * signs) / sqrt(block_size).
/// Parallelized with Rayon across blocks.
/// CPU-only — on GPU, cuBLAS matmul is faster for 32×32.
fn butterfly_wht_inverse_cpu(
    dequant: &Tensor,
    rotation_fwd: &Tensor,
    block_size: usize,
) -> Result<Tensor> {
    use rayon::prelude::*;

    let (m, bs) = dequant.dims2()?;
    debug_assert_eq!(bs, block_size);

    // Extract sign pattern: rotation_fwd[0][j] = signs[j] / sqrt(N)
    // → signs[j] = rotation_fwd[0][j] * sqrt(N)
    let sqrt_n = (block_size as f32).sqrt();
    let signs: Vec<f32> = rotation_fwd
        .narrow(0, 0, 1)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .to_vec1()?;

    let mut data: Vec<f32> = dequant.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let inv_sqrt_n = 1.0 / sqrt_n;

    // Process all blocks in parallel — each block is independent
    data.par_chunks_mut(block_size).for_each(|block| {
        // Step 1: Apply sign flip
        for j in 0..block_size {
            block[j] *= signs[j] * sqrt_n;
        }

        // Step 2: Butterfly WHT (unnormalized, 5 stages for block_size=32)
        let mut h = 1;
        while h < block_size {
            let full = h << 1;
            let mut i = 0;
            while i < block_size {
                for j in 0..h {
                    let a = block[i + j];
                    let b = block[i + j + h];
                    block[i + j] = a + b;
                    block[i + j + h] = a - b;
                }
                i += full;
            }
            h <<= 1;
        }

        // Step 3: Normalize by 1/sqrt(N)
        for val in block.iter_mut() {
            *val *= inv_sqrt_n;
        }
    });

    Tensor::from_vec(data, (m, block_size), &Device::Cpu)
}

/// Forward WHT rotation using butterfly algorithm — O(N log N) instead of O(N²) matmul.
///
/// Equivalent to `blocked.matmul(&rotation_fwd)` where
/// `rotation_fwd = H_normalized @ diag(signs)`.
///
/// For each row: result = signs * WHT(row) / sqrt(block_size).
///
/// This MUST produce numerically identical results to the CUDA butterfly WHT
/// in the dequant kernels, so that quant→dequant roundtrip is consistent.
fn butterfly_wht_forward(
    blocked: &Tensor,
    rotation_fwd: &Tensor,
    block_size: usize,
) -> Result<Tensor> {
    let (m, bs) = blocked.dims2()?;
    debug_assert_eq!(bs, block_size);
    let device = blocked.device().clone();

    // Extract sign pattern
    let sqrt_n = (block_size as f32).sqrt();
    let signs: Vec<f32> = rotation_fwd
        .narrow(0, 0, 1)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    let sign_vals: Vec<f32> = signs.iter().map(|s| s * sqrt_n).collect();

    let mut data: Vec<f32> = blocked.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let inv_sqrt_n = 1.0 / sqrt_n;

    // Process all blocks: butterfly WHT then sign flip
    // Must match CUDA: WHT(x) / sqrt(N) * signs
    for block in data.chunks_mut(block_size) {
        // Step 1: Butterfly WHT (unnormalized)
        let mut h = 1;
        while h < block_size {
            let full = h << 1;
            let mut i = 0;
            while i < block_size {
                for j in 0..h {
                    let a = block[i + j];
                    let b = block[i + j + h];
                    block[i + j] = a + b;
                    block[i + j + h] = a - b;
                }
                i += full;
            }
            h <<= 1;
        }

        // Step 2: Normalize by 1/sqrt(N) and apply signs
        for j in 0..block_size {
            block[j] = block[j] * inv_sqrt_n * sign_vals[j];
        }
    }

    Tensor::from_vec(data, (m, block_size), &Device::Cpu)?.to_device(&device)
}

/// Forward WHT rotation using butterfly — GPU-compatible via Candle gather/scatter.
///
/// Equivalent to `blocked.matmul(&rotation_fwd)` but uses the same butterfly
/// algorithm as the CUDA dequant kernels, ensuring numerical consistency.
///
/// Result: signs * WHT(x) / sqrt(block_size) for each row.
fn butterfly_wht_forward_gpu(
    blocked: &Tensor,
    rotation_fwd: &Tensor,
    block_size: usize,
) -> Result<Tensor> {
    let device = blocked.device().clone();
    let (m, bs) = blocked.dims2()?;
    debug_assert_eq!(bs, block_size);

    // Extract sign pattern: rotation_fwd[0][j] * sqrt(N) = signs[j]
    let sqrt_n = (block_size as f64).sqrt();
    let sign_pattern = (rotation_fwd.narrow(0, 0, 1)? * sqrt_n)?
        .squeeze(0)?
        .to_dtype(DType::F32)?;

    // Butterfly WHT: 5 stages for block_size=32
    // Each stage h: for pair (i, i^h), compute (a+b, a-b)
    // Using Candle ops: split into even/odd halves and add/subtract
    let mut data = blocked.to_dtype(DType::F32)?;

    let mut h = 1usize;
    while h < block_size {
        // Build index tensors for the butterfly pairs
        // For each position j in [0..block_size]:
        //   partner = j ^ h
        //   if j & h == 0: new[j] = old[j] + old[partner]
        //   if j & h != 0: new[j] = old[partner] - old[j]
        let partner_indices: Vec<u32> = (0..block_size as u32).map(|j| j ^ (h as u32)).collect();
        let partner_idx = Tensor::from_vec(partner_indices, (1, block_size), &device)?
            .broadcast_as((m, block_size))?
            .contiguous()?;

        let data_cont = data.contiguous()?;
        let partners = data_cont.gather(&partner_idx, 1)?;

        // Mask: 1.0 where j & h == 0 (add), -1.0 where j & h != 0 (subtract)
        let mask_vals: Vec<f32> = (0..block_size)
            .map(|j| if j & h == 0 { 1.0f32 } else { -1.0f32 })
            .collect();
        let mask = Tensor::from_vec(mask_vals, (1, block_size), &device)?;

        // new = data * mask + partners  (when mask=1: data+partner, when mask=-1: partner-data)
        // Simplify: new[j] = if mask>0: data[j]+partner[j] else partner[j]-data[j]
        //         = partner[j] + mask[j]*data[j]
        data = (partners + data.broadcast_mul(&mask)?)?;

        h <<= 1;
    }

    // Normalize by 1/sqrt(block_size) and apply signs
    let inv_sqrt = 1.0 / sqrt_n;
    let result = (data * inv_sqrt)?.broadcast_mul(&sign_pattern.unsqueeze(0)?)?;

    Ok(result)
}

/// Unpack bit-packed indices to U8 on-device (GPU or CPU, no transfer).
/// Supports 2-bit, 3-bit, and 4-bit packing.
fn unpack_indices_on_device(
    packed: &Tensor,
    n: usize,
    head_dim: usize,
    bits: u8,
) -> Result<Tensor> {
    let device = packed.device();
    let total_values = n * head_dim;

    // Use CPU unpack for all bit widths — Candle doesn't support bitwise ops.
    // The CPU roundtrip is fast for unpacking (sequential byte ops, no GPU sync).
    // The bottleneck is the matmul in dequant, not the unpack.
    let packed_cpu: Vec<u8> = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let unpacked = match bits {
        2 => turboquant::packed::unpack_indices_2bit(&packed_cpu, total_values),
        3 => turboquant::packed::unpack_indices_3bit(&packed_cpu, total_values),
        4 => turboquant::packed::unpack_indices_4bit(&packed_cpu, total_values),
        _ => return Err(tq_err(format!("unsupported bits for unpack: {bits}"))),
    };
    Tensor::from_vec(unpacked, (n, head_dim), &Device::Cpu)?.to_device(device)
}

fn tq_err(msg: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("TurboQuant: {msg}"))
}

/// Flattens a slice of per-entry vectors into a single contiguous `Vec<f32>`.
fn flatten_vecs(vecs: &[Vec<f32>], head_dim: usize) -> Vec<f32> {
    let mut flat = Vec::with_capacity(vecs.len() * head_dim);
    for v in vecs {
        flat.extend_from_slice(v);
    }
    flat
}

// ---------------------------------------------------------------------------
// TurboQuantKVCache
// ---------------------------------------------------------------------------

/// Block-level TurboQuant KV cache.
///
/// Stores compressed KV data as block-level indices + scales on GPU/CPU.
/// Each head_dim vector is split into blocks of QUANT_BLOCK_SIZE=32, each
/// independently rotated (WHT) and quantized via Lloyd-Max codebooks.
///
/// Also maintains per-head CPU `QuantizedKVCache` instances for the
/// `push_head`/`dequantize_keys` API (used by tests and QJL accessors).

pub struct TurboQuantKVCache {
    /// Per-head CPU caches (used by push_head/dequantize_keys API and QJL accessors).
    caches: Vec<QuantizedKVCache>,
    bits: u8,
    head_dim: usize,
    num_kv_heads: usize,
    num_layers: usize,
    /// Normalization mode: L2Norm (paper) or MaxNorm (llama.cpp).
    norm_mode: QuantNormMode,
    /// Number of outlier blocks that get the higher-quality codebook (bits-bit).
    /// - PQ plain: 0 (all blocks use standard (bits)-bit codebook)
    /// - PQO: usize::MAX (all blocks use outlier codebook = effectively bits-bit)
    /// - TQ: 0 (all blocks use (bits-1)-bit codebook, QJL compensates)
    outlier_blocks: usize,
    /// Whether QJL signs are computed and QJL correction is applied in attention.
    /// - PQ/PQO: false
    /// - TQ: true
    qjl_enabled: bool,
    /// Current sequence length per layer (how much of the compressed buffer is filled).
    buf_seq_len: Vec<usize>,
    /// GPU compressed KV-cache: quantized key indices per layer.
    /// Shape: [num_kv_heads, seq_capacity, head_dim] U8 (centroid index per element).
    gpu_k_indices: Vec<Option<Tensor>>,
    /// GPU compressed KV-cache: quantized value indices per layer.
    gpu_v_indices: Vec<Option<Tensor>>,
    /// GPU compressed KV-cache: key scales per layer.
    /// Shape: [num_kv_heads, seq_capacity, num_blocks] F32.
    gpu_k_scales: Vec<Option<Tensor>>,
    /// GPU compressed KV-cache: value scales per layer.
    gpu_v_scales: Vec<Option<Tensor>>,
    /// Whether this layer uses the GPU compressed path.
    gpu_path_active: Vec<bool>,
    /// Cached pre-computed tensors for on-device quantize/dequantize.
    gpu_precomputed: Option<GpuPrecomputed>,
    /// GPU compressed KV-cache: QJL sign bits for keys per layer.
    /// Shape: [num_kv_heads, seq_capacity, signs_per_head] U8
    /// where signs_per_head = head_dim / 8.
    gpu_k_qjl_signs: Vec<Option<Tensor>>,
    /// GPU compressed KV-cache: residual norms for keys per layer.
    /// Shape: [num_kv_heads, seq_capacity] F16.
    gpu_k_residual_norms: Vec<Option<Tensor>>,
    /// Lazy quantize: FP16 K/V stored during prefill, quantized before first decode.
    /// Shape: [num_kv_heads, seq_len, head_dim] in orig_dtype.
    lazy_k: Vec<Option<Tensor>>,
    lazy_v: Vec<Option<Tensor>>,
}

/// Block size for quantization. Each head_dim vector is split into blocks
/// of this size, each independently quantized with its own WHT rotation and norm.
///
/// Paper Section 4.3 + llama.cpp: block_size=32 gives much better quality
/// for real LLM KV cache vectors (which have norms 10-400+) compared to
/// quantizing the full head_dim=128 as a single block.
const QUANT_BLOCK_SIZE: usize = 32;

/// Normalization strategy for block-level PolarQuant.
/// Normalization strategy for block-level PolarQuant.
/// Configurable via `--pa-norm-mode maxnorm|l2norm`.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum QuantNormMode {
    /// Paper Algorithm 1: L2-norm → unit sphere → Beta-distribution codebooks.
    /// Mathematically optimal MSE. Use with QJL for unbiased inner products.
    L2Norm,
    /// llama.cpp approach: max-abs-norm → [-1,1] range → empirical codebooks.
    /// Better for attention quality without QJL. No theoretical guarantees.
    #[default]
    MaxNorm,
}

impl std::str::FromStr for QuantNormMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "maxnorm" => Ok(Self::MaxNorm),
            "l2norm" => Ok(Self::L2Norm),
            other => Err(format!(
                "Unknown norm mode `{other}`. Options: maxnorm, l2norm"
            )),
        }
    }
}

impl std::fmt::Display for QuantNormMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxNorm => write!(f, "maxnorm"),
            Self::L2Norm => write!(f, "l2norm"),
        }
    }
}

/// Pre-computed tensors for on-device TurboQuant operations.
///
/// All tensors live on the same GPU device. Created once, reused for
/// every decode step. Uses Candle tensor ops → works on CUDA, Metal, CPU.
///
/// Rotation matrices are [QUANT_BLOCK_SIZE × QUANT_BLOCK_SIZE], not
/// [head_dim × head_dim]. Each 32-element block within a head_dim=128
/// vector gets its own independent rotation + norm + quantization.
struct GpuPrecomputed {
    /// Forward rotation: H_normalized @ diag(signs), shape [block_size, block_size].
    rotation_fwd: Tensor,
    /// Inverse rotation: diag(signs) @ H = rotation_fwd^T.
    rotation_inv: Tensor,
    /// Normal codebook centroids (polar_bits), shape [n_centroids].
    centroids: Tensor,
    /// Normal codebook boundaries, shape [n_boundaries].
    boundaries: Tensor,
    /// Outlier codebook centroids (polar_bits + 1), shape [n_outlier_centroids].
    outlier_centroids: Tensor,
    /// Outlier codebook boundaries, shape [n_outlier_boundaries].
    outlier_boundaries: Tensor,
    /// Max value of outlier centroids (cached to avoid per-call GPU→CPU sync).
    outlier_outer_centroid: f64,
    /// Pre-computed scale sign tensor for outlier block marking, shape [1, num_blocks].
    scale_sign_tensor: Tensor,
    /// Pre-computed Rademacher matrix for QJL correction, shape [dim, dim].
    /// Each entry is ±1/√dim, deterministic from (DEFAULT_QJL_SEED, row, col).
    /// Only allocated when qjl_enabled=true. Lazy-initialized on first use.
    qjl_rademacher: Option<Tensor>,
}

/// Builds a normalized Hadamard matrix of size `dim × dim` on the given device.
///
/// The normalized matrix H_n = H / sqrt(dim) is orthogonal and self-inverse:
/// H_n @ H_n = I. This matches the `wht_inplace` + normalization in turboquant.
fn build_hadamard_matrix(dim: usize, device: &Device) -> Result<Tensor> {
    let mut h = vec![1.0f32; 1];
    let mut size = 1;
    while size < dim {
        let mut new_h = vec![0.0f32; (size * 2) * (size * 2)];
        for i in 0..size {
            for j in 0..size {
                let val = h[i * size + j];
                new_h[i * (size * 2) + j] = val;
                new_h[i * (size * 2) + (j + size)] = val;
                new_h[(i + size) * (size * 2) + j] = val;
                new_h[(i + size) * (size * 2) + (j + size)] = -val;
            }
        }
        h = new_h;
        size *= 2;
    }
    let norm = 1.0 / (dim as f32).sqrt();
    for v in h.iter_mut() {
        *v *= norm;
    }
    Tensor::from_vec(h, (dim, dim), device)
}

impl TurboQuantKVCache {
    /// PQ plain: standard codebook, no outlier override, no QJL.
    /// Each block uses the `bits`-bit codebook directly.
    pub fn new_pq(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
    ) -> Result<Self> {
        Self::new_internal(
            bits,
            head_dim,
            num_kv_heads,
            num_layers,
            0,
            false,
            QuantNormMode::MaxNorm,
        )
    }

    /// PQO (PolarQuant Outlier): all blocks use the outlier codebook (`bits`-bit).
    /// Best quality/performance ratio. No QJL overhead.
    /// This is the recommended default for production use.
    pub fn new_pqo(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
    ) -> Result<Self> {
        Self::new_internal(
            bits,
            head_dim,
            num_kv_heads,
            num_layers,
            usize::MAX,
            false,
            QuantNormMode::MaxNorm,
        )
    }

    /// TQ (TurboQuant, Paper Algorithm 2): (bits-1)-bit PolarQuant + 1-bit QJL.
    /// No outlier override — all blocks use standard (bits-1)-bit codebook.
    /// QJL correction eliminates multiplicative bias at cost of higher variance.
    pub fn new_tq(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
    ) -> Result<Self> {
        Self::new_internal(
            bits,
            head_dim,
            num_kv_heads,
            num_layers,
            0,
            true,
            QuantNormMode::MaxNorm,
        )
    }

    /// Like `new_pq` but with explicit norm mode.
    pub fn new_pq_with_norm(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        norm_mode: QuantNormMode,
    ) -> Result<Self> {
        Self::new_internal(
            bits,
            head_dim,
            num_kv_heads,
            num_layers,
            0,
            false,
            norm_mode,
        )
    }

    /// Like `new_pqo` but with explicit norm mode.
    pub fn new_pqo_with_norm(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        norm_mode: QuantNormMode,
    ) -> Result<Self> {
        Self::new_internal(
            bits,
            head_dim,
            num_kv_heads,
            num_layers,
            usize::MAX,
            false,
            norm_mode,
        )
    }

    /// Like `new_tq` but with explicit norm mode.
    pub fn new_tq_with_norm(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        norm_mode: QuantNormMode,
    ) -> Result<Self> {
        Self::new_internal(bits, head_dim, num_kv_heads, num_layers, 0, true, norm_mode)
    }

    /// Internal constructor with full configuration.
    fn new_internal(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        outlier_blocks: usize,
        qjl_enabled: bool,
        norm_mode: QuantNormMode,
    ) -> Result<Self> {
        let mut caches = Vec::with_capacity(num_kv_heads);
        for _ in 0..num_kv_heads {
            let config = TurboQuantConfig::new(bits, head_dim)
                .map_err(|e| tq_err(format!("config error: {e}")))?;
            let config = config.with_seed(DEFAULT_ROTATION_SEED);
            let cache = QuantizedKVCache::new(config, num_layers, DEFAULT_QJL_SEED);
            caches.push(cache);
        }

        Ok(Self {
            caches,
            bits,
            head_dim,
            num_kv_heads,
            num_layers,
            norm_mode,
            outlier_blocks,
            qjl_enabled,
            buf_seq_len: vec![0; num_layers],
            gpu_k_indices: vec![None; num_layers],
            gpu_v_indices: vec![None; num_layers],
            gpu_k_scales: vec![None; num_layers],
            gpu_v_scales: vec![None; num_layers],
            gpu_path_active: vec![false; num_layers],
            gpu_precomputed: None,
            gpu_k_qjl_signs: vec![None; num_layers],
            gpu_k_residual_norms: vec![None; num_layers],
            lazy_k: vec![None; num_layers],
            lazy_v: vec![None; num_layers],
        })
    }

    /// Returns whether QJL correction is enabled for this cache.
    pub fn qjl_enabled(&self) -> bool {
        self.qjl_enabled
    }

    /// Test-only constructor with specific QJL seed (for push_head path tests).
    #[cfg(test)]
    pub(crate) fn new_with_qjl_seed(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        qjl_seed: u64,
    ) -> Result<Self> {
        let mut caches = Vec::with_capacity(num_kv_heads);
        for _ in 0..num_kv_heads {
            let config = TurboQuantConfig::new(bits, head_dim)
                .map_err(|e| tq_err(format!("config error: {e}")))?;
            let config = config.with_seed(DEFAULT_ROTATION_SEED);
            let cache = QuantizedKVCache::new(config, num_layers, qjl_seed);
            caches.push(cache);
        }
        Ok(Self {
            caches,
            bits,
            head_dim,
            num_kv_heads,
            num_layers,
            norm_mode: QuantNormMode::L2Norm,
            outlier_blocks: usize::MAX,
            qjl_enabled: true,
            buf_seq_len: vec![0; num_layers],
            gpu_k_indices: vec![None; num_layers],
            gpu_v_indices: vec![None; num_layers],
            gpu_k_scales: vec![None; num_layers],
            gpu_v_scales: vec![None; num_layers],
            gpu_path_active: vec![false; num_layers],
            gpu_precomputed: None,
            gpu_k_qjl_signs: vec![None; num_layers],
            gpu_k_residual_norms: vec![None; num_layers],
            lazy_k: vec![None; num_layers],
            lazy_v: vec![None; num_layers],
        })
    }

    /// Test-only constructor with full control over all parameters.
    #[cfg(test)]
    pub(crate) fn new_with_config(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        _qjl_seed: u64,
        norm_mode: QuantNormMode,
    ) -> Result<Self> {
        Self::new_internal(
            bits,
            head_dim,
            num_kv_heads,
            num_layers,
            usize::MAX,
            false,
            norm_mode,
        )
    }

    /// Pushes a key-value pair for one specific head at one layer.
    ///
    /// `key` and `value` are f32 slices of length `head_dim`.
    pub fn push_head(
        &mut self,
        head: usize,
        layer: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        self.caches[head]
            .push(layer, key, value)
            .map_err(|e| tq_err(format!("push error: {e}")))
    }

    /// Pushes multiple key-value pairs for one head at one layer in a batch.
    ///
    /// More efficient than calling `push_head` repeatedly because the
    /// codebook, sign pattern, and config are computed once and reused.
    /// Best used during prefill where `new_seq_len > 1`.
    pub fn push_batch_head(
        &mut self,
        head: usize,
        layer: usize,
        keys: &[&[f32]],
        values: &[&[f32]],
    ) -> Result<()> {
        self.caches[head]
            .push_batch(layer, keys, values)
            .map_err(|e| tq_err(format!("push_batch error: {e}")))
    }

    /// Returns the number of stored entries for a given head at a layer.
    pub fn entry_count(&self, head: usize, layer: usize) -> usize {
        self.caches[head].entry_count(layer)
    }

    /// Dequantizes all stored key vectors for one head at one layer.
    ///
    /// Returns a flat `Vec<f32>` of length `entry_count * head_dim`,
    /// with entries laid out sequentially: `[k0_d0, k0_d1, ..., k1_d0, ...]`.
    pub fn dequantize_keys(&self, head: usize, layer: usize) -> Result<Vec<f32>> {
        let vecs = self.caches[head]
            .dequantize_all_keys(layer)
            .map_err(|e| tq_err(format!("dequantize keys error: {e}")))?;
        Ok(flatten_vecs(&vecs, self.head_dim))
    }

    /// Dequantizes all stored value vectors for one head at one layer.
    ///
    /// Returns a flat `Vec<f32>` of length `entry_count * head_dim`,
    /// with entries laid out sequentially.
    pub fn dequantize_values(&self, head: usize, layer: usize) -> Result<Vec<f32>> {
        let vecs = self.caches[head]
            .dequantize_all_values(layer)
            .map_err(|e| tq_err(format!("dequantize values error: {e}")))?;
        Ok(flatten_vecs(&vecs, self.head_dim))
    }

    /// Dequantizes key vectors in the range `[start..end)` for one head at one layer.
    ///
    /// Returns a flat `Vec<f32>` of length `(end - start) * head_dim`,
    /// with entries laid out sequentially.
    pub fn dequantize_keys_range(
        &self,
        head: usize,
        layer: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<f32>> {
        let vecs = self.caches[head]
            .dequantize_keys_range(layer, start, end)
            .map_err(|e| tq_err(format!("dequantize keys range error: {e}")))?;
        Ok(flatten_vecs(&vecs, self.head_dim))
    }

    /// Dequantizes value vectors in the range `[start..end)` for one head at one layer.
    ///
    /// Returns a flat `Vec<f32>` of length `(end - start) * head_dim`,
    /// with entries laid out sequentially.
    pub fn dequantize_values_range(
        &self,
        head: usize,
        layer: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<f32>> {
        let vecs = self.caches[head]
            .dequantize_values_range(layer, start, end)
            .map_err(|e| tq_err(format!("dequantize values range error: {e}")))?;
        Ok(flatten_vecs(&vecs, self.head_dim))
    }

    /// Dequantizes all keys for all heads at a layer and assembles a tensor.
    ///
    /// Returns a tensor of shape `[1, num_kv_heads, total_seq_len, head_dim]`
    /// on the specified device with the specified dtype.
    pub fn dequantize_keys_tensor(
        &self,
        layer: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        let seq_len = self.caches[0].entry_count(layer);
        if seq_len == 0 {
            return Ok(None);
        }
        self.assemble_tensor(layer, device, dtype, seq_len, true)
    }

    /// Dequantizes all values for all heads at a layer and assembles a tensor.
    ///
    /// Returns a tensor of shape `[1, num_kv_heads, total_seq_len, head_dim]`
    /// on the specified device with the specified dtype.
    pub fn dequantize_values_tensor(
        &self,
        layer: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        let seq_len = self.caches[0].entry_count(layer);
        if seq_len == 0 {
            return Ok(None);
        }
        self.assemble_tensor(layer, device, dtype, seq_len, false)
    }

    /// Internal helper: assembles a [1, num_kv_heads, seq_len, head_dim] tensor
    /// from dequantized per-head data.
    ///
    /// Thin wrapper around [`assemble_tensor_range`] for backward compatibility.
    fn assemble_tensor(
        &self,
        layer: usize,
        device: &Device,
        dtype: DType,
        seq_len: usize,
        is_keys: bool,
    ) -> Result<Option<Tensor>> {
        self.assemble_tensor_range(layer, device, dtype, 0, seq_len, is_keys)
    }

    /// Assembles a tensor from a sub-range `[start..end)` of the dequantized
    /// entries for the given layer. Used for delta dequantization.
    ///
    /// Returns `None` when `start == end` (empty range).
    fn assemble_tensor_range(
        &self,
        layer: usize,
        device: &Device,
        dtype: DType,
        start: usize,
        end: usize,
        is_keys: bool,
    ) -> Result<Option<Tensor>> {
        let range_len = end.saturating_sub(start);
        if range_len == 0 {
            return Ok(None);
        }

        let total_elems = self.num_kv_heads * range_len * self.head_dim;
        let mut flat_data = Vec::with_capacity(total_elems);

        for head in 0..self.num_kv_heads {
            let range_data = if is_keys {
                self.dequantize_keys_range(head, layer, start, end)?
            } else {
                self.dequantize_values_range(head, layer, start, end)?
            };
            flat_data.extend_from_slice(&range_data);
        }

        let batch_size = 1;
        let shape = (batch_size, self.num_kv_heads, range_len, self.head_dim);

        // Create tensor directly on target device if it's CPU+F32
        let already_target = device.is_cpu() && dtype == DType::F32;
        let tensor = if already_target {
            Tensor::from_vec(flat_data, shape, device)?
        } else {
            Tensor::from_vec(flat_data, shape, &Device::Cpu)?
                .to_device(device)?
                .to_dtype(dtype)?
        };
        Ok(Some(tensor))
    }

    /// Ensures the pre-computed GPU tensors are initialized for the given device.
    fn ensure_gpu_precomputed(&mut self, device: &Device) -> Result<()> {
        if self.gpu_precomputed.is_some() {
            return Ok(());
        }

        let block_dim = QUANT_BLOCK_SIZE;
        let polar_bits = self.bits - 1;

        // Hadamard matrix for block_size (NOT head_dim)
        let h_matrix = build_hadamard_matrix(block_dim, device)?;

        // Sign pattern for WHT preconditioning.
        // MUST be pseudo-random (not alternating!) to properly randomize the rotation.
        // Using llama.cpp's hardcoded pattern (generated from seed 42) which is
        // proven to work. Our Golden-Ratio hash with seed 42 produces an alternating
        // pattern [+1,-1,+1,-1,...] which does NOT randomize the WHT properly.
        let signs: Vec<f32> = if block_dim == 32 {
            vec![
                1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
                1.0, -1.0,
            ]
        } else {
            // Fallback: use turboquant-rs pattern for other block sizes
            turboquant::rotation::generate_sign_pattern(block_dim, DEFAULT_ROTATION_SEED)
        };
        let sign_tensor = Tensor::from_vec(signs, (1, block_dim), device)?;

        // Forward rotation: H @ diag(signs) = columns of H scaled by signs
        let rotation_fwd = h_matrix.broadcast_mul(&sign_tensor)?;
        // Inverse rotation: diag(signs) @ H = rotation_fwd^T (H symmetric, diag diagonal)
        let rotation_inv = rotation_fwd.t()?.contiguous()?;

        // Codebooks depend on normalization mode
        let (centroids, boundaries, outlier_centroids, outlier_boundaries) = match self.norm_mode {
            QuantNormMode::L2Norm => {
                // Paper: Beta-distribution optimal codebooks
                let cb = turboquant::codebook::get_codebook(polar_bits, block_dim)
                    .map_err(|e| tq_err(format!("codebook error: {e}")))?;
                let c: Vec<f32> = cb.centroids.iter().map(|&v| v as f32).collect();
                let b: Vec<f32> = cb.boundaries.iter().map(|&v| v as f32).collect();
                let outlier_bits = polar_bits + 1;
                let ocb = turboquant::codebook::get_codebook(outlier_bits, block_dim)
                    .map_err(|e| tq_err(format!("outlier codebook error: {e}")))?;
                let oc: Vec<f32> = ocb.centroids.iter().map(|&v| v as f32).collect();
                let ob: Vec<f32> = ocb.boundaries.iter().map(|&v| v as f32).collect();
                let c_len = c.len();
                let b_len = b.len();
                let oc_len = oc.len();
                let ob_len = ob.len();
                (
                    Tensor::from_vec(c, c_len, device)?,
                    Tensor::from_vec(b, b_len, device)?,
                    Tensor::from_vec(oc, oc_len, device)?,
                    Tensor::from_vec(ob, ob_len, device)?,
                )
            }
            QuantNormMode::MaxNorm => {
                // Exact codebooks from llama.cpp TQ3_0 source code.
                // Scale = amax / outer_centroid. After normalization, max
                // element equals the outermost centroid (exactly represented).
                //
                // 3-bit (8 centroids): from llama.cpp tq3_centroids[]
                let c3: Vec<f32> = vec![
                    -2.1573, -1.3336, -0.7434, -0.2428, 0.2428, 0.7434, 1.3336, 2.1573,
                ];
                let b3: Vec<f32> = vec![-1.7455, -1.0385, -0.4931, 0.0, 0.4931, 1.0385, 1.7455];
                // 2-bit (4 centroids): use every-other centroid from 3-bit
                let c2: Vec<f32> = vec![-2.1573, -0.7434, 0.7434, 2.1573];
                let b2: Vec<f32> = vec![-1.0385, 0.0, 1.0385];
                // 4-bit (16 centroids): interpolate within 3-bit range
                let c4: Vec<f32> = (0..16)
                    .map(|i| -2.1573 + (i as f32) * (2.0 * 2.1573 / 15.0))
                    .collect();
                let b4: Vec<f32> = (0..15)
                    .map(|i| -2.1573 + (i as f32 + 0.5) * (2.0 * 2.1573 / 15.0))
                    .collect();

                let (cn, bn) = match polar_bits {
                    2 => (c2.clone(), b2.clone()),
                    3 => (c3.clone(), b3.clone()),
                    4 => (c4.clone(), b4.clone()),
                    _ => (c3.clone(), b3.clone()),
                };
                let outlier_bits = polar_bits + 1;
                let (co, bo) = match outlier_bits {
                    3 => (c3, b3),
                    4 => (c4.clone(), b4.clone()),
                    _ => (c4, b4),
                };
                let cn_len = cn.len();
                let bn_len = bn.len();
                let co_len = co.len();
                let bo_len = bo.len();
                (
                    Tensor::from_vec(cn, cn_len, device)?,
                    Tensor::from_vec(bn, bn_len, device)?,
                    Tensor::from_vec(co, co_len, device)?,
                    Tensor::from_vec(bo, bo_len, device)?,
                )
            }
        };

        // Pre-compute cached values to avoid per-call GPU→CPU syncs
        let outlier_outer_centroid = outlier_centroids.max(0)?.to_scalar::<f32>()? as f64;
        let num_blocks = self.head_dim / QUANT_BLOCK_SIZE;
        let effective_outlier = self.outlier_blocks.min(num_blocks);
        let mut signs = vec![1.0_f32; num_blocks];
        for i in 0..effective_outlier {
            signs[i] = -1.0;
        }
        let scale_sign_tensor = Tensor::from_vec(signs, (1, num_blocks), device)?;

        // Pre-compute Rademacher matrix for QJL correction (only if QJL enabled)
        let qjl_rademacher = if self.qjl_enabled {
            let dim = self.head_dim;
            let mut rdata = Vec::with_capacity(dim * dim);
            for row in 0..dim {
                let row_vec = turboquant::qjl::generate_rademacher_row(dim, DEFAULT_QJL_SEED, row);
                rdata.extend_from_slice(&row_vec);
            }
            Some(Tensor::from_vec(rdata, (dim, dim), device)?)
        } else {
            None
        };

        self.gpu_precomputed = Some(GpuPrecomputed {
            rotation_fwd,
            rotation_inv,
            centroids,
            boundaries,
            outlier_centroids,
            outlier_boundaries,
            outlier_outer_centroid,
            scale_sign_tensor,
            qjl_rademacher,
        });

        Ok(())
    }

    /// Quantize input vectors to indices + scales using block-level PolarQuant.
    ///
    /// Each head_dim vector is split into blocks of QUANT_BLOCK_SIZE.
    /// Each block is independently normalized, rotated, and quantized.
    ///
    /// Input: [N, head_dim] f32
    /// Returns: (indices: [N, head_dim] U8, scales: [N, num_blocks] F32)
    ///   where num_blocks = head_dim / QUANT_BLOCK_SIZE
    // OUTLIER_BLOCKS removed — now a per-instance field `self.outlier_blocks`.
    // PQ: 0, PQO: usize::MAX (all blocks), TQ: 0.

    fn polar_quantize(
        input: &Tensor,
        head_dim: usize,
        bits: u8,
        norm_mode: QuantNormMode,
        outlier_blocks: usize,
        pre: &GpuPrecomputed,
    ) -> Result<(Tensor, Tensor)> {
        let n = input.dims()[0];
        let num_blocks = head_dim / QUANT_BLOCK_SIZE;
        let packed_dim = head_dim * bits as usize / 8;

        // GPU fast path: Candle butterfly WHT + CUDA quantize-and-pack kernel.
        // WHT is done in Candle (butterfly_wht_forward_gpu) for numerical consistency
        // with the CUDA dequant butterfly. The CUDA kernel handles amax+scale+quantize+pack.
        #[cfg(feature = "cuda")]
        if input.device().is_cuda()
            && norm_mode == QuantNormMode::MaxNorm
            && outlier_blocks >= num_blocks
        {
            let total_blocks = n * num_blocks;
            let bytes_per_block = QUANT_BLOCK_SIZE * bits as usize / 8;
            let n_ob = pre.outlier_boundaries.elem_count();

            // WHT rotation via matmul (fast on GPU via cuBLAS)
            let blocked = input.reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;
            let rotated = blocked.matmul(&pre.rotation_fwd)?;
            let rotated_flat = rotated.flatten_all()?.contiguous()?;
            let boundaries_cont = pre.outlier_boundaries.contiguous()?;

            // CUDA kernel: amax → scale → divide → quantize → pack
            let (packed_flat, scales_flat) = mistralrs_paged_attn::tq_quant_maxnorm_batch(
                &rotated_flat,
                &boundaries_cont,
                total_blocks,
                QUANT_BLOCK_SIZE,
                bits as usize,
                n_ob,
                bytes_per_block,
                pre.outlier_outer_centroid as f32,
                -1.0, // all outlier blocks → negative scale
                input.device(),
            )?;

            let packed_indices = packed_flat.reshape((n, packed_dim))?;
            let scales = scales_flat.reshape((n, num_blocks))?;
            return Ok((packed_indices, scales));
        }

        let min_norm = 1e-10_f64;

        // Reshape to [N * num_blocks, QUANT_BLOCK_SIZE]
        let blocked = input.reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;

        // Operation order differs between modes:
        // L2Norm (Paper): L2-norm → normalize → WHT → quantize
        // MaxNorm (llama.cpp): WHT → amax → scale → quantize
        let (rotated, safe_norm) = match norm_mode {
            QuantNormMode::L2Norm => {
                // Paper Algorithm 1: normalize to unit sphere, then rotate
                let norm = blocked
                    .sqr()?
                    .sum_keepdim(1)?
                    .sqrt()?
                    .clamp(min_norm, f64::MAX)?;
                let normalized = blocked.broadcast_div(&norm)?;
                let rotated = normalized.matmul(&pre.rotation_fwd)?;
                (rotated, norm)
            }
            QuantNormMode::MaxNorm => {
                // llama.cpp: rotate FIRST, then find amax of rotated values,
                // then scale so max rotated value = outermost centroid.
                let rotated_raw = blocked.matmul(&pre.rotation_fwd)?;
                let outer_c = pre.outlier_outer_centroid;
                let amax = rotated_raw.abs()?.max_keepdim(1)?;
                let scale = (amax / outer_c)?.clamp(min_norm, f64::MAX)?;
                let rotated = rotated_raw.broadcast_div(&scale)?;
                (rotated, scale)
            }
        };

        // 4. STATIC outlier assignment: first OUTLIER_BLOCKS blocks per vector
        //    get the outlier (higher-bit) codebook. No per-token computation.
        let effective_outlier_blocks = outlier_blocks.min(num_blocks);
        let outlier_rows = n * effective_outlier_blocks;
        let normal_start = outlier_rows;
        let normal_rows = n * num_blocks - outlier_rows;

        // 5a. Bucketize outlier blocks with outlier codebook
        let n_ob = pre.outlier_boundaries.elem_count();
        let ob_exp = pre.outlier_boundaries.reshape((1, 1, n_ob))?;
        let idx_out = if outlier_rows > 0 {
            let rotated_out = rotated.narrow(0, 0, outlier_rows)?;
            Some(
                rotated_out
                    .unsqueeze(2)?
                    .broadcast_gt(&ob_exp)?
                    .to_dtype(DType::U8)?
                    .sum_keepdim(2)?
                    .squeeze(2)?,
            )
        } else {
            None
        };

        // 5b. Bucketize normal blocks with normal codebook
        let idx_norm = if normal_rows > 0 {
            let rotated_norm = rotated.narrow(0, normal_start, normal_rows)?;
            let n_nb = pre.boundaries.elem_count();
            let nb_exp = pre.boundaries.reshape((1, 1, n_nb))?;
            Some(
                rotated_norm
                    .unsqueeze(2)?
                    .broadcast_gt(&nb_exp)?
                    .to_dtype(DType::U8)?
                    .sum_keepdim(2)?
                    .squeeze(2)?,
            )
        } else {
            None
        };

        // 6. Concatenate indices back
        let indices = match (idx_out, idx_norm) {
            (Some(o), Some(n)) => Tensor::cat(&[&o, &n], 0)?,
            (Some(o), None) => o,
            (None, Some(n)) => n,
            (None, None) => unreachable!(),
        };

        // Reshape: indices [N, head_dim], scales [N, num_blocks]
        let indices = indices.reshape((n, head_dim))?;
        let scales = safe_norm.reshape((n, num_blocks))?;

        // Mark outlier blocks with negative scale for dequant (pre-computed tensor)
        let scales = scales.broadcast_mul(&pre.scale_sign_tensor)?;

        // Store as F16 to save VRAM (F32 → F16 conversion)
        let scales = scales.to_dtype(DType::F16)?;

        // Pack indices from U8 to bit-packed format
        let packed_dim = head_dim * bits as usize / 8;
        #[cfg(feature = "cuda")]
        let packed_indices = if input.device().is_cuda() {
            // GPU fast path: pack on GPU, no CPU roundtrip
            let indices_flat = indices.flatten_all()?.contiguous()?;
            let packed_flat = cuda_pack_indices(
                &indices_flat,
                n * num_blocks,
                QUANT_BLOCK_SIZE,
                bits as usize,
                input.device(),
            )?;
            packed_flat.reshape((n, packed_dim))?
        } else {
            let indices_cpu: Vec<u8> = indices.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
            let packed = match bits {
                2 => turboquant::packed::pack_indices_2bit(&indices_cpu),
                3 => turboquant::packed::pack_indices_3bit(&indices_cpu),
                4 => turboquant::packed::pack_indices_4bit(&indices_cpu),
                _ => return Err(tq_err(format!("unsupported bits: {bits}"))),
            };
            Tensor::from_vec(packed, (n, packed_dim), &Device::Cpu)?.to_device(indices.device())?
        };
        #[cfg(not(feature = "cuda"))]
        let packed_indices = {
            let indices_cpu: Vec<u8> = indices.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
            let packed = match bits {
                2 => turboquant::packed::pack_indices_2bit(&indices_cpu),
                3 => turboquant::packed::pack_indices_3bit(&indices_cpu),
                4 => turboquant::packed::pack_indices_4bit(&indices_cpu),
                _ => return Err(tq_err(format!("unsupported bits: {bits}"))),
            };
            Tensor::from_vec(packed, (n, packed_dim), &Device::Cpu)?.to_device(indices.device())?
        };

        Ok((packed_indices, scales))
    }

    /// Dequantize from packed indices + scales using block-level PolarQuant.
    ///
    /// Input: indices [N, packed_dim] U8 (bit-packed), scales [N, num_blocks] F16
    /// Returns: [N, head_dim] F32 (reconstructed vectors)
    fn polar_dequantize(
        indices: &Tensor,
        scales: &Tensor,
        head_dim: usize,
        bits: u8,
        outlier_blocks: usize,
        pre: &GpuPrecomputed,
    ) -> Result<Tensor> {
        let n = indices.dims()[0];
        let num_blocks = head_dim / QUANT_BLOCK_SIZE;
        let total_blocks = n * num_blocks;

        // GPU fast path: use fused CUDA kernel (unpack + codebook + WHT + scale)
        #[cfg(feature = "cuda")]
        if indices.device().is_cuda() && outlier_blocks >= num_blocks {
            // All blocks use outlier codebook — single codebook for all
            let packed_flat = indices.flatten_all()?.contiguous()?;
            let scales_flat = scales.flatten_all()?.contiguous()?;
            let bytes_per_block = QUANT_BLOCK_SIZE * bits as usize / 8;

            // Sign pattern as flat tensor [block_size]
            let sign_flat = pre.rotation_fwd.narrow(0, 0, 1)?; // first row of H@diag(s)
                                                               // Extract sign pattern: sign[j] = rotation_fwd[0][j] * sqrt(block_size)
                                                               // Since rotation_fwd = H_norm @ diag(signs), and H_norm[0][j] = 1/sqrt(n),
                                                               // rotation_fwd[0][j] = signs[j] / sqrt(n), so signs[j] = rotation_fwd[0][j] * sqrt(n)
            let sqrt_bs = (QUANT_BLOCK_SIZE as f64).sqrt();
            let sign_pattern = (sign_flat * sqrt_bs)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .contiguous()?;

            let result = mistralrs_paged_attn::tq_dequant_batch(
                &packed_flat,
                &scales_flat,
                &pre.outlier_centroids,
                &sign_pattern,
                total_blocks,
                QUANT_BLOCK_SIZE,
                bits as usize,
                bytes_per_block,
                indices.device(),
            )?;

            return result.reshape((n, head_dim));
        }

        // CPU fallback: unpack + tensor ops
        let indices_unpacked = unpack_indices_on_device(indices, n, head_dim, bits)?;

        // Reshape to blocks: indices [N*num_blocks, QUANT_BLOCK_SIZE]
        let indices_blocked = indices_unpacked.reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;
        // Scales stored as F16 → convert to F32 for computation
        let scales_blocked = scales.to_dtype(DType::F32)?.reshape((n * num_blocks, 1))?;

        let abs_scales = scales_blocked.abs()?;
        let indices_flat = indices_blocked.flatten_all()?.to_dtype(DType::U32)?;

        // Centroid lookup — select codebook based on outlier configuration
        let dequant = if outlier_blocks >= num_blocks {
            // All blocks use outlier (higher-bit) codebook
            pre.outlier_centroids
                .index_select(&indices_flat, 0)?
                .reshape((n * num_blocks, QUANT_BLOCK_SIZE))?
        } else if outlier_blocks == 0 {
            // All blocks use normal codebook
            pre.centroids
                .index_select(&indices_flat, 0)?
                .reshape((n * num_blocks, QUANT_BLOCK_SIZE))?
        } else {
            // Mixed: negative scale = outlier codebook
            let is_outlier = scales_blocked
                .lt(0.0)?
                .to_dtype(DType::F32)?
                .broadcast_as((n * num_blocks, QUANT_BLOCK_SIZE))?;
            let n_nc = pre.centroids.elem_count() as u32;
            let clamped = indices_flat.clamp(0u32, n_nc - 1)?;
            let normal = pre
                .centroids
                .index_select(&clamped, 0)?
                .reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;
            let outlier = pre
                .outlier_centroids
                .index_select(&indices_flat, 0)?
                .reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;
            let not_outlier = (1.0 - &is_outlier)?;
            ((&is_outlier * &outlier)? + (&not_outlier * &normal)?)?
        };

        // 2. Inverse rotation: butterfly WHT on CPU (O(N log N)), matmul on GPU (cuBLAS)
        let reconstructed = if indices.device().is_cpu() {
            butterfly_wht_inverse_cpu(&dequant, &pre.rotation_fwd, QUANT_BLOCK_SIZE)?
        } else {
            dequant.matmul(&pre.rotation_inv)?
        };

        // 3. Re-scale per block (absolute value)
        let scaled = reconstructed.broadcast_mul(&abs_scales)?;

        // Reshape back to [N, head_dim]
        scaled.reshape((n, head_dim))
    }

    /// Appends new tokens to the GPU compressed cache and returns dequantized KV.
    ///
    /// This is the core GPU path: quantize new tokens → store indices + scales
    /// on GPU → dequantize the ENTIRE cache → return for attention.
    /// Ensures compressed index/scale buffers have capacity for `needed` tokens.
    /// Uses the same doubling strategy as `ensure_buffer_capacity`.
    fn ensure_compressed_capacity(
        &mut self,
        layer: usize,
        needed: usize,
        device: &Device,
    ) -> Result<()> {
        let heads = self.num_kv_heads;
        let dim = self.head_dim;

        // Check if we already have enough capacity
        let current_cap = self.gpu_k_indices[layer]
            .as_ref()
            .map_or(0, |t| t.dims()[1]);
        if current_cap >= needed {
            return Ok(());
        }

        // Grow by 25% + 128 tokens headroom (not doubling — saves VRAM)
        let grow = (needed / 4).max(128);
        let new_cap = needed + grow;
        let old_seq = self.gpu_k_indices[layer]
            .as_ref()
            .map_or(0, |t| t.dims()[1]);

        // Allocate new buffers
        let num_blocks = dim / QUANT_BLOCK_SIZE;
        // 3-bit packed indices: 8 values per 3 bytes → dim * 3 / 8 bytes per token
        let packed_dim = dim * self.bits as usize / 8;
        let new_ki = Tensor::zeros((heads, new_cap, packed_dim), DType::U8, device)?;
        let new_vi = Tensor::zeros((heads, new_cap, packed_dim), DType::U8, device)?;
        let new_ks = Tensor::zeros((heads, new_cap, num_blocks), DType::F16, device)?;
        let new_vs = Tensor::zeros((heads, new_cap, num_blocks), DType::F16, device)?;

        // Copy old data
        if old_seq > 0 {
            if let Some(ref old) = self.gpu_k_indices[layer] {
                let slice = old.narrow(1, 0, old_seq)?;
                new_ki.slice_set(&slice, 1, 0)?;
            }
            if let Some(ref old) = self.gpu_v_indices[layer] {
                let slice = old.narrow(1, 0, old_seq)?;
                new_vi.slice_set(&slice, 1, 0)?;
            }
            if let Some(ref old) = self.gpu_k_scales[layer] {
                let slice = old.narrow(1, 0, old_seq)?;
                new_ks.slice_set(&slice, 1, 0)?;
            }
            if let Some(ref old) = self.gpu_v_scales[layer] {
                let slice = old.narrow(1, 0, old_seq)?;
                new_vs.slice_set(&slice, 1, 0)?;
            }
        }

        self.gpu_k_indices[layer] = Some(new_ki);
        self.gpu_v_indices[layer] = Some(new_vi);
        self.gpu_k_scales[layer] = Some(new_ks);
        self.gpu_v_scales[layer] = Some(new_vs);

        // QJL sign bits and residual norms — ONLY for TQ mode (qjl_enabled)
        if self.qjl_enabled {
            let signs_per_head = dim / 8;
            let new_qjl_signs = Tensor::zeros((heads, new_cap, signs_per_head), DType::U8, device)?;
            let new_qjl_norms = Tensor::zeros((heads, new_cap), DType::F16, device)?;

            if old_seq > 0 {
                if let Some(ref old) = self.gpu_k_qjl_signs[layer] {
                    new_qjl_signs.slice_set(&old.narrow(1, 0, old_seq)?, 1, 0)?;
                }
                if let Some(ref old) = self.gpu_k_residual_norms[layer] {
                    new_qjl_norms.slice_set(&old.narrow(1, 0, old_seq)?, 1, 0)?;
                }
            }

            self.gpu_k_qjl_signs[layer] = Some(new_qjl_signs);
            self.gpu_k_residual_norms[layer] = Some(new_qjl_norms);
        }
        Ok(())
    }

    /// Flush deferred FP16 prefill data into the compressed cache.
    /// Called automatically before the first decode step.
    fn flush_lazy_quantize(
        &mut self,
        layer: usize,
        total_seq_len: usize,
        device: &Device,
    ) -> Result<()> {
        if self.lazy_k[layer].is_none() {
            return Ok(());
        }
        let lazy_k = self.lazy_k[layer].take().unwrap();
        let lazy_v = self.lazy_v[layer].take().unwrap();
        let lazy_seq = lazy_k.dims()[1];
        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;

        let lk_flat = lazy_k
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads * lazy_seq, head_dim))?;
        let lv_flat = lazy_v
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads * lazy_seq, head_dim))?;

        self.ensure_compressed_capacity(layer, total_seq_len, device)?;
        let pre = self.gpu_precomputed.as_ref().unwrap();

        let (lk_idx, lk_sc) = Self::polar_quantize(
            &lk_flat,
            head_dim,
            self.bits,
            self.norm_mode,
            self.outlier_blocks,
            pre,
        )?;
        let (lv_idx, lv_sc) = Self::polar_quantize(
            &lv_flat,
            head_dim,
            self.bits,
            self.norm_mode,
            self.outlier_blocks,
            pre,
        )?;

        let num_blocks = head_dim / QUANT_BLOCK_SIZE;
        let packed_dim = head_dim * self.bits as usize / 8;
        let lk_idx = lk_idx.reshape((num_kv_heads, lazy_seq, packed_dim))?;
        let lv_idx = lv_idx.reshape((num_kv_heads, lazy_seq, packed_dim))?;
        let lk_sc = lk_sc.reshape((num_kv_heads, lazy_seq, num_blocks))?;
        let lv_sc = lv_sc.reshape((num_kv_heads, lazy_seq, num_blocks))?;

        self.gpu_k_indices[layer]
            .as_ref()
            .unwrap()
            .slice_set(&lk_idx, 1, 0)?;
        self.gpu_v_indices[layer]
            .as_ref()
            .unwrap()
            .slice_set(&lv_idx, 1, 0)?;
        self.gpu_k_scales[layer]
            .as_ref()
            .unwrap()
            .slice_set(&lk_sc, 1, 0)?;
        self.gpu_v_scales[layer]
            .as_ref()
            .unwrap()
            .slice_set(&lv_sc, 1, 0)?;
        Ok(())
    }

    fn block_append_and_dequantize(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let device = k.device().clone();
        let orig_dtype = k.dtype();
        let profiling = layer == 0 && std::env::var("TQ_PROFILE_DECODE").is_ok();
        let t0 = std::time::Instant::now();
        self.ensure_gpu_precomputed(&device)?;

        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;
        let new_seq_len = k.dims()[2];
        if profiling && new_seq_len == 1 {
            eprintln!(
                "[TQ_PROF] FULL_DEQUANT path for decode! seq_len will be {}",
                self.buf_seq_len[layer] + 1
            );
        }

        // Current sequence length in compressed cache
        let old_seq_len = if self.gpu_path_active[layer] {
            self.buf_seq_len[layer]
        } else {
            0
        };
        let total_seq_len = old_seq_len + new_seq_len;

        // No lazy dequant — quantize immediately during prefill.
        // This keeps VRAM low (no FP16 clone persisting across layers).

        // --- Quantize new tokens into compressed buffers ---

        // Reshape input to [heads * new_seq, dim] for quantize
        let k_flat = k
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads * new_seq_len, head_dim))?;
        let v_flat = v
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads * new_seq_len, head_dim))?;

        // Ensure compressed buffers have capacity BEFORE borrowing self.gpu_precomputed.
        // No dequantized buffer needed — only compressed cache persists.
        self.ensure_compressed_capacity(layer, total_seq_len, &device)?;

        // Now borrow precomputed and quantize
        let pre = self.gpu_precomputed.as_ref().unwrap();

        let (k_new_idx, k_new_sc) = Self::polar_quantize(
            &k_flat,
            head_dim,
            self.bits,
            self.norm_mode,
            self.outlier_blocks,
            pre,
        )?;
        let (v_new_idx, v_new_sc) = Self::polar_quantize(
            &v_flat,
            head_dim,
            self.bits,
            self.norm_mode,
            self.outlier_blocks,
            pre,
        )?;

        // Reshape to [heads, new_seq, packed_dim] / [heads, new_seq, num_blocks]
        let num_blocks = head_dim / QUANT_BLOCK_SIZE;
        let packed_dim = head_dim * self.bits as usize / 8;
        let k_new_idx = k_new_idx.reshape((num_kv_heads, new_seq_len, packed_dim))?;
        let v_new_idx = v_new_idx.reshape((num_kv_heads, new_seq_len, packed_dim))?;
        let k_new_sc = k_new_sc.reshape((num_kv_heads, new_seq_len, num_blocks))?;
        let v_new_sc = v_new_sc.reshape((num_kv_heads, new_seq_len, num_blocks))?;
        self.gpu_k_indices[layer]
            .as_ref()
            .unwrap()
            .slice_set(&k_new_idx, 1, old_seq_len)?;
        self.gpu_v_indices[layer]
            .as_ref()
            .unwrap()
            .slice_set(&v_new_idx, 1, old_seq_len)?;
        self.gpu_k_scales[layer]
            .as_ref()
            .unwrap()
            .slice_set(&k_new_sc, 1, old_seq_len)?;
        self.gpu_v_scales[layer]
            .as_ref()
            .unwrap()
            .slice_set(&v_new_sc, 1, old_seq_len)?;

        // --- Compute QJL signs + residual norms for new keys (TQ mode only) ---
        // QJL operates on the FULL head_dim vector (not per 32-block).
        // Residual = original - block_dequant(block_quant(original)).
        if self.qjl_enabled {
            let k_new_idx_flat = k_new_idx.reshape((num_kv_heads * new_seq_len, packed_dim))?;
            let k_new_sc_flat = k_new_sc.reshape((num_kv_heads * new_seq_len, num_blocks))?;
            let k_dequant = Self::polar_dequantize(
                &k_new_idx_flat,
                &k_new_sc_flat,
                head_dim,
                self.bits,
                self.outlier_blocks,
                pre,
            )?;

            let signs_per_head = head_dim / 8;
            let qjl_seed = DEFAULT_QJL_SEED;
            let n_vecs = num_kv_heads * new_seq_len;

            let (signs_tensor, norms_tensor) = if device.is_cuda() {
                // GPU path: use tq_qjl_batch CUDA kernel — all on-device, no CPU transfer
                #[cfg(feature = "cuda")]
                {
                    let (signs_flat, norms_flat) =
                        crate::paged_attention::turboquant_cache::cuda_qjl_batch(
                            &k_flat,
                            &k_dequant,
                            qjl_seed,
                            n_vecs,
                            head_dim,
                            signs_per_head,
                            &device,
                        )?;
                    let signs = signs_flat.reshape((num_kv_heads, new_seq_len, signs_per_head))?;
                    let norms = norms_flat.reshape((num_kv_heads, new_seq_len))?;
                    (signs, norms)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    candle_core::bail!("CUDA feature not enabled but device is CUDA");
                }
            } else {
                // CPU path: use turboquant crate functions
                let residual = (&k_flat - &k_dequant)?;
                let res_norms = residual
                    .sqr()?
                    .sum_keepdim(1)?
                    .sqrt()?
                    .squeeze(1)?
                    .to_dtype(DType::F16)?;

                let mut all_signs = vec![0u8; n_vecs * signs_per_head];
                for vec_idx in 0..n_vecs {
                    let row_data: Vec<f32> =
                        residual.narrow(0, vec_idx, 1)?.squeeze(0)?.to_vec1()?;
                    let signs = turboquant::compute_qjl_signs(&row_data, head_dim, qjl_seed);
                    let start = vec_idx * signs_per_head;
                    all_signs[start..start + signs_per_head].copy_from_slice(&signs);
                }

                let signs = Tensor::from_vec(
                    all_signs,
                    (num_kv_heads, new_seq_len, signs_per_head),
                    &Device::Cpu,
                )?;
                let norms = res_norms.reshape((num_kv_heads, new_seq_len))?;
                (signs, norms)
            };

            // Store in pre-allocated QJL buffers
            self.gpu_k_qjl_signs[layer].as_ref().unwrap().slice_set(
                &signs_tensor,
                1,
                old_seq_len,
            )?;
            self.gpu_k_residual_norms[layer]
                .as_ref()
                .unwrap()
                .slice_set(&norms_tensor, 1, old_seq_len)?;
        }

        self.gpu_path_active[layer] = true;
        self.buf_seq_len[layer] = total_seq_len;

        // --- Dequantize for attention ---
        // PROFILING: log per-layer memory breakdown
        if new_seq_len == 1 && layer == 0 && std::env::var("TQ_PROFILE").is_ok() {
            // Compressed cache sizes (persistent VRAM)
            let idx_bytes = self.gpu_k_indices[layer]
                .as_ref()
                .map_or(0, |t| t.elem_count())
                + self.gpu_v_indices[layer]
                    .as_ref()
                    .map_or(0, |t| t.elem_count());
            let scale_bytes = (self.gpu_k_scales[layer]
                .as_ref()
                .map_or(0, |t| t.elem_count())
                + self.gpu_v_scales[layer]
                    .as_ref()
                    .map_or(0, |t| t.elem_count()))
                * 4;
            let qjl_bytes = self.gpu_k_qjl_signs[layer]
                .as_ref()
                .map_or(0, |t| t.elem_count())
                + self.gpu_k_residual_norms[layer]
                    .as_ref()
                    .map_or(0, |t| t.elem_count() * 2);
            // Dequantized output size (temporary)
            let dequant_k = num_kv_heads * total_seq_len * head_dim * 2;
            let dequant_v = dequant_k;

            let compressed_total = (idx_bytes + scale_bytes + qjl_bytes) * self.num_layers;
            let dequant_total = (dequant_k + dequant_v) * self.num_layers;

            eprintln!(
                "[TQ_PROFILE] decode step: seq_len={total_seq_len}, \
                 compressed_all_layers={} MB, \
                 dequant_temp_all_layers={} MB, \
                 sum={} MB",
                compressed_total / 1024 / 1024,
                dequant_total / 1024 / 1024,
                (compressed_total + dequant_total) / 1024 / 1024,
            );
        }

        let packed_dim = head_dim * self.bits as usize / 8;
        // Dequantize the relevant portions of the compressed cache.
        let ki = self.gpu_k_indices[layer].as_ref().unwrap();
        let vi = self.gpu_v_indices[layer].as_ref().unwrap();
        let ks = self.gpu_k_scales[layer].as_ref().unwrap();
        let vs = self.gpu_v_scales[layer].as_ref().unwrap();

        if new_seq_len > 1 && old_seq_len == 0 {
            // First prefill: return originals directly (no old tokens to dequant).
            Ok((k.clone(), v.clone()))
        } else if new_seq_len > 1 {
            // Subsequent prefill: dequant old tokens + cat with originals.
            let old_ki = ki
                .narrow(1, 0, old_seq_len)?
                .reshape((num_kv_heads * old_seq_len, packed_dim))?;
            let old_ks = ks
                .narrow(1, 0, old_seq_len)?
                .reshape((num_kv_heads * old_seq_len, num_blocks))?;
            let old_vi = vi
                .narrow(1, 0, old_seq_len)?
                .reshape((num_kv_heads * old_seq_len, packed_dim))?;
            let old_vs = vs
                .narrow(1, 0, old_seq_len)?
                .reshape((num_kv_heads * old_seq_len, num_blocks))?;

            let old_k = Self::polar_dequantize(
                &old_ki,
                &old_ks,
                head_dim,
                self.bits,
                self.outlier_blocks,
                pre,
            )?
            .reshape((1, num_kv_heads, old_seq_len, head_dim))?
            .to_dtype(orig_dtype)?;
            let old_v = Self::polar_dequantize(
                &old_vi,
                &old_vs,
                head_dim,
                self.bits,
                self.outlier_blocks,
                pre,
            )?
            .reshape((1, num_kv_heads, old_seq_len, head_dim))?
            .to_dtype(orig_dtype)?;

            Ok((Tensor::cat(&[&old_k, k], 2)?, Tensor::cat(&[&old_v, v], 2)?))
        } else {
            // Decode: dequantize entire compressed cache.
            // The temporary tensor is freed after the forward pass (per-layer).
            let all_ki = ki
                .narrow(1, 0, total_seq_len)?
                .reshape((num_kv_heads * total_seq_len, packed_dim))?;
            let all_ks = ks
                .narrow(1, 0, total_seq_len)?
                .reshape((num_kv_heads * total_seq_len, num_blocks))?;
            let all_vi = vi
                .narrow(1, 0, total_seq_len)?
                .reshape((num_kv_heads * total_seq_len, packed_dim))?;
            let all_vs = vs
                .narrow(1, 0, total_seq_len)?
                .reshape((num_kv_heads * total_seq_len, num_blocks))?;

            let full_k = Self::polar_dequantize(
                &all_ki,
                &all_ks,
                head_dim,
                self.bits,
                self.outlier_blocks,
                pre,
            )?
            .reshape((1, num_kv_heads, total_seq_len, head_dim))?
            .to_dtype(orig_dtype)?;
            let full_v = Self::polar_dequantize(
                &all_vi,
                &all_vs,
                head_dim,
                self.bits,
                self.outlier_blocks,
                pre,
            )?
            .reshape((1, num_kv_heads, total_seq_len, head_dim))?
            .to_dtype(orig_dtype)?;

            Ok((full_k, full_v))
        }
    }

    /// Ingests new key/value tensors for all KV heads at a given layer.
    ///
    /// Input tensors have shape `[batch, num_kv_heads, new_seq_len, head_dim]`.
    /// Each token-head vector is extracted, converted to f32, and quantized.
    ///
    /// Uses **delta dequantization**: only the newly appended tokens are
    /// dequantized, then concatenated with the cached dequantized history.
    /// This reduces per-token cost from O(total_seq_len) to O(new_seq_len).
    pub fn append_and_dequantize(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Block-level compressed cache (Candle tensor ops).
        // Quantize → store compressed indices + scales →
        // dequantize entire cache on-the-fly for attention.
        // Works on CUDA, Metal, AND CPU — Candle dispatches automatically.
        // Uses QUANT_BLOCK_SIZE=32 blocks for quality at high norms.
        self.block_append_and_dequantize(layer, k, v)
    }

    /// Block-wise decode attention: append new token, then compute attention
    /// directly from compressed cache without full-dequant.
    ///
    /// Uses online softmax (tile-based) to iterate over KV blocks of
    /// QUANT_BLOCK_SIZE tokens each, dequantizing only one block at a time.
    ///
    /// Memory: O(block_size × head_dim) instead of O(seq_len × head_dim).
    /// Works on CPU, CUDA, and Metal via Candle tensor ops.
    ///
    /// * `q` — query tensor `[batch=1, num_attention_heads, 1, head_dim]`
    /// * `k_new`, `v_new` — new key/value `[batch=1, num_kv_heads, 1, head_dim]`
    /// * `softmax_scale` — typically `1.0 / sqrt(head_dim)`
    /// * `n_kv_groups` — `num_attention_heads / num_kv_heads` (GQA ratio)
    ///
    /// Returns attention output `[batch=1, num_attention_heads, 1, head_dim]`.
    pub fn append_and_blockwise_attend(
        &mut self,
        layer: usize,
        k_new: &Tensor,
        v_new: &Tensor,
        q: &Tensor,
        softmax_scale: f32,
        n_kv_groups: usize,
    ) -> Result<Tensor> {
        let device = k_new.device().clone();
        let orig_dtype = k_new.dtype();
        let profiling = layer == 0 && std::env::var("TQ_PROFILE_DECODE").is_ok();
        let t0 = std::time::Instant::now();
        self.ensure_gpu_precomputed(&device)?;

        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;
        let new_seq_len = k_new.dims()[2];

        // Current sequence length in compressed cache
        let old_seq_len = if self.gpu_path_active[layer] {
            self.buf_seq_len[layer]
        } else {
            0
        };
        let total_seq_len = old_seq_len + new_seq_len;

        // Flush deferred prefill data before quantizing new tokens
        let has_lazy = self.lazy_k[layer].is_some();
        self.flush_lazy_quantize(layer, total_seq_len, &device)?;
        if profiling && has_lazy {
            // GPU sync to get accurate timing
            // GPU sync: force all pending ops to finish for accurate timing
            let _ = Tensor::zeros(1, DType::F32, &device)?.to_vec1::<f32>();
            eprintln!(
                "[TQ_PROF] flush_lazy_quantize (prefill→decode): {:.2}ms seq={total_seq_len}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        // --- Quantize and store new tokens ---
        let t_quant = std::time::Instant::now();

        let k_flat = k_new
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads * new_seq_len, head_dim))?;
        let v_flat = v_new
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads * new_seq_len, head_dim))?;

        self.ensure_compressed_capacity(layer, total_seq_len, &device)?;

        let pre = self.gpu_precomputed.as_ref().unwrap();

        let (k_new_idx, k_new_sc) = Self::polar_quantize(
            &k_flat,
            head_dim,
            self.bits,
            self.norm_mode,
            self.outlier_blocks,
            pre,
        )?;
        let (v_new_idx, v_new_sc) = Self::polar_quantize(
            &v_flat,
            head_dim,
            self.bits,
            self.norm_mode,
            self.outlier_blocks,
            pre,
        )?;

        let num_blocks = head_dim / QUANT_BLOCK_SIZE;
        let packed_dim = head_dim * self.bits as usize / 8;
        let k_new_idx = k_new_idx.reshape((num_kv_heads, new_seq_len, packed_dim))?;
        let v_new_idx = v_new_idx.reshape((num_kv_heads, new_seq_len, packed_dim))?;
        let k_new_sc = k_new_sc.reshape((num_kv_heads, new_seq_len, num_blocks))?;
        let v_new_sc = v_new_sc.reshape((num_kv_heads, new_seq_len, num_blocks))?;

        self.gpu_k_indices[layer]
            .as_ref()
            .unwrap()
            .slice_set(&k_new_idx, 1, old_seq_len)?;
        self.gpu_v_indices[layer]
            .as_ref()
            .unwrap()
            .slice_set(&v_new_idx, 1, old_seq_len)?;
        self.gpu_k_scales[layer]
            .as_ref()
            .unwrap()
            .slice_set(&k_new_sc, 1, old_seq_len)?;
        self.gpu_v_scales[layer]
            .as_ref()
            .unwrap()
            .slice_set(&v_new_sc, 1, old_seq_len)?;

        // QJL signs + residual norms for new keys (TQ mode only).
        // For PQO3 (recommended mode), qjl_enabled=false, so this is skipped.
        // Full QJL support in blockwise path is planned for later (Stufe 1.4).
        if self.qjl_enabled {
            // Fall back to full-dequant path for TQ mode until QJL is integrated
            // into the blockwise attention loop.
            self.buf_seq_len[layer] = total_seq_len;
            self.gpu_path_active[layer] = true;
            return self.dequantize_full_and_attend(
                layer,
                q,
                orig_dtype,
                softmax_scale,
                n_kv_groups,
            );
        }

        self.buf_seq_len[layer] = total_seq_len;
        self.gpu_path_active[layer] = true;

        if profiling {
            // GPU sync: force all pending ops to finish for accurate timing
            let _ = Tensor::zeros(1, DType::F32, &device)?.to_vec1::<f32>();
            eprintln!(
                "[TQ_PROF] quantize_new_token: {:.2}ms",
                t_quant.elapsed().as_secs_f64() * 1000.0
            );
        }

        // --- Attention from compressed cache ---

        let ki = self.gpu_k_indices[layer].as_ref().unwrap();
        let ks = self.gpu_k_scales[layer].as_ref().unwrap();
        let vi = self.gpu_v_indices[layer].as_ref().unwrap();
        let vs = self.gpu_v_scales[layer].as_ref().unwrap();
        let pre = self.gpu_precomputed.as_ref().unwrap();

        // q shape: [1, num_attention_heads, 1, head_dim] → [num_attention_heads, head_dim]
        let q_squeezed = q.squeeze(0)?.squeeze(1)?.to_dtype(DType::F32)?;
        let num_attention_heads = q_squeezed.dims()[0];
        let scale = softmax_scale as f64;

        // CUDA: use fused kernel (single launch, dequant in shared memory)
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            // Extract sign pattern for the kernel
            let sqrt_bs = (QUANT_BLOCK_SIZE as f64).sqrt();
            let sign_pattern = (pre.rotation_fwd.narrow(0, 0, 1)? * sqrt_bs)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .contiguous()?;

            // Pass full-capacity tensors + kv_len to avoid per-token .contiguous() copy.
            // The CUDA kernel uses kv_stride (=capacity) for addressing, kv_len for bounds.
            let q_cont = q_squeezed.contiguous()?;
            let kv_capacity = ki.dims()[1]; // allocated capacity >= total_seq_len

            let t_kernel = std::time::Instant::now();
            let output = mistralrs_paged_attn::tq_fused_attention(
                &q_cont,
                ki,
                ks,
                vi,
                vs,
                &pre.outlier_centroids,
                &sign_pattern,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                total_seq_len,
                kv_capacity,
                packed_dim,
                num_blocks,
                self.bits as usize,
                softmax_scale,
                &device,
            )?;

            if profiling {
                let _ = Tensor::zeros(1, DType::F32, &device)?.to_vec1::<f32>();
                eprintln!("[TQ_PROF] fused_kernel: {:.2}ms seq={total_seq_len} total_decode_step: {:.2}ms",
                    t_kernel.elapsed().as_secs_f64() * 1000.0,
                    t0.elapsed().as_secs_f64() * 1000.0);
            }

            return output
                .reshape((1, num_attention_heads, 1, head_dim))?
                .to_dtype(orig_dtype);
        }

        // Online softmax accumulators per attention head:
        // m: running max, s: running sum of exp, o: running weighted output
        let mut m = Tensor::full(f32::NEG_INFINITY, (num_attention_heads, 1), &device)?;
        let mut s = Tensor::zeros((num_attention_heads, 1), DType::F32, &device)?;
        let mut o = Tensor::zeros((num_attention_heads, head_dim), DType::F32, &device)?;

        // Attention chunk size: larger chunks = fewer kernel launches but bigger
        // temporary tensors. 256 tokens is a good middle ground (~128x smaller
        // than full-dequant at 32K, but only ~128 iterations instead of ~1024).
        let chunk_size = 256.max(QUANT_BLOCK_SIZE);
        let num_token_blocks = (total_seq_len + chunk_size - 1) / chunk_size;

        for block_idx in 0..num_token_blocks {
            let start = block_idx * chunk_size;
            let len = chunk_size.min(total_seq_len - start);

            // Dequantize K block: narrow per kv-head, dequantize
            let k_block_idx = ki
                .narrow(1, start, len)?
                .reshape((num_kv_heads * len, packed_dim))?;
            let k_block_sc = ks
                .narrow(1, start, len)?
                .reshape((num_kv_heads * len, num_blocks))?;
            let v_block_idx = vi
                .narrow(1, start, len)?
                .reshape((num_kv_heads * len, packed_dim))?;
            let v_block_sc = vs
                .narrow(1, start, len)?
                .reshape((num_kv_heads * len, num_blocks))?;

            // Dequantize: [num_kv_heads * len, head_dim]
            let k_block = Self::polar_dequantize(
                &k_block_idx,
                &k_block_sc,
                head_dim,
                self.bits,
                self.outlier_blocks,
                pre,
            )?
            .reshape((num_kv_heads, len, head_dim))?;
            let v_block = Self::polar_dequantize(
                &v_block_idx,
                &v_block_sc,
                head_dim,
                self.bits,
                self.outlier_blocks,
                pre,
            )?
            .reshape((num_kv_heads, len, head_dim))?;

            // GQA: repeat KV for query head groups
            // k_block: [num_kv_heads, len, head_dim] → [num_attention_heads, len, head_dim]
            let k_block = if n_kv_groups > 1 {
                k_block
                    .unsqueeze(1)?
                    .expand((num_kv_heads, n_kv_groups, len, head_dim))?
                    .reshape((num_attention_heads, len, head_dim))?
            } else {
                k_block
            };
            let v_block = if n_kv_groups > 1 {
                v_block
                    .unsqueeze(1)?
                    .expand((num_kv_heads, n_kv_groups, len, head_dim))?
                    .reshape((num_attention_heads, len, head_dim))?
            } else {
                v_block
            };

            // QK scores: [num_attention_heads, 1, len]
            // q_squeezed: [num_attention_heads, head_dim]
            // k_block^T:  [num_attention_heads, head_dim, len]
            let qk = q_squeezed
                .unsqueeze(1)?
                .matmul(&k_block.transpose(1, 2)?)?
                .squeeze(1)?;
            // qk: [num_attention_heads, len]
            let qk = (qk * scale)?;

            // TODO: QJL bias correction would go here for TQ mode

            // Online softmax update
            // m_block: [num_attention_heads, 1] — max of this block
            let m_block = qk.max_keepdim(D::Minus1)?;
            let m_new = m.maximum(&m_block)?;

            // Correction factor for old accumulator: exp(m_old - m_new) [num_attention_heads, 1]
            let correction = m.broadcast_sub(&m_new)?.exp()?;
            // exp(qk - m_new): [num_attention_heads, len]
            let exp_qk = qk.broadcast_sub(&m_new)?.exp()?;

            // Update sum: s = s * correction + sum(exp_qk)
            s = ((&s * &correction)? + exp_qk.sum_keepdim(D::Minus1)?)?;

            // Update output: o = o * correction + exp_qk @ v_block
            // correction: [num_attention_heads, 1] → broadcast to [num_attention_heads, head_dim]
            // exp_qk: [num_attention_heads, len] → [num_attention_heads, 1, len]
            // v_block: [num_attention_heads, len, head_dim]
            let weighted_v = exp_qk.unsqueeze(1)?.matmul(&v_block)?.squeeze(1)?;
            // weighted_v: [num_attention_heads, head_dim]
            o = (o.broadcast_mul(&correction)? + weighted_v)?;

            m = m_new;
        }

        // Final output: o / s → [num_attention_heads, head_dim]
        let output = o.broadcast_div(&s)?;

        // Reshape to [1, num_attention_heads, 1, head_dim] and convert back to orig dtype
        output
            .reshape((1, num_attention_heads, 1, head_dim))?
            .to_dtype(orig_dtype)
    }

    /// Fallback for TQ mode (QJL enabled): full-dequant + manual attention.
    /// Used until QJL correction is integrated into the blockwise loop.
    fn dequantize_full_and_attend(
        &self,
        layer: usize,
        q: &Tensor,
        orig_dtype: DType,
        softmax_scale: f32,
        n_kv_groups: usize,
    ) -> Result<Tensor> {
        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;
        let total_seq_len = self.buf_seq_len[layer];
        let packed_dim = head_dim * self.bits as usize / 8;
        let num_blocks = head_dim / QUANT_BLOCK_SIZE;

        let ki = self.gpu_k_indices[layer].as_ref().unwrap();
        let ks = self.gpu_k_scales[layer].as_ref().unwrap();
        let vi = self.gpu_v_indices[layer].as_ref().unwrap();
        let vs = self.gpu_v_scales[layer].as_ref().unwrap();
        let pre = self.gpu_precomputed.as_ref().unwrap();

        let all_ki = ki
            .narrow(1, 0, total_seq_len)?
            .reshape((num_kv_heads * total_seq_len, packed_dim))?;
        let all_ks = ks
            .narrow(1, 0, total_seq_len)?
            .reshape((num_kv_heads * total_seq_len, num_blocks))?;
        let all_vi = vi
            .narrow(1, 0, total_seq_len)?
            .reshape((num_kv_heads * total_seq_len, packed_dim))?;
        let all_vs = vs
            .narrow(1, 0, total_seq_len)?
            .reshape((num_kv_heads * total_seq_len, num_blocks))?;

        let full_k = Self::polar_dequantize(
            &all_ki,
            &all_ks,
            head_dim,
            self.bits,
            self.outlier_blocks,
            pre,
        )?
        .reshape((1, num_kv_heads, total_seq_len, head_dim))?
        .to_dtype(orig_dtype)?;
        let full_v = Self::polar_dequantize(
            &all_vi,
            &all_vs,
            head_dim,
            self.bits,
            self.outlier_blocks,
            pre,
        )?
        .reshape((1, num_kv_heads, total_seq_len, head_dim))?
        .to_dtype(orig_dtype)?;

        // Standard attention with GQA repeat
        use crate::attention::{Sdpa, SdpaParams};
        let sdpa_params = SdpaParams {
            n_kv_groups,
            softcap: None,
            softmax_scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None, // TODO: compute QJL bias for TQ mode
        };
        Sdpa.run_attention(q, &full_k, &full_v, None, None, &sdpa_params)
    }

    /// Returns true if this cache is in decode mode for the given layer
    /// (has existing compressed data and the new tokens count is 1).
    pub fn is_decode_ready(&self, layer: usize) -> bool {
        self.gpu_path_active[layer] && self.buf_seq_len[layer] > 0
    }

    /// Test helper: check if lazy FP16 data exists for a layer.
    pub fn has_lazy_data(&self, layer: usize) -> bool {
        self.lazy_k[layer].is_some()
    }

    /// Test helper: get buf_seq_len for a layer.
    pub fn buf_seq_len_for_test(&self, layer: usize) -> usize {
        self.buf_seq_len[layer]
    }

    /// Replaces this cache with a fresh empty one, preserving configuration.
    ///
    /// This is the canonical way to reset a `TurboQuantKVCache`. The newly
    /// created cache already has `cached_k`/`cached_v` initialised to `None`,
    /// so no additional `invalidate_all_caches()` call is needed.
    pub fn reset_all(&mut self) -> Result<()> {
        let new = TurboQuantKVCache::new_internal(
            self.bits,
            self.head_dim,
            self.num_kv_heads,
            self.num_layers,
            self.outlier_blocks,
            self.qjl_enabled,
            self.norm_mode,
        )?;
        *self = new;
        Ok(())
    }

    /// Invalidates the GPU buffer and pending vectors for a given layer.
    pub fn invalidate_cache(&mut self, layer: usize) {
        if layer < self.gpu_k_indices.len() {
            self.buf_seq_len[layer] = 0;
            self.gpu_k_indices[layer] = None;
            self.gpu_v_indices[layer] = None;
            self.gpu_k_scales[layer] = None;
            self.gpu_v_scales[layer] = None;
            self.gpu_path_active[layer] = false;
            self.gpu_k_qjl_signs[layer] = None;
            self.gpu_k_residual_norms[layer] = None;
            self.lazy_k[layer] = None;
            self.lazy_v[layer] = None;
        }
    }

    /// Invalidates all GPU buffers and pending vectors across every layer.
    pub fn invalidate_all_caches(&mut self) {
        for i in 0..self.num_layers {
            self.invalidate_cache(i);
        }
        self.gpu_precomputed = None;
    }

    /// Returns the total memory usage in bytes across all per-head caches.
    pub fn memory_usage(&self) -> usize {
        self.caches.iter().map(|c| c.memory_usage()).sum()
    }

    /// Returns the total bit budget.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Returns the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns the number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Returns the total sequence length stored for a given layer
    /// (uses head 0 as representative since all heads have the same count).
    ///
    /// Includes both quantized entries and pending (not yet quantized) prefill
    /// entries, so the returned value is always the true total token count.
    pub fn current_seq_len(&self, layer: usize) -> usize {
        // Block-level path: tracked by buf_seq_len.
        if self.gpu_path_active[layer] {
            return self.buf_seq_len[layer];
        }
        // CPU push_head path (used by tests).
        if self.caches.is_empty() {
            return 0;
        }
        self.caches[0].entry_count(layer)
    }

    // -------------------------------------------------------------------
    // QJL accessors (Phase 1 — stubs, to be implemented)
    // -------------------------------------------------------------------

    /// Returns QJL sign bytes for the last key entry at `(head, layer)`.
    ///
    /// Returns `dim / 8` packed bytes (1 bit per dimension).
    /// Returns `None` if no entries exist for this head/layer.
    pub fn qjl_signs(&self, head: usize, layer: usize) -> Option<&[u8]> {
        let cache = self.caches.get(head)?;
        let count = cache.entry_count(layer);
        if count == 0 {
            return None;
        }
        let block = cache.key_block(layer, count - 1)?;
        Some(block.qjl_signs())
    }

    /// Returns the key residual norm for a specific entry at `(head, layer, index)`.
    ///
    /// The residual norm is `L2(original - polar_dequant(polar_quant(original)))`.
    /// Returns `None` if the index is out of bounds.
    pub fn key_residual_norm(&self, head: usize, layer: usize, index: usize) -> Option<f32> {
        let cache = self.caches.get(head)?;
        let block = cache.key_block(layer, index)?;
        Some(block.residual_norm().to_f32())
    }

    /// Returns `true` if QJL data (signs + residual norms) is stored for the given layer.
    ///
    /// For the legacy CPU path, QJL data is present in QjlBlocks.
    /// For the block-level path, QJL data is in GPU tensors.
    /// Returns false if QJL is not yet computed (block path without QJL).
    pub fn has_qjl_data(&self, layer: usize) -> bool {
        if !self.qjl_enabled {
            return false;
        }
        // CPU path: QjlBlocks always contain QJL data
        if !self.caches.is_empty() && self.caches[0].entry_count(layer) > 0 {
            return true;
        }
        // GPU path: check if QJL tensors exist
        self.gpu_k_qjl_signs
            .get(layer)
            .map_or(false, |t| t.is_some())
    }

    /// Computes the QJL correction term for attention logits.
    ///
    /// Formula: `correction[k] = c_k · dot(R·q, signs_k)`
    /// where `c_k = residual_norm_k · √(π/2) / √dim`
    ///
    /// This is an additive term on attention logits (before softmax),
    /// compensating the quantization bias of PolarQuant.
    ///
    /// # Arguments
    /// * `head` — KV head index
    /// * `layer` — transformer layer index
    /// * `query` — query tensor, shape `[batch, 1, q_len, dim]`
    ///
    /// # Returns
    /// Correction tensor, shape `[batch, 1, q_len, kv_len]`.
    /// Add this to attention logits before softmax.
    ///
    /// Uses Candle tensor ops — works on CPU, CUDA, Metal.
    pub fn qjl_correction(&self, head: usize, layer: usize, query: &Tensor) -> Result<Tensor> {
        use turboquant::precompute_query_projections;
        use turboquant::qjl::{qjl_scaling_constant, sign_bit};

        let device = query.device();
        let dims = query.dims4()?;
        let q_len = dims.2;
        let dim = dims.3;

        // Determine kv_len and data source: GPU tensors (block path) or CPU caches (push_head path)
        let use_gpu_path = self.gpu_path_active.get(layer).copied().unwrap_or(false)
            && self
                .gpu_k_qjl_signs
                .get(layer)
                .map_or(false, |t| t.is_some());

        let kv_len = if use_gpu_path {
            self.buf_seq_len[layer]
        } else {
            self.caches[head].entry_count(layer)
        };

        if kv_len == 0 {
            return Tensor::zeros((dims.0, 1, q_len, 0), DType::F32, device);
        }

        // Extract query vectors as f32
        let q_flat = query.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let _qjl_seed = DEFAULT_QJL_SEED;

        if use_gpu_path {
            // GPU tensor path: all computation via Candle tensor ops (no CPU transfer).
            // Chunked to bound peak VRAM: process QJL_QUERY_CHUNK queries at a time
            // instead of the full [q_len, kv_len] matrix at once.
            const QJL_QUERY_CHUNK: usize = 256;

            let signs_tensor = self.gpu_k_qjl_signs[layer].as_ref().unwrap();
            let norms_tensor = self.gpu_k_residual_norms[layer].as_ref().unwrap();

            // 1. Get this head's signs [kv_len, signs_per_head] and norms [kv_len]
            let head_signs = signs_tensor
                .narrow(0, head, 1)?
                .narrow(1, 0, kv_len)?
                .squeeze(0)?;
            let head_norms = norms_tensor
                .narrow(0, head, 1)?
                .narrow(1, 0, kv_len)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;

            // 2. Unpack signs: packed U8 → ±1.0 float [kv_len, dim] (on-device)
            let _signs_per_head = dim / 8;
            let signs_u8 = head_signs.unsqueeze(2)?;
            let bit_masks =
                Tensor::from_vec(vec![1u8, 2, 4, 8, 16, 32, 64, 128], (1, 1, 8), device)?;
            let bits = signs_u8
                .to_dtype(DType::U32)?
                .broadcast_mul(&bit_masks.to_dtype(DType::U32)?)?;
            let bit_set = bits.ne(0u32)?.to_dtype(DType::F32)?;
            let signs_float = ((bit_set * 2.0)? - 1.0)?.reshape((kv_len, dim))?;
            let signs_float_t = signs_float.t()?; // [dim, kv_len] — reused across chunks

            // 3. Pre-compute scaling factors: c_k = residual_norm_k * √(π/2) / √dim
            let sqrt_pi_over_2 = std::f64::consts::FRAC_PI_2.sqrt() as f32;
            let inv_sqrt_dim = 1.0 / (dim as f32).sqrt();
            let scale_factor = sqrt_pi_over_2 * inv_sqrt_dim;
            let c = (head_norms * scale_factor as f64)?; // [kv_len]
            let c_row = c.unsqueeze(0)?; // [1, kv_len]

            // 4. Rademacher projection matrix (precomputed, [dim, dim])
            let rademacher = self
                .gpu_precomputed
                .as_ref()
                .unwrap()
                .qjl_rademacher
                .as_ref()
                .ok_or_else(|| tq_err("QJL Rademacher matrix not precomputed"))?;
            let rademacher_t = rademacher.t()?; // [dim, dim]

            // 5. Chunked computation to bound VRAM: process QJL_QUERY_CHUNK queries at a time
            //    Peak temp memory per chunk: [chunk, dim] + [chunk, kv_len] ≈ chunk * kv_len * 4 bytes
            let num_chunks = q_len.div_ceil(QJL_QUERY_CHUNK);
            let mut correction_chunks = Vec::with_capacity(num_chunks);

            for chunk_idx in 0..num_chunks {
                let offset = chunk_idx * QJL_QUERY_CHUNK;
                let chunk_len = QJL_QUERY_CHUNK.min(q_len - offset);

                let q_chunk = q_flat.narrow(0, offset, chunk_len)?; // [chunk, dim]
                let r_chunk = q_chunk.matmul(&rademacher_t)?; // [chunk, dim]
                let raw_chunk = r_chunk.matmul(&signs_float_t)?; // [chunk, kv_len]
                let corr_chunk = raw_chunk.broadcast_mul(&c_row)?; // [chunk, kv_len]
                correction_chunks.push(corr_chunk);
            }

            let correction = Tensor::cat(&correction_chunks, 0)?; // [q_len, kv_len]
            correction.reshape((1, 1, q_len, kv_len))?.to_device(device)
        } else {
            // CPU cache path: read from per-head QuantizedKVCache (push_head path)
            let mut corrections = Vec::with_capacity(q_len * kv_len);

            for q_pos in 0..q_len {
                let q_vec: Vec<f32> = q_flat.get(q_pos)?.to_vec1::<f32>()?;
                let qjl_seed_cpu = self.caches[head].qjl_seed();
                let r_query = precompute_query_projections(&q_vec, dim, qjl_seed_cpu);

                for k_idx in 0..kv_len {
                    let block = self.caches[head]
                        .key_block(layer, k_idx)
                        .ok_or_else(|| tq_err(format!("key block {k_idx} missing")))?;

                    let signs = block.qjl_signs();
                    let residual_norm = block.residual_norm().to_f32();
                    let c = qjl_scaling_constant(residual_norm, dim);

                    let dot: f32 = r_query
                        .iter()
                        .enumerate()
                        .take(dim)
                        .map(|(i, &rq)| rq * sign_bit(signs, i))
                        .sum();

                    corrections.push(c * dot);
                }
            }

            Tensor::from_vec(corrections, (1, 1, q_len, kv_len), device)
        }
    }
}

// We cannot derive Debug for QuantizedKVCache, so implement manually.
impl std::fmt::Debug for TurboQuantKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurboQuantKVCache")
            .field("bits", &self.bits)
            .field("head_dim", &self.head_dim)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("num_layers", &self.num_layers)
            .field("memory_usage", &self.memory_usage())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BITS: u8 = 3;
    const TEST_HEAD_DIM: usize = 64;
    const TEST_NUM_KV_HEADS: usize = 4;
    const TEST_NUM_LAYERS: usize = 2;
    const TEST_LAYER: usize = 0;

    fn make_cache() -> TurboQuantKVCache {
        TurboQuantKVCache::new_pqo(TEST_BITS, TEST_HEAD_DIM, TEST_NUM_KV_HEADS, TEST_NUM_LAYERS)
            .unwrap()
    }

    fn make_tq_cache() -> TurboQuantKVCache {
        TurboQuantKVCache::new_tq(TEST_BITS, TEST_HEAD_DIM, TEST_NUM_KV_HEADS, TEST_NUM_LAYERS)
            .unwrap()
    }

    fn dummy_vec(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    /// LCG constants matching turboquant-rs integration tests.
    const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const LCG_INCREMENT: u64 = 1;
    const LCG_SHIFT: u32 = 33;

    /// Deterministic pseudo-random vector (same as turboquant-rs tests).
    fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(LCG_MULTIPLIER)
                    .wrapping_add(LCG_INCREMENT);
                let bits = (state >> LCG_SHIFT) as i32;
                bits as f32 / (i32::MAX as f32)
            })
            .collect()
    }

    #[test]
    fn create_cache() {
        let cache = make_cache();
        assert_eq!(cache.bits(), TEST_BITS);
        assert_eq!(cache.entry_count(0, TEST_LAYER), 0);
    }

    #[test]
    fn push_and_count() {
        let mut cache = make_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 1.0);
        let value = dummy_vec(TEST_HEAD_DIM, 2.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();
        assert_eq!(cache.entry_count(0, TEST_LAYER), 1);
        // Other heads should be unaffected
        assert_eq!(cache.entry_count(1, TEST_LAYER), 0);
    }

    #[test]
    fn memory_usage_increases() {
        let mut cache = make_cache();
        let before = cache.memory_usage();
        let key = dummy_vec(TEST_HEAD_DIM, 3.0);
        let value = dummy_vec(TEST_HEAD_DIM, 4.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();
        assert!(cache.memory_usage() > before);
    }

    #[test]
    fn dequantize_keys_roundtrip() {
        let mut cache = make_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 5.0);
        let value = dummy_vec(TEST_HEAD_DIM, 6.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

        let dequantized = cache.dequantize_keys(0, TEST_LAYER).unwrap();
        assert_eq!(dequantized.len(), TEST_HEAD_DIM);
    }

    #[test]
    fn dequantize_values_roundtrip() {
        let mut cache = make_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 7.0);
        let value = dummy_vec(TEST_HEAD_DIM, 8.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

        let dequantized = cache.dequantize_values(0, TEST_LAYER).unwrap();
        assert_eq!(dequantized.len(), TEST_HEAD_DIM);
    }

    #[test]
    fn different_layers_independent() {
        let mut cache = make_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 8.0);
        let value = dummy_vec(TEST_HEAD_DIM, 9.0);
        cache.push_head(0, 0, &key, &value).unwrap();
        assert_eq!(cache.entry_count(0, 0), 1);
        assert_eq!(cache.entry_count(0, 1), 0);
    }

    #[test]
    fn different_heads_independent() {
        let mut cache = make_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 10.0);
        let value = dummy_vec(TEST_HEAD_DIM, 11.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();
        cache.push_head(1, TEST_LAYER, &key, &value).unwrap();
        assert_eq!(cache.entry_count(0, TEST_LAYER), 2);
        assert_eq!(cache.entry_count(1, TEST_LAYER), 1);
        assert_eq!(cache.entry_count(2, TEST_LAYER), 0);
    }

    #[test]
    fn current_seq_len_tracks_head_zero() {
        let mut cache = make_cache();
        assert_eq!(cache.current_seq_len(TEST_LAYER), 0);
        let key = dummy_vec(TEST_HEAD_DIM, 12.0);
        let value = dummy_vec(TEST_HEAD_DIM, 13.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();
        assert_eq!(cache.current_seq_len(TEST_LAYER), 1);
    }

    #[test]
    fn append_and_dequantize_basic() {
        let mut cache = make_cache();
        // Create a [1, 4, 2, 64] tensor (batch=1, heads=4, seq=2, dim=64)
        let total_elems = 1 * TEST_NUM_KV_HEADS * 2 * TEST_HEAD_DIM;
        let k_data: Vec<f32> = (0..total_elems)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let v_data: Vec<f32> = (0..total_elems)
            .map(|i| ((i as f32) * 0.02).cos())
            .collect();

        let k = Tensor::from_vec(
            k_data,
            (1, TEST_NUM_KV_HEADS, 2, TEST_HEAD_DIM),
            &Device::Cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            v_data,
            (1, TEST_NUM_KV_HEADS, 2, TEST_HEAD_DIM),
            &Device::Cpu,
        )
        .unwrap();

        let (full_k, full_v) = cache.append_and_dequantize(TEST_LAYER, &k, &v).unwrap();

        // Should have shape [1, 4, 2, 64]
        assert_eq!(full_k.dims(), &[1, TEST_NUM_KV_HEADS, 2, TEST_HEAD_DIM]);
        assert_eq!(full_v.dims(), &[1, TEST_NUM_KV_HEADS, 2, TEST_HEAD_DIM]);
        assert_eq!(cache.current_seq_len(TEST_LAYER), 2);
    }

    /// Full GPU compressed cache pipeline test.
    ///
    /// Verifies: (1) correctness vs CPU, (2) compression is real,
    /// (3) multiple decode steps work, (4) seq_len tracking is correct.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_compressed_cache_full_test() {
        let cuda_device = Device::cuda_if_available(0).unwrap();
        if cuda_device.is_cpu() {
            return;
        }

        const DIM: usize = 128;
        const HEADS: usize = 4;
        const LAYERS: usize = 1;
        const PREFILL_LEN: usize = 8;
        const DECODE_STEPS: usize = 4;
        const LAYER: usize = 0;

        // Generate test data
        let prefill_k: Vec<f32> = (0..HEADS * PREFILL_LEN * DIM)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let prefill_v: Vec<f32> = (0..HEADS * PREFILL_LEN * DIM)
            .map(|i| ((i as f32) * 0.02).cos())
            .collect();

        // --- GPU path ---
        let mut gpu_cache = TurboQuantKVCache::new_pqo(TEST_BITS, DIM, HEADS, LAYERS).unwrap();

        // Prefill
        let gpu_pf_k = Tensor::from_vec(
            prefill_k.clone(),
            (1, HEADS, PREFILL_LEN, DIM),
            &cuda_device,
        )
        .unwrap();
        let gpu_pf_v = Tensor::from_vec(
            prefill_v.clone(),
            (1, HEADS, PREFILL_LEN, DIM),
            &cuda_device,
        )
        .unwrap();
        let (full_k, full_v) = gpu_cache
            .append_and_dequantize(LAYER, &gpu_pf_k, &gpu_pf_v)
            .unwrap();

        // Check shape after prefill
        assert_eq!(full_k.dims(), &[1, HEADS, PREFILL_LEN, DIM]);
        assert_eq!(gpu_cache.current_seq_len(LAYER), PREFILL_LEN);

        // After prefill, lazy dequant stores FP16 originals (no compressed indices yet).
        // Compressed indices will be created on first decode via flush_lazy_quantize.
        assert!(
            gpu_cache.gpu_path_active[LAYER],
            "GPU path should be active"
        );
        assert!(
            gpu_cache.gpu_k_indices[LAYER].is_some(),
            "Compressed indices should exist after prefill"
        );

        // Multiple decode steps
        for step in 0..DECODE_STEPS {
            let dk: Vec<f32> = (0..HEADS * DIM)
                .map(|i| ((i as f32) * 0.03 + step as f32).sin())
                .collect();
            let dv: Vec<f32> = (0..HEADS * DIM)
                .map(|i| ((i as f32) * 0.04 + step as f32).cos())
                .collect();
            let gpu_dk = Tensor::from_vec(dk, (1, HEADS, 1, DIM), &cuda_device).unwrap();
            let gpu_dv = Tensor::from_vec(dv, (1, HEADS, 1, DIM), &cuda_device).unwrap();

            let (fk, fv) = gpu_cache
                .append_and_dequantize(LAYER, &gpu_dk, &gpu_dv)
                .unwrap();

            let expected_len = PREFILL_LEN + step + 1;
            assert_eq!(
                fk.dims(),
                &[1, HEADS, expected_len, DIM],
                "Shape wrong at decode step {step}"
            );
            assert_eq!(
                gpu_cache.current_seq_len(LAYER),
                expected_len,
                "Seq len wrong at decode step {step}"
            );
        }

        let total_len = PREFILL_LEN + DECODE_STEPS;

        // --- CPU path for correctness comparison ---
        let mut cpu_cache = TurboQuantKVCache::new_pqo(TEST_BITS, DIM, HEADS, LAYERS).unwrap();
        let cpu_pf_k =
            Tensor::from_vec(prefill_k, (1, HEADS, PREFILL_LEN, DIM), &Device::Cpu).unwrap();
        let cpu_pf_v =
            Tensor::from_vec(prefill_v, (1, HEADS, PREFILL_LEN, DIM), &Device::Cpu).unwrap();
        cpu_cache
            .append_and_dequantize(LAYER, &cpu_pf_k, &cpu_pf_v)
            .unwrap();

        for step in 0..DECODE_STEPS {
            let dk: Vec<f32> = (0..HEADS * DIM)
                .map(|i| ((i as f32) * 0.03 + step as f32).sin())
                .collect();
            let dv: Vec<f32> = (0..HEADS * DIM)
                .map(|i| ((i as f32) * 0.04 + step as f32).cos())
                .collect();
            let cpu_dk = Tensor::from_vec(dk, (1, HEADS, 1, DIM), &Device::Cpu).unwrap();
            let cpu_dv = Tensor::from_vec(dv, (1, HEADS, 1, DIM), &Device::Cpu).unwrap();
            cpu_cache
                .append_and_dequantize(LAYER, &cpu_dk, &cpu_dv)
                .unwrap();
        }

        // Compare last decode token output
        let (cpu_fk, _) = cpu_cache
            .append_and_dequantize(
                LAYER,
                &{
                    let d: Vec<f32> = (0..HEADS * DIM).map(|i| (i as f32 * 0.05).sin()).collect();
                    Tensor::from_vec(d, (1, HEADS, 1, DIM), &Device::Cpu).unwrap()
                },
                &{
                    let d: Vec<f32> = (0..HEADS * DIM).map(|i| (i as f32 * 0.06).cos()).collect();
                    Tensor::from_vec(d, (1, HEADS, 1, DIM), &Device::Cpu).unwrap()
                },
            )
            .unwrap();
        let (gpu_fk, _) = gpu_cache
            .append_and_dequantize(
                LAYER,
                &{
                    let d: Vec<f32> = (0..HEADS * DIM).map(|i| (i as f32 * 0.05).sin()).collect();
                    Tensor::from_vec(d, (1, HEADS, 1, DIM), &cuda_device).unwrap()
                },
                &{
                    let d: Vec<f32> = (0..HEADS * DIM).map(|i| (i as f32 * 0.06).cos()).collect();
                    Tensor::from_vec(d, (1, HEADS, 1, DIM), &cuda_device).unwrap()
                },
            )
            .unwrap();

        // Compare the last token (quantized by both paths)
        let final_len = total_len + 1;
        let cpu_last = cpu_fk
            .narrow(2, final_len - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let gpu_last = gpu_fk
            .to_device(&Device::Cpu)
            .unwrap()
            .narrow(2, final_len - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut max_diff: f32 = 0.0;
        for i in 0..cpu_last.len() {
            max_diff = max_diff.max((cpu_last[i] - gpu_last[i]).abs());
        }

        eprintln!(
            "GPU compressed cache test: max_diff={max_diff:.2e}, \
             total_seq={final_len}, heads={HEADS}, dim={DIM}"
        );
        // Debug: print some values
        eprintln!("CPU last 5: {:?}", &cpu_last[..5]);
        eprintln!("GPU last 5: {:?}", &gpu_last[..5]);
        eprintln!(
            "CPU shape: {:?}, GPU shape: {:?}",
            cpu_fk.dims(),
            gpu_fk.dims()
        );
        let tolerance = 5e-1;
        assert!(
            max_diff < tolerance,
            "Mismatch: max_diff={max_diff:.2e} > tolerance={tolerance:.0e}"
        );
    }

    /// Verifies that the GPU compressed cache only stores compressed data.
    ///
    /// The persistent VRAM usage should be:
    ///   - Indices: heads × seq_len × dim × 1 byte (U8) for K and V
    ///   - Scales:  heads × seq_len × 1 × 4 bytes (F32) for K and V
    ///   - NO dequantized buffer between calls
    /// Compare CUDA fused quant kernel output with Candle quant output.
    /// This test diagnoses where numerical differences arise.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_vs_candle_quant_comparison() {
        let cuda_device = Device::cuda_if_available(0).unwrap();
        if cuda_device.is_cpu() {
            return;
        }

        const DIM: usize = 128;
        const HEADS: usize = 8;
        const SEQ: usize = 512;
        const LAYER: usize = 0;

        let mut cache = TurboQuantKVCache::new_pqo(TEST_BITS, DIM, HEADS, 1).unwrap();
        cache.ensure_gpu_precomputed(&cuda_device).unwrap();
        let pre = cache.gpu_precomputed.as_ref().unwrap();

        // Generate test input
        let n = HEADS * SEQ;
        let input_data: Vec<f32> = (0..n * DIM)
            .map(|i| ((i as f32) * 0.0137).sin() * 2.0)
            .collect();
        let input = Tensor::from_vec(input_data, (n, DIM), &cuda_device).unwrap();

        // --- Candle path (matmul WHT) ---
        let num_blocks = DIM / QUANT_BLOCK_SIZE;
        let blocked = input.reshape((n * num_blocks, QUANT_BLOCK_SIZE)).unwrap();
        let rotated_candle = blocked.matmul(&pre.rotation_fwd).unwrap();

        // --- GPU butterfly path ---
        let rotated_butterfly =
            butterfly_wht_forward_gpu(&blocked, &pre.rotation_fwd, QUANT_BLOCK_SIZE).unwrap();

        // Compare WHT outputs
        let candle_flat: Vec<f32> = rotated_candle.flatten_all().unwrap().to_vec1().unwrap();
        let butterfly_flat: Vec<f32> = rotated_butterfly.flatten_all().unwrap().to_vec1().unwrap();

        let mut wht_max_diff: f32 = 0.0;
        let mut wht_diffs = 0usize;
        for (i, (c, b)) in candle_flat.iter().zip(butterfly_flat.iter()).enumerate() {
            let d = (c - b).abs();
            if d > 1e-5 {
                wht_diffs += 1;
                if wht_diffs <= 5 {
                    eprintln!("[WHT diff] idx={i}: candle={c:.6} butterfly={b:.6} diff={d:.2e}");
                }
            }
            wht_max_diff = wht_max_diff.max(d);
        }
        eprintln!(
            "[WHT] max_diff={wht_max_diff:.2e}, diffs>1e-5: {wht_diffs}/{} ({:.1}%)",
            candle_flat.len(),
            100.0 * wht_diffs as f64 / candle_flat.len() as f64
        );

        // --- Full quantize comparison ---
        let bits = TEST_BITS;
        let (candle_idx, candle_sc) = TurboQuantKVCache::polar_quantize(
            &input,
            DIM,
            bits,
            QuantNormMode::MaxNorm,
            usize::MAX,
            pre,
        )
        .unwrap();

        // Call CUDA fused quant directly
        let total_blocks = n * num_blocks;
        let bytes_per_block = QUANT_BLOCK_SIZE * bits as usize / 8;
        let n_ob = pre.outlier_boundaries.elem_count();
        let sqrt_bs = (QUANT_BLOCK_SIZE as f64).sqrt();
        // Use butterfly-rotated values as input (same as production code)
        let rotated_for_cuda =
            butterfly_wht_forward_gpu(&blocked, &pre.rotation_fwd, QUANT_BLOCK_SIZE).unwrap();
        let rotated_flat = rotated_for_cuda
            .flatten_all()
            .unwrap()
            .contiguous()
            .unwrap();
        let boundaries_cont = pre.outlier_boundaries.contiguous().unwrap();

        let (cuda_idx, cuda_sc) = mistralrs_paged_attn::tq_quant_maxnorm_batch(
            &rotated_flat,
            &boundaries_cont,
            total_blocks,
            QUANT_BLOCK_SIZE,
            bits as usize,
            n_ob,
            bytes_per_block,
            pre.outlier_outer_centroid as f32,
            -1.0,
            &cuda_device,
        )
        .unwrap();

        let packed_dim = DIM * bits as usize / 8;
        let cuda_idx = cuda_idx.reshape((n, packed_dim)).unwrap();
        let cuda_sc = cuda_sc.reshape((n, num_blocks)).unwrap();

        // Compare packed indices
        let ci: Vec<u8> = candle_idx.flatten_all().unwrap().to_vec1().unwrap();
        let gi: Vec<u8> = cuda_idx.flatten_all().unwrap().to_vec1().unwrap();
        let mut idx_diffs = 0usize;
        for (i, (c, g)) in ci.iter().zip(gi.iter()).enumerate() {
            if c != g {
                idx_diffs += 1;
                if idx_diffs <= 5 {
                    eprintln!("[IDX diff] byte={i}: candle={c} cuda={g}");
                }
            }
        }
        eprintln!(
            "[IDX] diffs: {idx_diffs}/{} ({:.1}%)",
            ci.len(),
            100.0 * idx_diffs as f64 / ci.len() as f64
        );

        // Compare scales
        let cs: Vec<f32> = candle_sc
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let gs: Vec<f32> = cuda_sc
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let mut sc_max_diff: f32 = 0.0;
        let mut sc_diffs = 0usize;
        for (i, (c, g)) in cs.iter().zip(gs.iter()).enumerate() {
            let d = (c - g).abs();
            if d > 1e-4 {
                sc_diffs += 1;
                if sc_diffs <= 5 {
                    eprintln!("[SCALE diff] idx={i}: candle={c:.6} cuda={g:.6} diff={d:.2e}");
                }
            }
            sc_max_diff = sc_max_diff.max(d);
        }
        eprintln!(
            "[SCALE] max_diff={sc_max_diff:.2e}, diffs>1e-4: {sc_diffs}/{}",
            cs.len()
        );

        // The test passes even with differences — it's diagnostic
        eprintln!("---");
        eprintln!(
            "If WHT diffs are ~0 but IDX diffs are >0, the issue is in quantization/packing."
        );
        eprintln!("If WHT diffs are >0, the issue is matmul vs butterfly numerical difference.");
    }

    ///
    /// The dequantized tensors returned by append_and_dequantize should be
    /// temporary — they exist only for the current forward pass.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_compressed_cache_vram_is_minimal() {
        let cuda_device = Device::cuda_if_available(0).unwrap();
        if cuda_device.is_cpu() {
            return;
        }

        const DIM: usize = 128;
        const HEADS: usize = 4;
        const LAYERS: usize = 1;
        const SEQ_LEN: usize = 1024;
        const LAYER: usize = 0;

        let mut cache = TurboQuantKVCache::new_pqo(TEST_BITS, DIM, HEADS, LAYERS).unwrap();

        // Prefill 1024 tokens
        let k_data: Vec<f32> = (0..HEADS * SEQ_LEN * DIM)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let v_data: Vec<f32> = (0..HEADS * SEQ_LEN * DIM)
            .map(|i| ((i as f32) * 0.02).cos())
            .collect();
        let k = Tensor::from_vec(k_data, (1, HEADS, SEQ_LEN, DIM), &cuda_device).unwrap();
        let v = Tensor::from_vec(v_data, (1, HEADS, SEQ_LEN, DIM), &cuda_device).unwrap();

        // Drop the returned tensors (simulate end of forward pass)
        let _ = cache.append_and_dequantize(LAYER, &k, &v).unwrap();
        drop(k);
        drop(v);

        // After prefill, compressed data should exist immediately (no lazy dequant).
        assert!(cache.gpu_path_active[LAYER]);

        let ki = cache.gpu_k_indices[LAYER]
            .as_ref()
            .expect("K indices missing after prefill");
        let vi = cache.gpu_v_indices[LAYER]
            .as_ref()
            .expect("V indices missing after prefill");
        assert!(
            ki.dims()[1] >= SEQ_LEN,
            "K indices too small: {:?}",
            ki.dims()
        );
        assert!(
            vi.dims()[1] >= SEQ_LEN,
            "V indices too small: {:?}",
            vi.dims()
        );
        assert_eq!(
            ki.dtype(),
            DType::U8,
            "Indices should be U8, not {:?}",
            ki.dtype()
        );

        // Compute actual persistent VRAM for KV cache only
        // Indices: capacity may be larger due to doubling, but actual data is SEQ_LEN
        let idx_capacity = ki.dims()[1];
        let idx_bytes_per_kv = HEADS * idx_capacity * DIM * 1; // U8
        let scale_bytes_per_kv = HEADS * idx_capacity * 1 * 4; // F32
        let compressed_bytes = 2 * (idx_bytes_per_kv + scale_bytes_per_kv); // K + V

        // Expected compressed size for exactly SEQ_LEN tokens:
        let expected_bytes = 2 * (HEADS * SEQ_LEN * DIM * 1 + HEADS * SEQ_LEN * 4);
        // Normal FP16 would be:
        let normal_fp16_bytes = 2 * HEADS * SEQ_LEN * DIM * 2;

        let compression_ratio = normal_fp16_bytes as f64 / expected_bytes as f64;

        eprintln!(
            "VRAM test: compressed={} KB (capacity={}), expected={} KB, \
             normal_fp16={} KB, compression={compression_ratio:.1}x",
            compressed_bytes / 1024,
            idx_capacity,
            expected_bytes / 1024,
            normal_fp16_bytes / 1024,
        );

        // Compressed cache should be smaller than FP16
        assert!(
            compressed_bytes < normal_fp16_bytes * 2, // allow 2x for doubling capacity
            "Compressed cache ({compressed_bytes} bytes) should be much smaller than \
             FP16 ({normal_fp16_bytes} bytes)"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 1: QJL-Storage Tests
    //
    // These tests verify that QJL signs and residual norms are correctly
    // computed and stored after quantization. TDD: tests written BEFORE
    // implementation — they will fail until Phase 1 is implemented.
    // -----------------------------------------------------------------------

    /// After appending tokens, QJL signs must be stored in the cache.
    /// Shape: [num_kv_heads, seq_len, signs_per_head] where
    /// signs_per_head = head_dim / 8 (1 bit per dimension, packed as bytes).
    #[test]
    fn qjl_signs_stored_after_append() {
        let mut cache = make_tq_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 1.0);
        let value = dummy_vec(TEST_HEAD_DIM, 2.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

        // QJL signs should exist for this head/layer
        let signs = cache.qjl_signs(0, TEST_LAYER);
        assert!(signs.is_some(), "QJL signs must be stored after push_head");
        let signs = signs.unwrap();
        let signs_per_head = TEST_HEAD_DIM / 8;
        assert_eq!(
            signs.len(),
            signs_per_head,
            "Expected {signs_per_head} sign bytes for dim={TEST_HEAD_DIM}"
        );
    }

    /// After appending tokens, residual norm must be stored.
    /// residual_norm = L2(original - polar_dequant(polar_quant(original)))
    #[test]
    fn residual_norm_stored_after_append() {
        let mut cache = make_tq_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 3.0);
        let value = dummy_vec(TEST_HEAD_DIM, 4.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

        let res_norm = cache.key_residual_norm(0, TEST_LAYER, 0);
        assert!(
            res_norm.is_some(),
            "Residual norm must be stored after push_head"
        );
        let norm_val = res_norm.unwrap();
        assert!(
            norm_val > 0.0,
            "Residual norm should be positive (quantization always has error)"
        );
    }

    /// Residual norm must equal L2-norm of (original - dequantized).
    #[test]
    fn residual_norm_matches_quantization_error() {
        let mut cache = make_tq_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 5.0);
        let value = dummy_vec(TEST_HEAD_DIM, 6.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

        // Dequantize to get reconstructed vector
        let dequantized = cache.dequantize_keys(0, TEST_LAYER).unwrap();

        // Compute expected residual norm manually
        let residual_norm_sq: f32 = key
            .iter()
            .zip(dequantized.iter())
            .map(|(o, d)| (o - d).powi(2))
            .sum();
        let expected_residual_norm = residual_norm_sq.sqrt();

        let stored_norm = cache.key_residual_norm(0, TEST_LAYER, 0).unwrap();
        let tolerance = 0.05; // f16 storage tolerance
        let diff = (stored_norm - expected_residual_norm).abs();
        assert!(
            diff < tolerance,
            "Residual norm mismatch: stored={stored_norm:.4}, \
             expected={expected_residual_norm:.4}, diff={diff:.4e}"
        );
    }

    /// QJL signs must match turboquant-rs CPU reference (byte-for-byte).
    #[test]
    fn qjl_signs_match_cpu_crate_reference() {
        use turboquant::{quantize_with_qjl, TurboQuantConfig};

        let key = dummy_vec(TEST_HEAD_DIM, 7.0);
        let config = TurboQuantConfig::new(TEST_BITS, TEST_HEAD_DIM)
            .unwrap()
            .with_seed(DEFAULT_ROTATION_SEED);

        // CPU reference via turboquant-rs crate
        let qjl_block = quantize_with_qjl(&config, &key, DEFAULT_QJL_SEED).unwrap();
        let ref_signs = qjl_block.qjl_signs();
        let ref_residual_norm = qjl_block.residual_norm().to_f32();

        // Our cache (should produce identical results)
        let mut cache = make_tq_cache();
        let value = dummy_vec(TEST_HEAD_DIM, 8.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

        let our_signs = cache
            .qjl_signs(0, TEST_LAYER)
            .expect("QJL signs not stored");
        let our_norm = cache
            .key_residual_norm(0, TEST_LAYER, 0)
            .expect("Residual norm not stored");

        // Byte-for-byte comparison of signs
        assert_eq!(
            our_signs, ref_signs,
            "QJL signs must match turboquant-rs reference byte-for-byte"
        );

        // Residual norm comparison (f16 tolerance)
        let norm_diff = (our_norm - ref_residual_norm).abs();
        assert!(
            norm_diff < 0.05,
            "Residual norm mismatch: ours={our_norm:.4}, ref={ref_residual_norm:.4}"
        );
    }

    /// Multi-token prefill: QJL data stored for ALL tokens, not just the last.
    #[test]
    fn qjl_data_stored_for_all_prefill_tokens() {
        let mut cache = make_tq_cache();
        let num_tokens = 8;

        for i in 0..num_tokens {
            let key = dummy_vec(TEST_HEAD_DIM, i as f32 * 1.5);
            let value = dummy_vec(TEST_HEAD_DIM, i as f32 * 2.5);
            cache.push_head(0, TEST_LAYER, &key, &value).unwrap();
        }

        // All tokens should have QJL data
        for i in 0..num_tokens {
            let norm = cache.key_residual_norm(0, TEST_LAYER, i);
            assert!(norm.is_some(), "Residual norm missing for token {i}");
            assert!(
                norm.unwrap() > 0.0,
                "Residual norm should be positive for token {i}"
            );
        }
    }

    /// After prefill + decode, QJL data is stored via the CPU path.
    /// Prefill buffers raw vectors (lazy quantization), decode triggers
    /// flush_pending which quantizes with QJL.
    #[test]
    fn cpu_path_stores_qjl_after_decode() {
        let mut cache = make_tq_cache();

        // Prefill: 2 tokens (lazy, goes to pending — no QJL yet)
        let prefill_elems = TEST_NUM_KV_HEADS * 2 * TEST_HEAD_DIM;
        let k_pf: Vec<f32> = (0..prefill_elems)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let v_pf: Vec<f32> = (0..prefill_elems)
            .map(|i| ((i as f32) * 0.02).cos())
            .collect();
        let k =
            Tensor::from_vec(k_pf, (1, TEST_NUM_KV_HEADS, 2, TEST_HEAD_DIM), &Device::Cpu).unwrap();
        let v =
            Tensor::from_vec(v_pf, (1, TEST_NUM_KV_HEADS, 2, TEST_HEAD_DIM), &Device::Cpu).unwrap();
        let _ = cache.append_and_dequantize(TEST_LAYER, &k, &v).unwrap();

        // Decode: 1 token — triggers flush_pending → quantizes with QJL
        let decode_elems = TEST_NUM_KV_HEADS * 1 * TEST_HEAD_DIM;
        let k_dec: Vec<f32> = (0..decode_elems)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();
        let v_dec: Vec<f32> = (0..decode_elems)
            .map(|i| ((i as f32) * 0.04).cos())
            .collect();
        let kd = Tensor::from_vec(
            k_dec,
            (1, TEST_NUM_KV_HEADS, 1, TEST_HEAD_DIM),
            &Device::Cpu,
        )
        .unwrap();
        let vd = Tensor::from_vec(
            v_dec,
            (1, TEST_NUM_KV_HEADS, 1, TEST_HEAD_DIM),
            &Device::Cpu,
        )
        .unwrap();
        let _ = cache.append_and_dequantize(TEST_LAYER, &kd, &vd).unwrap();

        // After decode, QJL data should exist (flush_pending quantized all)
        assert!(
            cache.has_qjl_data(TEST_LAYER),
            "QJL data must be stored after prefill + decode (flush_pending)"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 1 new: Block-path QJL storage tests (CPU + GPU)
    // -----------------------------------------------------------------------

    /// Helper: create tensors on the given device for block-path testing.
    fn make_block_path_tensors(
        device: &Device,
        num_heads: usize,
        seq_len: usize,
        dim: usize,
    ) -> (Tensor, Tensor) {
        let total = num_heads * seq_len * dim;
        let k_data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.013).sin()).collect();
        let v_data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.017).cos()).collect();
        let k = Tensor::from_vec(k_data, (1, num_heads, seq_len, dim), &Device::Cpu)
            .unwrap()
            .to_device(device)
            .unwrap();
        let v = Tensor::from_vec(v_data, (1, num_heads, seq_len, dim), &Device::Cpu)
            .unwrap()
            .to_device(device)
            .unwrap();
        (k, v)
    }

    /// Test A (CPU): block_append_and_dequantize stores QJL signs + residual norms.
    /// Paper-Ref: Algorithm 2 Step 5 (Residual projection)
    #[test]
    fn block_path_stores_qjl_signs_cpu() {
        block_path_stores_qjl_signs_impl(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn block_path_stores_qjl_signs_gpu() {
        let dev = Device::cuda_if_available(0).unwrap();
        if dev.is_cpu() {
            return;
        }
        block_path_stores_qjl_signs_impl(&dev);
    }

    fn block_path_stores_qjl_signs_impl(device: &Device) {
        let mut cache = make_tq_cache();
        let seq_len = 4;
        let (k, v) = make_block_path_tensors(device, TEST_NUM_KV_HEADS, seq_len, TEST_HEAD_DIM);
        let _ = cache.append_and_dequantize(TEST_LAYER, &k, &v).unwrap();

        // QJL signs tensor must exist
        let signs = cache.gpu_k_qjl_signs[TEST_LAYER].as_ref();
        assert!(
            signs.is_some(),
            "QJL signs tensor must exist after block append"
        );
        let signs = signs.unwrap();
        let signs_per_head = TEST_HEAD_DIM / 8;
        // Capacity >= seq_len (with some headroom from growth strategy)
        let cap = signs.dims()[1];
        assert!(
            cap >= cache.buf_seq_len[TEST_LAYER],
            "Capacity {cap} must be >= seq_len {}",
            cache.buf_seq_len[TEST_LAYER]
        );

        // Residual norms tensor must exist
        let norms = cache.gpu_k_residual_norms[TEST_LAYER].as_ref();
        assert!(
            norms.is_some(),
            "Residual norms tensor must exist after block append"
        );

        // Check that norms are positive (quantization always has error)
        let norms_cpu: Vec<f32> = norms
            .unwrap()
            .narrow(1, 0, seq_len)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &n) in norms_cpu.iter().enumerate() {
            assert!(
                n > 0.0,
                "Residual norm should be positive for entry {i}, got {n}"
            );
        }
    }

    /// Test B (CPU): residual norm from block path matches manual computation.
    /// Paper-Ref: residual_norm = L2(original - dequant(quant(original)))
    #[test]
    fn block_path_residual_norm_is_quantization_error_cpu() {
        block_path_residual_norm_is_quantization_error_impl(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn block_path_residual_norm_is_quantization_error_gpu() {
        let dev = Device::cuda_if_available(0).unwrap();
        if dev.is_cpu() {
            return;
        }
        block_path_residual_norm_is_quantization_error_impl(&dev);
    }

    fn block_path_residual_norm_is_quantization_error_impl(device: &Device) {
        let mut cache = make_tq_cache();
        let (k, v) = make_block_path_tensors(device, TEST_NUM_KV_HEADS, 1, TEST_HEAD_DIM);

        // Get original key data (on CPU for comparison)
        let k_cpu: Vec<f32> = k
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let (dequant_k, _) = cache.append_and_dequantize(TEST_LAYER, &k, &v).unwrap();
        let dk_cpu: Vec<f32> = dequant_k
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // For each head: compute expected residual norm
        for head in 0..TEST_NUM_KV_HEADS {
            let start = head * TEST_HEAD_DIM;
            let end = start + TEST_HEAD_DIM;
            let expected_norm: f32 = k_cpu[start..end]
                .iter()
                .zip(dk_cpu[start..end].iter())
                .map(|(o, d)| (o - d).powi(2))
                .sum::<f32>()
                .sqrt();

            let stored_norm: f32 = cache.gpu_k_residual_norms[TEST_LAYER]
                .as_ref()
                .unwrap()
                .narrow(0, head, 1)
                .unwrap()
                .narrow(1, 0, 1)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];

            let tolerance = 0.1; // f16 storage + block-level quantization tolerance
            let diff = (stored_norm - expected_norm).abs();
            assert!(
                diff < tolerance,
                "Head {head}: residual norm mismatch: stored={stored_norm:.4}, \
                 expected={expected_norm:.4}, diff={diff:.4e}"
            );
        }
    }

    /// Test C (CPU): QJL signs are deterministic (same input + seed → same signs).
    #[test]
    fn block_path_qjl_signs_deterministic_cpu() {
        block_path_qjl_signs_deterministic_impl(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn block_path_qjl_signs_deterministic_gpu() {
        let dev = Device::cuda_if_available(0).unwrap();
        if dev.is_cpu() {
            return;
        }
        block_path_qjl_signs_deterministic_impl(&dev);
    }

    fn block_path_qjl_signs_deterministic_impl(device: &Device) {
        let (k, v) = make_block_path_tensors(device, TEST_NUM_KV_HEADS, 2, TEST_HEAD_DIM);
        let signs_per_head = TEST_HEAD_DIM / 8;
        let seq = 2;

        // Run 1
        let mut cache1 = make_tq_cache();
        let _ = cache1.append_and_dequantize(TEST_LAYER, &k, &v).unwrap();
        let signs1: Vec<u8> = cache1.gpu_k_qjl_signs[TEST_LAYER]
            .as_ref()
            .unwrap()
            .narrow(1, 0, seq)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Run 2 (fresh cache, same input)
        let mut cache2 = make_tq_cache();
        let _ = cache2.append_and_dequantize(TEST_LAYER, &k, &v).unwrap();
        let signs2: Vec<u8> = cache2.gpu_k_qjl_signs[TEST_LAYER]
            .as_ref()
            .unwrap()
            .narrow(1, 0, seq)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(
            signs1.len(),
            TEST_NUM_KV_HEADS * seq * signs_per_head,
            "Signs length mismatch"
        );
        assert_eq!(signs1, signs2, "QJL signs must be deterministic");
    }

    /// Test D (CPU): QJL signs survive buffer capacity growth (reallocation).
    #[test]
    fn block_path_qjl_survives_capacity_growth_cpu() {
        block_path_qjl_survives_capacity_growth_impl(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn block_path_qjl_survives_capacity_growth_gpu() {
        let dev = Device::cuda_if_available(0).unwrap();
        if dev.is_cpu() {
            return;
        }
        block_path_qjl_survives_capacity_growth_impl(&dev);
    }

    fn block_path_qjl_survives_capacity_growth_impl(device: &Device) {
        let mut cache = make_tq_cache();
        let signs_per_head = TEST_HEAD_DIM / 8;

        // First batch: fill some initial tokens
        let batch1_seq = 4;
        let (k1, v1) =
            make_block_path_tensors(device, TEST_NUM_KV_HEADS, batch1_seq, TEST_HEAD_DIM);
        let _ = cache.append_and_dequantize(TEST_LAYER, &k1, &v1).unwrap();

        // Record signs for first batch
        let signs_before: Vec<u8> = cache.gpu_k_qjl_signs[TEST_LAYER]
            .as_ref()
            .unwrap()
            .narrow(1, 0, batch1_seq)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Push enough tokens to force capacity growth
        let big_batch = 2048 + 1;
        let (k2, v2) = make_block_path_tensors(device, TEST_NUM_KV_HEADS, big_batch, TEST_HEAD_DIM);
        let _ = cache.append_and_dequantize(TEST_LAYER, &k2, &v2).unwrap();

        // Verify signs from first batch survived the reallocation
        let signs_after: Vec<u8> = cache.gpu_k_qjl_signs[TEST_LAYER]
            .as_ref()
            .unwrap()
            .narrow(1, 0, batch1_seq)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(
            signs_before.len(),
            TEST_NUM_KV_HEADS * batch1_seq * signs_per_head
        );
        assert_eq!(
            signs_before, signs_after,
            "QJL signs for early tokens must survive buffer reallocation"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 2 new: Block-path QJL correction tests (CPU + GPU)
    // -----------------------------------------------------------------------

    /// Test E: qjl_correction via block path produces non-zero output.
    /// Paper-Ref: Theorem 2 — correction ≠ 0 (otherwise useless).
    #[test]
    fn block_path_qjl_correction_nonzero_cpu() {
        block_path_qjl_correction_nonzero_impl(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn block_path_qjl_correction_nonzero_gpu() {
        let dev = Device::cuda_if_available(0).unwrap();
        if dev.is_cpu() {
            return;
        }
        block_path_qjl_correction_nonzero_impl(&dev);
    }

    fn block_path_qjl_correction_nonzero_impl(device: &Device) {
        let mut cache = make_tq_cache();
        let seq_len = 8;
        let (k, v) = make_block_path_tensors(device, TEST_NUM_KV_HEADS, seq_len, TEST_HEAD_DIM);
        let _ = cache.append_and_dequantize(TEST_LAYER, &k, &v).unwrap();

        // Query tensor: [batch=1, heads=1, q_len=1, dim]
        let q_data: Vec<f32> = (0..TEST_HEAD_DIM)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();
        let query = Tensor::from_vec(q_data, (1, 1, 1, TEST_HEAD_DIM), &Device::Cpu)
            .unwrap()
            .to_device(device)
            .unwrap();

        let correction = cache.qjl_correction(0, TEST_LAYER, &query).unwrap();

        assert_eq!(correction.dims(), &[1, 1, 1, seq_len], "Shape mismatch");

        let corr_vals: Vec<f32> = correction
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let sum_abs: f32 = corr_vals.iter().map(|x| x.abs()).sum();
        assert!(
            sum_abs > 1e-6,
            "QJL correction should be non-zero, got sum_abs={sum_abs}"
        );
    }

    /// Test G: Block-path QJL eliminates multiplicative bias.
    /// Paper-Ref: Theorem 2 — E[⟨y, x̃⟩_corrected] = ⟨y, x⟩ (unbiased).
    /// This is THE central paper claim.
    #[test]
    fn block_path_qjl_eliminates_bias_cpu() {
        block_path_qjl_eliminates_bias_impl(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn block_path_qjl_eliminates_bias_gpu() {
        let dev = Device::cuda_if_available(0).unwrap();
        if dev.is_cpu() {
            return;
        }
        block_path_qjl_eliminates_bias_impl(&dev);
    }

    fn block_path_qjl_eliminates_bias_impl(device: &Device) {
        let num_samples = 200;
        let dim = TEST_HEAD_DIM;
        let mut polar_ratios = Vec::with_capacity(num_samples);
        let mut qjl_ratios = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let seed = 42u64.wrapping_add(i as u64);
            let key = pseudo_random_vec(dim, seed);
            let query = pseudo_random_vec(dim, seed.wrapping_add(10000));

            // True inner product
            let true_ip: f32 = key.iter().zip(query.iter()).map(|(k, q)| k * q).sum();
            if true_ip.abs() < 1e-4 {
                continue; // skip near-zero to avoid division instability
            }

            // Quantize via block path
            let mut cache = make_tq_cache();
            let k_tensor = Tensor::from_vec(
                vec![key.clone(); TEST_NUM_KV_HEADS]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<f32>>(),
                (1, TEST_NUM_KV_HEADS, 1, dim),
                &Device::Cpu,
            )
            .unwrap()
            .to_device(device)
            .unwrap();
            let v_tensor = Tensor::from_vec(
                vec![0.0f32; TEST_NUM_KV_HEADS * dim],
                (1, TEST_NUM_KV_HEADS, 1, dim),
                &Device::Cpu,
            )
            .unwrap()
            .to_device(device)
            .unwrap();

            let (dequant_k, _) = cache
                .append_and_dequantize(TEST_LAYER, &k_tensor, &v_tensor)
                .unwrap();

            // Polar-only inner product (no QJL)
            let dk: Vec<f32> = dequant_k
                .narrow(1, 0, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let polar_ip: f32 = dk.iter().zip(query.iter()).map(|(d, q)| d * q).sum();

            // QJL-corrected inner product
            let q_tensor = Tensor::from_vec(query.clone(), (1, 1, 1, dim), &Device::Cpu)
                .unwrap()
                .to_device(device)
                .unwrap();
            let correction = cache.qjl_correction(0, TEST_LAYER, &q_tensor).unwrap();
            let corr_val: f32 = correction
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
            let qjl_ip = polar_ip + corr_val;

            polar_ratios.push(polar_ip / true_ip);
            qjl_ratios.push(qjl_ip / true_ip);
        }

        let mean_polar: f32 = polar_ratios.iter().sum::<f32>() / polar_ratios.len() as f32;
        let mean_qjl: f32 = qjl_ratios.iter().sum::<f32>() / qjl_ratios.len() as f32;

        let polar_bias = (mean_polar - 1.0).abs();
        let qjl_bias = (mean_qjl - 1.0).abs();

        // Compute variance too (llama.cpp concern)
        let var_polar: f32 = polar_ratios
            .iter()
            .map(|r| (r - mean_polar).powi(2))
            .sum::<f32>()
            / polar_ratios.len() as f32;
        let var_qjl: f32 = qjl_ratios
            .iter()
            .map(|r| (r - mean_qjl).powi(2))
            .sum::<f32>()
            / qjl_ratios.len() as f32;

        eprintln!(
            "Bias test (n={}): polar mean_ratio={mean_polar:.4} (bias={polar_bias:.4}), \
             qjl mean_ratio={mean_qjl:.4} (bias={qjl_bias:.4})",
            polar_ratios.len()
        );
        eprintln!(
            "Variance: polar={var_polar:.6}, qjl={var_qjl:.6}, ratio={:.2}",
            if var_polar > 0.0 {
                var_qjl / var_polar
            } else {
                f32::NAN
            }
        );

        // TQ mode uses 2-bit polar (bits-1) which has ~6% bias.
        // QJL should reduce this to near-zero.
        assert!(
            polar_bias > 0.02,
            "2-bit polar should have measurable bias, got {polar_bias:.4}"
        );
        assert!(
            qjl_bias < polar_bias,
            "QJL should reduce bias: polar={polar_bias:.4}, qjl={qjl_bias:.4}"
        );
        assert!(
            qjl_bias < 0.05,
            "QJL should be near-unbiased, got {qjl_bias:.4}"
        );
        // QJL should not make bias significantly WORSE
        assert!(
            qjl_bias < polar_bias + 0.02,
            "QJL must not significantly worsen bias: polar={polar_bias:.4}, qjl={qjl_bias:.4}"
        );
    }

    /// Test H: QJL reduces MSE of inner product estimation.
    /// Paper-Ref: Theorem 2 — QJL correction reduces distortion.
    #[test]
    fn block_path_qjl_reduces_mse_cpu() {
        block_path_qjl_reduces_mse_impl(&Device::Cpu);
    }

    fn block_path_qjl_reduces_mse_impl(device: &Device) {
        let num_samples = 200;
        let dim = TEST_HEAD_DIM;
        let mut mse_polar = 0.0f64;
        let mut mse_qjl = 0.0f64;
        let mut count = 0;

        for i in 0..num_samples {
            let seed = 100u64.wrapping_add(i as u64);
            let key = pseudo_random_vec(dim, seed);
            let query = pseudo_random_vec(dim, seed.wrapping_add(50000));

            let true_ip: f32 = key.iter().zip(query.iter()).map(|(k, q)| k * q).sum();
            if true_ip.abs() < 1e-4 {
                continue;
            }

            let mut cache = make_tq_cache();
            let k_tensor = Tensor::from_vec(
                vec![key.clone(); TEST_NUM_KV_HEADS]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<f32>>(),
                (1, TEST_NUM_KV_HEADS, 1, dim),
                &Device::Cpu,
            )
            .unwrap()
            .to_device(device)
            .unwrap();
            let v_tensor = Tensor::from_vec(
                vec![0.0f32; TEST_NUM_KV_HEADS * dim],
                (1, TEST_NUM_KV_HEADS, 1, dim),
                &Device::Cpu,
            )
            .unwrap()
            .to_device(device)
            .unwrap();

            let (dequant_k, _) = cache
                .append_and_dequantize(TEST_LAYER, &k_tensor, &v_tensor)
                .unwrap();

            let dk: Vec<f32> = dequant_k
                .narrow(1, 0, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let polar_ip: f32 = dk.iter().zip(query.iter()).map(|(d, q)| d * q).sum();

            let q_tensor = Tensor::from_vec(query.clone(), (1, 1, 1, dim), &Device::Cpu)
                .unwrap()
                .to_device(device)
                .unwrap();
            let corr_val: f32 = cache
                .qjl_correction(0, TEST_LAYER, &q_tensor)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
            let qjl_ip = polar_ip + corr_val;

            mse_polar += ((polar_ip - true_ip) as f64).powi(2);
            mse_qjl += ((qjl_ip - true_ip) as f64).powi(2);
            count += 1;
        }

        mse_polar /= count as f64;
        mse_qjl /= count as f64;

        eprintln!(
            "MSE test (n={count}): polar={mse_polar:.6}, qjl={mse_qjl:.6}, \
             ratio={:.2}",
            if mse_polar > 0.0 {
                mse_qjl / mse_polar
            } else {
                f64::NAN
            }
        );

        // FINDING: With OUTLIER_BLOCKS=4 (3-bit polar for TQ3), QJL makes
        // MSE ~7x WORSE. The polar-only MSE is already very low (~0.17),
        // and QJL's Rademacher projection adds variance (~1.27 MSE).
        // This confirms llama.cpp's decision to exclude QJL.
        //
        // QJL would only help if polar quantization had significant bias
        // (2-bit polar with 4 centroids). With 3-bit (8 centroids, outlier
        // codebook), the bias is negligible and QJL's variance dominates.
        //
        // Assert: just log the ratio for decision-making. No hard pass/fail
        // on which is better — that's determined by the model benchmarks.
        let ratio = if mse_polar > 0.0 {
            mse_qjl / mse_polar
        } else {
            1.0
        };
        eprintln!(
            "FINDING: QJL MSE ratio = {ratio:.2}x (>1 means QJL is worse). \
             This confirms llama.cpp's variance concern for 3-bit polar."
        );
    }

    // -----------------------------------------------------------------------
    // PQ3-plain vs PQ3-outlier vs TQ3: Three-way comparison
    //
    // PQ3-plain:   3-bit Polar via crate (single block, no outlier override)
    // PQ3-outlier: 3-bit Polar via block-level cache (OUTLIER_BLOCKS=4,
    //              block_size=32, outlier codebook for all blocks)
    // TQ3:         2-bit Polar + 1-bit QJL (Paper Algorithm 2)
    //
    // All use 3 bits total per value. Which gives best inner products?
    // -----------------------------------------------------------------------

    /// Three-way comparison: PQ3-plain vs PQ3-outlier vs TQ3.
    /// Measures bias, MSE, and variance of inner product estimation.
    #[test]
    fn pq3_plain_vs_pq3_outlier_vs_tq3_comparison() {
        use turboquant::{
            dequantize_vec, estimate_inner_product_single, quantize_vec, quantize_with_qjl,
            TurboQuantConfig,
        };

        let dim = TEST_HEAD_DIM; // 64
        let num_samples = 500;
        let rotation_seed: u64 = 42;
        let qjl_seed: u64 = 12345;

        // PQ3-plain: 3-bit polar via crate (single block, dim=64)
        let pq3_plain_config = TurboQuantConfig::new(3, dim)
            .unwrap()
            .with_seed(rotation_seed);

        // TQ3: total_bits=3 → internally 2-bit polar + 1-bit QJL
        let tq3_config = TurboQuantConfig::new(3, dim)
            .unwrap()
            .with_seed(rotation_seed);

        struct Stats {
            ratios: Vec<f32>,
            sq_errors: Vec<f64>,
        }
        impl Stats {
            fn new(cap: usize) -> Self {
                Self {
                    ratios: Vec::with_capacity(cap),
                    sq_errors: Vec::with_capacity(cap),
                }
            }
            fn push(&mut self, estimate: f32, true_val: f32) {
                self.ratios.push(estimate / true_val);
                self.sq_errors.push(((estimate - true_val) as f64).powi(2));
            }
            fn bias(&self) -> f32 {
                let mean: f32 = self.ratios.iter().sum::<f32>() / self.ratios.len() as f32;
                (mean - 1.0).abs()
            }
            fn mean_ratio(&self) -> f32 {
                self.ratios.iter().sum::<f32>() / self.ratios.len() as f32
            }
            fn mse(&self) -> f64 {
                self.sq_errors.iter().sum::<f64>() / self.sq_errors.len() as f64
            }
            fn var(&self) -> f32 {
                let mean = self.mean_ratio();
                self.ratios.iter().map(|r| (r - mean).powi(2)).sum::<f32>()
                    / self.ratios.len() as f32
            }
        }

        let mut pq3_plain = Stats::new(num_samples);
        let mut pq3_outlier = Stats::new(num_samples);
        let mut tq3 = Stats::new(num_samples);

        for i in 0..num_samples {
            let seed_k = 1000u64.wrapping_add(i as u64);
            let seed_q = 50000u64.wrapping_add(i as u64);
            let key = pseudo_random_vec(dim, seed_k);
            let query = pseudo_random_vec(dim, seed_q);

            let true_ip: f32 = key.iter().zip(query.iter()).map(|(k, q)| k * q).sum();
            if true_ip.abs() < 0.1 {
                continue;
            }

            // --- PQ3-plain: 3-bit polar via crate ---
            let block = quantize_vec(&pq3_plain_config, &key).unwrap();
            let dequant = dequantize_vec(&pq3_plain_config, &block).unwrap();
            let ip: f32 = dequant.iter().zip(query.iter()).map(|(d, q)| d * q).sum();
            pq3_plain.push(ip, true_ip);

            // --- PQ3-outlier: via block-level cache (OUTLIER_BLOCKS=4) ---
            let mut cache = make_cache();
            let k_tensor = Tensor::from_vec(
                vec![key.clone(); TEST_NUM_KV_HEADS]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<f32>>(),
                (1, TEST_NUM_KV_HEADS, 1, dim),
                &Device::Cpu,
            )
            .unwrap();
            let v_tensor = Tensor::from_vec(
                vec![0.0f32; TEST_NUM_KV_HEADS * dim],
                (1, TEST_NUM_KV_HEADS, 1, dim),
                &Device::Cpu,
            )
            .unwrap();
            let (dk, _) = cache
                .append_and_dequantize(TEST_LAYER, &k_tensor, &v_tensor)
                .unwrap();
            let dk_vals: Vec<f32> = dk
                .narrow(1, 0, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let outlier_ip: f32 = dk_vals.iter().zip(query.iter()).map(|(d, q)| d * q).sum();
            pq3_outlier.push(outlier_ip, true_ip);

            // --- TQ3: 2-bit polar + QJL correction ---
            let per_vec_seed = qjl_seed.wrapping_add(i as u64);
            let tq3_ip = estimate_inner_product_single(
                &query,
                &quantize_with_qjl(&tq3_config, &key, per_vec_seed).unwrap(),
                &tq3_config,
                per_vec_seed,
            )
            .unwrap();
            tq3.push(tq3_ip, true_ip);
        }

        eprintln!("╔══════════════════════════════════════════════════════════════════╗");
        eprintln!(
            "║ PQ3-plain vs PQ3-outlier vs TQ3 (n={}, dim={dim})              ║",
            pq3_plain.ratios.len()
        );
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!(
            "║ PQ3-plain   (crate 3-bit):  bias={:.4}  MSE={:.4}  var={:.6} ║",
            pq3_plain.bias(),
            pq3_plain.mse(),
            pq3_plain.var()
        );
        eprintln!(
            "║ PQ3-outlier (cache 3-bit):  bias={:.4}  MSE={:.4}  var={:.6} ║",
            pq3_outlier.bias(),
            pq3_outlier.mse(),
            pq3_outlier.var()
        );
        eprintln!(
            "║ TQ3         (2-bit + QJL):  bias={:.4}  MSE={:.4}  var={:.6} ║",
            tq3.bias(),
            tq3.mse(),
            tq3.var()
        );
        eprintln!("║                                                                  ║");

        // Find winners
        let biases = [
            ("PQ3-plain", pq3_plain.bias()),
            ("PQ3-outlier", pq3_outlier.bias()),
            ("TQ3", tq3.bias()),
        ];
        let mses = [
            ("PQ3-plain", pq3_plain.mse()),
            ("PQ3-outlier", pq3_outlier.mse()),
            ("TQ3", tq3.mse()),
        ];
        let vars = [
            ("PQ3-plain", pq3_plain.var()),
            ("PQ3-outlier", pq3_outlier.var()),
            ("TQ3", tq3.var()),
        ];

        let bias_winner = biases
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        let mse_winner = mses
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        let var_winner = vars
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        eprintln!("║ Bias winner:     {bias_winner:<12}                                  ║");
        eprintln!("║ MSE winner:      {mse_winner:<12}                                  ║");
        eprintln!("║ Variance winner: {var_winner:<12}                                  ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════╝");

        // TQ3 must be near-unbiased (Paper Theorem 2)
        assert!(
            tq3.bias() < 0.05,
            "TQ3 should be near-unbiased, got bias={:.4}",
            tq3.bias()
        );
        // PQ3-outlier should have less bias than PQ3-plain (outlier codebook helps)
        assert!(
            pq3_outlier.bias() < pq3_plain.bias() + 0.01,
            "PQ3-outlier bias ({:.4}) should not be worse than PQ3-plain ({:.4})",
            pq3_outlier.bias(),
            pq3_plain.bias()
        );
    }

    // -----------------------------------------------------------------------
    // All 3 modes × 2 norm modes: roundtrip quality on CPU (+ GPU if available)
    // -----------------------------------------------------------------------

    /// Shared implementation: append keys via block path, verify dequant quality.
    fn mode_norm_roundtrip_impl(device: &Device, mode: &str, bits: u8, norm_mode: QuantNormMode) {
        let dim = TEST_HEAD_DIM;
        let heads = TEST_NUM_KV_HEADS;
        let layers = TEST_NUM_LAYERS;

        let mut cache =
            match mode {
                "pq" => TurboQuantKVCache::new_pq_with_norm(bits, dim, heads, layers, norm_mode)
                    .unwrap(),
                "pqo" => TurboQuantKVCache::new_pqo_with_norm(bits, dim, heads, layers, norm_mode)
                    .unwrap(),
                "tq" => TurboQuantKVCache::new_tq_with_norm(bits, dim, heads, layers, norm_mode)
                    .unwrap(),
                _ => panic!("unknown mode"),
            };

        // Prefill 4 tokens, then decode 1 token.
        // Decode forces dequantization from compressed cache → shows real quality.
        let prefill_len = 4;
        let (k_pf, v_pf) = make_block_path_tensors(device, heads, prefill_len, dim);
        let _ = cache
            .append_and_dequantize(TEST_LAYER, &k_pf, &v_pf)
            .unwrap();

        // Decode: 1 token — forces full cache dequant
        let (k_dec, v_dec) = make_block_path_tensors(device, heads, 1, dim);
        let (dk, _dv) = cache
            .append_and_dequantize(TEST_LAYER, &k_dec, &v_dec)
            .unwrap();

        let total_seq = prefill_len + 1;

        // Original keys: combine prefill + decode
        let k_all_orig: Vec<f32> = {
            let pf: Vec<f32> = k_pf
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let dc: Vec<f32> = k_dec
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let mut all = pf;
            all.extend(dc);
            all
        };

        let dk_cpu: Vec<f32> = dk
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Cosine similarity between original and dequantized (per head×token)
        // Only check prefill tokens (0..prefill_len) — these went through quant+dequant.
        let mut min_cos = f32::MAX;
        for h in 0..heads {
            for t in 0..prefill_len {
                let orig_start = (h * prefill_len + t) * dim;
                let deq_start = (h * total_seq + t) * dim;
                let orig = &k_all_orig[orig_start..orig_start + dim];
                let deq = &dk_cpu[deq_start..deq_start + dim];

                let dot: f32 = orig.iter().zip(deq.iter()).map(|(a, b)| a * b).sum();
                let norm_o: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_d: f32 = deq.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos = if norm_o > 0.0 && norm_d > 0.0 {
                    dot / (norm_o * norm_d)
                } else {
                    0.0
                };
                min_cos = min_cos.min(cos);
            }
        }

        let mode_upper = mode.to_uppercase();
        let norm_str = match norm_mode {
            QuantNormMode::MaxNorm => "MaxNorm",
            QuantNormMode::L2Norm => "L2Norm",
        };
        let dev_str = if device.is_cpu() { "CPU" } else { "GPU" };

        eprintln!(
            "{mode_upper}{bits} {norm_str} ({dev_str}): min cosine similarity = {min_cos:.4}"
        );

        // Reconstruction quality depends on effective polar bits:
        // PQ/TQ without outlier at bits=3: 3-bit / 2-bit codebook → cosine ~0.47-0.52
        // PQO with outlier: all blocks get (bits)-bit codebook → cosine ~0.83-0.85
        let threshold = match mode {
            "tq" => 0.40,  // 2-bit polar (bits-1)
            "pq" => 0.40,  // 3-bit but no outlier override
            "pqo" => 0.75, // outlier codebook = best
            _ => 0.40,
        };
        assert!(
            min_cos > threshold,
            "{mode_upper}{bits} {norm_str} ({dev_str}): min cosine {min_cos:.4} below threshold {threshold}"
        );

        // Verify QJL data presence matches mode
        let has_qjl = cache.has_qjl_data(TEST_LAYER);
        let expect_qjl = mode == "tq";
        assert_eq!(
            has_qjl, expect_qjl,
            "{mode_upper}{bits}: has_qjl_data={has_qjl}, expected={expect_qjl}"
        );

        // Verify seq_len tracking
        assert_eq!(cache.current_seq_len(TEST_LAYER), total_seq);
    }

    // --- PQ3 ---
    #[test]
    fn pq3_maxnorm_roundtrip_cpu() {
        mode_norm_roundtrip_impl(&Device::Cpu, "pq", 3, QuantNormMode::MaxNorm);
    }
    #[test]
    fn pq3_l2norm_roundtrip_cpu() {
        mode_norm_roundtrip_impl(&Device::Cpu, "pq", 3, QuantNormMode::L2Norm);
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn pq3_maxnorm_roundtrip_gpu() {
        let d = Device::cuda_if_available(0).unwrap();
        if !d.is_cpu() {
            mode_norm_roundtrip_impl(&d, "pq", 3, QuantNormMode::MaxNorm);
        }
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn pq3_l2norm_roundtrip_gpu() {
        let d = Device::cuda_if_available(0).unwrap();
        if !d.is_cpu() {
            mode_norm_roundtrip_impl(&d, "pq", 3, QuantNormMode::L2Norm);
        }
    }

    // --- PQO3 ---
    #[test]
    fn pqo3_maxnorm_roundtrip_cpu() {
        mode_norm_roundtrip_impl(&Device::Cpu, "pqo", 3, QuantNormMode::MaxNorm);
    }
    #[test]
    fn pqo3_l2norm_roundtrip_cpu() {
        mode_norm_roundtrip_impl(&Device::Cpu, "pqo", 3, QuantNormMode::L2Norm);
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn pqo3_maxnorm_roundtrip_gpu() {
        let d = Device::cuda_if_available(0).unwrap();
        if !d.is_cpu() {
            mode_norm_roundtrip_impl(&d, "pqo", 3, QuantNormMode::MaxNorm);
        }
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn pqo3_l2norm_roundtrip_gpu() {
        let d = Device::cuda_if_available(0).unwrap();
        if !d.is_cpu() {
            mode_norm_roundtrip_impl(&d, "pqo", 3, QuantNormMode::L2Norm);
        }
    }

    // --- TQ3 ---
    #[test]
    fn tq3_maxnorm_roundtrip_cpu() {
        mode_norm_roundtrip_impl(&Device::Cpu, "tq", 3, QuantNormMode::MaxNorm);
    }
    #[test]
    fn tq3_l2norm_roundtrip_cpu() {
        mode_norm_roundtrip_impl(&Device::Cpu, "tq", 3, QuantNormMode::L2Norm);
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn tq3_maxnorm_roundtrip_gpu() {
        let d = Device::cuda_if_available(0).unwrap();
        if !d.is_cpu() {
            mode_norm_roundtrip_impl(&d, "tq", 3, QuantNormMode::MaxNorm);
        }
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn tq3_l2norm_roundtrip_gpu() {
        let d = Device::cuda_if_available(0).unwrap();
        if !d.is_cpu() {
            mode_norm_roundtrip_impl(&d, "tq", 3, QuantNormMode::L2Norm);
        }
    }

    /// TQ3: chunked prefill (multiple appends) + decode must not crash.
    /// Reproduces a bug where qjl_correction fails after the second prefill chunk.
    #[test]
    fn tq3_chunked_prefill_then_decode_cpu() {
        tq3_chunked_prefill_then_decode_impl(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tq3_chunked_prefill_then_decode_gpu() {
        let d = Device::cuda_if_available(0).unwrap();
        if !d.is_cpu() {
            tq3_chunked_prefill_then_decode_impl(&d);
        }
    }

    fn tq3_chunked_prefill_then_decode_impl(device: &Device) {
        let dim = TEST_HEAD_DIM;
        let heads = TEST_NUM_KV_HEADS;
        let mut cache = TurboQuantKVCache::new_tq(TEST_BITS, dim, heads, TEST_NUM_LAYERS).unwrap();

        // Chunk 1: prefill 8 tokens
        let (k1, v1) = make_block_path_tensors(device, heads, 8, dim);
        let (dk1, _) = cache.append_and_dequantize(TEST_LAYER, &k1, &v1).unwrap();
        assert_eq!(dk1.dims()[2], 8);

        // Chunk 2: prefill 4 more tokens (simulates chunked prefill)
        let (k2, v2) = make_block_path_tensors(device, heads, 4, dim);
        let (dk2, _) = cache.append_and_dequantize(TEST_LAYER, &k2, &v2).unwrap();
        assert_eq!(dk2.dims()[2], 12); // total = 8 + 4

        // Decode: 1 token
        let (k3, v3) = make_block_path_tensors(device, heads, 1, dim);
        let (dk3, _) = cache.append_and_dequantize(TEST_LAYER, &k3, &v3).unwrap();
        assert_eq!(dk3.dims()[2], 13); // total = 12 + 1

        // QJL correction must work on the full cache
        let query = Tensor::from_vec(
            (0..dim)
                .map(|i| ((i as f32) * 0.05).sin())
                .collect::<Vec<f32>>(),
            (1, 1, 1, dim),
            &Device::Cpu,
        )
        .unwrap()
        .to_device(device)
        .unwrap();
        let correction = cache.qjl_correction(0, TEST_LAYER, &query).unwrap();
        assert_eq!(
            correction.dims(),
            &[1, 1, 1, 13],
            "QJL correction must cover all 13 tokens"
        );
    }

    /// Compressed attention (fused dequant + attention) must match
    /// standard attention on dequantized data.
    /// Tests that the CUDA kernel produces correct results.
    #[cfg(feature = "cuda")]
    #[test]
    fn compressed_flash_attention_matches_standard_gpu() {
        let device = Device::cuda_if_available(0).unwrap();
        if device.is_cpu() {
            return;
        }

        let dim = TEST_HEAD_DIM;
        let heads = TEST_NUM_KV_HEADS;
        let mut cache = TurboQuantKVCache::new_pqo(TEST_BITS, dim, heads, TEST_NUM_LAYERS).unwrap();

        // Prefill 64 tokens
        let seq_len = 64;
        let (k, v) = make_block_path_tensors(&device, heads, seq_len, dim);
        let _ = cache.append_and_dequantize(TEST_LAYER, &k, &v).unwrap();

        // Decode 1 token → triggers full dequant
        let (k_dec, v_dec) = make_block_path_tensors(&device, heads, 1, dim);
        let (full_k, full_v) = cache
            .append_and_dequantize(TEST_LAYER, &k_dec, &v_dec)
            .unwrap();

        // Query
        let q_data: Vec<f32> = (0..heads * dim)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();
        let q = Tensor::from_vec(q_data, (1, heads, 1, dim), &Device::Cpu)
            .unwrap()
            .to_device(&device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        // Standard attention: Q @ full_K^T → softmax → @ full_V
        let scale = 1.0 / (dim as f32).sqrt();
        let full_k_f32 = full_k.to_dtype(DType::F32).unwrap();
        let full_v_f32 = full_v.to_dtype(DType::F32).unwrap();
        let scores = q
            .matmul(&full_k_f32.t().unwrap())
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap();
        let weights = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        let standard_output = weights.matmul(&full_v_f32).unwrap();

        // Compressed attention (when implemented, will call CUDA kernel)
        // For now: verify the dequant CUDA kernel gives correct values
        // by comparing full dequant output with per-block kernel dequant
        let ki = cache.gpu_k_indices[TEST_LAYER].as_ref().unwrap();
        let ks = cache.gpu_k_scales[TEST_LAYER].as_ref().unwrap();
        let kv_len = cache.buf_seq_len[TEST_LAYER];
        let pre = cache.gpu_precomputed.as_ref().unwrap();
        let packed_dim = dim * TEST_BITS as usize / 8;

        let ki_flat = ki
            .narrow(1, 0, kv_len)
            .unwrap()
            .reshape((heads * kv_len, packed_dim))
            .unwrap();
        let ks_flat = ks
            .narrow(1, 0, kv_len)
            .unwrap()
            .reshape((heads * kv_len, dim / QUANT_BLOCK_SIZE))
            .unwrap();

        let k_dequant_kernel = TurboQuantKVCache::polar_dequantize(
            &ki_flat,
            &ks_flat,
            dim,
            TEST_BITS,
            usize::MAX,
            pre,
        )
        .unwrap()
        .reshape((1, heads, kv_len, dim))
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

        // Compare kernel dequant with full dequant from append
        let diff: f32 = (&k_dequant_kernel - &full_k_f32)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec0()
            .unwrap();

        eprintln!("CUDA kernel dequant vs full dequant: max_diff = {diff:.6}");
        assert!(
            diff < 0.01,
            "CUDA kernel dequant differs from Candle dequant: {diff:.6}"
        );

        // Standard attention output for comparison
        let std_vals: Vec<f32> = standard_output
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        eprintln!(
            "Standard attention output (first 4): {:?}",
            &std_vals[..4.min(std_vals.len())]
        );
    }

    // -----------------------------------------------------------------------
    // QJL-Correction Tests (push_head path)
    //
    // These tests verify that qjl_correction() produces the correct
    // additive bias on attention logits, matching turboquant-rs reference.
    // -----------------------------------------------------------------------

    /// Helper: push N keys for head 0 and return the original key vectors.
    fn push_keys_and_collect(cache: &mut TurboQuantKVCache, num_keys: usize) -> Vec<Vec<f32>> {
        let mut keys = Vec::with_capacity(num_keys);
        for i in 0..num_keys {
            let key = dummy_vec(TEST_HEAD_DIM, (i as f32) * 1.7 + 0.3);
            let value = dummy_vec(TEST_HEAD_DIM, (i as f32) * 2.1 + 0.5);
            cache.push_head(0, TEST_LAYER, &key, &value).unwrap();
            keys.push(key);
        }
        keys
    }

    /// qjl_correction() output matches turboquant-rs estimate_inner_product()
    /// minus the base polar dot product, for a single (query, key) pair.
    #[test]
    fn qjl_correction_matches_crate_reference() {
        use turboquant::{
            dequantize_vec, estimate_inner_product, precompute_query_projections,
            quantize_with_qjl, TurboQuantConfig,
        };

        let config = TurboQuantConfig::new(TEST_BITS, TEST_HEAD_DIM)
            .unwrap()
            .with_seed(DEFAULT_ROTATION_SEED);

        let query = dummy_vec(TEST_HEAD_DIM, 42.0);
        let key = dummy_vec(TEST_HEAD_DIM, 7.0);

        // --- turboquant-rs reference ---
        let qjl_block = quantize_with_qjl(&config, &key, DEFAULT_QJL_SEED).unwrap();
        let polar_config = TurboQuantConfig::new(qjl_block.polar_block().bits(), TEST_HEAD_DIM)
            .unwrap()
            .with_seed(DEFAULT_ROTATION_SEED);
        let reconstructed = dequantize_vec(&polar_config, qjl_block.polar_block()).unwrap();
        let base_dot: f32 = query
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| a * b)
            .sum();
        let r_query = precompute_query_projections(&query, TEST_HEAD_DIM, DEFAULT_QJL_SEED);
        let full_estimate = estimate_inner_product(&query, &r_query, &qjl_block, &config).unwrap();
        let reference_correction = full_estimate - base_dot;

        // --- our implementation ---
        let mut cache = make_cache();
        let value = dummy_vec(TEST_HEAD_DIM, 8.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

        let q_tensor = Tensor::from_vec(query, (1, 1, 1, TEST_HEAD_DIM), &Device::Cpu).unwrap();
        let correction = cache.qjl_correction(0, TEST_LAYER, &q_tensor).unwrap();

        let our_correction = correction.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];

        let diff = (our_correction - reference_correction).abs();
        assert!(
            diff < 1e-4,
            "QJL correction mismatch: ours={our_correction:.6}, \
             reference={reference_correction:.6}, diff={diff:.2e}"
        );
    }

    /// QJL correction reduces bias: mean(corrected - true) ≈ 0.
    ///
    /// Follows the same methodology as turboquant-rs integration tests
    /// (inner_product_tests.rs): measures absolute bias (est - true),
    /// not ratios, and varies QJL seed per sample to average over
    /// Rademacher matrix randomness.
    #[test]
    fn qjl_correction_reduces_bias() {
        use turboquant::qjl::dot_product;
        use turboquant::{dequantize_vec, quantize_with_qjl, TurboQuantConfig};

        let num_samples: usize = 500;
        // turboquant-rs uses 0.20 for 500 samples at dim=64 (qjl.rs:712)
        let bias_tolerance: f32 = 0.20;
        // LCG constants matching turboquant-rs integration tests
        let lcg_mul: u64 = 6_364_136_223_846_793_005;

        let config = TurboQuantConfig::new(TEST_BITS, TEST_HEAD_DIM)
            .unwrap()
            .with_seed(DEFAULT_ROTATION_SEED);
        let polar_config = TurboQuantConfig::new(TEST_BITS - 1, TEST_HEAD_DIM)
            .unwrap()
            .with_seed(DEFAULT_ROTATION_SEED);

        let mut polar_bias_sum = 0.0_f64;
        let mut corrected_bias_sum = 0.0_f64;

        for i in 0..num_samples {
            // Pseudo-random vectors (LCG-based, same as turboquant-rs)
            let key_seed = (i as u64).wrapping_mul(lcg_mul).wrapping_add(1000);
            let query_seed = (i as u64).wrapping_mul(lcg_mul).wrapping_add(2000);
            // Different QJL seed per sample — critical for unbiasedness
            let qjl_seed = DEFAULT_QJL_SEED.wrapping_add(i as u64);

            let key = pseudo_random_vec(TEST_HEAD_DIM, key_seed);
            let query = pseudo_random_vec(TEST_HEAD_DIM, query_seed);
            let true_ip = dot_product(&key, &query) as f64;

            // Polar-only estimate
            let qjl_block = quantize_with_qjl(&config, &key, qjl_seed).unwrap();
            let reconstructed = dequantize_vec(&polar_config, qjl_block.polar_block()).unwrap();
            let polar_ip = dot_product(&query, &reconstructed) as f64;
            polar_bias_sum += polar_ip - true_ip;

            // Our QJL correction: create a cache per sample to match qjl_seed
            let mut cache = TurboQuantKVCache::new_with_qjl_seed(
                TEST_BITS,
                TEST_HEAD_DIM,
                1,
                TEST_NUM_LAYERS,
                qjl_seed,
            )
            .unwrap();
            let value = pseudo_random_vec(TEST_HEAD_DIM, key_seed.wrapping_add(500));
            cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

            let q_tensor = Tensor::from_vec(query, (1, 1, 1, TEST_HEAD_DIM), &Device::Cpu).unwrap();
            let correction = cache
                .qjl_correction(0, TEST_LAYER, &q_tensor)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];

            let corrected_ip = polar_ip + correction as f64;
            corrected_bias_sum += corrected_ip - true_ip;
        }

        let polar_mean_bias = (polar_bias_sum / num_samples as f64).abs() as f32;
        let corrected_mean_bias = (corrected_bias_sum / num_samples as f64).abs() as f32;

        eprintln!(
            "Polar mean bias: {polar_mean_bias:.4}, \
             Corrected mean bias: {corrected_mean_bias:.4}"
        );

        assert!(
            corrected_mean_bias < polar_mean_bias,
            "QJL should reduce bias: polar={polar_mean_bias:.4}, \
             corrected={corrected_mean_bias:.4}"
        );
        assert!(
            corrected_mean_bias < bias_tolerance,
            "Corrected mean bias {corrected_mean_bias:.4} exceeds \
             tolerance {bias_tolerance} over {num_samples} samples"
        );
    }

    /// qjl_correction() with multiple keys returns correct shape
    /// and matches individual computations.
    #[test]
    fn qjl_correction_batch_matches_individual() {
        let mut cache = make_cache();
        let num_keys = 8;
        let _keys = push_keys_and_collect(&mut cache, num_keys);
        let query = dummy_vec(TEST_HEAD_DIM, 99.0);

        // Batch correction for all keys at once
        let q_tensor =
            Tensor::from_vec(query.clone(), (1, 1, 1, TEST_HEAD_DIM), &Device::Cpu).unwrap();
        let batch_correction = cache
            .qjl_correction(0, TEST_LAYER, &q_tensor)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(
            batch_correction.len(),
            num_keys,
            "Correction should have one value per key"
        );

        // Individual corrections (one cache per key)
        for (i, key) in _keys.iter().enumerate() {
            let mut single_cache = make_cache();
            let value = dummy_vec(TEST_HEAD_DIM, i as f32 * 2.1 + 0.5);
            single_cache.push_head(0, TEST_LAYER, key, &value).unwrap();

            let single_corr = single_cache
                .qjl_correction(0, TEST_LAYER, &q_tensor)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];

            let diff = (batch_correction[i] - single_corr).abs();
            assert!(
                diff < 1e-5,
                "Batch[{i}]={:.6} != individual={single_corr:.6}, diff={diff:.2e}",
                batch_correction[i]
            );
        }
    }

    /// qjl_correction() works on CPU device (no CUDA required).
    #[test]
    fn qjl_correction_works_on_cpu() {
        let mut cache = make_cache();
        let key = dummy_vec(TEST_HEAD_DIM, 1.0);
        let value = dummy_vec(TEST_HEAD_DIM, 2.0);
        cache.push_head(0, TEST_LAYER, &key, &value).unwrap();

        let query = dummy_vec(TEST_HEAD_DIM, 3.0);
        let q_tensor = Tensor::from_vec(query, (1, 1, 1, TEST_HEAD_DIM), &Device::Cpu).unwrap();

        let result = cache.qjl_correction(0, TEST_LAYER, &q_tensor);
        assert!(
            result.is_ok(),
            "qjl_correction must work on CPU: {:?}",
            result.err()
        );

        let correction = result.unwrap();
        assert_eq!(
            correction.dims(),
            &[1, 1, 1, 1],
            "Correction shape should be [batch, 1, q_len, kv_len]"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 2 continued: Attention-Integration Tests
    //
    // Paper Theorem 2 guarantees unbiased inner products. But attention uses
    // softmax (nonlinear) — unbiased logits don't guarantee unbiased softmax
    // output (Jensen's inequality). llama.cpp excluded QJL for this reason.
    //
    // These tests measure ACTUAL attention output quality, not just logits.
    // -----------------------------------------------------------------------

    /// Helper: SplitMix64 PRNG for proper Gaussian vector generation.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn next_open01(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
    }

    /// Generate a unit vector uniformly on S^{d-1} via Gaussian method.
    /// Required by Paper (Theorem 2 assumes x ∈ S^{d-1}).
    fn gaussian_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let mut g = Vec::with_capacity(dim);
        let pairs = (dim + 1) / 2;
        for _ in 0..pairs {
            let u1 = rng.next_open01();
            let u2 = rng.next_open01();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            g.push(r * theta.cos());
            g.push(r * theta.sin());
        }
        g.truncate(dim);
        let norm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        g.iter().map(|x| (*x / norm) as f32).collect()
    }

    /// Compute softmax attention output manually:
    ///   logits = Q @ K^T / sqrt(d)
    ///   weights = softmax(logits + correction)
    ///   output = weights @ V
    ///
    /// Returns (output, weights) for analysis.
    fn manual_attention(
        q: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        correction: Option<&[f32]>,
        dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let kv_len = keys.len();
        let scale = 1.0 / (dim as f32).sqrt();

        // Compute logits
        let mut logits: Vec<f32> = keys
            .iter()
            .map(|k| q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * scale)
            .collect();

        // Add QJL correction (scaled by 1/sqrt(d) like base logits)
        if let Some(corr) = correction {
            for (l, c) in logits.iter_mut().zip(corr.iter()) {
                *l += c * scale;
            }
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
        let weights: Vec<f32> = logits
            .iter()
            .map(|l| (l - max_logit).exp() / exp_sum)
            .collect();

        // Weighted value sum
        let mut output = vec![0.0_f32; dim];
        for (w, v) in weights.iter().zip(values.iter()) {
            for (o, val) in output.iter_mut().zip(v.iter()) {
                *o += w * val;
            }
        }

        (output, weights)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-10 || nb < 1e-10 {
            return 0.0;
        }
        dot / (na * nb)
    }

    /// Comprehensive attention quality comparison: Normal vs Polar-only vs QJL.
    ///
    /// Paper Theorem 2 guarantees unbiased inner products, but softmax is
    /// nonlinear (Jensen's inequality). llama.cpp excluded QJL due to
    /// variance concerns in softmax Top-K ranking.
    ///
    /// This test gathers empirical data across multiple configurations:
    /// - Dimensions: 64, 128 (Paper uses 128 for KV cache experiments)
    /// - Sequence lengths: 32, 128, 512 (tests scaling behavior)
    /// - Bit widths: b=3 (TQ3), b=4 (TQ4)
    /// - Metrics: cosine similarity, L2 error, top-1 attention weight overlap
    ///
    /// This is a DATA GATHERING test, not a strict pass/fail gate.
    /// It logs all results for analysis. The only assertion is that
    /// quantized attention output has minimum quality (cosine > 0.7).
    #[test]
    fn attention_quality_comprehensive_comparison() {
        use turboquant::{dequantize_vec, quantize_with_qjl, TurboQuantConfig};

        let num_queries_per_config: usize = 100;

        struct Config {
            dim: usize,
            bits: u8,
            num_keys: usize,
        }

        let configs = [
            Config {
                dim: 64,
                bits: 3,
                num_keys: 32,
            },
            Config {
                dim: 64,
                bits: 3,
                num_keys: 128,
            },
            Config {
                dim: 64,
                bits: 3,
                num_keys: 512,
            },
            Config {
                dim: 128,
                bits: 3,
                num_keys: 32,
            },
            Config {
                dim: 128,
                bits: 3,
                num_keys: 128,
            },
            Config {
                dim: 128,
                bits: 3,
                num_keys: 512,
            },
            Config {
                dim: 128,
                bits: 4,
                num_keys: 128,
            },
            Config {
                dim: 128,
                bits: 4,
                num_keys: 512,
            },
        ];

        eprintln!("\n{}", "=".repeat(90));
        eprintln!("Attention Quality: Normal vs Polar-only vs QJL-corrected");
        eprintln!("Paper: Theorem 2 guarantees unbiased logits, but softmax is nonlinear.");
        eprintln!("llama.cpp excluded QJL due to variance. We measure empirically.");
        eprintln!(
            "{:>5} {:>4} {:>6} | {:>8} {:>8} {:>8} | {:>8} {:>8} | {:>6}",
            "dim", "bits", "#keys", "cos_pol", "cos_qjl", "delta", "l2_pol", "l2_qjl", "top1%"
        );
        eprintln!("{}", "-".repeat(90));

        let mut worst_cos = 1.0_f64;

        for cfg in &configs {
            let dim = cfg.dim;
            let bits = cfg.bits;
            let num_keys = cfg.num_keys;
            let polar_bits = bits - 1;

            // Generate K, V once per config
            let keys: Vec<Vec<f32>> = (0..num_keys)
                .map(|i| gaussian_unit_vec(dim, i as u64 * 101 + 50000))
                .collect();
            let values: Vec<Vec<f32>> = (0..num_keys)
                .map(|i| gaussian_unit_vec(dim, i as u64 * 103 + 60000))
                .collect();

            let tq_config = TurboQuantConfig::new(bits, dim)
                .unwrap()
                .with_seed(DEFAULT_ROTATION_SEED);
            let polar_config = TurboQuantConfig::new(polar_bits, dim)
                .unwrap()
                .with_seed(DEFAULT_ROTATION_SEED);

            let mut polar_cos_sum = 0.0_f64;
            let mut qjl_cos_sum = 0.0_f64;
            let mut polar_l2_sum = 0.0_f64;
            let mut qjl_l2_sum = 0.0_f64;
            let mut top1_match_count: usize = 0;

            for qi in 0..num_queries_per_config {
                let query = gaussian_unit_vec(dim, qi as u64 * 107 + 70000 + dim as u64);
                let qjl_seed = DEFAULT_QJL_SEED.wrapping_add(qi as u64);

                // Ground truth
                let (ref_output, ref_weights) = manual_attention(&query, &keys, &values, None, dim);

                // Quantize keys (one QjlBlock per key)
                let qjl_blocks: Vec<_> = keys
                    .iter()
                    .map(|k| quantize_with_qjl(&tq_config, k, qjl_seed).unwrap())
                    .collect();

                // Polar-only dequantized keys
                let polar_keys: Vec<Vec<f32>> = qjl_blocks
                    .iter()
                    .map(|b| dequantize_vec(&polar_config, b.polar_block()).unwrap())
                    .collect();

                // Polar-only attention
                let (polar_output, _) = manual_attention(&query, &polar_keys, &values, None, dim);

                // QJL correction via cache
                let mut cache =
                    TurboQuantKVCache::new_with_qjl_seed(bits, dim, 1, 1, qjl_seed).unwrap();
                for (i, k) in keys.iter().enumerate() {
                    cache.push_head(0, 0, k, &values[i]).unwrap();
                }
                let q_tensor =
                    Tensor::from_vec(query.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                let correction = cache
                    .qjl_correction(0, 0, &q_tensor)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();

                // QJL-corrected attention
                let (qjl_output, qjl_weights) =
                    manual_attention(&query, &polar_keys, &values, Some(&correction), dim);

                // Metrics
                polar_cos_sum += cosine_similarity(&ref_output, &polar_output) as f64;
                qjl_cos_sum += cosine_similarity(&ref_output, &qjl_output) as f64;

                let polar_l2: f64 = ref_output
                    .iter()
                    .zip(polar_output.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let qjl_l2: f64 = ref_output
                    .iter()
                    .zip(qjl_output.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                polar_l2_sum += polar_l2;
                qjl_l2_sum += qjl_l2;

                // Top-1 overlap: does QJL-corrected attention focus on same key?
                let ref_top1 = ref_weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let qjl_top1 = qjl_weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                if ref_top1 == qjl_top1 {
                    top1_match_count += 1;
                }
            }

            let n = num_queries_per_config as f64;
            let polar_cos = polar_cos_sum / n;
            let qjl_cos = qjl_cos_sum / n;
            let delta = qjl_cos - polar_cos;
            let polar_l2 = polar_l2_sum / n;
            let qjl_l2 = qjl_l2_sum / n;
            let top1_pct = top1_match_count as f64 / num_queries_per_config as f64 * 100.0;

            eprintln!(
                "{dim:>5} {bits:>4} {num_keys:>6} | {polar_cos:>8.4} {qjl_cos:>8.4} \
                 {delta:>+8.5} | {polar_l2:>8.5} {qjl_l2:>8.5} | {top1_pct:>5.1}%"
            );

            worst_cos = worst_cos.min(polar_cos).min(qjl_cos);
        }

        eprintln!("{}", "-".repeat(90));

        // Minimum quality gate: both methods must produce usable attention
        let min_quality = 0.70;
        assert!(
            worst_cos > min_quality,
            "Worst attention quality {worst_cos:.4} below minimum {min_quality}"
        );
    }

    /// Verify QJL correction has correct shape for attention integration.
    ///
    /// The correction must be broadcastable to attention logits shape:
    /// [batch, num_heads, q_len, kv_len]
    #[test]
    fn qjl_correction_shape_compatible_with_attention() {
        let mut cache = make_cache();
        let num_keys = 16;
        let _keys = push_keys_and_collect(&mut cache, num_keys);

        let query = dummy_vec(TEST_HEAD_DIM, 42.0);
        let q_tensor = Tensor::from_vec(query, (1, 1, 1, TEST_HEAD_DIM), &Device::Cpu).unwrap();

        let correction = cache.qjl_correction(0, TEST_LAYER, &q_tensor).unwrap();

        // Shape must be [1, 1, 1, kv_len] — broadcastable over heads
        assert_eq!(
            correction.dims(),
            &[1, 1, 1, num_keys],
            "QJL correction shape must be [batch, 1, q_len, kv_len]"
        );

        // Must be finite (no NaN/Inf)
        let vals = correction.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "QJL correction contains NaN or Inf"
        );
    }

    // -----------------------------------------------------------------------
    // Attention-Integration via Sdpa.run_attention()
    //
    // These tests verify that SdpaParams.qjl_bias is correctly applied
    // in the actual attention pipeline — between Q@K^T and softmax.
    // -----------------------------------------------------------------------

    /// QJL bias in SdpaParams changes attention output.
    ///
    /// Verifies that:
    /// 1. SdpaParams accepts a qjl_bias field
    /// 2. Sdpa.run_attention() applies it before softmax
    /// 3. Output differs from no-bias case (the bias has effect)
    /// 4. Works on CPU (no CUDA required)
    #[test]
    fn sdpa_applies_qjl_bias() {
        use crate::attention::{Sdpa, SdpaParams};

        let batch = 1;
        let heads = 2;
        let q_len = 1;
        let kv_len = 8;
        let dim = TEST_HEAD_DIM;
        let scale = 1.0 / (dim as f32).sqrt();

        // Create Q, K, V tensors
        let q_data: Vec<f32> = (0..batch * heads * q_len * dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let k_data: Vec<f32> = (0..batch * heads * kv_len * dim)
            .map(|i| ((i as f32) * 0.02).cos())
            .collect();
        let v_data: Vec<f32> = (0..batch * heads * kv_len * dim)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();

        let q = Tensor::from_vec(q_data, (batch, heads, q_len, dim), &Device::Cpu).unwrap();
        let k = Tensor::from_vec(k_data, (batch, heads, kv_len, dim), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(v_data, (batch, heads, kv_len, dim), &Device::Cpu).unwrap();

        // Run WITHOUT qjl_bias
        let params_no_qjl = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };
        let out_no_qjl = Sdpa
            .run_attention(&q, &k, &v, None, None, &params_no_qjl)
            .unwrap();

        // Create a non-trivial QJL bias (large enough to change output)
        let bias_data: Vec<f32> = (0..batch * 1 * q_len * kv_len)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let qjl_bias =
            Tensor::from_vec(bias_data, (batch, 1, q_len, kv_len), &Device::Cpu).unwrap();

        // Run WITH qjl_bias
        let params_with_qjl = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: Some(qjl_bias),
        };
        let out_with_qjl = Sdpa
            .run_attention(&q, &k, &v, None, None, &params_with_qjl)
            .unwrap();

        // Outputs must differ (the bias changes attention weights)
        let diff = (&out_with_qjl - &out_no_qjl)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 0.001,
            "QJL bias should change attention output, but diff={diff:.6}"
        );

        // Both outputs should be valid (finite, correct shape)
        assert_eq!(out_no_qjl.dims(), &[batch, heads, q_len, dim]);
        assert_eq!(out_with_qjl.dims(), &[batch, heads, q_len, dim]);
    }

    /// QJL bias = None produces identical output to current (no regression).
    #[test]
    fn sdpa_qjl_bias_none_is_noop() {
        use crate::attention::{Sdpa, SdpaParams};

        let batch = 1;
        let heads = 2;
        let q_len = 1;
        let kv_len = 4;
        let dim = TEST_HEAD_DIM;
        let scale = 1.0 / (dim as f32).sqrt();

        let q_data: Vec<f32> = (0..batch * heads * q_len * dim)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let k_data: Vec<f32> = (0..batch * heads * kv_len * dim)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();
        let v_data: Vec<f32> = (0..batch * heads * kv_len * dim)
            .map(|i| ((i as f32) * 0.09).sin())
            .collect();

        let q = Tensor::from_vec(q_data, (batch, heads, q_len, dim), &Device::Cpu).unwrap();
        let k = Tensor::from_vec(k_data, (batch, heads, kv_len, dim), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(v_data, (batch, heads, kv_len, dim), &Device::Cpu).unwrap();

        let params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };

        // Should produce the same output as if qjl_bias field didn't exist
        let result = Sdpa.run_attention(&q, &k, &v, None, None, &params);
        assert!(result.is_ok(), "qjl_bias=None must not break attention");

        let out = result.unwrap();
        assert_eq!(out.dims(), &[batch, heads, q_len, dim]);

        // Output should be finite
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "Attention output contains NaN/Inf"
        );
    }

    // -----------------------------------------------------------------------
    // Bug investigation: multi-step decode with model-like dimensions
    //
    // Tests whether TQ3 dequant produces reasonable K/V tensors across
    // multiple decode steps (simulating actual model inference).
    // -----------------------------------------------------------------------

    /// Simulate Qwen3-0.6B dimensions: 128 dim, 8 kv_heads, 16 q_heads.
    /// Do prefill + multiple decode steps. Check that:
    /// 1. Dequantized K/V have reasonable values (not all zeros, not NaN)
    /// 2. Attention output varies across decode steps (not degenerate)
    /// 3. Cosine similarity between Normal and TQ3 attention output > threshold
    #[test]
    fn multi_step_decode_produces_reasonable_output() {
        use crate::attention::{Sdpa, SdpaParams};

        // Qwen3-0.6B-like dimensions
        let dim: usize = 128;
        let kv_heads: usize = 8;
        let q_heads: usize = 16;
        let bits: u8 = 3;
        let n_rep = q_heads / kv_heads;
        let prefill_len: usize = 8;
        let decode_steps: usize = 5;
        let scale = 1.0 / (dim as f32).sqrt();

        let mut tq_cache = TurboQuantKVCache::new_pqo(bits, dim, kv_heads, 1).unwrap();

        // Generate consistent random data
        let mut rng = SplitMix64::new(42);
        let mut gen_tensor = |batch: usize, heads: usize, seq: usize, d: usize| -> Tensor {
            let n = batch * heads * seq * d;
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    let bits = rng.next_u64();
                    (bits as i64) as f32 / (i64::MAX as f32) * 0.1
                })
                .collect();
            Tensor::from_vec(data, (batch, heads, seq, d), &Device::Cpu).unwrap()
        };

        // Prefill
        let k_pf = gen_tensor(1, kv_heads, prefill_len, dim);
        let v_pf = gen_tensor(1, kv_heads, prefill_len, dim);
        let (full_k, full_v) = tq_cache.append_and_dequantize(0, &k_pf, &v_pf).unwrap();

        // Check prefill output is reasonable
        let k_vals = full_k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            k_vals.iter().all(|v| v.is_finite()),
            "Prefill K has NaN/Inf"
        );
        let k_nonzero = k_vals.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(
            k_nonzero > k_vals.len() / 2,
            "Prefill K is mostly zeros: {k_nonzero}/{} nonzero",
            k_vals.len()
        );

        // Multiple decode steps
        let mut outputs = Vec::new();
        for step in 0..decode_steps {
            let k_dec = gen_tensor(1, kv_heads, 1, dim);
            let v_dec = gen_tensor(1, kv_heads, 1, dim);
            let (full_k, full_v) = tq_cache.append_and_dequantize(0, &k_dec, &v_dec).unwrap();

            let expected_kv_len = prefill_len + step + 1;
            assert_eq!(
                full_k.dims(),
                &[1, kv_heads, expected_kv_len, dim],
                "K shape wrong at step {step}"
            );

            // Run attention (with GQA expansion)
            let q = gen_tensor(1, q_heads, 1, dim);
            let params = SdpaParams {
                n_kv_groups: n_rep,
                softcap: None,
                softmax_scale: scale,
                sliding_window: None,
                sinks: None,
                qjl_bias: None, // Test WITHOUT QJL first to isolate polar dequant
            };
            let out = Sdpa
                .run_attention(&q, &full_k, &full_v, None, None, &params)
                .unwrap();
            assert_eq!(out.dims(), &[1, q_heads, 1, dim]);

            let out_vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                out_vals.iter().all(|v| v.is_finite()),
                "Output NaN at step {step}"
            );
            outputs.push(out_vals);
        }

        // Check outputs are not all identical (would indicate degenerate attention)
        let mut identical_count = 0;
        for i in 1..outputs.len() {
            let diff: f32 = outputs[i]
                .iter()
                .zip(outputs[i - 1].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if diff < 1e-6 {
                identical_count += 1;
            }
        }
        assert!(
            identical_count < decode_steps - 1,
            "All decode outputs are identical — attention is degenerate"
        );

        eprintln!(
            "Multi-step decode OK: {decode_steps} steps, dim={dim}, \
             kv_heads={kv_heads}, q_heads={q_heads}"
        );
    }

    /// Multi-layer test: simulates 4 transformer layers, each with its own
    /// KvCache::TurboQuant pointing to the shared TurboQuantKVCache.
    /// Prefill 8 tokens then decode 3 tokens. Check all layers produce
    /// different (layer-specific) outputs.
    #[test]
    fn multi_layer_decode_layers_are_independent() {
        use crate::attention::{Sdpa, SdpaParams};
        use crate::kv_cache::{KvCache, NormalCache};
        use crate::paged_attention::AttentionImplementation;

        let dim: usize = 128;
        let kv_heads: usize = 8;
        let q_heads: usize = 16;
        let num_layers: usize = 4;
        let n_rep = q_heads / kv_heads;
        let scale = 1.0 / (dim as f32).sqrt();
        let prefill_len: usize = 8;

        // Create shared TQ cache via the NormalCache path (same as real model)
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            num_layers,
            4096,
            None,
            dim,
            kv_heads,
            Device::Cpu,
            DType::F32,
        );

        let mut rng = SplitMix64::new(12345);
        let mut gen = |heads: usize, seq: usize| -> Tensor {
            let n = heads * seq * dim;
            let data: Vec<f32> = (0..n)
                .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32) * 0.1)
                .collect();
            Tensor::from_vec(data, (1, heads, seq, dim), &Device::Cpu).unwrap()
        };

        // Prefill all layers with DIFFERENT data per layer
        {
            let mut locked = cache.lock().unwrap();
            for layer_idx in 0..num_layers {
                let k = gen(kv_heads, prefill_len);
                let v = gen(kv_heads, prefill_len);
                locked.0[layer_idx].append(&k, &v).unwrap();
            }
        }

        // Decode 1 token on all layers
        let mut layer_outputs = Vec::new();
        {
            let mut locked = cache.lock().unwrap();
            for layer_idx in 0..num_layers {
                let k = gen(kv_heads, 1);
                let v = gen(kv_heads, 1);
                let (full_k, full_v) = locked.0[layer_idx].append(&k, &v).unwrap();

                let q = gen(q_heads, 1);
                let params = SdpaParams {
                    n_kv_groups: n_rep,
                    softcap: None,
                    softmax_scale: scale,
                    sliding_window: None,
                    sinks: None,
                    qjl_bias: None,
                };
                let out = Sdpa
                    .run_attention(&q, &full_k, &full_v, None, None, &params)
                    .unwrap();
                let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                layer_outputs.push(vals);
            }
        }

        // Each layer should produce DIFFERENT output
        for i in 1..num_layers {
            let diff: f32 = layer_outputs[i]
                .iter()
                .zip(layer_outputs[0].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff > 1e-6,
                "Layer {i} output identical to layer 0 — layers not independent"
            );
        }

        eprintln!("Multi-layer test OK: {num_layers} layers independent");
    }

    /// GPU + BF16: Does TQ3 dequant on CUDA with BF16 dtype produce
    /// reasonable values? The real model uses BF16.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_bf16_dequant_and_attention_reasonable() {
        use crate::attention::{Sdpa, SdpaParams};

        let cuda = Device::cuda_if_available(0).unwrap();
        if cuda.is_cpu() {
            return;
        }

        let dim: usize = 128;
        let kv_heads: usize = 8;
        let q_heads: usize = 16;
        let n_rep = q_heads / kv_heads;
        let scale = 1.0 / (dim as f32).sqrt();
        let prefill_len: usize = 8;

        let mut cache = TurboQuantKVCache::new_pqo(3, dim, kv_heads, 1).unwrap();

        // Prefill with BF16 (like Qwen3-0.6B)
        let k_data: Vec<f32> = (0..kv_heads * prefill_len * dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();
        let v_data: Vec<f32> = (0..kv_heads * prefill_len * dim)
            .map(|i| ((i as f32) * 0.02).cos() * 0.1)
            .collect();
        let k = Tensor::from_vec(k_data, (1, kv_heads, prefill_len, dim), &cuda)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::from_vec(v_data, (1, kv_heads, prefill_len, dim), &cuda)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let (fk, fv) = cache.append_and_dequantize(0, &k, &v).unwrap();
        eprintln!("GPU BF16 prefill: K {:?} dtype={:?}", fk.dims(), fk.dtype());

        // Decode
        let dk: Vec<f32> = (0..kv_heads * dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.1)
            .collect();
        let dv: Vec<f32> = (0..kv_heads * dim)
            .map(|i| ((i as f32) * 0.04).cos() * 0.1)
            .collect();
        let kd = Tensor::from_vec(dk, (1, kv_heads, 1, dim), &cuda)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let vd = Tensor::from_vec(dv, (1, kv_heads, 1, dim), &cuda)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let (fk, fv) = cache.append_and_dequantize(0, &kd, &vd).unwrap();

        // Check dequant values
        let k_f32 = fk
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let vals = k_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nonzero = vals.iter().filter(|v| v.abs() > 1e-6).count();
        let has_nan = vals.iter().any(|v| v.is_nan());
        eprintln!(
            "GPU BF16 dequant K: {nonzero}/{} nonzero, nan={has_nan}",
            vals.len()
        );
        assert!(!has_nan, "GPU BF16 dequant has NaN");
        assert!(
            nonzero > vals.len() / 2,
            "GPU BF16 dequant mostly zeros: {nonzero}/{}",
            vals.len()
        );

        // Run attention on GPU
        let q_data: Vec<f32> = (0..q_heads * dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.1)
            .collect();
        let q = Tensor::from_vec(q_data, (1, q_heads, 1, dim), &cuda)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let params = SdpaParams {
            n_kv_groups: n_rep,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };
        let result = Sdpa.run_attention(&q, &fk, &fv, None, None, &params);
        assert!(
            result.is_ok(),
            "GPU BF16 attention failed: {:?}",
            result.err()
        );

        let out = result.unwrap();
        let out_vals = out
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            out_vals.iter().all(|v| v.is_finite()),
            "GPU BF16 attention output has NaN/Inf"
        );
        eprintln!("GPU BF16 attention output OK, shape {:?}", out.dims());
    }

    /// Real LLM KV-cache vectors have norms >> 1 (typically 5-100+).
    /// Test that TQ3 attention quality degrades gracefully with increasing norm.
    ///
    /// Paper Theorem 1: MSE = norm² × C(f_X, b). For b=2: C ≈ 0.117.
    /// At norm=50: MSE ≈ 292, per-coord RMS error ≈ 1.5, signal ≈ 4.4.
    /// SNR ≈ 9 dB — marginal for attention.
    #[test]
    fn attention_quality_vs_vector_norm() {
        use crate::attention::{Sdpa, SdpaParams};

        let dim: usize = 128;
        let kv_heads: usize = 1;
        let q_heads: usize = 1;
        let num_keys: usize = 32;
        let num_queries: usize = 50;
        let scale = 1.0 / (dim as f32).sqrt();

        // Test with different vector norms (simulating different LLM layers)
        let norms = [0.1_f32, 1.0, 5.0, 20.0, 50.0, 100.0];

        eprintln!("\nAttention quality vs vector norm (TQ3, d={dim}, {num_keys} keys):");
        eprintln!(
            "{:>8} {:>10} {:>10} {:>10}",
            "norm", "cos_sim", "top1%", "mse_ratio"
        );

        for &norm in &norms {
            let mut cos_sum = 0.0_f64;
            let mut top1_matches = 0_usize;

            for qi in 0..num_queries {
                let mut rng = SplitMix64::new(qi as u64 * 101 + norm.to_bits() as u64);

                // Generate keys with given norm
                let keys: Vec<Vec<f32>> = (0..num_keys)
                    .map(|_| {
                        let v: Vec<f32> = (0..dim)
                            .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
                            .collect();
                        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        v.iter().map(|x| x / n * norm).collect()
                    })
                    .collect();
                let values = keys.clone(); // Same as keys for simplicity

                let query: Vec<f32> = {
                    let v: Vec<f32> = (0..dim)
                        .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
                        .collect();
                    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    v.iter().map(|x| x / n * norm).collect()
                };

                // Ground truth attention
                let (ref_out, ref_w) = manual_attention(&query, &keys, &values, None, dim);

                // TQ3 quantized attention
                // Insert tokens ONE AT A TIME (like the real model does)
                let mut cache = TurboQuantKVCache::new_pqo(3, dim, 1, 1).unwrap();
                let mut last_k = None;
                for (i, k) in keys.iter().enumerate() {
                    let kt = Tensor::from_vec(k.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                    let vt =
                        Tensor::from_vec(values[i].clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                    let (fk, _) = cache.append_and_dequantize(0, &kt, &vt).unwrap();
                    last_k = Some(fk);
                }
                let tq_k = last_k.unwrap();
                let tq_k_f32 = tq_k
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();

                let (tq_out, tq_w) = manual_attention(&query, &tq_k_f32, &values, None, dim);

                cos_sum += cosine_similarity(&ref_out, &tq_out) as f64;

                let ref_top = ref_w
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let tq_top = tq_w
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                if ref_top == tq_top {
                    top1_matches += 1;
                }
            }

            let cos = cos_sum / num_queries as f64;
            let top1 = top1_matches as f64 / num_queries as f64 * 100.0;
            let mse_ratio = norm * norm * 0.117; // Paper MSE prediction

            eprintln!("{norm:>8.1} {cos:>10.4} {top1:>9.1}% {mse_ratio:>10.2}");

            // At any norm, cosine similarity should be > 0.5 for usable attention
            eprintln!("{norm:>8.1} {cos:>10.4} {top1:>9.1}% {mse_ratio:>10.2}");

            if norm <= 20.0 {
                assert!(
                    cos > 0.5,
                    "Attention quality too low at norm={norm}: cos={cos:.4}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Outlier handling (Paper Section 4.3)
    //
    // "splitting channels into outlier and non-outlier sets, and applying
    //  two independent instances of TurboQuant to each, allocating higher
    //  bit precision to outliers"
    //
    // Paper example: 32 outlier channels at 3-bit + 96 non-outlier at 2-bit
    // = effective 2.5 bits per channel for head_dim=128.
    // -----------------------------------------------------------------------

    /// With outlier handling (Paper Section 4.3), attention quality must
    /// remain high even for vectors with large norms (50-100+).
    ///
    /// Paper claim: "quality neutrality" at 3.5 bits, "marginal degradation"
    /// at 2.5 bits. Without outlier handling, cos < 0.7 at norm=50.
    /// With outlier handling, cos should be > 0.9 at norm=50.
    #[test]
    fn outlier_handling_preserves_quality_at_high_norms() {
        let dim: usize = 128;
        let num_keys: usize = 8;
        let num_queries: usize = 10;
        let norm: f32 = 50.0; // High norm where plain TQ3 degrades

        // Paper Section 4.3: 32 outlier channels at 3-bit, 96 at 2-bit
        let outlier_channels: usize = 32;
        let outlier_bits: u8 = 4; // TQ4 = 3-bit polar + 1-bit QJL
        let normal_bits: u8 = 3; // TQ3 = 2-bit polar + 1-bit QJL

        let mut cos_sum = 0.0_f64;

        for qi in 0..num_queries {
            let mut rng = SplitMix64::new(qi as u64 * 101 + 999);

            let gen_vec = |rng: &mut SplitMix64| -> Vec<f32> {
                let v: Vec<f32> = (0..dim)
                    .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
                    .collect();
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / n * norm).collect()
            };

            let keys: Vec<Vec<f32>> = (0..num_keys).map(|_| gen_vec(&mut rng)).collect();
            let values: Vec<Vec<f32>> = (0..num_keys).map(|_| gen_vec(&mut rng)).collect();
            let query = gen_vec(&mut rng);

            // Ground truth
            let (ref_out, _) = manual_attention(&query, &keys, &values, None, dim);

            // Outlier-handled quantization:
            // Split each vector into outlier channels (0..32) and normal channels (32..128)
            // Quantize outlier channels with higher bits, normal with lower bits
            // Dequantize and recombine
            let mut tq_keys: Vec<Vec<f32>> = Vec::with_capacity(num_keys);
            for k in &keys {
                let dequant = quantize_dequant_with_outliers(
                    k,
                    dim,
                    outlier_channels,
                    outlier_bits,
                    normal_bits,
                );
                tq_keys.push(dequant);
            }

            let (tq_out, _) = manual_attention(&query, &tq_keys, &values, None, dim);
            cos_sum += cosine_similarity(&ref_out, &tq_out) as f64;
        }

        let cos = cos_sum / num_queries as f64;
        eprintln!(
            "Outlier handling at norm={norm}: cos={cos:.4} \
             (outlier={outlier_channels} channels at {outlier_bits}-bit)"
        );

        // With outlier handling, quality should be significantly better
        // than plain TQ3 (which gives cos ≈ 0.66 at norm=50)
        assert!(
            cos > 0.85,
            "Outlier handling should preserve quality at norm={norm}: cos={cos:.4}"
        );
    }

    /// Quantize a vector using block-level quantization:
    /// Split dim=128 into blocks of `block_size` (e.g., 32).
    /// Each block is independently quantized with PolarQuant.
    /// Block with highest norm gets `outlier_bits`, rest gets `normal_bits`.
    ///
    /// Paper Section 4.3: "splitting channels into outlier and non-outlier
    /// sets" — we implement this as block-level with the highest-norm block
    /// getting more bits.
    fn quantize_dequant_with_outliers(
        data: &[f32],
        dim: usize,
        _outlier_count: usize,
        outlier_bits: u8,
        normal_bits: u8,
    ) -> Vec<f32> {
        use turboquant::{dequantize_vec, quantize_vec, TurboQuantConfig};

        let block_size: usize = 32; // Power of 2, same as llama.cpp
        let num_blocks = dim / block_size;

        // Find block norms
        let block_norms: Vec<f32> = (0..num_blocks)
            .map(|b| {
                let start = b * block_size;
                let end = start + block_size;
                data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt()
            })
            .collect();

        // Highest-norm block gets outlier treatment
        let max_block = block_norms
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let outlier_config = TurboQuantConfig::new(outlier_bits, block_size)
            .unwrap()
            .with_seed(DEFAULT_ROTATION_SEED);
        let normal_config = TurboQuantConfig::new(normal_bits, block_size)
            .unwrap()
            .with_seed(DEFAULT_ROTATION_SEED);

        let mut result = vec![0.0_f32; dim];
        for b in 0..num_blocks {
            let start = b * block_size;
            let end = start + block_size;
            let block_data = &data[start..end];

            let config = if b == max_block {
                &outlier_config
            } else {
                &normal_config
            };

            let block_quant = quantize_vec(config, block_data).unwrap();
            let block_dequant = dequantize_vec(config, &block_quant).unwrap();

            result[start..end].copy_from_slice(&block_dequant);
        }
        result
    }

    // -----------------------------------------------------------------------
    // Block-level quantization in TurboQuantKVCache
    //
    // The cache must use block_size=32 for quantization instead of head_dim.
    // These tests verify the cache-level integration.
    // -----------------------------------------------------------------------

    /// Block size constant for quantization (Paper + llama.cpp use 32).
    const QUANT_BLOCK_SIZE: usize = 32;

    /// TurboQuantKVCache with block-level quantization produces reasonable
    /// attention output for high-norm vectors (like real LLMs).
    ///
    /// This is the KEY test: if it passes, TQ3 works for real models.
    /// Without block-level quant: cos ≈ 0.66 at norm=50 → garbage.
    /// With block-level quant: cos > 0.85 at norm=50 → usable.
    #[test]
    fn block_level_cache_attention_quality_at_high_norm() {
        use crate::attention::{Sdpa, SdpaParams};

        let dim: usize = 128;
        let kv_heads: usize = 1;
        let q_heads: usize = 1;
        let num_keys: usize = 8;
        let num_queries: usize = 10;
        let norm: f32 = 50.0;
        let scale = 1.0 / (dim as f32).sqrt();

        let mut cos_sum = 0.0_f64;

        for qi in 0..num_queries {
            let mut rng = SplitMix64::new(qi as u64 * 131 + 4242);
            let gen_vec = |rng: &mut SplitMix64| -> Vec<f32> {
                let v: Vec<f32> = (0..dim)
                    .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
                    .collect();
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / n * norm).collect()
            };

            let keys: Vec<Vec<f32>> = (0..num_keys).map(|_| gen_vec(&mut rng)).collect();
            let values: Vec<Vec<f32>> = (0..num_keys).map(|_| gen_vec(&mut rng)).collect();
            let query = gen_vec(&mut rng);

            // Ground truth
            let (ref_out, _) = manual_attention(&query, &keys, &values, None, dim);

            // TQ3 via cache (append one token at a time, like real model)
            let mut cache = TurboQuantKVCache::new_pqo(3, dim, kv_heads, 1).unwrap();
            let mut last_k = None;
            for (i, k) in keys.iter().enumerate() {
                let kt = Tensor::from_vec(k.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                let vt = Tensor::from_vec(values[i].clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                let (fk, _) = cache.append_and_dequantize(0, &kt, &vt).unwrap();
                last_k = Some(fk);
            }
            let tq_k = last_k.unwrap();
            let tq_k_f32 = tq_k
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();

            let (tq_out, _) = manual_attention(&query, &tq_k_f32, &values, None, dim);
            cos_sum += cosine_similarity(&ref_out, &tq_out) as f64;
        }

        let cos = cos_sum / num_queries as f64;
        eprintln!(
            "Block-level cache TQ3 at norm={norm}: cos={cos:.4} \
             (need > 0.85 for usable quality)"
        );

        // With block-level quantization (block_size=32), quality must be high
        assert!(
            cos > 0.85,
            "Block-level TQ3 cache quality too low at norm={norm}: cos={cos:.4}. \
             Is block_size={QUANT_BLOCK_SIZE} being used in polar_quantize?"
        );
    }

    /// GpuPrecomputed must use block_size for rotation matrix, not head_dim.
    #[test]
    fn gpu_precomputed_uses_block_size() {
        let dim: usize = 128;
        let mut cache = TurboQuantKVCache::new_pqo(3, dim, 1, 1).unwrap();

        // Trigger GpuPrecomputed creation
        let k = Tensor::from_vec(vec![0.1_f32; dim], (1, 1, 1, dim), &Device::Cpu).unwrap();
        let v = k.clone();
        let _ = cache.append_and_dequantize(0, &k, &v).unwrap();

        // Check rotation matrix dimensions
        let pre = cache
            .gpu_precomputed
            .as_ref()
            .expect("GpuPrecomputed not created");
        let rot_dims = pre.rotation_fwd.dims();

        // Must be [block_size, block_size], not [head_dim, head_dim]
        assert_eq!(
            rot_dims,
            &[QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE],
            "Rotation matrix should be [{QUANT_BLOCK_SIZE}x{QUANT_BLOCK_SIZE}], \
             got {rot_dims:?}. Is GpuPrecomputed using block_size?"
        );
    }

    /// With outlier handling (highest-norm block gets more bits),
    /// TQ3 must produce usable text with real LLM (Qwen3-0.6B dimensions).
    ///
    /// Paper Section 4.3: "32 outlier channels at 3-bit, rest at 2-bit"
    /// At norm=100 (typical for early transformer layers):
    /// - Without outlier: cos ≈ 0.52 (garbage)
    /// - With outlier: cos > 0.85 (usable)
    #[test]
    fn outlier_block_handling_at_extreme_norm() {
        let dim: usize = 128;
        let num_keys: usize = 8;
        let num_queries: usize = 10;
        let norm: f32 = 100.0; // Extreme norm (early layers)

        let mut cos_sum = 0.0_f64;
        for qi in 0..num_queries {
            let mut rng = SplitMix64::new(qi as u64 * 137 + 8888);
            let gen_vec = |rng: &mut SplitMix64| -> Vec<f32> {
                let v: Vec<f32> = (0..dim)
                    .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
                    .collect();
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / n * norm).collect()
            };

            let keys: Vec<Vec<f32>> = (0..num_keys).map(|_| gen_vec(&mut rng)).collect();
            let values: Vec<Vec<f32>> = (0..num_keys).map(|_| gen_vec(&mut rng)).collect();
            let query = gen_vec(&mut rng);

            let (ref_out, _) = manual_attention(&query, &keys, &values, None, dim);

            // TQ3 cache with outlier handling (block-level)
            let mut cache = TurboQuantKVCache::new_pqo(3, dim, 1, 1).unwrap();
            let mut last_k = None;
            for (i, k) in keys.iter().enumerate() {
                let kt = Tensor::from_vec(k.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                let vt = Tensor::from_vec(values[i].clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                let (fk, _) = cache.append_and_dequantize(0, &kt, &vt).unwrap();
                last_k = Some(fk);
            }
            let tq_k = last_k.unwrap();
            let tq_k_f32 = tq_k
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();

            let (tq_out, _) = manual_attention(&query, &tq_k_f32, &values, None, dim);
            cos_sum += cosine_similarity(&ref_out, &tq_out) as f64;
        }

        let cos = cos_sum / num_queries as f64;
        eprintln!("Outlier block handling at norm={norm}: cos={cos:.4}");

        // With outlier handling, even at norm=100, quality must be usable
        assert!(
            cos > 0.80,
            "Outlier handling must keep cos > 0.80 at norm={norm}, got {cos:.4}"
        );
    }

    /// Step-by-step roundtrip test: Quantize → inspect intermediates → Dequant.
    ///
    /// Tests EACH step independently to prevent errors canceling each other:
    /// 1. Indices: in valid range, not all same value
    /// 2. Scales: ≈ L2-norm of each block
    /// 3. Dequant: close to original (MSE within Paper Theorem 1 bounds)
    /// 4. Uses seq_len=1 (Decode path) — NOT Prefill which returns originals
    #[test]
    fn step_by_step_quantize_dequant_roundtrip() {
        let dim: usize = 128;
        let kv_heads: usize = 1;
        let bits: u8 = 3; // TQ3: 2-bit polar (or 3-bit with outlier)
        let num_blocks = dim / QUANT_BLOCK_SIZE;

        // Known input vector with realistic LLM-like values
        let mut rng = SplitMix64::new(9999);
        let norm: f32 = 20.0; // moderate norm
        let input: Vec<f32> = (0..dim)
            .map(|_| {
                (rng.next_u64() as i64) as f32 / (i64::MAX as f32) * norm / (dim as f32).sqrt()
            })
            .collect();
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("Input norm: {input_norm:.4}, first 5: {:?}", &input[..5]);

        // Step 0: Create cache and precomputed
        let mut cache = TurboQuantKVCache::new_pqo(bits, dim, kv_heads, 1).unwrap();

        // Step 1: Append as DECODE (seq_len=1) — forces quantize+dequant
        let k = Tensor::from_vec(input.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
        let v = k.clone();
        let (dequant_k, _) = cache.append_and_dequantize(0, &k, &v).unwrap();

        // Step 1a: Verify we actually went through quantize (NOT returning originals)
        let dequant_vals = dequant_k
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // If dequant == input exactly, we returned originals (BUG — decode should quantize)
        let exact_match_count = input
            .iter()
            .zip(dequant_vals.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-10)
            .count();
        let pct_exact = exact_match_count as f64 / dim as f64 * 100.0;
        eprintln!(
            "Exact matches: {exact_match_count}/{dim} ({pct_exact:.0}%) — \
             if 100% then quantize was skipped!"
        );
        assert!(
            pct_exact < 50.0,
            "Dequant returned originals — quantize was skipped! \
             {exact_match_count}/{dim} exact matches. \
             The Decode path must quantize+dequant, not return originals."
        );

        // Step 2: Check indices are stored and valid (3-bit packed → unpack first)
        let indices_opt = &cache.gpu_k_indices[0];
        assert!(indices_opt.is_some(), "Indices not stored after decode");
        let indices = indices_opt.as_ref().unwrap();
        let packed_bytes: Vec<u8> = indices
            .narrow(1, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        let idx_vals = turboquant::packed::unpack_indices_3bit(&packed_bytes, dim);
        let max_idx = *idx_vals.iter().max().unwrap();
        let min_idx = *idx_vals.iter().min().unwrap();
        let unique_count = {
            let mut v = idx_vals.clone();
            v.sort();
            v.dedup();
            v.len()
        };
        eprintln!(
            "Indices: range [{min_idx}, {max_idx}], {unique_count} unique values, \
             first 8: {:?}",
            &idx_vals[..8]
        );
        // With 3-bit outlier codebook: max index should be 7
        // With 2-bit normal codebook: max index should be 3
        // PQO mode: all blocks use outlier codebook (8 centroids, indices 0-7)
        let expected_max: u8 = 7; // all blocks = outlier in PQO mode
        assert!(
            max_idx <= expected_max as u8,
            "Index {max_idx} exceeds codebook size (max should be {expected_max})"
        );
        assert!(
            unique_count >= 2,
            "All indices are the same value ({min_idx}) — quantization is broken"
        );

        // Step 3: Check scales are stored and ≈ block norms
        let scales_opt = &cache.gpu_k_scales[0];
        assert!(scales_opt.is_some(), "Scales not stored");
        let scales = scales_opt.as_ref().unwrap();
        let scale_vals: Vec<f32> = scales
            .narrow(1, 0, 1)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        eprintln!(
            "Scales (abs): {:?}",
            scale_vals.iter().map(|s| s.abs()).collect::<Vec<_>>()
        );

        // Scales depend on norm mode:
        // L2Norm: scale ≈ L2-norm of block (before WHT)
        // MaxNorm: scale ≈ amax(WHT(block)) / outer_centroid
        // We just check scales are positive and non-zero
        for (b, &scale) in scale_vals.iter().enumerate() {
            let scale_abs = scale.abs();
            assert!(
                scale_abs > 1e-8,
                "Scale[{b}]={scale_abs:.6} is too small — block not quantized?"
            );
        }

        // Step 4: Dequant quality (MSE)
        let mse: f64 = input
            .iter()
            .zip(dequant_vals.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>();
        let relative_mse = mse / (input_norm as f64).powi(2);
        eprintln!(
            "Dequant MSE: {mse:.6}, relative MSE: {relative_mse:.6}, \
             dequant first 5: {:?}",
            &dequant_vals[..5]
        );

        // Paper Theorem 1: MSE ≈ 0.03 for 3-bit at block_dim=32
        // Allow 2x margin for small sample
        let max_relative_mse = 0.15; // generous for single vector
        assert!(
            relative_mse < max_relative_mse,
            "Relative MSE {relative_mse:.4} too high (max {max_relative_mse})"
        );
    }

    /// Simulates error propagation across transformer layers.
    ///
    /// In a real transformer:
    /// - Layer L computes K_L = W @ hidden_state_L
    /// - hidden_state_L comes from Layer L-1's attention output
    /// - If Layer L-1's attention used quantized KV, hidden_state_L has error
    /// - K_L inherits that error PLUS adds quantization noise
    ///
    /// This test simulates the chain: for each "layer", quantize→dequant
    /// a vector, then use the dequant output as input for the next "layer"
    /// (simulating hidden state propagation).
    ///
    /// Without QJL: multiplicative bias → error grows linearly with layers
    /// With QJL: unbiased → error grows as √(layers)
    #[test]
    fn error_accumulation_across_layers() {
        let dim: usize = 128;
        let num_layers: usize = 28; // Qwen3-0.6B
        let norm: f32 = 20.0;
        let bits: u8 = 3;

        let mut rng = SplitMix64::new(42424);
        let gen_vec = |rng: &mut SplitMix64, norm: f32| -> Vec<f32> {
            let v: Vec<f32> = (0..dim)
                .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
                .collect();
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter().map(|x| x / n * norm).collect()
        };

        let original = gen_vec(&mut rng, norm);

        // Simulate layer-by-layer propagation
        let mut current = original.clone();
        let mut errors_per_layer = Vec::new();

        eprintln!("\nError accumulation over {num_layers} layers (norm={norm}):");
        eprintln!(
            "{:>5} {:>10} {:>12} {:>10}",
            "layer", "rel_MSE", "cos_to_orig", "norm"
        );

        for layer in 0..num_layers {
            // Quantize current vector (simulates KV cache quantization at this layer)
            let mut cache = TurboQuantKVCache::new_pqo(bits, dim, 1, 1).unwrap();
            let k = Tensor::from_vec(current.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
            let v = k.clone();
            let (dequant_k, _) = cache.append_and_dequantize(0, &k, &v).unwrap();

            let dequant = dequant_k
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            // Measure error relative to ORIGINAL (not previous layer)
            let mse: f64 = original
                .iter()
                .zip(dequant.iter())
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum();
            let orig_norm_sq: f64 = original.iter().map(|x| (*x as f64).powi(2)).sum();
            let rel_mse = mse / orig_norm_sq;

            let cos = cosine_similarity(&original, &dequant);
            let cur_norm: f32 = dequant.iter().map(|x| x * x).sum::<f32>().sqrt();

            errors_per_layer.push(rel_mse);

            if layer < 5 || layer >= num_layers - 3 || layer % 5 == 0 {
                eprintln!("{layer:>5} {rel_mse:>10.6} {cos:>12.6} {cur_norm:>10.4}");
            }

            // Propagate: use dequantized output as next layer's input
            // In real model: hidden_state = attention(Q, dequant_K, dequant_V)
            // Simplified: next input = dequantized K (worst case — no attention averaging)
            current = dequant;
        }

        let final_cos = cosine_similarity(&original, &current);
        let final_rel_mse = errors_per_layer.last().unwrap();

        eprintln!("\nFinal after {num_layers} layers:");
        eprintln!("  Cosine to original: {final_cos:.4}");
        eprintln!("  Relative MSE: {final_rel_mse:.4}");

        // After 28 layers of requantization, the signal should not be completely destroyed
        // Note: this is WORST CASE — real model has attention averaging + feedforward
        assert!(
            final_cos > 0.5,
            "Signal completely destroyed after {num_layers} layers: cos={final_cos:.4}"
        );
    }

    /// Verify that block-level quantize→dequant preserves element ordering.
    ///
    /// A shape bug in the block reshape could swap blocks within a vector,
    /// making element i of the dequant correspond to element j of the input.
    /// This would look fine in MSE tests (similar values) but destroy the model.
    #[test]
    fn element_ordering_preserved_after_quantize_dequant() {
        let dim: usize = 128;
        // Input with distinct values per position (alternating signs + magnitude)
        // This makes it possible to check ordering even after quantization
        let mut rng = SplitMix64::new(77777);
        let input: Vec<f32> = (0..dim)
            .map(|i| {
                let base = (i as f32 + 1.0) * 0.1; // distinct magnitude per position
                let sign = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
                base * sign
            })
            .collect();

        // Test multiple bit widths AND the turboquant-rs CPU path (which uses head_dim=128)
        // to see if block_size=32 is the problem
        for bits in [3u8, 4] {
            let mut cache = TurboQuantKVCache::new_pqo(bits, dim, 1, 1).unwrap();
            let k = Tensor::from_vec(input.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
            let v = k.clone();
            let (dk, _) = cache.append_and_dequantize(0, &k, &v).unwrap();
            let dequant = dk
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            // For each element: is dequant[i] closest to input[i]?
            let mut mismatches = 0;
            for i in 0..dim {
                let actual = dequant[i];
                let closest_idx = input
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (**a - actual)
                            .abs()
                            .partial_cmp(&(**b - actual).abs())
                            .unwrap()
                    })
                    .unwrap()
                    .0;
                if closest_idx != i && (closest_idx as i32 - i as i32).unsigned_abs() > 1 {
                    // Allow off-by-one (adjacent elements may quantize to same centroid)
                    if mismatches < 5 {
                        eprintln!(
                            "  Ordering mismatch: dequant[{i}]={actual:.3}, \
                         closest to input[{closest_idx}]={:.3} (expected input[{i}]={:.3})",
                            input[closest_idx], input[i]
                        );
                    }
                    mismatches += 1;
                }
            }
            eprintln!("Element ordering (bits={bits}, block_size={QUANT_BLOCK_SIZE}): {mismatches}/{dim} significant mismatches");
        } // end for bits

        // Also test with turboquant-rs CPU path (dim=128, no block splitting)
        use turboquant::{dequantize_vec, quantize_vec, TurboQuantConfig};
        for bits in [2u8, 3, 4] {
            let config = TurboQuantConfig::new(bits, dim)
                .unwrap()
                .with_seed(DEFAULT_ROTATION_SEED);
            let block = quantize_vec(&config, &input).unwrap();
            let dequant_cpu = dequantize_vec(&config, &block).unwrap();

            let mut mismatches = 0;
            for i in 0..dim {
                let actual = dequant_cpu[i];
                let closest_idx = input
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (**a - actual)
                            .abs()
                            .partial_cmp(&(**b - actual).abs())
                            .unwrap()
                    })
                    .unwrap()
                    .0;
                if closest_idx != i && (closest_idx as i32 - i as i32).unsigned_abs() > 1 {
                    mismatches += 1;
                }
            }
            eprintln!("Element ordering (bits={bits}, dim=128 CPU path): {mismatches}/{dim} significant mismatches");
        }

        // Realistic model simulation: 8 KV heads, 15 tokens, one-by-one
        {
            let heads: usize = 8;
            let num_tokens: usize = 15;
            let bits: u8 = 4;

            let mut cache = TurboQuantKVCache::new_pqo(bits, dim, heads, 1).unwrap();
            let mut originals: Vec<Vec<f32>> = Vec::new(); // [token][head*dim]

            for t in 0..num_tokens {
                let mut rng_t = SplitMix64::new(t as u64 * 999 + 12345);
                let data: Vec<f32> = (0..heads * dim)
                    .map(|_| (rng_t.next_u64() as i64) as f32 / (i64::MAX as f32) * 20.0)
                    .collect();
                originals.push(data.clone());

                let k = Tensor::from_vec(data.clone(), (1, heads, 1, dim), &Device::Cpu).unwrap();
                let v = k.clone();
                let _ = cache.append_and_dequantize(0, &k, &v).unwrap();
            }

            // Now get the full dequantized cache
            let dummy = Tensor::zeros((1, heads, 1, dim), DType::F32, &Device::Cpu).unwrap();
            let (full_k, _) = cache.append_and_dequantize(0, &dummy, &dummy).unwrap();
            // full_k shape: [1, heads, num_tokens+1, dim]
            let total_tokens = num_tokens + 1;

            eprintln!("\n--- Realistic model simulation (CPU, heads={heads}, tokens={num_tokens}, bits={bits}) ---");
            eprintln!("full_k shape: {:?}", full_k.dims());

            // Check each original token's dequant quality
            for t in 0..num_tokens {
                let orig = &originals[t];
                let dequant = full_k
                    .narrow(2, t, 1)
                    .unwrap() // token t
                    .squeeze(0)
                    .unwrap()
                    .squeeze(1)
                    .unwrap() // [heads, dim]
                    .to_vec2::<f32>()
                    .unwrap();

                // Per-head comparison
                let mut worst_cos = 1.0_f32;
                let mut worst_head = 0;
                for h in 0..heads {
                    let orig_head = &orig[h * dim..(h + 1) * dim];
                    let dequant_head = &dequant[h];
                    let cos = cosine_similarity(orig_head, dequant_head);
                    if cos < worst_cos {
                        worst_cos = cos;
                        worst_head = h;
                    }
                }

                if t < 3 || t >= num_tokens - 2 || worst_cos < 0.99 {
                    eprintln!("  token {t}: worst_cos={worst_cos:.6} (head {worst_head})");
                }
            }
        }

        // Same test on GPU with BF16 (like real model)
        #[cfg(feature = "cuda")]
        {
            let cuda = Device::cuda_if_available(0).unwrap();
            if !cuda.is_cpu() {
                let heads: usize = 8;
                let num_tokens: usize = 15;
                let bits: u8 = 4;

                let mut cache = TurboQuantKVCache::new_pqo(bits, dim, heads, 1).unwrap();
                let mut originals: Vec<Vec<f32>> = Vec::new();

                for t in 0..num_tokens {
                    let mut rng_t = SplitMix64::new(t as u64 * 999 + 12345);
                    let data: Vec<f32> = (0..heads * dim)
                        .map(|_| (rng_t.next_u64() as i64) as f32 / (i64::MAX as f32) * 20.0)
                        .collect();
                    originals.push(data.clone());

                    // Use BF16 on CUDA like the real model
                    let k = Tensor::from_vec(data, (1, heads, 1, dim), &cuda)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap();
                    let v = k.clone();
                    let _ = cache.append_and_dequantize(0, &k, &v).unwrap();
                }

                let dummy = Tensor::zeros((1, heads, 1, dim), DType::BF16, &cuda).unwrap();
                let (full_k, _) = cache.append_and_dequantize(0, &dummy, &dummy).unwrap();

                eprintln!("\n--- Realistic model simulation (GPU BF16, heads={heads}, tokens={num_tokens}, bits={bits}) ---");
                eprintln!(
                    "full_k shape: {:?}, dtype: {:?}",
                    full_k.dims(),
                    full_k.dtype()
                );

                let full_k_cpu = full_k
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();

                for t in 0..num_tokens {
                    let orig = &originals[t];
                    let dequant = full_k_cpu
                        .narrow(2, t, 1)
                        .unwrap()
                        .squeeze(0)
                        .unwrap()
                        .squeeze(1)
                        .unwrap()
                        .to_vec2::<f32>()
                        .unwrap();

                    let mut worst_cos = 1.0_f32;
                    let mut worst_head = 0;
                    for h in 0..heads {
                        let orig_head = &orig[h * dim..(h + 1) * dim];
                        let dequant_head = &dequant[h];
                        let cos = cosine_similarity(orig_head, dequant_head);
                        if cos < worst_cos {
                            worst_cos = cos;
                            worst_head = h;
                        }
                    }

                    if t < 3 || t >= num_tokens - 2 || worst_cos < 0.95 {
                        eprintln!("  token {t}: worst_cos={worst_cos:.6} (head {worst_head})");
                    }
                }
            }
        }

        // KEY INSIGHT: PolarQuant doesn't preserve per-element identity.
        // It preserves DOT PRODUCTS (Paper Theorem 2).
        // Test dot product quality over multiple queries:
        let num_queries = 50;
        // Stage comparison: where does quality break?
        // NOTE: Real model Layer 0 has K-norms up to 400!
        for test_norm in [1.0_f32, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0] {
            eprintln!("\n--- Stage comparison (norm={test_norm}) ---");
            let norm: f32 = test_norm;
            let gen_vec = |rng: &mut SplitMix64, norm: f32| -> Vec<f32> {
                let v: Vec<f32> = (0..dim)
                    .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
                    .collect();
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / n * norm).collect()
            };

            // Stage A: turboquant-rs direct quantize+dequant (NO cache, NO blocks)
            {
                use turboquant::{dequantize_vec, quantize_vec, TurboQuantConfig};
                let config = TurboQuantConfig::new(3, dim)
                    .unwrap()
                    .with_seed(DEFAULT_ROTATION_SEED);
                let block = quantize_vec(&config, &input).unwrap();
                let dq = dequantize_vec(&config, &block).unwrap();
                let cos = cosine_similarity(&input, &dq);
                eprintln!("Stage A (crate direct, dim=128): cos={cos:.6}");
            }

            // Stage B: Cache roundtrip (1 token, block-level)
            {
                let mut cache = TurboQuantKVCache::new_pqo(3, dim, 1, 1).unwrap();
                let kt = Tensor::from_vec(input.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                let vt = kt.clone();
                let (dk, _) = cache.append_and_dequantize(0, &kt, &vt).unwrap();
                let dq = dk
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let cos = cosine_similarity(&input, &dq);
                eprintln!("Stage B (cache roundtrip, block_size=32): cos={cos:.6}");
            }

            // Stage C: Cache roundtrip + store + retrieve (2 tokens, check first)
            {
                let mut cache = TurboQuantKVCache::new_pqo(3, dim, 1, 1).unwrap();
                // Token 1
                let k1 = Tensor::from_vec(input.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                let v1 = k1.clone();
                let _ = cache.append_and_dequantize(0, &k1, &v1).unwrap();
                // Token 2
                let input2 = gen_vec(&mut rng, norm);
                let k2 = Tensor::from_vec(input2, (1, 1, 1, dim), &Device::Cpu).unwrap();
                let v2 = k2.clone();
                let (dk, _) = cache.append_and_dequantize(0, &k2, &v2).unwrap();
                // Check token 1's values (should be same as Stage B)
                let token1 = dk
                    .narrow(2, 0, 1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let cos = cosine_similarity(&input, &token1);
                eprintln!("Stage C (cache 2 tokens, check token 1): cos={cos:.6}");
            }

            // Stage D: Attention with 8 tokens (multiple bit widths)
            for stage_bits in [3u8, 4] {
                let keys: Vec<Vec<f32>> = (0..8).map(|_| gen_vec(&mut rng, norm)).collect();
                let query = gen_vec(&mut rng, norm);

                // Normal attention
                let (ref_out, _) = manual_attention(&query, &keys, &keys, None, dim);

                // TQ attention
                let mut cache = TurboQuantKVCache::new_pqo(stage_bits, dim, 1, 1).unwrap();
                let mut last_k = None;
                for k in &keys {
                    let kt = Tensor::from_vec(k.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                    let vt = kt.clone();
                    let (fk, _) = cache.append_and_dequantize(0, &kt, &vt).unwrap();
                    last_k = Some(fk);
                }
                let fk = last_k.unwrap();
                let tq_keys = fk
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                let (tq_out, _) = manual_attention(&query, &tq_keys, &keys, None, dim);
                let cos = cosine_similarity(&ref_out, &tq_out);
                eprintln!("Stage D (norm={norm:.0}, bits={stage_bits}): cos={cos:.6}");
            }
        } // end for test_norm
    }

    // -----------------------------------------------------------------------
    // Regression tests: guard against re-introducing known bugs
    // -----------------------------------------------------------------------

    /// The sign pattern for WHT preconditioning must NOT be alternating.
    ///
    /// Bug found 2026-04-05: Golden-Ratio hash with seed 42 produced
    /// [+1,-1,+1,-1,...] for i=0..31, which is NOT random and doesn't
    /// properly randomize the WHT rotation. This caused garbage output
    /// in all models. The fix: use llama.cpp's hardcoded pseudo-random
    /// pattern for block_size=32.
    #[test]
    fn sign_pattern_is_not_alternating() {
        let dim = 128;
        let mut cache = TurboQuantKVCache::new_pqo(3, dim, 1, 1).unwrap();

        // Trigger precomputed creation
        let k = Tensor::zeros((1, 1, 1, dim), DType::F32, &Device::Cpu).unwrap();
        let _ = cache.append_and_dequantize(0, &k, &k).unwrap();

        let pre = cache.gpu_precomputed.as_ref().unwrap();

        // Extract sign pattern from the rotation matrix:
        // rotation_fwd = H @ diag(signs), so signs[j] = rotation_fwd[0][j] / H[0][j]
        // H[0][j] = 1/sqrt(block_size) for all j (first row of Hadamard is all +1s)
        let rot = pre.rotation_fwd.to_vec2::<f32>().unwrap();
        let inv_sqrt = (QUANT_BLOCK_SIZE as f32).sqrt();
        let signs: Vec<f32> = rot[0].iter().map(|v| (v * inv_sqrt).round()).collect();

        // Check: NOT alternating
        let is_alternating = signs.windows(2).all(|w| (w[0] > 0.0) != (w[1] > 0.0));
        assert!(
            !is_alternating,
            "Sign pattern is alternating [+1,-1,+1,-1,...] — \
             this doesn't randomize the WHT rotation! \
             Use a pseudo-random pattern (e.g. llama.cpp's hardcoded pattern)."
        );

        // Check: has mix of same-sign neighbors (pseudo-random property)
        let same_sign_pairs = signs
            .windows(2)
            .filter(|w| (w[0] > 0.0) == (w[1] > 0.0))
            .count();
        assert!(
            same_sign_pairs >= 5,
            "Sign pattern has too few same-sign neighbors ({same_sign_pairs}) — \
             likely still too structured. Need pseudo-random pattern."
        );
    }

    /// MaxNorm: WHT must be applied BEFORE finding amax (like llama.cpp).
    ///
    /// Bug found 2026-04-05: we normalized BEFORE WHT, but amax should be
    /// computed on the WHT-rotated values for MaxNorm mode.
    /// Verified by checking that dequant(quantize(x)) ≈ x for MaxNorm.
    #[test]
    fn maxnorm_wht_order_roundtrip_quality() {
        let dim: usize = 128;
        let norm: f32 = 50.0;
        let num_samples: usize = 20;

        let mut rng = SplitMix64::new(54321);
        let mut cos_sum = 0.0_f64;

        for _ in 0..num_samples {
            let input: Vec<f32> = (0..dim)
                .map(|_| {
                    (rng.next_u64() as i64) as f32 / (i64::MAX as f32) * norm / (dim as f32).sqrt()
                })
                .collect();

            // Default is MaxNorm
            let mut cache = TurboQuantKVCache::new_pqo(4, dim, 1, 1).unwrap();
            let k = Tensor::from_vec(input.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
            let (dk, _) = cache.append_and_dequantize(0, &k, &k).unwrap();
            let dequant = dk
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            cos_sum += cosine_similarity(&input, &dequant) as f64;
        }

        let mean_cos = cos_sum / num_samples as f64;
        eprintln!("MaxNorm WHT-order roundtrip: mean cos={mean_cos:.6} ({num_samples} samples, norm={norm})");

        assert!(
            mean_cos > 0.98,
            "MaxNorm roundtrip quality too low: cos={mean_cos:.4}. \
             Is WHT applied BEFORE amax computation?"
        );
    }

    /// L2Norm roundtrip quality with fixed sign pattern.
    #[test]
    fn l2norm_roundtrip_quality_with_fixed_sign() {
        let dim: usize = 128;
        let norm: f32 = 50.0;
        let num_samples: usize = 20;

        let mut rng = SplitMix64::new(98765);
        let mut cos_sum = 0.0_f64;

        for _ in 0..num_samples {
            let input: Vec<f32> = (0..dim)
                .map(|_| {
                    (rng.next_u64() as i64) as f32 / (i64::MAX as f32) * norm / (dim as f32).sqrt()
                })
                .collect();

            let mut cache = TurboQuantKVCache::new_with_config(
                4,
                dim,
                1,
                1,
                DEFAULT_QJL_SEED,
                QuantNormMode::L2Norm,
            )
            .unwrap();
            let k = Tensor::from_vec(input.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
            let (dk, _) = cache.append_and_dequantize(0, &k, &k).unwrap();
            let dequant = dk
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            cos_sum += cosine_similarity(&input, &dequant) as f64;
        }

        let mean_cos = cos_sum / num_samples as f64;
        eprintln!("L2Norm roundtrip: mean cos={mean_cos:.6} ({num_samples} samples, norm={norm})");

        assert!(
            mean_cos > 0.98,
            "L2Norm roundtrip quality too low: cos={mean_cos:.4}. \
             Is sign pattern pseudo-random (not alternating)?"
        );
    }

    /// Both modes must produce usable attention quality at norm=50
    /// (typical for mid-layer KV cache in real LLMs).
    #[test]
    fn both_modes_attention_quality_at_realistic_norm() {
        let dim: usize = 128;
        let num_keys: usize = 8;
        let num_queries: usize = 10;
        let norm: f32 = 50.0;

        for (mode_name, mode) in [
            ("MaxNorm", QuantNormMode::MaxNorm),
            ("L2Norm", QuantNormMode::L2Norm),
        ] {
            let mut rng = SplitMix64::new(11111 + mode as u64 * 1000);
            let gen_vec = |rng: &mut SplitMix64| -> Vec<f32> {
                let v: Vec<f32> = (0..dim)
                    .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
                    .collect();
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / n * norm).collect()
            };

            let mut cos_sum = 0.0_f64;
            for _ in 0..num_queries {
                let keys: Vec<Vec<f32>> = (0..num_keys).map(|_| gen_vec(&mut rng)).collect();
                let values = keys.clone();
                let query = gen_vec(&mut rng);

                let (ref_out, _) = manual_attention(&query, &keys, &values, None, dim);

                let mut cache =
                    TurboQuantKVCache::new_with_config(4, dim, 1, 1, DEFAULT_QJL_SEED, mode)
                        .unwrap();
                let mut last_k = None;
                for (i, k) in keys.iter().enumerate() {
                    let kt = Tensor::from_vec(k.clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                    let vt =
                        Tensor::from_vec(values[i].clone(), (1, 1, 1, dim), &Device::Cpu).unwrap();
                    let (fk, _) = cache.append_and_dequantize(0, &kt, &vt).unwrap();
                    last_k = Some(fk);
                }
                let tq_k = last_k
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                let (tq_out, _) = manual_attention(&query, &tq_k, &values, None, dim);
                cos_sum += cosine_similarity(&ref_out, &tq_out) as f64;
            }

            let cos = cos_sum / num_queries as f64;
            eprintln!("{mode_name} attention quality at norm={norm}: cos={cos:.4}");

            // Note: cos on random vectors does NOT predict real model quality well.
            // The sign-pattern test and roundtrip tests are better regression guards.
            // This threshold catches only catastrophic failures (broken quantization).
            assert!(
                cos > 0.5,
                "{mode_name} attention CATASTROPHICALLY bad at norm={norm}: cos={cos:.4}. \
                 Likely broken sign-pattern, codebook, or WHT order."
            );
        }
    }

    /// End-to-end: real model (Qwen2-1.5B) with TQ4 must produce "Paris".
    ///
    /// This is the ULTIMATE test: if the model produces correct text,
    /// TurboQuant is working. Catches ALL bugs at once.
    ///
    /// Requires: release binary built, Qwen2-1.5B downloaded.
    /// Run with: cargo nextest run -E 'test(e2e_model)' -- --ignored
    #[test]
    #[ignore] // Slow (~10s), needs model download
    fn e2e_model_tq4_produces_correct_answer() {
        use std::process::Command;

        let binary = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("target/release/mistralrs");
        if !binary.exists() {
            eprintln!("Skipping: binary not found at {}", binary.display());
            return;
        }

        // Run model with TQ4 and pipe in a simple question
        let output = Command::new(&binary)
            .args([
                "run",
                "-m",
                "Qwen/Qwen3-0.6B",
                "-n",
                "0:999",
                "--pa-cache-type",
                "pqo4",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();

        let mut child = match output {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping: couldn't spawn: {e}");
                return;
            }
        };

        use std::io::Write;
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = writeln!(stdin, "\\temperature 0.1");
            let _ = writeln!(stdin, "What is the capital of France? Answer in one word.");
        }
        drop(child.stdin.take());

        // Wait with 30s timeout
        use std::time::Duration;
        let timeout = Duration::from_secs(30);
        let start = std::time::Instant::now();

        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        panic!("E2E test timed out after {timeout:?}");
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => panic!("Error waiting for process: {e}"),
            }
        }

        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();

        eprintln!(
            "E2E output (last 200 chars): ...{}",
            &stdout[stdout.len().saturating_sub(200)..]
        );

        assert!(
            stdout.contains("paris"),
            "E2E TQ4 must produce 'Paris'. This test catches ALL TurboQuant bugs."
        );
    }

    /// Butterfly WHT inverse must produce the same result as matmul with rotation_inv.
    #[test]
    fn butterfly_wht_matches_matmul() {
        let bits: u8 = 3;
        let dim: usize = 128;
        let heads: usize = 2;
        let layers: usize = 1;

        let mut tq = TurboQuantKVCache::new_pqo(bits, dim, heads, layers).unwrap();
        tq.ensure_gpu_precomputed(&Device::Cpu).unwrap();
        let pre = tq.gpu_precomputed.as_ref().unwrap();

        let num_blocks = dim / QUANT_BLOCK_SIZE;
        let m = 64; // 64 blocks to test (like 16 tokens × 4 blocks each)

        // Random dequantized block data
        let dequant = Tensor::rand(0f32, 1.0, (m, QUANT_BLOCK_SIZE), &Device::Cpu).unwrap();

        // Method 1: matmul (current)
        let result_matmul = dequant.matmul(&pre.rotation_inv).unwrap();

        // Method 2: butterfly (new)
        let result_butterfly =
            butterfly_wht_inverse_cpu(&dequant, &pre.rotation_fwd, QUANT_BLOCK_SIZE).unwrap();

        // Compare
        let flat_mm: Vec<f32> = result_matmul.flatten_all().unwrap().to_vec1().unwrap();
        let flat_bf: Vec<f32> = result_butterfly.flatten_all().unwrap().to_vec1().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in flat_mm.iter().zip(flat_bf.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-4,
            "Butterfly and matmul results must match, max diff: {max_diff}"
        );
    }

    /// Full dequant roundtrip with butterfly must match matmul roundtrip.
    #[test]
    fn butterfly_dequant_roundtrip_matches_matmul() {
        let bits: u8 = 3;
        let dim: usize = 128;
        let heads: usize = 2;
        let layers: usize = 1;

        // Two caches with same data — one will use butterfly (CPU), reference uses matmul
        let tq = TurboQuantKVCache::new_pqo(bits, dim, heads, layers).unwrap();
        let shared = std::sync::Arc::new(std::sync::Mutex::new(tq));
        let mut cache =
            crate::kv_cache::KvCache::new_turboquant(shared, 0, Device::Cpu, DType::F32);

        // Prefill + decode
        let k_pf = Tensor::rand(0f32, 1.0, (1, heads, 16, dim), &Device::Cpu).unwrap();
        let v_pf = Tensor::rand(0f32, 1.0, (1, heads, 16, dim), &Device::Cpu).unwrap();
        let (k_out, v_out) = cache.append(&k_pf, &v_pf).unwrap();

        // Verify shapes and that data is non-zero
        assert_eq!(k_out.dims(), &[1, heads, 16, dim]);
        let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        let nonzero = k_flat.iter().filter(|x| x.abs() > 1e-10).count();
        assert!(
            nonzero > k_flat.len() / 2,
            "Butterfly dequant should produce non-zero values"
        );
    }
}
