//! TurboQuant KV-Cache — multi-head quantized storage for keys and values.
//!
//! This module provides [`TurboQuantKVCache`] which stores one
//! [`QuantizedKVCache`] per KV head, allowing transparent quantize-on-write
//! and dequantize-on-read integration with the standard mistral.rs KvCache.
//!
//! When `KvCache::TurboQuant` is selected, key/value vectors are quantized
//! per head with TurboQuant (PolarQuant + QJL bias correction) and stored
//! here. On read, only newly appended tokens are dequantized and concatenated
//! with the cached dequantized history (delta dequantization), avoiding O(N²)
//! overhead during autoregressive decoding.

use candle_core::{DType, Device, Result, Storage, Tensor};
use rayon::prelude::*;
use turboquant::attention::QuantizedKVCache;
use turboquant::packed::TurboQuantConfig;

// ---------------------------------------------------------------------------
// Named constants (no magic numbers)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Zero-copy tensor data extraction
// ---------------------------------------------------------------------------

/// Extracts f32 data from a Candle tensor into `buf`, using zero-copy when possible.
///
/// If the tensor is CPU + F32 + contiguous, reads directly from Candle's internal
/// storage (no allocation, no copy). Otherwise falls back to to_vec1 (one copy).
fn extract_f32_data(tensor: &Tensor, buf: &mut Vec<f32>) -> Result<()> {
    let tensor = tensor.contiguous()?;
    if tensor.device().is_cpu() && tensor.dtype() == DType::F32 {
        // Zero-copy path: read directly from Candle storage
        let (storage, layout) = tensor.storage_and_layout();
        if let Storage::Cpu(cpu_storage) = &*storage {
            let data = cpu_storage.as_slice::<f32>()?;
            let offset = layout.start_offset();
            let len = tensor.elem_count();
            buf.extend_from_slice(&data[offset..offset + len]);
            return Ok(());
        }
    }
    // Fallback: convert to CPU + F32 and copy
    let cpu_f32 = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    buf.extend_from_slice(&cpu_f32.flatten_all()?.to_vec1::<f32>()?);
    Ok(())
}
const DEFAULT_ROTATION_SEED: u64 = 42;

/// Default seed for the QJL Rademacher matrix.
const DEFAULT_QJL_SEED: u64 = 12345;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

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

/// Multi-head TurboQuant KV cache.
///
/// Stores one [`QuantizedKVCache`] per KV head so that each head is
/// quantized and dequantized independently. This matches the per-head
/// nature of attention and allows transparent integration via `KvCache::TurboQuant`.
///
/// Delta dequantization: previously dequantized data is cached as flat f32 buffers
/// per layer. On each `append_and_dequantize` call only the *new* tokens are
/// dequantized and appended to the buffer (amortized O(1) via Vec growth).
/// The Tensor is only created once at the end from the full buffer.
/// Initial capacity for the pre-allocated GPU tensor buffer (in tokens).
/// Grows automatically when exceeded.
const INITIAL_BUFFER_CAPACITY: usize = 2048;

pub struct TurboQuantKVCache {
    caches: Vec<QuantizedKVCache>,
    bits: u8,
    head_dim: usize,
    num_kv_heads: usize,
    num_layers: usize,
    /// Pre-allocated key tensor buffers on target device: [layer].
    /// Shape: [1, num_kv_heads, buffer_capacity, head_dim].
    /// New tokens are written via slice_set; reads use narrow().
    gpu_k_buf: Vec<Option<Tensor>>,
    /// Pre-allocated value tensor buffers on target device.
    gpu_v_buf: Vec<Option<Tensor>>,
    /// Current sequence length per layer (how much of the buffer is filled).
    buf_seq_len: Vec<usize>,
    /// Current buffer capacity (in tokens) per layer.
    buf_capacity: Vec<usize>,
    /// Reusable buffer for flattened key data (avoids per-call allocation).
    k_data_buf: Vec<f32>,
    /// Reusable buffer for flattened value data (avoids per-call allocation).
    v_data_buf: Vec<f32>,
    /// Pending raw key vectors per layer per head that haven't been quantized yet.
    /// Layout: `pending_k[layer][head]` = `Vec` of `Vec<f32>` (one per position).
    /// Populated during prefill; flushed (batch-quantized) before the first decode step.
    pending_k: Vec<Vec<Vec<Vec<f32>>>>,
    /// Pending raw value vectors (same layout as `pending_k`).
    pending_v: Vec<Vec<Vec<Vec<f32>>>>,
}

impl TurboQuantKVCache {
    /// Creates a new multi-head TurboQuant KV cache.
    ///
    /// # Arguments
    ///
    /// * `bits` - Total bit budget (3 or 4)
    /// * `head_dim` - Dimension per attention head (must be power of two)
    /// * `num_kv_heads` - Number of key-value heads
    /// * `num_layers` - Number of transformer layers
    pub fn new(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
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
            gpu_k_buf: vec![None; num_layers],
            gpu_v_buf: vec![None; num_layers],
            buf_seq_len: vec![0; num_layers],
            buf_capacity: vec![0; num_layers],
            k_data_buf: Vec::new(),
            v_data_buf: Vec::new(),
            pending_k: (0..num_layers)
                .map(|_| vec![Vec::new(); num_kv_heads])
                .collect(),
            pending_v: (0..num_layers)
                .map(|_| vec![Vec::new(); num_kv_heads])
                .collect(),
        })
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

    /// Batch-quantizes all pending (prefill) vectors for the given layer.
    ///
    /// During prefill, raw key/value vectors are buffered in `pending_k`/`pending_v`
    /// without quantization. This method flushes them into the quantized caches
    /// using `push_batch` with rayon parallelism across heads. It is called
    /// automatically before the first decode step for each layer.
    ///
    /// No-op if there are no pending vectors for this layer.
    fn flush_pending(&mut self, layer: usize) -> Result<()> {
        // Early exit: nothing to flush.
        if !self.has_pending(layer) {
            return Ok(());
        }

        // Split borrows: immutable refs to pending buffers, mutable ref to caches.
        let pending_k = &self.pending_k[layer];
        let pending_v = &self.pending_v[layer];
        let caches = &mut self.caches;

        caches
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(head, cache)| {
                let keys: Vec<&[f32]> =
                    pending_k[head].iter().map(|v| v.as_slice()).collect();
                let vals: Vec<&[f32]> =
                    pending_v[head].iter().map(|v| v.as_slice()).collect();
                if !keys.is_empty() {
                    cache.push_batch(layer, &keys, &vals).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "TurboQuant: flush_pending error: {e}"
                        ))
                    })?;
                }
                Ok::<(), candle_core::Error>(())
            })?;

        // Clear pending buffers (keep allocated capacity for potential reuse).
        for head_bufs in &mut self.pending_k[layer] {
            head_bufs.clear();
        }
        for head_bufs in &mut self.pending_v[layer] {
            head_bufs.clear();
        }
        Ok(())
    }

    /// Returns `true` if there are pending (not yet quantized) vectors for the
    /// given layer. Used to decide whether `flush_pending` must run.
    pub fn has_pending(&self, layer: usize) -> bool {
        self.pending_k[layer].iter().any(|h| !h.is_empty())
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
        let orig_device = k.device().clone();
        let orig_dtype = k.dtype();

        let dims = k.dims4()?;
        let batch_size = dims.0;
        let num_heads = dims.1;
        let new_seq_len = dims.2;
        let head_dim = dims.3;

        debug_assert_eq!(num_heads, self.num_kv_heads);
        debug_assert_eq!(head_dim, self.head_dim);

        // Extract f32 data from tensors into reusable buffers.
        // Zero-copy path: if tensor is already CPU+F32+contiguous, read directly
        // from Candle's internal storage without copying.
        self.k_data_buf.clear();
        self.v_data_buf.clear();
        extract_f32_data(k, &mut self.k_data_buf)?;
        extract_f32_data(v, &mut self.v_data_buf)?;

        // Lazy quantization: during prefill (new_seq_len > 1), store raw vectors
        // in pending buffers WITHOUT quantizing. The quantization is deferred to
        // `flush_pending()` which runs before the first decode step.
        // During decode (new_seq_len == 1), quantize immediately (single token).

        if new_seq_len > 1 {
            // Prefill: store raw vectors in pending buffers — skip quantization.
            // Split borrows: take mutable refs to pending_k/pending_v and immutable
            // refs to k_data_buf/v_data_buf simultaneously.
            let k_buf = &self.k_data_buf;
            let v_buf = &self.v_data_buf;
            let pending_k_layer = &mut self.pending_k[layer];
            let pending_v_layer = &mut self.pending_v[layer];
            for b in 0..batch_size {
                for head in 0..num_heads {
                    for pos in 0..new_seq_len {
                        let offset = ((b * num_heads + head) * new_seq_len + pos) * head_dim;
                        pending_k_layer[head].push(k_buf[offset..offset + head_dim].to_vec());
                        pending_v_layer[head].push(v_buf[offset..offset + head_dim].to_vec());
                    }
                }
            }
        } else {
            // Decode: flush any pending prefill vectors before the first decode token.
            self.flush_pending(layer)?;

            // Sequential path (rayon overhead not worth it for 1 token).
            let k_buf = &self.k_data_buf;
            let v_buf = &self.v_data_buf;
            let caches = &mut self.caches;
            for b in 0..batch_size {
                for head in 0..num_heads {
                    let offset = ((b * num_heads + head) * new_seq_len) * head_dim;
                    let k_vec = &k_buf[offset..offset + head_dim];
                    let v_vec = &v_buf[offset..offset + head_dim];
                    caches[head]
                        .push(layer, k_vec, v_vec)
                        .map_err(|e| tq_err(format!("push error: {e}")))?;
                }
            }
        }

        // Total sequence length = quantized entries + pending (not yet quantized) entries.
        let quantized_len = self.caches[0].entry_count(layer);
        let pending_len = self.pending_k[layer].first().map_or(0, |h| h.len());
        let total_seq_len = quantized_len + pending_len;

        // Log memory stats periodically (layer 0 only)
        const LOG_INTERVAL: usize = 100;
        if layer == 0 && total_seq_len % LOG_INTERVAL == 0 && total_seq_len > 0 {
            let tq_bytes = self.memory_usage();
            let fp16_bytes = total_seq_len * self.num_kv_heads * self.head_dim * 2 * 2;
            let ratio = if tq_bytes > 0 {
                fp16_bytes as f64 / tq_bytes as f64
            } else {
                0.0
            };
            tracing::info!(
                "TurboQuant KV cache: {total_seq_len} tokens, \
                 TQ={:.1} KB, FP16={:.1} KB, compression={ratio:.1}x",
                tq_bytes as f64 / 1024.0,
                fp16_bytes as f64 / 1024.0,
            );
        }

        // --- Optimized buffer update ---
        // Ensure GPU buffer is allocated and large enough
        self.ensure_buffer_capacity(layer, total_seq_len, &orig_device, orig_dtype)?;

        let start_idx = total_seq_len - new_seq_len;

        // PREFILL OPTIMIZATION: During prefill (new_seq_len > 1), we already have
        // the original K/V tensors on the correct device/dtype. Write them directly
        // into the GPU buffer — no dequantize needed (we only quantized for storage).
        // During decode (new_seq_len == 1), we must dequantize from the quantized
        // store since we don't retain the original tensors between steps.
        let is_prefill = new_seq_len > 1;

        if is_prefill {
            // Write original tensors directly into buffer (skip dequantize entirely)
            self.gpu_k_buf[layer].as_ref().unwrap()
                .slice_set(k, 2, start_idx)?;
            self.gpu_v_buf[layer].as_ref().unwrap()
                .slice_set(v, 2, start_idx)?;
        } else {
            // Decode: dequantize the new token(s) and write into buffer
            let delta_k = self.assemble_tensor_range(
                layer, &orig_device, orig_dtype,
                start_idx, total_seq_len, true,
            )?;
            let delta_v = self.assemble_tensor_range(
                layer, &orig_device, orig_dtype,
                start_idx, total_seq_len, false,
            )?;
            if let Some(ref delta) = delta_k {
                self.gpu_k_buf[layer].as_ref().unwrap()
                    .slice_set(delta, 2, start_idx)?;
            }
            if let Some(ref delta) = delta_v {
                self.gpu_v_buf[layer].as_ref().unwrap()
                    .slice_set(delta, 2, start_idx)?;
            }
        }
        self.buf_seq_len[layer] = total_seq_len;

        // Return narrow views (no copy — just a view into the buffer)
        let full_k = self.gpu_k_buf[layer].as_ref().unwrap()
            .narrow(2, 0, total_seq_len)?;
        let full_v = self.gpu_v_buf[layer].as_ref().unwrap()
            .narrow(2, 0, total_seq_len)?;

        Ok((full_k, full_v))
    }

    /// Replaces this cache with a fresh empty one, preserving configuration.
    ///
    /// This is the canonical way to reset a `TurboQuantKVCache`. The newly
    /// created cache already has `cached_k`/`cached_v` initialised to `None`,
    /// so no additional `invalidate_all_caches()` call is needed.
    pub fn reset_all(&mut self) -> Result<()> {
        let new = TurboQuantKVCache::new(self.bits, self.head_dim, self.num_kv_heads, self.num_layers)?;
        *self = new;
        Ok(())
    }

    /// Invalidates the GPU buffer and pending vectors for a given layer.
    pub fn invalidate_cache(&mut self, layer: usize) {
        if layer < self.gpu_k_buf.len() {
            self.gpu_k_buf[layer] = None;
            self.gpu_v_buf[layer] = None;
            self.buf_seq_len[layer] = 0;
            self.buf_capacity[layer] = 0;
            for head_bufs in &mut self.pending_k[layer] {
                head_bufs.clear();
            }
            for head_bufs in &mut self.pending_v[layer] {
                head_bufs.clear();
            }
        }
    }

    /// Invalidates all GPU buffers and pending vectors across every layer.
    pub fn invalidate_all_caches(&mut self) {
        for i in 0..self.gpu_k_buf.len() {
            self.gpu_k_buf[i] = None;
            self.gpu_v_buf[i] = None;
            self.buf_seq_len[i] = 0;
            self.buf_capacity[i] = 0;
            for head_bufs in &mut self.pending_k[i] {
                head_bufs.clear();
            }
            for head_bufs in &mut self.pending_v[i] {
                head_bufs.clear();
            }
        }
    }

    /// Ensures the GPU buffer for a layer has enough capacity for `needed_len` tokens.
    /// Allocates or re-allocates (doubling) if needed.
    fn ensure_buffer_capacity(
        &mut self,
        layer: usize,
        needed_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        if self.buf_capacity[layer] >= needed_len && self.gpu_k_buf[layer].is_some() {
            return Ok(());
        }

        // Double capacity or use INITIAL_BUFFER_CAPACITY, whichever is larger
        let new_cap = needed_len
            .max(self.buf_capacity[layer] * 2)
            .max(INITIAL_BUFFER_CAPACITY);

        let shape = (1, self.num_kv_heads, new_cap, self.head_dim);

        // Allocate new buffers on the target device
        let new_k = Tensor::zeros(shape, dtype, device)?;
        let new_v = Tensor::zeros(shape, dtype, device)?;

        // Copy old data if it exists
        let old_len = self.buf_seq_len[layer];
        if old_len > 0 {
            if let Some(ref old_k) = self.gpu_k_buf[layer] {
                let old_slice = old_k.narrow(2, 0, old_len)?;
                new_k.slice_set(&old_slice, 2, 0)?;
            }
            if let Some(ref old_v) = self.gpu_v_buf[layer] {
                let old_slice = old_v.narrow(2, 0, old_len)?;
                new_v.slice_set(&old_slice, 2, 0)?;
            }
        }

        self.gpu_k_buf[layer] = Some(new_k);
        self.gpu_v_buf[layer] = Some(new_v);
        self.buf_capacity[layer] = new_cap;
        Ok(())
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
        if self.caches.is_empty() {
            return 0;
        }
        let quantized = self.caches[0].entry_count(layer);
        let pending = self.pending_k[layer].first().map_or(0, |h| h.len());
        quantized + pending
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
        TurboQuantKVCache::new(
            TEST_BITS,
            TEST_HEAD_DIM,
            TEST_NUM_KV_HEADS,
            TEST_NUM_LAYERS,
        )
        .unwrap()
    }

    fn dummy_vec(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
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
}
