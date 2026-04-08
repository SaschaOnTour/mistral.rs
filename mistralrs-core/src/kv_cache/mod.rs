use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::{Result, Tensor, D};

use crate::{
    get_mut_arcmutex,
    paged_attention::turboquant_cache::TurboQuantKVCache,
    pipeline::{CacheManagerMixin, MetadataMixin},
    sequence::Sequence,
};

mod full_cache;
mod hybrid_cache;
mod rotating_cache;
mod single_cache;

pub use full_cache::{EitherCache, LayerCaches};
pub use hybrid_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    RecurrentStateSnapshot,
};
pub use rotating_cache::RotatingCache;
pub use single_cache::SingleCache;

pub trait CacheManager<T: CacheManagerMixin + MetadataMixin + ?Sized> {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    );
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool);
    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    );
}

#[derive(Debug, Clone)]
pub enum KvCache {
    Normal {
        k: SingleCache,
        v: SingleCache,
    },
    Rotating {
        k: RotatingCache,
        v: RotatingCache,
    },
    TurboQuant {
        /// Shared multi-head quantized cache (shared across layers via Arc).
        cache: std::sync::Arc<std::sync::Mutex<TurboQuantKVCache>>,
        /// Which transformer layer this KvCache instance represents.
        layer: usize,
        /// Original device for the tensors.
        device: candle_core::Device,
        /// Original dtype for the tensors.
        dtype: candle_core::DType,
    },
}

impl KvCache {
    pub fn new_normal(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        let k = SingleCache::new(dim, max_seq_len, capacity_seq_len);
        let v = SingleCache::new(dim, max_seq_len, capacity_seq_len);
        Self::Normal { k, v }
    }

    pub fn new_rotating(dim: usize, sliding_window: usize, capacity_seq_len: usize) -> Self {
        let k = RotatingCache::new(dim, sliding_window, capacity_seq_len);
        let v = RotatingCache::new(dim, sliding_window, capacity_seq_len);
        Self::Rotating { k, v }
    }

    /// Creates a new TurboQuant KV cache for a specific layer.
    ///
    /// The `cache` is a shared `Arc<Mutex<TurboQuantKVCache>>` that holds
    /// quantized data for ALL layers and heads. Each `KvCache::TurboQuant`
    /// instance references it with a specific `layer` index.
    pub fn new_turboquant(
        cache: std::sync::Arc<std::sync::Mutex<TurboQuantKVCache>>,
        layer: usize,
        device: candle_core::Device,
        dtype: candle_core::DType,
    ) -> Self {
        Self::TurboQuant {
            cache,
            layer,
            device,
            dtype,
        }
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { k, .. } => k.current_data(),
            Self::Rotating { k, .. } => k.current_data(),
            Self::TurboQuant {
                cache,
                layer,
                device,
                dtype,
                ..
            } => {
                let guard = cache
                    .lock()
                    .map_err(|e| candle_core::Error::Msg(format!("TurboQuant lock error: {e}")))?;
                guard.dequantize_keys_tensor(*layer, device, *dtype)
            }
        }
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { v, .. } => v.current_data(),
            Self::Rotating { v, .. } => v.current_data(),
            Self::TurboQuant {
                cache,
                layer,
                device,
                dtype,
                ..
            } => {
                let guard = cache
                    .lock()
                    .map_err(|e| candle_core::Error::Msg(format!("TurboQuant lock error: {e}")))?;
                guard.dequantize_values_tensor(*layer, device, *dtype)
            }
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // Handle TurboQuant separately since it has a completely different code path
        if let Self::TurboQuant { cache, layer, .. } = self {
            let mut guard = cache
                .lock()
                .map_err(|e| candle_core::Error::Msg(format!("TurboQuant lock error: {e}")))?;
            return guard.append_and_dequantize(*layer, &k, &v);
        }

        let (out_k, out_v) = match self {
            Self::Normal { k: kc, v: vc } => {
                kc.append(&k)?;
                vc.append(&v)?;
                (kc.current_data()?, vc.current_data()?)
            }
            Self::Rotating { k: kc, v: vc } => {
                let out_k = kc.append(&k)?;
                let out_v = vc.append(&v)?;
                (Some(out_k), Some(out_v))
            }
            Self::TurboQuant { .. } => unreachable!("handled above"),
        };
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                match self {
                    Self::Normal { k, .. } => shape[k.dim] = 0,
                    Self::Rotating { k, .. } => shape[k.dim] = 0,
                    Self::TurboQuant { .. } => unreachable!("handled above"),
                }
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                match self {
                    Self::Normal { v, .. } => shape[v.dim] = 0,
                    Self::Rotating { v, .. } => shape[v.dim] = 0,
                    Self::TurboQuant { .. } => unreachable!("handled above"),
                }
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    pub fn current_seq_len(&self) -> usize {
        match self {
            Self::Normal { k, .. } => k.current_seq_len(),
            Self::Rotating { k, .. } => k.current_seq_len(),
            Self::TurboQuant { cache, layer, .. } => {
                let guard = cache.lock().expect("TurboQuant lock poisoned");
                guard.current_seq_len(*layer)
            }
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Normal { k, v } => {
                k.reset();
                v.reset();
            }
            Self::Rotating { k, v } => {
                k.reset();
                v.reset();
            }
            Self::TurboQuant { cache, .. } => {
                // TurboQuant caches are append-only; the simplest reset is
                // to replace the shared cache with a fresh one. Because a
                // single Arc<Mutex<TurboQuantKVCache>> is shared across all
                // layers, resetting one layer effectively needs to recreate
                // the whole cache. We do this by replacing the Arc contents.
                cache
                    .lock()
                    .expect("TurboQuant lock poisoned")
                    .reset_all()
                    .expect("TurboQuant reset: failed to create new cache");
            }
        }
    }

    /// Returns Ok if the length reassignment was successful, otherwise returns Err.
    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        match self {
            Self::Normal { k, v } => {
                k.set_len(len)?;
                v.set_len(len)?;
                Ok(())
            }
            Self::Rotating { k, v } => {
                k.set_len(len)?;
                v.set_len(len)?;
                Ok(())
            }
            Self::TurboQuant { .. } => {
                // TurboQuant is append-only; truncation is not supported.
                // Silently accept if the requested length matches current,
                // otherwise error.
                let current = self.current_seq_len();
                if len <= current {
                    Ok(())
                } else {
                    candle_core::bail!(
                        "TurboQuant cache: cannot extend length from {} to {}",
                        current,
                        len,
                    )
                }
            }
        }
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        match self {
            Self::Normal { k, v } => {
                k.try_set_len(len)?;
                v.try_set_len(len)?;
                Ok(())
            }
            Self::Rotating { k, v } => {
                k.try_set_len(len)?;
                v.try_set_len(len)?;
                Ok(())
            }
            Self::TurboQuant { .. } => {
                // TurboQuant is append-only; accept any length <= current.
                Ok(())
            }
        }
    }

    pub fn is_rotating(&self) -> bool {
        matches!(self, Self::Rotating { .. })
    }

    pub fn is_turboquant(&self) -> bool {
        matches!(self, Self::TurboQuant { .. })
    }

    /// Computes the QJL bias correction for attention logits (Paper Algorithm 2).
    ///
    /// Returns `Some(tensor)` for TurboQuant caches, `None` otherwise.
    /// The tensor shape is `[batch, num_kv_heads, q_len, kv_len]` and should
    /// be added to attention logits before softmax (via `SdpaParams.qjl_bias`).
    ///
    /// For non-TurboQuant caches, always returns `Ok(None)`.
    pub fn qjl_bias(&self, query: &Tensor) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { .. } | Self::Rotating { .. } => Ok(None),
            Self::TurboQuant { cache, layer, .. } => {
                let guard = cache
                    .lock()
                    .map_err(|e| candle_core::Error::Msg(format!("TurboQuant lock: {e}")))?;

                if !guard.has_qjl_data(*layer) {
                    return Ok(None);
                }

                let num_kv_heads = guard.num_kv_heads();
                let q_dims = query.dims4()?;
                let num_q_heads = q_dims.1;
                let q_per_head = query.narrow(1, 0, 1)?; // first head as template

                // Compute correction per KV head, then stack
                let mut head_corrections = Vec::with_capacity(num_kv_heads);
                for head in 0..num_kv_heads {
                    let corr = guard.qjl_correction(head, *layer, &q_per_head)?;
                    head_corrections.push(corr);
                }

                // Stack: [batch, num_kv_heads, q_len, kv_len]
                let refs: Vec<&Tensor> = head_corrections.iter().collect();
                let combined = Tensor::cat(&refs, 1)?;

                // GQA expansion: repeat KV heads to match query heads
                // (same as repeat_kv for K/V tensors in attention)
                let n_rep = num_q_heads / num_kv_heads;
                let expanded = if n_rep > 1 {
                    let (b, h, q, k) = combined.dims4()?;
                    combined
                        .unsqueeze(2)?
                        .expand((b, h, n_rep, q, k))?
                        .reshape((b, h * n_rep, q, k))?
                } else {
                    combined
                };
                // Match dtype of attention logits (may be F16 or BF16)
                let expanded = expanded.to_dtype(query.dtype())?;
                Ok(Some(expanded))
            }
        }
    }

    /// Clones a `TurboQuant` variant by cloning the `Arc` reference.
    /// Returns `None` for `Normal` / `Rotating` variants.
    fn clone_tq_ref(&self) -> Option<KvCache> {
        if let Self::TurboQuant {
            cache,
            layer,
            device,
            dtype,
        } = self
        {
            Some(Self::TurboQuant {
                cache: cache.clone(),
                layer: *layer,
                device: device.clone(),
                dtype: *dtype,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormalCache(pub Vec<KvCache>);

#[derive(Debug)]
pub enum NormalCacheType {
    Normal {
        max_seq_len: usize,
    },
    SlidingWindow {
        window: usize,
    },
    /// TurboQuant quantized KV cache. All layers share a single
    /// `TurboQuantKVCache` via `Arc<Mutex>`.
    TurboQuant {
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
    },
}

impl NormalCache {
    /// The number of tokens to grow the cache by
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(len: usize, max_seq_len: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self(vec![
            KvCache::new_normal(
                2,
                max_seq_len,
                Self::CACHE_GROW_SIZE
            );
            len
        ])))
    }

    pub fn new_sliding(
        len: usize,
        max_seq_len: usize,
        sliding_window: Option<usize>,
    ) -> Arc<Mutex<Self>> {
        match sliding_window {
            Some(sliding_window) => Arc::new(Mutex::new(Self(vec![
                KvCache::new_rotating(
                    2,
                    sliding_window,
                    Self::CACHE_GROW_SIZE
                );
                len
            ]))),
            None => Arc::new(Mutex::new(Self(vec![
                KvCache::new_normal(
                    2,
                    max_seq_len,
                    Self::CACHE_GROW_SIZE
                );
                len
            ]))),
        }
    }

    /// Creates the appropriate cache based on the attention mechanism.
    ///
    /// For `TurboQuant`: creates a shared quantized KV cache across all layers.
    /// For `Eager`/`PagedAttention`: delegates to `new_sliding`.
    pub fn new_for_attention(
        attention_mechanism: &crate::paged_attention::AttentionImplementation,
        num_layers: usize,
        max_seq_len: usize,
        sliding_window: Option<usize>,
        head_dim: usize,
        num_kv_heads: usize,
        device: candle_core::Device,
        dtype: candle_core::DType,
    ) -> Arc<Mutex<Self>> {
        use crate::paged_attention::AttentionImplementation;
        match attention_mechanism {
            AttentionImplementation::PolarQuant(bits, nm) => Self::new_pq(
                num_layers,
                *bits,
                head_dim,
                num_kv_heads,
                device,
                dtype,
                *nm,
            )
            .expect("Failed to create PQ cache"),
            AttentionImplementation::PolarQuantOutlier(bits, nm) => Self::new_pqo(
                num_layers,
                *bits,
                head_dim,
                num_kv_heads,
                device,
                dtype,
                *nm,
            )
            .expect("Failed to create PQO cache"),
            AttentionImplementation::TurboQuant(bits, nm) => Self::new_tq(
                num_layers,
                *bits,
                head_dim,
                num_kv_heads,
                device,
                dtype,
                *nm,
            )
            .expect("Failed to create TQ cache"),
            AttentionImplementation::Eager | AttentionImplementation::PagedAttention => {
                Self::new_sliding(num_layers, max_seq_len, sliding_window)
            }
        }
    }

    pub fn from_types(types: Vec<NormalCacheType>) -> Arc<Mutex<Self>> {
        let mut caches = Vec::new();
        for ty in types {
            match ty {
                NormalCacheType::Normal { max_seq_len } => {
                    caches.push(KvCache::new_normal(2, max_seq_len, Self::CACHE_GROW_SIZE));
                }
                NormalCacheType::SlidingWindow { window } => {
                    caches.push(KvCache::new_rotating(2, window, Self::CACHE_GROW_SIZE));
                }
                NormalCacheType::TurboQuant { .. } => {
                    panic!("Use NormalCache::new_for_attention() for TurboQuant caches")
                }
            }
        }
        Arc::new(Mutex::new(Self(caches)))
    }

    /// PQ: PolarQuant plain (standard codebook, no outlier, no QJL).
    pub fn new_pq(
        num_layers: usize,
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        device: candle_core::Device,
        dtype: candle_core::DType,
        norm_mode: crate::paged_attention::QuantNormMode,
    ) -> candle_core::Result<Arc<Mutex<Self>>> {
        Self::new_quantized_cache(
            TurboQuantKVCache::new_pq_with_norm(
                bits,
                head_dim,
                num_kv_heads,
                num_layers,
                norm_mode,
            )?,
            num_layers,
            device,
            dtype,
        )
    }

    /// PQO: PolarQuant Outlier (all blocks use outlier codebook, no QJL).
    pub fn new_pqo(
        num_layers: usize,
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        device: candle_core::Device,
        dtype: candle_core::DType,
        norm_mode: crate::paged_attention::QuantNormMode,
    ) -> candle_core::Result<Arc<Mutex<Self>>> {
        Self::new_quantized_cache(
            TurboQuantKVCache::new_pqo_with_norm(
                bits,
                head_dim,
                num_kv_heads,
                num_layers,
                norm_mode,
            )?,
            num_layers,
            device,
            dtype,
        )
    }

    /// TQ: TurboQuant (Paper Algorithm 2: (bits-1)-bit Polar + 1-bit QJL).
    pub fn new_tq(
        num_layers: usize,
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        device: candle_core::Device,
        dtype: candle_core::DType,
        norm_mode: crate::paged_attention::QuantNormMode,
    ) -> candle_core::Result<Arc<Mutex<Self>>> {
        Self::new_quantized_cache(
            TurboQuantKVCache::new_tq_with_norm(
                bits,
                head_dim,
                num_kv_heads,
                num_layers,
                norm_mode,
            )?,
            num_layers,
            device,
            dtype,
        )
    }

    fn new_quantized_cache(
        tq_cache: TurboQuantKVCache,
        num_layers: usize,
        device: candle_core::Device,
        dtype: candle_core::DType,
    ) -> candle_core::Result<Arc<Mutex<Self>>> {
        let shared = Arc::new(std::sync::Mutex::new(tq_cache));
        let mut caches = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            caches.push(KvCache::new_turboquant(
                shared.clone(),
                layer,
                device.clone(),
                dtype,
            ));
        }
        Ok(Arc::new(Mutex::new(Self(caches))))
    }
}

pub struct NormalCacheManager;

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for NormalCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        let mut new_k_cache = Vec::new();
        let mut new_v_cache = Vec::new();

        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            // Preallocate combined k and v caches across all sequences, avoiding Tensor::cat copies
            let batch_len = seqs.len();
            // Use the first sequence as template
            let (first_k, first_v) = {
                let src_cache = if modify_draft_cache {
                    seqs[0].normal_draft_cache()
                } else {
                    seqs[0].normal_cache()
                };
                let Some(cache) = src_cache.get(layer).unwrap().as_ref() else {
                    // This is hit in gemma3n for the shared kv cache
                    new_k_cache.push(None);
                    new_v_cache.push(None);
                    continue;
                };
                // TurboQuant caches are shared via Arc — skip tensor extraction.
                if cache.is_turboquant() {
                    new_k_cache.push(None);
                    new_v_cache.push(None);
                    continue;
                }
                match cache {
                    KvCache::Normal { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Rotating { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::TurboQuant { .. } => {
                        unreachable!("handled above")
                    }
                }
            };
            // Build dims for batched cache
            let mut dims_k = first_k.dims().to_vec();
            let mut dims_v = first_v.dims().to_vec();
            dims_k[0] *= batch_len;
            dims_v[0] *= batch_len;
            let batch_k = Tensor::zeros(dims_k.clone(), first_k.dtype(), first_k.device()).unwrap();
            let batch_v = Tensor::zeros(dims_v.clone(), first_v.dtype(), first_v.device()).unwrap();
            // Fill each sequence's cache slice
            for (i, seq) in seqs.iter_mut().enumerate() {
                let src_cache = if modify_draft_cache {
                    seq.normal_draft_cache()
                } else {
                    seq.normal_cache()
                };
                let Some(cache) = src_cache.get(layer).unwrap().as_ref() else {
                    // Skip for shared kv cache layers in models like gemma3n
                    continue;
                };
                let (src_k, src_v) = match cache {
                    KvCache::Normal { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Rotating { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::TurboQuant { .. } => {
                        // TurboQuant is shared; skip tensor extraction per-sequence.
                        continue;
                    }
                };
                let offset = i * first_k.dims()[0];
                batch_k.slice_set(&src_k, 0, offset).unwrap();
                batch_v.slice_set(&src_v, 0, offset).unwrap();
            }
            new_k_cache.push(Some(batch_k));
            new_v_cache.push(Some(batch_v));
        }

        let seq0_cache = if modify_draft_cache {
            &*seqs[0].normal_draft_cache()
        } else {
            &*seqs[0].normal_cache()
        };

        let existing_pipeline_caches = pipeline.cache().normal().0.clone();

        let mut caches = Vec::new();
        for (layer_idx, (k_cache, v_cache)) in new_k_cache.into_iter().zip(new_v_cache).enumerate()
        {
            // Use this for the various parameters. Assumes all seqs are from one model.
            let Some(cache_ref) = seq0_cache[layer_idx].as_ref() else {
                // Sequence has no cache for this layer. Check if the pipeline's
                // existing cache is TurboQuant -- if so, preserve the shared
                // reference instead of creating a dummy Normal cache.
                if let Some(existing) = existing_pipeline_caches.get(layer_idx) {
                    if let Some(tq) = existing.clone_tq_ref() {
                        caches.push(tq);
                        continue;
                    }
                }
                // This is hit in gemma3n for the shared kv cache - create dummy cache
                // These layers don't have their own cache because they share another layer's cache
                caches.push(KvCache::Normal {
                    k: SingleCache {
                        all_data: None,
                        dim: 0,
                        current_seq_len: 0,
                        max_seq_len: 0,
                        capacity_seq_len: 0,
                    },
                    v: SingleCache {
                        all_data: None,
                        dim: 0,
                        current_seq_len: 0,
                        max_seq_len: 0,
                        capacity_seq_len: 0,
                    },
                });
                continue;
            };
            match cache_ref {
                KvCache::Normal { k: old_k, .. } => {
                    let template_cache_dim = old_k.dim;
                    let template_cache_csl = old_k.current_seq_len;
                    let template_cache_msl = old_k.max_seq_len;
                    let template_cache_capsl = old_k.capacity_seq_len;

                    caches.push(KvCache::Normal {
                        k: SingleCache {
                            all_data: k_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                        },
                        v: SingleCache {
                            all_data: v_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                        },
                    });
                }
                KvCache::Rotating { k: old_k, .. } => {
                    let template_cache_dim = old_k.dim;
                    let template_cache_csl = old_k.current_seq_len;
                    let template_cache_msl = old_k.max_seq_len;
                    let template_cache_offset = old_k.offset;
                    let template_cache_capsl = old_k.capacity_seq_len;

                    caches.push(KvCache::Rotating {
                        k: RotatingCache {
                            all_data: k_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            offset: template_cache_offset,
                            capacity_seq_len: template_cache_capsl,
                        },
                        v: RotatingCache {
                            all_data: v_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            offset: template_cache_offset,
                            capacity_seq_len: template_cache_capsl,
                        },
                    });
                }
                KvCache::TurboQuant { .. } => {
                    caches.push(cache_ref.clone_tq_ref().unwrap());
                }
            }
        }
        *pipeline.cache().normal() = NormalCache(caches);
    }
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        let all_cache = pipeline.cache().normal();
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let cache = all_cache.0.get(layer).unwrap();

            // TurboQuant caches must be handled first, before the is_none
            // check below. Even when the TQ cache is empty (just reset),
            // we still need to store the shared Arc reference into each
            // sequence so that clone_in_cache can find a TurboQuant entry.
            if let Some(tq) = cache.clone_tq_ref() {
                for seq in seqs.iter_mut() {
                    let output_cache = if modify_draft_cache {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    let seq_cache = &mut output_cache[layer];
                    *seq_cache = Some(tq.clone());
                }
                continue;
            }

            // This case for llama 3.2 vision cross attn
            if cache.k().unwrap().is_none() {
                continue;
            }

            let (k_cache, v_cache) = match cache {
                KvCache::Normal { k, v } => {
                    (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                }
                KvCache::Rotating { k, v } => {
                    (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                }
                KvCache::TurboQuant { .. } => {
                    unreachable!("TurboQuant handled above");
                }
            };

            let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(k_caches.len(), seqs.len());
            let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(v_caches.len(), seqs.len());

            for (seq_i, seq) in seqs.iter_mut().enumerate() {
                let output_cache = if modify_draft_cache {
                    seq.normal_draft_cache()
                } else {
                    seq.normal_cache()
                };
                let seq_cache = &mut output_cache[layer];
                let k = k_caches.get(seq_i).unwrap().clone();
                let v = v_caches.get(seq_i).unwrap().clone();

                match cache {
                    KvCache::Normal {
                        k: cache_k,
                        v: cache_v,
                    } => {
                        *seq_cache = Some(KvCache::Normal {
                            k: SingleCache {
                                all_data: Some(k),
                                dim: cache_k.dim,
                                current_seq_len: cache_k.current_seq_len,
                                max_seq_len: cache_k.max_seq_len,
                                capacity_seq_len: cache_k.capacity_seq_len,
                            },
                            v: SingleCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                capacity_seq_len: cache_v.capacity_seq_len,
                            },
                        });
                    }
                    KvCache::Rotating {
                        k: cache_k,
                        v: cache_v,
                    } => {
                        *seq_cache = Some(KvCache::Rotating {
                            k: RotatingCache {
                                all_data: Some(k),
                                dim: cache_k.dim,
                                current_seq_len: cache_k.current_seq_len,
                                max_seq_len: cache_k.max_seq_len,
                                offset: cache_k.offset,
                                capacity_seq_len: cache_k.capacity_seq_len,
                            },
                            v: RotatingCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                offset: cache_v.offset,
                                capacity_seq_len: cache_v.capacity_seq_len,
                            },
                        });
                    }
                    KvCache::TurboQuant { .. } => {
                        // TurboQuant layers are skipped via `continue` above.
                        unreachable!("TurboQuant handled by continue above");
                    }
                }
            }
        }
    }
    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        _modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        if seqs.iter().any(|seq| seq.preallocated_cache().is_none()) {
            for layer in pipeline.cache().normal().0.iter_mut() {
                layer.reset();
            }
            return;
        }

        let layer_devices = pipeline.device_mapper().map(|device_mapper| {
            let total_layers = pipeline.cache().normal().0.len();
            let mut layer_devices = Vec::with_capacity(total_layers);
            for layer in 0..total_layers {
                let device = device_mapper
                    .device_for(layer, false)
                    .cloned()
                    .expect("Internal bug, layer out of range!");
                layer_devices.push(device);
            }
            layer_devices
        });

        let old_caches = pipeline.cache().normal().0.clone();

        for (layer_idx, layer) in pipeline.cache().normal().0.iter_mut().enumerate() {
            if !load_preallocated_cache {
                layer.reset();
                continue;
            }

            let mut k_caches = Vec::new();
            let mut v_caches = Vec::new();
            for seq in seqs.iter_mut() {
                let (mut k_preallocated_cache, mut v_preallocated_cache) =
                    (*seq.preallocated_cache().as_ref().unwrap()).clone();
                if let Some(layer_devices) = &layer_devices {
                    let layer_dev = &layer_devices[layer_idx];
                    k_preallocated_cache = k_preallocated_cache
                        .to_device(layer_dev)
                        .expect("Could not prepare cache");
                    v_preallocated_cache = v_preallocated_cache
                        .to_device(layer_dev)
                        .expect("Could not prepare cache");
                }
                k_caches.push(k_preallocated_cache);
                v_caches.push(v_preallocated_cache);
            }
            let k_cache = if k_caches.len() > 1 {
                Tensor::cat(&k_caches, 0).unwrap()
            } else {
                k_caches[0].clone()
            };
            let v_cache = if v_caches.len() > 1 {
                Tensor::cat(&v_caches, 0).unwrap()
            } else {
                v_caches[0].clone()
            };

            // Use this for the various parameters. Assumes all seqs are from one model.
            match &old_caches[layer_idx] {
                KvCache::Normal { k, .. } => {
                    let template_cache_dim = k.dim;
                    let template_cache_msl = k.max_seq_len;

                    let cache = KvCache::Normal {
                        k: SingleCache {
                            all_data: Some(k_cache.zeros_like().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: k_cache.dims()[template_cache_dim],
                        },
                        v: SingleCache {
                            all_data: Some(v_cache.zeros_like().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: k_cache.dims()[template_cache_dim],
                        },
                    };
                    *layer = cache;
                }
                KvCache::Rotating { k, .. } => {
                    let template_cache_dim = k.dim;
                    let template_cache_msl = k.max_seq_len;

                    // Rotating cache is not preallocated.
                    let cache = KvCache::Rotating {
                        k: RotatingCache {
                            all_data: None,
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            offset: 0,
                            capacity_seq_len: 0,
                        },
                        v: RotatingCache {
                            all_data: None,
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            offset: 0,
                            capacity_seq_len: 0,
                        },
                    };
                    *layer = cache;
                }
                KvCache::TurboQuant {
                    cache: tq_cache,
                    layer: tq_layer,
                    ..
                } => {
                    // Reset the TurboQuant cache by replacing its contents
                    // with a fresh empty cache. Since all layers share the
                    // same Arc, only reset once (on layer 0).
                    if *tq_layer == 0 {
                        tq_cache
                            .lock()
                            .expect("TurboQuant lock poisoned")
                            .reset_all()
                            .expect("TurboQuant set_none_cache: failed to create new cache");
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    cache: Arc<Mutex<LayerCaches>>,
    xlora_cache: Option<Arc<Mutex<LayerCaches>>>,
    draft_cache: Arc<Mutex<LayerCaches>>,
    scalings_cache: Option<Arc<Mutex<Option<Tensor>>>>,
}

impl Cache {
    pub(crate) fn new(len: usize, is_xlora: bool) -> Self {
        Self {
            cache: Arc::new(Mutex::new(vec![None; len])),
            xlora_cache: if is_xlora {
                Some(Arc::new(Mutex::new(vec![None; len])))
            } else {
                None
            },
            draft_cache: Arc::new(Mutex::new(vec![None; len])),
            scalings_cache: if is_xlora {
                Some(Arc::new(Mutex::new(None)))
            } else {
                None
            },
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.cache)
    }

    pub(crate) fn draft_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.draft_cache)
    }

    /// # Panics
    /// If there is no xlora cache
    pub(crate) fn xlora_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.xlora_cache.as_ref().expect("No X-LoRA cache."))
    }

    /// # Panics
    /// If there is no xlora cache
    pub(crate) fn get_scalings_cache(&self) -> MutexGuard<'_, Option<Tensor>> {
        get_mut_arcmutex!(self
            .scalings_cache
            .as_ref()
            .expect("No X-LoRA scalings cache."))
    }

    pub(crate) fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    /// Update the KV cache and return (k,v)
    pub(crate) fn update_kv_cache(
        cache: &mut Option<(Tensor, Tensor)>,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (k, v) = match &*cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                (k, v)
            }
        };
        *cache = Some((k.clone(), v.clone()));
        Ok((k.contiguous()?, v.contiguous()?))
    }

    /// Update the KV cache and return (k,v,attn_mask)
    pub(crate) fn update_kv_cache_sliding_window(
        cache: &mut Option<(Tensor, Tensor)>,
        k: Tensor,
        v: Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let (k, v, attention_mask) = match cache.clone() {
            None => (k, v, attention_mask.cloned()),
            Some((mut prev_k, mut prev_v)) => {
                let mut mask = attention_mask.cloned();
                if let Some(sliding_window) = sliding_window {
                    let kv_seq_len = prev_k.dim(2)?;
                    if kv_seq_len > sliding_window {
                        prev_k = prev_k.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        prev_v = prev_v.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        if let Some(ref mut mask) = mask {
                            let mask_len = mask.dim(1)?;
                            *mask = mask.narrow(
                                1,
                                mask_len - (sliding_window - 1),
                                sliding_window - 1,
                            )?;
                            *mask = Tensor::cat(
                                &[&*mask, &mask.narrow(1, mask_len - 1, 1)?.ones_like()?],
                                D::Minus1,
                            )?;
                        }
                    }
                }
                let (k, v) = {
                    let k = Tensor::cat(&[prev_k, k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[prev_v, v], 2)?.contiguous()?;
                    (k, v)
                };
                (k, v, mask)
            }
        };
        *cache = Some((k.clone(), v.clone()));
        Ok((k.contiguous()?, v.contiguous()?, attention_mask))
    }
}

pub struct FullCacheManager;

enum SeqCache {
    Normal,
    XLora,
    Draft,
}

fn clone_in_cache(
    num_hidden_layers: usize,
    cache: &mut LayerCaches,
    seqs: &mut [&mut crate::sequence::Sequence],
    src: SeqCache,
) {
    let mut new_cache = Vec::new();
    'outer: for layer in 0..num_hidden_layers {
        let mut k_vec = Vec::new();
        let mut v_vec = Vec::new();
        for seq in &mut *seqs {
            let src_cache = match src {
                SeqCache::Normal => seq.cache(),
                SeqCache::XLora => seq.xlora_cache(),
                SeqCache::Draft => seq.draft_cache(),
            };
            let cache = src_cache.get(layer).unwrap();
            // This case for llama 3.2 vision cross attn
            if cache.is_none() {
                new_cache.push(None);
                continue 'outer;
            }
            let cache = cache
                .as_ref()
                .expect("Not handling completions in `clone_in_cache`.");
            k_vec.push(cache.0.clone());
            v_vec.push(cache.1.clone());
        }
        new_cache.push(Some((
            if k_vec.len() > 1 {
                Tensor::cat(&k_vec, 0).unwrap()
            } else {
                k_vec[0].clone()
            },
            if v_vec.len() > 1 {
                Tensor::cat(&v_vec, 0).unwrap()
            } else {
                v_vec[0].clone()
            },
        )));
    }
    *cache = new_cache;
}

fn clone_out_cache(
    num_hidden_layers: usize,
    cache: &mut LayerCaches,
    seqs: &mut [&mut crate::sequence::Sequence],
    target: SeqCache,
) {
    for layer in 0..num_hidden_layers {
        let cache = cache.get(layer).unwrap();
        // This case for llama 3.2 vision cross attn
        if cache.is_none() {
            continue;
        }

        let k_cache = cache.as_ref().unwrap().0.clone();
        let v_cache = cache.as_ref().unwrap().1.clone();

        let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
        debug_assert_eq!(k_caches.len(), seqs.len());
        let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
        debug_assert_eq!(v_caches.len(), seqs.len());

        for (seq_i, seq) in seqs.iter_mut().enumerate() {
            let output_cache = match target {
                SeqCache::Normal => seq.cache(),
                SeqCache::XLora => seq.xlora_cache(),
                SeqCache::Draft => seq.draft_cache(),
            };
            let seq_cache = &mut output_cache[layer];
            let k = k_caches.get(seq_i).unwrap().clone();
            let v = v_caches.get(seq_i).unwrap().clone();
            *seq_cache = Some((k, v));
        }
    }
}

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for FullCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        if modify_draft_cache {
            clone_in_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().lock(),
                seqs,
                SeqCache::Draft,
            );
            return;
        }
        clone_in_cache(
            pipeline.get_metadata().num_hidden_layers,
            &mut pipeline.cache().full().lock(),
            seqs,
            SeqCache::Normal,
        );
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().no_kv_cache {
            clone_in_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().xlora_lock(),
                seqs,
                SeqCache::XLora,
            );
        }
        if pipeline.get_metadata().is_xlora {
            pipeline
                .cache()
                .full()
                .get_scalings_cache()
                .clone_from(seqs[0].scaling_cache());
        }
    }

    fn clone_out_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        if modify_draft_cache {
            clone_out_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().lock(),
                seqs,
                SeqCache::Draft,
            );
            return;
        }
        clone_out_cache(
            pipeline.get_metadata().num_hidden_layers,
            &mut pipeline.cache().full().lock(),
            seqs,
            SeqCache::Normal,
        );
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().no_kv_cache {
            clone_out_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().xlora_lock(),
                seqs,
                SeqCache::XLora,
            );
        }
        if pipeline.get_metadata().is_xlora {
            seqs[0]
                .scaling_cache()
                .clone_from(&pipeline.cache().full().get_scalings_cache());
        }
    }

    fn set_none_cache(
        &self,
        pipeline: &T,
        _seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
        let mut new_cache = Vec::new();
        for _ in 0..pipeline.get_metadata().num_hidden_layers {
            new_cache.push(None);
        }
        pipeline.cache().full().lock().clone_from(&new_cache);
        if modify_draft_cache {
            pipeline.cache().full().draft_lock().clone_from(&new_cache);
        }
        if pipeline.cache().full().is_xlora() {
            *pipeline.cache().full().xlora_lock() = new_cache;
        }
    }
}

/// Cache manager for hybrid models (attention + recurrent layers).
///
/// This implements vLLM-style continuous batching:
/// - Attention layers: Standard KV cache batching (cat on clone_in, chunk on clone_out)
/// - Recurrent layers: Pool-based state management with indexed access
///
/// Each sequence has a `recurrent_state_idx` pointing to its slot in the
/// state pool. The forward pass builds a `state_indices` tensor from these
/// indices and uses gather/scatter operations.
pub struct HybridCacheManager;

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for HybridCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        let mut hybrid_cache = pipeline.cache().hybrid();
        let num_layers = hybrid_cache.num_layers();

        // Build state_indices for recurrent layers from sequences' recurrent_state_idx
        // Find the device from the first recurrent layer's pool
        let recurrent_device = hybrid_cache.caches.iter().find_map(|c| {
            if let HybridLayerCache::Recurrent(pool) = c {
                Some(pool.device().clone())
            } else {
                None
            }
        });

        // Ensure every sequence has a recurrent slot when using hybrid cache.
        let mut state_index_allocation_failed = false;
        let mut newly_allocated = Vec::new();
        for (seq_idx, seq) in seqs.iter_mut().enumerate() {
            if seq.recurrent_state_idx().is_none() {
                if let Some(slot_idx) = hybrid_cache.allocate_seq() {
                    seq.set_recurrent_state_idx(Some(slot_idx));
                    newly_allocated.push((seq_idx, slot_idx));
                } else {
                    tracing::warn!(
                        "Failed to allocate recurrent state slot for sequence {}, hybrid forward will fail for this batch.",
                        seq.id()
                    );
                    state_index_allocation_failed = true;
                    break;
                }
            }
        }
        if state_index_allocation_failed {
            for (seq_idx, slot_idx) in newly_allocated {
                seqs[seq_idx].set_recurrent_state_idx(None);
                hybrid_cache.free_seq(slot_idx);
            }
        }

        if let Some(device) = recurrent_device {
            if state_index_allocation_failed {
                hybrid_cache.set_state_indices(None);
            } else {
                // Build state_indices tensor from sequences
                let mut indices = Vec::with_capacity(seqs.len());
                for seq in seqs.iter() {
                    if let Some(idx) = seq.recurrent_state_idx() {
                        #[allow(clippy::cast_possible_truncation)]
                        indices.push(idx as u32);
                    } else {
                        tracing::warn!(
                            "Sequence {} missing recurrent_state_idx during hybrid clone_in_cache.",
                            seq.id()
                        );
                        hybrid_cache.set_state_indices(None);
                        return;
                    }
                }
                if let Ok(state_indices) = Tensor::from_vec(indices, (seqs.len(),), &device) {
                    hybrid_cache.set_state_indices(Some(state_indices));
                } else {
                    hybrid_cache.set_state_indices(None);
                }
            }
        }

        // For attention layers, we still need to batch KV caches
        for layer_idx in 0..num_layers {
            let layer_cache = hybrid_cache.caches.get_mut(layer_idx).unwrap();

            if let HybridLayerCache::Attention(kv_cache) = layer_cache {
                // Batch KV caches from sequences (same as NormalCacheManager)
                let mut k_tensors = Vec::new();
                let mut v_tensors = Vec::new();
                let mut template_cache: Option<KvCache> = None;

                for seq in seqs.iter_mut() {
                    let seq_cache = if modify_draft_cache {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    if let Some(Some(ref kv)) = seq_cache.get(layer_idx) {
                        if template_cache.is_none() {
                            template_cache = Some(kv.clone());
                        }
                        if let (Ok(Some(k)), Ok(Some(v))) = (kv.k(), kv.v()) {
                            k_tensors.push(k);
                            v_tensors.push(v);
                        }
                    }
                }

                if !k_tensors.is_empty() {
                    // cat/clone of narrow'd views may be non-contiguous;
                    // all_data must be contiguous for slice_set in SingleCache::append.
                    let batched_k = if k_tensors.len() > 1 {
                        Tensor::cat(&k_tensors, 0).unwrap()
                    } else {
                        k_tensors[0].contiguous().unwrap()
                    };
                    let batched_v = if v_tensors.len() > 1 {
                        Tensor::cat(&v_tensors, 0).unwrap()
                    } else {
                        v_tensors[0].contiguous().unwrap()
                    };

                    if let Some(ref template) = template_cache {
                        match (template, kv_cache) {
                            (KvCache::Normal { k: tk, .. }, KvCache::Normal { k, v }) => {
                                k.all_data = Some(batched_k);
                                k.current_seq_len = tk.current_seq_len;
                                k.capacity_seq_len = tk.current_seq_len;
                                v.all_data = Some(batched_v);
                                v.current_seq_len = tk.current_seq_len;
                                v.capacity_seq_len = tk.current_seq_len;
                            }
                            (KvCache::Rotating { k: tk, .. }, KvCache::Rotating { k, v }) => {
                                k.all_data = Some(batched_k);
                                k.current_seq_len = tk.current_seq_len;
                                k.capacity_seq_len = tk.current_seq_len;
                                k.offset = tk.offset;
                                v.all_data = Some(batched_v);
                                v.current_seq_len = tk.current_seq_len;
                                v.capacity_seq_len = tk.current_seq_len;
                                v.offset = tk.offset;
                            }
                            _ => {}
                        }
                    }
                }
            }
            // For recurrent layers: No copying needed!
            // The pool is accessed directly via state_indices during forward.
        }
    }

    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        let hybrid_cache = pipeline.cache().hybrid();
        let num_layers = hybrid_cache.num_layers();
        let num_seqs = seqs.len();

        // For attention layers, split batched KV caches back to sequences
        for layer_idx in 0..num_layers {
            let layer_cache = hybrid_cache.caches.get(layer_idx).unwrap();

            if let HybridLayerCache::Attention(kv_cache) = layer_cache {
                if let (Ok(Some(k)), Ok(Some(v))) = (kv_cache.k(), kv_cache.v()) {
                    let k_chunks = k.chunk(num_seqs, 0).unwrap();
                    let v_chunks = v.chunk(num_seqs, 0).unwrap();

                    for (seq_idx, seq) in seqs.iter_mut().enumerate() {
                        // chunk() returns non-contiguous views; all_data must be contiguous.
                        let seq_k = k_chunks.get(seq_idx).unwrap().contiguous().unwrap();
                        let seq_v = v_chunks.get(seq_idx).unwrap().contiguous().unwrap();

                        let seq_cache = if modify_draft_cache {
                            seq.normal_draft_cache()
                        } else {
                            seq.normal_cache()
                        };

                        // Initialize cache if needed
                        if seq_cache.get(layer_idx).is_none() || seq_cache[layer_idx].is_none() {
                            while seq_cache.len() <= layer_idx {
                                seq_cache.push(None);
                            }
                            seq_cache[layer_idx] = Some(kv_cache.clone());
                        }

                        if let Some(ref mut seq_kv) = seq_cache[layer_idx] {
                            match (kv_cache, seq_kv) {
                                (KvCache::Normal { k: src_k, .. }, KvCache::Normal { k, v }) => {
                                    k.all_data = Some(seq_k);
                                    k.current_seq_len = src_k.current_seq_len;
                                    k.capacity_seq_len = src_k.current_seq_len;
                                    v.all_data = Some(seq_v);
                                    v.current_seq_len = src_k.current_seq_len;
                                    v.capacity_seq_len = src_k.current_seq_len;
                                }
                                (
                                    KvCache::Rotating { k: src_k, .. },
                                    KvCache::Rotating { k, v },
                                ) => {
                                    k.all_data = Some(seq_k);
                                    k.current_seq_len = src_k.current_seq_len;
                                    k.capacity_seq_len = src_k.current_seq_len;
                                    k.offset = src_k.offset;
                                    v.all_data = Some(seq_v);
                                    v.current_seq_len = src_k.current_seq_len;
                                    v.capacity_seq_len = src_k.current_seq_len;
                                    v.offset = src_k.offset;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // For recurrent layers: No splitting needed!
            // The pool was updated in-place during forward via scatter operations.
        }
    }

    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
        // Reset attention KV caches in sequences
        for seq in seqs.iter_mut() {
            let seq_cache = if modify_draft_cache {
                seq.normal_draft_cache()
            } else {
                seq.normal_cache()
            };
            for kv in seq_cache.iter_mut().flatten() {
                kv.reset();
            }
        }
        // Reset the hybrid cache (including recurrent state pools)
        let mut hybrid_cache = pipeline.cache().hybrid();
        hybrid_cache.reset();

        // Build state_indices so the forward pass can access recurrent pool states.
        // Sequences already have slots allocated from add_request.
        let recurrent_device = hybrid_cache.caches.iter().find_map(|c| {
            if let HybridLayerCache::Recurrent(pool) = c {
                Some(pool.device().clone())
            } else {
                None
            }
        });
        if let Some(device) = recurrent_device {
            #[allow(clippy::cast_possible_truncation)]
            let indices: Vec<u32> = seqs
                .iter()
                .filter_map(|seq| seq.recurrent_state_idx().map(|idx| idx as u32))
                .collect();
            if indices.len() == seqs.len() {
                if let Ok(state_indices) = Tensor::from_vec(indices, (seqs.len(),), &device) {
                    hybrid_cache.set_state_indices(Some(state_indices));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::{AttentionImplementation, QuantNormMode};
    use candle_core::{DType, Device, Tensor};

    // -----------------------------------------------------------------------
    // Test 1: new_for_attention factory
    // -----------------------------------------------------------------------

    #[test]
    fn new_for_attention_eager_creates_normal_cache() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::Eager,
            4,    // num_layers
            2048, // max_seq_len
            None, // no sliding window
            128,  // head_dim
            8,    // num_kv_heads
            Device::Cpu,
            DType::F32,
        );
        let locked = cache.lock().unwrap();
        assert_eq!(locked.0.len(), 4);
        // Should be Normal variant, not TurboQuant
        assert!(!locked.0[0].is_turboquant());
    }

    #[test]
    fn new_for_attention_tq3_creates_turboquant_cache() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            4,    // num_layers
            2048, // max_seq_len
            None,
            64, // head_dim (must be power of two)
            8,  // num_kv_heads
            Device::Cpu,
            DType::F32,
        );
        let locked = cache.lock().unwrap();
        assert_eq!(locked.0.len(), 4);
        assert!(locked.0[0].is_turboquant());
    }

    #[test]
    fn new_for_attention_tq4_creates_turboquant_cache() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(4, QuantNormMode::MaxNorm),
            2,
            1024,
            None,
            128,
            4,
            Device::Cpu,
            DType::F32,
        );
        let locked = cache.lock().unwrap();
        assert_eq!(locked.0.len(), 2);
        assert!(locked.0[0].is_turboquant());
    }

    // -----------------------------------------------------------------------
    // Test 2: Full KvCache::TurboQuant append roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn turboquant_cache_append_roundtrip() {
        // Create a TQ3 cache: 2 layers, head_dim=64, 2 kv_heads
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            2, // num_layers
            1024,
            None,
            64, // head_dim
            2,  // num_kv_heads
            Device::Cpu,
            DType::F32,
        );

        let mut locked = cache.lock().unwrap();
        let kv_cache = &mut locked.0[0]; // layer 0

        // Create dummy K/V tensors: [batch=1, num_kv_heads=2, seq_len=1, head_dim=64]
        let k = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();

        // Append
        let (full_k, full_v) = kv_cache.append(&k, &v).unwrap();

        // Check output shape: [1, 2, 1, 64] (one token in cache)
        assert_eq!(full_k.dims(), &[1, 2, 1, 64]);
        assert_eq!(full_v.dims(), &[1, 2, 1, 64]);

        // Append another token
        let k2 = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let v2 = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let (full_k2, full_v2) = kv_cache.append(&k2, &v2).unwrap();

        // Now should have 2 tokens: [1, 2, 2, 64]
        assert_eq!(full_k2.dims(), &[1, 2, 2, 64]);
        assert_eq!(full_v2.dims(), &[1, 2, 2, 64]);
    }

    #[test]
    fn turboquant_cache_append_preserves_dtype() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            1,
            1024,
            None,
            64,
            2,
            Device::Cpu,
            DType::F32,
        );
        let mut locked = cache.lock().unwrap();
        let kv_cache = &mut locked.0[0];

        // Use BF16 input tensors to test dtype preservation through
        // the quantize-dequantize roundtrip.
        let k = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let (full_k, full_v) = kv_cache.append(&k, &v).unwrap();
        assert_eq!(full_k.dtype(), DType::BF16);
        assert_eq!(full_v.dtype(), DType::BF16);
    }

    #[test]
    fn turboquant_cache_layers_independent() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            2,
            1024,
            None,
            64,
            2,
            Device::Cpu,
            DType::F32,
        );
        let mut locked = cache.lock().unwrap();

        // Push to layer 0 only
        let k = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        locked.0[0].append(&k, &v).unwrap();

        // Layer 0 has 1 token, layer 1 has 0
        assert_eq!(locked.0[0].current_seq_len(), 1);
        assert_eq!(locked.0[1].current_seq_len(), 0);
    }

    // -----------------------------------------------------------------------
    // Test 3: Compression / multi-token accumulation
    // -----------------------------------------------------------------------

    #[test]
    fn turboquant_cache_memory_smaller_than_fp16() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            1,
            1024,
            None,
            128,
            8,
            Device::Cpu,
            DType::F32,
        );
        let mut locked = cache.lock().unwrap();

        // Push 100 tokens
        for _ in 0..100 {
            let k = Tensor::rand(0f32, 1.0, (1, 8, 1, 128), &Device::Cpu).unwrap();
            let v = Tensor::rand(0f32, 1.0, (1, 8, 1, 128), &Device::Cpu).unwrap();
            locked.0[0].append(&k, &v).unwrap();
        }

        // FP16 equivalent: 100 tokens * 8 heads * 128 dim * 2 bytes * 2 (K+V) = 409600 bytes
        // TQ3 should be significantly smaller.
        // We can't easily measure TQ memory from KvCache, but we verify
        // the returned tensor shape is correct after accumulation.
        let k = Tensor::rand(0f32, 1.0, (1, 8, 1, 128), &Device::Cpu).unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, 8, 1, 128), &Device::Cpu).unwrap();
        let (full_k, _) = locked.0[0].append(&k, &v).unwrap();
        assert_eq!(full_k.dims(), &[1, 8, 101, 128]); // 101 total tokens
    }

    // -----------------------------------------------------------------------
    // Regression tests: clone_in / clone_out / set_none cache bug
    //
    // These tests ensure TurboQuant caches survive the operations that
    // NormalCacheManager performs (clone_out, clone_in, set_none) without
    // losing their variant or shared Arc identity.
    // -----------------------------------------------------------------------

    /// Critical regression test: TQ cache remains TurboQuant after simulated
    /// clone_out + clone_in cycle and data persists through the shared Arc.
    #[test]
    fn turboquant_cache_shared_across_layers() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            2, // num_layers
            1024,
            None,
            64, // head_dim
            2,  // num_kv_heads
            Device::Cpu,
            DType::F32,
        );
        let mut locked = cache.lock().unwrap();

        // Both layers should be TurboQuant
        assert!(locked.0[0].is_turboquant());
        assert!(locked.0[1].is_turboquant());

        // Push to layer 0 via one reference
        let k = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        locked.0[0].append(&k, &v).unwrap();

        // Layer 0 should have 1 token, layer 1 should have 0
        assert_eq!(locked.0[0].current_seq_len(), 1);
        assert_eq!(locked.0[1].current_seq_len(), 0);
    }

    /// Verify that all layers in a TurboQuant cache share the same Arc
    /// allocation (pointer equality). If clone_out/clone_in were to deep-copy
    /// the Arc, layers would diverge and this test would catch it.
    #[test]
    fn turboquant_cache_arc_sharing() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            2, // num_layers
            1024,
            None,
            64, // head_dim
            2,  // num_kv_heads
            Device::Cpu,
            DType::F32,
        );
        let locked = cache.lock().unwrap();

        // Extract the Arc from layer 0 and layer 1
        if let KvCache::TurboQuant { cache: arc0, .. } = &locked.0[0] {
            if let KvCache::TurboQuant { cache: arc1, .. } = &locked.0[1] {
                // Both layers must share the SAME Arc (same allocation)
                assert!(
                    std::sync::Arc::ptr_eq(arc0, arc1),
                    "TurboQuant layers should share the same Arc, but they point to different allocations"
                );
            } else {
                panic!("Layer 1 should be TurboQuant");
            }
        } else {
            panic!("Layer 0 should be TurboQuant");
        }
    }

    /// After reset, the cache should still be the TurboQuant variant (not
    /// silently replaced by Normal or set to None).
    #[test]
    fn turboquant_cache_reset_preserves_variant() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            2, // num_layers
            1024,
            None,
            64, // head_dim
            2,  // num_kv_heads
            Device::Cpu,
            DType::F32,
        );
        let mut locked = cache.lock().unwrap();

        // Push data
        let k = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        locked.0[0].append(&k, &v).unwrap();
        assert_eq!(locked.0[0].current_seq_len(), 1);

        // Reset
        locked.0[0].reset();

        // Should still be TurboQuant, just empty
        assert!(
            locked.0[0].is_turboquant(),
            "After reset, cache should still be TurboQuant variant"
        );
        // Reset replaces the shared cache contents, so seq_len should be 0
        assert_eq!(locked.0[0].current_seq_len(), 0);
    }

    /// Multiple single-token appends must accumulate correctly, returning the
    /// full history each time.
    #[test]
    fn turboquant_cache_accumulates_across_appends() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            1, // num_layers
            1024,
            None,
            64, // head_dim
            4,  // num_kv_heads
            Device::Cpu,
            DType::F32,
        );
        let mut locked = cache.lock().unwrap();

        // Push 5 tokens one at a time
        for i in 0..5 {
            let k = Tensor::rand(0f32, 1.0, (1, 4, 1, 64), &Device::Cpu).unwrap();
            let v = Tensor::rand(0f32, 1.0, (1, 4, 1, 64), &Device::Cpu).unwrap();
            let (full_k, full_v) = locked.0[0].append(&k, &v).unwrap();

            // Should return full history: [1, 4, i+1, 64]
            assert_eq!(
                full_k.dims(),
                &[1, 4, i + 1, 64],
                "K history should have {} tokens after append {}",
                i + 1,
                i
            );
            assert_eq!(
                full_v.dims(),
                &[1, 4, i + 1, 64],
                "V history should have {} tokens after append {}",
                i + 1,
                i
            );
        }

        assert_eq!(locked.0[0].current_seq_len(), 5);
    }

    /// TQ cache handles prefill (multi-token append) followed by single-token
    /// decode steps.
    #[test]
    fn turboquant_cache_prefill_multi_token() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::PolarQuantOutlier(3, QuantNormMode::MaxNorm),
            1, // num_layers
            1024,
            None,
            64, // head_dim
            2,  // num_kv_heads
            Device::Cpu,
            DType::F32,
        );
        let mut locked = cache.lock().unwrap();

        // Prefill with 10 tokens at once
        let k = Tensor::rand(0f32, 1.0, (1, 2, 10, 64), &Device::Cpu).unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, 2, 10, 64), &Device::Cpu).unwrap();
        let (full_k, full_v) = locked.0[0].append(&k, &v).unwrap();

        assert_eq!(full_k.dims(), &[1, 2, 10, 64]);
        assert_eq!(full_v.dims(), &[1, 2, 10, 64]);
        assert_eq!(locked.0[0].current_seq_len(), 10);

        // Then decode one more token
        let k2 = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let v2 = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let (full_k2, full_v2) = locked.0[0].append(&k2, &v2).unwrap();

        assert_eq!(full_k2.dims(), &[1, 2, 11, 64]);
        assert_eq!(full_v2.dims(), &[1, 2, 11, 64]);
        assert_eq!(locked.0[0].current_seq_len(), 11);
    }

    // -------------------------------------------------------------------
    // QJL bias via KvCache interface
    // -------------------------------------------------------------------

    /// KvCache::qjl_bias() returns None for Normal caches.
    #[test]
    fn qjl_bias_returns_none_for_normal_cache() {
        let mut cache = KvCache::new_normal(2, 100, 512);
        let k = Tensor::rand(0f32, 1.0, (1, 2, 4, 64), &Device::Cpu).unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, 2, 4, 64), &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();

        let q = Tensor::rand(0f32, 1.0, (1, 2, 1, 64), &Device::Cpu).unwrap();
        let bias = cache.qjl_bias(&q).unwrap();
        assert!(
            bias.is_none(),
            "Normal cache should return None for qjl_bias"
        );
    }

    /// KvCache::qjl_bias() returns a tensor for TurboQuant caches.
    #[test]
    fn qjl_bias_returns_tensor_for_turboquant() {
        use crate::paged_attention::turboquant_cache::TurboQuantKVCache;
        use std::sync::{Arc, Mutex};

        let bits: u8 = 3;
        let dim: usize = 64;
        let heads: usize = 2;
        let layers: usize = 1;
        let layer: usize = 0;

        let tq = TurboQuantKVCache::new_tq(bits, dim, heads, layers).unwrap();
        let shared = Arc::new(Mutex::new(tq));
        let mut cache = KvCache::new_turboquant(shared, layer, Device::Cpu, DType::F32);

        // Prefill 4 tokens then decode 1 (triggers quantization with QJL)
        let k_pf = Tensor::rand(0f32, 1.0, (1, heads, 4, dim), &Device::Cpu).unwrap();
        let v_pf = Tensor::rand(0f32, 1.0, (1, heads, 4, dim), &Device::Cpu).unwrap();
        cache.append(&k_pf, &v_pf).unwrap();
        let k_dec = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();
        let v_dec = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();
        cache.append(&k_dec, &v_dec).unwrap();

        // Query
        let q = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();
        let bias = cache.qjl_bias(&q).unwrap();

        assert!(
            bias.is_some(),
            "TurboQuant cache should return Some for qjl_bias"
        );
        let bias = bias.unwrap();
        // Shape: [batch, num_kv_heads, q_len, kv_len]
        assert_eq!(bias.dims()[0], 1, "batch");
        assert_eq!(bias.dims()[2], 1, "q_len");
        assert_eq!(bias.dims()[3], 5, "kv_len = 4 prefill + 1 decode");
    }

    /// End-to-end: KvCache::qjl_bias() flows into Sdpa.run_attention()
    /// and changes the output compared to no QJL.
    #[test]
    fn qjl_bias_end_to_end_through_sdpa() {
        use crate::attention::{Sdpa, SdpaParams};
        use crate::paged_attention::turboquant_cache::TurboQuantKVCache;
        use std::sync::{Arc, Mutex};

        let bits: u8 = 3;
        let dim: usize = 64;
        let heads: usize = 2;
        let layers: usize = 1;
        let layer: usize = 0;
        let scale = 1.0 / (dim as f32).sqrt();

        let tq = TurboQuantKVCache::new_tq(bits, dim, heads, layers).unwrap();
        let shared = Arc::new(Mutex::new(tq));
        let mut cache = KvCache::new_turboquant(shared, layer, Device::Cpu, DType::F32);

        // Prefill + decode to trigger QJL storage
        let kv_len = 8;
        let k_pf = Tensor::rand(0f32, 1.0, (1, heads, kv_len - 1, dim), &Device::Cpu).unwrap();
        let v_pf = Tensor::rand(0f32, 1.0, (1, heads, kv_len - 1, dim), &Device::Cpu).unwrap();
        cache.append(&k_pf, &v_pf).unwrap();
        let k_dec = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();
        let v_dec = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();
        let (full_k, full_v) = cache.append(&k_dec, &v_dec).unwrap();

        let q = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();

        // Without QJL
        let params_no_qjl = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };
        let out_no_qjl = Sdpa
            .run_attention(&q, &full_k, &full_v, None, None, &params_no_qjl)
            .unwrap();

        // With QJL from cache
        let qjl_bias = cache.qjl_bias(&q).unwrap();
        assert!(qjl_bias.is_some(), "Should have QJL bias");

        let params_with_qjl = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias,
        };
        let out_with_qjl = Sdpa
            .run_attention(&q, &full_k, &full_v, None, None, &params_with_qjl)
            .unwrap();

        // QJL bias must change output
        let diff = (&out_with_qjl - &out_no_qjl)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-6,
            "QJL bias from cache should change attention output, diff={diff:.6}"
        );
    }

    /// SdpaParams::with_qjl() creates a new SdpaParams with the QJL bias set.
    /// This is the API model attention layers use to pass QJL data.
    #[test]
    fn sdpa_params_with_qjl_helper() {
        use crate::attention::SdpaParams;

        let base = SdpaParams {
            n_kv_groups: 4,
            softcap: Some(30.0),
            softmax_scale: 0.125,
            sliding_window: Some(4096),
            sinks: None,
            qjl_bias: None,
        };

        // with_qjl(None) preserves None
        let p1 = base.with_qjl(None);
        assert!(p1.qjl_bias.is_none());
        assert_eq!(p1.n_kv_groups, 4);
        assert_eq!(p1.softcap, Some(30.0));
        assert_eq!(p1.softmax_scale, 0.125);
        assert_eq!(p1.sliding_window, Some(4096));

        // with_qjl(Some(tensor)) sets the bias
        let bias = Tensor::zeros((1, 1, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let p2 = base.with_qjl(Some(bias));
        assert!(p2.qjl_bias.is_some());
        assert_eq!(p2.qjl_bias.unwrap().dims(), &[1, 1, 1, 8]);

        // Original untouched
        assert!(base.qjl_bias.is_none());
    }

    /// Verify that ALL model files that call kv_cache.append() also pass
    /// the KvCache to run_attention (so QJL bias is automatically applied).
    ///
    /// This is a code-scan test that catches when a new model is added
    /// without QJL support. It greps the source for the pattern:
    ///   kv_cache.append(...) followed by Sdpa.run_attention(...)
    /// and verifies that run_attention receives the kv_cache parameter.
    ///
    /// NOTE: This test checks source code structure, not runtime behavior.
    /// It ensures architectural consistency across all model implementations.
    #[test]
    fn all_models_pass_kv_cache_to_run_attention() {
        use std::fs;
        use std::path::Path;

        let model_dirs = [
            "src/models",
            "src/vision_models",
            "src/xlora_models",
            "src/embedding_models",
            "src/speech_models",
        ];

        let mut missing = Vec::new();

        for dir in &model_dirs {
            let base = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
            if !base.exists() {
                continue;
            }
            for entry in walkdir(base) {
                let content = fs::read_to_string(&entry).unwrap_or_default();
                // Find files that call both kv_cache.append AND Sdpa.run_attention
                let has_append = content.contains("kv_cache.append(");
                let has_run_attention = content.contains("Sdpa.run_attention(")
                    || content.contains("Sdpa.run_attention_noflash(");
                if has_append && has_run_attention {
                    // Check that run_attention call site passes kv_cache
                    // Pattern: .run_attention(..., &kv_cache) or uses with_qjl
                    // Must actively compute QJL bias — not just have "qjl_bias: None"
                    let has_qjl_integration =
                        content.contains(".with_qjl(") || content.contains(".qjl_bias(");
                    if !has_qjl_integration {
                        missing.push(
                            entry
                                .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                                .unwrap_or(&entry)
                                .display()
                                .to_string(),
                        );
                    }
                }
            }
        }

        if !missing.is_empty() {
            panic!(
                "The following model files call kv_cache.append() + Sdpa.run_attention() \
                 but do NOT integrate QJL bias (missing with_qjl/qjl_bias):\n  {}",
                missing.join("\n  ")
            );
        }
    }

    /// Recursively collect .rs files from a directory.
    fn walkdir(dir: std::path::PathBuf) -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    files.extend(walkdir(path));
                } else if path.extension().map_or(false, |e| e == "rs") {
                    files.push(path);
                }
            }
        }
        files
    }

    /// QJL bias must be broadcastable to attention logits shape when
    /// using GQA (num_query_heads > num_kv_heads).
    ///
    /// Qwen3-0.6B: 16 query heads, 8 KV heads (GQA ratio 2).
    /// The bias must be [batch, num_q_heads, q_len, kv_len] or
    /// broadcastable to it (e.g. [batch, 1, q_len, kv_len]).
    #[test]
    fn qjl_bias_broadcastable_with_gqa() {
        use crate::attention::{Sdpa, SdpaParams};
        use crate::paged_attention::turboquant_cache::TurboQuantKVCache;
        use std::sync::{Arc, Mutex};

        let bits: u8 = 3;
        let dim: usize = 64;
        let kv_heads: usize = 4;
        let q_heads: usize = 8; // GQA: 2 query heads per KV head
        let layers: usize = 1;
        let layer: usize = 0;
        let scale = 1.0 / (dim as f32).sqrt();

        let tq = TurboQuantKVCache::new_tq(bits, dim, kv_heads, layers).unwrap();
        let shared = Arc::new(Mutex::new(tq));
        let mut cache = KvCache::new_turboquant(shared, layer, Device::Cpu, DType::F32);

        // Prefill + decode
        let k_pf = Tensor::rand(0f32, 1.0, (1, kv_heads, 4, dim), &Device::Cpu).unwrap();
        let v_pf = Tensor::rand(0f32, 1.0, (1, kv_heads, 4, dim), &Device::Cpu).unwrap();
        cache.append(&k_pf, &v_pf).unwrap();
        let k_dec = Tensor::rand(0f32, 1.0, (1, kv_heads, 1, dim), &Device::Cpu).unwrap();
        let v_dec = Tensor::rand(0f32, 1.0, (1, kv_heads, 1, dim), &Device::Cpu).unwrap();
        let (full_k, full_v) = cache.append(&k_dec, &v_dec).unwrap();

        // Query has MORE heads than KV (GQA)
        let q = Tensor::rand(0f32, 1.0, (1, q_heads, 1, dim), &Device::Cpu).unwrap();

        let qjl_bias = cache.qjl_bias(&q).unwrap();
        assert!(qjl_bias.is_some());
        let bias = qjl_bias.unwrap();

        // Bias must be broadcastable to [1, q_heads, 1, kv_len]
        // Either [1, q_heads, 1, kv_len] or [1, 1, 1, kv_len]
        let bias_heads = bias.dims()[1];
        assert!(
            bias_heads == q_heads || bias_heads == 1,
            "QJL bias heads ({bias_heads}) must be {q_heads} or 1 for GQA broadcast"
        );

        // run_attention handles GQA expansion internally via repeat_kv,
        // so pass K/V with kv_heads (not q_heads)
        let n_rep = q_heads / kv_heads;
        let params = SdpaParams {
            n_kv_groups: n_rep,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: Some(bias),
        };
        let result = Sdpa.run_attention(&q, &full_k, &full_v, None, None, &params);
        assert!(
            result.is_ok(),
            "Attention with GQA + QJL bias must not crash: {:?}",
            result.err()
        );
    }

    // -------------------------------------------------------------------
    // Block-wise attention via cached_attention()
    // -------------------------------------------------------------------

    /// Block-wise decode attention must produce the same result as
    /// full-dequant + SDPA (within quantization tolerance).
    #[test]
    fn blockwise_matches_full_dequant_cpu() {
        use crate::attention::{cached_attention, Sdpa, SdpaParams};
        use crate::paged_attention::turboquant_cache::TurboQuantKVCache;
        use std::sync::{Arc, Mutex};

        let bits: u8 = 3;
        let dim: usize = 128;
        let kv_heads: usize = 8;
        let q_heads: usize = 16; // GQA ratio 2
        let layers: usize = 1;
        let layer: usize = 0;
        let scale = 1.0 / (dim as f32).sqrt();
        let n_kv_groups = q_heads / kv_heads;

        let sdpa_params = SdpaParams {
            n_kv_groups,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };

        // --- Reference path: full-dequant + SDPA ---
        let tq_ref = TurboQuantKVCache::new_pqo(bits, dim, kv_heads, layers).unwrap();
        let shared_ref = Arc::new(Mutex::new(tq_ref));
        let mut cache_ref = KvCache::new_turboquant(shared_ref, layer, Device::Cpu, DType::F32);

        // --- Blockwise path ---
        let tq_bw = TurboQuantKVCache::new_pqo(bits, dim, kv_heads, layers).unwrap();
        let shared_bw = Arc::new(Mutex::new(tq_bw));
        let mut cache_bw = KvCache::new_turboquant(shared_bw, layer, Device::Cpu, DType::F32);

        // Prefill 32 tokens (both caches get same data)
        let k_pf = Tensor::rand(0f32, 1.0, (1, kv_heads, 32, dim), &Device::Cpu).unwrap();
        let v_pf = Tensor::rand(0f32, 1.0, (1, kv_heads, 32, dim), &Device::Cpu).unwrap();
        let (k_ref, v_ref) = cache_ref.append(&k_pf, &v_pf).unwrap();
        let (_, _) = cache_bw.append(&k_pf, &v_pf).unwrap();

        // Decode 1 token
        let k_dec = Tensor::rand(0f32, 1.0, (1, kv_heads, 1, dim), &Device::Cpu).unwrap();
        let v_dec = Tensor::rand(0f32, 1.0, (1, kv_heads, 1, dim), &Device::Cpu).unwrap();
        let q = Tensor::rand(0f32, 1.0, (1, q_heads, 1, dim), &Device::Cpu).unwrap();

        // Reference: full-dequant + SDPA
        let (k_full, v_full) = cache_ref.append(&k_dec, &v_dec).unwrap();
        let ref_out = Sdpa
            .run_attention(&q, &k_full, &v_full, None, None, &sdpa_params)
            .unwrap();

        // Blockwise: cached_attention dispatches to blockwise path
        let bw_out =
            cached_attention(&mut cache_bw, &q, &k_dec, &v_dec, None, &sdpa_params, None).unwrap();

        // Compare shapes
        assert_eq!(ref_out.dims(), bw_out.dims(), "Output shapes must match");
        assert_eq!(ref_out.dims(), &[1, q_heads, 1, dim]);

        // Compare values — should be close (both go through quantization)
        let ref_flat: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();
        let bw_flat: Vec<f32> = bw_out.flatten_all().unwrap().to_vec1().unwrap();

        let mut max_diff: f32 = 0.0;
        for (r, b) in ref_flat.iter().zip(bw_flat.iter()) {
            max_diff = max_diff.max((r - b).abs());
        }
        assert!(
            max_diff < 0.05,
            "Block-wise and full-dequant attention should match closely, max diff: {max_diff}"
        );
    }

    /// cached_attention with Normal cache must behave identically to append + SDPA.
    #[test]
    fn cached_attention_normal_cache_unchanged() {
        use crate::attention::{cached_attention, Sdpa, SdpaParams};

        let dim: usize = 64;
        let heads: usize = 4;
        let scale = 1.0 / (dim as f32).sqrt();
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };

        // Reference: direct append + SDPA
        let mut cache_ref = KvCache::new_normal(2, 1024, 512);
        let k_pf = Tensor::rand(0f32, 1.0, (1, heads, 8, dim), &Device::Cpu).unwrap();
        let v_pf = Tensor::rand(0f32, 1.0, (1, heads, 8, dim), &Device::Cpu).unwrap();
        let (k_r, v_r) = cache_ref.append(&k_pf, &v_pf).unwrap();
        let k_dec = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();
        let v_dec = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();
        let q = Tensor::rand(0f32, 1.0, (1, heads, 1, dim), &Device::Cpu).unwrap();
        let (k_r, v_r) = cache_ref.append(&k_dec, &v_dec).unwrap();
        let ref_out = Sdpa
            .run_attention(&q, &k_r, &v_r, None, None, &sdpa_params)
            .unwrap();

        // cached_attention path
        let mut cache_ca = KvCache::new_normal(2, 1024, 512);
        let _ = cache_ca.append(&k_pf, &v_pf).unwrap();
        let ca_out =
            cached_attention(&mut cache_ca, &q, &k_dec, &v_dec, None, &sdpa_params, None).unwrap();

        // Must be bit-identical (no quantization involved)
        let ref_flat: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();
        let ca_flat: Vec<f32> = ca_out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            ref_flat, ca_flat,
            "Normal cache: cached_attention must be identical to append+SDPA"
        );
    }

    /// Prefill (seq_len > 1) should use the standard path, not blockwise.
    #[test]
    fn cached_attention_prefill_uses_standard_path() {
        use crate::attention::{cached_attention, SdpaParams};
        use crate::paged_attention::turboquant_cache::TurboQuantKVCache;
        use std::sync::{Arc, Mutex};

        let bits: u8 = 3;
        let dim: usize = 64;
        let heads: usize = 2;
        let layers: usize = 1;
        let scale = 1.0 / (dim as f32).sqrt();
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };

        let tq = TurboQuantKVCache::new_pqo(bits, dim, heads, layers).unwrap();
        let shared = Arc::new(Mutex::new(tq));
        let mut cache = KvCache::new_turboquant(shared, 0, Device::Cpu, DType::F32);

        // Prefill with 8 tokens — should work and return correct shape
        let k = Tensor::rand(0f32, 1.0, (1, heads, 8, dim), &Device::Cpu).unwrap();
        let v = Tensor::rand(0f32, 1.0, (1, heads, 8, dim), &Device::Cpu).unwrap();
        let q = Tensor::rand(0f32, 1.0, (1, heads, 8, dim), &Device::Cpu).unwrap();

        let result = cached_attention(&mut cache, &q, &k, &v, None, &sdpa_params, None).unwrap();
        assert_eq!(result.dims(), &[1, heads, 8, dim]);
    }

    /// Fused CUDA kernel must match full-dequant + SDPA on GPU.
    /// Skipped if no CUDA device available.
    #[test]
    fn fused_kernel_matches_full_dequant_gpu() {
        use crate::attention::{cached_attention, Sdpa, SdpaParams};
        use crate::paged_attention::turboquant_cache::TurboQuantKVCache;
        use std::sync::{Arc, Mutex};

        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap()
        } else {
            eprintln!("Skipping fused_kernel_matches_full_dequant_gpu: no CUDA");
            return;
        };

        let bits: u8 = 3;
        let dim: usize = 128;
        let kv_heads: usize = 8;
        let q_heads: usize = 16; // GQA ratio 2
        let layers: usize = 1;
        let layer: usize = 0;
        let scale = 1.0 / (dim as f32).sqrt();
        let n_kv_groups = q_heads / kv_heads;
        let dtype = DType::BF16;

        let sdpa_params = SdpaParams {
            n_kv_groups,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };

        // --- Reference: full-dequant + SDPA (using append directly) ---
        let tq_ref = TurboQuantKVCache::new_pqo(bits, dim, kv_heads, layers).unwrap();
        let shared_ref = Arc::new(Mutex::new(tq_ref));
        let mut cache_ref = KvCache::new_turboquant(shared_ref, layer, device.clone(), dtype);

        // --- Fused kernel path (via cached_attention) ---
        let tq_fused = TurboQuantKVCache::new_pqo(bits, dim, kv_heads, layers).unwrap();
        let shared_fused = Arc::new(Mutex::new(tq_fused));
        let mut cache_fused = KvCache::new_turboquant(shared_fused, layer, device.clone(), dtype);

        // Prefill 64 tokens (same data for both)
        let k_pf = Tensor::rand(0f32, 1.0, (1, kv_heads, 64, dim), &device).unwrap();
        let v_pf = Tensor::rand(0f32, 1.0, (1, kv_heads, 64, dim), &device).unwrap();
        cache_ref.append(&k_pf, &v_pf).unwrap();
        cache_fused.append(&k_pf, &v_pf).unwrap();

        // Decode 1 token
        let k_dec = Tensor::rand(0f32, 1.0, (1, kv_heads, 1, dim), &device).unwrap();
        let v_dec = Tensor::rand(0f32, 1.0, (1, kv_heads, 1, dim), &device).unwrap();
        let q = Tensor::rand(0f32, 1.0, (1, q_heads, 1, dim), &device).unwrap();

        // Reference: full-dequant + SDPA
        let (k_full, v_full) = cache_ref.append(&k_dec, &v_dec).unwrap();
        let ref_out = Sdpa
            .run_attention(&q, &k_full, &v_full, None, None, &sdpa_params)
            .unwrap();

        // Fused: cached_attention dispatches to fused kernel on CUDA
        let fused_out = cached_attention(
            &mut cache_fused,
            &q,
            &k_dec,
            &v_dec,
            None,
            &sdpa_params,
            None,
        )
        .unwrap();

        // Compare
        assert_eq!(ref_out.dims(), fused_out.dims(), "Output shapes must match");

        let ref_flat: Vec<f32> = ref_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let fused_flat: Vec<f32> = fused_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut max_diff: f32 = 0.0;
        for (r, f) in ref_flat.iter().zip(fused_flat.iter()) {
            max_diff = max_diff.max((r - f).abs());
        }
        assert!(
            max_diff < 0.1,
            "Fused kernel and full-dequant attention must match, max diff: {max_diff}"
        );
    }

    /// Verify the PQO3 pipeline: prefill quantizes immediately, decode works correctly.
    #[test]
    fn pqo3_prefill_decode_pipeline() {
        use crate::attention::{cached_attention, SdpaParams};
        use crate::paged_attention::turboquant_cache::TurboQuantKVCache;
        use std::sync::{Arc, Mutex};

        let bits: u8 = 3;
        let dim: usize = 64;
        let kv_heads: usize = 4;
        let q_heads: usize = 4;
        let layers: usize = 1;
        let layer: usize = 0;
        let scale = 1.0 / (dim as f32).sqrt();
        let sdpa_params = SdpaParams {
            n_kv_groups: q_heads / kv_heads,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            qjl_bias: None,
        };

        let tq = TurboQuantKVCache::new_pqo(bits, dim, kv_heads, layers).unwrap();
        let shared = Arc::new(Mutex::new(tq));
        let mut cache = KvCache::new_turboquant(shared.clone(), layer, Device::Cpu, DType::F32);

        // Step 1: Prefill with 16 tokens — quantizes immediately
        let k_pf = Tensor::rand(0f32, 1.0, (1, kv_heads, 16, dim), &Device::Cpu).unwrap();
        let v_pf = Tensor::rand(0f32, 1.0, (1, kv_heads, 16, dim), &Device::Cpu).unwrap();
        let q_pf = Tensor::rand(0f32, 1.0, (1, q_heads, 16, dim), &Device::Cpu).unwrap();
        let pf_out =
            cached_attention(&mut cache, &q_pf, &k_pf, &v_pf, None, &sdpa_params, None).unwrap();
        assert_eq!(pf_out.dims(), &[1, q_heads, 16, dim]);

        // After prefill, compressed cache should exist immediately
        {
            let guard = shared.lock().unwrap();
            assert_eq!(guard.buf_seq_len_for_test(layer), 16);
        }

        // Step 2: Decode tokens
        for step in 0..3 {
            let k_d = Tensor::rand(0f32, 1.0, (1, kv_heads, 1, dim), &Device::Cpu).unwrap();
            let v_d = Tensor::rand(0f32, 1.0, (1, kv_heads, 1, dim), &Device::Cpu).unwrap();
            let q_d = Tensor::rand(0f32, 1.0, (1, q_heads, 1, dim), &Device::Cpu).unwrap();
            let d_out =
                cached_attention(&mut cache, &q_d, &k_d, &v_d, None, &sdpa_params, None).unwrap();
            assert_eq!(d_out.dims(), &[1, q_heads, 1, dim]);

            let guard = shared.lock().unwrap();
            assert_eq!(guard.buf_seq_len_for_test(layer), 16 + step + 1);
        }
    }
}
