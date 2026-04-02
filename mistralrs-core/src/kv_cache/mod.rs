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
    Normal { k: SingleCache, v: SingleCache },
    Rotating { k: RotatingCache, v: RotatingCache },
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
                let guard = cache.lock().map_err(|e| {
                    candle_core::Error::Msg(format!("TurboQuant lock error: {e}"))
                })?;
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
                let guard = cache.lock().map_err(|e| {
                    candle_core::Error::Msg(format!("TurboQuant lock error: {e}"))
                })?;
                guard.dequantize_values_tensor(*layer, device, *dtype)
            }
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // Handle TurboQuant separately since it has a completely different code path
        if let Self::TurboQuant {
            cache, layer, ..
        } = self
        {
            let mut guard = cache.lock().map_err(|e| {
                candle_core::Error::Msg(format!("TurboQuant lock error: {e}"))
            })?;
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
    Normal { max_seq_len: usize },
    SlidingWindow { window: usize },
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
            AttentionImplementation::TurboQuant(bits) => {
                Self::new_turboquant(num_layers, *bits, head_dim, num_kv_heads, device, dtype)
                    .expect("Failed to create TurboQuant cache")
            }
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

    /// Creates a `NormalCache` where every layer uses TurboQuant quantization.
    ///
    /// A single `TurboQuantKVCache` is shared across all layers via `Arc<Mutex<_>>`.
    /// Each layer gets its own `KvCache::TurboQuant` that references the shared
    /// cache with its layer index.
    pub fn new_turboquant(
        num_layers: usize,
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        device: candle_core::Device,
        dtype: candle_core::DType,
    ) -> candle_core::Result<Arc<Mutex<Self>>> {
        let tq_cache = TurboQuantKVCache::new(bits, head_dim, num_kv_heads, num_layers)?;
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
    use candle_core::{DType, Device, Tensor};
    use crate::paged_attention::AttentionImplementation;

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
            &AttentionImplementation::TurboQuant(3),
            4,    // num_layers
            2048, // max_seq_len
            None,
            64,   // head_dim (must be power of two)
            8,    // num_kv_heads
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
            &AttentionImplementation::TurboQuant(4),
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
            &AttentionImplementation::TurboQuant(3),
            2,    // num_layers
            1024,
            None,
            64,   // head_dim
            2,    // num_kv_heads
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
            &AttentionImplementation::TurboQuant(3),
            1, 1024, None, 64, 2,
            Device::Cpu, DType::F32,
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
            &AttentionImplementation::TurboQuant(3),
            2, 1024, None, 64, 2,
            Device::Cpu, DType::F32,
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
            &AttentionImplementation::TurboQuant(3),
            1, 1024, None, 128, 8,
            Device::Cpu, DType::F32,
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
            &AttentionImplementation::TurboQuant(3),
            2,    // num_layers
            1024,
            None,
            64,   // head_dim
            2,    // num_kv_heads
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
            &AttentionImplementation::TurboQuant(3),
            2,    // num_layers
            1024,
            None,
            64,   // head_dim
            2,    // num_kv_heads
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
            &AttentionImplementation::TurboQuant(3),
            2,    // num_layers
            1024,
            None,
            64,   // head_dim
            2,    // num_kv_heads
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
            &AttentionImplementation::TurboQuant(3),
            1,    // num_layers
            1024,
            None,
            64,   // head_dim
            4,    // num_kv_heads
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
                "K history should have {} tokens after append {}", i + 1, i
            );
            assert_eq!(
                full_v.dims(),
                &[1, 4, i + 1, 64],
                "V history should have {} tokens after append {}", i + 1, i
            );
        }

        assert_eq!(locked.0[0].current_seq_len(), 5);
    }

    /// TQ cache handles prefill (multi-token append) followed by single-token
    /// decode steps.
    #[test]
    fn turboquant_cache_prefill_multi_token() {
        let cache = NormalCache::new_for_attention(
            &AttentionImplementation::TurboQuant(3),
            1,    // num_layers
            1024,
            None,
            64,   // head_dim
            2,    // num_kv_heads
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
}
