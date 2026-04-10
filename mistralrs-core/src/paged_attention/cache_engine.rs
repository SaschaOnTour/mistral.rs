use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::config::{KvCacheLayout, ModelConfigLike};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
// Note: pyo3::pyclass removed — complex enum with data variants not supported by PyO3.
// Python users should use string-based cache type selection instead.
pub enum PagedCacheType {
    Auto,
    F8E4M3,
    /// PolarQuant plain: block-level polar quantization, standard codebook, no QJL.
    /// CLI: --pa-cache-type pq3 / pq4
    PolarQuant(u8),
    /// PolarQuant Outlier: all blocks use outlier codebook (bits-bit).
    /// Best quality/performance ratio. No QJL overhead.
    /// CLI: --pa-cache-type pqo3 / pqo4
    PolarQuantOutlier(u8),
    /// TurboQuant (Paper Alg. 2): (bits-1)-bit PolarQuant + 1-bit QJL.
    /// Unbiased inner products at cost of higher variance + compute.
    /// CLI: --pa-cache-type tq3 / tq4
    TurboQuant(u8),
}

impl PagedCacheType {
    pub fn to_dtype(&self, act_dtype: DType) -> DType {
        match self {
            PagedCacheType::F8E4M3 => DType::F8E4M3,
            PagedCacheType::Auto
            | PagedCacheType::PolarQuant(_)
            | PagedCacheType::PolarQuantOutlier(_)
            | PagedCacheType::TurboQuant(_) => act_dtype,
        }
    }

    /// Returns `true` if this cache type uses compressed/quantized KV storage (PQ, PQO, or TQ).
    pub fn is_compressed_cache(&self) -> bool {
        matches!(
            self,
            PagedCacheType::PolarQuant(_)
                | PagedCacheType::PolarQuantOutlier(_)
                | PagedCacheType::TurboQuant(_)
        )
    }

    /// Returns the bit width for quantized cache types, or `None` for Auto/F8E4M3.
    pub fn tq_bits(&self) -> Option<u8> {
        match self {
            PagedCacheType::PolarQuant(bits)
            | PagedCacheType::PolarQuantOutlier(bits)
            | PagedCacheType::TurboQuant(bits) => Some(*bits),
            _ => None,
        }
    }
}

impl FromStr for PagedCacheType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "f8e4m3" => Ok(Self::F8E4M3),
            "pq3" => Ok(Self::PolarQuant(3)),
            "pq4" => Ok(Self::PolarQuant(4)),
            "pqo3" => Ok(Self::PolarQuantOutlier(3)),
            "pqo4" => Ok(Self::PolarQuantOutlier(4)),
            "tq3" => Ok(Self::TurboQuant(3)),
            "tq4" => Ok(Self::TurboQuant(4)),
            other => Err(format!(
                "Unknown cache type `{other}`. Options: auto, f8e4m3, pq3, pq4, pqo3, pqo4, tq3, tq4"
            )),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for PagedCacheType {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        let dtype = cache_config.cache_type.to_dtype(dtype);
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                cache_config,
                dtype,
                device,
                layer_devices,
            )?)),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        // Use blocking lock instead of busy-wait spin loop to avoid CPU waste
        // and potential thread starvation issues
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let kv_cache_layout = model_config.kv_cache_layout();
        let mut gpu_cache = Vec::new();

        for (layer_idx, device) in layer_devices
            .iter()
            .take(model_config.num_layers())
            .map(|x| x.as_ref().unwrap_or(device))
            .enumerate()
        {
            let (key_blocks, value_blocks) = match kv_cache_layout {
                KvCacheLayout::Standard => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                        layer_idx,
                    );
                    let value_block_shape = Self::calculate_value_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2
                                * key_block_shape.3;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * value_block_shape.0
                                * value_block_shape.1
                                * value_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::Mla {
                    kv_lora_rank,
                    kpe_head_dim,
                } => {
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kv_lora_rank;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kpe_head_dim;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.k_head_dim_for_layer(layer_idx) / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.v_head_dim_for_layer(layer_idx),
            block_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_auto() {
        assert_eq!(
            "auto".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::Auto
        );
    }

    #[test]
    fn parse_f8e4m3() {
        assert_eq!(
            "f8e4m3".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::F8E4M3
        );
    }

    #[test]
    fn parse_tq3() {
        assert_eq!(
            "tq3".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::TurboQuant(3)
        );
    }

    #[test]
    fn parse_tq4() {
        assert_eq!(
            "tq4".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::TurboQuant(4)
        );
    }

    #[test]
    fn parse_unknown_fails() {
        assert!("tq5".parse::<PagedCacheType>().is_err());
        assert!("fp16".parse::<PagedCacheType>().is_err());
    }

    #[test]
    fn is_compressed_cache_detects_quantized() {
        assert!(PagedCacheType::TurboQuant(3).is_compressed_cache());
        assert!(PagedCacheType::TurboQuant(4).is_compressed_cache());
        assert!(!PagedCacheType::Auto.is_compressed_cache());
        assert!(!PagedCacheType::F8E4M3.is_compressed_cache());
    }

    #[test]
    fn tq_bits_returns_correct_value() {
        assert_eq!(PagedCacheType::TurboQuant(3).tq_bits(), Some(3));
        assert_eq!(PagedCacheType::TurboQuant(4).tq_bits(), Some(4));
        assert_eq!(PagedCacheType::Auto.tq_bits(), None);
        assert_eq!(PagedCacheType::F8E4M3.tq_bits(), None);
    }

    #[test]
    fn to_dtype_tq_returns_activation_dtype() {
        let act = DType::F16;
        assert_eq!(PagedCacheType::TurboQuant(3).to_dtype(act), DType::F16);
        assert_eq!(PagedCacheType::TurboQuant(4).to_dtype(act), DType::F16);
    }

    #[test]
    fn to_dtype_f8e4m3_returns_f8() {
        assert_eq!(PagedCacheType::F8E4M3.to_dtype(DType::F16), DType::F8E4M3);
    }

    #[test]
    fn default_is_auto() {
        assert_eq!(PagedCacheType::default(), PagedCacheType::Auto);
    }
}
