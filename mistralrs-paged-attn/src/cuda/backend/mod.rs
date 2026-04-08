mod cache;
mod context_attention_mla;
mod flash_attn_sinks;
mod gather_kv;
mod mla;
mod paged_attention;
mod scale_update;
mod tq_attention;
mod tq_dequant;
mod tq_quant;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, DevicePtr, DeviceRepr},
};
pub use context_attention_mla::context_attention_fwd_mla;
pub use flash_attn_sinks::{flash_attn_sinks, flash_attn_sinks_varlen};
pub use gather_kv::gather_kv_cache;
pub use mla::{concat_and_cache_mla, flashinfer_mla_decode, gather_mla_cache};
pub use paged_attention::{paged_attention, reshape_and_cache};
pub use scale_update::kv_scale_update;
pub use tq_attention::tq_fused_attention;
pub use tq_dequant::tq_dequant_batch;
pub use tq_quant::{tq_pack_indices, tq_qjl_batch, tq_quant_batch, tq_quant_maxnorm_batch};

pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    let (_, guard) = v.device_ptr(v.stream());
    let (ptr, _) = v.slice(lo..).device_ptr(v.stream());
    (ptr, guard)
}
