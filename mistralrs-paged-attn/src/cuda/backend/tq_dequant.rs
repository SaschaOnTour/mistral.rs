use candle_core::cuda::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Storage, Tensor};

use crate::cuda::ffi;

/// Launch the TurboQuant dequantize kernel on GPU.
///
/// Takes packed polar block data (already on GPU) and produces f32 output
/// directly in GPU memory — fused unpack + codebook + WHT + scale in one kernel.
pub fn tq_dequant_batch(
    packed_indices: &Tensor,
    scales: &Tensor,
    codebook: &Tensor,
    sign_pattern: &Tensor,
    num_blocks: usize,
    block_size: usize,
    bits: usize,
    bytes_per_block: usize,
    device: &Device,
) -> Result<Tensor> {
    let Device::Cuda(dev) = device else {
        candle_core::bail!("tq_dequant_batch requires CUDA device");
    };

    let output = Tensor::zeros((num_blocks, block_size), DType::F32, device)?;

    if num_blocks == 0 {
        return Ok(output);
    }

    let stream = dev.cuda_stream().cu_stream() as _;

    {
        let Storage::Cuda(packed_cuda) = &*packed_indices.storage_and_layout().0 else {
            candle_core::bail!("packed_indices must be on CUDA");
        };
        let Storage::Cuda(scales_cuda) = &*scales.storage_and_layout().0 else {
            candle_core::bail!("scales must be on CUDA");
        };
        let Storage::Cuda(codebook_cuda) = &*codebook.storage_and_layout().0 else {
            candle_core::bail!("codebook must be on CUDA");
        };
        let Storage::Cuda(sign_cuda) = &*sign_pattern.storage_and_layout().0 else {
            candle_core::bail!("sign_pattern must be on CUDA");
        };
        let Storage::Cuda(output_cuda) = &*output.storage_and_layout().0 else {
            candle_core::bail!("output must be on CUDA");
        };

        let packed_slice = packed_cuda.as_cuda_slice::<u8>()?;
        let scales_slice = scales_cuda.as_cuda_slice::<half::f16>()?;
        let codebook_slice = codebook_cuda.as_cuda_slice::<f32>()?;
        let sign_slice = sign_cuda.as_cuda_slice::<f32>()?;
        let output_slice = output_cuda.as_cuda_slice::<f32>()?;

        let (packed_ptr, _g1) = packed_slice.device_ptr(packed_slice.stream());
        let (scales_ptr, _g2) = scales_slice.device_ptr(scales_slice.stream());
        let (codebook_ptr, _g3) = codebook_slice.device_ptr(codebook_slice.stream());
        let (sign_ptr, _g4) = sign_slice.device_ptr(sign_slice.stream());
        let (output_ptr, _g5) = output_slice.device_ptr(output_slice.stream());

        unsafe {
            ffi::tq_dequant_batch(
                packed_ptr as *const u8,
                scales_ptr as *const u16,
                codebook_ptr as *const f32,
                sign_ptr as *const f32,
                output_ptr as *mut f32,
                num_blocks as i32,
                block_size as i32,
                bits as i32,
                bytes_per_block as i32,
                stream,
            );
        }
    }

    Ok(output)
}
