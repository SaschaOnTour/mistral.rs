use candle_core::cuda::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Storage, Tensor};

use crate::cuda::ffi;

/// Launch the TurboQuant PolarQuant kernel on GPU.
///
/// Quantizes f32 input vectors to packed polar blocks directly on GPU.
/// Returns `(packed_indices, scales)` where scales are `DType::F16`.
pub fn tq_quant_batch(
    input: &Tensor,
    boundaries: &Tensor,
    sign_seed: u64,
    num_blocks: usize,
    head_dim: usize,
    bits: usize,
    num_boundaries: usize,
    bytes_per_block: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let Device::Cuda(dev) = device else {
        candle_core::bail!("tq_quant_batch requires CUDA device");
    };

    let packed_out = Tensor::zeros(num_blocks * bytes_per_block, DType::U8, device)?;
    let scales_out = Tensor::zeros(num_blocks, DType::F16, device)?;

    if num_blocks == 0 {
        return Ok((packed_out, scales_out));
    }

    let stream = dev.cuda_stream().cu_stream() as _;

    {
        let Storage::Cuda(input_cuda) = &*input.storage_and_layout().0 else {
            candle_core::bail!("input must be on CUDA");
        };
        let Storage::Cuda(boundaries_cuda) = &*boundaries.storage_and_layout().0 else {
            candle_core::bail!("boundaries must be on CUDA");
        };
        let Storage::Cuda(packed_cuda) = &*packed_out.storage_and_layout().0 else {
            candle_core::bail!("packed_out must be on CUDA");
        };
        let Storage::Cuda(scales_cuda) = &*scales_out.storage_and_layout().0 else {
            candle_core::bail!("scales_out must be on CUDA");
        };

        let input_slice = input_cuda.as_cuda_slice::<f32>()?;
        let boundaries_slice = boundaries_cuda.as_cuda_slice::<f32>()?;
        let packed_slice = packed_cuda.as_cuda_slice::<u8>()?;
        let scales_slice = scales_cuda.as_cuda_slice::<half::f16>()?;

        let (input_ptr, _g1) = input_slice.device_ptr(input_slice.stream());
        let (boundaries_ptr, _g2) = boundaries_slice.device_ptr(boundaries_slice.stream());
        let (packed_ptr, _g3) = packed_slice.device_ptr(packed_slice.stream());
        let (scales_ptr, _g4) = scales_slice.device_ptr(scales_slice.stream());

        unsafe {
            ffi::tq_quant_batch(
                input_ptr as *const f32,
                boundaries_ptr as *const f32,
                sign_seed,
                packed_ptr as *mut u8,
                scales_ptr as *mut u16,
                num_blocks as i32,
                head_dim as i32,
                bits as i32,
                num_boundaries as i32,
                bytes_per_block as i32,
                stream,
            );
        }
    }

    Ok((packed_out, scales_out))
}

/// Quantize-and-pack kernel for MaxNorm mode on GPU.
///
/// Input: `rotated_input` — ALREADY rotated (WHT + signs applied) flat F32
///        tensor `[num_blocks * block_size]`.
/// The WHT is done externally (Candle butterfly_wht_forward_gpu) so that
/// quant and dequant are numerically consistent.
/// Returns `(packed_indices, scales)` on GPU.
pub fn tq_quant_maxnorm_batch(
    rotated_input: &Tensor,
    boundaries: &Tensor,
    num_blocks: usize,
    block_size: usize,
    bits: usize,
    num_boundaries: usize,
    bytes_per_block: usize,
    outer_centroid: f32,
    scale_sign: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let Device::Cuda(dev) = device else {
        candle_core::bail!("tq_quant_maxnorm_batch requires CUDA device");
    };

    let packed_out = Tensor::zeros(num_blocks * bytes_per_block, DType::U8, device)?;
    let scales_out = Tensor::zeros(num_blocks, DType::F16, device)?;

    if num_blocks == 0 {
        return Ok((packed_out, scales_out));
    }

    let stream = dev.cuda_stream().cu_stream() as _;

    {
        let Storage::Cuda(input_cuda) = &*rotated_input.storage_and_layout().0 else {
            candle_core::bail!("rotated_input must be on CUDA");
        };
        let Storage::Cuda(bound_cuda) = &*boundaries.storage_and_layout().0 else {
            candle_core::bail!("boundaries must be on CUDA");
        };
        let Storage::Cuda(packed_cuda) = &*packed_out.storage_and_layout().0 else {
            candle_core::bail!("packed_out must be on CUDA");
        };
        let Storage::Cuda(scales_cuda) = &*scales_out.storage_and_layout().0 else {
            candle_core::bail!("scales_out must be on CUDA");
        };

        let input_slice = input_cuda.as_cuda_slice::<f32>()?;
        let bound_slice = bound_cuda.as_cuda_slice::<f32>()?;
        let packed_slice = packed_cuda.as_cuda_slice::<u8>()?;
        let scales_slice = scales_cuda.as_cuda_slice::<half::f16>()?;

        let (input_ptr, _g1) = input_slice.device_ptr(input_slice.stream());
        let (bound_ptr, _g2) = bound_slice.device_ptr(bound_slice.stream());
        let (packed_ptr, _g3) = packed_slice.device_ptr(packed_slice.stream());
        let (scales_ptr, _g4) = scales_slice.device_ptr(scales_slice.stream());

        unsafe {
            ffi::tq_quant_maxnorm_batch(
                input_ptr as *const f32,
                bound_ptr as *const f32,
                packed_ptr as *mut u8,
                scales_ptr as *mut u16,
                num_blocks as i32,
                block_size as i32,
                bits as i32,
                num_boundaries as i32,
                bytes_per_block as i32,
                outer_centroid,
                scale_sign,
                stream,
            );
        }
    }

    Ok((packed_out, scales_out))
}

/// Pack U8 indices to bit-packed format on GPU (no CPU roundtrip).
///
/// Input: `indices` — flat U8 tensor `[num_vectors * block_size]` with values in 0..2^bits.
/// Returns: packed U8 tensor `[num_vectors * bytes_per_block]`.
pub fn tq_pack_indices(
    indices: &Tensor,
    num_vectors: usize,
    block_size: usize,
    bits: usize,
    device: &Device,
) -> Result<Tensor> {
    let Device::Cuda(dev) = device else {
        candle_core::bail!("tq_pack_indices requires CUDA device");
    };

    let bytes_per_block = block_size * bits / 8;
    let packed_out = Tensor::zeros(num_vectors * bytes_per_block, DType::U8, device)?;

    if num_vectors == 0 {
        return Ok(packed_out);
    }

    let stream = dev.cuda_stream().cu_stream() as _;

    {
        let Storage::Cuda(idx_cuda) = &*indices.storage_and_layout().0 else {
            candle_core::bail!("indices must be on CUDA");
        };
        let Storage::Cuda(out_cuda) = &*packed_out.storage_and_layout().0 else {
            candle_core::bail!("packed_out must be on CUDA");
        };

        let idx_slice = idx_cuda.as_cuda_slice::<u8>()?;
        let out_slice = out_cuda.as_cuda_slice::<u8>()?;

        let (idx_ptr, _g1) = idx_slice.device_ptr(idx_slice.stream());
        let (out_ptr, _g2) = out_slice.device_ptr(out_slice.stream());

        unsafe {
            ffi::tq_pack_indices(
                idx_ptr as *const u8,
                out_ptr as *mut u8,
                num_vectors as i32,
                block_size as i32,
                bits as i32,
                bytes_per_block as i32,
                stream,
            );
        }
    }

    Ok(packed_out)
}

/// Launch the TurboQuant QJL sign kernel on GPU.
///
/// Computes residual between original and dequantized vectors, then
/// projects through Rademacher matrix to generate QJL sign bits.
/// Returns `(qjl_signs, residual_norms)` where norms are `DType::F16`.
pub fn tq_qjl_batch(
    original: &Tensor,
    dequantized: &Tensor,
    qjl_seed: u64,
    num_blocks: usize,
    head_dim: usize,
    signs_per_block: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let Device::Cuda(dev) = device else {
        candle_core::bail!("tq_qjl_batch requires CUDA device");
    };

    let signs_out = Tensor::zeros(num_blocks * signs_per_block, DType::U8, device)?;
    let norms_out = Tensor::zeros(num_blocks, DType::F16, device)?;

    if num_blocks == 0 {
        return Ok((signs_out, norms_out));
    }

    let stream = dev.cuda_stream().cu_stream() as _;

    {
        let Storage::Cuda(original_cuda) = &*original.storage_and_layout().0 else {
            candle_core::bail!("original must be on CUDA");
        };
        let Storage::Cuda(dequantized_cuda) = &*dequantized.storage_and_layout().0 else {
            candle_core::bail!("dequantized must be on CUDA");
        };
        let Storage::Cuda(signs_cuda) = &*signs_out.storage_and_layout().0 else {
            candle_core::bail!("signs_out must be on CUDA");
        };
        let Storage::Cuda(norms_cuda) = &*norms_out.storage_and_layout().0 else {
            candle_core::bail!("norms_out must be on CUDA");
        };

        let original_slice = original_cuda.as_cuda_slice::<f32>()?;
        let dequantized_slice = dequantized_cuda.as_cuda_slice::<f32>()?;
        let signs_slice = signs_cuda.as_cuda_slice::<u8>()?;
        let norms_slice = norms_cuda.as_cuda_slice::<half::f16>()?;

        let (original_ptr, _g1) = original_slice.device_ptr(original_slice.stream());
        let (dequantized_ptr, _g2) = dequantized_slice.device_ptr(dequantized_slice.stream());
        let (signs_ptr, _g3) = signs_slice.device_ptr(signs_slice.stream());
        let (norms_ptr, _g4) = norms_slice.device_ptr(norms_slice.stream());

        unsafe {
            ffi::tq_qjl_batch(
                original_ptr as *const f32,
                dequantized_ptr as *const f32,
                qjl_seed,
                signs_ptr as *mut u8,
                norms_ptr as *mut u16,
                num_blocks as i32,
                head_dim as i32,
                signs_per_block as i32,
                stream,
            );
        }
    }

    Ok((signs_out, norms_out))
}
