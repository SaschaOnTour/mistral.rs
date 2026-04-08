#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::{
    attention::backends::cpu, kv_cache::KvCache,
    pipeline::text_models_inputs_processor::FlashParams,
};

use candle_core::{DType, Device, Result, Tensor};

mod backends;

#[allow(unused)]
pub(crate) use backends::{flash_attn, maybe_synchronize, naive_sdpa, sinks_attn};

/// Chunk size for attention computation to avoid OOM on long sequences
pub(crate) const ATTENTION_CHUNK_SIZE: usize = 1024;

/// Generic chunked attention computation that can be used by different backends
pub(crate) fn chunked_attention<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>, usize, usize) -> Result<Tensor>,
{
    let seq_len = q.dim(2)?;

    if seq_len <= ATTENTION_CHUNK_SIZE {
        // For short sequences, use the regular path (offset=0, len=seq_len)
        return attention_fn(q, k, v, mask, 0, seq_len);
    }

    // Chunk the query to avoid OOM on long sequences
    let num_chunks = seq_len.div_ceil(ATTENTION_CHUNK_SIZE);
    let mut attn_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let offset = chunk_idx * ATTENTION_CHUNK_SIZE;
        let chunk_len = ATTENTION_CHUNK_SIZE.min(seq_len - offset);

        // Extract query chunk
        let q_chunk = q.narrow(2, offset, chunk_len)?;

        // Extract mask chunk if present.
        // Small broadcast masks (e.g. [1,1]) are passed through unchanged.
        let mask_chunk = mask
            .map(|m| {
                let narrow_dim = match m.rank() {
                    2 => 0,
                    3 => 1,
                    _ => 2,
                };
                if m.dim(narrow_dim)? >= offset + chunk_len {
                    m.narrow(narrow_dim, offset, chunk_len)
                } else {
                    // Mask is smaller than chunk (broadcast mask) — pass through
                    Ok(m.clone())
                }
            })
            .transpose()?;

        // Compute attention for this chunk
        let att_chunk = attention_fn(&q_chunk, k, v, mask_chunk.as_ref(), offset, chunk_len)?;

        attn_chunks.push(att_chunk);
    }

    // Concatenate all chunks along the sequence dimension
    Tensor::cat(&attn_chunks, 2)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
    pub sinks: Option<Tensor>,
    /// QJL bias correction for TurboQuant attention (Paper Algorithm 2).
    /// Shape: `[batch, num_heads_or_1, q_len, kv_len]`.
    /// Added to attention logits before softmax. `None` for non-TurboQuant.
    pub qjl_bias: Option<Tensor>,
}

impl SdpaParams {
    /// Returns a new `SdpaParams` with the QJL bias set, keeping all other fields.
    ///
    /// Use in model attention layers:
    /// ```ignore
    /// let sdpa_params = self.sdpa_params.with_qjl(kv_cache.qjl_bias(&q)?);
    /// Sdpa.run_attention(&q, &k, &v, mask, flash_params, &sdpa_params)?
    /// ```
    pub fn with_qjl(&self, qjl_bias: Option<Tensor>) -> Self {
        Self {
            n_kv_groups: self.n_kv_groups,
            softcap: self.softcap,
            softmax_scale: self.softmax_scale,
            sliding_window: self.sliding_window,
            sinks: self.sinks.clone(),
            qjl_bias,
        }
    }
}

pub struct Sdpa;

impl Sdpa {
    /// Computes softmax(QK^T*sqrt(d_k))V
    ///
    /// Inputs:
    /// - q: (b_sz, n_attn_heads, q_len, head_dim)
    /// - k: (b_sz, n_kv_heads, q_len, head_dim)
    /// - v: (b_sz, n_kv_heads, q_len, head_dim)
    ///
    /// The attention implementation is dispatched as follows:
    /// 1) If using flash attn (CUDA), use a flash attention V2/V3 kernel
    /// 2) If decoding and using a Metal device, use a fused kkernel
    /// 2) Otherwise, use the "naive" SDPA implementation (with optimized mask+softmax+scale application)
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        // If sinks are present, dispatch to the sinks backend
        if let Some(sinks) = &sdpa_params.sinks {
            return sinks_attn(q, k, v, sinks, mask, flash_params, sdpa_params);
        }

        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // QJL bias correction requires naive SDPA — flash attention kernels
        // cannot inject a custom additive bias on attention logits.
        let can_use_flash = sdpa_params.qjl_bias.is_none()
            && (q.device().is_cpu()
                || q.device().is_cuda() && crate::using_flash_attn() && q.dtype() != DType::F32);

        if can_use_flash {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            if q.device().is_cpu() {
                match q.dtype() {
                    DType::F32 => {
                        return cpu::run_flash_attn_cpu::<f32>(&q, &k, &v, mask, sdpa_params);
                    }
                    DType::F16 => {
                        return cpu::run_flash_attn_cpu::<half::f16>(&q, &k, &v, mask, sdpa_params)
                    }
                    DType::BF16 => {
                        return cpu::run_flash_attn_cpu::<half::bf16>(
                            &q,
                            &k,
                            &v,
                            mask,
                            sdpa_params,
                        );
                    }
                    _ => {
                        return Err(candle_core::Error::Msg("Unsupported data type".into()));
                    }
                }
            } else {
                return flash_attn(&q, &k, &v, flash_params, sdpa_params)?.transpose(1, 2);
            }
        }

        self.run_attention_noflash(q, k, v, mask, sdpa_params)
    }

    /// Same as `run_attention`, but no flash attention
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention_noflash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // We can use Metal SDPA (vector/full) if the mask is the correct size and head dims match.
        // If the mask is provided, then softcapping isn't allowed - default back to naive SDPA
        // Softcapping is implemented for vector SDPA.
        let all_head_dims_match = head_dim == k_head_dim && k_head_dim == v_head_dim;
        let tgt_mask_shape = vec![b_sz, n_attn_heads, seq_len, k.dim(2)?];
        let can_use_mask = mask.is_none_or(|mask| {
            mask.layout().broadcast_as(tgt_mask_shape.clone()).is_ok()
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        });
        let valid_head_dims: &[usize] = if seq_len == 1 {
            &[32, 64, 72, 80, 96, 128, 256]
        } else {
            // Not sure why the full kernel doesn't like 256.
            // [32, 64, 72, 80, 96, 128, 256]
            &[32, 64, 72, 80, 96, 128]
        };
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && all_head_dims_match
            && valid_head_dims.contains(&head_dim)
            && can_use_mask
        {
            let mask = match mask {
                Some(mask) => Some(mask.broadcast_as(tgt_mask_shape)?),
                None => None,
            };
            return candle_nn::ops::sdpa(
                q,
                k,
                v,
                mask.as_ref(),
                false,
                sdpa_params.softmax_scale,
                sdpa_params.softcap.unwrap_or(1.0),
            );
        }

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

        if mask.is_some_and(|x| x.rank() == 2) || mistralrs_quant::distributed::use_nccl() {
            return naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
            );
        }

        // QJL bias requires naive SDPA — cuBLASLt path doesn't support additive bias.
        if sdpa_params.qjl_bias.is_some() {
            return naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
            );
        }

        // TODO: bench?
        #[allow(unused)]
        if let (Device::Cuda(_), Some(cublaslt)) = (
            q.device(),
            mistralrs_quant::cublaslt::CUBLASLT_CONTROLLER.get_for_device(q.device()),
        ) {
            #[cfg(feature = "cuda")]
            {
                maybe_synchronize(q.device())?;

                // Use chunked attention for cuBLASLt path
                let k_flat = k.flatten(0, 1)?;
                let v_flat = v.flatten(0, 1)?;

                chunked_attention(
                    q,
                    &k,
                    &v,
                    mask,
                    |q_chunk, _k, _v, mask_chunk, _offset, _len| {
                        // cuBLASLt batch matmul implementation requires inputs to be dims3
                        let (chunk_b_sz, chunk_n_heads, chunk_seq_len, chunk_head_dim) =
                            q_chunk.dims4()?;
                        let q_flat = q_chunk.flatten(0, 1)?;

                        let attention_bias = match mask_chunk {
                            Some(mask) if mask.rank() == 3 && mask.dims()[0] == 1 => {
                                Some(mask.repeat((chunk_n_heads, 1, 1))?)
                            }
                            Some(mask) if mask.rank() == 3 => Some(mask.clone()),
                            Some(mask) if mask.rank() == 4 => {
                                let tgt_shape =
                                    vec![chunk_b_sz, chunk_n_heads, chunk_seq_len, k.dim(2)?];
                                Some(mask.broadcast_as(tgt_shape)?.flatten(0, 1)?)
                            }
                            Some(mask) => {
                                candle_core::bail!("cublaslt attn mask: rank must be 3 or 4")
                            }
                            None => None,
                        };

                        // If attention_bias is set, we fuse the add by giving it as the output matrix
                        // and setting beta to 1.0
                        let beta = match attention_bias.is_some() {
                            true => Some(1.0),
                            false => None,
                        };

                        // Batch matrix multiplication
                        // Fuse softmax scale and attention_bias add
                        let mut attention_scores = cublaslt.batch_matmul(
                            &k_flat,
                            &q_flat,
                            attention_bias.as_ref(),
                            Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0)),
                            beta,
                            None,
                            None,
                        )?;
                        if let Some(softcap) = sdpa_params.softcap {
                            attention_scores = (attention_scores.tanh()? * softcap as f64)?;
                        }
                        attention_scores = candle_nn::ops::softmax_last_dim(&attention_scores)?;

                        let context_layer = cublaslt.batch_matmul(
                            &v_flat.t()?.contiguous()?,
                            &attention_scores,
                            // We save one allocation
                            Some(&q_flat),
                            None,
                            None,
                            None,
                            None,
                        )?;

                        // Reshape to dims4
                        context_layer.reshape((
                            chunk_b_sz,
                            chunk_n_heads,
                            chunk_seq_len,
                            v_head_dim,
                        ))
                    },
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::bail!("`cuda` feature is not enabled")
            }
        } else {
            naive_sdpa(q, &k, &v, mask, sdpa_params)
        }
    }
}

/// Append K/V to cache and compute attention in one step.
///
/// For TurboQuant decode (single-token generation with existing cache):
/// uses block-wise attention directly from compressed data — no full
/// dequantization, O(block_size) memory instead of O(seq_len).
///
/// For everything else (Normal, Rotating, TurboQuant prefill):
/// delegates to the standard `append()` + `Sdpa.run_attention()` path.
///
/// This function is the single integration point for all models — changing
/// the attention strategy here affects all 28+ model implementations.
pub fn cached_attention(
    kv_cache: &mut KvCache,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
    flash_params: Option<&FlashParams>,
) -> Result<Tensor> {
    // TurboQuant: blockwise decode on CUDA (saves VRAM), full-dequant on CPU/Metal
    // (CPU/Metal have optimized flash attention that beats our generic online softmax)
    let is_decode = k.dim(2)? == 1;
    if let KvCache::TurboQuant { cache, layer, .. } = kv_cache {
        let mut guard = cache
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("TurboQuant lock: {e}")))?;
        if is_decode && guard.is_decode_ready(*layer) && k.device().is_cuda() {
            return guard.append_and_blockwise_attend(
                *layer,
                k,
                v,
                q,
                sdpa_params.softmax_scale,
                sdpa_params.n_kv_groups,
            );
        }
        // CPU/Metal decode, prefill, or first token: full-dequant + SDPA
        let (k_out, v_out) = guard.append_and_dequantize(*layer, k, v)?;
        drop(guard); // release lock before qjl_bias (which re-locks)
        let qjl_bias = kv_cache.qjl_bias(q)?;
        let sdpa_params = sdpa_params.with_qjl(qjl_bias);
        return Sdpa.run_attention(q, &k_out, &v_out, mask, flash_params, &sdpa_params);
    }

    // Normal / Rotating cache: standard path
    let (k_out, v_out) = kv_cache.append(k, v)?;
    let qjl_bias = kv_cache.qjl_bias(q)?;
    let sdpa_params = sdpa_params.with_qjl(qjl_bias);
    Sdpa.run_attention(q, &k_out, &v_out, mask, flash_params, &sdpa_params)
}
