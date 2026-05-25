use std::ffi::CString;

use llama_rs_sys as ffi;

use crate::ggml_runtime::weights::GgmlWeights;
use crate::{Error, Result};

const LATENT: i64 = 256;
const PATCH_CHANNELS: usize = 512;
const PATCHES_PER_LATENT: usize = 16;
const DIM: i64 = 768;
const HEADS: i64 = 12;
const HEAD_DIM: i64 = 64;
const ROPE_DIMS: i32 = 32;
const WINDOW: usize = 34;
const SUB_CHUNK: usize = 17;
const SHIFT: usize = 17;
const BLOCKS: usize = 6;
const FF_INNER: i64 = 2304;

impl GgmlWeights {
    pub fn encode_same_s(&mut self, audio_patches: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        validate_audio_patches(audio_patches, t_lat)?;
        if t_lat == 0 || t_lat % 2 != 0 {
            return Err(Error::Ggml(
                "SAME-S encoder requires a positive even latent length".into(),
            ));
        }

        let patch_t = t_lat * PATCHES_PER_LATENT;
        let mapped = self.encoder_mapping(audio_patches, patch_t)?;
        let new_tokens = self.tensor_f32("encoder.ly.0.new_tokens")?;

        let internal_t = t_lat * SUB_CHUNK;
        let mut seq = vec![0.0f32; DIM as usize * internal_t];
        for latent_idx in 0..t_lat {
            for sub in 0..PATCHES_PER_LATENT {
                copy_token(
                    &mapped,
                    patch_t,
                    latent_idx * PATCHES_PER_LATENT + sub,
                    &mut seq,
                    internal_t,
                    latent_idx * SUB_CHUNK + sub,
                );
            }
            copy_token(
                &new_tokens,
                1,
                0,
                &mut seq,
                internal_t,
                latent_idx * SUB_CHUNK + PATCHES_PER_LATENT,
            );
        }

        let mut first = vec![0.0f32; seq.len()];
        for (chunk_idx, chunk) in seq.chunks_exact(DIM as usize * WINDOW).enumerate() {
            let out = self.encoder_window(chunk, 0, BLOCKS / 2)?;
            first[chunk_idx * DIM as usize * WINDOW..(chunk_idx + 1) * DIM as usize * WINDOW]
                .copy_from_slice(&out);
        }

        let shifted_t = internal_t + 2 * SHIFT;
        let mut shifted = vec![0.0f32; DIM as usize * shifted_t];
        for token in 0..SHIFT {
            copy_token(&first, internal_t, token, &mut shifted, shifted_t, token);
        }
        for token in 0..internal_t {
            copy_token(
                &first,
                internal_t,
                token,
                &mut shifted,
                shifted_t,
                token + SHIFT,
            );
        }
        for token in 0..SHIFT {
            copy_token(
                &first,
                internal_t,
                internal_t - SHIFT + token,
                &mut shifted,
                shifted_t,
                SHIFT + internal_t + token,
            );
        }

        let mut second = vec![0.0f32; shifted.len()];
        for (chunk_idx, chunk) in shifted.chunks_exact(DIM as usize * WINDOW).enumerate() {
            let out = self.encoder_window(chunk, BLOCKS / 2, BLOCKS)?;
            second[chunk_idx * DIM as usize * WINDOW..(chunk_idx + 1) * DIM as usize * WINDOW]
                .copy_from_slice(&out);
        }

        let mut cropped = vec![0.0f32; DIM as usize * internal_t];
        for token in 0..internal_t {
            copy_token(
                &second,
                shifted_t,
                token + SHIFT,
                &mut cropped,
                internal_t,
                token,
            );
        }

        let mut latent_tokens = vec![0.0f32; DIM as usize * t_lat];
        for latent_idx in 0..t_lat {
            copy_token(
                &cropped,
                internal_t,
                latent_idx * SUB_CHUNK + PATCHES_PER_LATENT,
                &mut latent_tokens,
                t_lat,
                latent_idx,
            );
        }

        let mut latents = self.encoder_project_out(&latent_tokens, t_lat)?;
        let scaling = self.tensor_f32("bn.scaling_factor")?;
        let bias = self.tensor_f32("bn.bias")?;
        let running_std = self.tensor_f32("bn.running_std")?;
        let std = running_std.first().copied().unwrap_or(1.0);
        for t in 0..t_lat {
            for ch in 0..LATENT as usize {
                let idx = ch + t * LATENT as usize;
                latents[idx] = (latents[idx] * scaling[ch] + bias[ch]) / std;
            }
        }
        Ok(latents)
    }

    pub fn encode_same_l(&mut self, audio_patches: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        validate_audio_patches(audio_patches, t_lat)?;
        Err(Error::Incomplete(
            "SAME-L encoder ggml runtime is not implemented yet".into(),
        ))
    }

    fn encoder_mapping(&mut self, audio_patches: &[f32], patch_t: usize) -> Result<Vec<f32>> {
        let x_name = CString::new("same_s_encoder_audio").unwrap();
        let out_name = CString::new("same_s_encoder_mapped").unwrap();
        let mut input = vec![0.0f32; audio_patches.len()];
        for t in 0..patch_t {
            for ch in 0..PATCH_CHANNELS {
                input[t + ch * patch_t] = audio_patches[ch + t * PATCH_CHANNELS];
            }
        }
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 4096, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to create SAME-S encoder mapping graph".into(),
                ));
            }
            let x = ffi::ggml_new_tensor_2d(
                ctx0,
                ffi::ggml_type_GGML_TYPE_F32,
                patch_t as i64,
                PATCH_CHANNELS as i64,
            );
            ffi::ggml_set_name(x, x_name.as_ptr());
            ffi::ggml_set_input(x);
            let mut h =
                ffi::ggml_conv_1d_ph(ctx0, self.tensor("encoder.ly.0.map.weight")?, x, 1, 0);
            let bias = ffi::ggml_reshape_2d(ctx0, self.tensor("encoder.ly.0.map.bias")?, 1, DIM);
            h = ffi::ggml_add(ctx0, h, bias);
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            let raw = self.compute_encoder_single_input_graph(
                ctx0,
                gf,
                &x_name,
                &input,
                &out_name,
                DIM as usize * patch_t,
            )?;
            let mut mapped = vec![0.0f32; raw.len()];
            for t in 0..patch_t {
                for ch in 0..DIM as usize {
                    mapped[ch + t * DIM as usize] = raw[t + ch * patch_t];
                }
            }
            Ok(mapped)
        }
    }

    fn encoder_project_out(&mut self, tokens: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        let x_name = CString::new("same_s_encoder_tokens").unwrap();
        let out_name = CString::new("same_s_encoder_latents").unwrap();
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 4096, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to create SAME-S encoder projection graph".into(),
                ));
            }
            let x = ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, DIM, t_lat as i64);
            ffi::ggml_set_name(x, x_name.as_ptr());
            ffi::ggml_set_input(x);
            let mut h = ffi::ggml_mul_mat(ctx0, self.tensor("encoder.ly.2.weight")?, x);
            h = ffi::ggml_add(ctx0, h, self.tensor("encoder.ly.2.bias")?);
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            self.compute_encoder_single_input_graph(
                ctx0,
                gf,
                &x_name,
                tokens,
                &out_name,
                LATENT as usize * t_lat,
            )
        }
    }

    fn encoder_window(
        &mut self,
        input: &[f32],
        start_block: usize,
        end_block: usize,
    ) -> Result<Vec<f32>> {
        let x_name = CString::new("same_s_encoder_window").unwrap();
        let pos_name = CString::new("same_s_encoder_pos").unwrap();
        let out_name = CString::new("same_s_encoder_window_out").unwrap();
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 16384, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to create SAME-S encoder window graph".into(),
                ));
            }
            let mut h =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, DIM, WINDOW as i64);
            ffi::ggml_set_name(h, x_name.as_ptr());
            ffi::ggml_set_input(h);
            let pos = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, WINDOW as i64);
            ffi::ggml_set_name(pos, pos_name.as_ptr());
            ffi::ggml_set_input(pos);
            for block in start_block..end_block {
                h = self.encoder_block(ctx0, block, h, pos)?;
            }
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            if !ffi::ggml_backend_sched_alloc_graph(self.scheduler(), gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to allocate SAME-S encoder window graph".into(),
                ));
            }
            set_input(gf, &x_name, input);
            let positions = (0..WINDOW as i32).collect::<Vec<_>>();
            let pos_graph = ffi::ggml_graph_get_tensor(gf, pos_name.as_ptr());
            ffi::ggml_backend_tensor_set(
                pos_graph,
                positions.as_ptr().cast(),
                0,
                std::mem::size_of_val(positions.as_slice()),
            );
            let status = ffi::ggml_backend_sched_graph_compute(self.scheduler(), gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.scheduler());
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "SAME-S encoder window compute failed: status={status}"
                )));
            }
            let out = ffi::ggml_graph_get_tensor(gf, out_name.as_ptr());
            let mut data = vec![0.0f32; DIM as usize * WINDOW];
            ffi::ggml_backend_tensor_get(
                out,
                data.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(data.as_slice()),
            );
            ffi::ggml_backend_sched_reset(self.scheduler());
            ffi::ggml_free(ctx0);
            Ok(data)
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn encoder_block(
        &self,
        ctx: *mut ffi::ggml_context,
        block: usize,
        x: *mut ffi::ggml_tensor,
        pos: *mut ffi::ggml_tensor,
    ) -> Result<*mut ffi::ggml_tensor> {
        let p = format!("encoder.ly.0.tr.{block}.");
        let residual = x;
        let h = self.encoder_dyt(ctx, x, &(p.to_owned() + "pn."))?;
        let qkv = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "sa.qkv.weight"))?, h);
        let row_bytes = (*qkv).nb[1];
        let mut q = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, DIM, WINDOW as i64, row_bytes, 0),
        );
        let mut k = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, DIM, WINDOW as i64, row_bytes, DIM as usize * 4),
        );
        let mut v = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, DIM, WINDOW as i64, row_bytes, DIM as usize * 8),
        );
        let mut qd = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, DIM, WINDOW as i64, row_bytes, DIM as usize * 12),
        );
        let mut kd = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, DIM, WINDOW as i64, row_bytes, DIM as usize * 16),
        );
        q = ffi::ggml_reshape_3d(ctx, q, HEAD_DIM, HEADS, WINDOW as i64);
        k = ffi::ggml_reshape_3d(ctx, k, HEAD_DIM, HEADS, WINDOW as i64);
        v = ffi::ggml_reshape_3d(ctx, v, HEAD_DIM, HEADS, WINDOW as i64);
        qd = ffi::ggml_reshape_3d(ctx, qd, HEAD_DIM, HEADS, WINDOW as i64);
        kd = ffi::ggml_reshape_3d(ctx, kd, HEAD_DIM, HEADS, WINDOW as i64);
        q = self.encoder_dyt(ctx, q, &(p.to_owned() + "sa.qn."))?;
        k = self.encoder_dyt(ctx, k, &(p.to_owned() + "sa.kn."))?;
        qd = self.encoder_dyt(ctx, qd, &(p.to_owned() + "sa.qn."))?;
        kd = self.encoder_dyt(ctx, kd, &(p.to_owned() + "sa.kn."))?;
        q = encoder_rope(ctx, q, pos);
        k = encoder_rope(ctx, k, pos);
        qd = encoder_rope(ctx, qd, pos);
        kd = encoder_rope(ctx, kd, pos);
        let main = encoder_attention(ctx, q, k, v);
        let diff = encoder_attention(ctx, qd, kd, v);
        let mut attn = ffi::ggml_sub(ctx, main, diff);
        attn = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "sa.out.weight"))?, attn);
        let mut x = ffi::ggml_add(ctx, residual, attn);

        let residual = x;
        let mut h = self.encoder_dyt(ctx, x, &(p.to_owned() + "fn."))?;
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ff0.weight"))?, h);
        h = ffi::ggml_add(ctx, h, self.tensor(&(p.to_owned() + "ff0.bias"))?);
        let row_bytes = (*h).nb[1];
        let value = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, h, FF_INNER, WINDOW as i64, row_bytes, 0),
        );
        let gate = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(
                ctx,
                h,
                FF_INNER,
                WINDOW as i64,
                row_bytes,
                FF_INNER as usize * 4,
            ),
        );
        h = ffi::ggml_mul(ctx, value, ffi::ggml_silu(ctx, gate));
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ff2.weight"))?, h);
        h = ffi::ggml_add(ctx, h, self.tensor(&(p.to_owned() + "ff2.bias"))?);
        x = ffi::ggml_add(ctx, residual, h);
        Ok(x)
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn encoder_dyt(
        &self,
        ctx: *mut ffi::ggml_context,
        x: *mut ffi::ggml_tensor,
        prefix: &str,
    ) -> Result<*mut ffi::ggml_tensor> {
        let mut h = ffi::ggml_mul(ctx, x, self.tensor(&(prefix.to_owned() + "alpha"))?);
        h = ffi::ggml_tanh(ctx, h);
        h = ffi::ggml_mul(ctx, h, self.tensor(&(prefix.to_owned() + "gamma"))?);
        h = ffi::ggml_add(ctx, h, self.tensor(&(prefix.to_owned() + "beta"))?);
        Ok(h)
    }

    unsafe fn compute_encoder_single_input_graph(
        &mut self,
        ctx0: *mut ffi::ggml_context,
        gf: *mut ffi::ggml_cgraph,
        input_name: &CString,
        input: &[f32],
        output_name: &CString,
        output_len: usize,
    ) -> Result<Vec<f32>> {
        unsafe {
            if !ffi::ggml_backend_sched_alloc_graph(self.scheduler(), gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to allocate SAME-S encoder graph".into(),
                ));
            }
            set_input(gf, input_name, input);
            let status = ffi::ggml_backend_sched_graph_compute(self.scheduler(), gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.scheduler());
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "SAME-S encoder compute failed: status={status}"
                )));
            }
            let out = ffi::ggml_graph_get_tensor(gf, output_name.as_ptr());
            let mut data = vec![0.0f32; output_len];
            ffi::ggml_backend_tensor_get(
                out,
                data.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(data.as_slice()),
            );
            ffi::ggml_backend_sched_reset(self.scheduler());
            ffi::ggml_free(ctx0);
            Ok(data)
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn encoder_attention(
    ctx: *mut ffi::ggml_context,
    q: *mut ffi::ggml_tensor,
    k: *mut ffi::ggml_tensor,
    mut v: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    let q = ffi::ggml_permute(ctx, q, 0, 2, 1, 3);
    let k = ffi::ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ffi::ggml_permute(ctx, v, 0, 2, 1, 3);
    let mut kq = ffi::ggml_mul_mat(ctx, k, q);
    kq = ffi::ggml_scale(ctx, kq, 1.0 / (HEAD_DIM as f32).sqrt());
    kq = ffi::ggml_soft_max(ctx, kq);
    v = ffi::ggml_cont(ctx, ffi::ggml_transpose(ctx, v));
    let mut attn = ffi::ggml_mul_mat(ctx, v, kq);
    attn = ffi::ggml_permute(ctx, attn, 0, 2, 1, 3);
    ffi::ggml_cont_2d(ctx, attn, DIM, WINDOW as i64)
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn encoder_rope(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    pos: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    ffi::ggml_rope_ext(
        ctx,
        x,
        pos,
        std::ptr::null_mut(),
        ROPE_DIMS,
        ffi::GGML_ROPE_TYPE_NEOX as i32,
        0,
        10000.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    )
}

fn validate_audio_patches(audio_patches: &[f32], t_lat: usize) -> Result<()> {
    let expected = PATCH_CHANNELS * PATCHES_PER_LATENT * t_lat;
    if audio_patches.len() != expected {
        return Err(Error::Ggml(format!(
            "SAME encoder patch length mismatch: got {}, expected {}",
            audio_patches.len(),
            expected
        )));
    }
    Ok(())
}

fn copy_token(
    src: &[f32],
    src_t: usize,
    src_idx: usize,
    dst: &mut [f32],
    dst_t: usize,
    dst_idx: usize,
) {
    for ch in 0..DIM as usize {
        dst[ch + dst_idx * DIM as usize] = src[ch + src_idx * DIM as usize];
    }
    let _ = (src_t, dst_t);
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn set_input(gf: *mut ffi::ggml_cgraph, name: &CString, data: &[f32]) {
    let tensor = ffi::ggml_graph_get_tensor(gf, name.as_ptr());
    ffi::ggml_backend_tensor_set(tensor, data.as_ptr().cast(), 0, std::mem::size_of_val(data));
}
