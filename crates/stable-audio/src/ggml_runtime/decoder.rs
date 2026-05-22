use std::ffi::CString;

use llama_rs_sys as ffi;

use crate::ggml_runtime::weights::GgmlWeights;
use crate::{Error, Result};

const LATENT: i64 = 256;
const DIM: i64 = 768;
const HEADS: i64 = 12;
const HEAD_DIM: i64 = 64;
const ROPE_DIMS: i32 = 32;
const WINDOW: usize = 34;
const SUB_CHUNK: usize = 17;
const SIN_PER_POS: usize = 16;
const OUT_CHANNELS: usize = 512;
const L_DIM: i64 = 1536;
const L_HEADS: i64 = 24;
const L_BLOCKS: usize = 12;
const L_FF_INNER: i64 = 4608;
const L_SIN_START_BLOCK: usize = 5;

impl GgmlWeights {
    pub fn decode_same_s(&mut self, latents: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        if latents.len() != LATENT as usize * t_lat {
            return Err(Error::Ggml("SAME-S latent length mismatch".into()));
        }
        if t_lat == 0 || t_lat % 2 != 0 {
            return Err(Error::Ggml(
                "SAME-S requires a positive even latent length".into(),
            ));
        }

        let running_std = self.tensor_f32("bn.running_std")?;
        let std = running_std.first().copied().unwrap_or(1.0);
        let scaled = latents.iter().map(|v| v * std).collect::<Vec<_>>();
        let projected = self.decoder_project_in(&scaled, t_lat)?;
        let new_tokens = self.tensor_f32("decoder.ly.3.new_tokens")?;

        let internal_t = t_lat * SUB_CHUNK;
        let mut seq = vec![0.0f32; DIM as usize * internal_t];
        for latent_idx in 0..t_lat {
            let dst_token = latent_idx * SUB_CHUNK;
            copy_token(
                &projected, t_lat, latent_idx, &mut seq, internal_t, dst_token,
            );
            for sub in 1..SUB_CHUNK {
                copy_token(&new_tokens, 1, 0, &mut seq, internal_t, dst_token + sub);
            }
        }

        let mut first = vec![0.0f32; seq.len()];
        for (chunk_idx, chunk) in seq.chunks_exact(DIM as usize * WINDOW).enumerate() {
            let out = self.decoder_window(chunk, 0, 3)?;
            first[chunk_idx * DIM as usize * WINDOW..(chunk_idx + 1) * DIM as usize * WINDOW]
                .copy_from_slice(&out);
        }

        let shifted_t = internal_t + WINDOW;
        let mut shifted = vec![0.0f32; DIM as usize * shifted_t];
        for token in 0..SUB_CHUNK {
            copy_token(&first, internal_t, token, &mut shifted, shifted_t, token);
        }
        for token in 0..internal_t {
            copy_token(
                &first,
                internal_t,
                token,
                &mut shifted,
                shifted_t,
                token + SUB_CHUNK,
            );
        }
        for token in 0..SUB_CHUNK {
            copy_token(
                &first,
                internal_t,
                internal_t - SUB_CHUNK + token,
                &mut shifted,
                shifted_t,
                SUB_CHUNK + internal_t + token,
            );
        }

        let mut second = vec![0.0f32; shifted.len()];
        for (chunk_idx, chunk) in shifted.chunks_exact(DIM as usize * WINDOW).enumerate() {
            let out = self.decoder_window(chunk, 3, 6)?;
            second[chunk_idx * DIM as usize * WINDOW..(chunk_idx + 1) * DIM as usize * WINDOW]
                .copy_from_slice(&out);
        }

        let mut cropped = vec![0.0f32; DIM as usize * internal_t];
        for token in 0..internal_t {
            copy_token(
                &second,
                shifted_t,
                token + SUB_CHUNK,
                &mut cropped,
                internal_t,
                token,
            );
        }

        let patch_t = t_lat * SIN_PER_POS;
        let mut tokens = vec![0.0f32; DIM as usize * patch_t];
        for latent_idx in 0..t_lat {
            for sub in 0..SIN_PER_POS {
                copy_token(
                    &cropped,
                    internal_t,
                    latent_idx * SUB_CHUNK + 1 + sub,
                    &mut tokens,
                    patch_t,
                    latent_idx * SIN_PER_POS + sub,
                );
            }
        }
        self.decoder_mapping(&tokens, patch_t)
    }

    pub fn decode_same_l(&mut self, latents: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        if latents.len() != LATENT as usize * t_lat {
            return Err(Error::Ggml("SAME-L latent length mismatch".into()));
        }
        if t_lat == 0 {
            return Err(Error::Ggml(
                "SAME-L requires a positive latent length".into(),
            ));
        }
        if t_lat > 6 {
            return self.decode_same_l_chunked(latents, t_lat, 2, 2);
        }

        self.decode_same_l_inner(latents, t_lat)
    }

    fn decode_same_l_inner(&mut self, latents: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        let running_std = self.tensor_f32("bn.running_std")?;
        let std = running_std.first().copied().unwrap_or(1.0);
        let scaled = latents.iter().map(|v| v * std).collect::<Vec<_>>();
        let projected = self.decoder_project_in_l(&scaled, t_lat)?;
        let new_tokens = self.tensor_f32("decoder.ly.3.new_tokens")?;

        let internal_t = t_lat * SUB_CHUNK;
        let mut seq = vec![0.0f32; L_DIM as usize * internal_t];
        for latent_idx in 0..t_lat {
            let dst_token = latent_idx * SUB_CHUNK;
            copy_token_dim(&projected, L_DIM as usize, latent_idx, &mut seq, dst_token);
            for sub in 1..SUB_CHUNK {
                copy_token_dim(&new_tokens, L_DIM as usize, 0, &mut seq, dst_token + sub);
            }
        }

        let decoded = self.decoder_same_l_sequence(&seq, internal_t)?;
        let patch_t = t_lat * SIN_PER_POS;
        let mut tokens = vec![0.0f32; L_DIM as usize * patch_t];
        for latent_idx in 0..t_lat {
            for sub in 0..SIN_PER_POS {
                copy_token_dim(
                    &decoded,
                    L_DIM as usize,
                    latent_idx * SUB_CHUNK + 1 + sub,
                    &mut tokens,
                    latent_idx * SIN_PER_POS + sub,
                );
            }
        }
        self.decoder_mapping_l(&tokens, patch_t)
    }

    fn decode_same_l_chunked(
        &mut self,
        latents: &[f32],
        t_lat: usize,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Vec<f32>> {
        let kernel = chunk_size + 2 * overlap;
        if t_lat <= kernel {
            return self.decode_same_l_inner(latents, t_lat);
        }

        let mut pieces = Vec::new();
        let first = self.decode_same_l_inner(&latents[..LATENT as usize * kernel], kernel)?;
        let valid_first = chunk_size + overlap;
        pieces.extend_from_slice(&first[..valid_first * SIN_PER_POS * OUT_CHANNELS]);
        let mut i = valid_first;

        while i + chunk_size + overlap <= t_lat {
            let start = i - overlap;
            let end = i + chunk_size + overlap;
            let out = self.decode_same_l_inner(
                &latents[start * LATENT as usize..end * LATENT as usize],
                end - start,
            )?;
            let keep_start = overlap * SIN_PER_POS * OUT_CHANNELS;
            let keep_end = (overlap + chunk_size) * SIN_PER_POS * OUT_CHANNELS;
            pieces.extend_from_slice(&out[keep_start..keep_end]);
            i += chunk_size;
        }

        let remaining = t_lat - i;
        if remaining > 0 {
            let start = t_lat - kernel;
            let out = self.decode_same_l_inner(
                &latents[start * LATENT as usize..t_lat * LATENT as usize],
                kernel,
            )?;
            let keep = remaining * SIN_PER_POS * OUT_CHANNELS;
            pieces.extend_from_slice(&out[out.len() - keep..]);
        }
        Ok(pieces)
    }

    fn decoder_project_in(&mut self, latents: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        let x_name = CString::new("same_s_latents").unwrap();
        let out_name = CString::new("same_s_projected").unwrap();
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 4096, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create SAME-S project graph".into()));
            }
            let x =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, LATENT, t_lat as i64);
            ffi::ggml_set_name(x, x_name.as_ptr());
            ffi::ggml_set_input(x);
            let mut h = ffi::ggml_mul_mat(ctx0, self.tensor("decoder.ly.1.weight")?, x);
            h = ffi::ggml_add(ctx0, h, self.tensor("decoder.ly.1.bias")?);
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            self.compute_single_input_graph(
                ctx0,
                gf,
                &x_name,
                latents,
                &out_name,
                DIM as usize * t_lat,
            )
        }
    }

    fn decoder_project_in_l(&mut self, latents: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        let x_name = CString::new("same_l_latents").unwrap();
        let out_name = CString::new("same_l_projected").unwrap();
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 4096, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create SAME-L project graph".into()));
            }
            let x =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, LATENT, t_lat as i64);
            ffi::ggml_set_name(x, x_name.as_ptr());
            ffi::ggml_set_input(x);
            let mut h = ffi::ggml_mul_mat(ctx0, self.tensor("decoder.ly.1.weight")?, x);
            h = ffi::ggml_add(ctx0, h, self.tensor("decoder.ly.1.bias")?);
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            self.compute_single_input_graph(
                ctx0,
                gf,
                &x_name,
                latents,
                &out_name,
                L_DIM as usize * t_lat,
            )
        }
    }

    fn decoder_same_l_sequence(&mut self, input: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        let x_name = CString::new("same_l_seq").unwrap();
        let pos_name = CString::new("same_l_pos").unwrap();
        let mask_name = CString::new("same_l_mask").unwrap();
        let out_name = CString::new("same_l_seq_out").unwrap();
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 32768, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create SAME-L sequence graph".into()));
            }
            let mut h =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, L_DIM, n_tokens as i64);
            ffi::ggml_set_name(h, x_name.as_ptr());
            ffi::ggml_set_input(h);
            let pos = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, n_tokens as i64);
            ffi::ggml_set_name(pos, pos_name.as_ptr());
            ffi::ggml_set_input(pos);
            let mask = ffi::ggml_new_tensor_2d(
                ctx0,
                ffi::ggml_type_GGML_TYPE_F32,
                n_tokens as i64,
                n_tokens as i64,
            );
            ffi::ggml_set_name(mask, mask_name.as_ptr());
            ffi::ggml_set_input(mask);
            for block in 0..L_BLOCKS {
                h = self.decoder_block_l(ctx0, block, h, pos, mask, n_tokens as i64)?;
            }
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            if !ffi::ggml_backend_sched_alloc_graph(self.scheduler(), gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to allocate SAME-L sequence graph".into(),
                ));
            }
            set_input(gf, &x_name, input);
            let positions = (0..n_tokens as i32).collect::<Vec<_>>();
            let pos_graph = ffi::ggml_graph_get_tensor(gf, pos_name.as_ptr());
            ffi::ggml_backend_tensor_set(
                pos_graph,
                positions.as_ptr().cast(),
                0,
                std::mem::size_of_val(positions.as_slice()),
            );
            let mask_values = same_l_swa_mask(n_tokens);
            set_input(gf, &mask_name, &mask_values);
            let status = ffi::ggml_backend_sched_graph_compute(self.scheduler(), gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.scheduler());
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "SAME-L sequence compute failed: status={status}"
                )));
            }
            let out = ffi::ggml_graph_get_tensor(gf, out_name.as_ptr());
            let mut data = vec![0.0f32; L_DIM as usize * n_tokens];
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

    fn decoder_window(
        &mut self,
        input: &[f32],
        start_block: usize,
        end_block: usize,
    ) -> Result<Vec<f32>> {
        let x_name = CString::new("same_s_window").unwrap();
        let pos_name = CString::new("same_s_pos").unwrap();
        let out_name = CString::new("same_s_window_out").unwrap();
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 16384, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create SAME-S window graph".into()));
            }
            let mut h =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, DIM, WINDOW as i64);
            ffi::ggml_set_name(h, x_name.as_ptr());
            ffi::ggml_set_input(h);
            let pos = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, WINDOW as i64);
            ffi::ggml_set_name(pos, pos_name.as_ptr());
            ffi::ggml_set_input(pos);
            for block in start_block..end_block {
                h = self.decoder_block(ctx0, block, h, pos)?;
            }
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            if !ffi::ggml_backend_sched_alloc_graph(self.scheduler(), gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to allocate SAME-S window graph".into()));
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
                    "SAME-S window compute failed: status={status}"
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

    fn decoder_mapping(&mut self, tokens: &[f32], patch_t: usize) -> Result<Vec<f32>> {
        let x_name = CString::new("same_s_tokens").unwrap();
        let out_name = CString::new("same_s_patches").unwrap();
        let mut tokens_time_major = vec![0.0f32; tokens.len()];
        for t in 0..patch_t {
            for ch in 0..DIM as usize {
                tokens_time_major[t + ch * patch_t] = tokens[ch + t * DIM as usize];
            }
        }
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 4096, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create SAME-S mapping graph".into()));
            }
            let x =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, patch_t as i64, DIM);
            ffi::ggml_set_name(x, x_name.as_ptr());
            ffi::ggml_set_input(x);
            let mut h =
                ffi::ggml_conv_1d_ph(ctx0, self.tensor("decoder.ly.3.map.weight")?, x, 1, 1);
            let bias = ffi::ggml_reshape_2d(
                ctx0,
                self.tensor("decoder.ly.3.map.bias")?,
                1,
                OUT_CHANNELS as i64,
            );
            h = ffi::ggml_add(ctx0, h, bias);
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            let raw = self.compute_single_input_graph(
                ctx0,
                gf,
                &x_name,
                &tokens_time_major,
                &out_name,
                OUT_CHANNELS * patch_t,
            )?;
            let mut patches = vec![0.0f32; raw.len()];
            for t in 0..patch_t {
                for ch in 0..OUT_CHANNELS {
                    patches[ch + t * OUT_CHANNELS] = raw[t + ch * patch_t];
                }
            }
            Ok(patches)
        }
    }

    fn decoder_mapping_l(&mut self, tokens: &[f32], patch_t: usize) -> Result<Vec<f32>> {
        let x_name = CString::new("same_l_tokens").unwrap();
        let out_name = CString::new("same_l_patches").unwrap();
        let mut tokens_time_major = vec![0.0f32; tokens.len()];
        for t in 0..patch_t {
            for ch in 0..L_DIM as usize {
                tokens_time_major[t + ch * patch_t] = tokens[ch + t * L_DIM as usize];
            }
        }
        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 4096, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create SAME-L mapping graph".into()));
            }
            let x =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, patch_t as i64, L_DIM);
            ffi::ggml_set_name(x, x_name.as_ptr());
            ffi::ggml_set_input(x);
            let mut h =
                ffi::ggml_conv_1d_ph(ctx0, self.tensor("decoder.ly.3.map.weight")?, x, 1, 1);
            let bias = ffi::ggml_reshape_2d(
                ctx0,
                self.tensor("decoder.ly.3.map.bias")?,
                1,
                OUT_CHANNELS as i64,
            );
            h = ffi::ggml_add(ctx0, h, bias);
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            let raw = self.compute_single_input_graph(
                ctx0,
                gf,
                &x_name,
                &tokens_time_major,
                &out_name,
                OUT_CHANNELS * patch_t,
            )?;
            let mut patches = vec![0.0f32; raw.len()];
            for t in 0..patch_t {
                for ch in 0..OUT_CHANNELS {
                    patches[ch + t * OUT_CHANNELS] = raw[t + ch * patch_t];
                }
            }
            Ok(patches)
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn decoder_block(
        &self,
        ctx: *mut ffi::ggml_context,
        block: usize,
        x: *mut ffi::ggml_tensor,
        pos: *mut ffi::ggml_tensor,
    ) -> Result<*mut ffi::ggml_tensor> {
        let p = format!("decoder.ly.3.tr.{block}.");
        let residual = x;
        let h = self.decoder_dyt(ctx, x, &(p.to_owned() + "pn."))?;
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
        q = self.decoder_dyt(ctx, q, &(p.to_owned() + "sa.qn."))?;
        k = self.decoder_dyt(ctx, k, &(p.to_owned() + "sa.kn."))?;
        qd = self.decoder_dyt(ctx, qd, &(p.to_owned() + "sa.qn."))?;
        kd = self.decoder_dyt(ctx, kd, &(p.to_owned() + "sa.kn."))?;
        q = decoder_rope(ctx, q, pos);
        k = decoder_rope(ctx, k, pos);
        qd = decoder_rope(ctx, qd, pos);
        kd = decoder_rope(ctx, kd, pos);
        let main = decoder_attention(ctx, q, k, v);
        let diff = decoder_attention(ctx, qd, kd, v);
        let mut attn = ffi::ggml_sub(ctx, main, diff);
        attn = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "sa.out.weight"))?, attn);
        let mut x = ffi::ggml_add(ctx, residual, attn);

        let residual = x;
        let mut h = self.decoder_dyt(ctx, x, &(p.to_owned() + "fn."))?;
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ff0.weight"))?, h);
        h = ffi::ggml_add(ctx, h, self.tensor(&(p.to_owned() + "ff0.bias"))?);
        let row_bytes = (*h).nb[1];
        let value = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, h, 2304, WINDOW as i64, row_bytes, 0),
        );
        let gate = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, h, 2304, WINDOW as i64, row_bytes, 2304 * 4),
        );
        h = ffi::ggml_mul(ctx, value, ffi::ggml_silu(ctx, gate));
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ff2.weight"))?, h);
        h = ffi::ggml_add(ctx, h, self.tensor(&(p.to_owned() + "ff2.bias"))?);
        x = ffi::ggml_add(ctx, residual, h);
        Ok(x)
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn decoder_dyt(
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

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn decoder_block_l(
        &self,
        ctx: *mut ffi::ggml_context,
        block: usize,
        x: *mut ffi::ggml_tensor,
        pos: *mut ffi::ggml_tensor,
        mask: *mut ffi::ggml_tensor,
        n_tokens: i64,
    ) -> Result<*mut ffi::ggml_tensor> {
        let p = format!("decoder.ly.3.tr.{block}.");
        let residual = x;
        let h = self.decoder_dyt(ctx, x, &(p.to_owned() + "pn."))?;
        let qkv = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "sa.qkv.weight"))?, h);
        let row_bytes = (*qkv).nb[1];
        let mut q = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, L_DIM, n_tokens, row_bytes, 0),
        );
        let mut k = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, L_DIM, n_tokens, row_bytes, L_DIM as usize * 4),
        );
        let mut v = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, L_DIM, n_tokens, row_bytes, L_DIM as usize * 8),
        );
        let mut qd = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, L_DIM, n_tokens, row_bytes, L_DIM as usize * 12),
        );
        let mut kd = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, L_DIM, n_tokens, row_bytes, L_DIM as usize * 16),
        );
        q = ffi::ggml_reshape_3d(ctx, q, HEAD_DIM, L_HEADS, n_tokens);
        k = ffi::ggml_reshape_3d(ctx, k, HEAD_DIM, L_HEADS, n_tokens);
        v = ffi::ggml_reshape_3d(ctx, v, HEAD_DIM, L_HEADS, n_tokens);
        qd = ffi::ggml_reshape_3d(ctx, qd, HEAD_DIM, L_HEADS, n_tokens);
        kd = ffi::ggml_reshape_3d(ctx, kd, HEAD_DIM, L_HEADS, n_tokens);
        q = self.decoder_dyt(ctx, q, &(p.to_owned() + "sa.qn."))?;
        k = self.decoder_dyt(ctx, k, &(p.to_owned() + "sa.kn."))?;
        qd = self.decoder_dyt(ctx, qd, &(p.to_owned() + "sa.qn."))?;
        kd = self.decoder_dyt(ctx, kd, &(p.to_owned() + "sa.kn."))?;
        q = decoder_rope(ctx, q, pos);
        k = decoder_rope(ctx, k, pos);
        qd = decoder_rope(ctx, qd, pos);
        kd = decoder_rope(ctx, kd, pos);
        let main = decoder_attention_l(ctx, q, k, v, mask, n_tokens);
        let diff = decoder_attention_l(ctx, qd, kd, v, mask, n_tokens);
        let mut attn = ffi::ggml_sub(ctx, main, diff);
        attn = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "sa.out.weight"))?, attn);
        let mut x = ffi::ggml_add(ctx, residual, attn);

        let residual = x;
        let mut h = self.decoder_dyt(ctx, x, &(p.to_owned() + "fn."))?;
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ff0.weight"))?, h);
        h = ffi::ggml_add(ctx, h, self.tensor(&(p.to_owned() + "ff0.bias"))?);
        let row_bytes = (*h).nb[1];
        let value = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, h, L_FF_INNER, n_tokens, row_bytes, 0),
        );
        let mut gate = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(
                ctx,
                h,
                L_FF_INNER,
                n_tokens,
                row_bytes,
                L_FF_INNER as usize * 4,
            ),
        );
        gate = if block >= L_SIN_START_BLOCK {
            ffi::ggml_sin(ctx, ffi::ggml_scale(ctx, gate, std::f32::consts::PI))
        } else {
            ffi::ggml_silu(ctx, gate)
        };
        h = ffi::ggml_mul(ctx, value, gate);
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ff2.weight"))?, h);
        h = ffi::ggml_add(ctx, h, self.tensor(&(p.to_owned() + "ff2.bias"))?);
        x = ffi::ggml_add(ctx, residual, h);
        Ok(x)
    }

    unsafe fn compute_single_input_graph(
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
                return Err(Error::Ggml("failed to allocate SAME-S graph".into()));
            }
            set_input(gf, input_name, input);
            let status = ffi::ggml_backend_sched_graph_compute(self.scheduler(), gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.scheduler());
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "SAME-S compute failed: status={status}"
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
unsafe fn decoder_attention(
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
unsafe fn decoder_attention_l(
    ctx: *mut ffi::ggml_context,
    q: *mut ffi::ggml_tensor,
    k: *mut ffi::ggml_tensor,
    mut v: *mut ffi::ggml_tensor,
    mask: *mut ffi::ggml_tensor,
    n_tokens: i64,
) -> *mut ffi::ggml_tensor {
    let q = ffi::ggml_permute(ctx, q, 0, 2, 1, 3);
    let k = ffi::ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ffi::ggml_permute(ctx, v, 0, 2, 1, 3);
    let mut kq = ffi::ggml_mul_mat(ctx, k, q);
    kq = ffi::ggml_soft_max_ext(ctx, kq, mask, 1.0 / (HEAD_DIM as f32).sqrt(), 0.0);
    v = ffi::ggml_cont(ctx, ffi::ggml_transpose(ctx, v));
    let mut attn = ffi::ggml_mul_mat(ctx, v, kq);
    attn = ffi::ggml_permute(ctx, attn, 0, 2, 1, 3);
    ffi::ggml_cont_2d(ctx, attn, L_DIM, n_tokens)
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn decoder_rope(
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

fn copy_token_dim(src: &[f32], dim: usize, src_idx: usize, dst: &mut [f32], dst_idx: usize) {
    for ch in 0..dim {
        dst[ch + dst_idx * dim] = src[ch + src_idx * dim];
    }
}

fn same_l_swa_mask(n_tokens: usize) -> Vec<f32> {
    let mut mask = vec![-1.0e9_f32; n_tokens * n_tokens];
    for query in 0..n_tokens {
        let group = query / SUB_CHUNK;
        let query_in_group = query % SUB_CHUNK;
        for window_idx in query_in_group..=query_in_group + 2 * SUB_CHUNK {
            let key = group * SUB_CHUNK + window_idx;
            if key >= SUB_CHUNK {
                let key = key - SUB_CHUNK;
                if key < n_tokens {
                    mask[key + query * n_tokens] = 0.0;
                }
            }
        }
    }
    mask
}

unsafe fn set_input(gf: *mut ffi::ggml_cgraph, name: &CString, data: &[f32]) {
    unsafe {
        let tensor = ffi::ggml_graph_get_tensor(gf, name.as_ptr());
        ffi::ggml_backend_tensor_set(tensor, data.as_ptr().cast(), 0, std::mem::size_of_val(data));
    }
}
