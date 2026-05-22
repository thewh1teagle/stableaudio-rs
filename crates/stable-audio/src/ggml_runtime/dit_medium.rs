use std::ffi::CString;

use llama_rs_sys as ffi;

use crate::ggml_runtime::weights::GgmlWeights;
use crate::{Error, Result};

const IO: i64 = 256;
const COND: i64 = 768;
const EMBED: i64 = 1536;
const N_HEAD: i64 = 24;
const HEAD_DIM: i64 = 64;
const MEMORY: i64 = 64;
const ROPE_DIMS: i32 = 32;
const NORM_EPS: f32 = 1e-5;
const QK_EPS: f32 = 1e-6;
const PROMPT_PLUS_SECONDS: i64 = 257;
const DEPTH: usize = 24;
const FF_INNER: i64 = 6144;

impl GgmlWeights {
    pub fn dit_medium_velocity(
        &mut self,
        noise: &[f32],
        t_lat: usize,
        cross_attn: &[f32],
        global_cond: &[f32],
        timestep_embed: &[f32],
    ) -> Result<MediumDitPrepared> {
        let x_name = CString::new("inp_dit_x").unwrap();
        let cross_name = CString::new("inp_dit_cross").unwrap();
        let global_name = CString::new("inp_dit_global").unwrap();
        let time_name = CString::new("inp_dit_time").unwrap();
        let pos_name = CString::new("inp_dit_pos").unwrap();
        let local_name = CString::new("inp_dit_local_zero").unwrap();
        let mem_zero_name = CString::new("inp_dit_mem_zero").unwrap();
        let one_name = CString::new("inp_dit_one").unwrap();
        let out_name = CString::new("dit_velocity").unwrap();

        if noise.len() != IO as usize * t_lat {
            return Err(Error::Ggml("DiT noise length mismatch".into()));
        }
        if cross_attn.len() != PROMPT_PLUS_SECONDS as usize * COND as usize {
            return Err(Error::Ggml("DiT cross attention length mismatch".into()));
        }
        if global_cond.len() != COND as usize || timestep_embed.len() != 256 {
            return Err(Error::Ggml("DiT conditioning length mismatch".into()));
        }

        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 32768, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create DiT prep graph".into()));
            }

            let x = ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, IO, t_lat as i64);
            ffi::ggml_set_name(x, x_name.as_ptr());
            ffi::ggml_set_input(x);
            let cross = ffi::ggml_new_tensor_2d(
                ctx0,
                ffi::ggml_type_GGML_TYPE_F32,
                COND,
                PROMPT_PLUS_SECONDS,
            );
            ffi::ggml_set_name(cross, cross_name.as_ptr());
            ffi::ggml_set_input(cross);
            let global = ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, COND, 1);
            ffi::ggml_set_name(global, global_name.as_ptr());
            ffi::ggml_set_input(global);
            let time = ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, 256, 1);
            ffi::ggml_set_name(time, time_name.as_ptr());
            ffi::ggml_set_input(time);
            let pos =
                ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, t_lat as i64 + MEMORY);
            ffi::ggml_set_name(pos, pos_name.as_ptr());
            ffi::ggml_set_input(pos);
            let local_zero = ffi::ggml_new_tensor_2d(
                ctx0,
                ffi::ggml_type_GGML_TYPE_F32,
                PROMPT_PLUS_SECONDS,
                t_lat as i64,
            );
            ffi::ggml_set_name(local_zero, local_name.as_ptr());
            ffi::ggml_set_input(local_zero);
            let mem_zero =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, EMBED, MEMORY);
            ffi::ggml_set_name(mem_zero, mem_zero_name.as_ptr());
            ffi::ggml_set_input(mem_zero);
            let one = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_F32, 1);
            ffi::ggml_set_name(one, one_name.as_ptr());
            ffi::ggml_set_input(one);

            let mut context = ffi::ggml_mul_mat(ctx0, self.tensor("cond_emb.0.weight")?, cross);
            context = ffi::ggml_silu(ctx0, context);
            context = ffi::ggml_mul_mat(ctx0, self.tensor("cond_emb.2.weight")?, context);

            let mut global_pre = ffi::ggml_mul_mat(ctx0, self.tensor("glob_emb.0.weight")?, global);
            global_pre = ffi::ggml_silu(ctx0, global_pre);
            global_pre = ffi::ggml_mul_mat(ctx0, self.tensor("glob_emb.2.weight")?, global_pre);

            let mut t_emb = ffi::ggml_mul_mat(ctx0, self.tensor("time_emb.0.weight")?, time);
            t_emb = ffi::ggml_add(ctx0, t_emb, self.tensor("time_emb.0.bias")?);
            t_emb = ffi::ggml_silu(ctx0, t_emb);
            t_emb = ffi::ggml_mul_mat(ctx0, self.tensor("time_emb.2.weight")?, t_emb);
            t_emb = ffi::ggml_add(ctx0, t_emb, self.tensor("time_emb.2.bias")?);
            let global_embed = ffi::ggml_add(ctx0, global_pre, t_emb);

            let pre_w = ffi::ggml_reshape_2d(ctx0, self.tensor("pre.weight")?, IO, IO);
            let x_pp = ffi::ggml_add(ctx0, ffi::ggml_mul_mat(ctx0, pre_w, x), x);
            let mut h = ffi::ggml_mul_mat(ctx0, self.tensor("transformer.pin.weight")?, x_pp);

            let mut g =
                ffi::ggml_mul_mat(ctx0, self.tensor("transformer.gce.0.weight")?, global_embed);
            g = ffi::ggml_add(ctx0, g, self.tensor("transformer.gce.0.bias")?);
            g = ffi::ggml_silu(ctx0, g);
            g = ffi::ggml_mul_mat(ctx0, self.tensor("transformer.gce.2.weight")?, g);
            g = ffi::ggml_add(ctx0, g, self.tensor("transformer.gce.2.bias")?);

            let mem = ffi::ggml_cast(
                ctx0,
                self.tensor("transformer.memory_tokens")?,
                ffi::ggml_type_GGML_TYPE_F32,
            );
            h = ffi::ggml_concat(ctx0, mem, h, 1);
            for layer in 0..DEPTH {
                h = self.dit_medium_block(
                    ctx0,
                    layer,
                    h,
                    context,
                    g,
                    pos,
                    local_zero,
                    mem_zero,
                    one,
                    t_lat as i64,
                )?;
            }

            h = ffi::ggml_cont(
                ctx0,
                ffi::ggml_view_2d(
                    ctx0,
                    h,
                    EMBED,
                    t_lat as i64,
                    (*h).nb[1],
                    MEMORY as usize * (*h).nb[1],
                ),
            );
            h = ffi::ggml_mul_mat(ctx0, self.tensor("transformer.pout.weight")?, h);
            let post_w = ffi::ggml_reshape_2d(ctx0, self.tensor("post.weight")?, IO, IO);
            h = ffi::ggml_add(ctx0, ffi::ggml_mul_mat(ctx0, post_w, h), h);
            ffi::ggml_set_name(h, out_name.as_ptr());
            ffi::ggml_set_output(h);
            ffi::ggml_build_forward_expand(gf, h);
            ffi::ggml_build_forward_expand(gf, context);
            ffi::ggml_build_forward_expand(gf, g);

            if !ffi::ggml_backend_sched_alloc_graph(self.scheduler(), gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to allocate DiT prep graph".into()));
            }
            set_input(gf, &x_name, noise);
            set_input(gf, &cross_name, cross_attn);
            set_input(gf, &global_name, global_cond);
            set_input(gf, &time_name, timestep_embed);
            let positions = (0..(t_lat + MEMORY as usize))
                .map(|idx| idx as i32)
                .collect::<Vec<_>>();
            let pos_graph = ffi::ggml_graph_get_tensor(gf, pos_name.as_ptr());
            ffi::ggml_backend_tensor_set(
                pos_graph,
                positions.as_ptr().cast(),
                0,
                std::mem::size_of_val(positions.as_slice()),
            );
            let local_zeros = vec![0.0f32; PROMPT_PLUS_SECONDS as usize * t_lat];
            set_input(gf, &local_name, &local_zeros);
            let mem_zeros = vec![0.0f32; EMBED as usize * MEMORY as usize];
            set_input(gf, &mem_zero_name, &mem_zeros);
            set_input(gf, &one_name, &[1.0f32]);
            let status = ffi::ggml_backend_sched_graph_compute(self.scheduler(), gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.scheduler());
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "DiT prep compute failed: status={status}"
                )));
            }
            let out = ffi::ggml_graph_get_tensor(gf, out_name.as_ptr());
            let mut projected_x = vec![0.0f32; IO as usize * t_lat];
            ffi::ggml_backend_tensor_get(
                out,
                projected_x.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(projected_x.as_slice()),
            );
            ffi::ggml_backend_sched_reset(self.scheduler());
            ffi::ggml_free(ctx0);
            Ok(MediumDitPrepared { projected_x })
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn dit_medium_block(
        &self,
        ctx: *mut ffi::ggml_context,
        layer: usize,
        x: *mut ffi::ggml_tensor,
        context: *mut ffi::ggml_tensor,
        global: *mut ffi::ggml_tensor,
        pos: *mut ffi::ggml_tensor,
        local_zero: *mut ffi::ggml_tensor,
        mem_zero: *mut ffi::ggml_tensor,
        one: *mut ffi::ggml_tensor,
        t_lat: i64,
    ) -> Result<*mut ffi::ggml_tensor> {
        let n_tokens = t_lat + MEMORY;
        let p = format!("tr.blk.{layer}.");
        let ss = ffi::ggml_add(ctx, self.tensor(&(p.to_owned() + "ssg"))?, global);
        let scale_self = ffi::ggml_view_1d(ctx, ss, EMBED, 0);
        let shift_self = ffi::ggml_view_1d(ctx, ss, EMBED, EMBED as usize * 4);
        let gate_self = ffi::ggml_view_1d(ctx, ss, EMBED, EMBED as usize * 8);

        let residual = x;
        let mut h = ffi::ggml_rms_norm(ctx, x, NORM_EPS);
        h = ffi::ggml_mul(ctx, h, self.tensor(&(p.to_owned() + "pn.gamma"))?);
        h = ffi::ggml_mul(ctx, h, ffi::ggml_add1(ctx, scale_self, one));
        h = ffi::ggml_add(ctx, h, shift_self);

        let qkv = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "sa.qkv.weight"))?, h);
        let row_bytes = (*qkv).nb[1];
        let q = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, EMBED, n_tokens, row_bytes, 0),
        );
        let k = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, EMBED, n_tokens, row_bytes, EMBED as usize * 4),
        );
        let v = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, EMBED, n_tokens, row_bytes, EMBED as usize * 8),
        );
        let q_diff = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, EMBED, n_tokens, row_bytes, EMBED as usize * 12),
        );
        let k_diff = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, qkv, EMBED, n_tokens, row_bytes, EMBED as usize * 16),
        );
        let mut q = ffi::ggml_reshape_3d(ctx, q, HEAD_DIM, N_HEAD, n_tokens);
        let mut k = ffi::ggml_reshape_3d(ctx, k, HEAD_DIM, N_HEAD, n_tokens);
        let v = ffi::ggml_reshape_3d(ctx, v, HEAD_DIM, N_HEAD, n_tokens);
        let mut q_diff = ffi::ggml_reshape_3d(ctx, q_diff, HEAD_DIM, N_HEAD, n_tokens);
        let mut k_diff = ffi::ggml_reshape_3d(ctx, k_diff, HEAD_DIM, N_HEAD, n_tokens);
        q = ffi::ggml_mul(
            ctx,
            ffi::ggml_rms_norm(ctx, q, QK_EPS),
            self.tensor(&(p.to_owned() + "sa.qn.gamma"))?,
        );
        k = ffi::ggml_mul(
            ctx,
            ffi::ggml_rms_norm(ctx, k, QK_EPS),
            self.tensor(&(p.to_owned() + "sa.kn.gamma"))?,
        );
        q_diff = ffi::ggml_mul(
            ctx,
            ffi::ggml_rms_norm(ctx, q_diff, QK_EPS),
            self.tensor(&(p.to_owned() + "sa.qn.gamma"))?,
        );
        k_diff = ffi::ggml_mul(
            ctx,
            ffi::ggml_rms_norm(ctx, k_diff, QK_EPS),
            self.tensor(&(p.to_owned() + "sa.kn.gamma"))?,
        );
        q = dit_rope(ctx, q, pos);
        k = dit_rope(ctx, k, pos);
        q_diff = dit_rope(ctx, q_diff, pos);
        k_diff = dit_rope(ctx, k_diff, pos);
        let main = medium_attention(ctx, q, k, v, n_tokens);
        let diff = medium_attention(ctx, q_diff, k_diff, v, n_tokens);
        let attn = ffi::ggml_sub(ctx, main, diff);
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "sa.out.weight"))?, attn);

        let gate = ffi::ggml_sigmoid(ctx, ffi::ggml_add1(ctx, ffi::ggml_neg(ctx, gate_self), one));
        h = ffi::ggml_mul(ctx, h, gate);
        let mut x = ffi::ggml_add(ctx, h, residual);

        let residual = x;
        h = ffi::ggml_rms_norm(ctx, x, NORM_EPS);
        h = ffi::ggml_mul(ctx, h, self.tensor(&(p.to_owned() + "ca_norm.gamma"))?);
        let q_all = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ca.q.weight"))?, h);
        let kv = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ca.kv.weight"))?, context);
        let row_bytes_q = (*q_all).nb[1];
        let mut q = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, q_all, EMBED, n_tokens, row_bytes_q, 0),
        );
        let mut q_diff = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, q_all, EMBED, n_tokens, row_bytes_q, EMBED as usize * 4),
        );
        let row_bytes = (*kv).nb[1];
        let mut k = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, kv, EMBED, PROMPT_PLUS_SECONDS, row_bytes, 0),
        );
        let mut k_diff = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(
                ctx,
                kv,
                EMBED,
                PROMPT_PLUS_SECONDS,
                row_bytes,
                EMBED as usize * 4,
            ),
        );
        let mut v = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(
                ctx,
                kv,
                EMBED,
                PROMPT_PLUS_SECONDS,
                row_bytes,
                EMBED as usize * 8,
            ),
        );
        q = ffi::ggml_reshape_3d(ctx, q, HEAD_DIM, N_HEAD, n_tokens);
        q_diff = ffi::ggml_reshape_3d(ctx, q_diff, HEAD_DIM, N_HEAD, n_tokens);
        k = ffi::ggml_reshape_3d(ctx, k, HEAD_DIM, N_HEAD, PROMPT_PLUS_SECONDS);
        k_diff = ffi::ggml_reshape_3d(ctx, k_diff, HEAD_DIM, N_HEAD, PROMPT_PLUS_SECONDS);
        v = ffi::ggml_reshape_3d(ctx, v, HEAD_DIM, N_HEAD, PROMPT_PLUS_SECONDS);
        q = ffi::ggml_mul(
            ctx,
            ffi::ggml_rms_norm(ctx, q, QK_EPS),
            self.tensor(&(p.to_owned() + "ca.qn.gamma"))?,
        );
        k = ffi::ggml_mul(
            ctx,
            ffi::ggml_rms_norm(ctx, k, QK_EPS),
            self.tensor(&(p.to_owned() + "ca.kn.gamma"))?,
        );
        q_diff = ffi::ggml_mul(
            ctx,
            ffi::ggml_rms_norm(ctx, q_diff, QK_EPS),
            self.tensor(&(p.to_owned() + "ca.qn.gamma"))?,
        );
        k_diff = ffi::ggml_mul(
            ctx,
            ffi::ggml_rms_norm(ctx, k_diff, QK_EPS),
            self.tensor(&(p.to_owned() + "ca.kn.gamma"))?,
        );
        let main = medium_attention(ctx, q, k, v, n_tokens);
        let diff = medium_attention(ctx, q_diff, k_diff, v, n_tokens);
        let mut ca = ffi::ggml_sub(ctx, main, diff);
        ca = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ca.out.weight"))?, ca);
        x = ffi::ggml_add(ctx, residual, ca);
        let mut local = ffi::ggml_mul_mat(
            ctx,
            self.tensor(&(p.to_owned() + "loc.0.weight"))?,
            local_zero,
        );
        local = ffi::ggml_add(ctx, local, self.tensor(&(p.to_owned() + "loc.0.bias"))?);
        local = ffi::ggml_silu(ctx, local);
        local = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "loc.2.weight"))?, local);
        local = ffi::ggml_add(ctx, local, self.tensor(&(p.to_owned() + "loc.2.bias"))?);
        local = ffi::ggml_concat(ctx, mem_zero, local, 1);
        x = ffi::ggml_add(ctx, x, local);

        let scale_ff = ffi::ggml_view_1d(ctx, ss, EMBED, EMBED as usize * 12);
        let shift_ff = ffi::ggml_view_1d(ctx, ss, EMBED, EMBED as usize * 16);
        let gate_ff = ffi::ggml_view_1d(ctx, ss, EMBED, EMBED as usize * 20);
        let residual = x;
        h = ffi::ggml_rms_norm(ctx, x, NORM_EPS);
        h = ffi::ggml_mul(ctx, h, self.tensor(&(p.to_owned() + "fn.gamma"))?);
        h = ffi::ggml_mul(ctx, h, ffi::ggml_add1(ctx, scale_ff, one));
        h = ffi::ggml_add(ctx, h, shift_ff);
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ff0.weight"))?, h);
        h = ffi::ggml_add(ctx, h, self.tensor(&(p.to_owned() + "ff0.bias"))?);
        let row_bytes = (*h).nb[1];
        let value = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, h, FF_INNER, n_tokens, row_bytes, 0),
        );
        let gate = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_2d(ctx, h, FF_INNER, n_tokens, row_bytes, FF_INNER as usize * 4),
        );
        h = ffi::ggml_mul(ctx, value, ffi::ggml_silu(ctx, gate));
        h = ffi::ggml_mul_mat(ctx, self.tensor(&(p.to_owned() + "ff2.weight"))?, h);
        h = ffi::ggml_add(ctx, h, self.tensor(&(p.to_owned() + "ff2.bias"))?);
        let gate = ffi::ggml_sigmoid(ctx, ffi::ggml_add1(ctx, ffi::ggml_neg(ctx, gate_ff), one));
        h = ffi::ggml_mul(ctx, h, gate);
        Ok(ffi::ggml_add(ctx, h, residual))
    }
}

pub struct MediumDitPrepared {
    pub projected_x: Vec<f32>,
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn medium_attention(
    ctx: *mut ffi::ggml_context,
    q: *mut ffi::ggml_tensor,
    k: *mut ffi::ggml_tensor,
    mut v: *mut ffi::ggml_tensor,
    n_tokens: i64,
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
    ffi::ggml_cont_2d(ctx, attn, EMBED, n_tokens)
}

unsafe fn set_input(gf: *mut ffi::ggml_cgraph, name: &CString, data: &[f32]) {
    unsafe {
        let tensor = ffi::ggml_graph_get_tensor(gf, name.as_ptr());
        ffi::ggml_backend_tensor_set(tensor, data.as_ptr().cast(), 0, std::mem::size_of_val(data));
    }
}

unsafe fn dit_rope(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    pos: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    unsafe {
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
}
