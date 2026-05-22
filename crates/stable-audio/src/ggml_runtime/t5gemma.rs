use std::ffi::CString;
use std::ptr;

use llama_rs_sys as ffi;

use crate::ggml_runtime::weights::GgmlWeights;
use crate::{Error, Result};

const HIDDEN: i64 = 768;
const N_HEAD: i64 = 12;
const HEAD_DIM: i64 = 64;
const N_LAYER: usize = 12;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10000.0;
const ATTN_SOFTCAP: f32 = 50.0;

impl GgmlWeights {
    pub fn encode_t5gemma(&mut self, token_ids: &[i32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let n_tokens = token_ids.len();
        let inp_name = CString::new("inp_t5_tokens").unwrap();
        let pos_name = CString::new("inp_t5_pos").unwrap();
        let out_name = CString::new("t5_hidden").unwrap();

        unsafe {
            let ctx0 = self.compute_context()?;
            let gf = ffi::ggml_new_graph_custom(ctx0, 32768, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create T5 graph".into()));
            }

            let inp = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, n_tokens as i64);
            ffi::ggml_set_name(inp, inp_name.as_ptr());
            ffi::ggml_set_input(inp);
            let pos = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, n_tokens as i64);
            ffi::ggml_set_name(pos, pos_name.as_ptr());
            ffi::ggml_set_input(pos);

            let mut cur = ffi::ggml_get_rows(ctx0, self.tensor("emb.weight")?, inp);
            cur = ffi::ggml_cast(ctx0, cur, ffi::ggml_type_GGML_TYPE_F32);
            cur = ffi::ggml_scale(ctx0, cur, (HIDDEN as f32).sqrt());

            for layer in 0..N_LAYER {
                let p = format!("ly.{layer}.");
                let residual = cur;
                let mut h = rms_mul(ctx0, cur, self.tensor(&(p.clone() + "pre_sa_ln.weight"))?);
                let mut q = ffi::ggml_mul_mat(ctx0, self.tensor(&(p.clone() + "sa.q.weight"))?, h);
                let mut k = ffi::ggml_mul_mat(ctx0, self.tensor(&(p.clone() + "sa.k.weight"))?, h);
                let mut v = ffi::ggml_mul_mat(ctx0, self.tensor(&(p.clone() + "sa.v.weight"))?, h);

                q = ffi::ggml_reshape_3d(ctx0, q, HEAD_DIM, N_HEAD, n_tokens as i64);
                k = ffi::ggml_reshape_3d(ctx0, k, HEAD_DIM, N_HEAD, n_tokens as i64);
                v = ffi::ggml_reshape_3d(ctx0, v, HEAD_DIM, N_HEAD, n_tokens as i64);
                q = rope(ctx0, q, pos);
                k = rope(ctx0, k, pos);

                let q = ffi::ggml_permute(ctx0, q, 0, 2, 1, 3);
                let k = ffi::ggml_permute(ctx0, k, 0, 2, 1, 3);
                let mut v = ffi::ggml_permute(ctx0, v, 0, 2, 1, 3);
                let mut kq = ffi::ggml_mul_mat(ctx0, k, q);
                kq = ffi::ggml_scale(ctx0, kq, 1.0 / (HEAD_DIM as f32).sqrt());
                kq = ffi::ggml_scale(
                    ctx0,
                    ffi::ggml_tanh(ctx0, ffi::ggml_scale(ctx0, kq, 1.0 / ATTN_SOFTCAP)),
                    ATTN_SOFTCAP,
                );
                kq = ffi::ggml_soft_max(ctx0, kq);
                v = ffi::ggml_cont(ctx0, ffi::ggml_transpose(ctx0, v));
                let mut attn = ffi::ggml_mul_mat(ctx0, v, kq);
                attn = ffi::ggml_permute(ctx0, attn, 0, 2, 1, 3);
                attn = ffi::ggml_cont_2d(ctx0, attn, HIDDEN, n_tokens as i64);
                attn = ffi::ggml_mul_mat(ctx0, self.tensor(&(p.clone() + "sa.o.weight"))?, attn);
                attn = rms_mul(ctx0, attn, self.tensor(&(p.clone() + "post_sa_ln.weight"))?);
                cur = ffi::ggml_add(ctx0, residual, attn);

                let residual = cur;
                h = rms_mul(ctx0, cur, self.tensor(&(p.clone() + "pre_ff_ln.weight"))?);
                let mut gate =
                    ffi::ggml_mul_mat(ctx0, self.tensor(&(p.clone() + "mlp.gate.weight"))?, h);
                let up = ffi::ggml_mul_mat(ctx0, self.tensor(&(p.clone() + "mlp.up.weight"))?, h);
                gate = ffi::ggml_gelu_quick(ctx0, gate);
                h = ffi::ggml_mul(ctx0, gate, up);
                h = ffi::ggml_mul_mat(ctx0, self.tensor(&(p.clone() + "mlp.down.weight"))?, h);
                h = rms_mul(ctx0, h, self.tensor(&(p + "post_ff_ln.weight"))?);
                cur = ffi::ggml_add(ctx0, residual, h);
            }

            cur = rms_mul(ctx0, cur, self.tensor("norm.weight")?);
            ffi::ggml_set_name(cur, out_name.as_ptr());
            ffi::ggml_set_output(cur);
            ffi::ggml_build_forward_expand(gf, cur);

            if !ffi::ggml_backend_sched_alloc_graph(self.scheduler(), gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to allocate T5 graph".into()));
            }
            let inp_graph = ffi::ggml_graph_get_tensor(gf, inp_name.as_ptr());
            ffi::ggml_backend_tensor_set(
                inp_graph,
                token_ids.as_ptr().cast(),
                0,
                std::mem::size_of_val(token_ids),
            );
            let positions = (0..n_tokens).map(|idx| idx as i32).collect::<Vec<_>>();
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
                return Err(Error::Ggml(format!("T5 compute failed: status={status}")));
            }
            let out = ffi::ggml_graph_get_tensor(gf, out_name.as_ptr());
            let mut output = vec![0.0f32; n_tokens * HIDDEN as usize];
            ffi::ggml_backend_tensor_get(
                out,
                output.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(output.as_slice()),
            );
            ffi::ggml_backend_sched_reset(self.scheduler());
            ffi::ggml_free(ctx0);
            Ok(output)
        }
    }
}

unsafe fn rms_mul(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    weight: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    unsafe { ffi::ggml_mul(ctx, ffi::ggml_rms_norm(ctx, x, RMS_EPS), weight) }
}

unsafe fn rope(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    pos: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    unsafe {
        ffi::ggml_rope_ext(
            ctx,
            x,
            pos,
            ptr::null_mut(),
            HEAD_DIM as i32,
            ffi::GGML_ROPE_TYPE_NEOX as i32,
            0,
            ROPE_THETA,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
    }
}
