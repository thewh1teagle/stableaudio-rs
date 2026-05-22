use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::Result;
use crate::ggml_runtime::weights::GgmlWeights;
use crate::text::TokenizedPrompt;

pub const SAMPLE_RATE: usize = 44_100;
pub const SAMPLES_PER_LATENT: usize = 4096;
pub const PROMPT_LEN: usize = 256;
pub const COND_DIM: usize = 768;
pub const IO_CHANNELS: usize = 256;

#[derive(Debug, Clone)]
pub struct Conditioning {
    pub t_lat: usize,
    pub target_samples: usize,
    pub cross_attn: Vec<f32>,
    pub global_cond: Vec<f32>,
    pub noise: Vec<f32>,
    pub sigmas: Vec<f32>,
    pub timestep_features: Vec<f32>,
}

pub fn build_conditioning(
    dit_weights: &GgmlWeights,
    tokens: &TokenizedPrompt,
    prompt_embeddings: &[f32],
    seconds: f32,
    steps: usize,
    seed: u64,
    require_even_t_lat: bool,
) -> Result<Conditioning> {
    let mut t_lat = ((seconds * SAMPLE_RATE as f32) / SAMPLES_PER_LATENT as f32).ceil() as usize;
    t_lat = t_lat.max(1);
    if require_even_t_lat && t_lat % 2 != 0 {
        t_lat += 1;
    }
    let target_samples = (seconds * SAMPLE_RATE as f32).round() as usize;

    let pad = dit_weights.tensor_f32("cond.prompt.pad_emb")?;
    let sec_w = dit_weights.tensor_f32("cond.sec.1.weight")?;
    let sec_b = dit_weights.tensor_f32("cond.sec.1.bias")?;
    let seconds_embed = seconds_embedding(seconds, &sec_w, &sec_b);

    let mut cross_attn = vec![0.0; (PROMPT_LEN + 1) * COND_DIM];
    for token in 0..PROMPT_LEN {
        let is_real = tokens.mask.get(token).copied().unwrap_or(0) != 0;
        let src = if is_real {
            &prompt_embeddings[token * COND_DIM..(token + 1) * COND_DIM]
        } else {
            &pad
        };
        cross_attn[token * COND_DIM..(token + 1) * COND_DIM].copy_from_slice(src);
    }
    cross_attn[PROMPT_LEN * COND_DIM..(PROMPT_LEN + 1) * COND_DIM].copy_from_slice(&seconds_embed);

    let mut rng = StdRng::seed_from_u64(seed);
    let noise_len = IO_CHANNELS * t_lat;
    let noise = (0..noise_len)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect::<Vec<f32>>();

    Ok(Conditioning {
        t_lat,
        target_samples,
        cross_attn,
        global_cond: seconds_embed,
        noise,
        sigmas: build_pingpong_schedule(steps, 1.0),
        timestep_features: timestep_features(1.0),
    })
}

fn seconds_embedding(seconds: f32, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    let norm = (seconds.clamp(0.0, 384.0) - 0.0) / 384.0;
    let half: usize = 128;
    for i in 0..half {
        let ramp = i as f32 / (half.saturating_sub(1)).max(1) as f32;
        let freq = (ramp * (10000.0_f32.ln() - 0.5_f32.ln()) + 0.5_f32.ln()).exp();
        let arg = norm * freq * std::f32::consts::TAU;
        features[i] = arg.cos();
        features[i + half] = arg.sin();
    }
    let mut out = vec![0.0; COND_DIM];
    for row in 0..COND_DIM {
        let mut sum = bias[row];
        for col in 0..256 {
            sum += weight[row * 256 + col] * features[col];
        }
        out[row] = sum;
    }
    out
}

fn build_pingpong_schedule(steps: usize, sigma_max: f32) -> Vec<f32> {
    let steps = steps.max(1);
    let mut out = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = sigma_max + (0.0 - sigma_max) * (i as f32 / steps as f32);
        let shifted = logsnr_shift(t);
        out.push(if i == 0 { sigma_max } else { shifted });
    }
    out
}

pub fn timestep_features(t: f32) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    let half: usize = 128;
    for i in 0..half {
        let ramp = i as f32 / (half.saturating_sub(1)).max(1) as f32;
        let freq = (ramp * (10000.0_f32.ln() - 0.5_f32.ln()) + 0.5_f32.ln()).exp();
        let arg = t * freq * std::f32::consts::TAU;
        features[i] = arg.cos();
        features[i + half] = arg.sin();
    }
    features
}

fn logsnr_shift(t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let anchor_logsnr = -6.2_f32;
    let logsnr_end = 2.0_f32;
    let logsnr = logsnr_end - t * (logsnr_end - anchor_logsnr);
    1.0 / (1.0 + logsnr.exp())
}
