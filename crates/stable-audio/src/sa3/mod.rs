pub mod conditioning;

use crate::Result;
use crate::audio::AudioBuffer;
use crate::ggml_runtime::gguf::GgufModel;
use crate::ggml_runtime::weights::GgmlWeights;
use crate::text::TokenizedPrompt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

pub fn generate_sfx(
    dit: &GgufModel,
    decoder: &GgufModel,
    text_encoder: &GgufModel,
    dit_weights: &mut GgmlWeights,
    decoder_weights: &mut GgmlWeights,
    text_encoder_weights: &mut GgmlWeights,
    tokens: &TokenizedPrompt,
    seconds: f32,
    steps: usize,
    seed: u64,
) -> Result<AudioBuffer> {
    let component = dit.get_string("sa3.component")?.unwrap_or_default();
    if !matches!(
        component.as_str(),
        "dit-small-sfx" | "dit-small-music" | "dit-medium"
    ) {
        return Err(crate::Error::Incomplete(format!(
            "Rust ggml runtime for {component} is not implemented yet"
        )));
    }

    let token_ids = tokens.ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
    let real_len = tokens.mask.iter().filter(|mask| **mask != 0).count();
    let _prompt_embeddings = text_encoder_weights.encode_t5gemma(&token_ids[..real_len])?;
    let conditioning = conditioning::build_conditioning(
        dit_weights,
        tokens,
        &_prompt_embeddings,
        seconds,
        steps,
        seed,
        component.as_str() != "dit-medium",
    )?;
    let latents = sample_latents(
        dit_weights,
        &conditioning.noise,
        conditioning.t_lat,
        &conditioning.cross_attn,
        &conditioning.global_cond,
        &conditioning.sigmas,
        seed,
        component.as_str() == "dit-medium",
    )?;
    let patches = if component == "dit-medium" {
        decoder_weights.decode_same_l(&latents, conditioning.t_lat)?
    } else {
        decoder_weights.decode_same_s(&latents, conditioning.t_lat)?
    };
    let samples = patched_decode_stereo(&patches, conditioning.t_lat, conditioning.target_samples);

    let _ = (
        dit,
        decoder,
        text_encoder,
        dit_weights,
        tokens,
        seconds,
        steps,
        seed,
    );
    Ok(AudioBuffer {
        sample_rate: conditioning::SAMPLE_RATE as u32,
        channels: 2,
        samples,
    })
}

fn sample_latents(
    dit_weights: &mut GgmlWeights,
    initial_noise: &[f32],
    t_lat: usize,
    cross_attn: &[f32],
    global_cond: &[f32],
    sigmas: &[f32],
    seed: u64,
    medium: bool,
) -> Result<Vec<f32>> {
    let mut x = initial_noise.to_vec();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5A17_3A3D_DEC0_DED5);
    for pair in sigmas.windows(2) {
        let t_curr = pair[0];
        let t_next = pair[1];
        let t_feat = conditioning::timestep_features(t_curr);
        let projected_x = if medium {
            dit_weights
                .dit_medium_velocity(&x, t_lat, cross_attn, global_cond, &t_feat)?
                .projected_x
        } else {
            dit_weights
                .dit_prepare_inputs(&x, t_lat, cross_attn, global_cond, &t_feat)?
                .projected_x
        };
        let mut denoised = vec![0.0f32; x.len()];
        for ((out, x_i), v_i) in denoised.iter_mut().zip(&x).zip(&projected_x) {
            *out = *x_i - t_curr * *v_i;
        }
        if t_next > 0.0 {
            for (x_i, d_i) in x.iter_mut().zip(&denoised) {
                let noise: f32 = StandardNormal.sample(&mut rng);
                *x_i = (1.0 - t_next) * *d_i + t_next * noise;
            }
        } else {
            x = denoised;
        }
    }
    Ok(x)
}

fn patched_decode_stereo(patches: &[f32], t_lat: usize, target_samples: usize) -> Vec<f32> {
    let patch_t = t_lat * 16;
    let total_samples = t_lat * conditioning::SAMPLES_PER_LATENT;
    let keep = target_samples.min(total_samples);
    let mut samples = Vec::with_capacity(keep * 2);
    for sample_idx in 0..keep {
        let patch_pos = sample_idx / 256;
        let within = sample_idx % 256;
        let left = patches[(within) + patch_pos * 512];
        let right = patches[(256 + within) + patch_pos * 512];
        samples.push(left);
        samples.push(right);
    }
    let _ = patch_t;
    samples
}
