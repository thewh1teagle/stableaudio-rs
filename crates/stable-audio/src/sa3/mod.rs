pub mod conditioning;

use std::path::Path;

use crate::Result;
use crate::audio::{self, AudioBuffer};
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
        None,
        &conditioning.sigmas,
        seed,
        component.as_str() == "dit-medium",
        None,
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

#[allow(clippy::too_many_arguments)]
pub fn edit_audio(
    dit: &GgufModel,
    encoder: &GgufModel,
    decoder: &GgufModel,
    text_encoder: &GgufModel,
    dit_weights: &mut GgmlWeights,
    encoder_weights: &mut GgmlWeights,
    decoder_weights: &mut GgmlWeights,
    text_encoder_weights: &mut GgmlWeights,
    tokens: &TokenizedPrompt,
    input_path: &Path,
    seconds: f32,
    steps: usize,
    seed: u64,
    init_noise_level: f32,
) -> Result<AudioBuffer> {
    generate_with_audio_conditioning(
        dit,
        Some(encoder),
        decoder,
        text_encoder,
        dit_weights,
        Some(encoder_weights),
        decoder_weights,
        text_encoder_weights,
        tokens,
        Some(input_path),
        seconds,
        steps,
        seed,
        AudioConditioningMode::AudioToAudio { init_noise_level },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn inpaint_audio(
    dit: &GgufModel,
    encoder: &GgufModel,
    decoder: &GgufModel,
    text_encoder: &GgufModel,
    dit_weights: &mut GgmlWeights,
    encoder_weights: &mut GgmlWeights,
    decoder_weights: &mut GgmlWeights,
    text_encoder_weights: &mut GgmlWeights,
    tokens: &TokenizedPrompt,
    input_path: &Path,
    seconds: f32,
    steps: usize,
    seed: u64,
    inpaint_start: f32,
    inpaint_end: f32,
) -> Result<AudioBuffer> {
    generate_with_audio_conditioning(
        dit,
        Some(encoder),
        decoder,
        text_encoder,
        dit_weights,
        Some(encoder_weights),
        decoder_weights,
        text_encoder_weights,
        tokens,
        Some(input_path),
        seconds,
        steps,
        seed,
        AudioConditioningMode::Inpaint {
            start_seconds: inpaint_start,
            end_seconds: inpaint_end,
        },
    )
}

enum AudioConditioningMode {
    AudioToAudio {
        init_noise_level: f32,
    },
    Inpaint {
        start_seconds: f32,
        end_seconds: f32,
    },
}

#[allow(clippy::too_many_arguments)]
fn generate_with_audio_conditioning(
    dit: &GgufModel,
    encoder: Option<&GgufModel>,
    decoder: &GgufModel,
    text_encoder: &GgufModel,
    dit_weights: &mut GgmlWeights,
    encoder_weights: Option<&mut GgmlWeights>,
    decoder_weights: &mut GgmlWeights,
    text_encoder_weights: &mut GgmlWeights,
    tokens: &TokenizedPrompt,
    input_path: Option<&Path>,
    seconds: f32,
    steps: usize,
    seed: u64,
    mode: AudioConditioningMode,
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
    let prompt_embeddings = text_encoder_weights.encode_t5gemma(&token_ids[..real_len])?;
    let conditioning = conditioning::build_conditioning(
        dit_weights,
        tokens,
        &prompt_embeddings,
        seconds,
        steps,
        seed,
        component.as_str() != "dit-medium",
    )?;

    let mut initial_noise = conditioning.noise.clone();
    let mut local_add_cond = None;
    let mut paste_back = None;
    let mut waveform_restore = None;
    if let Some(input_path) = input_path {
        let encoder = encoder.ok_or_else(|| {
            crate::Error::Incomplete("audio conditioning requires an encoder GGUF".into())
        })?;
        let encoder_weights = encoder_weights.ok_or_else(|| {
            crate::Error::Incomplete("audio conditioning requires encoder weights".into())
        })?;
        let input = audio::wav::read_f32_stereo_44k(input_path)?;
        let target_samples = conditioning.t_lat * conditioning::SAMPLES_PER_LATENT;
        let audio_patches = patch_audio_for_encoder(&input, target_samples);
        let init_latents =
            if encoder.get_string("sa3.component")?.as_deref() == Some("same-l-encoder") {
                encoder_weights.encode_same_l(&audio_patches, conditioning.t_lat)?
            } else {
                encoder_weights.encode_same_s(&audio_patches, conditioning.t_lat)?
            };
        match mode {
            AudioConditioningMode::AudioToAudio { init_noise_level } => {
                let sigma_max = init_noise_level.clamp(0.0, 1.0);
                for (noise, latent) in initial_noise.iter_mut().zip(&init_latents) {
                    *noise = *latent * (1.0 - sigma_max) + *noise * sigma_max;
                }
            }
            AudioConditioningMode::Inpaint {
                start_seconds,
                end_seconds,
            } => {
                let start_lat = seconds_to_latent(start_seconds, conditioning.t_lat);
                let end_lat = seconds_to_latent(end_seconds, conditioning.t_lat);
                let mut mask = vec![1.0f32; conditioning.t_lat];
                for value in mask.iter_mut().take(end_lat).skip(start_lat) {
                    *value = 0.0;
                }
                let mut local = vec![0.0f32; conditioning.t_lat * (conditioning::IO_CHANNELS + 1)];
                for t in 0..conditioning.t_lat {
                    local[t * (conditioning::IO_CHANNELS + 1)] = mask[t];
                    for ch in 0..conditioning::IO_CHANNELS {
                        local[t * (conditioning::IO_CHANNELS + 1) + 1 + ch] =
                            init_latents[ch * conditioning.t_lat + t] * mask[t];
                    }
                }
                local_add_cond = Some(local);
                paste_back = Some((init_latents, mask));
                waveform_restore = Some((input, start_seconds, end_seconds));
            }
        }
    }

    let latents = sample_latents(
        dit_weights,
        &initial_noise,
        conditioning.t_lat,
        &conditioning.cross_attn,
        &conditioning.global_cond,
        local_add_cond.as_deref(),
        &conditioning.sigmas,
        seed,
        component.as_str() == "dit-medium",
        paste_back.as_ref(),
    )?;
    let patches = if component == "dit-medium" {
        decoder_weights.decode_same_l(&latents, conditioning.t_lat)?
    } else {
        decoder_weights.decode_same_s(&latents, conditioning.t_lat)?
    };
    let mut samples =
        patched_decode_stereo(&patches, conditioning.t_lat, conditioning.target_samples);
    if let Some((input, start_seconds, end_seconds)) = waveform_restore {
        restore_inpaint_waveform(&mut samples, &input, start_seconds, end_seconds);
    }

    let _ = (dit, decoder, text_encoder, tokens, seconds, steps, seed);
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
    local_add_cond: Option<&[f32]>,
    sigmas: &[f32],
    seed: u64,
    medium: bool,
    paste_back: Option<&(Vec<f32>, Vec<f32>)>,
) -> Result<Vec<f32>> {
    let mut x = initial_noise.to_vec();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5A17_3A3D_DEC0_DED5);
    for pair in sigmas.windows(2) {
        let t_curr = pair[0];
        let t_next = pair[1];
        let t_feat = conditioning::timestep_features(t_curr);
        let projected_x = if medium {
            dit_weights
                .dit_medium_velocity(&x, t_lat, cross_attn, global_cond, local_add_cond, &t_feat)?
                .projected_x
        } else {
            dit_weights
                .dit_prepare_inputs(&x, t_lat, cross_attn, global_cond, local_add_cond, &t_feat)?
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
        if let Some((init_latents, mask)) = paste_back {
            for t in 0..t_lat {
                if mask[t] != 0.0 {
                    for ch in 0..conditioning::IO_CHANNELS {
                        x[ch * t_lat + t] = init_latents[ch * t_lat + t];
                    }
                }
            }
        }
    }
    Ok(x)
}

fn patch_audio_for_encoder(audio: &AudioBuffer, target_samples: usize) -> Vec<f32> {
    let mut channels = vec![vec![0.0f32; target_samples]; 2];
    let frames = audio.samples.len() / audio.channels as usize;
    let keep = frames.min(target_samples);
    for frame in 0..keep {
        channels[0][frame] = audio.samples[frame * 2];
        channels[1][frame] = audio.samples[frame * 2 + 1];
    }
    let patch_t = target_samples / 256;
    let mut patches = vec![0.0f32; 512 * patch_t];
    for patch in 0..patch_t {
        for sample in 0..256 {
            let src = patch * 256 + sample;
            patches[sample + patch * 512] = channels[0][src];
            patches[256 + sample + patch * 512] = channels[1][src];
        }
    }
    patches
}

fn seconds_to_latent(seconds: f32, t_lat: usize) -> usize {
    ((seconds * conditioning::SAMPLE_RATE as f32) / conditioning::SAMPLES_PER_LATENT as f32)
        .round()
        .clamp(0.0, t_lat as f32) as usize
}

fn restore_inpaint_waveform(
    samples: &mut [f32],
    input: &AudioBuffer,
    start_seconds: f32,
    end_seconds: f32,
) {
    if input.channels != 2 || samples.is_empty() {
        return;
    }
    let sample_rate = conditioning::SAMPLE_RATE;
    let frames = samples.len() / 2;
    let input_frames = input.samples.len() / input.channels as usize;
    let start_frame = seconds_to_frame(start_seconds, frames);
    let end_frame = seconds_to_frame(end_seconds, frames).max(start_frame);
    let fade_frames = ((sample_rate as f32 * 0.05).round() as usize)
        .max(1)
        .min(frames);

    let prefix_end = start_frame.min(frames).min(input_frames);
    for frame in 0..prefix_end {
        copy_input_frame(samples, input, frame);
    }

    let start_fade_end = (start_frame + fade_frames).min(frames);
    for frame in start_frame..start_fade_end {
        let ramp = (frame - start_frame) as f32 / fade_frames as f32;
        blend_input_to_generated(samples, input, frame, ramp);
    }

    if end_frame < frames {
        let end_fade_start = end_frame.saturating_sub(fade_frames).max(start_frame);
        for frame in end_fade_start..end_frame.min(frames) {
            let ramp = (frame - end_fade_start) as f32 / (end_frame - end_fade_start).max(1) as f32;
            blend_input_to_generated(samples, input, frame, 1.0 - ramp);
        }
        let suffix_start = end_frame.min(frames).min(input_frames);
        for frame in suffix_start..frames.min(input_frames) {
            copy_input_frame(samples, input, frame);
        }
    }
}

fn seconds_to_frame(seconds: f32, max_frames: usize) -> usize {
    (seconds * conditioning::SAMPLE_RATE as f32)
        .round()
        .clamp(0.0, max_frames as f32) as usize
}

fn copy_input_frame(samples: &mut [f32], input: &AudioBuffer, frame: usize) {
    samples[frame * 2] = input.samples[frame * 2];
    samples[frame * 2 + 1] = input.samples[frame * 2 + 1];
}

fn blend_input_to_generated(samples: &mut [f32], input: &AudioBuffer, frame: usize, gen_mix: f32) {
    let input_frames = input.samples.len() / input.channels as usize;
    let source_frame = frame.min(input_frames.saturating_sub(1));
    let input_mix = 1.0 - gen_mix;
    for ch in 0..2 {
        let idx = frame * 2 + ch;
        samples[idx] = input.samples[source_frame * 2 + ch] * input_mix + samples[idx] * gen_mix;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_inpaint_waveform_preserves_unmasked_regions() {
        let frames = conditioning::SAMPLE_RATE * 3;
        let input_samples = (0..frames)
            .flat_map(|frame| {
                let value = frame as f32 / frames as f32;
                [value, -value]
            })
            .collect::<Vec<_>>();
        let input = AudioBuffer {
            sample_rate: conditioning::SAMPLE_RATE as u32,
            channels: 2,
            samples: input_samples.clone(),
        };
        let mut output = vec![0.5f32; input_samples.len()];

        restore_inpaint_waveform(&mut output, &input, 1.0, 2.0);

        let start = conditioning::SAMPLE_RATE * 2;
        for frame in 0..conditioning::SAMPLE_RATE {
            assert_eq!(output[frame * 2], input_samples[frame * 2]);
            assert_eq!(output[frame * 2 + 1], input_samples[frame * 2 + 1]);
        }
        for frame in start..frames {
            assert_eq!(output[frame * 2], input_samples[frame * 2]);
            assert_eq!(output[frame * 2 + 1], input_samples[frame * 2 + 1]);
        }
    }
}
