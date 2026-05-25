use std::path::{Path, PathBuf};

pub mod audio;
pub mod error;
pub mod ggml_runtime;
pub mod sa3;
pub mod text;

pub use audio::AudioBuffer;
pub use error::{Error, Result};

use ggml_runtime::gguf::GgufModel;
use ggml_runtime::weights::GgmlWeights;
use text::Tokenizer;

#[derive(Debug, Clone)]
pub struct StableAudioConfig {
    pub dit_path: PathBuf,
    pub decoder_path: PathBuf,
    pub text_encoder_path: PathBuf,
    pub encoder_path: Option<PathBuf>,
    pub steps: usize,
    pub seed: u64,
}

impl StableAudioConfig {
    pub fn new(
        dit_path: impl Into<PathBuf>,
        decoder_path: impl Into<PathBuf>,
        text_encoder_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            dit_path: dit_path.into(),
            decoder_path: decoder_path.into(),
            text_encoder_path: text_encoder_path.into(),
            encoder_path: None,
            steps: 8,
            seed: 0,
        }
    }

    pub fn encoder_path(mut self, encoder_path: impl Into<PathBuf>) -> Self {
        self.encoder_path = Some(encoder_path.into());
        self
    }

    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub seconds: f32,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
}

impl GenerateRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            seconds: 8.0,
            steps: None,
            seed: None,
        }
    }

    pub fn seconds(mut self, seconds: f32) -> Self {
        self.seconds = seconds;
        self
    }

    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = Some(steps);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

#[derive(Debug, Clone)]
pub struct AudioEditRequest {
    pub prompt: String,
    pub input_path: PathBuf,
    pub seconds: f32,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
    pub init_noise_level: f32,
}

impl AudioEditRequest {
    pub fn new(prompt: impl Into<String>, input_path: impl Into<PathBuf>) -> Self {
        Self {
            prompt: prompt.into(),
            input_path: input_path.into(),
            seconds: 8.0,
            steps: None,
            seed: None,
            init_noise_level: 0.9,
        }
    }

    pub fn seconds(mut self, seconds: f32) -> Self {
        self.seconds = seconds;
        self
    }

    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = Some(steps);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn init_noise_level(mut self, init_noise_level: f32) -> Self {
        self.init_noise_level = init_noise_level;
        self
    }
}

#[derive(Debug, Clone)]
pub struct AudioInpaintRequest {
    pub prompt: String,
    pub input_path: PathBuf,
    pub seconds: f32,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
    pub inpaint_start: f32,
    pub inpaint_end: f32,
}

impl AudioInpaintRequest {
    pub fn new(prompt: impl Into<String>, input_path: impl Into<PathBuf>) -> Self {
        Self {
            prompt: prompt.into(),
            input_path: input_path.into(),
            seconds: 8.0,
            steps: None,
            seed: None,
            inpaint_start: 0.0,
            inpaint_end: 0.0,
        }
    }

    pub fn seconds(mut self, seconds: f32) -> Self {
        self.seconds = seconds;
        self
    }

    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = Some(steps);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn inpaint_range(mut self, start: f32, end: f32) -> Self {
        self.inpaint_start = start;
        self.inpaint_end = end;
        self
    }
}

#[derive(Debug, Clone)]
pub struct AudioContinuationRequest {
    pub prompt: String,
    pub input_path: PathBuf,
    pub extend_seconds: f32,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
}

impl AudioContinuationRequest {
    pub fn new(prompt: impl Into<String>, input_path: impl Into<PathBuf>) -> Self {
        Self {
            prompt: prompt.into(),
            input_path: input_path.into(),
            extend_seconds: 8.0,
            steps: None,
            seed: None,
        }
    }

    pub fn extend_seconds(mut self, extend_seconds: f32) -> Self {
        self.extend_seconds = extend_seconds;
        self
    }

    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = Some(steps);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

pub struct StableAudio {
    dit: GgufModel,
    encoder: Option<GgufModel>,
    decoder: GgufModel,
    text_encoder: GgufModel,
    dit_weights: GgmlWeights,
    encoder_weights: Option<GgmlWeights>,
    decoder_weights: GgmlWeights,
    text_encoder_weights: GgmlWeights,
    tokenizer: Tokenizer,
    config: StableAudioConfig,
}

impl StableAudio {
    pub fn load(config: StableAudioConfig) -> Result<Self> {
        let dit = GgufModel::open(&config.dit_path)?;
        let encoder = config
            .encoder_path
            .as_ref()
            .map(GgufModel::open)
            .transpose()?;
        let decoder = GgufModel::open(&config.decoder_path)?;
        let text_encoder = GgufModel::open(&config.text_encoder_path)?;
        let tokenizer = Tokenizer::from_gguf(&text_encoder)?;
        let dit_weights = GgmlWeights::load_all(&dit)?;
        let encoder_weights = encoder.as_ref().map(GgmlWeights::load_all).transpose()?;
        let decoder_weights = GgmlWeights::load_all(&decoder)?;
        let text_encoder_weights = GgmlWeights::load_all(&text_encoder)?;
        Ok(Self {
            dit,
            encoder,
            decoder,
            text_encoder,
            dit_weights,
            encoder_weights,
            decoder_weights,
            text_encoder_weights,
            tokenizer,
            config,
        })
    }

    pub fn dit(&self) -> &GgufModel {
        &self.dit
    }

    pub fn decoder(&self) -> &GgufModel {
        &self.decoder
    }

    pub fn encoder(&self) -> Option<&GgufModel> {
        self.encoder.as_ref()
    }

    pub fn text_encoder(&self) -> &GgufModel {
        &self.text_encoder
    }

    pub fn model_path(&self) -> &Path {
        &self.config.dit_path
    }

    pub fn generate(&mut self, request: GenerateRequest) -> Result<AudioBuffer> {
        let steps = request.steps.unwrap_or(self.config.steps);
        let seed = request.seed.unwrap_or(self.config.seed);
        let tokens = self.tokenizer.encode_sa3_prompt(&request.prompt, 256)?;
        sa3::generate_sfx(
            &self.dit,
            &self.decoder,
            &self.text_encoder,
            &mut self.dit_weights,
            &mut self.decoder_weights,
            &mut self.text_encoder_weights,
            &tokens,
            request.seconds,
            steps,
            seed,
        )
    }

    pub fn edit_audio(&mut self, request: AudioEditRequest) -> Result<AudioBuffer> {
        let steps = request.steps.unwrap_or(self.config.steps);
        let seed = request.seed.unwrap_or(self.config.seed);
        let tokens = self.tokenizer.encode_sa3_prompt(&request.prompt, 256)?;
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Incomplete("audio-to-audio requires an encoder GGUF".into()))?;
        let encoder_weights = self
            .encoder_weights
            .as_mut()
            .ok_or_else(|| Error::Incomplete("audio-to-audio requires encoder weights".into()))?;
        sa3::edit_audio(
            &self.dit,
            encoder,
            &self.decoder,
            &self.text_encoder,
            &mut self.dit_weights,
            encoder_weights,
            &mut self.decoder_weights,
            &mut self.text_encoder_weights,
            &tokens,
            &request.input_path,
            request.seconds,
            steps,
            seed,
            request.init_noise_level,
        )
    }

    pub fn inpaint_audio(&mut self, request: AudioInpaintRequest) -> Result<AudioBuffer> {
        let steps = request.steps.unwrap_or(self.config.steps);
        let seed = request.seed.unwrap_or(self.config.seed);
        let tokens = self.tokenizer.encode_sa3_prompt(&request.prompt, 256)?;
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Incomplete("inpainting requires an encoder GGUF".into()))?;
        let encoder_weights = self
            .encoder_weights
            .as_mut()
            .ok_or_else(|| Error::Incomplete("inpainting requires encoder weights".into()))?;
        sa3::inpaint_audio(
            &self.dit,
            encoder,
            &self.decoder,
            &self.text_encoder,
            &mut self.dit_weights,
            encoder_weights,
            &mut self.decoder_weights,
            &mut self.text_encoder_weights,
            &tokens,
            &request.input_path,
            request.seconds,
            steps,
            seed,
            request.inpaint_start,
            request.inpaint_end,
        )
    }

    pub fn continue_audio(&mut self, request: AudioContinuationRequest) -> Result<AudioBuffer> {
        let input = audio::wav::read_f32_stereo_44k(&request.input_path)?;
        let input_seconds =
            input.samples.len() as f32 / input.channels as f32 / input.sample_rate as f32;
        let steps = request.steps.unwrap_or(self.config.steps);
        let seed = request.seed.unwrap_or(self.config.seed);
        let prompt = request.prompt;
        let input_path = request.input_path;
        let extend_seconds = request.extend_seconds;
        let audio = self.inpaint_audio(
            AudioInpaintRequest::new(prompt.clone(), input_path)
                .seconds(input_seconds + extend_seconds)
                .steps(steps)
                .seed(seed)
                .inpaint_range(input_seconds, input_seconds + extend_seconds),
        )?;
        if tail_rms(&audio, input.samples.len() / input.channels as usize) >= 0.01 {
            return Ok(audio);
        }

        let tail_prompt =
            format!("continuous full-duration music with no fade out and no silence, {prompt}");
        let mut tail = self.generate(
            GenerateRequest::new(tail_prompt)
                .seconds(extend_seconds + 4.0)
                .steps(steps)
                .seed(seed ^ 0xC0A7_1A11),
        )?;
        let tail_len =
            (extend_seconds * tail.sample_rate as f32).round() as usize * tail.channels as usize;
        tail.samples.truncate(tail_len);
        Ok(splice_continuation(&input, &tail))
    }
}

fn tail_rms(audio: &AudioBuffer, start_frame: usize) -> f32 {
    let channels = audio.channels as usize;
    let start = start_frame
        .saturating_mul(channels)
        .min(audio.samples.len());
    let tail = &audio.samples[start..];
    if tail.is_empty() {
        return 0.0;
    }
    (tail.iter().map(|sample| sample * sample).sum::<f32>() / tail.len() as f32).sqrt()
}

fn splice_continuation(prefix: &AudioBuffer, tail: &AudioBuffer) -> AudioBuffer {
    let channels = prefix.channels as usize;
    let fade_frames = ((prefix.sample_rate as f32 * 0.05).round() as usize)
        .min(prefix.samples.len() / channels)
        .min(tail.samples.len() / channels);
    let mut samples = prefix.samples.clone();
    let prefix_last_frame = prefix.samples.len() / channels - 1;
    let tail_start = samples.len();
    samples.extend_from_slice(&tail.samples);
    for fade_frame in 0..fade_frames {
        let t = fade_frame as f32 / fade_frames.max(1) as f32;
        let tail_frame = fade_frame;
        for ch in 0..channels {
            let out_idx = tail_start + tail_frame * channels + ch;
            let prefix_idx = prefix_last_frame * channels + ch;
            let tail_idx = tail_frame * channels + ch;
            samples[out_idx] = prefix.samples[prefix_idx] * (1.0 - t) + tail.samples[tail_idx] * t;
        }
    }

    AudioBuffer {
        sample_rate: prefix.sample_rate,
        channels: prefix.channels,
        samples,
    }
}
