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
            steps: 8,
            seed: 0,
        }
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

pub struct StableAudio {
    dit: GgufModel,
    decoder: GgufModel,
    text_encoder: GgufModel,
    dit_weights: GgmlWeights,
    decoder_weights: GgmlWeights,
    text_encoder_weights: GgmlWeights,
    tokenizer: Tokenizer,
    config: StableAudioConfig,
}

impl StableAudio {
    pub fn load(config: StableAudioConfig) -> Result<Self> {
        let dit = GgufModel::open(&config.dit_path)?;
        let decoder = GgufModel::open(&config.decoder_path)?;
        let text_encoder = GgufModel::open(&config.text_encoder_path)?;
        let tokenizer = Tokenizer::from_gguf(&text_encoder)?;
        let dit_weights = GgmlWeights::load_all(&dit)?;
        let decoder_weights = GgmlWeights::load_all(&decoder)?;
        let text_encoder_weights = GgmlWeights::load_all(&text_encoder)?;
        Ok(Self {
            dit,
            decoder,
            text_encoder,
            dit_weights,
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
}
