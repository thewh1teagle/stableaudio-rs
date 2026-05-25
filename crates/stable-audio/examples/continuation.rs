/*
Download Q8 model bundle and the example source WAV:

    wget https://github.com/thewh1teagle/stableaudio-rs/releases/download/models-v0.1.0/stable-audio-3-small-music-q8_0.tar.gz
    tar -xzf stable-audio-3-small-music-q8_0.tar.gz
    mkdir -p input
    wget -O input/continuation-prefix.wav \
      https://github.com/thewh1teagle/stableaudio-rs/releases/download/models-v0.1.0/continuation-prefix.wav

Run:

    cargo run -p stable-audio --example continuation -- \
      --dit models/gguf-q8_0/sa3-small-music-dit.gguf \
      --encoder models/gguf-q8_0/sa3-small-music-same-s-encoder.gguf \
      --decoder models/gguf-q8_0/sa3-small-music-same-s-decoder.gguf \
      --text-encoder models/gguf-q8_0/t5gemma-b-b-ul2-encoder.gguf \
      --input input/continuation-prefix.wav \
      --prompt "continue as cinematic electronic music with rising drums" \
      --extend-seconds 6 \
      --output output/continuation.wav
*/

use std::path::PathBuf;

use clap::Parser;
use stable_audio::{AudioContinuationRequest, Result, StableAudio, StableAudioConfig};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "models/gguf/sa3-small-music-dit.gguf")]
    dit: PathBuf,

    #[arg(
        long,
        default_value = "models/gguf/sa3-small-music-same-s-encoder.gguf"
    )]
    encoder: PathBuf,

    #[arg(
        long,
        default_value = "models/gguf/sa3-small-music-same-s-decoder.gguf"
    )]
    decoder: PathBuf,

    #[arg(long, default_value = "models/gguf/t5gemma-b-b-ul2-encoder.gguf")]
    text_encoder: PathBuf,

    #[arg(long, default_value = "input/continuation-prefix.wav")]
    input: PathBuf,

    #[arg(
        long,
        short,
        default_value = "continue as cinematic electronic music with rising drums"
    )]
    prompt: String,

    #[arg(long, default_value_t = 6.0)]
    extend_seconds: f32,

    #[arg(long, default_value_t = 8)]
    steps: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, short, default_value = "output/continuation.wav")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = StableAudioConfig::new(args.dit, args.decoder, args.text_encoder)
        .encoder_path(args.encoder)
        .steps(args.steps)
        .seed(args.seed);
    let mut model = StableAudio::load(config)?;
    let audio = model.continue_audio(
        AudioContinuationRequest::new(args.prompt, args.input)
            .extend_seconds(args.extend_seconds)
            .steps(args.steps)
            .seed(args.seed),
    )?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    audio.write_wav(args.output)?;
    Ok(())
}
