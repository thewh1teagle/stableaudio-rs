/*
Download Q8 model bundle:

    wget https://github.com/thewh1teagle/stableaudio-rs/releases/download/models-v0.1.0/stable-audio-3-medium-q8_0.tar.gz
    tar -xzf stable-audio-3-medium-q8_0.tar.gz

Run:

    cargo run -p stable-audio --example medium -- \
      --dit models/gguf-q8_0/sa3-medium-dit.gguf \
      --decoder models/gguf-q8_0/sa3-medium-same-l-decoder.gguf \
      --text-encoder models/gguf-q8_0/t5gemma-b-b-ul2-encoder.gguf \
      --prompt "continuous 10 second futuristic electro funk track, tight steady drums throughout, warm slap bass groove" \
      --seconds 10 \
      --output output/medium.wav
*/

use std::path::PathBuf;

use clap::Parser;
use stable_audio::{GenerateRequest, Result, StableAudio, StableAudioConfig};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "models/gguf/sa3-medium-dit.gguf")]
    dit: PathBuf,

    #[arg(long, default_value = "models/gguf/sa3-medium-same-l-decoder.gguf")]
    decoder: PathBuf,

    #[arg(long, default_value = "models/gguf/t5gemma-b-b-ul2-encoder.gguf")]
    text_encoder: PathBuf,

    #[arg(
        long,
        short,
        default_value = "cinematic piano arpeggio building into orchestral electronic drums"
    )]
    prompt: String,

    #[arg(long, default_value_t = 1.0)]
    seconds: f32,

    #[arg(long, default_value_t = 8)]
    steps: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, short, default_value = "output/medium.wav")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = StableAudioConfig::new(args.dit, args.decoder, args.text_encoder)
        .steps(args.steps)
        .seed(args.seed);
    let mut model = StableAudio::load(config)?;
    let audio = model.generate(
        GenerateRequest::new(args.prompt)
            .seconds(args.seconds)
            .steps(args.steps)
            .seed(args.seed),
    )?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    audio.write_wav(args.output)?;
    Ok(())
}
