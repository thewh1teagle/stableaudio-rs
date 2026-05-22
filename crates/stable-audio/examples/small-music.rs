/*
Download Q8 model bundle:

    wget https://github.com/thewh1teagle/stableaudio-rs/releases/download/models-v0.1.0/stable-audio-3-small-music-q8_0.tar.gz
    tar -xzf stable-audio-3-small-music-q8_0.tar.gz

Run:

    cargo run -p stable-audio --example small-music -- \
      --dit models/gguf-q8_0/sa3-small-music-dit.gguf \
      --decoder models/gguf-q8_0/sa3-small-music-same-s-decoder.gguf \
      --text-encoder models/gguf-q8_0/t5gemma-b-b-ul2-encoder.gguf \
      --prompt "continuous upbeat electro funk groove, steady drums throughout, warm bassline" \
      --seconds 6 \
      --output output/small-music.wav
*/

use std::path::PathBuf;

use clap::Parser;
use stable_audio::{GenerateRequest, Result, StableAudio, StableAudioConfig};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "models/gguf/sa3-small-music-dit.gguf")]
    dit: PathBuf,

    #[arg(
        long,
        default_value = "models/gguf/sa3-small-music-same-s-decoder.gguf"
    )]
    decoder: PathBuf,

    #[arg(long, default_value = "models/gguf/t5gemma-b-b-ul2-encoder.gguf")]
    text_encoder: PathBuf,

    #[arg(
        long,
        short,
        default_value = "warm analog synth arpeggio with a tight electronic drum groove"
    )]
    prompt: String,

    #[arg(long, default_value_t = 8.0)]
    seconds: f32,

    #[arg(long, default_value_t = 8)]
    steps: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, short, default_value = "output/small-music.wav")]
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
