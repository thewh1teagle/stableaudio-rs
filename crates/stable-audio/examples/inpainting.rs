/*
Download Q8 model bundle and the example source WAV:

    wget https://github.com/thewh1teagle/stableaudio-rs/releases/download/models-v0.1.0/stable-audio-3-small-music-q8_0.tar.gz
    tar -xzf stable-audio-3-small-music-q8_0.tar.gz
    mkdir -p input
    wget -O input/inpainting-source.wav \
      https://github.com/thewh1teagle/stableaudio-rs/releases/download/models-v0.1.0/inpainting-source.wav

Run:

    cargo run -p stable-audio --example inpainting -- \
      --dit models/gguf-q8_0/sa3-small-music-dit.gguf \
      --encoder models/gguf-q8_0/sa3-small-music-same-s-encoder.gguf \
      --decoder models/gguf-q8_0/sa3-small-music-same-s-decoder.gguf \
      --text-encoder models/gguf-q8_0/t5gemma-b-b-ul2-encoder.gguf \
      --input input/inpainting-source.wav \
      --prompt "replace this section with a short distorted guitar fill" \
      --inpaint-start 4.0 \
      --inpaint-end 7.0 \
      --seconds 10 \
      --output output/inpainting.wav
*/

use std::path::PathBuf;

use clap::Parser;
use stable_audio::{AudioInpaintRequest, Result, StableAudio, StableAudioConfig};

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

    #[arg(long, default_value = "input/inpainting-source.wav")]
    input: PathBuf,

    #[arg(
        long,
        short,
        default_value = "replace this section with a short distorted guitar fill"
    )]
    prompt: String,

    #[arg(long, default_value_t = 4.0)]
    inpaint_start: f32,

    #[arg(long, default_value_t = 7.0)]
    inpaint_end: f32,

    #[arg(long, default_value_t = 10.0)]
    seconds: f32,

    #[arg(long, default_value_t = 8)]
    steps: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, short, default_value = "output/inpainting.wav")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = StableAudioConfig::new(args.dit, args.decoder, args.text_encoder)
        .encoder_path(args.encoder)
        .steps(args.steps)
        .seed(args.seed);
    let mut model = StableAudio::load(config)?;
    let audio = model.inpaint_audio(
        AudioInpaintRequest::new(args.prompt, args.input)
            .seconds(args.seconds)
            .steps(args.steps)
            .seed(args.seed)
            .inpaint_range(args.inpaint_start, args.inpaint_end),
    )?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    audio.write_wav(args.output)?;
    Ok(())
}
