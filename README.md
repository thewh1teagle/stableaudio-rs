# stableaudio-rs

Generate music and sound effects locally in Rust with Stable Audio 3 GGUF models.

Rust GGUF inference for Stable Audio 3 models.

## Features

- 🎵 Text-to-music and sound effect generation from Stable Audio 3 prompts.
- ⚡ GPU-accelerated inference through ggml backends such as Vulkan.
- 📦 Q8 GGUF model bundles for smaller local downloads.
- 🦀 Pure Rust examples for small SFX, small music, and medium models.
- 🔊 Writes standard stereo WAV output at 44.1 kHz.

Original models:

- [Stable Audio](https://stability.ai/stable-audio)
- 🤗 [Stable Audio 3 on Hugging Face](https://huggingface.co/collections/stabilityai/stable-audio-3)

See the examples in [`crates/stable-audio/examples`](crates/stable-audio/examples).
