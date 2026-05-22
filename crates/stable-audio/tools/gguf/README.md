# stable-audio-gguf-tools

GGUF conversion tools for the Rust `stable-audio` crate.

Convert a gated Hugging Face Stable Audio 3 small checkpoint from the local HF
cache or by downloading it:

```bash
uv run stable-audio-gguf-tools \
  --model-id stabilityai/stable-audio-3-small-sfx \
  --output-dir ../../models/gguf \
  --out-type f16
```

For Small-SFX, the converter writes:

- `sa3-small-sfx-dit.gguf`
- `sa3-same-s-decoder.gguf`
- `t5gemma-b-b-ul2-encoder.gguf`

For Small-Music:

```bash
uv run stable-audio-gguf-tools \
  --model-id stabilityai/stable-audio-3-small-music \
  --output-dir ../../models/gguf \
  --out-type f16
```

This writes:

- `sa3-small-music-dit.gguf`
- `sa3-small-music-same-s-decoder.gguf`
- `t5gemma-b-b-ul2-encoder.gguf`

For Medium:

```bash
uv run stable-audio-gguf-tools \
  --model-id stabilityai/stable-audio-3-medium \
  --output-dir ../../models/gguf \
  --out-type f16
```

This writes:

- `sa3-medium-dit.gguf`
- `sa3-medium-same-l-decoder.gguf`
- `t5gemma-b-b-ul2-encoder.gguf`

`--out-type` accepts `f32`, `f16`, `q8_0`, and `q6_k`. Quantized exports keep
1D/bias/norm tensors in FP32/FP16 and quantize eligible 2D/3D weights.
