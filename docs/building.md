# BUILDING

## RUST LIBRARY

```bash
cargo build -p stable-audio
```

## C API

```bash
cargo build -p stable-audio-capi
```

THE C HEADER IS GENERATED WITH `CBINDGEN`:

```bash
cargo binstall cbindgen -y
cbindgen crates/stable-audio-capi \
  --crate stable-audio-capi \
  --output crates/stable-audio-capi/include/stable_audio.h
```

THE GENERATED HEADER IS CHECKED IN AT:

```text
crates/stable-audio-capi/include/stable_audio.h
```

## PYTHON PACKAGE

BUILD THE C API FIRST, THEN RUN PYTHON EXAMPLES:

```bash
cargo build -p stable-audio-capi
cd python/stableaudio
uv run python examples/small_sfx.py
uv run python examples/gradio_app.py
```
