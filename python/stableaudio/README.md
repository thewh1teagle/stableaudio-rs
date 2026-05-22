# stableaudio

Python `ctypes` bindings for the local `stable-audio-capi` shared library.

Build the C API first:

```bash
cargo build -p stable-audio-capi
```

Run the small SFX example from this directory:

```bash
uv run python examples/small_sfx.py
```
