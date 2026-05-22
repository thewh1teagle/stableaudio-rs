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

Other examples:

```bash
uv run python examples/small_music.py
uv run python examples/medium.py
```

Run the local Gradio app:

```bash
uv run python examples/gradio_app.py
```
