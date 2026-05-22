from __future__ import annotations

from pathlib import Path

import gradio as gr

from stableaudio import StableAudio
from stableaudio.models import MODELS, ensure_model, model_paths


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output/gradio.wav"
_MODEL: StableAudio | None = None
_MODEL_KEY: str | None = None


def _choices() -> list[tuple[str, str]]:
    return [(spec.label, key) for key, spec in MODELS.items()]


def _prompt_choices(model_key: str) -> list[str]:
    spec = MODELS[model_key]
    return [spec.default_prompt, *spec.suggestions]


def on_model_change(model_key: str):
    spec = MODELS[model_key]
    prompts = _prompt_choices(model_key)
    return (
        gr.update(value=spec.default_prompt),
        gr.update(choices=prompts, value=prompts[0]),
        gr.update(value=spec.seconds),
    )


def use_suggestion(prompt: str) -> str:
    return prompt


def generate(model_key: str, prompt: str, seconds: float, steps: int, seed: int, progress=gr.Progress()):
    global _MODEL, _MODEL_KEY
    progress(0.05, desc="Checking model files")
    ensure_model(model_key, ROOT)

    if _MODEL_KEY != model_key:
        if _MODEL is not None:
            _MODEL.close()
        progress(0.25, desc="Loading model")
        dit, decoder, text_encoder = model_paths(model_key, ROOT)
        _MODEL = StableAudio(dit, decoder, text_encoder, steps=steps, seed=seed)
        _MODEL_KEY = model_key

    progress(0.55, desc="Generating audio")
    assert _MODEL is not None
    output = _MODEL.generate_wav(prompt, OUTPUT, seconds=seconds, steps=steps, seed=seed)
    progress(1.0, desc="Done")
    return str(output), f"Wrote {output}"


with gr.Blocks(title="stableaudio-rs") as demo:
    gr.Markdown("# stableaudio-rs")
    gr.Markdown("Generate music and sound effects locally with Stable Audio 3 Q8 GGUF models.")
    with gr.Row():
        model = gr.Dropdown(_choices(), value="small-sfx", label="Model")
        seconds = gr.Number(value=MODELS["small-sfx"].seconds, label="Seconds", precision=1)
        steps = gr.Slider(1, 16, value=8, step=1, label="Steps")
        seed = gr.Number(value=777, label="Seed", precision=0)
    prompt = gr.Textbox(value=MODELS["small-sfx"].default_prompt, label="Prompt", lines=3)
    suggestions = gr.Radio(_prompt_choices("small-sfx"), label="Prompt suggestions")
    run = gr.Button("Generate")
    audio = gr.Audio(label="Output", type="filepath")
    status = gr.Textbox(label="Status", interactive=False)

    model.change(on_model_change, inputs=model, outputs=[prompt, suggestions, seconds])
    suggestions.change(use_suggestion, inputs=suggestions, outputs=prompt)
    run.click(generate, inputs=[model, prompt, seconds, steps, seed], outputs=[audio, status])


if __name__ == "__main__":
    demo.launch()
