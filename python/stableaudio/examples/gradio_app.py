from __future__ import annotations

from pathlib import Path

import gradio as gr

from stableaudio import StableAudio
from stableaudio.models import MODELS, ensure_model, model_paths


ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = APP_DIR / "output"
_MODEL: StableAudio | None = None
_MODEL_KEY: str | None = None

CSS = """
#app { max-width: 1120px; margin: 0 auto; }
#header h1 { margin-bottom: 0.15rem; }
#header p { margin-top: 0; color: var(--body-text-color-subdued); }
#run-button { min-height: 44px; }
.compact-output textarea { font-size: 0.9rem !important; }
"""


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

    def model_progress(fraction: float | None, desc: str) -> None:
        value = 0.10 if fraction is None else 0.10 + (fraction * 0.35)
        progress(value, desc=desc)

    ensure_model(model_key, ROOT, progress=model_progress)

    if _MODEL_KEY != model_key:
        if _MODEL is not None:
            _MODEL.close()
        progress(0.25, desc="Loading model")
        dit, decoder, text_encoder = model_paths(model_key, ROOT)
        _MODEL = StableAudio(dit, decoder, text_encoder, steps=steps, seed=seed)
        _MODEL_KEY = model_key

    progress(0.55, desc="Generating audio")
    assert _MODEL is not None
    output = OUTPUT_DIR / f"{model_key}-{seed}.wav"
    output = _MODEL.generate_wav(prompt, output, seconds=seconds, steps=steps, seed=seed)
    progress(1.0, desc="Done")
    return str(output), f"Ready: {output}"


theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="zinc",
)

with gr.Blocks(title="stableaudio-rs", fill_width=True) as demo:
    with gr.Column(elem_id="app"):
        gr.Markdown(
            "# stableaudio-rs\nGenerate music and sound effects locally with Stable Audio 3 Q8 GGUF models.",
            elem_id="header",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=7, min_width=420):
                model = gr.Dropdown(_choices(), value="small-sfx", label="Model")
                prompt = gr.Textbox(
                    value=MODELS["small-sfx"].default_prompt,
                    label="Prompt",
                    lines=5,
                    max_lines=8,
                    placeholder="Describe the sound or music you want to generate...",
                )
                suggestions = gr.Radio(_prompt_choices("small-sfx"), label="Prompt suggestions")
                run = gr.Button("Generate", variant="primary", elem_id="run-button")

            with gr.Column(scale=5, min_width=320):
                audio = gr.Audio(label="Output", type="filepath", autoplay=False)
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=2,
                    elem_classes=["compact-output"],
                )
                with gr.Accordion("Generation settings", open=True):
                    seconds = gr.Number(
                        value=MODELS["small-sfx"].seconds,
                        label="Seconds",
                        precision=1,
                    )
                    steps = gr.Slider(1, 16, value=8, step=1, label="Steps")
                    seed = gr.Number(value=777, label="Seed", precision=0)

    model.change(on_model_change, inputs=model, outputs=[prompt, suggestions, seconds])
    suggestions.change(use_suggestion, inputs=suggestions, outputs=prompt)
    run.click(generate, inputs=[model, prompt, seconds, steps, seed], outputs=[audio, status])


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(OUTPUT_DIR)], theme=theme, css=CSS)
