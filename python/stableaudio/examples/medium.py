from pathlib import Path

from stableaudio import StableAudio
from stableaudio.models import ensure_model, model_paths


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    spec = ensure_model("medium", ROOT)
    dit, decoder, text_encoder = model_paths("medium", ROOT)
    with StableAudio(dit, decoder, text_encoder, steps=8, seed=779) as model:
        output = model.generate_wav(
            spec.default_prompt,
            ROOT / "output/python-medium-q8-test.wav",
            seconds=spec.seconds,
            steps=8,
            seed=779,
        )
    print(output)


if __name__ == "__main__":
    main()
