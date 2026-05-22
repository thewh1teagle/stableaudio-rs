from pathlib import Path

from stableaudio import StableAudio


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    with StableAudio(
        ROOT / "models/gguf-q8_0/sa3-small-sfx-dit.gguf",
        ROOT / "models/gguf-q8_0/sa3-same-s-decoder.gguf",
        ROOT / "models/gguf-q8_0/t5gemma-b-b-ul2-encoder.gguf",
        steps=8,
        seed=777,
    ) as model:
        output = model.generate_wav(
            "crystalline robot power up, sparkling servo chirps, deep digital whoosh",
            ROOT / "output/python-small-sfx-q8-test.wav",
            seconds=3.0,
            steps=8,
            seed=777,
        )
    print(output)


if __name__ == "__main__":
    main()
