from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile

import httpx


RELEASE_BASE = "https://github.com/thewh1teagle/stableaudio-rs/releases/download/models-v0.1.0"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    archive: str
    dit: str
    decoder: str
    text_encoder: str
    default_prompt: str
    seconds: float
    suggestions: tuple[str, ...]
    dependencies: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return f"{RELEASE_BASE}/{self.archive}"


MODELS = {
    "small-sfx": ModelSpec(
        key="small-sfx",
        label="Small SFX",
        archive="stable-audio-3-small-sfx-q8_0.tar.gz",
        dit="models/gguf-q8_0/sa3-small-sfx-dit.gguf",
        decoder="models/gguf-q8_0/sa3-same-s-decoder.gguf",
        text_encoder="models/gguf-q8_0/t5gemma-b-b-ul2-encoder.gguf",
        default_prompt="crystalline robot power up, sparkling servo chirps, deep digital whoosh",
        seconds=3.0,
        suggestions=(
            "cinematic laser blast with glassy tail and deep impact",
            "tiny magical UI notification, soft sparkle, clean chime",
            "heavy sci fi door opens with hydraulic hiss and low rumble",
        ),
    ),
    "small-music": ModelSpec(
        key="small-music",
        label="Small Music",
        archive="stable-audio-3-small-music-q8_0.tar.gz",
        dit="models/gguf-q8_0/sa3-small-music-dit.gguf",
        decoder="models/gguf-q8_0/sa3-small-music-same-s-decoder.gguf",
        text_encoder="models/gguf-q8_0/t5gemma-b-b-ul2-encoder.gguf",
        default_prompt="continuous upbeat electro funk groove, steady drums throughout, warm bassline",
        seconds=6.0,
        suggestions=(
            "continuous synthwave loop, punchy drums, warm bass, sparkling arpeggios",
            "lofi house groove, dusty drums, mellow keys, smooth bassline",
            "ambient piano and soft pads, gentle pulse, no fade out",
        ),
    ),
    "medium": ModelSpec(
        key="medium",
        label="Medium",
        archive="stable-audio-3-medium-q8_0.tar.gz",
        dit="models/gguf-q8_0/sa3-medium-dit.gguf",
        decoder="models/gguf-q8_0/sa3-medium-same-l-decoder.gguf",
        text_encoder="models/gguf-q8_0/t5gemma-b-b-ul2-encoder.gguf",
        default_prompt=(
            "continuous 10 second futuristic electro funk track, tight steady drums throughout, "
            "warm slap bass groove, bright synth stabs"
        ),
        seconds=10.0,
        suggestions=(
            "cinematic electronic score, pulsing analog bass, wide synth chords, heroic melody",
            "futuristic desert chase soundtrack, breakbeat drums, metallic percussion, tense strings",
            "dreamy piano arpeggio, orchestral electronic drums, warm evolving pads",
        ),
        dependencies=("small-music",),
    ),
}


def model_paths(model: str, root: str | Path = ".") -> tuple[Path, Path, Path]:
    spec = MODELS[model]
    root = Path(root)
    return root / spec.dit, root / spec.decoder, root / spec.text_encoder


def model_ready(model: str, root: str | Path = ".") -> bool:
    return all(path.exists() for path in model_paths(model, root))


def ensure_model(model: str, root: str | Path = ".", delete_archive: bool = True) -> ModelSpec:
    spec = MODELS[model]
    root = Path(root)
    for dependency in spec.dependencies:
        ensure_model(dependency, root=root, delete_archive=delete_archive)
    if not model_ready(model, root):
        _download_and_extract(spec, root, delete_archive=delete_archive)
    return spec


def _download_and_extract(spec: ModelSpec, root: Path, delete_archive: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / spec.archive
    if not archive_path.exists():
        with httpx.stream("GET", spec.url, follow_redirects=True, timeout=None) as response:
            response.raise_for_status()
            with archive_path.open("wb") as file:
                for chunk in response.iter_bytes():
                    if chunk:
                        file.write(chunk)
    _safe_extract(archive_path, root)
    if delete_archive:
        archive_path.unlink(missing_ok=True)


def _safe_extract(archive_path: Path, root: Path) -> None:
    root = root.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"refusing to extract unsafe path: {member.name}")
        archive.extractall(root)
