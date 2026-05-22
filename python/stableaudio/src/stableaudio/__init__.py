from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path


def _library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libstable_audio_capi.dylib"
    if system == "Windows":
        return "stable_audio_capi.dll"
    return "libstable_audio_capi.so"


def _candidate_library_paths() -> list[Path]:
    paths = []
    env_path = os.environ.get("STABLE_AUDIO_CAPI_LIB")
    if env_path:
        paths.append(Path(env_path))

    here = Path(__file__).resolve()
    for parent in here.parents:
        paths.append(parent / "native" / _library_name())
        paths.append(parent / "target" / "debug" / _library_name())
        paths.append(parent / "target" / "release" / _library_name())
    return paths


def _load_library() -> ctypes.CDLL:
    for path in _candidate_library_paths():
        if path.exists():
            return ctypes.CDLL(str(path))
    searched = "\n".join(str(path) for path in _candidate_library_paths())
    raise RuntimeError(
        "Could not find stable-audio C API library. "
        "Build it with `cargo build -p stable-audio-capi` or set "
        f"STABLE_AUDIO_CAPI_LIB.\nSearched:\n{searched}"
    )


_lib = _load_library()
_lib.stable_audio_last_error.restype = ctypes.c_char_p
_lib.stable_audio_model_load.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_uint64,
]
_lib.stable_audio_model_load.restype = ctypes.c_void_p
_lib.stable_audio_model_free.argtypes = [ctypes.c_void_p]
_lib.stable_audio_model_free.restype = None
_lib.stable_audio_generate_wav.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_float,
    ctypes.c_size_t,
    ctypes.c_uint64,
    ctypes.c_char_p,
]
_lib.stable_audio_generate_wav.restype = ctypes.c_int


def _bytes_path(path: str | os.PathLike[str]) -> bytes:
    return os.fsencode(Path(path))


def _check_status(status: int) -> None:
    if status == 0:
        return
    message = _lib.stable_audio_last_error()
    raise RuntimeError(message.decode("utf-8") if message else "stable-audio C API error")


class StableAudio:
    def __init__(
        self,
        dit: str | os.PathLike[str],
        decoder: str | os.PathLike[str],
        text_encoder: str | os.PathLike[str],
        *,
        steps: int = 8,
        seed: int = 0,
    ) -> None:
        handle = _lib.stable_audio_model_load(
            _bytes_path(dit),
            _bytes_path(decoder),
            _bytes_path(text_encoder),
            steps,
            seed,
        )
        if not handle:
            _check_status(-1)
        self._handle = ctypes.c_void_p(handle)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.stable_audio_model_free(self._handle)
            self._handle = None

    def generate_wav(
        self,
        prompt: str,
        output: str | os.PathLike[str],
        *,
        seconds: float = 8.0,
        steps: int = 8,
        seed: int = 0,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        status = _lib.stable_audio_generate_wav(
            self._handle,
            prompt.encode("utf-8"),
            seconds,
            steps,
            seed,
            _bytes_path(output),
        )
        _check_status(status)
        return output

    def __enter__(self) -> StableAudio:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["StableAudio"]
