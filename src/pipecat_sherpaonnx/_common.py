"""Shared helpers for the sherpa-onnx STT services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# 16-bit signed PCM max amplitude; used to normalise int16 samples to [-1, 1].
_INT16_MAX = 32768.0


def pcm_bytes_to_float32(audio: bytes) -> np.ndarray:
    """Convert raw 16-bit little-endian mono PCM bytes to float32 in [-1, 1].

    sherpa-onnx recognizers accept mono float32 samples normalised to the
    [-1, 1] range. Pipecat delivers microphone audio as 16-bit PCM bytes, so
    every chunk must be converted before being fed to a recognizer stream.
    """
    return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / _INT16_MAX


def resolve_transducer_files(
    model_dir: str | Path | None,
    encoder: str | None,
    decoder: str | None,
    joiner: str | None,
    tokens: str | None,
) -> tuple[str, str, str, str]:
    """Resolve the encoder/decoder/joiner/tokens paths for a transducer model.

    Explicit paths win when given. Otherwise the files are located by glob from
    ``model_dir`` using the common sherpa-onnx naming conventions, so a whole
    model directory can be pointed at without listing every file::

        model_dir/
        ├── encoder.int8.onnx   (or encoder.onnx)
        ├── decoder.int8.onnx   (or decoder.onnx)
        ├── joiner.int8.onnx    (or joiner.onnx)
        └── tokens.txt

    Returns ``(encoder, decoder, joiner, tokens)`` as absolute path strings.
    """
    base = Path(model_dir).expanduser().resolve() if model_dir else None

    def _find(name: str | None, patterns: list[str], label: str) -> str:
        if name:
            p = Path(name).expanduser()
            return str(p if p.is_absolute() or base is None else base / p)
        if base is None:
            raise ValueError(
                f"No {label} path given and no model_dir to resolve it from. "
                f"Pass either model_dir or an explicit {label} path."
            )
        for pat in patterns:
            matches = sorted(base.glob(pat))
            if matches:
                return str(matches[0])
        raise FileNotFoundError(
            f"Could not find a {label} file in {base} (tried {patterns})."
        )

    enc = _find(encoder, ["encoder*.onnx"], "encoder")
    dec = _find(decoder, ["decoder*.onnx"], "decoder")
    joi = _find(joiner, ["joiner*.onnx"], "joiner")
    tok = _find(tokens, ["tokens.txt"], "tokens")
    return enc, dec, joi, tok


def join_texts(a: Any, b: Any) -> str:
    """Join two (possibly empty) transcript fragments with a single space."""
    a = str(a or "").strip()
    b = str(b or "").strip()
    if a and b:
        return f"{a} {b}"
    return a or b
