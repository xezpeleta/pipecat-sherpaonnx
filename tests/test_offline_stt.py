"""Tests for SherpaOnnxOfflineSTTService (uses the fake sherpa_onnx)."""

from __future__ import annotations

import numpy as np
import pytest

from pipecat.frames.frames import (
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import run_test

from pipecat_sherpaonnx import SherpaOnnxOfflineSTTService
from pipecat_sherpaonnx._common import resolve_transducer_files

SAMPLE_RATE = 16000


def _pcm(duration_ms: int = 100) -> bytes:
    return (np.ones(SAMPLE_RATE * duration_ms // 1000, dtype=np.int16)).tobytes()


def test_resolve_transducer_files_auto_discovery(model_dir):
    enc, dec, joi, tok = resolve_transducer_files(model_dir, None, None, None, None)
    assert enc.endswith("encoder.int8.onnx")
    assert dec.endswith("decoder.int8.onnx")
    assert joi.endswith("joiner.int8.onnx")
    assert tok.endswith("tokens.txt")


def test_resolve_transducer_files_explicit_paths(model_dir):
    enc, dec, joi, tok = resolve_transducer_files(
        None,
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
    )
    assert enc.endswith("encoder.int8.onnx")


def test_resolve_transducer_files_missing_model_dir():
    with pytest.raises(ValueError):
        resolve_transducer_files(None, None, None, None, None)


async def test_offline_emits_final_transcript(fake_sherpa, model_dir):
    service = SherpaOnnxOfflineSTTService(
        model_dir=str(model_dir), model_type="nemo_transducer"
    )
    frames = [
        VADUserStartedSpeakingFrame(),
        InputAudioRawFrame(audio=_pcm(), sample_rate=SAMPLE_RATE, num_channels=1),
        VADUserStoppedSpeakingFrame(),
    ]
    down, _ = await run_test(service, frames_to_send=frames)

    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert len(finals) == 1
    assert finals[0].text == "kaixo"
    assert finals[0].finalized is True

    # The recognizer was built with the right model_type.
    rec = fake_sherpa.OfflineRecognizer
    assert rec.instances
    assert rec.instances[0].kwargs["model_type"] == "nemo_transducer"


async def test_offline_no_transcript_for_empty_result(fake_sherpa, model_dir):
    # Configure the fake to return empty text.
    fake_sherpa.OfflineRecognizer.text = ""
    try:
        service = SherpaOnnxOfflineSTTService(model_dir=str(model_dir))
        frames = [
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=_pcm(), sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ]
        down, _ = await run_test(service, frames_to_send=frames)
        finals = [f for f in down if isinstance(f, TranscriptionFrame)]
        assert finals == []
    finally:
        fake_sherpa.OfflineRecognizer.text = "kaixo"
