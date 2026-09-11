"""Tests for SherpaOnnxSTTService (uses the fake sherpa_onnx)."""

from __future__ import annotations

import numpy as np

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import run_test

from pipecat_sherpaonnx import SherpaOnnxSTTService

SAMPLE_RATE = 16000


def _pcm(duration_ms: int = 100) -> bytes:
    return (np.ones(SAMPLE_RATE * duration_ms // 1000, dtype=np.int16)).tobytes()


async def test_streaming_emits_interim_and_final(fake_sherpa, model_dir):
    """Feed audio -> interim partial; VAD stop -> finalized transcript."""
    service = SherpaOnnxSTTService(model_dir=str(model_dir), model_type="zipformer2")
    frames = [
        VADUserStartedSpeakingFrame(),
        InputAudioRawFrame(audio=_pcm(), sample_rate=SAMPLE_RATE, num_channels=1),
        VADUserStoppedSpeakingFrame(),
    ]
    down, _ = await run_test(service, frames_to_send=frames)

    interims = [f for f in down if isinstance(f, InterimTranscriptionFrame)]
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]

    assert len(interims) == 1
    assert interims[0].text == "kaixo"

    assert len(finals) == 1
    assert finals[0].text == "kaixo"
    assert finals[0].finalized is True


async def test_streaming_accumulates_across_endpoints(fake_sherpa, model_dir):
    """With endpoint detection on, segments accumulate into one final per turn."""
    rec = fake_sherpa.OnlineRecognizer
    rec.text = "kaixo"
    rec.endpoint = True
    try:
        service = SherpaOnnxSTTService(model_dir=str(model_dir))
        frames = [
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=_pcm(), sample_rate=SAMPLE_RATE, num_channels=1),
            # Second chunk: after the first endpoint reset, this feeds new audio
            # and the partial becomes "<committed> <current>".
            InputAudioRawFrame(audio=_pcm(), sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ]
        down, _ = await run_test(service, frames_to_send=frames)

        finals = [f for f in down if isinstance(f, TranscriptionFrame)]
        # One finalized transcript for the whole turn.
        assert len(finals) == 1
        assert finals[0].text == "kaixo kaixo"
    finally:
        rec.endpoint = False


async def test_streaming_emit_final_on_endpoint(fake_sherpa, model_dir):
    """emit_final_on_endpoint=True emits a final as soon as an endpoint fires."""
    rec = fake_sherpa.OnlineRecognizer
    rec.text = "kaixo"
    rec.endpoint = True
    try:
        service = SherpaOnnxSTTService(
            model_dir=str(model_dir), emit_final_on_endpoint=True
        )
        frames = [
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=_pcm(), sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ]
        down, _ = await run_test(service, frames_to_send=frames)

        finals = [f for f in down if isinstance(f, TranscriptionFrame)]
        # One final emitted on the endpoint.
        assert any(f.text == "kaixo" and f.finalized for f in finals)
    finally:
        rec.endpoint = False
