"""Streaming sherpa-onnx STT service for Pipecat.

Uses :class:`sherpa_onnx.OnlineRecognizer` for true streaming recognition.
Only for **online-capable** transducer models (e.g. streaming Zipformer) —
models whose sherpa-onnx export targets the *online* recognizer.

Non-streaming models (NeMo Parakeet TDT, Whisper, Paraformer, CTC, ...) are not
compatible with the online recognizer and must use
:class:`~pipecat_sherpaonnx.offline_stt.SherpaOnnxOfflineSTTService` instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from ._common import join_texts, pcm_bytes_to_float32, resolve_transducer_files


class SherpaOnnxSTTService(STTService):
    """Streaming STT backed by ``sherpa_onnx.OnlineRecognizer``.

    Audio chunks are fed to a single persistent online stream. Partial
    hypotheses are emitted as :class:`InterimTranscriptionFrame` while the user
    speaks, and finalized utterances as
    :class:`TranscriptionFrame`.

    Turn boundaries are driven by the pipeline's VAD: by default a single
    finalized transcript is emitted per user turn (on VAD stop). sherpa-onnx's
    endpoint detection is used internally to keep the recognition stream fresh
    across endpoints and to stitch multi-sentence turns together correctly.

    Set ``emit_final_on_endpoint=True`` to additionally emit a finalized
    transcript on each detected endpoint — useful for pipelines that run without
    a VAD.

    Example::

        stt = SherpaOnnxSTTService(
            model_dir="/models/sherpa-onnx-streaming-zipformer-...",
            model_type="zipformer2",
            num_threads=4,
            language=Language.EN,
        )

    Args:
        model_dir: Directory containing the encoder/decoder/joiner ``.onnx``
            files and ``tokens.txt``. Auto-discovers the files; explicit paths
            take precedence.
        encoder: Path to the encoder ONNX file (overrides auto-discovery).
        decoder: Path to the decoder ONNX file (overrides auto-discovery).
        joiner: Path to the joiner ONNX file (overrides auto-discovery).
        tokens: Path to ``tokens.txt`` (overrides auto-discovery).
        num_threads: Number of ONNX Runtime inference threads.
        decoding_method: ``"greedy_search"`` (default) or
            ``"modified_beam_search"``.
        model_type: sherpa-onnx online model type, e.g. ``"zipformer2"``.
        provider: ONNX Runtime execution provider (``"cpu"``, ``"cuda"``).
        enable_endpoint_detection: Whether sherpa-onnx endpoint detection runs.
            Kept on by default to segment the internal stream; does not emit
            final transcripts unless ``emit_final_on_endpoint`` is set.
        rule1_min_trailing_silence: Endpoint rule 1: trailing silence (s).
        rule2_min_trailing_silence: Endpoint rule 2: trailing silence (s).
        rule3_min_utterance_length: Endpoint rule 3: max utterance length (s).
        emit_final_on_endpoint: Emit a finalized transcript on each detected
            endpoint (in addition to the VAD-stop final). Defaults to ``False``.
        language: Language hint forwarded on every transcription frame.
        sample_rate: Expected audio sample rate (Hz). Defaults to 16000.
        hotwords_file: Optional path to a hotwords file (beam search only).
        hotwords_score: Score boost applied to hotwords.
        **kwargs: Additional arguments forwarded to ``STTService``.
    """

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        encoder: str | None = None,
        decoder: str | None = None,
        joiner: str | None = None,
        tokens: str | None = None,
        num_threads: int = 2,
        decoding_method: str = "greedy_search",
        model_type: str = "zipformer2",
        provider: str = "cpu",
        enable_endpoint_detection: bool = True,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        emit_final_on_endpoint: bool = False,
        language: Language | str | None = Language.EU,
        sample_rate: int = 16000,
        hotwords_file: str = "",
        hotwords_score: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model=None, language=language),
            **kwargs,
        )
        self._model_dir = model_dir
        self._encoder = encoder
        self._decoder = decoder
        self._joiner = joiner
        self._tokens = tokens
        self._num_threads = num_threads
        self._decoding_method = decoding_method
        self._model_type = model_type
        self._provider = provider
        self._enable_endpoint = enable_endpoint_detection
        self._rule1 = rule1_min_trailing_silence
        self._rule2 = rule2_min_trailing_silence
        self._rule3 = rule3_min_utterance_length
        self._emit_final_on_endpoint = emit_final_on_endpoint
        self._language = language
        self._hotwords_file = hotwords_file
        self._hotwords_score = hotwords_score

        self._recognizer: Any = None
        self._stream: Any = None
        # Text committed across sherpa-onnx endpoints within the current user
        # turn (used to stitch multi-sentence turns into one final transcript).
        self._segment_text = ""
        # Last partial we emitted, to suppress duplicate InterimTranscriptionFrames.
        self._last_partial = ""

    async def start(self, frame: Frame) -> None:
        await super().start(frame)
        await self._load()

    async def _load(self) -> None:
        if self._recognizer is not None:
            return
        # Lazy import — see SherpaOnnxOfflineSTTService._load for rationale.
        import sherpa_onnx

        enc, dec, joi, tok = resolve_transducer_files(
            self._model_dir, self._encoder, self._decoder, self._joiner, self._tokens
        )
        logger.info(f"{self.name}: loading sherpa-onnx online model (tokens={tok})")
        self._recognizer = await asyncio.to_thread(
            sherpa_onnx.OnlineRecognizer.from_transducer,
            tokens=tok,
            encoder=enc,
            decoder=dec,
            joiner=joi,
            num_threads=self._num_threads,
            decoding_method=self._decoding_method,
            model_type=self._model_type,
            provider=self._provider,
            enable_endpoint_detection=self._enable_endpoint,
            rule1_min_trailing_silence=self._rule1,
            rule2_min_trailing_silence=self._rule2,
            rule3_min_utterance_length=self._rule3,
            hotwords_file=self._hotwords_file,
            hotwords_score=self._hotwords_score,
        )
        self._stream = await asyncio.to_thread(self._recognizer.create_stream)
        logger.info(f"{self.name}: sherpa-onnx online model ready")

    def _process_chunk(self, audio: bytes) -> tuple[str, bool]:
        """Feed one chunk to the online stream and read the current hypothesis.

        Runs in a worker thread so the (potentially slow) ONNX forward pass
        does not block the event loop. Returns ``(text, is_endpoint)``.
        """
        samples = pcm_bytes_to_float32(audio)
        self._stream.accept_waveform(self.sample_rate, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._recognizer.get_result(self._stream).strip()
        is_endpoint = self._recognizer.is_endpoint(self._stream)
        return text, is_endpoint

    def _flush_remaining(self) -> str:
        """Decode any buffered audio left in the stream. Runs in a worker thread."""
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        return self._recognizer.get_result(self._stream).strip()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Feed an audio chunk and yield interim/final transcription frames."""
        if self._recognizer is None:
            await self._load()

        text, is_endpoint = await asyncio.to_thread(self._process_chunk, audio)

        partial = join_texts(self._segment_text, text)
        if partial and partial != self._last_partial:
            self._last_partial = partial
            yield InterimTranscriptionFrame(
                text=partial,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=self._language,
            )

        if is_endpoint:
            if text:
                if self._emit_final_on_endpoint:
                    final = join_texts(self._segment_text, text)
                    self._segment_text = ""
                    self._last_partial = ""
                    self._recognizer.reset(self._stream)
                    yield TranscriptionFrame(
                        text=final,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        language=self._language,
                        finalized=True,
                    )
                    return
                # Accumulate the committed text across endpoints within this turn
                # so a multi-sentence turn produces one correct final transcript.
                self._segment_text = partial
            self._recognizer.reset(self._stream)
            self._last_partial = ""

    async def _handle_vad_user_stopped_speaking(
        self, frame: VADUserStoppedSpeakingFrame
    ) -> None:
        """Flush a finalized transcript for the turn on VAD stop."""
        await super()._handle_vad_user_stopped_speaking(frame)
        if self._stream is None:
            return
        try:
            text = await asyncio.to_thread(self._flush_remaining)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception(f"{self.name}: flushing final transcript failed")
            await self.push_error(f"{self}: flushing failed: {e}", exception=e)
            return
        final = join_texts(self._segment_text, text)
        if final:
            await self.push_frame(
                TranscriptionFrame(
                    text=final,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=self._language,
                    finalized=True,
                )
            )
        self._recognizer.reset(self._stream)
        self._segment_text = ""
        self._last_partial = ""
