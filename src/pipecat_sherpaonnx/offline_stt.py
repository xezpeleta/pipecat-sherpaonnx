"""Offline (non-streaming) sherpa-onnx STT service for Pipecat.

Uses :class:`sherpa_onnx.OfflineRecognizer` to transcribe complete speech
segments delimited by VAD. This works with any offline transducer model (e.g.
NeMo Parakeet TDT), as well as Paraformer, CTC, and Whisper models exported for
sherpa-onnx — in short, any model whose sherpa-onnx export targets the
*offline* recognizer.

The base :class:`~pipecat.services.stt_service.SegmentedSTTService` buffers
audio between VAD start/stop events; each complete utterance is decoded offline
in a worker thread (so the event loop is never blocked) and emitted as a single
finalized :class:`~pipecat.frames.frames.TranscriptionFrame`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from ._common import pcm_bytes_to_float32, resolve_transducer_files

# Module-level recognizer cache, shared across service instances. Pipecat
# pipelines are built per connection, so without this cache every WebSocket
# reconnect would reload the (typically 100+ MB) ONNX weights from disk —
# measured at ~2.4 s for a 623 MB encoder. Recognizers are immutable and
# thread-safe for inference, so sharing one across connections is safe.
_RECOGNIZER_CACHE: dict[tuple, Any] = {}


class SherpaOnnxOfflineSTTService(SegmentedSTTService):
    """VAD-segmented STT backed by ``sherpa_onnx.OfflineRecognizer``.

    Audio is buffered by the inherited ``SegmentedSTTService`` between VAD
    events; on each VAD stop the buffered segment (padded with a little
    trailing silence so the model can finish the last word) is decoded offline
    and yielded as a finalized transcript.

    Example::

        stt = SherpaOnnxOfflineSTTService(
            model_dir="/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            model_type="nemo_transducer",
            num_threads=4,
            language=Language.EU,
        )

    Args:
        model_dir: Directory containing the encoder/decoder/joiner ``.onnx``
            files and ``tokens.txt``. If given, the files are auto-discovered;
            explicit ``encoder``/``decoder``/``joiner``/``tokens`` paths take
            precedence and may be relative to this directory.
        encoder: Path to the encoder ONNX file (overrides auto-discovery).
        decoder: Path to the decoder ONNX file (overrides auto-discovery).
        joiner: Path to the joiner ONNX file (overrides auto-discovery).
        tokens: Path to ``tokens.txt`` (overrides auto-discovery).
        num_threads: Number of ONNX Runtime inference threads.
        decoding_method: ``"greedy_search"`` (default) or
            ``"modified_beam_search"``.
        model_type: sherpa-onnx model type. ``"transducer"`` (default) works
            for standard Zipformer transducers; use ``"nemo_transducer"`` for
            NeMo Parakeet TDT models.
        provider: ONNX Runtime execution provider (``"cpu"``, ``"cuda"``,
            ``"coreml"``).
        language: Language hint forwarded on every transcription frame.
        sample_rate: Expected audio sample rate (Hz). Defaults to 16000.
        hotwords_file: Optional path to a hotwords file (beam search only).
        hotwords_score: Score boost applied to hotwords.
        trailing_silence_secs: Seconds of silence appended to each segment
            before decoding, so the model can finish the final word.
        **kwargs: Additional arguments forwarded to ``SegmentedSTTService``.
    """

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        encoder: str | None = None,
        decoder: str | None = None,
        joiner: str | None = None,
        tokens: str | None = None,
        num_threads: int = 1,
        decoding_method: str = "greedy_search",
        model_type: str = "transducer",
        provider: str = "cpu",
        language: Language | str | None = Language.EU,
        sample_rate: int = 16000,
        hotwords_file: str = "",
        hotwords_score: float = 1.5,
        trailing_silence_secs: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            trailing_silence_secs=trailing_silence_secs,
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
        self._hotwords_file = hotwords_file
        self._hotwords_score = hotwords_score
        self._language = language
        self._recognizer: Any = None

    @property
    def wants_wav_segments(self) -> bool:
        """Return ``False``: the recognizer reads raw float32 samples.

        The inherited default wraps segments in a WAV container (what cloud
        upload APIs expect). sherpa-onnx reads raw mono PCM via
        ``accept_waveform``, so a WAV header would be misread as audio samples.
        """
        return False

    async def start(self, frame: Frame) -> None:
        await super().start(frame)
        await self._load()

    async def _load(self) -> None:
        if self._recognizer is not None:
            return
        # Lazy import: sherpa-onnx is a heavy wheel and is only needed once the
        # pipeline actually starts. This also keeps `import pipecat_sherpaonnx`
        # cheap and lets the unit tests inject a fake recognizer.
        import sherpa_onnx

        enc, dec, joi, tok = resolve_transducer_files(
            self._model_dir, self._encoder, self._decoder, self._joiner, self._tokens
        )
        cache_key = (
            enc, dec, joi, tok,
            self._num_threads, self._decoding_method, self._model_type,
            self._provider, self._hotwords_file, self._hotwords_score,
        )
        cached = _RECOGNIZER_CACHE.get(cache_key)
        if cached is not None:
            self._recognizer = cached
            return
        logger.info(f"{self.name}: loading sherpa-onnx offline model (tokens={tok})")
        recognizer = await asyncio.to_thread(
            sherpa_onnx.OfflineRecognizer.from_transducer,
            encoder=enc,
            decoder=dec,
            joiner=joi,
            tokens=tok,
            num_threads=self._num_threads,
            decoding_method=self._decoding_method,
            model_type=self._model_type,
            provider=self._provider,
            hotwords_file=self._hotwords_file,
            hotwords_score=self._hotwords_score,
        )
        _RECOGNIZER_CACHE[cache_key] = recognizer
        self._recognizer = recognizer
        logger.info(f"{self.name}: sherpa-onnx offline model ready")

    def _decode(self, audio: bytes) -> str:
        """Decode one PCM segment. Runs in a worker thread.

        A fresh offline stream is created per segment (the standard sherpa-onnx
        pattern for offline recognition).
        """
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, pcm_bytes_to_float32(audio))
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a buffered segment and yield a finalized transcript."""
        if self._recognizer is None:
            await self._load()
        await self.start_processing_metrics()
        try:
            text = await asyncio.to_thread(self._decode, audio)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception(f"{self.name}: offline decoding failed")
            await self.push_error(f"{self}: offline decoding failed: {e}", exception=e)
            await self.stop_processing_metrics()
            return
        await self.stop_processing_metrics()
        if text:
            logger.debug(f"{self.name}: transcription: [{text}]")
            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=self._language,
            )
