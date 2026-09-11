# pipecat-sherpaonnx

[Pipecat](https://github.com/pipecat-ai/pipecat) STT services for
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

Run speech-to-text **locally on CPU** with any of the 100+ sherpa-onnx ASR
models, in any language — no cloud API keys, no GPU required.

## Services

| Service | Recognizer | Use when |
|---|---|---|
| `SherpaOnnxOfflineSTTService` | `OfflineRecognizer` | Your model is **non-streaming** (the common case: Parakeet TDT, Whisper, Paraformer, CTC). VAD segments audio; one finalized transcript per utterance. |
| `SherpaOnnxSTTService` | `OnlineRecognizer` | Your model is **online/streaming-capable** (streaming Zipformer, …). Emits interim partials as the user speaks. |

> **Which one do I need?** If your model's sherpa-onnx export targets the
> **offline** recognizer, use `SherpaOnnxOfflineSTTService`. The
> `OnlineRecognizer` only works with models specifically trained/exported for
> streaming. When in doubt, use the offline service — it works with the
> overwhelming majority of sherpa-onnx models.
>
> Example: the Basque
> [`xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx`](https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx)
> model is a non-streaming NeMo Parakeet TDT model → use
> `SherpaOnnxOfflineSTTService` with `model_type="nemo_transducer"`.

## Install

```bash
pip install pipecat-sherpaonnx
```

## Quick start (offline, with VAD)

```python
from pipecat_sherpaonnx import SherpaOnnxOfflineSTTService
from pipecat.transcriptions.language import Language

stt = SherpaOnnxOfflineSTTService(
    model_dir="/path/to/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    model_type="nemo_transducer",
    num_threads=4,
    language=Language.EU,
)
```

Place `stt` in a Pipecat pipeline behind a VAD (e.g. `SileroVADAnalyzer`) and an
audio input source. The VAD delimits utterances; each one is decoded offline and
emitted as a finalized `TranscriptionFrame`.

### File auto-discovery

Point `model_dir` at the unpacked model directory and the encoder/decoder/
joiner/tokens files are found automatically (matching `encoder*.onnx`,
`decoder*.onnx`, `joiner*.onnx`, `tokens.txt`). Pass `encoder` / `decoder` /
`joiner` / `tokens` explicitly to override.

## Streaming

```python
from pipecat_sherpaonnx import SherpaOnnxSTTService
from pipecat.transcriptions.language import Language

stt = SherpaOnnxSTTService(
    model_dir="/path/to/sherpa-onnx-streaming-zipformer-...",
    model_type="zipformer2",
    num_threads=4,
    language=Language.EN,
)
```

Interim partials are emitted while the user speaks; a finalized transcript is
emitted on VAD stop. Set `emit_final_on_endpoint=True` for pipelines without a
VAD.

## How it works

- **Audio conversion** — Pipecat delivers 16-bit PCM bytes; these are converted
  to float32 in `[-1, 1]` (what sherpa-onnx `accept_waveform` expects).
- **Offline service** subclasses Pipecat's `SegmentedSTTService`: audio is
  buffered between VAD events and each complete segment is decoded in a worker
  thread (the event loop is never blocked by inference).
- **Streaming service** subclasses `STTService` and feeds every chunk to a
  persistent `OnlineRecognizer` stream, emitting interim hypotheses and using
  sherpa-onnx endpoint detection to keep multi-sentence turns correct.
- `sherpa_onnx` is imported lazily on service start, so `import
  pipecat_sherpaonnx` is cheap.

## Development

```bash
uv sync                 # install deps
uv run pytest           # run tests (use a fake recognizer; no model download needed)
```

## License

BSD-2-Clause.
