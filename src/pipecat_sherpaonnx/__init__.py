"""Pipecat STT services for sherpa-onnx.

Two services are provided:

* :class:`SherpaOnnxOfflineSTTService` — VAD-segmented, backed by
  ``sherpa_onnx.OfflineRecognizer``. For non-streaming transducer, Paraformer,
  CTC, and Whisper models exported for sherpa-onnx (e.g. NeMo Parakeet TDT).
  This is the right choice for most models, including the Basque
  ``parakeet-tdt-0.6b-v3`` model.
* :class:`SherpaOnnxSTTService` — streaming, backed by
  ``sherpa_onnx.OnlineRecognizer``. For online-capable transducer models
  (e.g. streaming Zipformer). Emits interim partials as the user speaks.

``sherpa_onnx`` is imported lazily (on service start), so importing this
package does not require the (heavy) sherpa-onnx wheel to be installed.
"""

from .offline_stt import SherpaOnnxOfflineSTTService
from .streaming_stt import SherpaOnnxSTTService

__all__ = ["SherpaOnnxOfflineSTTService", "SherpaOnnxSTTService"]

__version__ = "0.1.0"
