"""Shared test fixtures.

The tests exercise the Pipecat wiring (frame emission, VAD handling, model
loading) without downloading real sherpa-onnx models. They inject a fake
``sherpa_onnx`` module into ``sys.modules`` before the service lazily imports it
(in ``_load``), so no heavy wheel is required to run the suite.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fake OfflineRecognizer
# ---------------------------------------------------------------------------


class FakeOfflineStream:
    """Mimics sherpa_onnx's OfflineStream for tests."""

    def __init__(self, text: str = "kaixo") -> None:
        self.fed = False
        self.result = SimpleNamespace(text=text)

    def accept_waveform(self, sample_rate: int, samples: Any) -> None:
        self.fed = True


class FakeOfflineRecognizer:
    """Mimics sherpa_onnx.OfflineRecognizer for tests.

    ``text`` and ``instances`` are class attributes so tests can inspect the
    created recognizers and flip the canned transcript from one place.
    """

    text: str = "kaixo"
    instances: list["FakeOfflineRecognizer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.created_streams: list[FakeOfflineStream] = []
        FakeOfflineRecognizer.instances.append(self)

    @classmethod
    def from_transducer(cls, **kwargs: Any) -> "FakeOfflineRecognizer":
        return cls(**kwargs)

    def create_stream(self) -> FakeOfflineStream:
        s = FakeOfflineStream(text=FakeOfflineRecognizer.text)
        self.created_streams.append(s)
        return s

    def decode_stream(self, stream: FakeOfflineStream) -> None:  # noqa: D401
        pass


# ---------------------------------------------------------------------------
# Fake OnlineRecognizer
# ---------------------------------------------------------------------------


class FakeOnlineStream:
    def __init__(self) -> None:
        self.fed = False

    def accept_waveform(self, sample_rate: int, samples: Any) -> None:
        self.fed = True


class FakeOnlineRecognizer:
    """Mimics sherpa_onnx.OnlineRecognizer for tests.

    ``text``/``endpoint``/``instances`` are class attributes so tests can drive
    behaviour and inspect state without touching instances directly.
    """

    text: str = "kaixo"
    endpoint: bool = False
    instances: list["FakeOnlineRecognizer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.stream: FakeOnlineStream | None = None
        FakeOnlineRecognizer.instances.append(self)

    @classmethod
    def from_transducer(cls, **kwargs: Any) -> "FakeOnlineRecognizer":
        return cls(**kwargs)

    def create_stream(self) -> FakeOnlineStream:
        self.stream = FakeOnlineStream()
        return self.stream

    def is_ready(self, stream: FakeOnlineStream) -> bool:
        return False  # decode loop is a no-op in tests

    def decode_stream(self, stream: FakeOnlineStream) -> None:  # noqa: D401
        pass

    def get_result(self, stream: FakeOnlineStream) -> str:
        return FakeOnlineRecognizer.text if stream.fed else ""

    def is_endpoint(self, stream: FakeOnlineStream) -> bool:
        return FakeOnlineRecognizer.endpoint

    def reset(self, stream: FakeOnlineStream) -> None:
        stream.fed = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_sherpa():
    """Install a fake ``sherpa_onnx`` module for the duration of the test."""
    mod = types.ModuleType("sherpa_onnx")
    mod.OfflineRecognizer = FakeOfflineRecognizer
    mod.OnlineRecognizer = FakeOnlineRecognizer
    # Reset all shared class state between tests.
    FakeOfflineRecognizer.instances.clear()
    FakeOfflineRecognizer.text = "kaixo"
    FakeOnlineRecognizer.instances.clear()
    FakeOnlineRecognizer.text = "kaixo"
    FakeOnlineRecognizer.endpoint = False
    saved = sys.modules.get("sherpa_onnx")
    sys.modules["sherpa_onnx"] = mod
    try:
        yield mod
    finally:
        if saved is not None:
            sys.modules["sherpa_onnx"] = saved
        else:
            sys.modules.pop("sherpa_onnx", None)


@pytest.fixture
def model_dir(tmp_path):
    """Create a fake model directory with the expected file layout."""
    for name in (
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ):
        (tmp_path / name).write_bytes(b"\x00")
    return tmp_path
