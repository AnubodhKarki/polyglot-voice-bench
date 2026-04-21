from abc import ABC, abstractmethod

from src.types import AudioSample, TranscriptionResult


class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MockProvider(TranscriptionProvider):
    """Returns a predetermined transcript. Used to test the runner without burning API quota."""

    def __init__(self, canned_text: str = "mock transcript"):
        self._canned_text = canned_text

    @property
    def name(self) -> str:
        return "mock"

    def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        return TranscriptionResult(
            text=self._canned_text,
            provider=self.name,
            model="mock-v0",
            language_hint=sample.language_code,
            latency_ms=0.0,
            audio_duration_s=0.0,
        )
