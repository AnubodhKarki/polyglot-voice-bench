from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptionResult:
    text: str
    provider: str
    model: str
    language_hint: str
    latency_ms: float = 0.0
    audio_duration_s: float = 0.0
    api_cost_usd: float | None = None
    raw_response: Any = field(default=None, repr=False)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class AudioSample:
    audio_path: str
    reference_text: str
    language_code: str
    speaker_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
