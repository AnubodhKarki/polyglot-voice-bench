import time
from pathlib import Path

import soundfile as sf
from groq import Groq
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.providers.base import TranscriptionProvider
from src.types import AudioSample, TranscriptionResult

MODEL = "whisper-large-v3"
COST_PER_MINUTE_USD = 0.00185  # $0.111/hour on Groq


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg or "503" in msg


class GroqWhisperLargeV3Provider(TranscriptionProvider):
    def __init__(self, api_key: str) -> None:
        self._client = Groq(api_key=api_key)

    @property
    def name(self) -> str:
        return "groq_whisper"

    def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        try:
            info = sf.info(sample.audio_path)
            duration_s = info.duration
        except Exception:
            duration_s = 0.0

        try:
            return self._transcribe_with_retry(sample, duration_s)
        except Exception as exc:
            return TranscriptionResult(
                text="",
                provider=self.name,
                model=MODEL,
                language_hint=sample.language_code,
                audio_duration_s=duration_s,
                api_cost_usd=None,
                error=str(exc),
            )

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _transcribe_with_retry(self, sample: AudioSample, duration_s: float) -> TranscriptionResult:
        # FLEURS uses "hi_in" / "ne_np"; Whisper expects ISO-639-1 "hi" / "ne"
        lang = sample.language_code.split("_")[0]

        with open(sample.audio_path, "rb") as f:
            audio_bytes = f.read()

        filename = Path(sample.audio_path).name
        t0 = time.monotonic()
        response = self._client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=MODEL,
            language=lang,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        cost = (duration_s / 60.0) * COST_PER_MINUTE_USD

        return TranscriptionResult(
            text=response.text,
            provider=self.name,
            model=MODEL,
            language_hint=lang,
            latency_ms=latency_ms,
            audio_duration_s=duration_s,
            api_cost_usd=cost,
        )
