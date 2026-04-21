import time

import soundfile as sf
from deepgram import DeepgramClient
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.providers.base import TranscriptionProvider
from src.types import AudioSample, TranscriptionResult

MODEL = "nova-3"
LANGUAGE = "multi"
COST_PER_MINUTE_USD = 0.0092


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg or "5" in str(getattr(exc, "status_code", ""))


class DeepgramNova3MultiProvider(TranscriptionProvider):
    def __init__(self, api_key: str) -> None:
        self._client = DeepgramClient(api_key=api_key)

    @property
    def name(self) -> str:
        return "deepgram"

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
                language_hint=LANGUAGE,
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
        with open(sample.audio_path, "rb") as f:
            audio_bytes = f.read()

        t0 = time.monotonic()
        response = self._client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=MODEL,
            language=LANGUAGE,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        transcript = (
            response.results.channels[0].alternatives[0].transcript
            if response.results and response.results.channels
            else ""
        )

        cost = (duration_s / 60.0) * COST_PER_MINUTE_USD

        return TranscriptionResult(
            text=transcript,
            provider=self.name,
            model=MODEL,
            language_hint=LANGUAGE,
            latency_ms=latency_ms,
            audio_duration_s=duration_s,
            api_cost_usd=cost,
            raw_response=response,
        )
