"""AssemblyAI Universal-2 batch transcription provider.

Universal-2 supports ~99 languages (including Hindi) via the standard REST API.
Contrast with Universal-3 Pro Streaming, which supports only 6 languages
(English, Spanish, German, French, Portuguese, Italian) via WebSocket.

This provider uses the synchronous Transcriber.transcribe() — no chunked
streaming, no real-time throttle. Each call uploads the file, waits for
AssemblyAI to process it, and returns the transcript.

Language mapping: for language codes not in AssemblyAI's Universal-2 supported
list (notably Nepali), we omit the language_code and enable language_detection
so AssemblyAI attempts its best guess. This is what a production developer
would do — and the result is a finding either way.
"""

import time

import assemblyai as aai
import soundfile as sf

from src.providers.base import TranscriptionProvider
from src.types import AudioSample, TranscriptionResult

MODEL = "universal"  # assemblyai.SpeechModel.universal
# Per-second pricing for Universal-2 batch (Pay As You Go, verified assemblyai.com/pricing)
COST_PER_SECOND_USD = 0.00041

# AssemblyAI BCP-47-compatible codes for languages we explicitly support.
# Languages absent from this map get language_detection=True instead.
_LANGUAGE_MAP: dict[str, str] = {
    "hi_in": "hi",
    "hi_en": "hi",   # Hinglish — closest supported code
    "en_us": "en_us",
    "en_au": "en_au",
    "en":    "en",
}


class AssemblyAIUniversal2Provider(TranscriptionProvider):
    def __init__(self, api_key: str) -> None:
        aai.settings.api_key = api_key

    @property
    def name(self) -> str:
        return "assemblyai_u2"

    def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        try:
            info = sf.info(sample.audio_path)
            duration_s = info.duration
        except Exception:
            duration_s = 0.0

        try:
            return self._transcribe(sample, duration_s)
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

    def _transcribe(self, sample: AudioSample, duration_s: float) -> TranscriptionResult:
        lang_code = _LANGUAGE_MAP.get(sample.language_code)

        if lang_code is not None:
            config = aai.TranscriptionConfig(
                speech_models=[aai.SpeechModel.universal],
                language_code=lang_code,
            )
        else:
            # Unsupported language (e.g. Nepali) — let AssemblyAI auto-detect.
            # Language-not-supported error or silent failure is itself a finding.
            config = aai.TranscriptionConfig(
                speech_models=[aai.SpeechModel.universal],
                language_detection=True,
            )

        transcriber = aai.Transcriber(config=config)

        t0 = time.monotonic()
        transcript = transcriber.transcribe(sample.audio_path)
        latency_ms = (time.monotonic() - t0) * 1000

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(transcript.error)

        return TranscriptionResult(
            text=transcript.text or "",
            provider=self.name,
            model=MODEL,
            language_hint=lang_code or "auto",
            latency_ms=latency_ms,
            audio_duration_s=duration_s,
            api_cost_usd=duration_s * COST_PER_SECOND_USD,
        )
