import time

import numpy as np
import soundfile as sf
from assemblyai.streaming.v3 import (
    Encoding,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
)

from src.providers.base import TranscriptionProvider
from src.types import AudioSample, TranscriptionResult

MODEL = "u3-rt-pro"
SAMPLE_RATE = 16_000
CHUNK_MS = 100
# 100ms of int16 mono at 16kHz = 1600 samples × 2 bytes = 3200 bytes per chunk
CHUNK_BYTES = (SAMPLE_RATE * CHUNK_MS // 1000) * 2
COST_PER_HOUR_USD = 0.45  # Universal-3 Pro streaming, verified assemblyai.com/pricing


class AssemblyAIUniversal3ProProvider(TranscriptionProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "assemblyai"

    def transcribe(self, sample: AudioSample) -> TranscriptionResult:
        try:
            data, sr = sf.read(sample.audio_path, dtype="int16", always_2d=False)
            duration_s = len(data) / sr
        except Exception:
            duration_s = 0.0
            data = np.array([], dtype="int16")

        try:
            return self._stream_and_transcribe(sample, data, duration_s)
        except Exception as exc:
            return TranscriptionResult(
                text="",
                provider=self.name,
                model=MODEL,
                language_hint=sample.language_code,
                audio_duration_s=duration_s,
                error=str(exc),
            )

    def _stream_and_transcribe(
        self,
        sample: AudioSample,
        pcm_data: np.ndarray,
        duration_s: float,
    ) -> TranscriptionResult:
        final_turns: list[str] = []
        error_holder: list[str | None] = [None]

        client = StreamingClient(StreamingClientOptions(api_key=self._api_key))

        def on_turn(_, event: TurnEvent) -> None:
            # Ignore partial turns (end_of_turn=False); only keep finalized turns.
            if event.end_of_turn and event.transcript:
                final_turns.append(event.transcript)

        def on_error(_, error: StreamingError) -> None:
            error_holder[0] = str(error)

        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Error, on_error)

        t0 = time.monotonic()
        client.connect(
            StreamingParameters(
                sample_rate=SAMPLE_RATE,
                speech_model=MODEL,
                encoding=Encoding.pcm_s16le,
                language_detection=True,
            )
        )

        # Stream raw PCM at 1× real-time to avoid error 4029 ("Client sent audio too fast").
        # Each CHUNK_MS of audio is sent then we sleep CHUNK_MS before the next chunk.
        raw_bytes = pcm_data.tobytes()
        chunk_sleep_s = CHUNK_MS / 1000.0
        for i in range(0, len(raw_bytes), CHUNK_BYTES):
            client.stream(raw_bytes[i : i + CHUNK_BYTES])
            time.sleep(chunk_sleep_s)

        # Sends TerminateSession then blocks until the server emits TerminationEvent.
        client.disconnect(terminate=True)
        latency_ms = (time.monotonic() - t0) * 1000

        if error_holder[0]:
            raise RuntimeError(error_holder[0])

        return TranscriptionResult(
            text=" ".join(final_turns),
            provider=self.name,
            model=MODEL,
            language_hint=sample.language_code,
            latency_ms=latency_ms,
            audio_duration_s=duration_s,
            api_cost_usd=(duration_s / 3600.0) * COST_PER_HOUR_USD,
        )
