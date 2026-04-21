import io
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

from src.types import AudioSample

CACHE_ROOT = Path.home() / ".cache" / "sa-voice-bench"
TARGET_SAMPLE_RATE = 16_000


def _to_16k_mono_wav(path: Path, raw_bytes: bytes | None, raw_path: str | None) -> None:
    """Read audio from bytes or file path, resample to 16kHz mono, save as WAV."""
    if raw_bytes:
        array, sr = sf.read(io.BytesIO(raw_bytes))
    elif raw_path:
        array, sr = sf.read(raw_path)
    else:
        raise ValueError("No audio data")

    if array.ndim > 1:
        array = array.mean(axis=1)

    if sr != TARGET_SAMPLE_RATE:
        from math import gcd
        g = gcd(TARGET_SAMPLE_RATE, sr)
        array = resample_poly(array, TARGET_SAMPLE_RATE // g, sr // g).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), array.astype(np.float32), TARGET_SAMPLE_RATE, subtype="PCM_16")


def load_fleurs_subset(
    language_code: str,
    split: str = "validation",
    n_samples: int = 5,
) -> list[AudioSample]:
    """Load up to n_samples from FLEURS for the given language code.

    Audio is cached to ~/.cache/sa-voice-bench/<language_code>/ as 16kHz
    mono WAV files. Re-running with the same arguments is instant.
    """
    cache_dir = CACHE_ROOT / language_code
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "google/fleurs",
        language_code,
        split=split,
        trust_remote_code=True,  # required for datasets<3.0 with loading scripts
    )
    # decode=False gives us raw bytes/path without needing librosa
    dataset = dataset.cast_column("audio", Audio(decode=False))

    samples: list[AudioSample] = []
    for row in dataset.select(range(min(n_samples, len(dataset)))):
        speaker_id = str(row.get("speaker_id", "unknown"))
        sample_id = str(row.get("id", row.get("num_samples", abs(hash(row["transcription"])) % 10**8)))

        wav_path = cache_dir / f"{sample_id}.wav"
        if not wav_path.exists():
            audio = row["audio"]
            _to_16k_mono_wav(wav_path, audio.get("bytes"), audio.get("path"))

        samples.append(
            AudioSample(
                audio_path=str(wav_path),
                reference_text=row["transcription"],
                language_code=language_code,
                speaker_id=speaker_id,
                metadata={"split": split, "sample_id": sample_id},
            )
        )

    return samples


if __name__ == "__main__":
    print("Loading 5 Hindi samples...")
    hindi = load_fleurs_subset("hi_in", n_samples=5)
    for s in hindi:
        print(f"  [{s.language_code}] {s.reference_text[:80]}")

    print("\nLoading 5 Nepali samples...")
    nepali = load_fleurs_subset("ne_np", n_samples=5)
    for s in nepali:
        print(f"  [{s.language_code}] {s.reference_text[:80]}")
