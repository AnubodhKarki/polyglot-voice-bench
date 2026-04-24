"""
Split a single long recording into per-utterance WAV files.

Usage:
    python scripts/split_recording.py <recording.wav> [options]

Record yourself reading every utterance in data/own_voice/manifest.csv in order,
pausing for at least 2 seconds between each sentence. Then run this script to
split the recording into individual files named after each row in the manifest.

Options:
    --manifest      Path to manifest CSV (default: data/own_voice/manifest.csv)
    --out-dir       Output directory for WAV files (default: data/own_voice)
    --silence-sec   Minimum pause duration in seconds to treat as a boundary (default: 1.5)
    --silence-db    Audio level in dBFS below which a frame is considered silent (default: -40)
    --dry-run       Print detected segments without writing files
    --padding-ms    Milliseconds of silence to keep at the start/end of each clip (default: 80)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

def rms_db(samples: np.ndarray) -> float:
    """Return RMS level in dBFS (full scale = 0 dB)."""
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms == 0:
        return -96.0
    return 20 * np.log10(rms / 32768.0)


def detect_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
    silence_db: float = -40.0,
    silence_sec: float = 1.5,
    frame_ms: int = 20,
    padding_ms: int = 80,
) -> list[tuple[int, int]]:
    """
    Return a list of (start_sample, end_sample) for each speech region.

    A speech region ends when audio stays below `silence_db` for at least
    `silence_sec` seconds, and starts again when audio exceeds `silence_db`.
    """
    frame_samples = int(sample_rate * frame_ms / 1000)
    silence_frames = int(silence_sec * 1000 / frame_ms)
    pad_samples = int(sample_rate * padding_ms / 1000)

    # Mono mix if multi-channel
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Convert float audio (−1..1) to int16-equivalent scale for dB calc
    if audio.dtype.kind == "f":
        pcm = (audio * 32767).astype(np.int16)
    else:
        pcm = audio.astype(np.int16)

    n_frames = len(pcm) // frame_samples
    is_silent = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        chunk = pcm[i * frame_samples : (i + 1) * frame_samples]
        is_silent[i] = rms_db(chunk) < silence_db

    # Smooth: require `silence_frames` consecutive silent frames for a gap
    regions: list[tuple[int, int]] = []
    in_speech = False
    speech_start = 0
    silent_run = 0

    for i, silent in enumerate(is_silent):
        if not in_speech:
            if not silent:
                in_speech = True
                speech_start = i
                silent_run = 0
        else:
            if silent:
                silent_run += 1
                if silent_run >= silence_frames:
                    # End of speech region
                    speech_end = i - silent_run + 1
                    start_s = max(0, speech_start * frame_samples - pad_samples)
                    end_s = min(len(pcm), speech_end * frame_samples + pad_samples)
                    regions.append((start_s, end_s))
                    in_speech = False
                    silent_run = 0
            else:
                silent_run = 0

    # Close final region if still in speech at end
    if in_speech:
        start_s = max(0, speech_start * frame_samples - pad_samples)
        end_s = len(pcm)
        regions.append((start_s, end_s))

    return regions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> list[str]:
    filenames: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(row["filename"])
    return filenames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a long recording into per-utterance WAV files."
    )
    parser.add_argument("recording", type=Path, help="Path to the long WAV recording")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/own_voice/manifest.csv"),
        help="Path to manifest CSV (default: data/own_voice/manifest.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/own_voice"),
        help="Output directory (default: data/own_voice)",
    )
    parser.add_argument(
        "--silence-sec",
        type=float,
        default=1.5,
        help="Min silence duration in seconds to split on (default: 1.5)",
    )
    parser.add_argument(
        "--silence-db",
        type=float,
        default=-40.0,
        help="Silence threshold in dBFS (default: -40)",
    )
    parser.add_argument(
        "--padding-ms",
        type=int,
        default=80,
        help="Milliseconds of audio to keep around each segment (default: 80)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detected segments without writing files",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        print(f"ERROR: Recording not found: {args.recording}", file=sys.stderr)
        sys.exit(1)
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    filenames = load_manifest(args.manifest)
    print(f"Manifest: {len(filenames)} utterances")

    print(f"Loading {args.recording} …")
    audio, sr = sf.read(str(args.recording), always_2d=False)
    duration = len(audio) / sr
    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {duration:.1f} s")
    print(f"  Channels    : {'mono' if audio.ndim == 1 else audio.shape[1]}")

    print(
        f"\nDetecting speech regions (silence ≥ {args.silence_sec}s below {args.silence_db} dBFS) …"
    )
    regions = detect_speech_regions(
        audio,
        sr,
        silence_db=args.silence_db,
        silence_sec=args.silence_sec,
        padding_ms=args.padding_ms,
    )
    print(f"  Found {len(regions)} region(s)")

    if len(regions) != len(filenames):
        print(
            f"\n⚠  WARNING: {len(regions)} regions detected but {len(filenames)} utterances in manifest."
        )
        print("   Try adjusting --silence-sec or --silence-db and re-run.")
        if not args.dry_run:
            print("   Use --dry-run to inspect without writing files.")

    print()
    n = min(len(regions), len(filenames))
    mono = audio.ndim == 1

    for i, ((start, end), fname) in enumerate(zip(regions, filenames), 1):
        seg_dur = (end - start) / sr
        print(f"  [{i:3d}/{n}] {fname:<25} {start/sr:6.2f}s – {end/sr:6.2f}s  ({seg_dur:.2f}s)")
        if not args.dry_run:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / fname
            segment = audio[start:end] if mono else audio[start:end, :]
            sf.write(str(out_path), segment, sr, subtype="PCM_16")

    if args.dry_run:
        print("\nDry-run — no files written.")
    else:
        print(f"\n✓ Wrote {n} file(s) to {args.out_dir}/")
        if len(regions) != len(filenames):
            print(
                f"  {abs(len(regions) - len(filenames))} utterance(s) were not matched."
            )


if __name__ == "__main__":
    main()
