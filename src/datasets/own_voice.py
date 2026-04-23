import csv
from pathlib import Path

from src.types import AudioSample

DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "own_voice"
AUDIO_DIR = DATA_ROOT / "audio"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"


def load_own_voice_subset(language: str | None = None) -> list[AudioSample]:
    """Load own-voice samples from data/own_voice/manifest.csv.

    Silently skips rows whose audio file doesn't exist yet — the manifest is
    pre-populated before recording so the infrastructure can be tested beforehand.

    language: ISO-style code used in manifest (ne_np / hi_en / en_au / ne_en).
              Pass None to load all available languages.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    samples: list[AudioSample] = []
    with open(MANIFEST_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if language is not None and row["language"] != language:
                continue

            audio_path = AUDIO_DIR / row["filename"]
            if not audio_path.exists():
                continue

            samples.append(
                AudioSample(
                    audio_path=str(audio_path),
                    reference_text=row["reference_text"],
                    language_code=row["language"],
                    speaker_id=row["speaker_id"],
                    metadata={
                        "filename": row["filename"],
                        "category": row["category"],
                        "code_switch_density": row.get("code_switch_density", ""),
                        "license": row["license"],
                        "sample_id": Path(row["filename"]).stem,
                    },
                )
            )

    return samples


def manifest_status() -> dict[str, tuple[int, int]]:
    """Return {language: (recorded, total)} counts from the manifest."""
    totals: dict[str, int] = {}
    recorded: dict[str, int] = {}

    if not MANIFEST_PATH.exists():
        return {}

    with open(MANIFEST_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lang = row["language"]
            totals[lang] = totals.get(lang, 0) + 1
            if (AUDIO_DIR / row["filename"]).exists():
                recorded[lang] = recorded.get(lang, 0) + 1

    return {lang: (recorded.get(lang, 0), totals[lang]) for lang in totals}


if __name__ == "__main__":
    print("Manifest status (recorded / total):")
    for lang, (done, total) in manifest_status().items():
        bar = "█" * done + "░" * (total - done)
        print(f"  {lang:8s}  {bar}  {done}/{total}")
    print()
    all_samples = load_own_voice_subset()
    print(f"Total loadable samples: {len(all_samples)}")
