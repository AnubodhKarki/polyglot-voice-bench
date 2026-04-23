"""v3 entry point: 3 providers × 4 own-voice language categories.

Language categories:
  ne_np  — Nepali (30 utterances)
  hi_en  — Hinglish, low/medium/high code-switch density (30 utterances)
  en_au  — English in own voice, Australian accent (30 utterances)
  ne_en  — Nepali-English code-switch (30 utterances)

Results are written to results/v3/ to keep them separate from FLEURS runs.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

console = Console()
RESULTS_DIR = Path(__file__).parent.parent / "results" / "v3"
LANGUAGES = ["ne_np", "hi_en", "en_au", "ne_en"]

LANG_LABELS = {
    "ne_np": "Nepali",
    "hi_en": "Hinglish",
    "en_au": "English (own voice)",
    "ne_en": "Nepali-English switch",
}


def _fmt_wer(metrics: dict) -> str:
    if metrics.get("wer") is None:
        n_err = metrics.get("n_errored", "?")
        n_tot = metrics.get("n_total", "?")
        return f"N/A ({n_err}/{n_tot} errors)"
    s = f"{metrics['wer']:.3f}"
    ci = metrics.get("wer_ci")
    if ci:
        s += f"  [{ci['lower']:.3f}–{ci['upper']:.3f}]"
    return s


def _fmt_cer(metrics: dict) -> str:
    if metrics.get("cer") is None:
        return "N/A"
    s = f"{metrics['cer']:.3f}"
    ci = metrics.get("cer_ci")
    if ci:
        s += f"  [{ci['lower']:.3f}–{ci['upper']:.3f}]"
    return s


def _print_comparison(all_metrics: dict, provider_names: list[str]) -> None:
    console.rule("[bold]v3 — Own-Voice 3-Provider Comparison[/bold]")
    table = Table(title="WER / CER — mean with 95% CI (own-voice dataset, CC-BY)")
    table.add_column("Language", style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Deepgram Nova-3")
    table.add_column("Whisper v3 (Groq)")
    table.add_column("AssemblyAI U3-Pro")

    for lang in LANGUAGES:
        label = LANG_LABELS[lang]
        for i, (metric_name, fmt_fn) in enumerate([("WER", _fmt_wer), ("CER", _fmt_cer)]):
            row = [label if i == 0 else "", metric_name]
            for name in provider_names:
                m = all_metrics.get((name, lang), {})
                row.append(fmt_fn(m))
            table.add_row(*row)

    console.print(table)


def main() -> None:
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    assemblyai_key = os.environ.get("ASSEMBLYAI_API_KEY", "")

    missing = [
        name
        for name, val in [
            ("DEEPGRAM_API_KEY", deepgram_key),
            ("GROQ_API_KEY", groq_key),
            ("ASSEMBLYAI_API_KEY", assemblyai_key),
        ]
        if not val
    ]
    if missing:
        for key in missing:
            console.print(f"[bold red]ERROR:[/bold red] {key} not set in .env")
        sys.exit(1)

    from src.datasets.own_voice import load_own_voice_subset, manifest_status
    from src.providers.assemblyai_universal3_pro import AssemblyAIUniversal3ProProvider
    from src.providers.deepgram_nova3_multi import DeepgramNova3MultiProvider
    from src.providers.groq_whisper_large_v3 import GroqWhisperLargeV3Provider
    from src.runner import run_benchmark

    providers = [
        DeepgramNova3MultiProvider(api_key=deepgram_key),
        GroqWhisperLargeV3Provider(api_key=groq_key),
        AssemblyAIUniversal3ProProvider(api_key=assemblyai_key),
    ]
    provider_names = [p.name for p in providers]

    console.rule("[bold]v3 — Own-Voice Benchmark[/bold]")

    status = manifest_status()
    console.print("Recording status:")
    for lang, (done, total) in status.items():
        bar = "█" * done + "░" * (total - done)
        console.print(f"  {lang:8s}  {bar}  {done}/{total}")
    console.print()

    total_available = sum(done for done, _ in status.values())
    if total_available == 0:
        console.print(
            "[bold yellow]No audio files found yet.[/bold yellow]\n"
            "Record your utterances following RECORDING_PROTOCOL.md, "
            "place WAV files in data/own_voice/audio/, then re-run."
        )
        sys.exit(0)

    samples_by_lang: dict = {}
    for lang in LANGUAGES:
        samples = load_own_voice_subset(language=lang)
        samples_by_lang[lang] = samples
        if samples:
            console.print(f"Loaded {len(samples)} {lang} samples.")

    all_metrics: dict = {}
    for lang in LANGUAGES:
        samples = samples_by_lang.get(lang, [])
        if not samples:
            console.print(f"[dim]Skipping {lang} — no audio files recorded yet.[/dim]")
            continue
        console.rule(f"[bold]{LANG_LABELS[lang]} ({lang})[/bold]")
        for provider in providers:
            metrics = run_benchmark(provider, samples, RESULTS_DIR, lang)
            all_metrics[(provider.name, lang)] = metrics

    if all_metrics:
        _print_comparison(all_metrics, provider_names)
        console.print(f"\nAll results written to [cyan]{RESULTS_DIR}[/cyan]")


if __name__ == "__main__":
    main()
