"""v2 entry point: Deepgram Nova-3 vs Whisper Large v3 (Groq) vs AssemblyAI Universal-3 Pro
on FLEURS Hindi, Nepali, and English (en_us).

Note: FLEURS has only en_us for English — no en_in or en_au. Accent variants
require Mozilla Common Voice and will be addressed in v3.
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
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_SAMPLES = 25
LANGUAGES = ["hi_in", "ne_np", "en_us"]


def _fmt_wer(metrics: dict) -> str:
    if metrics.get("wer") is None:
        return f"N/A ({metrics.get('n_errored', '?')} errors)"
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
    console.rule("[bold]3-Provider Side-by-Side[/bold]")
    table = Table(title="WER / CER — mean with 95% CI (n=25 per language)")
    table.add_column("Language", style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Deepgram Nova-3")
    table.add_column("Whisper v3 (Groq)")
    table.add_column("AssemblyAI U3-Pro")

    for lang in LANGUAGES:
        for i, (metric_name, fmt_fn) in enumerate([("WER", _fmt_wer), ("CER", _fmt_cer)]):
            row = [lang if i == 0 else "", metric_name]
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

    from src.datasets.fleurs import load_fleurs_subset
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

    console.rule("[bold]v2 — 3-Provider Benchmark: Hindi · Nepali · English[/bold]")
    console.print(
        f"[dim]{N_SAMPLES} samples per language · "
        "Deepgram/Whisper results cached where available from v0/v1[/dim]"
    )

    samples_by_lang: dict = {}
    for lang in LANGUAGES:
        console.print(f"Loading {N_SAMPLES} {lang} samples from FLEURS...")
        samples_by_lang[lang] = load_fleurs_subset(lang, n_samples=N_SAMPLES)

    all_metrics: dict = {}
    for lang in LANGUAGES:
        console.rule(f"[bold]{lang}[/bold]")
        for provider in providers:
            metrics = run_benchmark(provider, samples_by_lang[lang], RESULTS_DIR, lang)
            all_metrics[(provider.name, lang)] = metrics

    _print_comparison(all_metrics, provider_names)
    console.print(f"\nAll results written to [cyan]{RESULTS_DIR}[/cyan]")


if __name__ == "__main__":
    main()
