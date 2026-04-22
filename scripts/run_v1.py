"""v1 entry point: Deepgram Nova-3 vs Whisper Large v3 (via Groq) on 5 Hindi + 5 Nepali FLEURS samples."""

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
N_SAMPLES = 5


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


def _print_comparison(all_metrics: dict) -> None:
    console.rule("[bold]Side-by-side Comparison[/bold]")

    table = Table(title="WER / CER by Provider and Language (mean ± 95% CI)")
    table.add_column("Language", style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Deepgram Nova-3 Multi")
    table.add_column("Whisper Large v3 (Groq)")

    for lang in ["hi_in", "ne_np"]:
        dg = all_metrics.get(("deepgram", lang), {})
        wh = all_metrics.get(("groq_whisper", lang), {})
        table.add_row(lang, "WER", _fmt_wer(dg), _fmt_wer(wh))
        table.add_row("", "CER", _fmt_cer(dg), _fmt_cer(wh))

    console.print(table)


def main() -> None:
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if not deepgram_key:
        console.print("[bold red]ERROR:[/bold red] DEEPGRAM_API_KEY not set in .env")
        sys.exit(1)
    if not groq_key:
        console.print("[bold red]ERROR:[/bold red] GROQ_API_KEY not set in .env")
        sys.exit(1)

    from src.datasets.fleurs import load_fleurs_subset
    from src.providers.deepgram_nova3_multi import DeepgramNova3MultiProvider
    from src.providers.groq_whisper_large_v3 import GroqWhisperLargeV3Provider
    from src.runner import run_benchmark

    deepgram = DeepgramNova3MultiProvider(api_key=deepgram_key)
    whisper = GroqWhisperLargeV3Provider(api_key=groq_key)

    console.rule("[bold]v1 — Deepgram Nova-3 vs Whisper Large v3 (Groq)[/bold]")
    console.print(f"Loading {N_SAMPLES} Hindi + {N_SAMPLES} Nepali samples from FLEURS...")
    hindi_samples = load_fleurs_subset("hi_in", n_samples=N_SAMPLES)
    nepali_samples = load_fleurs_subset("ne_np", n_samples=N_SAMPLES)

    all_metrics: dict = {}

    for lang_label, samples in [("hi_in", hindi_samples), ("ne_np", nepali_samples)]:
        console.rule(f"[bold]{lang_label}[/bold]")
        for provider in [deepgram, whisper]:
            metrics = run_benchmark(provider, samples, RESULTS_DIR, lang_label)
            all_metrics[(provider.name, lang_label)] = metrics

    _print_comparison(all_metrics)
    console.print(f"\nAll results written to [cyan]{RESULTS_DIR}[/cyan]")
    console.print(
        "[dim]Note: Deepgram results are cached from v0 and will not re-call the API.[/dim]"
    )


if __name__ == "__main__":
    main()
