"""v2b: Add AssemblyAI Universal-2 (batch) to the v2 comparison.

Runs Universal-2 on the same 25 FLEURS samples per language already used in v2.
Results are stored alongside v2 results (same results/ dir) under provider name
'assemblyai_u2'. Previously computed transcripts for other providers are not
re-run — the runner loads them from the cache.

Prints a 4-provider comparison table: Deepgram / Whisper / AssemblyAI U3-Pro /
AssemblyAI U2. This directly shows the model-tier trade-off for Hindi.
"""

import json
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


def _fmt(metrics: dict, key: str) -> str:
    if metrics.get(key) is None:
        n_err = metrics.get("n_errored", "?")
        return f"N/A ({n_err} err)" if key == "wer" else "N/A"
    s = f"{metrics[key]:.3f}"
    ci_key = f"{key}_ci"
    ci = metrics.get(ci_key)
    if ci:
        s += f"  [{ci['lower']:.3f}–{ci['upper']:.3f}]"
    return s


def _load_existing_metrics(provider: str, language: str) -> dict:
    path = RESULTS_DIR / f"metrics_{provider}_{language}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _print_comparison(all_metrics: dict) -> None:
    console.rule("[bold]v2b — 4-Provider Comparison[/bold]")
    table = Table(
        title="WER / CER — mean with 95% CI (FLEURS validation, n=25 per language)",
        show_lines=True,
    )
    table.add_column("Language", style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Deepgram\nNova-3")
    table.add_column("Whisper v3\n(Groq)")
    table.add_column("AssemblyAI\nU3-Pro (stream)")
    table.add_column("AssemblyAI\nU2 (batch)")

    providers = ["deepgram", "groq_whisper", "assemblyai", "assemblyai_u2"]

    for lang in LANGUAGES:
        for i, metric in enumerate(["wer", "cer"]):
            row = [lang if i == 0 else "", metric.upper()]
            for prov in providers:
                m = all_metrics.get((prov, lang), {})
                row.append(_fmt(m, metric))
            table.add_row(*row)

    console.print(table)
    console.print()
    console.print(
        "[dim]U3-Pro is AssemblyAI's streaming flagship — the natural choice for "
        "a voice application. Its 6-language limit is not surfaced by the API: "
        "Hindi requests succeed silently with garbage output. U2 (batch) supports "
        "Hindi but requires giving up real-time streaming and their best model.[/dim]"
    )


def main() -> None:
    assemblyai_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not assemblyai_key:
        console.print("[bold red]ERROR:[/bold red] ASSEMBLYAI_API_KEY not set in .env")
        sys.exit(1)

    from src.datasets.fleurs import load_fleurs_subset
    from src.providers.assemblyai_universal2 import AssemblyAIUniversal2Provider
    from src.runner import run_benchmark

    provider = AssemblyAIUniversal2Provider(api_key=assemblyai_key)

    console.rule("[bold]v2b — AssemblyAI Universal-2 supplementary run[/bold]")
    console.print(
        "[dim]Runs Universal-2 (batch) on the same 25 FLEURS samples per language. "
        "Other providers load from v2 cache.[/dim]\n"
    )

    all_metrics: dict = {}

    # Load existing v2 results for comparison
    for prov in ["deepgram", "groq_whisper", "assemblyai"]:
        for lang in LANGUAGES:
            m = _load_existing_metrics(prov, lang)
            if m:
                all_metrics[(prov, lang)] = m

    # Run Universal-2 on each language
    for lang in LANGUAGES:
        console.print(f"Loading {N_SAMPLES} {lang} samples from FLEURS...")
        samples = load_fleurs_subset(lang, n_samples=N_SAMPLES)
        console.rule(f"[bold]Universal-2 / {lang}[/bold]")
        metrics = run_benchmark(provider, samples, RESULTS_DIR, lang)
        all_metrics[(provider.name, lang)] = metrics

    _print_comparison(all_metrics)
    console.print(f"\nNew metrics written to [cyan]{RESULTS_DIR}[/cyan]")


if __name__ == "__main__":
    main()
