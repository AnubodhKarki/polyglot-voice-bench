"""v0 entry point: Deepgram Nova-3 Multilingual on 5 Hindi + 5 Nepali FLEURS samples."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

# Allow running as `uv run python scripts/run_v0.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

console = Console()

RESULTS_DIR = Path(__file__).parent.parent / "results"
N_SAMPLES = 5


def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not api_key:
        console.print("[bold red]ERROR:[/bold red] DEEPGRAM_API_KEY not set.")
        console.print("Copy .env.example to .env and add your key.")
        sys.exit(1)

    from src.datasets.fleurs import load_fleurs_subset
    from src.providers.deepgram_nova3_multi import DeepgramNova3MultiProvider
    from src.runner import run_benchmark

    provider = DeepgramNova3MultiProvider(api_key=api_key)

    console.rule("[bold]v0 — Deepgram Nova-3 Multilingual[/bold]")
    console.print(f"Loading {N_SAMPLES} Hindi samples from FLEURS...")
    hindi_samples = load_fleurs_subset("hi_in", n_samples=N_SAMPLES)

    console.print(f"Loading {N_SAMPLES} Nepali samples from FLEURS...")
    nepali_samples = load_fleurs_subset("ne_np", n_samples=N_SAMPLES)

    console.rule("[bold]Running Hindi[/bold]")
    hindi_metrics = run_benchmark(provider, hindi_samples, RESULTS_DIR, "hi_in")

    console.rule("[bold]Running Nepali[/bold]")
    nepali_metrics = run_benchmark(provider, nepali_samples, RESULTS_DIR, "ne_np")

    console.rule("[bold]v0 Complete[/bold]")
    console.print(f"Results written to [cyan]{RESULTS_DIR}[/cyan]")
    console.print(
        f"Hindi WER: [bold]{hindi_metrics['wer']:.3f}[/bold]"
        if hindi_metrics["wer"] is not None
        else "Hindi WER: [yellow]N/A (no successful transcriptions)[/yellow]"
    )
    console.print(
        f"Nepali WER: [bold]{nepali_metrics['wer']:.3f}[/bold]"
        if nepali_metrics["wer"] is not None
        else f"Nepali: [red]{nepali_metrics['n_errored']} errors[/red] / "
             f"{nepali_metrics['n_succeeded']} succeeded  ← expected finding"
    )


if __name__ == "__main__":
    main()
