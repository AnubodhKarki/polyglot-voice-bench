import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.metrics import bootstrap_ci, compute_cer, compute_wer
from src.providers.base import TranscriptionProvider
from src.types import AudioSample, TranscriptionResult

console = Console()


def _result_path(output_dir: Path, provider: str, language: str, sample_id: str) -> Path:
    return output_dir / "transcripts" / provider / language / f"{sample_id}.json"


def _save_result(path: Path, result: TranscriptionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    data.pop("raw_response", None)  # not JSON-serialisable
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load_result(path: Path) -> TranscriptionResult:
    data = json.loads(path.read_text())
    return TranscriptionResult(**data)


def run_benchmark(
    provider: TranscriptionProvider,
    samples: list[AudioSample],
    output_dir: Path,
    language_label: str,
) -> dict:
    """Run provider over samples, persisting each result. Resumable: skips
    samples whose output file already exists. Returns a metrics summary dict."""
    output_dir = Path(output_dir)

    # pairs: (sample, result) in the same order as samples
    pairs: list[tuple[AudioSample, TranscriptionResult]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]{provider.name} / {language_label}",
            total=len(samples),
        )

        for sample in samples:
            sample_id = str(sample.metadata.get("sample_id", Path(sample.audio_path).stem))
            rpath = _result_path(output_dir, provider.name, language_label, sample_id)

            if rpath.exists():
                result = _load_result(rpath)
            else:
                result = provider.transcribe(sample)
                _save_result(rpath, result)

            pairs.append((sample, result))
            progress.advance(task)

    succeeded = [(s, r) for s, r in pairs if r.succeeded]
    errored = [(s, r) for s, r in pairs if not r.succeeded]

    metrics: dict = {
        "provider": provider.name,
        "language": language_label,
        "n_total": len(pairs),
        "n_succeeded": len(succeeded),
        "n_errored": len(errored),
        "errors": [r.error for _, r in errored],
        "wer": None,
        "cer": None,
        "wer_ci": None,
        "cer_ci": None,
    }

    if succeeded:
        refs = [s.reference_text for s, _ in succeeded]
        hyps = [r.text for _, r in succeeded]

        per_wer = [compute_wer(ref, hyp) for ref, hyp in zip(refs, hyps)]
        per_cer = [compute_cer(ref, hyp) for ref, hyp in zip(refs, hyps)]

        metrics["wer"] = float(np.mean(per_wer))
        metrics["cer"] = float(np.mean(per_cer))

        if len(refs) >= 3:
            metrics["wer_ci"] = bootstrap_ci(refs, hyps, compute_wer)
            metrics["cer_ci"] = bootstrap_ci(refs, hyps, compute_cer)

    metrics_path = output_dir / f"metrics_{provider.name}_{language_label}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    _print_summary(metrics, errored)
    return metrics


def _print_summary(metrics: dict, errored: list) -> None:
    table = Table(title=f"Results: {metrics['provider']} / {metrics['language']}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Samples", str(metrics["n_total"]))
    table.add_row("Succeeded", f"[green]{metrics['n_succeeded']}[/green]")
    table.add_row("Errored", f"[red]{metrics['n_errored']}[/red]")

    if metrics["wer"] is not None:
        wer_str = f"{metrics['wer']:.3f}"
        if metrics["wer_ci"]:
            ci = metrics["wer_ci"]
            wer_str += f"  (95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}])"
        table.add_row("WER (mean)", wer_str)

    if metrics["cer"] is not None:
        cer_str = f"{metrics['cer']:.3f}"
        if metrics["cer_ci"]:
            ci = metrics["cer_ci"]
            cer_str += f"  (95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}])"
        table.add_row("CER (mean)", cer_str)

    for _, r in errored:
        table.add_row("[red]Error[/red]", (r.error or "")[:120])

    console.print(table)
