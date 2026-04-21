from collections.abc import Callable

import numpy as np
import jiwer


def compute_wer(reference: str, hypothesis: str, normalize: Callable[[str], str] | None = None) -> float:
    if normalize:
        reference = normalize(reference)
        hypothesis = normalize(hypothesis)
    return jiwer.wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str, normalize: Callable[[str], str] | None = None) -> float:
    if normalize:
        reference = normalize(reference)
        hypothesis = normalize(hypothesis)
    return jiwer.cer(reference, hypothesis)


def compute_error_breakdown(reference: str, hypothesis: str) -> dict[str, int]:
    out = jiwer.process_words(reference, hypothesis)
    return {
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
    }


def bootstrap_ci(
    references: list[str],
    hypotheses: list[str],
    metric_fn: Callable[[str, str], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Return point estimate and bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    n = len(references)
    point = float(np.mean([metric_fn(r, h) for r, h in zip(references, hypotheses)]))

    boot = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_mean = float(np.mean([metric_fn(references[i], hypotheses[i]) for i in idx]))
        boot.append(sample_mean)

    alpha = 1 - confidence
    lower = float(np.percentile(boot, 100 * alpha / 2))
    upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))

    return {"point": point, "lower": lower, "upper": upper, "confidence": confidence}
