import pytest

from src.metrics import bootstrap_ci, compute_cer, compute_error_breakdown, compute_wer


def test_wer_identical():
    assert compute_wer("the cat sat", "the cat sat") == 0.0


def test_wer_completely_different():
    wer = compute_wer("the cat sat on the mat", "dog runs fast over hill now")
    assert wer >= 1.0


def test_wer_partial():
    # "the cat sat on the mat" -> "a cat sat on mat"
    # reference words: the cat sat on the mat  (6)
    # hypothesis words: a   cat sat on mat      (5)
    # edits: sub(the->a)=1, del(the)=1, ins=0 => WER = 2/6
    wer = compute_wer("the cat sat on the mat", "a cat sat on mat")
    assert abs(wer - 2 / 6) < 1e-6


def test_error_breakdown_partial():
    breakdown = compute_error_breakdown("the cat sat on the mat", "a cat sat on mat")
    assert breakdown["substitutions"] == 1
    assert breakdown["deletions"] == 1
    assert breakdown["insertions"] == 0


def test_cer_identical():
    assert compute_cer("hello", "hello") == 0.0


def test_cer_all_wrong():
    cer = compute_cer("abc", "xyz")
    assert cer >= 1.0


def test_bootstrap_ci_brackets_point_estimate():
    refs = ["the cat sat on the mat"] * 20
    hyps = ["a cat sat on mat"] * 20
    result = bootstrap_ci(refs, hyps, compute_wer, n_resamples=1000, seed=42)
    assert result["lower"] <= result["point"] <= result["upper"]


def test_bootstrap_ci_perfect_score():
    refs = ["hello world"] * 10
    hyps = ["hello world"] * 10
    result = bootstrap_ci(refs, hyps, compute_wer, seed=42)
    assert result["point"] == 0.0
    assert result["lower"] == 0.0
    assert result["upper"] == 0.0
