# DECISIONS.md

Every non-trivial choice made during development is recorded here with a date and rationale. This file exists so the methodology can be defended in interviews and reproduced by others.

---

## 2026-04-21 — Why uv over pip

`uv` (from Astral) is a Rust-based Python package manager that replaces both `pip` and `virtualenv`. Chosen because: (1) dramatically faster installs due to parallel dependency resolution, (2) `uv sync` produces a reproducible environment from `pyproject.toml` + `uv.lock` without needing a separate `requirements.txt`, (3) `uv run` executes scripts inside the managed venv without manual activation — this matters for CI and for collaborators who don't want to think about venvs. The lockfile means two machines running `uv sync` get byte-for-byte identical environments, which is essential for a benchmark claiming reproducibility.

---

## 2026-04-21 — Why this provider interface design

Every provider implements `TranscriptionProvider` (an ABC with a single `transcribe(sample: AudioSample) -> TranscriptionResult` method). This means: (1) the runner never imports a concrete provider — it only knows the interface, (2) adding a new provider is a single new file with no changes to runner logic, (3) the `MockProvider` lets us test the runner and metrics pipeline without any API calls. ABCs in Python enforce the contract at class-definition time — if a subclass forgets to implement `transcribe`, Python raises `TypeError` at import, not silently at runtime.

---

## 2026-04-21 — Why catch-don't-raise on transcription errors

A benchmark run over 100 samples should not die because sample 47 hit a 503. Errors are caught and stored in `TranscriptionResult.error` so the run continues, the error is preserved verbatim (including HTTP status codes and provider error messages), and the metrics phase skips errored samples while reporting `n_errored` explicitly. This also means Nepali requests to Deepgram — which are expected to fail because Deepgram doesn't officially support Nepali — will produce a clean, quotable error record rather than a stack trace. The failure mode is part of the finding.

---

## 2026-04-21 — Why include Nepali in v0 even though Deepgram doesn't support it

The entire thesis of this project is that voice AI has a coverage gap for South and Southeast Asian languages. The most honest way to demonstrate that in v0 is to run a language that a major provider doesn't support and show exactly what happens. Options: (a) garbage transcript with near-100% WER, (b) an explicit language-not-supported error, (c) a surprisingly coherent output if Nepali training data leaked into the multilingual model. Any of these is a real finding. Omitting Nepali from v0 because it might fail would be methodologically dishonest and would undermine the entire framing.

---

## 2026-04-21 — FLEURS dataset field names and language codes

_To be filled in at Checkpoint 5 after fetching the dataset card._

---

## 2026-04-21 — Deepgram SDK version and Nova-3 parameters

_To be filled in at Checkpoint 6 after fetching Deepgram docs._
