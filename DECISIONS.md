# DECISIONS.md

Every non-trivial choice made during development is recorded here with a date and rationale. This file exists so the methodology can be defended in interviews and reproduced by others.

---

## 2026-04-21 — v0 actual findings (Deepgram Nova-3 Multilingual)

Hindi (`hi_in`): WER 0.215, CER 0.090, all 5 samples succeeded. The model code-switches into English for certain words (numbers, proper nouns, technical terms) even when the reference is pure Hindi — an interesting secondary finding for v2.

Nepali (`ne_np`): WER 1.211, CER 0.508, all 5 samples "succeeded" (no API errors). The model does not return a language-not-supported error. Instead it silently maps Nepali phonemes to the nearest Hindi equivalents, producing fluent-sounding but semantically incorrect Hindi text. WER exceeds 1.0 because the model inserts extra words beyond substitutions and deletions. This silent failure mode is the central v0 finding.

Interpretation: a production system relying on Deepgram for Nepali transcription would produce wrong output with no error signal. The technology has a coverage gap, and it does not communicate that gap to callers.

---

## 2026-04-21 — Why use mean WER across samples rather than corpus WER

Per-sample WER is computed for each sample individually, then averaged. An alternative is corpus WER (total edit distance / total reference words across all samples). We use mean-per-sample because: (1) it weights each speaker equally regardless of utterance length, (2) it allows per-sample variance to feed into bootstrap CIs, and (3) it is the more common approach in recent STT benchmarks (see ESB leaderboard). For v0 with 5 samples this distinction is minor; it matters more at scale.

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

## 2026-04-21 — Why openai-whisper was dropped from v0 dependencies

`openai-whisper` was originally included to use `whisper.normalizers.EnglishTextNormalizer`. It could not be installed on macOS x86_64 (Intel): newer versions pull in `numba` -> `llvmlite` which requires `cmake` at build time; older pinned versions have a broken `pkg_resources` build dependency with modern uv. Rather than introduce a system-level cmake install as a prerequisite, we replaced the dependency with a minimal `normalize_english()` function (lowercase + unicode NFC + punctuation strip) that is sufficient for v0 WER comparisons. The plan is to revisit proper normalization in v4 using `faster-whisper`, which ships pre-built wheels.

---

## 2026-04-21 — Why datasets is pinned to <3.0 and audio decoded without librosa

`datasets>=3.0` dropped support for dataset loading scripts; `google/fleurs` still uses one (`fleurs.py`). Pinned to `datasets>=2.14,<3.0` to keep loading-script support. When `google/fleurs` migrates to Parquet this pin can be removed.

`datasets 2.x` auto-decodes the `Audio` feature using `librosa`. Since librosa was dropped (see below), we use `dataset.cast_column("audio", Audio(decode=False))` to get raw bytes/path, then decode with `soundfile` + `scipy.signal.resample_poly`. This avoids librosa entirely while giving us the same 16kHz mono float32 arrays.

---

## 2026-04-21 — Why librosa was dropped from v0 and jiwer API version

`librosa>=0.9` requires `numba`, which requires `llvmlite`, which requires `cmake` at build time — not available on this machine without a Homebrew install. Since v0 only needs basic audio loading and resampling (16kHz mono WAV), `soundfile` + `scipy.signal` are sufficient. `librosa` can be re-added in v4 if we need mel-spectrograms or more advanced audio analysis.

`jiwer` was installed as v4.0.0 (the current release at time of writing). It removed `compute_measures` in favor of `process_words()` which returns a `WordOutput` dataclass, and `wer()` / `cer()` as standalone functions. The metrics module was updated to use the v4 API. All 8 tests pass.

---

## 2026-04-21 — FLEURS dataset field names and language codes

Verified against https://huggingface.co/datasets/google/fleurs on 2026-04-21.
- Hindi code: `hi_in`, Nepali code: `ne_np`
- Native sample rate: 16,000 Hz (no resampling needed)
- Transcript field: `transcription` (Unicode-normalised by Google); raw form is `raw_transcription`
- Audio field: `audio` dict with keys `array` (numpy float32), `sampling_rate`, `path`
- Splits: `train`, `validation`, `test`; no HuggingFace auth required (CC-BY-4.0)

We use `transcription` (not `raw_transcription`) as the reference string because it is the form Google normalised for evaluation. Using `raw_transcription` would introduce inconsistencies between samples. In v4 we will revisit Devanagari normalisation on top of this.

---

## 2026-04-21 — Deepgram SDK version and Nova-3 parameters

Verified against developers.deepgram.com and deepgram.com/pricing on 2026-04-21.
- SDK: `deepgram-sdk==6.1.1` (already installed; no version change needed)
- Model string: `"nova-3"`
- Language parameter for multilingual mode: `language="multi"`
- Nepali (ne / ne-NP): **not listed** as a supported language for any Nova model
- Price: $0.0092 per minute (Nova-3, Pay As You Go tier)

Nepali being unsupported is the expected and intended finding for v0. The provider will pass `language="multi"` regardless of the input language; Deepgram will attempt transcription and either return garbage, an empty transcript, or an error — all of which are valid findings to record verbatim.
