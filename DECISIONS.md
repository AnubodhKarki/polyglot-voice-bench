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

---

## 2026-04-22 — v1 actual findings (Deepgram Nova-3 vs Whisper Large v3 via Groq)

Hindi (`hi_in`): Deepgram WER 0.215 / CER 0.090; Whisper WER 0.362 / CER 0.116. Deepgram leads by ~68% relative WER. The CI bands do not overlap (Deepgram [0.148–0.272] vs Whisper [0.244–0.479]) — but the upper bound of Deepgram's CI touches the lower bound of Whisper's, so n=5 is not enough to declare statistical significance. The direction is clear; the magnitude needs more samples.

Nepali (`ne_np`): Deepgram WER 1.211 / CER 0.508; Whisper WER 1.012 / CER 0.352. Both are catastrophic (WER > 1.0 means insertions exceed reference length), but Whisper fails less badly — ~17% lower WER, ~31% lower CER. The WER CIs overlap substantially ([1.000–1.422] vs [0.733–1.213]), so the WER difference is within noise at n=5. The CER gap (0.508 vs 0.352) is more pronounced and the intervals barely touch, suggesting Whisper gets more characters right even when word boundaries are wrong — consistent with it having been trained on Devanagari script. Neither model should be used for Nepali in production.

Interpretation: the v1 headline is a split verdict — Deepgram wins on Hindi, Whisper is less bad on Nepali. Increasing to n=50 in a future version will tighten the CIs and clarify whether the Hindi gap and Nepali ordering hold at scale.

---

## 2026-04-22 — v2 actual findings (3-provider × 3-language, n=25)

English (en_us): Deepgram WER 0.086 / CER 0.037 — clear best. AssemblyAI WER 0.227 / CER 0.054 (23/25 succeeded, 2 errors). Whisper WER 0.234 / CER 0.058. AssemblyAI and Whisper are essentially tied on English; Deepgram is 2.6× better than either.

Hindi (hi_in): Deepgram WER 0.165 / CER 0.083 — clear best. Whisper WER 0.286 / CER 0.082. AssemblyAI WER 1.008 / CER 0.981 — catastrophic silent failure. CI [1.001, 1.019] is extraordinarily tight, indicating a consistent systematic failure across all 25 samples, not noise. The model produced English-looking output for Hindi audio without raising any error.

Nepali (ne_np): All three providers fail (WER > 0.96). Whisper is least bad at WER 0.963 / CER 0.311. Deepgram WER 1.084 / CER 0.416. AssemblyAI WER 1.045 / CER 0.989 — same near-total CER failure pattern as Hindi.

The v2 central finding: every provider has at least one silent failure mode on South Asian languages. Deepgram fails on Nepali. AssemblyAI Universal-3 Pro fails on both Hindi and Nepali — its 6 supported languages appear to be European only. Only Whisper produces non-degenerate (WER < 1.0) output across all three languages, though still poor on Nepali. No provider surfaces a language-not-supported error for any of these cases.

---

## 2026-04-22 — v2 scope: FLEURS English is en_us only; accent variants deferred

The v2 spec listed FLEURS en_us, en_in, en_au. FLEURS (google/fleurs) has 102 language codes; English appears only as en_us (US English). There is no en_in or en_au subset. Indian English and Australian English accent variants require a different dataset — Mozilla Common Voice has self-reported accent metadata for English. This is deferred to v3 where the self-collected dataset and Common Voice accent slices will be introduced together.

---

## 2026-04-22 — AssemblyAI Universal-3 Pro: streaming architecture for pre-recorded files

Universal-3 Pro (`u3-rt-pro`) is a streaming-only model (WebSocket). It does not have a batch/async file endpoint. To benchmark it on pre-recorded FLEURS audio, we stream the audio file in 100ms PCM chunks at 1× real-time speed via `StreamingClient.stream(bytes)`.

The 1× real-time throttle is required because AssemblyAI's server enforces error 4029 ("Client sent audio too fast") when audio arrives faster than real-time. Consequence: each ~12s FLEURS sample takes ~12s to stream, so AssemblyAI calls are significantly slower than Deepgram (file upload, ~1s) and Groq Whisper (file upload, ~2s). This is an inherent property of the streaming-only API, not a benchmark artifact.

The latency reported for AssemblyAI therefore includes streaming time, not just inference time — it is not comparable to the other providers' latency figures. Metrics in the results table are WER/CER only; latency is stored in the per-sample JSON for reference.

`disconnect(terminate=True)` sends a `TerminateSession` WebSocket message and then blocks until the server responds with a `TerminationEvent`. This guarantees all in-flight turns are finalized before we collect transcripts.

---

## 2026-04-22 — AssemblyAI Universal-3 Pro: no language hint

`StreamingParameters` has no `language` field. Language is detected automatically when `language_detection=True`. This is a methodological asymmetry: Groq Whisper gets an explicit language hint, Deepgram uses `language="multi"`, and AssemblyAI auto-detects. All three are running under their recommended/available production configuration. The asymmetry is intentional and documented; correcting it would require either withholding hints from Whisper (making the benchmark unfair to Whisper) or misusing the AssemblyAI API.

---

## 2026-04-22 — v2 sample count: 25 instead of 100

The spec targets ~100 samples per language. v2 uses 25 because: (1) AssemblyAI streams at 1× real-time — 100 samples × ~12s = ~20 minutes of wall time for one language, or ~60 minutes for three; (2) 25 samples gives ≥10× more statistical power than v0/v1's n=5, and bootstrap CIs at n=25 are tight enough to support directional claims. The runner is resumable; bumping to 100 is a one-line change once we have the time budget.

---

## 2026-04-22 — Why Groq as the inference backend for Whisper Large v3

Three options: (1) run Whisper locally via `openai-whisper` or `faster-whisper`, (2) OpenAI's hosted Whisper API, (3) Groq's hosted Whisper endpoint. Local was already ruled out in v0 (llvmlite/cmake dependency on this machine). OpenAI's Whisper API is $0.006/minute; Groq's is $0.111/hour ($0.00185/minute) — 3× cheaper for the same model checkpoint. Groq also reports sub-second latency for short clips, which matters when running 10 samples in sequence. Both expose the same OpenAI-compatible transcription endpoint; the Groq Python SDK is a thin wrapper around it. If Groq's service is unavailable, switching to OpenAI requires one constructor swap and a key change — no logic changes needed.

---

## 2026-04-22 — Why pass an explicit language hint to Whisper but not to Deepgram

Deepgram Nova-3 Multilingual is called with `language="multi"` (auto-detect) because that is its documented multilingual mode — there is no per-language switch that improves accuracy. Whisper Large v3 is called with an explicit ISO-639-1 language code (`hi` for Hindi, `ne` for Nepali) because: (1) without a hint, Whisper's language-detection step consumes ~1 second and can misidentify languages with limited training data, (2) Whisper's Nepali training data is thin, so auto-detect may hallucinate English or Hindi, (3) giving Whisper the hint represents best-practice deployment — a real application that knows the user's language should pass it. This is a methodological asymmetry and is intentional: we are benchmarking each provider under its recommended production configuration, not under artificial parity.

---

## 2026-04-23 — v3 language codes for non-standard categories

v3 introduces four language categories, two of which have no standard IETF/ISO code:

- `ne_np` — Nepali (existing, follows FLEURS convention)
- `hi_en` — Hinglish (Hindi-English code-switching). No standard code exists. We use underscore-separated `hi_en` so that `language_code.split("_")[0]` → `"hi"` for the Groq Whisper provider, which requires an ISO-639-1 code. The manifest's `language` field uses `hi_en` as an internal label; it is not IETF BCP-47 compliant.
- `en_au` — Australian English in own voice. Standard IETF BCP-47 tag. `en_au`.split("_")[0]` → `"en"` for Whisper.
- `ne_en` — Nepali-English code-switching. Same convention as `hi_en`. `ne_en`.split("_")[0]` → `"ne"` for Whisper.

For Deepgram, all categories receive `language="multi"` (unchanged). For AssemblyAI, all receive auto-detection (unchanged). The existing provider code requires no modification for v3.

---

## 2026-04-23 — v3 reference text encoding for code-switched utterances

For Hinglish (`hi_en`): reference text is in Latin script throughout. This reflects how Hinglish is written in digital contexts (SMS, WhatsApp, Twitter). A provider that outputs Devanagari for a Hinglish utterance will score high WER; one that outputs romanized Hindi will score lower — this asymmetry is intentional and interesting.

For Nepali-English code-switch (`ne_en`): reference text uses Devanagari for the Nepali portions and Latin for the English portions, matching natural written code-switching. WER computation is purely string-based (no script-aware tokenization), so a provider that outputs all-Latin or all-Devanagari will be penalized even if the phonetic content is correct. This is a known limitation documented here for the v4 mixed-script WER work.

---

## 2026-04-23 — v3 results directory separation

v3 results are written to `results/v3/` rather than the top-level `results/`. This prevents collisions with FLEURS runs: both use `ne_np` as a language label, but the data sources, speakers, and sample distributions are different. Mixing them would make the cached transcript files unreliable. The runner's `output_dir` parameter makes this a one-line change per script.

---

## 2026-04-23 — Own-voice dataset pre-population strategy

The `data/own_voice/manifest.csv` is committed before the recording session (2026-04-26). The 120 reference texts are fixed in advance. This is intentional: (1) the utterances can be reviewed and corrected before recording, (2) the dataset loader and `run_v3.py` can be tested before any audio exists (they skip missing files gracefully), (3) it creates a discipline for the recording session — each file has an exact name to produce. Audio files are added to git after recording since `data/own_voice/audio/*.wav` is whitelisted in `.gitignore`.

---

## 2026-04-23 — Why include a CC-BY own-voice dataset

Public benchmark datasets for Nepali, Hinglish, and Nepali-English code-switching are scarce. FLEURS has no Hinglish or code-switch category. Mozilla Common Voice has Nepali but with no Hinglish or code-switch data. By releasing our own recordings under CC-BY 4.0, we: (1) give future researchers a reproducible test set for these categories, (2) make the benchmark results independently verifiable, (3) contribute a resource that did not exist before. The speaker is not anonymous — named attribution is intentional and matches the CC-BY license requirement.
