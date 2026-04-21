# polyglot-voice-bench

A reproducible benchmark of current voice AI providers on the languages and speech patterns the industry has overlooked: Hindi, Nepali, Hinglish, multi-accent English, and non-Western names.

## Why this exists

I'm a Nepali speaker in Sydney. I code-switch between Nepali, Hindi, and English every day at home and at work. Most voice AI treats this as an edge case. It isn't — it's the daily reality of a billion people. This project measures how bad the gap is and names the providers that are closing it.

v0 focuses on the simplest possible spine: Deepgram Nova-3 Multilingual, 5 Hindi samples + 5 Nepali samples from FLEURS. Nepali is not officially supported by Deepgram. That failure is not a bug in the benchmark — it *is* the benchmark.

## Framing

Findings are framed as **"the technology has a gap"**, not "these speakers are hard." The speakers are normal. The systems are incomplete.

## Running v0

```bash
# 1. Clone and install
git clone https://github.com/anubkarki/polyglot-voice-bench.git
cd polyglot-voice-bench
cp .env.example .env
# Edit .env and add your DEEPGRAM_API_KEY

# 2. Install deps via uv
uv sync

# 3. Run
uv run python scripts/run_v0.py
```

Results land in `results/`.

## v0 Results

Deepgram Nova-3 Multilingual, FLEURS validation split, 5 samples per language.

| Language | Samples | WER | 95% CI | CER |
|----------|---------|-----|--------|-----|
| Hindi (`hi_in`) | 5 | 0.215 | [0.148, 0.272] | 0.090 |
| Nepali (`ne_np`) | 5 | **1.211** | [1.000, 1.422] | 0.508 |

**Nepali sample (reference → hypothesis):**
```
REF: अन्तरिक्षमा उपग्रहले कल प्राप्त गर्छ र त्यसपछि यसलाई तुरुन्तै तल प्रतिबिम्ब गर्छ
HYP: अंतरिक्षियाँ में उपग्रहले कल प्राप्त घर से रोह तेज पचीस यसलाई तुरंतेज तल प्रतिबिंब घर से
```

## What this means

Deepgram Nova-3 does not return an error on Nepali audio. It returns **fluent-looking garbage in Hindi script** — silently mapping Nepali phonemes to the nearest Hindi words it recognises. The WER exceeds 1.0 (theoretically capped at 1.0 for substitutions and deletions alone) because the model also *inserts* extra words not present in the reference.

This is worse than an error. An error tells you the system failed. A confident wrong answer tells you nothing is wrong until you check the transcript against ground truth. A production system using Deepgram for Nepali would silently produce incorrect output with no signal that anything had gone wrong.

Hindi performs reasonably (WER 0.215), with the model code-switching into English for some words ("forty", "major", "wheelchair") — a separate finding for v2.

## Next: v1

- Add Whisper Large v3 via Groq (free tier)
- Run the same 10 samples through both providers
- First side-by-side comparison: does Whisper handle Nepali better?

## Next: v1

- Add Whisper Large v3 via Groq (free tier)
- Run the same 10 samples through both providers
- First side-by-side comparison table

## Versions

| v | Goal |
|---|------|
| v0 | Spine: Deepgram + FLEURS Hindi/Nepali (10 samples) |
| v1 | Cross-provider: add Whisper via Groq |
| v2 | Scale + English accents: AssemblyAI + en_us/en_in/en_au |
| v3 | Self-collected dataset (CC-BY release) |
| v3.5 | "Anubodh" name benchmark |
| v4 | Devanagari normalisation + CIs + Gradio app |
| v5 | Blog post + HN post + outreach |

## License

Code: MIT. Dataset (v3+): CC-BY 4.0.
