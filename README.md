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

_To be filled in after the first real run._

## What this means

_To be filled in after v0 findings are confirmed._

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
