# sarvam-tts-dataset

A small, high-curation **single-speaker text-to-speech dataset** for **Telugu + Indian English**, built for the Sarvam AI take-home. Every clip is one speaker, ~25 s, cut at silences, with an accurate transcript and a Parler-TTS-style natural-language description of *how* it's said (emotion, style, pace, pitch, whisper).

**Result:** 190 clips / 79 min, balanced 95 Telugu + 95 English across 14 sources (an over-collected review buffer; `pipeline/balance.py` re-trims to the final size).

- **Full write-up:** [`report/REPORT.pdf`](report/REPORT.pdf) — what I built, iterations, what worked/didn't, quality observations, and what I'd improve.
- **Dataset card** (pushed to HuggingFace): [`report/README.md`](report/README.md).

## How it works

A resumable, manifest-driven pipeline. `data/manifest.jsonl` is the single source of truth — one row per clip — and each stage reads it, does one job, and writes back, so any stage can be re-run safely.

```
s1 download  -> s2/s2b segment -> s3 filter -> s4/s4b features+emotion
   -> s5 ASR (saaras:v3) -> s6 tag (sarvam-30b) -> s7 describe -> [human review] -> s8 export
```

Sarvam APIs do the speech-to-text (`saaras:v3`), diarization (Batch STT), and all LLM tagging/description (`sarvam-30b`). A hard **no-overwrite rule** keeps machine columns (`asr_*`, `llm_*`) and human columns (`human_*`) separate; they're merged only at export, and human-verified clips form the `gold` split.

## Quickstart

```bash
pip install -r requirements.txt

# secrets: copy .env.example -> .env and set SARVAM_KEY and HF_TOKEN
# sources: edit config.yaml (the YouTube URLs, languages, per-source flags)

python pipeline/s1_download.py            # download + normalize to 16kHz mono
python pipeline/s2_segment.py             # silence-aware ~25s clips
python pipeline/s2b_diarized_segment.py   # only for channels marked `diarized: true`
python pipeline/s3_music_filter.py        # drop music / crowd / multi-speaker
python pipeline/s4_features.py
python pipeline/s4b_audio_emotion.py      # audio-grounded 2nd emotion opinion
python pipeline/s5_asr.py                 # Sarvam saaras:v3
python pipeline/s6_tag.py                 # Sarvam sarvam-30b: emotion/style/whisper
python pipeline/s7_describe.py            # Sarvam sarvam-30b: description

python review/gold_sample.py              # mark a small, source-balanced review set
python review/review_ui.py                # human verification (Gradio)
python eval/compute_wer.py                # WER/CER on verified gold clips
python pipeline/balance.py 60             # trim to a balanced ~60 min
python pipeline/s8_export.py              # build + push the public HuggingFace dataset
```

> `config.yaml` per-source flags: `solo: true` (verified single speaker), `diarized: true` (multi-speaker → keep dominant), `whisper: true` (ASMR/whispered), `gender`.

## Repo structure

```
config.yaml                # the only file you edit by hand: sources + knobs
data/manifest.jsonl        # single source of truth (one row per clip)
pipeline/                  # s1..s8 stages + state.py, balance.py, emotion_map.py
review/                    # gold_sample.py, review_ui.py (Gradio)
eval/                      # compute_wer.py, distributions.py, snapshot.py
report/                    # REPORT.pdf, the dataset card, figures + make_figures.py
CLAUDE.md                  # the detailed project contract / design notes
```

## Ethics

Sources are publicly available recordings; each clip records its `source_type`. Copyrighted film/music audio is avoided and music-bed clips are dropped. Intended for **research/educational use** — the licence tag covers the added annotations; verify rights for the underlying audio before commercial use.
