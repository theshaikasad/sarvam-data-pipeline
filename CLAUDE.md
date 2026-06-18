# sarvam-tts-dataset

Take-home for a Sarvam AI internship: build a **60-minute single-speaker TTS training
dataset** (~30 min Indian English + ~30 min Telugu, ~120 clips of ~25s each) with
accurate transcripts and Parler-TTS-style emotion/style tags, published as a **public
HuggingFace dataset**.

## Grading & philosophy (read this first)
Graded on **DATA QUALITY AND CURATION JUDGMENT, not code.** The pipeline is assumed
trivial. What matters: clean audio (no music bed), accurate transcripts/labels, honest
quality metrics, and good decisions.

- 120 clips × ~25s, single-speaker per clip, clean recordings.
- **Over-collect ~50%, then filter down.** Quality over quantity.
- **"Listen to the data" is the core of the grade** — manual review is not optional.

## Confirmed Sarvam APIs (use these EXACT strings — verified against current docs)
- **ASR:** model `saaras:v3`, `mode="transcribe"`, `POST https://api.sarvam.ai/speech-to-text`,
  header `api-subscription-key: $SARVAM_KEY`, multipart file upload.
  **The synchronous endpoint only accepts files UNDER 30 seconds** — target ~25s clips to
  stay safely under it.
- **ASR language:** `language_code` accepts `"unknown"` (auto-detect), but we KNOW each
  channel's language, so pass the explicit code (`te-IN` / `en-IN`) for higher accuracy.
  `"unknown"` is fallback only.
- **LLM:** model `sarvam-30b` (`sarvam-m` is legacy) via OpenAI-compatible client,
  `base_url="https://api.sarvam.ai/v1"`, `api_key=$SARVAM_KEY`.
- **Docs index for AI tools:** https://docs.sarvam.ai/llms-full.txt (full API reference, one file).
- **Report context:** Bulbul V3 (Sarvam's TTS) has **NO emotion parameter** — it's LLM-based
  and infers prosody from text/context. This is WHY we produce rich Parler-style
  natural-language descriptions rather than bare one-word labels: a prosody-inferring model
  trains on rich descriptive signal.

## Repo structure
```
config.yaml, requirements.txt, .env.example
data/{raw, clips, manifest.jsonl}
pipeline/{state.py, s1_download.py, s2_segment.py, s3_music_filter.py,
          s4_features.py, s5_asr.py, s6_tag.py, s7_describe.py, s8_export.py}
review/{gold_sample.py, review_ui.py}
eval/{compute_wer.py, distributions.py}
report/
```

## Single source of truth: `data/manifest.jsonl`
One JSON row per clip. Stages move through a state machine via a `stage` field:

```
downloaded → segmented → music_checked → transcribed → tagged → described → final
                                                                  (or "rejected" + rejected_reason)
```

Each pipeline stage loads the manifest, filters rows by `stage`, processes, writes back.
**Crash-resumable:** rerunning skips rows already past that stage.

### CRITICAL no-overwrite rule
Machine columns and human columns are **SEPARATE keys and never share a key.** Human review
writes `human_transcript` / `human_emotion` / `human_verified` / `reviewer` / `reviewed_at`.
It **NEVER** edits `asr_*` or `llm_*` fields. At **EXPORT only**, compute derived fields:
- `final_transcript = human_transcript or asr_transcript`
- `final_emotion = human_emotion or llm_emotion`

`human_verified=true` rows are the **gold subset** for WER and for the `gold` split.

### Manifest row fields
- **identity:** `clip_id` (`channel_video_segment`, immutable), `source_url`, `source_channel`,
  `source_type`, `language`, `start_time`, `end_time`, `duration`
- **stage:** `stage`, `rejected_reason`
- **quality:** `has_music`, `music_confidence`, `snr_estimate`
- **acoustic:** `pitch_mean`, `pitch_std`, `energy_rms`, `speaking_rate`
- **asr (machine, never edited):** `asr_transcript`, `asr_model`, `asr_language_detected`, `asr_confidence`
- **llm tags (machine, never edited):** `llm_emotion`, `llm_style`, `llm_confidence`,
  `llm_reasoning`, `llm_model`, `annotated_at`
- **human (only set during review):** `human_transcript`, `human_emotion`, `human_verified`,
  `reviewer`, `reviewed_at`
- **description:** `description`

## Taxonomy (Parler-aligned)
- **emotion** (pick one): `neutral, happy, sad, angry, excited, calm, fearful, surprised, serious, playful`
- **whisper**: boolean flag, **separate from emotion** (it's a phonation style and can co-occur).
- **style** (pick one): `narrative, conversational, oratorical, instructional, devotional, dramatic`
- **Parler axes derived from acoustic features:**
  - `speaking_rate_bin`: `very_slow / slow / measured / fast / very_fast`
  - `pitch_bin`: `low / moderate / high` (per-gender)
  - `pitch_variation`: `monotone / moderate / animated`
  - `recording_quality`: `clean / slight_noise / noisy`
- **description**: one natural-language sentence composed from the above, e.g.
  *"A male speaker narrates a Telugu story in a slow, sorrowful tone with moderate pitch
  variation, recorded clearly with almost no background noise."*

## Final HF dataset columns (`s8` pushes)
`audio` (Audio 16kHz), `text` (=`final_transcript`), `language`, `emotion` (=`final_emotion`),
`style`, `whisper`, `speaking_rate_bin`, `pitch_bin`, `pitch_variation`, `recording_quality`,
`description`, `speaker_id` (anonymized per source, e.g. `spk_harshaneeyam`), `duration`,
`source_url`, `source_channel`, `source_type`, `human_verified`,
`split` (`"gold"` if `human_verified` else `"train"`).

## Edge-case decisions (document these as deliberate)
- **Segmentation:** NEVER chop at fixed times (causes mid-word cuts). Use VAD/silence
  detection and cut only at silences ≥300ms, packing speech into ~22–28s windows.
  Boundaries land between words by construction.
- **Background music:** DETECT AND DROP, never repair. Source separation (Demucs) degrades
  audio and TTS needs clean recordings, so a repaired clip is worse than none. Use
  `inaSpeechSegmenter` to label speech/music/noise; if music overlaps **>10%** of a clip,
  set `has_music=true`, `stage="rejected"`, `rejected_reason="music_bed"`. **Keep rejected
  rows** — their counts are the report's iteration log.
- **Audio format:** 16kHz mono WAV, loudness-normalized.

## Source roster (6 channels)
**Verify each by listening to ~3 min before bulk download; swap if music bed / multi-speaker / bad mic.**

Telugu:
- Harshaneeyam — `te-IN`, `podcast_storytelling`
- BV Pattabhiram — `te-IN`, `public_lecture` / motivational
- Garikapati Narasimha Rao — `te-IN`, `public_lecture` / discourse

English (Indian):
- TED/TEDx India (e.g. Shashi Tharoor, Devdutt Pattanaik) — `en-IN`, `public_lecture`
- one solo Indian-English audiobook/storytelling channel — `en-IN`, `audiobook`
- one solo Indian-English podcast monologue (e.g. Ranveer / Raj Shamani solo segments) — `en-IN`, `podcast_independent`

## Ethics / licensing
Record `source_type` per clip (`government_broadcast` / `public_lecture` /
`podcast_independent` / `audiobook` / `podcast_storytelling`). Note licensing posture
**honestly** in the dataset card; avoid copyrighted film/music audio. AI4Bharat's
`ai4bharat/Mann-ki-Baat` (CC BY 4.0) can supply a small verified-transcript Telugu+English
slice usable as a **quality anchor**.

## Eval
- **WER and CER** via `jiwer`, computed on `human_verified` gold rows, reported **per-language**
  (CER matters for Telugu script).
- Emotion distribution and duration histograms.
- Report the **worst ASR failures** with examples.
