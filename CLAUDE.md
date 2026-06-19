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
- **LLM:** model `sarvam-30b` (`sarvam-m`/`sarvam-105b` are the other options; `sarvam-m`
  is deprecated) via OpenAI-compatible client, `base_url="https://api.sarvam.ai/v1"`,
  `api_key=$SARVAM_KEY`. **CRITICAL:** `sarvam-30b` is a REASONING model — by default it
  spends the whole token budget (4096 on the starter tier) on hidden chain-of-thought and
  returns `content=None`. Pass `extra_body={"reasoning_effort": None}` (an explicit JSON
  null, NOT the string `"none"` which 400s) to disable thinking for direct, cheap answers.
- **Docs index for AI tools:** https://docs.sarvam.ai/llms-full.txt (full API reference, one file).
- **Report context:** Bulbul V3 (Sarvam's TTS) has **NO emotion parameter** — it's LLM-based
  and infers prosody from text/context. This is WHY we produce rich Parler-style
  natural-language descriptions rather than bare one-word labels: a prosody-inferring model
  trains on rich descriptive signal.

## Repo structure
```
config.yaml, requirements.txt, .env.example
data/{raw, clips, manifest.jsonl}
pipeline/{state.py, emotion_map.py, s1_download.py, s2_segment.py, s2b_diarized_segment.py,
 s3_music_filter.py, s4_features.py, s4b_audio_emotion.py, s5_asr.py, s6_tag.py,
 s7_describe.py, s8_export.py}
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
  `source_type`, `language`, `gender` (per source; backs per-gender `pitch_bin`),
  `start_time`, `end_time`, `duration`
- **stage:** `stage`, `rejected_reason`
- **quality (s3):** `has_music`, `music_confidence`, `has_noise`, `noise_confidence`
 (applause/laughter/crowd), `has_multi_speaker`, `gender_detected`, `snr_estimate`
- **acoustic:** `pitch_mean`, `pitch_std`, `energy_rms`, `speaking_rate`
- **audio emotion (machine, never edited; s4b):** `audio_emotion` (on OUR taxonomy),
 `audio_emotion_raw` (model's native label), `audio_emotion_score`, `audio_emotion_model`,
 `emotion_agree` (`agree`/`disagree` vs `llm_emotion`, in valence/arousal quadrant space);
 dimensional backend also sets `audio_arousal`/`audio_valence`/`audio_dominance` ([0,1])
- **diarized v2 segmentation (s2b):** `speaker_id` (diarization label), `segmentation`
 (`diarized_v2` vs default VAD)
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
`clip_id` (stable per-clip key, maps back to the manifest), `audio` (Audio 16kHz),
`text` (=`final_transcript`), `language`, `emotion` (=`final_emotion`),
`style`, `whisper`, `speaking_rate_bin`, `pitch_bin` (per-gender), `pitch_variation`,
`recording_quality`, `description`, `speaker_id` (anonymized per source, e.g.
`spk_harshaneeyam`), `gender` (per source), `duration`, `source_url`, `source_channel`,
`source_type`, `human_verified`, `split` (`"gold"` if `human_verified` else `"train"`).

## Emotion: two independent opinions, human is ground truth (document as deliberate)
Emotion is fundamentally PROSODIC, but the LLM tag (s6) sees only text+binned features.
So `s4b_audio_emotion.py` runs a speech-emotion model ON THE WAVEFORM — exactly the signal
the text LLM cannot see. Default `backend: categorical` uses a label-matched SER classifier
(RAVDESS-trained → 7/10 of our taxonomy incl. `calm`; multilingual `MERaLiON-SER-v1` covers
Tamil for the Telugu half and is the recommended upgrade) mapped onto our taxonomy via
`label_map`; `backend: dimensional` instead regresses arousal/valence/dominance. We compare
the two opinions in valence/arousal quadrant space (`emotion_agree`) and
**route disagreements to the front of human review** (the real ground truth). This is
ALLOWED: the assignment only mandates Sarvam for ASR/diarization/LLM, and Sarvam has no
emotion API — s4b only ADDS an acoustic cross-check, it replaces no Sarvam call. Run order:
s4 (features) → s4b (audio emotion) → s5 (asr) → s6 (tag, sets `emotion_agree`).

## Edge-case decisions (document these as deliberate)
- **Segmentation:** NEVER chop at fixed times (causes mid-word cuts). Default (s2) uses
 VAD/silence detection and cuts only at silences ≥300ms, packing speech into ~22–28s
 windows. Boundaries land between words by construction. **v2 (s2b, opt-in per channel via
 `diarized: true`):** run the Sarvam **Batch API with diarization** on the full recording,
 keep only the DOMINANT speaker's phrase chunks, and pack contiguous chunks — clean phrase
 boundaries + single-speaker guaranteed (best for interview/podcast sources). Batch gives
 chunk-level, not word-level, timestamps.
- **Multiple speakers in one clip:** single-speaker per clip is required. Cheap guard (s3):
 inaSpeechSegmenter gender detection — if BOTH male & female speech each exceed
 `multispeaker_min_fraction`, reject `multi_speaker`. Strong guarantee: the s2b diarized
 path. True simultaneous overlap is unrepairable → reject, never separate.
- **Mixed-gender / 2-person podcast or interview:** set `diarized: true` on that channel.
 s2b keeps only the DOMINANT speaker (most total speech time) and drops the guest's turns, so
 the dataset stays single-speaker — we do NOT attempt source separation. Leave the channel
 `gender: unknown`; s3 detects gender per surviving clip (`gender_detected`) and s6 uses that
 to bin pitch, so you needn't know the dominant speaker up front. The cheap s3 guard is the
 fallback if you forget to set `diarized: true` (it just rejects the mixed clips).
- **Code-switching / two languages at once:** distinguish two cases. (1) *Same speaker mixes
 languages* (e.g. Telugu sentence with English words — extremely common in Indian
 podcasts): this is fine. `saaras:v3` handles code-switching; we pass the channel's DOMINANT
 `language_code` (`te-IN`/`en-IN`) and tag the clip with that language. (2) *Two different
 people each speaking a different language simultaneously*: that's just overlapping speech —
 unrepairable like a music bed. The s3 multi-speaker guard / s2b dominant-speaker selection
 already handle it (reject the overlap, or keep only the dominant voice). We never try to
 split two overlapping voices apart. If a source is so heavily bilingual that the dominant
 language is unclear, set `language: unknown` to let Saaras auto-detect (lower accuracy — a
 last resort, not the default).
- **Crowd noise / applause / laughter (e.g. standup):** great for emotion diversity but the
 crowd overlapping the voice is unrepairable like a music bed, and it's labeled `noise`
 (NOT `music`) by inaSpeechSegmenter. s3 rejects clips whose `noise` fraction exceeds
 `noise_overlap_reject_threshold` (`rejected_reason="crowd_noise"`); survivors get
 `recording_quality` downgraded. Prefer solo/studio sources for emotional range.
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

## As-built status (final)
- **Dataset:** currently a **193-clip / 80.2-min over-collected review buffer** (8 Indian
 English ~40 min, 6 Telugu ~40 min) so clips can be rejected during review and still clear
 the >=30 min per-language floor; re-run `pipeline/balance.py <min>` to cull to final after
 review. 617 clips collected -> dropped 60 music_bed + 79 crowd_noise + 10 manual + 275
 balance_trim. Gender: male 42 min / female 32 min (40%) / unknown 6.8. Whisper: 5 (ASMR).
 23 gold (human-verified).
 - **`En_Podcast_Raj` is Telugu + female, not English + male** — caught during review: it's a
   Telugu podcast with heavy English code-switching that was configured `en-IN`/`male`, so
   Saaras romanized it instead of writing Telugu script. Relabeled `te-IN`/`female` and re-ran
   s5->s6->s7 (clip_ids keep their original `En_Podcast_Raj_*` prefix since clip_id is
   immutable). Re-binned pitch per-gender; lifted the female share 34% -> 42%.
- **Per-channel config flags that emerged** (set on a channel block in config.yaml):
  - `solo: true` — verified single-speaker; s3 skips the noisy gender multi-speaker guard.
  - `diarized: true` — multi-speaker source; s2b keeps only the dominant speaker (s2 skips it).
  - `whisper: true` — ASMR/whispered throughout; s6 forces whisper=true (LLM can't hear it).
  - `gender: male|female|unknown` — backs per-gender pitch_bin; unknown falls back to s3's
    per-clip `gender_detected`.
- **New segmentation knobs:** `clip_pad_ms` (lead-in/out from surrounding silence so clips
  don't start flush on a word onset) and `clip_max_gap_ms` (close a clip on long internal
  silence — needed for ASMR dead air).
- **`pipeline/balance.py [min]`** — over-collect then cull to a balanced N-minute set. Splits
 the target evenly across languages, then fills each language by **round-robin across its
 sources** so a small source (e.g. the 2-min ASMR channel) can't strand budget — leftover is
 redistributed to the other sources, so each language reliably reaches its share (>=30 min).
 Keeps human_verified clips, normalizes kept rows to `described` (re-export-safe), marks the
 rest `balance_trim`.
- **`eval/snapshot.py`** — text corpus report at any stage. **`s8` shuffles** rows on export.
- **Known limitation:** English ASMR is whispered below the −38 dB silence threshold, so VAD
  reads it as silence (0 clips). Fix = per-channel lower `silence_thresh_db`.
