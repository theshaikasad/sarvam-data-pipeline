# Sarvam TTS-Dataset Take-Home — Report

**A 58.6-minute, single-speaker-per-clip TTS dataset (Telugu + Indian English) with accurate transcripts and Parler-TTS-style natural-language style/emotion descriptions, published on HuggingFace.**

Graded on data-quality and curation judgment, so this report front-loads the *decisions* and *iterations*, not the (deliberately simple) pipeline code.

---

## 1. What I built & how the pipeline works

A **resumable, manifest-driven state machine**. `data/manifest.jsonl` is the single source of truth — one JSON row per clip — and every stage filters rows by a `stage` field, processes them, and writes back. Rerunning skips finished rows, so the whole pipeline is crash- and credit-safe.

```
downloaded -> segmented -> music_checked -> transcribed -> tagged -> described -> final
                                                      (or rejected + rejected_reason)
```

| Stage | Does | Tool / model |
|------|------|--------------|
| s1 download | fetch audio, normalize to 16 kHz mono, loudness -23 LUFS | yt-dlp + ffmpeg |
| s2 segment | cut **only at silences >=300 ms** into 18–28 s clips (never mid-word) | pydub VAD |
| s2b diarize | *opt-in:* keep only the **dominant speaker** of a multi-speaker source | **Sarvam Batch STT + diarization** |
| s3 filter | detect & **drop** music beds / crowd noise / mixed-gender multi-speaker | inaSpeechSegmenter |
| s4 features | pitch (YIN), energy, speaking rate | librosa |
| s4b audio-emotion | a **second, audio-grounded** emotion opinion from the waveform | wav2vec2 SER |
| s5 ASR | transcribe each clip with the **known** language code | **Sarvam `saaras:v3`** |
| s6 tag | bin Parler axes + assign emotion/style/whisper | **Sarvam `sarvam-30b`** |
| s7 describe | compose one Parler-style sentence | **Sarvam `sarvam-30b`** |
| s8 export | build + push the HuggingFace dataset & card (shuffled) | datasets / hub |

**Sarvam usage (the assignment mandate):** ASR = `saaras:v3` (`mode=transcribe`, explicit `te-IN`/`en-IN`); diarization = Batch STT API; LLM tagging/description = `sarvam-30b` (OpenAI-compatible). Everything else is non-Sarvam helper tooling — allowed because Sarvam has no API for it.

**Human-in-the-loop, with a hard no-overwrite rule:** machine columns (`asr_*`, `llm_*`) and human columns (`human_*`) are *separate keys and never overwrite each other*. A Gradio review UI writes only `human_*`; the merge (`final_transcript = human_transcript or asr_transcript`, etc.) happens **only at export**. `human_verified` rows form the `gold` split and are the basis for WER/CER.

---

## 2. Iterations to improve data quality

This was the bulk of the work — most "quality" gains came from catching the pipeline silently doing the wrong thing.

1. **The music filter was silently failing on every clip.** `inaSpeechSegmenter`'s bundled `pyannote.algorithms` calls `np.vstack(<generator>)` (rejected by NumPy >=1.24) and its Keras-2 CNN models crash under Keras 3 (TF >=2.16). First run: it errored on all 235 clips -> kept 0, detected **no** music. Fixed with a repo-local viterbi shim + `TF_USE_LEGACY_KERAS=1` (tf-keras). *Quality impact:* music-bed detection started working at all (60 music clips ultimately dropped).

2. **False multi-speaker rejections on solo voices.** The cheap gender-based guard rejected 70 clips as "multi-speaker." Measuring the actual splits showed the gender CNN was labeling **39–51 % of a single, expressive male storyteller's speech as "female"** (pitch modulation fools a binary classifier). Fix: a per-channel `solo: true` opt-out (curator asserts "I listened, it's one speaker") that skips the noisy guard. *Quality impact:* recovered ~70 good Telugu clips and restored source balance.

3. **`sarvam-30b` is a reasoning model — it returned empty answers and drained credits.** With no `max_tokens`, it spent the entire 4096-token starter-tier cap on hidden chain-of-thought (`finish_reason=length`, `content=None`) -> **zero tags produced and maximal cost per call**. The fix (from Sarvam's docs) was `reasoning_effort=None` (explicit JSON null, *not* the string `"none"`, which 400s): direct JSON answers in ~70 tokens / ~0.5 s — **~60x cheaper, ~30x faster.** This was both a correctness *and* a cost bug.

4. **Filename parse bug skipped a whole source.** Channel/video-id parsing used `rpartition("_")`, but YouTube IDs contain underscores (`DV1zxu47_mA`) — so a podcast was silently mis-parsed and skipped. Fixed by matching filenames against known channel names (longest-first).

5. **Clips were cut flush to the first/last sample** (0 ms padding), risking clipped word onsets. Added `clip_pad_ms` (120 ms lead-in/out pulled from the surrounding silence).

6. **ASMR edge cases.** (a) Long internal pauses were being packed into clips -> added `clip_max_gap_ms` (close a clip when the silence gap exceeds 1 s). (b) The LLM can't *hear* whisper from text+features -> per-channel `whisper: true` override. (c) **English ASMR segmented to 0 clips** because it's whispered *below* the -38 dB silence threshold (VAD reads it as silence) — a real limitation, noted in §3.

7. **Over-collect, then balance.** Collected **617 clips** across 14 sources, then `balance.py` culled to a **balanced ~60 min** (even across languages *and* sources, keeping any human-reviewed clips), reporting balance-trims separately from quality rejects.

8. **Ran out of Sarvam credits twice** (once from the reasoning bug, once at scale) — but because the pipeline is fully resumable, each top-up resumed labeling exactly where it stopped with zero rework.

---

## 3. What worked & what didn't

**Worked**
- **Silence-based segmentation:** every clip lands on a silence; **0 of 140 clips exceed the 30 s `saaras:v3` sync limit** (range 18.3–28.2 s); no mid-word cuts.
- **Code-switching:** `saaras:v3` transcribes Telugu<->English mixing cleanly and *transliterates* English into Telugu script (only 1/73 Telugu clips contains Latin text) — a useful, documentable ASR convention.
- **Diarization** pulled clean single-speaker clips out of 3–4-person podcasts (e.g. En_Podcast_Raj, En_Comedy).
- **Crowd-noise filter** correctly dropped **79 standup clips** drowned in applause/laughter while keeping the clean ones.
- The **reasoning-effort fix** turned tagging from impossible/expensive into instant and cheap.

**Didn't (honest)**
- **English ASMR -> 0 clips** (whisper below the silence threshold). Telugu ASMR survived (5 clips). Fix would be a per-channel lower `silence_thresh_db`.
- **Text-LLM vs audio-model emotion agreement is only 24/140 (17 %).** The audio SER model is English/German-trained, so its calibration on Telugu is approximate; we therefore use it *only* as a relative cross-check to route disagreements to humans, never as a label of record.
- Style is still **narrative-leaning** (83/140) even after adding podcasts/standups, because long-form Indian content skews monologue.
- Credits ran out twice mid-run (recoverable, but it interrupted flow).

---

## 4. Quality observations & decisions

**Final dataset: 140 clips / 58.6 min, 14 sources.**

| Language | minutes | sources |
|---|---|---|
| Indian English | 32.0 | 9 (audiobook, solo podcast, 2 multi-host podcasts->diarized, speech, 3 female standups, TED) |
| Telugu | 26.7 | 5 (2 lectures, storytelling, standup, ASMR) |

- **Gender:** male 35.0 min / **female 19.9 min (34 %)** / unknown 3.7 — deliberately added female standups + ASMR to fight an initially 90 %-male roster.
- **Style:** narrative 83, **conversational 50**, dramatic 4, oratorical 3.
- **Emotion:** neutral 43, excited 22, sad 20, serious 15, playful 13, happy 12, angry 11, calm 3, surprised 1.
- **Whisper:** 5 (ASMR).
- **Source funnel:** 617 collected -> dropped 60 music_bed + 79 crowd_noise + 2 manual + 336 balance-trim -> **140 kept**.

**Key deliberate decisions**
- **Two independent emotion opinions.** Emotion is prosodic, but the LLM (s6) sees only text + binned features. So s4b runs a speech-emotion model on the *waveform* — the signal the LLM can't see — and disagreements (116/140) are pushed to the **front of human review**, where human time is most valuable.
- **Detect-and-drop, never repair.** Source separation (Demucs) degrades audio; a repaired clip is worse than none for TTS. Music/crowd over the voice is unrepairable -> reject.
- **Per-gender pitch binning** (male and female live in different Hz ranges), with a per-clip detected-gender fallback when the source gender is unknown.
- **Rich natural-language descriptions** (not bare labels) because Sarvam's Bulbul V3 TTS has no emotion parameter — it *infers* prosody from text, so it trains best on descriptive signal.
- **Honesty about label trust:** only the human-verified `gold` split should be fully trusted; non-gold emotion/style/transcripts are weak supervision from `sarvam-30b`.

**WER/CER:** computed by `eval/compute_wer.py` on the human-verified gold rows (reference = `human_transcript`, hypothesis = `asr_transcript`), reported per-language (CER matters for Telugu script). *Pending* the human pass on the 16-clip review queue.

---

## 5. What I'd improve given more time

- **Per-channel VAD thresholds** so quiet ASMR (and loud standup) segment correctly — recover the English ASMR.
- **A multilingual SER model** (e.g. `MERaLiON-SER-v1`, which covers Tamil and emits valence/arousal/dominance) to raise the audio-emotion agreement on Telugu from its current ~17 %.
- **More female Telugu sources** — Telugu is still 100 % male; the only female additions are English.
- **Finish the human review pass** and publish real WER/CER numbers + a worst-failures table.
- **Scale up** with the YouTube scraper and richer genre coverage (debate, devotional, news) to break the residual narrative lean.
- **Forced alignment** for sub-clip word timing and tighter onset trimming.

---

## Ethics & licensing
Each clip records a `source_type` (public_lecture / podcast_independent / podcast_storytelling / audiobook / standup_comedy / asmr). We avoid copyrighted film/music audio and drop any clip with a detected music bed. Redistribution rights for third-party source audio are not individually cleared, so the dataset is intended for **research/educational use**; the `cc-by-4.0` tag covers the annotations contributed here. Downstream users must verify rights for the underlying audio.
