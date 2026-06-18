"""Stage 8: build the HuggingFace dataset + dataset card and push (public).

Takes rows at stage "described" that are NOT rejected, computes the EXPORT-ONLY derived
fields (CLAUDE.md no-overwrite rule: machine/human keys stay separate; we only combine
them here at export time):
    final_transcript = human_transcript or asr_transcript
    final_emotion    = human_emotion    or llm_emotion
    final_style      = human_style      or llm_style
    final_whisper    = human_whisper if reviewed else machine whisper

Builds a datasets.Dataset with the final columns from CLAUDE.md (audio cast to a 16kHz
Audio feature), splits into "gold" (human_verified) vs "train", and push_to_hub as a
PUBLIC repo using HF_TOKEN. Also composes a dataset card (README.md) — sources table,
methodology, minutes per language, emotion distribution, WER/CER, limitations, licensing,
and the Bulbul-V3 prosody rationale for the description field — and uploads it as the card.

Idempotent: rows that export are advanced to stage "final" so a rerun won't duplicate work
(pass --no-advance to keep them at "described"). Use --dry-run to build locally and skip
the push (writes the card + a local Arrow snapshot under data/hf_export/).

Usage:
    python pipeline/s8_export.py            # build + push (needs HF_TOKEN + hf.repo_id)
    python pipeline/s8_export.py --dry-run  # build locally, write card, skip push
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from datetime import date

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"
CARD_PATH = os.path.join("report", "README.md")
LOCAL_EXPORT_DIR = os.path.join("data", "hf_export")

LANG_NAME = {"te-IN": "Telugu", "en-IN": "Indian English"}

# Final HF columns, in the order CLAUDE.md lists them.
FINAL_COLUMNS = [
    "audio", "text", "language", "emotion", "style", "whisper",
    "speaking_rate_bin", "pitch_bin", "pitch_variation", "recording_quality",
    "description", "speaker_id", "duration", "source_url", "source_channel",
    "source_type", "human_verified", "split",
]


def load_env() -> None:
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


def speaker_id(channel: str) -> str:
    """Anonymized, stable per-source speaker id, e.g. 'Harshaneeyam' -> 'spk_harshaneeyam'."""
    slug = "".join(c if c.isalnum() else "_" for c in (channel or "unknown").lower())
    slug = "_".join(p for p in slug.split("_") if p)  # collapse repeats
    return f"spk_{slug}"


def final_fields(row: dict) -> dict:
    """Compute the EXPORT-ONLY merged fields. Never mutates the manifest row."""
    human_verified = bool(row.get("human_verified"))
    return {
        "text": row.get("human_transcript") or row.get("asr_transcript") or "",
        "emotion": row.get("human_emotion") or row.get("llm_emotion") or "neutral",
        "style": row.get("human_style") or row.get("llm_style") or "narrative",
        # human_whisper only exists once reviewed; else trust the machine flag.
        "whisper": bool(row.get("human_whisper")
                        if "human_whisper" in row else row.get("whisper", False)),
        "human_verified": human_verified,
        "split": "gold" if human_verified else "train",
    }


def build_records(rows: dict, clips_dir: str) -> list[dict]:
    """One export record per non-rejected 'described' row that has an audio file."""
    records = []
    for row in state.by_stage(rows, "described"):
        cid = row["clip_id"]
        clip_path = os.path.join(clips_dir, f"{cid}.wav")
        if not os.path.exists(clip_path):
            print(f"  [warn] skipping {cid}: missing clip file {clip_path}")
            continue
        merged = final_fields(row)
        records.append({
            "clip_id": cid,
            "audio": clip_path,
            "text": merged["text"],
            "language": row.get("language", "unknown"),
            "emotion": merged["emotion"],
            "style": merged["style"],
            "whisper": merged["whisper"],
            "speaking_rate_bin": row.get("speaking_rate_bin"),
            "pitch_bin": row.get("pitch_bin"),
            "pitch_variation": row.get("pitch_variation"),
            "recording_quality": row.get("recording_quality"),
            "description": row.get("description", ""),
            "speaker_id": speaker_id(row.get("source_channel", "")),
            "duration": float(row.get("duration", 0.0)),
            "source_url": row.get("source_url", ""),
            "source_channel": row.get("source_channel", ""),
            "source_type": row.get("source_type", ""),
            "human_verified": merged["human_verified"],
            "split": merged["split"],
        })
    return records


def to_dataset(records: list[dict]):
    """Build a datasets.Dataset with audio cast to a 16kHz Audio feature."""
    from datasets import Audio, Dataset

    columns = {col: [r[col] for r in records] for col in FINAL_COLUMNS}
    ds = Dataset.from_dict(columns)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


# --------------------------------------------------------------------------- card

def _load_wer_report() -> dict | None:
    path = os.path.join("eval", "wer_report.json")
    if not os.path.exists(path):
        return None
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_card(records: list[dict], cfg: dict, repo_id: str) -> str:
    n = len(records)
    n_gold = sum(1 for r in records if r["human_verified"])

    minutes = defaultdict(float)
    for r in records:
        minutes[r["language"]] += r["duration"] / 60.0
    total_min = sum(minutes.values())

    emotions = Counter(r["emotion"] for r in records)
    styles = Counter(r["style"] for r in records)

    # sources table: channel -> (language, source_type, clips, minutes)
    src = {}
    for r in records:
        ch = r["source_channel"] or "unknown"
        s = src.setdefault(ch, {"lang": r["language"], "type": r["source_type"],
                                "clips": 0, "min": 0.0, "spk": r["speaker_id"]})
        s["clips"] += 1
        s["min"] += r["duration"] / 60.0

    source_rows = [
        [ch, s["spk"], LANG_NAME.get(s["lang"], s["lang"]), s["type"],
         s["clips"], f"{s['min']:.1f}"]
        for ch, s in sorted(src.items())
    ]
    sources_table = _md_table(
        ["channel", "speaker_id", "language", "source_type", "clips", "minutes"],
        source_rows,
    )

    minutes_table = _md_table(
        ["language", "clips", "minutes"],
        [[LANG_NAME.get(l, l),
          sum(1 for r in records if r["language"] == l),
          f"{minutes[l]:.1f}"] for l in sorted(minutes)]
        + [["**TOTAL**", n, f"{total_min:.1f}"]],
    )

    emotion_table = _md_table(
        ["emotion", "clips"],
        [[e, c] for e, c in emotions.most_common()],
    )
    style_table = _md_table(
        ["style", "clips"],
        [[s, c] for s, c in styles.most_common()],
    )

    # WER/CER
    wer = _load_wer_report()
    if wer:
        wer_rows = [[LANG_NAME.get(l, l), d["clips"], f"{d['wer']:.3f}", f"{d['cer']:.3f}"]
                    for l, d in sorted(wer["per_language"].items())]
        o = wer["overall"]
        wer_rows.append(["**ALL**", o["clips"], f"{o['wer']:.3f}", f"{o['cer']:.3f}"])
        wer_section = (
            "WER (word error rate) and CER (character error rate) are computed with "
            "`jiwer` over the **human-verified gold** rows only, comparing the raw "
            "`saaras:v3` hypothesis against the human reference. CER is the more meaningful "
            "metric for Telugu (Indic script, agglutinative morphology).\n\n"
            + _md_table(["language", "gold clips", "WER", "CER"], wer_rows)
        )
        worst = wer.get("worst", [])
        if worst:
            wer_section += "\n\n**Worst ASR failures (gold):**\n\n"
            for w in worst[:5]:
                wer_section += (
                    f"- `{w['clip_id']}` ({LANG_NAME.get(w['language'], w['language'])}, "
                    f"WER {w['wer']:.2f})\n"
                    f"  - human: {w['human']}\n"
                    f"  - asr&nbsp;&nbsp;: {w['asr']}\n"
                )
    else:
        wer_section = (
            "_No `eval/wer_report.json` found at export time. Run "
            "`python eval/compute_wer.py` after human verification to populate WER/CER._"
        )

    taxonomy = cfg["taxonomy"]
    today = date.today().isoformat()

    # YAML metadata header for the HF card.
    header = f"""---
license: cc-by-4.0
language:
- te
- en
pretty_name: Sarvam-style Single-Speaker TTS Dataset (Telugu + Indian English)
task_categories:
- text-to-speech
tags:
- tts
- speech
- telugu
- indian-english
- parler-tts
- emotion
- prosody
size_categories:
- n<1K
---
"""

    body = f"""# Sarvam-style TTS Training Dataset

A small, high-curation single-speaker-per-clip TTS dataset (~{total_min:.0f} minutes,
{n} clips) of **Telugu** and **Indian English** speech, each clip annotated with an
accurate transcript and a rich, Parler-TTS-style natural-language **style/emotion
description**. Built as a take-home; the emphasis is **data quality and curation
judgment**, not pipeline code.

- **Clips:** {n} (~25s each, always cut at silence so no word is ever clipped)
- **Total audio:** {total_min:.1f} min
- **Gold (human-verified) subset:** {n_gold} clips (`split="gold"`)
- **Audio:** 16kHz mono WAV, loudness-normalized
- **Repo:** `{repo_id}`
- **Generated:** {today}

## Sources

Each clip records its provenance and a `source_type`. Speaker identity is anonymized to a
stable `spk_*` id per source channel.

{sources_table}

## Minutes per language

{minutes_table}

## Label distributions

Emotion (`final_emotion = human_emotion or llm_emotion`):

{emotion_table}

Style (`final_style = human_style or llm_style`):

{style_table}

## Transcription quality (WER / CER)

{wer_section}

## Methodology

The pipeline is a resumable, manifest-driven state machine
(`downloaded -> segmented -> music_checked -> transcribed -> tagged -> described ->
final`), with `data/manifest.jsonl` as the single source of truth (one JSON row per clip).

1. **Download & normalize** — audio fetched per channel, resampled to 16kHz mono and
   loudness-normalized.
2. **Segmentation** — VAD/silence detection (silences >= {cfg['min_silence_ms']}ms);
   speech is packed into ~{cfg['min_clip_seconds']}-{cfg['max_clip_seconds']}s windows and
   boundaries land **only inside silences**, so clips never cut mid-word. The ~25s target
   also keeps every clip under the Saaras v3 30s synchronous limit.
3. **Music filter (detect & DROP, never repair)** — `inaSpeechSegmenter` labels
   speech/music; if music overlaps >
   {int(cfg['music_overlap_reject_threshold'] * 100)}% of a clip it is rejected
   (`rejected_reason="music_bed"`). Source separation degrades audio and TTS needs clean
   recordings, so a repaired clip is worse than none. Rejected rows are kept as the
   iteration log.
4. **Acoustic features** — `pitch_mean`/`pitch_std` (librosa YIN, NaNs dropped),
   `energy_rms`, and `speaking_rate`.
5. **ASR** — `saaras:v3`, `mode="transcribe"`, with the **known** language code per
   channel (not auto-detect) for higher accuracy. Empty transcripts are rejected
   (`rejected_reason="empty_asr"`).
6. **Tagging** — acoustic features are binned into Parler axes (speaking rate / pitch /
   pitch variation / recording quality); `sarvam-30b` then assigns emotion, style, and a
   whisper flag via a strict JSON-only prompt over the transcript + features.
7. **Description** — `sarvam-30b` composes ONE natural-language sentence from the
   structured fields.
8. **Human review & export** — a stratified gold sample is verified by a human in a Gradio
   UI; WER/CER are computed on that gold subset; this dataset is then exported.

### No-overwrite rule

Machine columns (`asr_*`, `llm_*`) and human columns (`human_*`) are **separate keys** and
never overwrite each other. The `final_*` fields here are computed **only at export**:
`final_transcript = human_transcript or asr_transcript`,
`final_emotion = human_emotion or llm_emotion`. `human_verified=true` rows form the `gold`
split and are the basis for the WER/CER numbers above.

## Why a natural-language `description` (Bulbul V3 rationale)

Sarvam's TTS model **Bulbul V3 has no explicit emotion parameter** — it is LLM-based and
infers prosody from the text/context it is given. So instead of bare one-word labels, every
clip carries a rich Parler-TTS-style sentence (e.g. *"A male speaker narrates a Telugu
story in a slow, sorrowful tone with moderate pitch variation, recorded clearly with almost
no background noise."*). A prosody-inferring model trains best on that descriptive signal,
and the structured columns remain available for filtering/conditioning.

## Columns

`audio` (16kHz), `text` (=`final_transcript`), `language`, `emotion` (=`final_emotion`),
`style`, `whisper`, `speaking_rate_bin`, `pitch_bin`, `pitch_variation`,
`recording_quality`, `description`, `speaker_id` (anonymized per source), `duration`,
`source_url`, `source_channel`, `source_type`, `human_verified`, `split`
(`"gold"` if human-verified else `"train"`).

### Taxonomy

- **emotion:** {", ".join(taxonomy['emotion'])}
- **style:** {", ".join(taxonomy['style'])}
- **speaking_rate_bin:** {", ".join(taxonomy['speaking_rate_bin'])}
- **pitch_bin:** {", ".join(taxonomy['pitch_bin'])} (per-gender)
- **pitch_variation:** {", ".join(taxonomy['pitch_variation'])}
- **recording_quality:** {", ".join(taxonomy['recording_quality'])}

## Known limitations

- **Small scale.** Designed as a curated seed (~1 hour), not a production corpus.
- **Machine labels are weak supervision.** Emotion/style/description come from `sarvam-30b`
  over acoustic features; only the `gold` split is human-verified. Treat non-gold labels as
  noisy.
- **ASR errors remain** in non-gold `text`. See the WER/CER section for measured error and
  worst-case examples; CER is higher for Telugu by script nature.
- **Pitch bins are gender-agnostic heuristics** unless gender was annotated.
- **Per-clip single speaker**, but the dataset spans multiple speakers (one per source).

## Ethics & licensing

- Audio is sourced from publicly available recordings; each clip records its `source_type`
  ({", ".join(sorted(set(r['source_type'] for r in records if r['source_type'])))}).
- We **avoid copyrighted film/music audio** and drop any clip with a detected music bed.
- Licensing posture is stated **honestly**: redistribution rights for third-party source
  audio are not individually cleared, so this dataset is intended for **research and
  educational use**. The card is marked `cc-by-4.0` for the *annotations* contributed here;
  downstream users must verify rights for the underlying audio before commercial use.
- `ai4bharat/Mann-ki-Baat` (CC BY 4.0) can serve as a verified-transcript quality anchor.

## Reproduce

```bash
pip install -r requirements.txt
# set SARVAM_KEY and HF_TOKEN in .env, edit config.yaml channels + hf.repo_id
python pipeline/s1_download.py && python pipeline/s2_segment.py
python pipeline/s3_music_filter.py && python pipeline/s4_features.py
python pipeline/s5_asr.py && python pipeline/s6_tag.py && python pipeline/s7_describe.py
python review/gold_sample.py && python review/review_ui.py   # human verification
python eval/compute_wer.py && python eval/distributions.py
python pipeline/s8_export.py
```
"""
    return header + body


# --------------------------------------------------------------------------- run

def run(config_path: str = CONFIG_PATH, dry_run: bool = False,
        advance: bool = True) -> None:
    load_env()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    clips_dir = cfg["paths"]["clips"]
    manifest = cfg["paths"]["manifest"]
    repo_id = os.environ.get("HF_REPO_ID") or cfg.get("hf", {}).get("repo_id", "")
    private = bool(cfg.get("hf", {}).get("private", False))

    rows = state.load(manifest)
    records = build_records(rows, clips_dir)
    if not records:
        print("s8: nothing at stage 'described' to export. Done.")
        return

    n_gold = sum(1 for r in records if r["human_verified"])
    print(f"s8: exporting {len(records)} clips ({n_gold} gold / "
          f"{len(records) - n_gold} train).")

    # Dataset card first (so it exists even on a dry run / failed push).
    card = build_card(records, cfg, repo_id or "<unset-repo-id>")
    os.makedirs(os.path.dirname(CARD_PATH), exist_ok=True)
    with open(CARD_PATH, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"  wrote dataset card -> {CARD_PATH}")

    ds = to_dataset(records)
    print(f"  built dataset: {ds}")

    if dry_run:
        os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)
        ds.save_to_disk(LOCAL_EXPORT_DIR)
        print(f"  [dry-run] saved local snapshot -> {LOCAL_EXPORT_DIR} (push skipped).")
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN not set (see .env.example). Use --dry-run to skip pushing.")
        if not repo_id or "your-hf-username" in repo_id:
            sys.exit("Set a real hf.repo_id in config.yaml (or HF_REPO_ID env) before push.")

        print(f"  pushing PUBLIC dataset -> {repo_id} ...")
        ds.push_to_hub(repo_id, private=private, token=token)

        # Overwrite the auto-generated card with ours.
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=CARD_PATH,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"  pushed + uploaded dataset card. "
              f"https://huggingface.co/datasets/{repo_id}")

    # Advance exported rows to 'final' (idempotent rerun-safe).
    if advance:
        for r in records:
            state.update(rows, r["clip_id"], stage="final")
        state.save(rows, manifest)
        print(f"  advanced {len(records)} rows to stage 'final'.")

    print("s8 done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry-run" in args
    no_advance = "--no-advance" in args
    positional = [a for a in args if not a.startswith("--")]
    cfg_path = positional[0] if positional else CONFIG_PATH
    run(cfg_path, dry_run=dry, advance=not no_advance)
