"""Stage 2b (v2 segmentation): diarization-based, single-speaker clips.

The default segmenter (s2) packs VAD speech spans — clean boundaries, but it has no idea WHO
is speaking, so an interview/podcast with a guest would mix speakers across clips. This v2
path uses the Sarvam **Batch Speech-to-Text API** with `with_diarization=True` on the WHOLE
raw recording (up to 2h), which returns phrase-level chunks each tagged with a `speaker_id`.
We then:
  1. pick the DOMINANT speaker (most total speech time) — the single voice for this source;
  2. keep only that speaker's chunks, dropping every other-speaker interjection;
  3. pack temporally-CONTIGUOUS dominant-speaker chunks into ~target-second clips, cutting
     only at chunk (phrase) boundaries, so clips never cut mid-word AND are guaranteed
     single-speaker by construction.

Note: the Batch API gives chunk-level (sentence/phrase) timestamps, not word-level, so we
cut at phrase boundaries — clean, and stronger than blind/VAD for multi-speaker sources.

Opt-in per channel via `diarized: true` in config.yaml (s2 skips those channels). Produces
manifest rows at stage "segmented" (then the normal s3 -> s4 -> s4b -> s5 ... pipeline runs),
adding `speaker_id` (diarization label) and `segmentation="diarized_v2"`. Resumable: a raw
file whose clips already exist is skipped. Requires SARVAM_KEY and the `sarvamai` SDK.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

import yaml
from pydub import AudioSegment

sys.path.insert(0, os.path.dirname(__file__))
import state  # noqa: E402
from s2_segment import channel_lookup, parse_raw_name  # noqa: E402

CONFIG_PATH = "config.yaml"
MAX_GAP_MS = 1200  # break a clip if the dominant speaker is silent/interrupted this long


def load_env() -> None:
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


def diarize(raw_path: str, language_code: str, model: str, mode: str, key: str) -> list[dict]:
    """Run a Sarvam batch diarization job on one file; return diarized_transcript entries."""
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=key)
    job = client.speech_to_text_job.create_job(
        model=model, language_code=language_code, mode=mode, with_diarization=True)
    job.upload_files(file_paths=[raw_path])
    job.start()
    job.wait_until_complete()

    out_dir = tempfile.mkdtemp(prefix="sarvam_diar_")
    job.download_outputs(output_dir=out_dir)
    entries: list[dict] = []
    for jf in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        entries.extend((data.get("diarized_transcript") or {}).get("entries") or [])
    return entries


def dominant_speaker(entries: list[dict]) -> str | None:
    totals: dict[str, float] = {}
    for e in entries:
        dur = float(e["end_time_seconds"]) - float(e["start_time_seconds"])
        totals[str(e.get("speaker_id"))] = totals.get(str(e.get("speaker_id")), 0.0) + dur
    return max(totals, key=totals.get) if totals else None


def pack_entries(entries, speaker, target_ms, min_ms, max_ms):
    """Yield (start_ms, end_ms, transcript) for contiguous runs of `speaker`'s chunks.

    Boundaries land at chunk (phrase) ends. A run breaks when another speaker intervenes or
    the dominant speaker is silent for > MAX_GAP_MS, so each clip is contiguous same-speaker
    audio (no splicing across removed segments).
    """
    run_start = run_end = None
    parts: list[str] = []

    def flush(start, end, texts):
        if start is not None and min_ms <= end - start <= max_ms:
            yield start, end, " ".join(t.strip() for t in texts if t).strip()

    prev_end = None
    for e in entries:
        s = int(float(e["start_time_seconds"]) * 1000)
        en = int(float(e["end_time_seconds"]) * 1000)
        if str(e.get("speaker_id")) != speaker:
            continue
        # Break the current clip if this chunk isn't contiguous with the last kept one.
        if run_start is not None and (prev_end is None or s - prev_end > MAX_GAP_MS
                                      or en - run_start > max_ms):
            yield from flush(run_start, run_end, parts)
            run_start, parts = None, []
        if run_start is None:
            run_start = s
        run_end = en
        parts.append(e.get("transcript", ""))
        prev_end = en
        if run_end - run_start >= target_ms:
            yield from flush(run_start, run_end, parts)
            run_start, parts = None, []
    if run_start is not None:
        yield from flush(run_start, run_end, parts)


def segment_raw(raw_path, cfg, channels, rows, key) -> int:
    channel, videoid = parse_raw_name(raw_path)
    ch_cfg = channels.get(channel)
    if ch_cfg is None or not ch_cfg.get("diarized"):
        return 0
    prefix = f"{channel}_{videoid}_"
    if any(cid.startswith(prefix) for cid in rows):
        print(f"  [skip] {os.path.basename(raw_path)}: already segmented.")
        return 0

    clips_dir = cfg["paths"]["clips"]
    os.makedirs(clips_dir, exist_ok=True)
    asr = cfg["asr"]
    entries = diarize(raw_path, ch_cfg["language"], asr["model"], asr["mode"], key)
    if not entries:
        print(f"  [warn] {os.path.basename(raw_path)}: no diarized entries returned.")
        return 0
    speaker = dominant_speaker(entries)

    audio = AudioSegment.from_wav(raw_path)
    target_ms = int(cfg["target_clip_seconds"] * 1000)
    min_ms = int(cfg["min_clip_seconds"] * 1000)
    max_ms = int(cfg["max_clip_seconds"] * 1000)

    count = 0
    for start_ms, end_ms, transcript in pack_entries(entries, speaker, target_ms, min_ms, max_ms):
        clip_id = f"{channel}_{videoid}_{count:03d}"
        audio[start_ms:end_ms].export(os.path.join(clips_dir, f"{clip_id}.wav"), format="wav")
        state.update(
            rows, clip_id,
            source_url=f"https://youtu.be/{videoid}",
            source_channel=channel,
            source_type=ch_cfg["source_type"],
            language=ch_cfg["language"],
            gender=ch_cfg.get("gender", "unknown"),
            start_time=round(start_ms / 1000.0, 3),
            end_time=round(end_ms / 1000.0, 3),
            duration=round((end_ms - start_ms) / 1000.0, 3),
            speaker_id=str(speaker),
            segmentation="diarized_v2",
            stage="segmented",
        )
        count += 1
    print(f"  [diarized] {os.path.basename(raw_path)} -> {count} clips "
          f"(dominant speaker {speaker} of {len({str(e.get('speaker_id')) for e in entries})})")
    return count


def run(config_path: str = CONFIG_PATH) -> None:
    load_env()
    key = os.environ.get("SARVAM_KEY")
    if not key:
        sys.exit("SARVAM_KEY not set (see .env.example).")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    channels = channel_lookup(cfg)
    diarized = [c["name"] for c in cfg["channels"] if c.get("diarized")]
    if not diarized:
        print("s2b: no channels have `diarized: true` in config. Nothing to do.")
        return
    rows = state.load(cfg["paths"]["manifest"])

    raw_files = sorted(glob.glob(os.path.join(cfg["paths"]["raw"], "*.wav")))
    raw_files = [p for p in raw_files if not p.endswith(".src.wav")]

    total = 0
    for raw_path in raw_files:
        total += segment_raw(raw_path, cfg, channels, rows, key)
    state.save(rows, cfg["paths"]["manifest"])
    print(f"s2b done. {total} new diarized clips created for channels: {', '.join(diarized)}.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
