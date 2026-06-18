"""Stage 2: VAD/silence-based segmentation.

NEVER cuts at fixed time offsets (that causes mid-word cuts). Instead:
  1. detect_nonsilent() finds speech spans separated by silences >= min_silence_ms.
  2. Greedily pack consecutive speech spans into a clip; only CLOSE a clip at a silence
     gap (i.e. at the end of a span), once the accumulated duration reaches
     target_clip_seconds.
  3. Keep a clip only if its duration is within [min_clip_seconds, max_clip_seconds].

Because every boundary lands inside a silence, clips never cut mid-word, and capping at
max_clip_seconds keeps each clip safely under the Saaras v3 30s sync limit.

Creates one manifest row per kept clip (stage="segmented"), inheriting language and
source_type from the channel config. Resumable: a raw file is skipped if any clip rows
for it already exist.
"""

from __future__ import annotations

import glob
import os
import sys

import yaml
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import state

CONFIG_PATH = "config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_raw_name(path: str) -> tuple[str, str]:
    """data/raw/<channel>_<videoid>.wav -> (channel, videoid). channel may contain '_'."""
    base = os.path.splitext(os.path.basename(path))[0]
    channel, _, videoid = base.rpartition("_")
    return channel, videoid


def channel_lookup(cfg: dict) -> dict[str, dict]:
    return {c["name"]: c for c in cfg["channels"]}


def pack_spans(spans, target_ms, min_ms, max_ms):
    """Yield (start_ms, end_ms) clips greedily packed from non-silent spans.

    Boundaries only ever land at span ends (i.e. inside a silence), so clips never cut
    mid-word. A span is added only if it keeps the clip within max_ms; once the clip
    reaches target_ms it is closed. Clips outside [min_ms, max_ms] are dropped.
    """
    clip_start = clip_end = None
    for start, end in spans:
        if clip_start is None:
            clip_start, clip_end = start, end
        elif end - clip_start <= max_ms:
            clip_end = end
        else:
            # adding this span would overflow max -> close current clip at last silence
            if clip_end - clip_start >= min_ms:
                yield clip_start, clip_end
            clip_start, clip_end = start, end
        # close proactively once we've packed enough (and stayed within max)
        if clip_end - clip_start >= target_ms:
            if min_ms <= clip_end - clip_start <= max_ms:
                yield clip_start, clip_end
            clip_start = clip_end = None
    if clip_start is not None and min_ms <= clip_end - clip_start <= max_ms:
        yield clip_start, clip_end


def segment_raw(raw_path: str, cfg: dict, channels: dict, rows: dict) -> int:
    channel, videoid = parse_raw_name(raw_path)
    ch_cfg = channels.get(channel)
    if ch_cfg is None:
        print(f"  [skip] {os.path.basename(raw_path)}: channel '{channel}' not in config.")
        return 0
    if ch_cfg.get("diarized"):
        print(f"  [skip] {os.path.basename(raw_path)}: diarized channel (handled by s2b).")
        return 0

    prefix = f"{channel}_{videoid}_"
    if any(cid.startswith(prefix) for cid in rows):
        print(f"  [skip] {os.path.basename(raw_path)}: already segmented.")
        return 0

    clips_dir = cfg["paths"]["clips"]
    os.makedirs(clips_dir, exist_ok=True)

    audio = AudioSegment.from_wav(raw_path)
    spans = detect_nonsilent(
        audio,
        min_silence_len=int(cfg["min_silence_ms"]),
        silence_thresh=int(cfg["silence_thresh_db"]),
    )

    target_ms = int(cfg["target_clip_seconds"] * 1000)
    min_ms = int(cfg["min_clip_seconds"] * 1000)
    max_ms = int(cfg["max_clip_seconds"] * 1000)
    # A little breathing room so clips don't start/end flush on a word onset; pulled from the
    # surrounding silence (boundaries are at >= min_silence_ms gaps, so this stays in silence).
    pad_ms = int(cfg.get("clip_pad_ms", 0))

    count = 0
    for start_ms, end_ms in pack_spans(spans, target_ms, min_ms, max_ms):
        clip_id = f"{channel}_{videoid}_{count:03d}"
        clip_path = os.path.join(clips_dir, f"{clip_id}.wav")
        ps = max(0, start_ms - pad_ms)
        pe = min(len(audio), end_ms + pad_ms)
        audio[ps:pe].export(clip_path, format="wav")
        state.update(
            rows, clip_id,
            source_url=f"https://youtu.be/{videoid}",
            source_channel=channel,
            source_type=ch_cfg["source_type"],
            language=ch_cfg["language"],
            gender=ch_cfg.get("gender", "unknown"),
            start_time=round(ps / 1000.0, 3),
            end_time=round(pe / 1000.0, 3),
            duration=round((pe - ps) / 1000.0, 3),
            stage="segmented",
        )
        count += 1
    print(f"  [segment] {os.path.basename(raw_path)} -> {count} clips")
    return count


def run(config_path: str = CONFIG_PATH) -> None:
    cfg = load_config(config_path)
    channels = channel_lookup(cfg)
    rows = state.load(cfg["paths"]["manifest"])

    raw_files = sorted(glob.glob(os.path.join(cfg["paths"]["raw"], "*.wav")))
    raw_files = [p for p in raw_files if not p.endswith(".src.wav")]

    total = 0
    for raw_path in raw_files:
        total += segment_raw(raw_path, cfg, channels, rows)

    state.save(rows, cfg["paths"]["manifest"])
    print(f"s2 done. {total} new clips created.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
