"""Stage 3: music-bed detection (detect and DROP, never repair).

For every row at stage "segmented", run inaSpeechSegmenter on the clip and compute the
fraction of clip duration labeled "music". Source separation degrades audio and TTS needs
clean recordings, so a clip with a music bed is rejected rather than repaired.

  - has_music = (music fraction > music_overlap_reject_threshold)
  - music_confidence = music fraction
  - if music fraction > threshold: stage="rejected", rejected_reason="music_bed"
    else:                          stage="music_checked"

Rejected rows are KEPT in the manifest — their counts are the report's iteration log.
Resumable: only rows still at stage "segmented" are processed.
"""

from __future__ import annotations

import os
import sys

import yaml

import state

CONFIG_PATH = "config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def music_fraction(segmentation, clip_duration: float) -> float:
    """Fraction of clip_duration labeled 'music' by inaSpeechSegmenter."""
    if clip_duration <= 0:
        return 0.0
    music = sum(end - start for label, start, end in segmentation if label == "music")
    return min(music / clip_duration, 1.0)


def run(config_path: str = CONFIG_PATH) -> None:
    cfg = load_config(config_path)
    clips_dir = cfg["paths"]["clips"]
    threshold = float(cfg["music_overlap_reject_threshold"])
    rows = state.load(cfg["paths"]["manifest"])

    pending = list(state.by_stage(rows, "segmented"))
    if not pending:
        print("s3: nothing at stage 'segmented'. Done.")
        return

    # Import + init the segmenter once (heavy: loads TF models).
    from inaSpeechSegmenter import Segmenter
    seg = Segmenter(vad_engine="smn", detect_gender=False)

    segmented = len(pending)
    rejected = 0
    kept = 0
    for row in pending:
        clip_id = row["clip_id"]
        clip_path = os.path.join(clips_dir, f"{clip_id}.wav")
        if not os.path.exists(clip_path):
            print(f"  [warn] missing clip file: {clip_path}")
            continue
        try:
            segmentation = seg(clip_path)
        except Exception as e:  # noqa: BLE001 - keep the batch going
            print(f"  [error] segmenter failed on {clip_id}: {e}")
            continue

        frac = music_fraction(segmentation, float(row["duration"]))
        has_music = frac > threshold
        if has_music:
            state.update(rows, clip_id, has_music=True, music_confidence=round(frac, 4),
                         stage="rejected", rejected_reason="music_bed")
            rejected += 1
        else:
            state.update(rows, clip_id, has_music=False, music_confidence=round(frac, 4),
                         stage="music_checked")
            kept += 1

    state.save(rows, cfg["paths"]["manifest"])
    print(f"s3 done. segmented {segmented} -> rejected {rejected} for music -> kept {kept}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
