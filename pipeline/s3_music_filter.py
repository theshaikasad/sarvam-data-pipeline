"""Stage 3: music / crowd-noise / multi-speaker filter (detect and DROP, never repair).

For every row at stage "segmented", run inaSpeechSegmenter (with gender detection so speech
is split into male/female) and measure, as a fraction of clip duration:
  - music  -> music bed (intros, background score)
  - noise  -> applause / laughter / crowd (the standup-special problem)
  - male / female speech -> if BOTH exceed multispeaker_min_fraction the clip almost
    certainly has >1 speaker (cheap mixed-gender guard; true single-speaker enforcement is
    the diarization-based s2b path).

Source separation degrades audio and TTS needs clean single-speaker recordings, so any of
these is REJECTED rather than repaired (a repaired clip is worse than none). Rejection
priority: music_bed -> crowd_noise -> multi_speaker. Kept clips advance to "music_checked"
and carry music_confidence / noise_confidence / gender_detected for downstream use.

  fields: has_music, music_confidence, has_noise, noise_confidence, gender_detected,
          has_multi_speaker; stage + rejected_reason on rejection.

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


def label_fractions(segmentation, clip_duration: float) -> dict:
    """Fraction of clip_duration per inaSpeechSegmenter label (music/noise/male/female)."""
    if clip_duration <= 0:
        return {}
    totals: dict[str, float] = {}
    for label, start, end in segmentation:
        totals[label] = totals.get(label, 0.0) + (end - start)
    return {k: min(v / clip_duration, 1.0) for k, v in totals.items()}


def run(config_path: str = CONFIG_PATH) -> None:
    cfg = load_config(config_path)
    clips_dir = cfg["paths"]["clips"]
    music_thr = float(cfg["music_overlap_reject_threshold"])
    noise_thr = float(cfg.get("noise_overlap_reject_threshold", 1.0))
    spk_min = float(cfg.get("multispeaker_min_fraction", 1.0))
    rows = state.load(cfg["paths"]["manifest"])

    pending = list(state.by_stage(rows, "segmented"))
    if not pending:
        print("s3: nothing at stage 'segmented'. Done.")
        return

    # Import + init the segmenter once (heavy: loads TF models). detect_gender=True so we
    # can both label speaker gender and catch mixed-gender (multi-speaker) clips.
    from inaSpeechSegmenter import Segmenter
    seg = Segmenter(vad_engine="smn", detect_gender=True)

    segmented = len(pending)
    rej_music = rej_noise = rej_spk = kept = 0
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

        frac = label_fractions(segmentation, float(row["duration"]))
        music = frac.get("music", 0.0)
        noise = frac.get("noise", 0.0)
        male = frac.get("male", 0.0)
        female = frac.get("female", 0.0)
        multi = male >= spk_min and female >= spk_min
        gender_detected = ("male" if male >= female else "female") if (male or female) else "unknown"

        common = dict(
            music_confidence=round(music, 4), has_music=music > music_thr,
            noise_confidence=round(noise, 4), has_noise=noise > noise_thr,
            gender_detected=gender_detected, has_multi_speaker=multi,
        )
        # Rejection priority: music -> crowd noise -> multi-speaker.
        if music > music_thr:
            state.update(rows, clip_id, **common, stage="rejected", rejected_reason="music_bed")
            rej_music += 1
        elif noise > noise_thr:
            state.update(rows, clip_id, **common, stage="rejected", rejected_reason="crowd_noise")
            rej_noise += 1
        elif multi:
            state.update(rows, clip_id, **common, stage="rejected", rejected_reason="multi_speaker")
            rej_spk += 1
        else:
            state.update(rows, clip_id, **common, stage="music_checked")
            kept += 1

    state.save(rows, cfg["paths"]["manifest"])
    print(f"s3 done. {segmented} segmented -> rejected music={rej_music}, "
          f"crowd_noise={rej_noise}, multi_speaker={rej_spk} -> kept {kept}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
