"""Stage 4: acoustic features (additive — does not advance the stage).

For rows at stage "music_checked", compute pitch_mean, pitch_std (librosa.yin, NaNs
dropped), energy_rms. speaking_rate needs the transcript, which doesn't exist until s5,
so it is left null here and filled in s6.

Run s4 BEFORE s5 (both consume "music_checked"; s5 advances rows to "transcribed").
Resumable/idempotent: rows that already have pitch_mean are skipped.
"""

from __future__ import annotations

import os
import sys

import librosa
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"


def features(path: str):
    y, sr = librosa.load(path, sr=None, mono=True)
    f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr)
    f0 = f0[np.isfinite(f0)]
    pitch_mean = float(np.mean(f0)) if f0.size else None
    pitch_std = float(np.std(f0)) if f0.size else None
    energy_rms = float(np.mean(librosa.feature.rms(y=y)))
    return pitch_mean, pitch_std, energy_rms


def run(config_path: str = CONFIG_PATH) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clips_dir = cfg["paths"]["clips"]
    rows = state.load(cfg["paths"]["manifest"])

    done = 0
    for row in state.by_stage(rows, "music_checked"):
        if row.get("pitch_mean") is not None:
            continue  # already computed
        clip_path = os.path.join(clips_dir, f"{row['clip_id']}.wav")
        if not os.path.exists(clip_path):
            print(f"  [warn] missing clip: {clip_path}")
            continue
        pitch_mean, pitch_std, energy_rms = features(clip_path)
        state.update(rows, row["clip_id"],
                     pitch_mean=round(pitch_mean, 2) if pitch_mean else None,
                     pitch_std=round(pitch_std, 2) if pitch_std else None,
                     energy_rms=round(energy_rms, 6),
                     speaking_rate=None)  # filled in s6 once transcript exists
        done += 1
    state.save(rows, cfg["paths"]["manifest"])
    print(f"s4 done. features computed for {done} clips.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
