"""Dataset distribution plots for the report.

Over non-rejected rows, plots:
  - emotion histogram (final emotion = human_emotion or llm_emotion)
  - clip-duration histogram
  - total minutes per language (bar)
  - text-LLM vs audio-model emotion agreement (bar), if s4b has run
Saves PNGs to report/ and prints the per-language minute totals + the agreement rate.

Usage: python eval/distributions.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import emotion_map  # noqa: E402
import state  # noqa: E402

CONFIG_PATH = "config.yaml"
OUT_DIR = "report"


def run(config_path: str = CONFIG_PATH) -> None:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rows = state.load(cfg["paths"]["manifest"])
    os.makedirs(OUT_DIR, exist_ok=True)

    kept = [r for r in rows.values() if r.get("stage") not in (None, "rejected")]
    if not kept:
        print("No non-rejected rows to plot.")
        return

    emotions = Counter((r.get("human_emotion") or r.get("llm_emotion") or "unknown")
                       for r in kept)
    durations = [r.get("duration", 0) for r in kept]
    minutes = defaultdict(float)
    for r in kept:
        minutes[r.get("language", "unknown")] += r.get("duration", 0) / 60.0

    # emotion histogram
    plt.figure(figsize=(8, 4))
    labels, counts = zip(*sorted(emotions.items()))
    plt.bar(labels, counts)
    plt.title("Emotion distribution")
    plt.ylabel("clips")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "emotion_hist.png"), dpi=120)
    plt.close()

    # duration histogram
    plt.figure(figsize=(8, 4))
    plt.hist(durations, bins=20)
    plt.title("Clip duration distribution")
    plt.xlabel("seconds")
    plt.ylabel("clips")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "duration_hist.png"), dpi=120)
    plt.close()

    # minutes per language
    plt.figure(figsize=(6, 4))
    langs = sorted(minutes)
    plt.bar(langs, [minutes[l] for l in langs])
    plt.title("Total minutes per language")
    plt.ylabel("minutes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "language_minutes.png"), dpi=120)
    plt.close()

    # text-LLM vs audio-model emotion agreement (valence/arousal quadrant)
    agree = Counter()
    for r in kept:
        a = emotion_map.agreement(r.get("llm_emotion"), r.get("audio_emotion"))
        if a is not None:
            agree[a] += 1
    n_judged = sum(agree.values())
    extra_plot = ""
    if n_judged:
        plt.figure(figsize=(5, 4))
        labels = ["agree", "disagree"]
        plt.bar(labels, [agree.get(l, 0) for l in labels],
                color=["#1a7f37", "#c0392b"])
        plt.title("Emotion: text-LLM vs audio model")
        plt.ylabel("clips")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "emotion_agreement.png"), dpi=120)
        plt.close()
        extra_plot = ", emotion_agreement"

    print(f"Clips (non-rejected): {len(kept)}")
    for lang in langs:
        print(f"  {lang}: {minutes[lang]:.1f} min")
    print(f"  TOTAL: {sum(minutes.values()):.1f} min")
    if n_judged:
        pct = 100.0 * agree.get("agree", 0) / n_judged
        print(f"LLM<->audio emotion agreement: {agree.get('agree', 0)}/{n_judged} "
              f"({pct:.0f}%); {agree.get('disagree', 0)} disagreements flagged for review.")
    else:
        print("LLM<->audio agreement: n/a (run pipeline/s4b_audio_emotion.py).")
    print(f"Saved plots to {OUT_DIR}/ "
          f"(emotion_hist, duration_hist, language_minutes{extra_plot}).")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
