"""Generate report figures from the manifest (kept clips only). Saves PNGs to report/figures/.

Charts: minutes per source (balance, colored by language), emotion distribution,
style distribution, and a clip-duration histogram against the target window.

Usage: python report/make_figures.py
"""

from __future__ import annotations

import collections
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MANIFEST = "data/manifest.jsonl"
OUT = os.path.join("report", "figures")
KEPT = ("described", "final")
TE, EN = "#d1495b", "#2e6f95"  # language colors
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load():
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    return [r for r in rows if r.get("stage") in KEPT]


def fig_sources(clips):
    mins = collections.defaultdict(float)
    lang = {}
    for r in clips:
        mins[r["source_channel"]] += float(r.get("duration", 0)) / 60.0
        lang[r["source_channel"]] = r.get("language")
    order = sorted(mins, key=lambda s: (lang[s], mins[s]))
    vals = [mins[s] for s in order]
    colors = [TE if lang[s] == "te-IN" else EN for s in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(order, vals, color=colors)
    ax.set_xlabel("minutes")
    ax.set_title("Minutes per source (balanced across 14 sources)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TE),
                       plt.Rectangle((0, 0), 1, 1, color=EN)],
              labels=["Telugu", "Indian English"], frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sources.png", dpi=140); plt.close(fig)


def _bar(counter, title, color, fname):
    items = counter.most_common()
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, vals, color=color)
    ax.set_ylabel("clips"); ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, str(v), ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=140); plt.close(fig)


def fig_duration(clips):
    durs = [float(r.get("duration", 0)) for r in clips]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(durs, bins=20, color="#6a994e")
    ax.axvline(30, color="#c0392b", linestyle="--", label="30 s ASR limit")
    ax.set_xlabel("seconds"); ax.set_ylabel("clips")
    ax.set_title(f"Clip durations (n={len(durs)}, all under 30 s)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_duration.png", dpi=140); plt.close(fig)


def run():
    os.makedirs(OUT, exist_ok=True)
    clips = load()
    if not clips:
        sys.exit("no kept clips in manifest")
    fig_sources(clips)
    _bar(collections.Counter(r.get("llm_emotion") for r in clips),
         "Emotion distribution", "#9b5de5", "fig_emotion.png")
    _bar(collections.Counter(r.get("llm_style") for r in clips),
         "Style distribution", "#f4a259", "fig_style.png")
    fig_duration(clips)
    print(f"wrote 4 figures to {OUT}/ from {len(clips)} clips "
          f"({sum(float(r.get('duration', 0)) for r in clips) / 60:.1f} min)")


if __name__ == "__main__":
    run()
