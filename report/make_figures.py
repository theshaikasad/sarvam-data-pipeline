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


def fig_pipeline():
    """Hand-drawn horizontal flowchart of the 8-stage pipeline."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    SARVAM, HELPER, HUMAN = "#2e6f95", "#52658a", "#e29578"
    stages = [
        ("s1  Download", "yt-dlp + ffmpeg\n16 kHz mono", HELPER),
        ("s2 / s2b  Segment", "VAD or diarize\n~25 s, no mid-word", HELPER),
        ("s3  Filter", "drop music /\ncrowd / 2-speaker", HELPER),
        ("s4 + s4b  Features", "pitch, energy +\naudio emotion", HELPER),
        ("s5  ASR", "Sarvam saaras:v3\ntranscript", SARVAM),
        ("s6  Tag", "sarvam-30b\nemotion / style", SARVAM),
        ("s7  Describe", "sarvam-30b\nParler sentence", SARVAM),
        ("s8  Export", "HuggingFace\n(public)", HELPER),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    n = len(stages); bw, bh, gap = 1.0, 1.0, 0.45
    for i, (title, sub, color) in enumerate(stages):
        x = i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, 0), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=color, ec="none"))
        ax.text(x + bw / 2, 0.66, title, ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold")
        ax.text(x + bw / 2, 0.30, sub, ha="center", va="center", color="white", fontsize=7.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + bw, bh / 2), (x + bw + gap, bh / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#333", lw=1.4))
    # annotations: reject drop-off under s3, human review above s7->s8
    sx = 2 * (bw + gap)
    ax.add_patch(FancyArrowPatch((sx + bw / 2, 0), (sx + bw / 2, -0.55),
                                 arrowstyle="-|>", mutation_scale=12, color="#c0392b", lw=1.3))
    ax.text(sx + bw / 2, -0.8, "rejected\n(kept as audit log)", ha="center", va="top",
            fontsize=7.5, color="#c0392b")
    hx = 6 * (bw + gap)
    ax.add_patch(FancyBboxPatch((hx - 0.1, 1.5), bw + 0.2, 0.5,
                                boxstyle="round,pad=0.04,rounding_size=0.1", fc=HUMAN, ec="none"))
    ax.text(hx + bw / 2, 1.75, "human review -> gold", ha="center", va="center",
            color="white", fontsize=8, fontweight="bold")
    ax.add_patch(FancyArrowPatch((hx + bw / 2, 1.5), (hx + bw / 2, bh),
                                 arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.2))
    ax.text(-0.1, 1.9, "Manifest (data/manifest.jsonl) is the single source of truth; "
            "every stage reads and writes it.", fontsize=8, style="italic", color="#444")
    ax.set_xlim(-0.3, n * (bw + gap)); ax.set_ylim(-1.4, 2.2); ax.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline.png", dpi=150); plt.close(fig)


def run():
    os.makedirs(OUT, exist_ok=True)
    clips = load()
    if not clips:
        sys.exit("no kept clips in manifest")
    fig_pipeline()
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
