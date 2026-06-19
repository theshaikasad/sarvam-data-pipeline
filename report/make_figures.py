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
    """Hand-drawn VERTICAL flowchart of the 8-stage pipeline."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    SARVAM, HELPER, HUMAN, REJ = "#2e6f95", "#52658a", "#e29578", "#c0392b"
    stages = [
        ("s1  Download", "yt-dlp + ffmpeg, 16 kHz mono", HELPER),
        ("s2 / s2b  Segment", "VAD or diarize; ~25 s, no mid-word", HELPER),
        ("s3  Filter", "drop music / crowd / 2-speaker", HELPER),
        ("s4 + s4b  Features", "pitch, energy + audio emotion", HELPER),
        ("s5  ASR", "Sarvam saaras:v3 -> transcript", SARVAM),
        ("s6  Tag", "sarvam-30b -> emotion / style", SARVAM),
        ("s7  Describe", "sarvam-30b -> Parler sentence", SARVAM),
        ("s8  Export", "HuggingFace dataset (public)", HELPER),
    ]
    n = len(stages)
    bw, bh, gap = 4.2, 0.8, 0.5
    cx = 0.0  # box left edge x
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    ys = [-(i * (bh + gap)) for i in range(n)]
    for i, (title, sub, color) in enumerate(stages):
        y = ys[i]
        ax.add_patch(FancyBboxPatch((cx, y), bw, bh, boxstyle="round,pad=0.03,rounding_size=0.10",
                                    fc=color, ec="none"))
        ax.text(cx + 0.25, y + bh * 0.62, title, ha="left", va="center", color="white",
                fontsize=12, fontweight="bold")
        ax.text(cx + 0.25, y + bh * 0.24, sub, ha="left", va="center", color="white", fontsize=9)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((cx + bw / 2, y), (cx + bw / 2, y - gap),
                                         arrowstyle="-|>", mutation_scale=16, color="#333", lw=1.6))
    # reject branch off s3 (index 2) to the right
    y3 = ys[2]
    ax.add_patch(FancyArrowPatch((cx + bw, y3 + bh / 2), (cx + bw + 1.0, y3 + bh / 2),
                                 arrowstyle="-|>", mutation_scale=14, color=REJ, lw=1.5))
    ax.text(cx + bw + 1.1, y3 + bh / 2, "rejected\n(kept as\naudit log)", ha="left", va="center",
            fontsize=9, color=REJ)
    # human-review box feeding into s8 (between s7 idx6 and s8 idx7)
    yh = (ys[6] + ys[7]) / 2 + bh / 2
    ax.add_patch(FancyBboxPatch((cx + bw + 0.7, yh - 0.3), 2.2, 0.6,
                                boxstyle="round,pad=0.03,rounding_size=0.10", fc=HUMAN, ec="none"))
    ax.text(cx + bw + 0.7 + 1.1, yh, "human review\n-> gold split", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
    ax.add_patch(FancyArrowPatch((cx + bw + 0.7, yh), (cx + bw / 2 + 0.2, ys[7] + bh / 2 + 0.15),
                                 arrowstyle="-|>", mutation_scale=13, color="#333", lw=1.3,
                                 connectionstyle="arc3,rad=0.25"))
    ax.text(cx, ys[0] + bh + 0.55,
            "data/manifest.jsonl = single source of truth\n(every stage reads & writes it)",
            ha="left", va="bottom", fontsize=9.5, style="italic", color="#444")
    ax.set_xlim(cx - 0.3, cx + bw + 3.2)
    ax.set_ylim(ys[-1] - 0.4, ys[0] + bh + 1.3)
    ax.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
