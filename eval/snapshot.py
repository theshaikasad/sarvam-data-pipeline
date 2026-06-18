"""Text corpus snapshot — an honest, at-a-glance summary of the dataset's current state.

Complements eval/distributions.py (which draws the emotion/duration PLOTS once s6 has run).
This one prints TEXT tables that are meaningful at every pipeline stage: corpus size, the
reject log, minutes per language/channel, duration compliance, acoustic-feature ranges,
transcript length, Telugu<->English code-switching, and (once present) the machine emotion
distributions. Read-only.

Usage: python eval/snapshot.py
"""

from __future__ import annotations

import collections
import os
import re
import statistics
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"
LANG_NAME = {"te-IN": "Telugu", "en-IN": "Indian English"}
LATIN = re.compile(r"[A-Za-z]")
DEVA_OR_INDIC = re.compile(r"[^\x00-\x7F]")  # any non-ASCII (Telugu script etc.)


def fmt_min(seconds: float) -> str:
    return f"{seconds / 60.0:.1f}"


def bar(n: int, total: int, width: int = 24) -> str:
    filled = int(round(width * n / total)) if total else 0
    return "█" * filled + "·" * (width - filled)


def run(config_path: str = CONFIG_PATH) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rows = list(state.load(cfg["paths"]["manifest"]).values())
    if not rows:
        print("Manifest is empty — nothing to analyze yet.")
        return

    kept = [r for r in rows if r.get("stage") != "rejected"]
    rej = [r for r in rows if r.get("stage") == "rejected"]

    print("=" * 64)
    print("CORPUS SNAPSHOT")
    print("=" * 64)
    print(f"Total clips collected : {len(rows)}")
    print(f"  kept (non-rejected) : {len(kept)}")
    print(f"  rejected            : {len(rej)}")
    by_stage = collections.Counter(r.get("stage") for r in rows)
    print("  by stage            : "
          + ", ".join(f"{s}={n}" for s, n in sorted(by_stage.items(), key=lambda x: -x[1])))

    # ---- reject log (the iteration story) ----
    if rej:
        print("\n--- Reject log ---")
        for reason, n in collections.Counter(r.get("rejected_reason") for r in rows
                                              if r.get("stage") == "rejected").most_common():
            print(f"  {reason or '?':<16} {n:>4}")

    # ---- minutes per language ----
    print("\n--- Kept clips per language ---")
    lang_secs: dict[str, float] = collections.defaultdict(float)
    lang_n: dict[str, int] = collections.Counter()
    for r in kept:
        lang_secs[r.get("language", "?")] += float(r.get("duration", 0))
        lang_n[r.get("language", "?")] += 1
    total_secs = sum(lang_secs.values())
    for lang in sorted(lang_secs):
        print(f"  {LANG_NAME.get(lang, lang):<15} {lang_n[lang]:>4} clips  "
              f"{fmt_min(lang_secs[lang]):>6} min  {bar(lang_n[lang], len(kept))}")
    print(f"  {'TOTAL':<15} {len(kept):>4} clips  {fmt_min(total_secs):>6} min")

    # ---- per channel ----
    print("\n--- Kept clips per channel ---")
    ch_secs: dict[str, float] = collections.defaultdict(float)
    ch_n: dict[str, int] = collections.Counter()
    ch_meta: dict[str, tuple] = {}
    for r in kept:
        ch = r.get("source_channel", "?")
        ch_secs[ch] += float(r.get("duration", 0))
        ch_n[ch] += 1
        ch_meta[ch] = (r.get("language", "?"), r.get("gender", "?"),
                       r.get("segmentation", "vad"))
    print(f"  {'channel':<20}{'lang':<8}{'gender':<8}{'seg':<11}{'clips':>6}{'min':>7}")
    for ch in sorted(ch_secs):
        lang, gen, seg = ch_meta[ch]
        print(f"  {ch:<20}{lang:<8}{gen:<8}{seg:<11}{ch_n[ch]:>6}{fmt_min(ch_secs[ch]):>7}")

    # ---- gender balance (minutes) ----
    print("\n--- Speaker gender balance (by minutes) ---")
    gen_secs: dict[str, float] = collections.defaultdict(float)
    for r in kept:
        gen_secs[r.get("gender", "unknown")] += float(r.get("duration", 0))
    for g in sorted(gen_secs):
        pct = 100.0 * gen_secs[g] / total_secs if total_secs else 0
        print(f"  {g:<10} {fmt_min(gen_secs[g]):>6} min  ({pct:4.0f}%)  {bar(int(gen_secs[g]), int(total_secs) or 1)}")

    # ---- duration compliance ----
    durs = [float(r.get("duration", 0)) for r in kept if r.get("duration")]
    if durs:
        lo, hi = cfg.get("min_clip_seconds", 0), cfg.get("max_clip_seconds", 99)
        pad = cfg.get("clip_pad_ms", 0) / 1000.0
        over = [d for d in durs if d > 30]
        print("\n--- Clip duration (s) ---")
        print(f"  min={min(durs):.1f}  mean={statistics.mean(durs):.1f}  "
              f"median={statistics.median(durs):.1f}  max={max(durs):.1f}")
        print(f"  target window [{lo}-{hi}]s (+{pad:.2f}s pad each end); "
              f"clips over 30s (ASR sync limit): {len(over)}")

    # ---- acoustic features ----
    pitches = [(r.get("gender", "unknown"), r.get("pitch_mean")) for r in kept
               if r.get("pitch_mean") is not None]
    if pitches:
        print("\n--- Pitch (Hz, mean F0) by declared gender ---")
        by_g: dict[str, list] = collections.defaultdict(list)
        for g, p in pitches:
            by_g[g].append(p)
        for g in sorted(by_g):
            vals = by_g[g]
            print(f"  {g:<10} n={len(vals):>4}  "
                  f"mean={statistics.mean(vals):5.0f}  "
                  f"range={min(vals):.0f}-{max(vals):.0f}")

    # ---- transcript length + code-switching ----
    print("\n--- Transcripts ---")
    have_asr = [r for r in kept if r.get("asr_transcript")]
    print(f"  clips with ASR transcript: {len(have_asr)}/{len(kept)}")
    if have_asr:
        lens = [len(r["asr_transcript"]) for r in have_asr]
        print(f"  chars/clip: mean={statistics.mean(lens):.0f}  "
              f"min={min(lens)}  max={max(lens)}")
        # Code-switching: fraction of Telugu clips whose transcript also contains Latin script.
        te = [r for r in have_asr if r.get("language") == "te-IN"]
        if te:
            mixed = sum(1 for r in te if LATIN.search(r["asr_transcript"]))
            print(f"  Telugu clips containing English (Latin) tokens: "
                  f"{mixed}/{len(te)} ({100.0 * mixed / len(te):.0f}%)  [code-switching]")

    # ---- machine emotion (audio model + LLM), if present ----
    ae = [r.get("audio_emotion") for r in kept if r.get("audio_emotion")]
    if ae:
        print(f"\n--- Audio-model emotion (s4b) — {len(ae)}/{len(kept)} scored ---")
        for emo, n in collections.Counter(ae).most_common():
            print(f"  {emo:<12} {n:>4}  {bar(n, len(ae))}")
    le = [r.get("llm_emotion") for r in kept if r.get("llm_emotion")]
    if le:
        print(f"\n--- LLM emotion (s6) — {len(le)}/{len(kept)} tagged ---")
        for emo, n in collections.Counter(le).most_common():
            print(f"  {emo:<12} {n:>4}  {bar(n, len(le))}")
        agree = collections.Counter(r.get("emotion_agree") for r in kept
                                    if r.get("emotion_agree"))
        if agree:
            judged = sum(agree.values())
            print(f"  text-LLM vs audio agreement: "
                  f"{agree.get('agree', 0)}/{judged} agree "
                  f"({100.0 * agree.get('agree', 0) / judged:.0f}%)")
    else:
        print("\n(LLM emotion/style + descriptions not yet generated — run s6/s7 for the full "
              "label analysis, then eval/distributions.py for plots.)")

    print("=" * 64)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
