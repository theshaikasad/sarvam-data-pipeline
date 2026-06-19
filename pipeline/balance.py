"""Trim to a balanced ~target-minute dataset (keep reviewed clips; drop unreviewed excess).

The pipeline over-collects on purpose; this culls down to a balanced submission set:
  - split the target evenly across LANGUAGES, then evenly across each language's SOURCES,
  - always KEEP every human_verified clip (never throw away reviewed work),
  - fill the rest of each source's budget with a deterministic shuffle of its unreviewed clips,
  - mark everything not selected as stage="rejected", rejected_reason="balance_trim".

Re-runnable: it first un-trims any previous balance_trim, so you can re-balance freely.
"balance_trim" is distinct from quality rejects (music_bed/crowd_noise/...) so the report can
report culling separately from filtering. Recoverable (just flip the stage back).

Usage: python pipeline/balance.py [target_minutes=60] [seed=42]
"""

from __future__ import annotations

import collections
import os
import random
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"
ELIGIBLE = ("described", "final")


def run(target_minutes: float = 60.0, seed: int = 42, config_path: str = CONFIG_PATH) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    manifest = cfg["paths"]["manifest"]
    rows = state.load(manifest)
    rng = random.Random(seed)

    # Un-trim a previous balance pass so re-running re-balances from the full pool.
    for cid, r in rows.items():
        if r.get("rejected_reason") == "balance_trim":
            state.update(rows, cid, stage="described", rejected_reason=None)

    pool = [r for r in rows.values() if r.get("stage") in ELIGIBLE]
    by_lang: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        by_lang[r.get("language", "?")].append(r)

    per_lang_sec = (target_minutes * 60.0) / max(1, len(by_lang))
    keep: set[str] = set()
    for lang, clips in by_lang.items():
        by_src: dict[str, list[dict]] = collections.defaultdict(list)
        for r in clips:
            by_src[r.get("source_channel", "?")].append(r)

        # Always keep reviewed clips, then fill toward the language budget by ROUND-ROBIN
        # across sources. Round-robin keeps sources balanced AND redistributes any budget a
        # small source (e.g. a 2-min ASMR channel) can't absorb to the other sources, so the
        # language reliably reaches its target instead of undershooting.
        sec = 0.0
        queues: dict[str, collections.deque] = {}
        for src in sorted(by_src):
            sclips = by_src[src]
            verified = [r for r in sclips if r.get("human_verified")]
            others = [r for r in sclips if not r.get("human_verified")]
            rng.shuffle(others)
            for r in verified:                      # never throw away reviewed work
                keep.add(r["clip_id"]); sec += float(r.get("duration", 0))
            queues[src] = collections.deque(others)

        while sec < per_lang_sec:
            progressed = False
            for src in sorted(queues):
                if sec >= per_lang_sec:
                    break
                q = queues[src]
                if q:
                    r = q.popleft()
                    keep.add(r["clip_id"]); sec += float(r.get("duration", 0))
                    progressed = True
            if not progressed:                       # whole language pool exhausted
                break

    # Normalize: kept rows -> "described" (the pre-export stage) so a fresh s8 rebuilds the
    # complete set; everything else in the pool becomes a recoverable balance_trim reject.
    ntrim = 0
    for r in pool:
        if r["clip_id"] in keep:
            state.update(rows, r["clip_id"], stage="described", rejected_reason=None)
        else:
            state.update(rows, r["clip_id"], stage="rejected", rejected_reason="balance_trim")
            ntrim += 1
    state.save(rows, manifest)

    # report
    kept = [r for r in rows.values() if r.get("stage") in ELIGIBLE]
    print(f"Balanced to ~{target_minutes:.0f} min target: kept {len(kept)} clips, "
          f"trimmed {ntrim} unreviewed.")
    bylang_sec: dict[str, float] = collections.defaultdict(float)
    bysrc: dict[tuple, list] = collections.defaultdict(lambda: [0, 0.0])
    for r in kept:
        d = float(r.get("duration", 0))
        bylang_sec[r.get("language")] += d
        s = bysrc[(r.get("language"), r.get("source_channel"))]
        s[0] += 1; s[1] += d
    for lang in sorted(bylang_sec):
        print(f"  {lang}: {bylang_sec[lang] / 60:.1f} min")
        for (lg, src), (n, sec) in sorted(bysrc.items()):
            if lg == lang:
                print(f"      {src:<18} {n:>3} clips  {sec / 60:>5.1f} min")
    print(f"  TOTAL: {sum(bylang_sec.values()) / 60:.1f} min")


if __name__ == "__main__":
    tmin = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run(tmin, sd)
