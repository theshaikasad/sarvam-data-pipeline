"""Draw a reproducible, stratified gold sample for human verification.

From rows at stage "final" (or "described"), select N rows PER LANGUAGE and, within each
language, spread across llm_emotion buckets so the gold set isn't all one emotion. Within
each bucket, clips where the LLM and the audio model DISAGREE on emotion (emotion_agree ==
"disagree", from s4b/s6) are picked FIRST — human review time is most valuable exactly where
the two automatic opinions conflict. Selection is deterministic for a fixed seed. Marks
chosen rows with gold_candidate=true and prints their clip_ids.

So `python review/gold_sample.py 25` gives 25 Telugu + 25 English = 50 gold candidates.

Usage: python review/gold_sample.py [N_per_language=25] [seed=42]
"""

from __future__ import annotations

import os
import random
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"
DEFAULT_N = 50  # split evenly across languages -> 25 Telugu + 25 Indian English
DEFAULT_SEED = 42
ELIGIBLE_STAGES = ("final", "described")


def _channel_queue(clips: list[dict], rng: random.Random) -> list[dict]:
    """Per-channel pick order: emotion-spread, with the clips that most NEED a human ear
    (LLM<->audio emotion disagreement, or whispered) pulled to the FRONT."""
    buckets: dict[str, list[dict]] = {}
    for r in clips:
        buckets.setdefault(r.get("llm_emotion", "unknown"), []).append(r)
    for e in buckets:
        buckets[e].sort(key=lambda r: r["clip_id"])  # determinism before shuffle
        rng.shuffle(buckets[e])
    q: list[dict] = []
    order = sorted(buckets)
    while any(buckets.values()):                    # round-robin across emotions -> spread
        for e in order:
            if buckets[e]:
                q.append(buckets[e].pop())
    q.sort(key=lambda r: 0 if (r.get("emotion_agree") == "disagree" or r.get("whisper"))
           else 1)                                   # stable: priority clips first
    return q


def stratified_sample(rows: dict, n: int, seed: int) -> list[dict]:
    """~n gold candidates PER LANGUAGE, spread EVENLY across that language's sources (so the
    review set isn't dominated by one chatty channel), disagreement/whisper clips first."""
    rng = random.Random(seed)
    cands = [r for r in rows.values() if r.get("stage") in ELIGIBLE_STAGES]
    if not cands:
        return []

    by_lang: dict[str, list[dict]] = {}
    for r in cands:
        by_lang.setdefault(r.get("language", "unknown"), []).append(r)

    selected: list[dict] = []
    for lang in sorted(by_lang):
        by_chan: dict[str, list[dict]] = {}
        for r in by_lang[lang]:
            by_chan.setdefault(r.get("source_channel", "?"), []).append(r)
        queues = {c: _channel_queue(by_chan[c], rng) for c in by_chan}
        target = min(n, len(by_lang[lang]))
        picked: list[dict] = []
        order = sorted(queues)
        while len(picked) < target and any(queues.values()):
            for c in order:                          # round-robin across channels -> balance
                if queues[c]:
                    picked.append(queues[c].pop(0))
                    if len(picked) >= target:
                        break
        selected.extend(picked)
    return selected


def run(n: int = DEFAULT_N, seed: int = DEFAULT_SEED, config_path: str = CONFIG_PATH) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    manifest = cfg["paths"]["manifest"]
    rows = state.load(manifest)

    # Fresh sample each run: clear any previous gold_candidate (human_verified is untouched,
    # so already-reviewed clips stay reviewed).
    for r in rows.values():
        r.pop("gold_candidate", None)

    selected = stratified_sample(rows, n, seed)
    for r in selected:
        state.update(rows, r["clip_id"], gold_candidate=True)
    state.save(rows, manifest)

    n_disagree = sum(1 for r in selected if r.get("emotion_agree") == "disagree")
    print(f"Sampled {len(selected)} gold candidates (N={n} per language, seed={seed}); "
          f"{n_disagree} are LLM<->audio disagreements (prioritised):")
    for r in selected:
        flag = "  <- LLM/audio disagree" if r.get("emotion_agree") == "disagree" else ""
        print(f"  {r['clip_id']}  [{r.get('language')}, llm={r.get('llm_emotion')}, "
              f"audio={r.get('audio_emotion')}]{flag}")


if __name__ == "__main__":
    argn = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    argseed = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEED
    run(argn, argseed)
