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


def stratified_sample(rows: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cands = [r for r in rows.values() if r.get("stage") in ELIGIBLE_STAGES]
    if not cands:
        return []

    by_lang: dict[str, list[dict]] = {}
    for r in cands:
        by_lang.setdefault(r.get("language", "unknown"), []).append(r)

    langs = sorted(by_lang)
    # n is PER LANGUAGE: each language independently targets n gold candidates.
    targets = {l: n for l in langs}

    selected: list[dict] = []
    for lang in langs:
        buckets: dict[str, list[dict]] = {}
        for r in by_lang[lang]:
            buckets.setdefault(r.get("llm_emotion", "unknown"), []).append(r)
        for e in buckets:
            buckets[e].sort(key=lambda r: r["clip_id"])  # determinism before shuffle
            rng.shuffle(buckets[e])
            # Stable sort so LLM<->audio disagreements sit at the END of the bucket and are
            # therefore pop()'d (i.e. selected) first; ties keep the shuffled order.
            buckets[e].sort(key=lambda r: 1 if r.get("emotion_agree") == "disagree" else 0)

        target = min(targets[lang], len(by_lang[lang]))
        picked: list[dict] = []
        order = sorted(buckets)
        while len(picked) < target:
            progressed = False
            for e in order:
                if buckets[e]:
                    picked.append(buckets[e].pop())
                    progressed = True
                    if len(picked) >= target:
                        break
            if not progressed:
                break
        selected.extend(picked)
    return selected


def run(n: int = DEFAULT_N, seed: int = DEFAULT_SEED, config_path: str = CONFIG_PATH) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    manifest = cfg["paths"]["manifest"]
    rows = state.load(manifest)

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
