"""Draw a reproducible, stratified gold sample for human verification.

From rows at stage "final" (or "described"), select N rows balanced across language and,
within each language, across llm_emotion buckets. Selection is deterministic for a fixed
seed. Marks chosen rows with gold_candidate=true and prints their clip_ids.

Usage: python review/gold_sample.py [N=25] [seed=42]
"""

from __future__ import annotations

import os
import random
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"
DEFAULT_N = 25
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
    base, rem = divmod(n, len(langs))
    targets = {l: base + (1 if i < rem else 0) for i, l in enumerate(langs)}

    selected: list[dict] = []
    for lang in langs:
        buckets: dict[str, list[dict]] = {}
        for r in by_lang[lang]:
            buckets.setdefault(r.get("llm_emotion", "unknown"), []).append(r)
        for e in buckets:
            buckets[e].sort(key=lambda r: r["clip_id"])  # determinism before shuffle
            rng.shuffle(buckets[e])

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

    print(f"Sampled {len(selected)} gold candidates (N={n}, seed={seed}):")
    for r in selected:
        print(f"  {r['clip_id']}  [{r.get('language')}, {r.get('llm_emotion')}]")


if __name__ == "__main__":
    argn = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    argseed = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEED
    run(argn, argseed)
