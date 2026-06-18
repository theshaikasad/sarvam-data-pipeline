"""WER / CER on the human-verified gold rows, per language.

Reference = human_transcript, hypothesis = asr_transcript. Prints a per-language table,
saves eval/wer_report.json, and lists the 5 worst clips by WER (asr vs human) so the
report can show concrete ASR failure modes. CER matters most for Telugu script.

Usage: python eval/compute_wer.py
"""

from __future__ import annotations

import json
import os
import sys

import jiwer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"
REPORT_PATH = os.path.join("eval", "wer_report.json")


def run(config_path: str = CONFIG_PATH) -> None:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rows = state.load(cfg["paths"]["manifest"])

    gold = [r for r in rows.values()
            if r.get("human_verified") and r.get("human_transcript") is not None]
    if not gold:
        print("No human_verified rows yet. Run review/review_ui.py first.")
        return

    by_lang: dict[str, list[dict]] = {}
    for r in gold:
        by_lang.setdefault(r.get("language", "unknown"), []).append(r)

    report = {"overall": {}, "per_language": {}, "worst": []}
    print(f"{'language':<10}{'clips':>7}{'WER':>9}{'CER':>9}")
    print("-" * 35)
    all_refs, all_hyps = [], []
    for lang in sorted(by_lang):
        refs = [r["human_transcript"] for r in by_lang[lang]]
        hyps = [r.get("asr_transcript", "") for r in by_lang[lang]]
        all_refs += refs
        all_hyps += hyps
        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        report["per_language"][lang] = {"clips": len(refs), "wer": wer, "cer": cer}
        print(f"{lang:<10}{len(refs):>7}{wer:>9.3f}{cer:>9.3f}")

    report["overall"] = {"clips": len(all_refs),
                         "wer": jiwer.wer(all_refs, all_hyps),
                         "cer": jiwer.cer(all_refs, all_hyps)}
    print("-" * 35)
    print(f"{'ALL':<10}{report['overall']['clips']:>7}"
          f"{report['overall']['wer']:>9.3f}{report['overall']['cer']:>9.3f}")

    # 5 worst clips by individual WER
    scored = []
    for r in gold:
        w = jiwer.wer(r["human_transcript"], r.get("asr_transcript", ""))
        scored.append((w, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    print("\nWorst 5 clips by WER:")
    for w, r in scored[:5]:
        report["worst"].append({
            "clip_id": r["clip_id"], "language": r.get("language"), "wer": w,
            "human": r["human_transcript"], "asr": r.get("asr_transcript", ""),
        })
        print(f"  {r['clip_id']} (WER {w:.3f}, {r.get('language')})")
        print(f"    human: {r['human_transcript']}")
        print(f"    asr  : {r.get('asr_transcript', '')}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
