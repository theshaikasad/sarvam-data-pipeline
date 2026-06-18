"""Gradio human-verification UI.

Iterates over gold_candidate rows not yet human_verified. Reads/writes
data/manifest.jsonl directly (via state.py). Honors the no-overwrite rule: only writes
human_* fields + stage/rejected_reason, NEVER touches asr_* or llm_*.

Run: python review/review_ui.py   (set REVIEWER env var for your name)
"""

from __future__ import annotations

import datetime
import os
import sys

import gradio as gr
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import state  # noqa: E402

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
EMOTIONS = CFG["taxonomy"]["emotion"]
STYLES = CFG["taxonomy"]["style"]
CLIPS_DIR = CFG["paths"]["clips"]
MANIFEST = CFG["paths"]["manifest"]

rows = state.load(MANIFEST)
pending = [cid for cid, r in sorted(rows.items())
           if r.get("gold_candidate") and not r.get("human_verified")
           and r.get("stage") != "rejected"]


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def view(idx: int):
    if idx >= len(pending):
        return (None, "", EMOTIONS[0], STYLES[0], False, "",
                f"All {len(pending)} clips reviewed ✅")
    r = rows[pending[idx]]
    audio = os.path.join(CLIPS_DIR, f"{pending[idx]}.wav")
    info = f"Clip {idx + 1}/{len(pending)} — {r.get('source_channel', '?')} — {pending[idx]}"
    return (audio, r.get("asr_transcript", ""),
            r.get("llm_emotion") or EMOTIONS[0],
            r.get("llm_style") or STYLES[0],
            bool(r.get("whisper", False)),
            r.get("llm_reasoning", ""), info)


def verify(idx, transcript, emotion, style, whisper, reviewer):
    cid = pending[idx]
    fields = dict(human_transcript=transcript, human_emotion=emotion,
                  human_whisper=bool(whisper), human_verified=True,
                  reviewer=reviewer or os.environ.get("REVIEWER", "reviewer"),
                  reviewed_at=now())
    if style != rows[cid].get("llm_style"):
        fields["human_style"] = style
    state.update(rows, cid, **fields)
    state.save(rows, MANIFEST)
    return (idx + 1, *view(idx + 1))


def reject(idx):
    state.update(rows, pending[idx], stage="rejected", rejected_reason="manual_review")
    state.save(rows, MANIFEST)
    return (idx + 1, *view(idx + 1))


with gr.Blocks(title="TTS gold review") as app:
    idx_state = gr.State(0)
    info = gr.Markdown()
    reviewer = gr.Textbox(label="Reviewer", value=os.environ.get("REVIEWER", "reviewer"))
    audio = gr.Audio(type="filepath", label="Clip")
    transcript = gr.Textbox(label="Transcript (editable — ASR prefilled)", lines=4)
    with gr.Row():
        emotion = gr.Dropdown(EMOTIONS, label="Emotion")
        style = gr.Dropdown(STYLES, label="Style")
        whisper = gr.Checkbox(label="Whisper")
    reasoning = gr.Textbox(label="LLM reasoning (read-only)", interactive=False, lines=3)
    with gr.Row():
        verify_btn = gr.Button("Verify & Next", variant="primary")
        reject_btn = gr.Button("Skip / Reject")

    outs = [idx_state, audio, transcript, emotion, style, whisper, reasoning, info]
    verify_btn.click(verify, [idx_state, transcript, emotion, style, whisper, reviewer], outs)
    reject_btn.click(reject, [idx_state], outs)
    app.load(lambda: (0, *view(0)), None, outs)


if __name__ == "__main__":
    if not pending:
        print("No gold candidates pending review. Run review/gold_sample.py first.")
    app.launch()
