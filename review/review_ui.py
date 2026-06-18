"""Minimal Gradio review UI: listen -> fix the transcript -> Update saves to the manifest.

Deliberately plain: uniform text size, Back / Next navigation, a single Update button that
writes any edits, and Reject. No reviewer-name box, no emoji clutter. Language + scope
toggles at the top (grandma -> Telugu; scope = the gold subset, all kept clips, or rejected).

Every Update/Reject writes data/manifest.jsonl via state.py and HONORS the no-overwrite rule:
it only sets human_* fields (+ stage/rejected_reason) and NEVER touches asr_* or llm_*.
Reviewer name comes from the REVIEWER env var (default "reviewer").

Run:  python review/review_ui.py        (or REVIEWER="Ammamma" python review/review_ui.py)
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
REVIEWER = os.environ.get("REVIEWER", "reviewer")

LANGS = [("Telugu", "te-IN"), ("English", "en-IN")]
SCOPES = [("Gold sample", "gold"), ("All clips", "all"), ("Rejected", "rejected")]

rows = state.load(MANIFEST)


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def pending_for(lang: str, scope: str) -> list[str]:
    def keep(r) -> bool:
        if r.get("language") != lang:
            return False
        if scope == "gold":
            return (bool(r.get("gold_candidate")) and not r.get("human_verified")
                    and r.get("stage") != "rejected")
        if scope == "rejected":
            return r.get("stage") == "rejected"
        return r.get("stage") != "rejected"
    return [cid for cid, r in sorted(rows.items()) if keep(r)]


def render(plist: list[str], idx: int):
    total = len(plist)
    if total == 0:
        return (None, "", EMOTIONS[0], STYLES[0], False, "No clips here.",
                gr.update(value="Update"))
    idx = max(0, min(idx, total - 1))
    cid = plist[idx]
    r = rows[cid]
    verified = bool(r.get("human_verified"))
    ndone = sum(1 for c in plist if rows[c].get("human_verified"))
    status = "✅ VERIFIED" if verified else "⬜ not reviewed"
    info = f"Clip {idx + 1} of {total}  ·  {status}  ·  {ndone}/{total} done"
    btn = gr.update(value="✓ Update again" if verified else "Update")
    return (
        os.path.join(CLIPS_DIR, f"{cid}.wav"),
        r.get("human_transcript") or r.get("asr_transcript", ""),
        r.get("human_emotion") or r.get("llm_emotion") or EMOTIONS[0],
        r.get("human_style") or r.get("llm_style") or STYLES[0],
        bool(r.get("human_whisper") if "human_whisper" in r else r.get("whisper", False)),
        info,
        btn,
    )


def load_lang(lang: str, scope: str):
    plist = pending_for(lang, scope)
    return (plist, 0, *render(plist, 0))


def go(plist, idx, step):
    return (max(0, min(idx + step, len(plist) - 1)) if plist else 0,
            *render(plist, idx + step))


def update(plist, idx, transcript, emotion, style, whisper):
    """Save edits to the current clip's human_* fields, then move to the next."""
    if plist and 0 <= idx < len(plist):
        cid = plist[idx]
        fields = dict(human_transcript=transcript, human_emotion=emotion,
                      human_whisper=bool(whisper), human_verified=True,
                      reviewer=REVIEWER, reviewed_at=now())
        if style != rows[cid].get("llm_style"):
            fields["human_style"] = style
        state.update(rows, cid, **fields)
        state.save(rows, MANIFEST)
    nxt = min(idx + 1, len(plist) - 1) if plist else 0
    return (nxt, *render(plist, nxt))


def reject(plist, idx):
    if plist and 0 <= idx < len(plist):
        state.update(rows, plist[idx], stage="rejected", rejected_reason="manual_review")
        state.save(rows, MANIFEST)
    nxt = min(idx + 1, len(plist) - 1) if plist else 0
    return (nxt, *render(plist, nxt))


# Uniform TEXT size — scoped to text elements only, NOT the audio player (a wildcard there
# collapses its play-button icon). Then explicitly size the player's control icons.
CSS = """
.gradio-container {max-width: 880px !important; margin: auto !important;}
#info {text-align: center; font-weight: 700; font-size: 22px !important; padding: 6px 0;}
#transcript textarea {font-size: 22px !important; line-height: 1.6 !important;}
label span, .gradio-container label {font-size: 20px !important;}
#back, #update, #reject, #next {font-size: 22px !important; padding: 16px !important;
    font-weight: 700 !important;}
#update {background: #1a7f37 !important; color: #fff !important;}
#reject {background: #c0392b !important; color: #fff !important;}
/* keep the audio player's play / skip icons clearly visible */
#audio svg {width: 26px !important; height: 26px !important;}
#audio button {opacity: 1 !important;}
"""

with gr.Blocks(title="Review", elem_id="app") as app:
    plist_state = gr.State([])
    idx_state = gr.State(0)

    with gr.Row():
        lang = gr.Radio(LANGS, value="te-IN", label="Language")
        scope = gr.Radio(SCOPES, value="all", label="Show")

    info = gr.Markdown(elem_id="info")
    audio = gr.Audio(type="filepath", label="Clip", interactive=False, autoplay=True,
                     elem_id="audio")
    transcript = gr.Textbox(label="Transcript (edit if wrong)", lines=5, elem_id="transcript")
    with gr.Row():
        emotion = gr.Dropdown(EMOTIONS, label="Emotion")
        style = gr.Dropdown(STYLES, label="Style")
        whisper = gr.Checkbox(label="Whisper?")

    with gr.Row():
        back_btn = gr.Button("⬅ Back", elem_id="back")
        update_btn = gr.Button("Update", elem_id="update")
        reject_btn = gr.Button("Reject", elem_id="reject")
        next_btn = gr.Button("Next ➡", elem_id="next")

    displays = [audio, transcript, emotion, style, whisper, info, update_btn]

    lang.change(load_lang, [lang, scope], [plist_state, idx_state, *displays])
    scope.change(load_lang, [lang, scope], [plist_state, idx_state, *displays])
    app.load(load_lang, [lang, scope], [plist_state, idx_state, *displays])

    back_btn.click(lambda p, i: go(p, i, -1), [plist_state, idx_state], [idx_state, *displays])
    next_btn.click(lambda p, i: go(p, i, 1), [plist_state, idx_state], [idx_state, *displays])
    update_btn.click(update, [plist_state, idx_state, transcript, emotion, style, whisper],
                     [idx_state, *displays])
    reject_btn.click(reject, [plist_state, idx_state], [idx_state, *displays])


if __name__ == "__main__":
    counts = {lbl: len(pending_for(code, "all")) for lbl, code in LANGS}
    print("Clips (all):", ", ".join(f"{k}={v}" for k, v in counts.items()))
    app.launch(theme=gr.themes.Soft(), css=CSS)
