"""Gradio human-verification UI — built to be READABLE and EASY.

TWO MODES (set with the REVIEW_MODE env var):

  * simple  (DEFAULT) — for a non-technical reviewer (e.g. a Telugu-speaking grandparent).
    HUGE transcript, autoplaying audio, and just two big buttons: ✅ Correct / ❌ Reject.
    Her ONLY job is the transcript: accept it, or fix a wrong word and accept. Emotion /
    style / description are hidden and left as the machine's guess (refine later in full).

  * full — adds emotion / style / whisper controls for a technical reviewer who also wants
    to correct the prosody labels.

Both iterate over rows marked gold_candidate (by review/gold_sample.py) that aren't yet
human_verified, with a one-click LANGUAGE SWITCH so each reviewer only sees their language.
Reads/writes data/manifest.jsonl via state.py and HONORS the no-overwrite rule: only writes
human_* fields + stage/rejected_reason, and NEVER touches asr_* or llm_*.

Run:
    REVIEWER="Ammamma" python review/review_ui.py                  # simple mode (default)
    REVIEW_MODE=full REVIEWER="Asad" python review/review_ui.py    # full mode
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

SIMPLE = os.environ.get("REVIEW_MODE", "simple").lower() != "full"

# Emoji labels make the choices intuitive at a glance. The stored VALUE is always the
# plain taxonomy word; the emoji is only shown in the UI.
EMOTION_EMOJI = {
    "neutral": "😐", "happy": "😊", "sad": "😢", "angry": "😠", "excited": "🤩",
    "calm": "😌", "fearful": "😨", "surprised": "😲", "serious": "🧐", "playful": "😜",
}
STYLE_EMOJI = {
    "narrative": "📖", "conversational": "💬", "oratorical": "🗣️",
    "instructional": "🎓", "devotional": "🙏", "dramatic": "🎭",
}
EMOTION_CHOICES = [(f"{EMOTION_EMOJI.get(e, '')} {e}", e) for e in EMOTIONS]
STYLE_CHOICES = [(f"{STYLE_EMOJI.get(s, '')} {s}", s) for s in STYLES]

LANGS = [("🟠 Telugu", "te-IN"), ("🔵 English", "en-IN")]
DEFAULT_LANG = "te-IN" if SIMPLE else "te-IN"

# A single in-memory copy of the manifest for this session, persisted on every action.
rows = state.load(MANIFEST)


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def pending_for(lang: str) -> list[str]:
    """Snapshot of clip_ids still needing review for the chosen language."""
    return [
        cid for cid, r in sorted(rows.items())
        if r.get("gold_candidate")
        and not r.get("human_verified")
        and r.get("stage") != "rejected"
        and r.get("language") == lang
    ]


def render(plist: list[str], idx: int):
    """Return component updates for the clip at plist[idx] (or a done message)."""
    total = len(plist)
    if total == 0:
        return (None, "", EMOTIONS[0], STYLES[0], False, "",
                "## 🎉 No clips to review in this language yet.")
    if idx >= total:
        return (None, "", EMOTIONS[0], STYLES[0], False, "",
                f"## 🎉 All {total} clips done — thank you! 🙏")
    cid = plist[idx]
    r = rows[cid]
    audio = os.path.join(CLIPS_DIR, f"{cid}.wav")
    hint = r.get("llm_reasoning", "")
    if SIMPLE:
        # Keep it calm: just progress. No AI guesses to distract the reader.
        info = f"## Clip {idx + 1} of {total}"
    else:
        # Two independent automatic opinions: the text LLM (s6) and the audio model (s4b).
        second = ""
        if r.get("audio_emotion"):
            warn = " ⚠️ they DISAGREE — listen carefully" \
                if r.get("emotion_agree") == "disagree" else ""
            second = (f"\n*Audio model heard: {EMOTION_EMOJI.get(r.get('audio_emotion'), '')} "
                      f"{r.get('audio_emotion')}*{warn}")
        info = (f"## Clip {idx + 1} of {total}  ·  {r.get('source_channel', '?')}\n"
                f"*AI guessed: {EMOTION_EMOJI.get(r.get('llm_emotion'), '')} "
                f"{r.get('llm_emotion', '?')} · {r.get('llm_style', '?')}*" + second)
    return (
        audio,
        r.get("human_transcript") or r.get("asr_transcript", ""),
        r.get("human_emotion") or r.get("llm_emotion") or EMOTIONS[0],
        r.get("human_style") or r.get("llm_style") or STYLES[0],
        bool(r.get("human_whisper") if "human_whisper" in r else r.get("whisper", False)),
        hint,
        info,
    )


def load_lang(lang: str):
    plist = pending_for(lang)
    return (plist, 0, *render(plist, 0))


def verify(plist, idx, transcript, emotion, style, whisper, reviewer):
    if idx < len(plist):
        cid = plist[idx]
        fields = dict(
            human_transcript=transcript,
            human_verified=True,
            reviewer=reviewer or os.environ.get("REVIEWER", "reviewer"),
            reviewed_at=now(),
        )
        # In SIMPLE mode the reviewer only judges the transcript; we deliberately do NOT
        # write human_emotion/style/whisper, so the machine (LLM) labels stand at export.
        if not SIMPLE:
            fields["human_emotion"] = emotion
            fields["human_whisper"] = bool(whisper)
            # Only record human_style when it differs from the machine guess.
            if style != rows[cid].get("llm_style"):
                fields["human_style"] = style
        state.update(rows, cid, **fields)
        state.save(rows, MANIFEST)
    return (idx + 1, *render(plist, idx + 1))


def reject(plist, idx):
    if idx < len(plist):
        state.update(rows, plist[idx], stage="rejected", rejected_reason="manual_review")
        state.save(rows, MANIFEST)
    return (idx + 1, *render(plist, idx + 1))


def go_back(plist, idx):
    nidx = max(0, idx - 1)
    return (nidx, *render(plist, nidx))


def skip(plist, idx):
    return (idx + 1, *render(plist, idx + 1))


# Simple mode goes BIG. Full mode is a touch smaller so the extra controls fit.
TX_FONT = "44px" if SIMPLE else "30px"
CSS = f"""
.gradio-container {{max-width: 1000px !important; margin: auto !important;}}
#info {{text-align: center;}}
#info h2 {{font-size: 40px !important; margin-bottom: 4px;}}
#instructions {{font-size: 26px !important; text-align: center; color: #333;}}
#transcript textarea {{font-size: {TX_FONT} !important; line-height: 1.6 !important;
    padding: 18px !important;}}
#transcript label, #hint label, #reviewer label,
#emotion span, #style span, #whisper span,
#emotion legend, #style legend {{font-size: 24px !important; font-weight: 600;}}
#emotion label, #style label {{font-size: 24px !important; padding: 10px 14px !important;}}
#hint textarea {{font-size: 20px !important; color: #555;}}
button {{font-size: 28px !important; padding: 22px !important; font-weight: 700 !important;}}
#verify {{background: #1a7f37 !important; color: white !important;}}
#reject {{background: #c0392b !important; color: white !important;}}
#whisper label {{font-size: 24px !important;}}
"""


with gr.Blocks(title="TTS gold review", theme=gr.themes.Soft(), css=CSS) as app:
    plist_state = gr.State([])
    idx_state = gr.State(0)

    gr.Markdown("# 🎧 వినండి & సరిచూడండి  (Listen & Check)")
    if SIMPLE:
        gr.Markdown(
            "**1.** ▶️ వినండి (listen).  **2.** అక్షరాలు సరిగ్గా ఉంటే  ✅ **సరి (Correct)** నొక్కండి.  "
            "**3.** తప్పు ఉంటే బాక్స్‌లో సరిచేసి ✅ నొక్కండి.  **4.** క్లిప్ చెడ్డదైతే  ❌ **Reject**.",
            elem_id="instructions",
        )
    else:
        gr.Markdown(
            "**1.** Listen.  **2.** Fix any wrong words.  **3.** Tap how it *sounds*.  "
            "**4.** Press the green **Verify** button.",
            elem_id="instructions",
        )

    with gr.Row():
        lang = gr.Radio(LANGS, value=DEFAULT_LANG, label="Language to review",
                        elem_id="langpick")
        reviewer = gr.Textbox(label="Your name",
                              value=os.environ.get("REVIEWER", "reviewer"),
                              elem_id="reviewer")

    info = gr.Markdown(elem_id="info")
    audio = gr.Audio(type="filepath", label="▶️ Clip", autoplay=True, elem_id="audio")
    transcript = gr.Textbox(label="📝 ఇది సరిగ్గా ఉందా? (edit if wrong)", lines=4,
                            elem_id="transcript")

    # Prosody controls — only shown/used in full mode.
    emotion = gr.Radio(EMOTION_CHOICES, label="😊 How does it sound? (emotion)",
                       elem_id="emotion", visible=not SIMPLE)
    style = gr.Radio(STYLE_CHOICES, label="🎭 Speaking style",
                     elem_id="style", visible=not SIMPLE)
    whisper = gr.Checkbox(label="🤫 Whispering?", elem_id="whisper", visible=not SIMPLE)
    hint = gr.Textbox(label="🤖 AI reasoning (read-only hint)", interactive=False,
                      lines=2, elem_id="hint", visible=not SIMPLE)

    with gr.Row():
        back_btn = gr.Button("⬅️ Back")
        skip_btn = gr.Button("⏭️ Skip")
        reject_btn = gr.Button("❌ Reject", elem_id="reject")
        verify_label = "✅ సరి · Correct & Next" if SIMPLE else "✅ Verify & Next"
        verify_btn = gr.Button(verify_label, variant="primary", elem_id="verify")

    displays = [audio, transcript, emotion, style, whisper, hint, info]

    # Switching language reloads the queue for that language and resets to clip 1.
    lang.change(load_lang, [lang], [plist_state, idx_state, *displays])
    app.load(load_lang, [lang], [plist_state, idx_state, *displays])

    verify_btn.click(verify,
                     [plist_state, idx_state, transcript, emotion, style, whisper, reviewer],
                     [idx_state, *displays])
    reject_btn.click(reject, [plist_state, idx_state], [idx_state, *displays])
    back_btn.click(go_back, [plist_state, idx_state], [idx_state, *displays])
    skip_btn.click(skip, [plist_state, idx_state], [idx_state, *displays])


if __name__ == "__main__":
    counts = {label: len(pending_for(code)) for label, code in LANGS}
    print(f"Mode: {'SIMPLE (transcript accept/reject)' if SIMPLE else 'FULL (with prosody)'}")
    if not any(counts.values()):
        print("No gold candidates pending review. Run review/gold_sample.py first.")
    else:
        print("Pending review:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    app.launch()
