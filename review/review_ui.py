"""Gradio human-verification UI — ONE screen, big and readable, everything a human checks.

Built for a non-technical reviewer (e.g. a Telugu-speaking grandparent) AND for you. The
SAME app serves both: a one-click LANGUAGE SWITCH at the top means each person only sees
their language (grandma -> 🟠 Telugu, you -> 🔵 English).

Per clip the reviewer judges the four things only a HUMAN (with ears) can settle, and that
export actually merges (`final_* = human_* or machine_*`):
  1. 📝 transcript   — read along, fix any wrong word
  2. 😊 emotion      — how does it FEEL? (big emoji buttons, so no English reading needed)
  3. 🤫 whisper      — is it whispered?
  4. 🎭 style        — how is it spoken? (emoji buttons)
The acoustic Parler axes (pitch / speaking-rate / recording-quality bins) are machine-only
by design — there is no human override for them — so they are deliberately NOT shown here.

Reads/writes data/manifest.jsonl via state.py and HONORS the no-overwrite rule: only writes
human_* fields + stage/rejected_reason, and NEVER touches asr_* or llm_*. The machine guess
is pre-selected so the job is "confirm, or change" — and when the audio model DISAGREES with
the text-LLM emotion, the clip is flagged "listen carefully" so hard cases aren't rubber-stamped.

Run:
    REVIEWER="Ammamma" python review/review_ui.py
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

# Emoji labels make the choices intuitive at a glance — judged by SOUND, not by reading an
# English word. The stored VALUE is always the plain taxonomy word; emoji is only shown.
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
DEFAULT_LANG = "te-IN"

# A single in-memory copy of the manifest for this session, persisted on every action.
rows = state.load(MANIFEST)


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# Review scope: the curated gold subset (default), every kept clip (to listen to the WHOLE
# dataset), or the rejected clips (to audit the music/noise filters by ear).
SCOPES = [("⭐ Gold sample", "gold"), ("📋 All clips", "all"), ("🗑️ Rejected (audit)", "rejected")]
DEFAULT_SCOPE = "gold"


def pending_for(lang: str, scope: str = DEFAULT_SCOPE) -> list[str]:
    """Clip_ids for the chosen language and review scope."""
    def keep(r) -> bool:
        if r.get("language") != lang:
            return False
        if scope == "gold":      # the normal review queue: stratified, not-yet-verified
            return (bool(r.get("gold_candidate")) and not r.get("human_verified")
                    and r.get("stage") != "rejected")
        if scope == "rejected":  # audit what the pipeline dropped
            return r.get("stage") == "rejected"
        return r.get("stage") != "rejected"  # "all": every kept clip
    return [cid for cid, r in sorted(rows.items()) if keep(r)]


def render(plist: list[str], idx: int):
    """Return component updates for the clip at plist[idx] (or a done message)."""
    total = len(plist)
    if total == 0:
        return (None, "", EMOTIONS[0], False, STYLES[0],
                "## 🎉 No clips to review in this language yet.")
    if idx >= total:
        return (None, "", EMOTIONS[0], False, STYLES[0],
                f"## 🎉 All {total} clips done — thank you! 🙏")
    cid = plist[idx]
    r = rows[cid]
    audio = os.path.join(CLIPS_DIR, f"{cid}.wav")

    # Progress + a gentle nudge when the two automatic emotion opinions (text LLM vs audio
    # model, s6/s4b) disagree — those are exactly the clips worth a careful listen.
    info = f"## Clip {idx + 1} of {total}"
    if r.get("audio_emotion") and r.get("emotion_agree") == "disagree":
        info += "\n### ⚠️ చెవి పెట్టి వినండి · listen carefully to the feeling"
    return (
        audio,
        r.get("human_transcript") or r.get("asr_transcript", ""),
        r.get("human_emotion") or r.get("llm_emotion") or EMOTIONS[0],
        bool(r.get("human_whisper") if "human_whisper" in r else r.get("whisper", False)),
        r.get("human_style") or r.get("llm_style") or STYLES[0],
        info,
    )


def load_lang(lang: str, scope: str = DEFAULT_SCOPE):
    plist = pending_for(lang, scope)
    return (plist, 0, *render(plist, 0))


def verify(plist, idx, transcript, emotion, whisper, style, reviewer):
    if idx < len(plist):
        cid = plist[idx]
        fields = dict(
            human_transcript=transcript,
            human_emotion=emotion,
            human_whisper=bool(whisper),
            human_verified=True,
            reviewer=reviewer or os.environ.get("REVIEWER", "reviewer"),
            reviewed_at=now(),
        )
        # Only record human_style when it differs from the machine guess (keeps the manifest
        # honest about what the human actually changed; export falls back to llm_style).
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


CSS = """
.gradio-container {max-width: 1000px !important; margin: auto !important;}
#info {text-align: center;}
#info h2 {font-size: 40px !important; margin-bottom: 4px;}
#info h3 {font-size: 26px !important; color: #b54708 !important; margin-top: 6px;}
#instructions {font-size: 24px !important; text-align: center; color: #333;}
#transcript textarea {font-size: 40px !important; line-height: 1.6 !important;
    padding: 18px !important;}
#transcript label, #reviewer label,
#emotion span, #style span, #whisper span,
#emotion legend, #style legend {font-size: 24px !important; font-weight: 600;}
#emotion label, #style label {font-size: 24px !important; padding: 10px 14px !important;}
#whisper label {font-size: 24px !important;}
button {font-size: 28px !important; padding: 22px !important; font-weight: 700 !important;}
#verify {background: #1a7f37 !important; color: white !important;}
#reject {background: #c0392b !important; color: white !important;}
"""


with gr.Blocks(title="TTS gold review", theme=gr.themes.Soft(), css=CSS) as app:
    plist_state = gr.State([])
    idx_state = gr.State(0)

    gr.Markdown("# 🎧 వినండి & సరిచూడండి  (Listen & Check)")
    gr.Markdown(
        "**1.** ▶️ వినండి (listen).  "
        "**2.** 📝 అక్షరాలు సరిచూసి తప్పుంటే సరిచేయండి (fix wrong words).  "
        "**3.** 😊 ఏ భావం? · 🎭 ఎలా మాట్లాడుతున్నారు? బొమ్మ ఎంచుకోండి (tap how it feels / sounds).  "
        "**4.** ✅ **సరి (Correct)** నొక్కండి · క్లిప్ చెడ్డదైతే ❌ **Reject**.",
        elem_id="instructions",
    )

    with gr.Row():
        lang = gr.Radio(LANGS, value=DEFAULT_LANG, label="Language to review · భాష",
                        elem_id="langpick")
        scope = gr.Radio(SCOPES, value=DEFAULT_SCOPE, label="Show",
                         elem_id="scopepick")
        reviewer = gr.Textbox(label="Your name · మీ పేరు",
                              value=os.environ.get("REVIEWER", "reviewer"),
                              elem_id="reviewer")

    info = gr.Markdown(elem_id="info")
    audio = gr.Audio(type="filepath", label="▶️ Clip", autoplay=True, elem_id="audio")
    transcript = gr.Textbox(label="📝 ఇది సరిగ్గా ఉందా? (edit if wrong)", lines=4,
                            elem_id="transcript")
    emotion = gr.Radio(EMOTION_CHOICES, label="😊 ఏ భావం? · How does it FEEL?",
                       elem_id="emotion")
    whisper = gr.Checkbox(label="🤫 గుసగుసలా మాట్లాడుతున్నారా? · Whispering?",
                          elem_id="whisper")
    style = gr.Radio(STYLE_CHOICES, label="🎭 ఎలా మాట్లాడుతున్నారు? · Speaking style",
                     elem_id="style")

    with gr.Row():
        back_btn = gr.Button("⬅️ Back")
        skip_btn = gr.Button("⏭️ Skip")
        reject_btn = gr.Button("❌ Reject", elem_id="reject")
        verify_btn = gr.Button("✅ సరి · Correct & Next", variant="primary",
                               elem_id="verify")

    displays = [audio, transcript, emotion, whisper, style, info]

    # Switching language OR scope reloads the queue and resets to clip 1.
    lang.change(load_lang, [lang, scope], [plist_state, idx_state, *displays])
    scope.change(load_lang, [lang, scope], [plist_state, idx_state, *displays])
    app.load(load_lang, [lang, scope], [plist_state, idx_state, *displays])

    verify_btn.click(verify,
                     [plist_state, idx_state, transcript, emotion, whisper, style, reviewer],
                     [idx_state, *displays])
    reject_btn.click(reject, [plist_state, idx_state], [idx_state, *displays])
    back_btn.click(go_back, [plist_state, idx_state], [idx_state, *displays])
    skip_btn.click(skip, [plist_state, idx_state], [idx_state, *displays])


if __name__ == "__main__":
    counts = {label: len(pending_for(code)) for label, code in LANGS}
    if not any(counts.values()):
        print("No gold candidates pending review. Run review/gold_sample.py first.")
    else:
        print("Pending review:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    app.launch()
