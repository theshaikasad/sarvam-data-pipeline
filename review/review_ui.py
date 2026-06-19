"""Gradio review UI — BIG and bilingual for a non-technical reviewer, with a progress tracker.

Listen -> fix the transcript -> tap how it feels -> Save. Large text, Telugu+English labels,
and emoji choices so emotion/style are judged by SOUND, not by reading an English word.

DONE vs LEFT is tracked by `human_verified` in data/manifest.jsonl (the single source of
truth). A reviewed clip stays reviewed forever: the pipeline's no-overwrite rule means
re-running it to add more clips NEVER touches human_* fields, so done never reverts and new
clips just appear as ⬜ not-reviewed. The header shows ✅ done / ⬜ left counts, and the
"Jump" panel is an online-test-style palette of every clip (✅/⬜/❌) you can click to jump to.
🔄 Refresh re-reads the manifest so clips processed while this app is open show up live.

Every Save/Reject writes only human_* fields (+ stage/rejected_reason) and NEVER touches
asr_* or llm_*. Reviewer name comes from the REVIEWER env var (default "reviewer").

Run:  REVIEWER="Ammamma" python review/review_ui.py
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

# Emoji make the choices intuitive at a glance — judged by SOUND, not by reading the word.
# The stored VALUE is always the plain taxonomy word; the emoji is only shown in the UI.
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

LANGS = [("🟠 తెలుగు Telugu", "te-IN"), ("🔵 English", "en-IN")]
SCOPES = [("⭐ సమీక్ష · Review set", "review"),
          ("📝 మిగిలినవి · Left to do", "todo"), ("✅ అయినవి · Done", "done"),
          ("అన్నీ · All", "all"), ("🗑 Rejected", "rejected")]
LANG_NAME = {"te-IN": "తెలుగు Telugu", "en-IN": "English"}

# Default style HINT by source_type (see review/LABELING_GUIDE.md). Only a starting nudge —
# the per-clip delivery wins. Shown on screen next to the source so style is easy to pick.
STYLE_HINT = {
    "podcast_storytelling": "narrative",
    "audiobook": "narrative",
    "public_lecture": "oratorical",
    "government_broadcast": "oratorical",
    "podcast_independent": "conversational",
}

# Single in-memory copy of the manifest for this session, persisted on every action and
# re-readable via the 🔄 Refresh button (so newly processed clips appear without a restart).
rows = state.load(MANIFEST)


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def pending_for(lang: str, scope: str) -> list[str]:
    def keep(r) -> bool:
        if r.get("language") != lang:
            return False
        if scope == "review":  # the small curated gold queue (gold_sample), not yet verified
            return (bool(r.get("gold_candidate")) and not r.get("human_verified")
                    and r.get("stage") != "rejected")
        if scope == "todo":
            return not r.get("human_verified") and r.get("stage") != "rejected"
        if scope == "done":
            return bool(r.get("human_verified"))
        if scope == "rejected":
            return r.get("stage") == "rejected"
        return r.get("stage") != "rejected"  # "all"
    return [cid for cid, r in sorted(rows.items()) if keep(r)]


def lang_stats(lang: str) -> tuple[int, int, int]:
    """(done, left, total) over all non-rejected clips for this language."""
    clips = [r for r in rows.values()
             if r.get("language") == lang and r.get("stage") != "rejected"]
    done = sum(1 for r in clips if r.get("human_verified"))
    return done, len(clips) - done, len(clips)


def gold_stats(lang: str) -> tuple[int, int, int]:
    """(verified, left, total) over the curated GOLD set (gold_candidate) for this language.

    This is the small set that actually needs reviewing for WER/CER — distinct from the
    whole-language counts above, so the bar can't make 9 gold clips look like 90.
    """
    gold = [r for r in rows.values()
            if r.get("language") == lang and r.get("gold_candidate")
            and r.get("stage") != "rejected"]
    verified = sum(1 for r in gold if r.get("human_verified"))
    return verified, len(gold) - verified, len(gold)


def palette_choices(plist: list[str]) -> list[tuple[str, int]]:
    """Online-test-style chips: each clip's number + status, value = its position."""
    out = []
    for i, cid in enumerate(plist):
        r = rows[cid]
        mark = "❌" if r.get("stage") == "rejected" else ("✅" if r.get("human_verified")
                                                          else "⬜")
        out.append((f"{i + 1} {mark}", i))
    return out


def render(plist: list[str], idx: int, lang: str):
    done, left, total_lang = lang_stats(lang)
    g_done, g_left, g_total = gold_stats(lang)
    bar = (f"### ✅ {done} అయినవి (done)  ·  ⬜ {left} మిగిలినవి (left)  ·  "
           f"{total_lang} మొత్తం — {LANG_NAME.get(lang, lang)}\n"
           f"#### ⭐ Gold set: {g_done}/{g_total} verified  ·  {g_left} left to review "
           f"(this is all you need for WER/CER)")
    if not plist:
        return (None, "", EMOTIONS[0], STYLES[0], False,
                bar + "\n\n*ఈ జాబితా ఖాళీ · nothing here*",
                gr.update(choices=[], value=None), gr.update(value="✅ సరి · Save"))
    idx = max(0, min(idx, len(plist) - 1))
    cid = plist[idx]
    r = rows[cid]
    verified = bool(r.get("human_verified"))
    status = "✅ సరిచూసారు · VERIFIED" if verified else "⬜ ఇంకా చూడలేదు · not reviewed"
    # A gentle nudge when the text-LLM and audio-model emotions disagree (s6/s4b) — those
    # are exactly the clips worth a careful listen.
    warn = ""
    if r.get("audio_emotion") and r.get("emotion_agree") == "disagree":
        warn = "  ·  ⚠️ చెవి పెట్టి వినండి · listen carefully"
    # Source + a suggested default style (by source_type) — a nudge so style is easy to pick.
    src = r.get("source_channel") or "?"
    stype = r.get("source_type") or ""
    hint = STYLE_HINT.get(stype)
    src_line = f"\n\n**{src}**" + (f"  ·  {stype}" if stype else "")
    if hint:
        src_line += f"  ·  usually **{STYLE_EMOJI.get(hint, '')} {hint}** style"
    info = f"{bar}\n\n## క్లిప్ {idx + 1} / {len(plist)}  ·  {status}{warn}{src_line}"
    return (
        os.path.join(CLIPS_DIR, f"{cid}.wav"),
        r.get("human_transcript") or r.get("asr_transcript", ""),
        r.get("human_emotion") or r.get("llm_emotion") or EMOTIONS[0],
        r.get("human_style") or r.get("llm_style") or STYLES[0],
        bool(r.get("human_whisper") if "human_whisper" in r else r.get("whisper", False)),
        info,
        gr.update(choices=palette_choices(plist), value=idx),
        gr.update(value="✅ మళ్ళీ సరి · Save again" if verified else "✅ సరి · Save & Next"),
    )


def load_lang(lang: str, scope: str):
    plist = pending_for(lang, scope)
    return (plist, 0, *render(plist, 0, lang))


def refresh(lang: str, scope: str):
    """Re-read the manifest from disk so clips processed since launch appear live."""
    global rows
    rows = state.load(MANIFEST)
    return load_lang(lang, scope)


def go(plist, idx, lang, step):
    nidx = max(0, min(idx + step, len(plist) - 1)) if plist else 0
    return (nidx, *render(plist, nidx, lang))


def jump(plist, pos, lang):
    nidx = int(pos) if pos is not None else 0
    return (nidx, *render(plist, nidx, lang))


def save(plist, idx, lang, transcript, emotion, style, whisper):
    """Write edits to the current clip's human_* fields, then advance to the next."""
    if plist and 0 <= idx < len(plist):
        cid = plist[idx]
        fields = dict(human_transcript=transcript, human_emotion=emotion,
                      human_whisper=bool(whisper), human_verified=True,
                      reviewer=REVIEWER, reviewed_at=now())
        # Only record human_style when it differs from the machine guess (export falls back
        # to llm_style otherwise; keeps the manifest honest about what the human changed).
        if style != rows[cid].get("llm_style"):
            fields["human_style"] = style
        state.update(rows, cid, **fields)
        state.save(rows, MANIFEST)
    nxt = min(idx + 1, len(plist) - 1) if plist else 0
    return (nxt, *render(plist, nxt, lang))


def reject(plist, idx, lang):
    if plist and 0 <= idx < len(plist):
        state.update(rows, plist[idx], stage="rejected", rejected_reason="manual_review")
        state.save(rows, MANIFEST)
    nxt = min(idx + 1, len(plist) - 1) if plist else 0
    return (nxt, *render(plist, nxt, lang))


# Big, readable text everywhere. Scoped to text elements only — a wildcard on the audio
# player collapses its play-button icon — then the player's control icons are sized back up.
CSS = """
.gradio-container {max-width: 940px !important; margin: auto !important;}
#info {text-align: center;}
#info h2 {font-size: 34px !important; margin: 4px 0;}
#info h3 {font-size: 24px !important; color: #1a7f37 !important; margin: 2px 0;}
#instructions {font-size: 24px !important; text-align: center; color: #333;}
#transcript textarea {font-size: 36px !important; line-height: 1.6 !important;
    padding: 16px !important;}
#transcript label, label span, .gradio-container label {font-size: 22px !important;}
#emotion label, #style label {font-size: 24px !important; padding: 10px 14px !important;}
#whisper label {font-size: 24px !important;}
/* size ONLY the action buttons — never a global `button` rule, which mangles the audio
   player's own play/skip controls. */
#back, #save, #reject, #next, #refresh {font-size: 26px !important; padding: 20px !important;
    font-weight: 700 !important;}
#save {background: #1a7f37 !important; color: #fff !important;}
#reject {background: #c0392b !important; color: #fff !important;}
/* online-test palette: numbered status chips that wrap into a grid */
#palette label {font-size: 18px !important; padding: 8px 10px !important; min-width: 54px;
    text-align: center; justify-content: center;}
/* keep the audio player's play / skip icons clearly visible */
#audio svg {width: 28px !important; height: 28px !important;}
#audio button {opacity: 1 !important;}
"""

with gr.Blocks(title="Review", theme=gr.themes.Soft(), css=CSS) as app:
    plist_state = gr.State([])
    idx_state = gr.State(0)

    gr.Markdown("# 🎧 వినండి & సరిచూడండి  ·  Listen & Check")
    gr.Markdown(
        "**1.** ▶️ వినండి (listen).  "
        "**2.** 📝 తప్పుంటే అక్షరాలు సరిచేయండి (fix wrong words).  "
        "**3.** 😊 ఏ భావం? 🎭 ఎలా? బొమ్మ నొక్కండి (tap how it feels).  "
        "**4.** ✅ **సరి (Save)** · చెడ్డదైతే ❌ **Reject**.",
        elem_id="instructions",
    )

    with gr.Row():
        lang = gr.Radio(LANGS, value="te-IN", label="భాష · Language")
        scope = gr.Radio(SCOPES, value="review", label="చూపించు · Show")

    info = gr.Markdown(elem_id="info")

    with gr.Accordion("📋 జాబితా · Jump to a clip (✅ done · ⬜ left)", open=False):
        palette = gr.Radio([], label="", elem_id="palette")

    audio = gr.Audio(type="filepath", label="▶️ Clip", interactive=False, autoplay=True,
                     elem_id="audio")
    transcript = gr.Textbox(label="📝 ఇది సరిగ్గా ఉందా? · Is the text right? (edit if wrong)",
                            lines=4, elem_id="transcript")
    emotion = gr.Radio(EMOTION_CHOICES, label="😊 ఏ భావం? · How does it FEEL?",
                       elem_id="emotion")
    style = gr.Radio(STYLE_CHOICES, label="🎭 ఎలా మాట్లాడుతున్నారు? · Speaking style",
                     elem_id="style")
    whisper = gr.Checkbox(label="🤫 గుసగుసనా? · Whispering?", elem_id="whisper")

    with gr.Accordion("❓ ఏది ఎంచుకోవాలి? · Not sure how to label?", open=False):
        gr.Markdown(
            "**Leave the AI's guess unless it's clearly wrong.** Label how it *sounds*, "
            "not the topic.\n\n"
            "**Style — what is the speaker doing?**\n"
            "- 📖 narrative = telling a *story* / reading prose\n"
            "- 🗣️ oratorical = addressing a *crowd* (speech, lecture, TED, motivational)\n"
            "- 💬 conversational = casual chat with one person\n"
            "- 🎓 instructional = teaching steps / how-to · 🙏 devotional = prayer/spiritual · "
            "🎭 dramatic = acting/performed\n\n"
            "**Emotion — what do you HEAR?** `neutral` 😐 is fine and common. Upgrade only on "
            "a clear cue: 😊 bright→happy · 🤩 hyped→excited · 😢 heavy/slow→sad · 😠 harsh→angry "
            "· 😌 soft/soothing→calm · 🧐 grave/firm→serious. Torn? pick **neutral**.\n\n"
            "⚠️ flag = the two AIs disagreed — replay and trust your ears. "
            "Full guide: `review/LABELING_GUIDE.md`."
        )

    with gr.Row():
        back_btn = gr.Button("⬅️ వెనుక · Back", elem_id="back")
        save_btn = gr.Button("✅ సరి · Save & Next", elem_id="save")
        reject_btn = gr.Button("❌ వద్దు · Reject", elem_id="reject")
        next_btn = gr.Button("ముందు ➡️ · Next", elem_id="next")
    refresh_btn = gr.Button("🔄 కొత్త క్లిప్‌లు · Refresh (load newly processed clips)",
                            elem_id="refresh")

    displays = [audio, transcript, emotion, style, whisper, info, palette, save_btn]

    lang.change(load_lang, [lang, scope], [plist_state, idx_state, *displays])
    scope.change(load_lang, [lang, scope], [plist_state, idx_state, *displays])
    app.load(load_lang, [lang, scope], [plist_state, idx_state, *displays])
    refresh_btn.click(refresh, [lang, scope], [plist_state, idx_state, *displays])

    back_btn.click(lambda p, i, l: go(p, i, l, -1), [plist_state, idx_state, lang],
                   [idx_state, *displays])
    next_btn.click(lambda p, i, l: go(p, i, l, 1), [plist_state, idx_state, lang],
                   [idx_state, *displays])
    palette.change(jump, [plist_state, palette, lang], [idx_state, *displays])
    save_btn.click(save, [plist_state, idx_state, lang, transcript, emotion, style, whisper],
                   [idx_state, *displays])
    reject_btn.click(reject, [plist_state, idx_state, lang], [idx_state, *displays])


if __name__ == "__main__":
    counts = {lbl: len(pending_for(code, "todo")) for lbl, code in LANGS}
    print("Left to do:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    app.launch()
