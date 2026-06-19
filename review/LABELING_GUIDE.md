# Labeling guide — style & emotion

Quick rules for the human review UI (`review/review_ui.py`). Goal: **fast, consistent,
honest** labels. Consistency beats agonizing — label the same kind of clip the same way
every time.

## Rule 0 — the button starts on the AI's guess. Only change it if it's CLEARLY wrong.
Emotion/style radios are pre-selected to the machine's guess. If the guess is plausible,
just hit **Save**. Don't overthink a coin-flip — only override when it's obviously off.

## Rule 1 — label how it SOUNDS, not what it's about.
A sad *story* read in a flat voice is `neutral`, not `sad`. A cheerful voice reading bad
news is `happy`/`excited`. You're rating the **delivery**, not the topic.

## Rule 2 — judge the whole clip's DOMINANT impression.
~25s clips drift. Pick what dominates. Ignore one stray excited sentence in an otherwise
calm clip.

---

## STYLE — pick by what the speaker is *doing* (one per clip)

Walk down this list; **take the first that fits**:

1. **devotional** 🙏 — religious/spiritual: prayer, praise, chanting, sermon about god/faith.
2. **dramatic** 🎭 — theatrical performance / acting: voices for characters, exaggerated
   emotional delivery, like a stage play. (Rare — only if it really sounds *performed*.)
3. **instructional** 🎓 — teaching a how-to / steps / a lesson ("first you do this…", advice,
   explaining a concept to learn).
4. **oratorical** 🗣️ — addressing an AUDIENCE/CROWD: speech, lecture, motivational talk, TED.
   Persuasive, rhetorical, projected voice, "you all…".
5. **narrative** 📖 — telling a STORY or reading prose: events, characters, a plot, audiobook.
6. **conversational** 💬 — casual, informal, like chatting to ONE person; relaxed, spontaneous.

If torn between two, the **source** usually settles it (next table).

### Default style by source (use unless the clip clearly says otherwise)
| Source / channel | Default style |
|---|---|
| Harshaneeyam (podcast storytelling) | **narrative** |
| Audiobook channel | **narrative** |
| Garikapati (discourse — when retelling epics/stories) | **narrative** |
| BV Pattabhiram (motivational lecture) | **oratorical** |
| Garikapati (when addressing the audience / pravachanam) | **oratorical** |
| TED / TEDx India | **oratorical** |
| Solo podcast monologue (Ranveer / Raj Shamani solo) | **conversational** |
| Any clearly devotional/spiritual passage | **devotional** |

**narrative vs oratorical** (the common confusion): is the speaker telling a *story*
(events/characters) → narrative; or *addressing the audience directly* (persuading,
"you should…", rhetorical questions) → oratorical.

**narrative vs conversational**: scripted/storytelling flow → narrative; loose, casual,
unscripted chat → conversational.

---

## EMOTION — pick by vocal cues (one per clip)

`neutral` is a **legitimate, common** answer — most lecture/narration is neutral. Don't
invent emotion that isn't in the voice. But distinguish:

- **neutral** 😐 — flat, informational, no strong feeling. The default.
- **calm** 😌 — soft, slow, soothing, gentle, peaceful (more *relaxed* than plain neutral).
- **serious** 🧐 — grave, weighty, firm, no-nonsense (heavier than neutral).

Override to a stronger emotion only when you actually HEAR it:

| You hear… | Label |
|---|---|
| laughter, bright/up tone, smiling voice | **happy** 😊 |
| high energy, fast, animated, hyped | **excited** 🤩 |
| heavy, low, slow, mournful, voice cracking | **sad** 😢 |
| harsh, loud, sharp, confrontational | **angry** 😠 |
| shaky, tense, anxious, trembling | **fearful** 😨 |
| sudden gasp, "oh!", shock | **surprised** 😲 |
| teasing, joking, light and silly | **playful** 😜 |
| soft/slow/soothing | **calm** 😌 |
| grave/firm/weighty | **serious** 🧐 |

**Whisper** 🤫 is separate — tick it only if the voice is actually whispered/hushed; it can
co-occur with any emotion.

### Emotion tie-breakers
- Can't decide between neutral and a faint emotion? → **neutral** (don't over-label).
- Between two real emotions? → the **stronger/more dominant** one over the clip.
- See the **⚠️ listen carefully** flag? The text-AI and audio-AI disagreed here — replay
  it and trust your ears; this clip is worth the extra few seconds.

---

## TL;DR
1. Listen. 2. Fix wrong words. 3. Leave the AI guess unless it's clearly wrong.
4. Style = what they're *doing* (story→narrative, speech→oratorical, chat→conversational).
5. Emotion = what you *hear* (neutral is fine; only upgrade on a clear cue). 6. Save.
