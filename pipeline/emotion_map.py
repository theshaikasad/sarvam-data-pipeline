"""Shared emotion mapping between the audio model (s4b) and the LLM tagger (s6).

The audio SER model emits dimensional arousal/valence/dominance in [0,1]. The LLM emits a
categorical emotion from the taxonomy. To compare the two opinions honestly we project BOTH
into a small, robust valence/arousal QUADRANT space (a 5-way SER model cannot reproduce a
fine 10-way label, so we never compare labels directly):

    pos_high  positive valence, high arousal  -> happy / excited / playful / surprised
    neg_high  negative valence, high arousal  -> angry / fearful
    pos_low   positive valence, low arousal   -> calm
    neg_low   negative valence, low arousal   -> sad
    neutral   mid valence or mid arousal      -> neutral / serious

`dims_to_emotion` picks one representative taxonomy word per quadrant (the audio model's
coarse vote); `agreement` returns "agree" / "disagree" by comparing quadrants.
"""

from __future__ import annotations

from typing import Optional

# Map every taxonomy emotion to its valence/arousal quadrant.
EMOTION_QUADRANT = {
    "happy": "pos_high",
    "excited": "pos_high",
    "playful": "pos_high",
    "surprised": "pos_high",
    "angry": "neg_high",
    "fearful": "neg_high",
    "calm": "pos_low",
    "sad": "neg_low",
    "neutral": "neutral",
    "serious": "neutral",
}

# One representative taxonomy emotion the audio model "votes" for, per quadrant.
QUADRANT_EMOTION = {
    "pos_high": "excited",
    "neg_high": "angry",
    "pos_low": "calm",
    "neg_low": "sad",
    "neutral": "neutral",
}

DEFAULT_CUTOFFS = {
    "arousal_high": 0.55,
    "arousal_low": 0.45,
    "valence_high": 0.55,
    "valence_low": 0.45,
}


def dims_to_quadrant(arousal: Optional[float], valence: Optional[float],
                     cutoffs: dict | None = None) -> Optional[str]:
    """Project (arousal, valence) in [0,1] onto a valence/arousal quadrant."""
    if arousal is None or valence is None:
        return None
    c = {**DEFAULT_CUTOFFS, **(cutoffs or {})}
    high_a = arousal >= c["arousal_high"]
    low_a = arousal <= c["arousal_low"]
    high_v = valence >= c["valence_high"]
    low_v = valence <= c["valence_low"]
    if not (high_a or low_a) or not (high_v or low_v):
        return "neutral"  # anything mid-range on either axis reads as neutral
    if high_v and high_a:
        return "pos_high"
    if low_v and high_a:
        return "neg_high"
    if high_v and low_a:
        return "pos_low"
    return "neg_low"


def dims_to_emotion(arousal: Optional[float], valence: Optional[float],
                    cutoffs: dict | None = None) -> Optional[str]:
    """The audio model's coarse categorical vote (taxonomy word)."""
    q = dims_to_quadrant(arousal, valence, cutoffs)
    return QUADRANT_EMOTION.get(q) if q else None


def agreement(llm_emotion: Optional[str], audio_emotion: Optional[str]) -> Optional[str]:
    """'agree' / 'disagree' by comparing valence/arousal quadrants; None if unknown."""
    if not llm_emotion or not audio_emotion:
        return None
    ql = EMOTION_QUADRANT.get(llm_emotion)
    qa = EMOTION_QUADRANT.get(audio_emotion)
    if ql is None or qa is None:
        return None
    return "agree" if ql == qa else "disagree"
