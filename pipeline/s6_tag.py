"""Stage 6: derive Parler axes + LLM emotion/style tagging.

For rows at stage "transcribed":
  1. Fill speaking_rate (non-space chars / second) if missing.
  2. Derive speaking_rate_bin / pitch_bin / pitch_variation / recording_quality from the
     acoustic features (heuristic thresholds, documented below).
  3. Call sarvam-30b with a strict JSON-only prompt over transcript + acoustic features,
     returning {emotion, style, whisper, confidence, reasoning}. Parsed robustly.

Stores llm_* fields + the derived bins + machine `whisper`, advances to "tagged".
NEVER touches human_* fields. Resumable: only "transcribed" rows are processed; rows that
error stay at "transcribed" for retry.

Bin thresholds are heuristic (graded on judgment, so kept explicit and tweakable):
  speaking_rate (chars/s): <7 very_slow, <10 slow, <14 measured, <18 fast, else very_fast
  pitch_mean (Hz, gender-agnostic fallback): <140 low, <=220 moderate, else high
  pitch_std (Hz): <20 monotone, <=50 moderate, else animated
  recording_quality (from music_confidence): <0.02 clean, <0.10 slight_noise, else noisy
"""

from __future__ import annotations

import json
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"


def load_env() -> None:
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


def speaking_rate(transcript: str, duration: float) -> float | None:
    if not transcript or not duration:
        return None
    chars = len(re.sub(r"\s+", "", transcript))
    return round(chars / duration, 2)


def rate_bin(r):
    if r is None:
        return None
    return ("very_slow" if r < 7 else "slow" if r < 10 else "measured"
            if r < 14 else "fast" if r < 18 else "very_fast")


def pitch_bin(p):
    if p is None:
        return None
    return "low" if p < 140 else "moderate" if p <= 220 else "high"


def variation_bin(s):
    if s is None:
        return None
    return "monotone" if s < 20 else "moderate" if s <= 50 else "animated"


def quality_bin(music_conf):
    mc = music_conf or 0.0
    return "clean" if mc < 0.02 else "slight_noise" if mc < 0.10 else "noisy"


def parse_json(text: str) -> dict:
    """Strip code fences and extract the first {...} object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(text[start:end + 1])


PROMPT = """You are an expert audio annotator for a TTS dataset. Given a transcript and \
measured acoustic features of ONE speech clip, classify its delivery. Respond with ONLY a \
JSON object, no prose, no code fences.

Allowed emotion (pick exactly one): {emotions}
Allowed style (pick exactly one): {styles}

Transcript ({language}): {transcript}
Acoustic features: speaking_rate={rate} ({rate_bin}), pitch_mean_hz={pitch} ({pitch_bin}), \
pitch_variation={variation}, recording_quality={quality}

Return JSON: {{"emotion": <one allowed emotion>, "style": <one allowed style>, \
"whisper": <true|false, true only if the speaker is whispering>, \
"confidence": <0..1 float>, "reasoning": "<one short sentence>"}}"""


def tag_one(client, model, row, emotions, styles):
    prompt = PROMPT.format(
        emotions=", ".join(emotions), styles=", ".join(styles),
        language=row.get("language", "unknown"),
        transcript=row.get("asr_transcript", ""),
        rate=row.get("speaking_rate"), rate_bin=row.get("speaking_rate_bin"),
        pitch=row.get("pitch_mean"), pitch_bin=row.get("pitch_bin"),
        variation=row.get("pitch_variation"), quality=row.get("recording_quality"),
    )
    resp = client.chat.completions.create(
        model=model, temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    data = parse_json(resp.choices[0].message.content)
    emotion = data.get("emotion") if data.get("emotion") in emotions else "neutral"
    style = data.get("style") if data.get("style") in styles else "narrative"
    return {
        "llm_emotion": emotion,
        "llm_style": style,
        "whisper": bool(data.get("whisper", False)),
        "llm_confidence": data.get("confidence"),
        "llm_reasoning": data.get("reasoning", ""),
    }


def run(config_path: str = CONFIG_PATH) -> None:
    load_env()
    key = os.environ.get("SARVAM_KEY")
    if not key:
        sys.exit("SARVAM_KEY not set (see .env.example).")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    emotions = cfg["taxonomy"]["emotion"]
    styles = cfg["taxonomy"]["style"]
    model = cfg["llm"]["model"]
    rows = state.load(cfg["paths"]["manifest"])

    from datetime import datetime
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=cfg["llm"]["base_url"])

    ok = errors = 0
    for row in list(state.by_stage(rows, "transcribed")):
        cid = row["clip_id"]
        # 1-2) derived axes
        if row.get("speaking_rate") is None:
            row["speaking_rate"] = speaking_rate(row.get("asr_transcript", ""),
                                                 row.get("duration", 0))
        row["speaking_rate_bin"] = rate_bin(row.get("speaking_rate"))
        row["pitch_bin"] = pitch_bin(row.get("pitch_mean"))
        row["pitch_variation"] = variation_bin(row.get("pitch_std"))
        row["recording_quality"] = quality_bin(row.get("music_confidence"))
        # 3) LLM tag
        try:
            tags = tag_one(client, model, row, emotions, styles)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {cid}: {e}")
            errors += 1
            continue
        state.update(rows, cid, **tags, llm_model=model,
                     annotated_at=datetime.now().isoformat(timespec="seconds"),
                     stage="tagged")
        state.save(rows, cfg["paths"]["manifest"])
        ok += 1

    print(f"s6 done. tagged {ok}, errors {errors}.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
