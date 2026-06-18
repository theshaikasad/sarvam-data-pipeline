"""Stage 7: compose one Parler-style natural-language description sentence.

For rows at stage "tagged", call sarvam-30b to turn the structured fields (gender if
known, language, emotion, style, speaking_rate_bin, pitch info, recording_quality) into
ONE descriptive sentence — the rich prosody signal Bulbul V3 (no emotion param) trains on.
Stores `description`, advances to "described". Resumable; errors leave the row at "tagged".
"""

from __future__ import annotations

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"

LANG_NAME = {"te-IN": "Telugu", "en-IN": "Indian English"}

PROMPT = """Write ONE natural-language sentence describing how this speech clip sounds, in \
the style of Parler-TTS voice descriptions. Use the attributes below; do not invent \
others. Output ONLY the sentence, no quotes, no preamble.

speaker_gender: {gender}
language: {language}
emotion: {emotion}
style: {style}
speaking_rate: {rate_bin}
pitch_level: {pitch_bin}
pitch_variation: {variation}
recording_quality: {quality}

Example: "A male speaker narrates a Telugu story in a slow, sorrowful tone with moderate \
pitch variation, recorded clearly with almost no background noise."""


def describe_one(client, model, row):
    prompt = PROMPT.format(
        gender=row.get("gender", "a speaker of unknown gender"),
        language=LANG_NAME.get(row.get("language"), row.get("language", "unknown")),
        emotion=row.get("llm_emotion", "neutral"),
        style=row.get("llm_style", "narrative"),
        rate_bin=row.get("speaking_rate_bin", "measured"),
        pitch_bin=row.get("pitch_bin", "moderate"),
        variation=row.get("pitch_variation", "moderate"),
        quality=row.get("recording_quality", "clean"),
    )
    resp = client.chat.completions.create(
        model=model, temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip().strip('"')


def load_env() -> None:
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


def run(config_path: str = CONFIG_PATH) -> None:
    load_env()
    key = os.environ.get("SARVAM_KEY")
    if not key:
        sys.exit("SARVAM_KEY not set (see .env.example).")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = cfg["llm"]["model"]
    rows = state.load(cfg["paths"]["manifest"])

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=cfg["llm"]["base_url"])

    ok = errors = 0
    for row in list(state.by_stage(rows, "tagged")):
        cid = row["clip_id"]
        try:
            description = describe_one(client, model, row)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {cid}: {e}")
            errors += 1
            continue
        state.update(rows, cid, description=description, stage="described")
        state.save(rows, cfg["paths"]["manifest"])
        ok += 1

    print(f"s7 done. described {ok}, errors {errors}.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
