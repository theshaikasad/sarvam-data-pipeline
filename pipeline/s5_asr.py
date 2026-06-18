"""Stage 5: ASR via Saaras v3.

For rows at stage "music_checked", POST the clip to /speech-to-text with
model="saaras:v3", mode="transcribe", and the explicit language_code from the row
(higher accuracy than auto-detect). Stores asr_* fields and advances to "transcribed".
Empty transcripts are rejected (rejected_reason="empty_asr"). HTTP/network errors leave
the row at "music_checked" for a later retry (resumable). Polite rate-limiting between
calls; 429/503 backoff.
"""

from __future__ import annotations

import os
import sys
import time

import requests
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import state  # noqa: E402

CONFIG_PATH = "config.yaml"
REQUEST_PAUSE_S = 0.5  # be polite between calls


def load_env() -> None:
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


def transcribe(clip_path, language_code, endpoint, model, mode, key, retries=3):
    """Return the parsed JSON response, retrying on 429/503."""
    for attempt in range(retries):
        with open(clip_path, "rb") as fh:
            resp = requests.post(
                endpoint,
                headers={"api-subscription-key": key},
                files={"file": (os.path.basename(clip_path), fh, "audio/wav")},
                data={"model": model, "mode": mode, "language_code": language_code},
                timeout=120,
            )
        if resp.status_code in (429, 503):
            wait = 2 ** attempt
            print(f"    {resp.status_code} — backing off {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()


def run(config_path: str = CONFIG_PATH) -> None:
    load_env()
    key = os.environ.get("SARVAM_KEY")
    if not key:
        sys.exit("SARVAM_KEY not set (see .env.example).")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clips_dir = cfg["paths"]["clips"]
    endpoint = cfg["asr"]["endpoint"]
    model = cfg["asr"]["model"]
    mode = cfg["asr"]["mode"]
    rows = state.load(cfg["paths"]["manifest"])

    pending = list(state.by_stage(rows, "music_checked"))
    ok = empty = errors = 0
    for row in pending:
        cid = row["clip_id"]
        clip_path = os.path.join(clips_dir, f"{cid}.wav")
        if not os.path.exists(clip_path):
            print(f"  [warn] missing clip: {clip_path}")
            continue
        try:
            data = transcribe(clip_path, row.get("language", "unknown"),
                              endpoint, model, mode, key)
        except Exception as e:  # noqa: BLE001 - keep batch alive, leave row for retry
            print(f"  [error] {cid}: {e}")
            errors += 1
            continue

        transcript = (data.get("transcript") or "").strip()
        if not transcript:
            state.update(rows, cid, asr_transcript="", asr_model=model,
                         stage="rejected", rejected_reason="empty_asr")
            empty += 1
        else:
            state.update(rows, cid,
                         asr_transcript=transcript,
                         asr_model=model,
                         asr_language_detected=data.get("language_code"),
                         asr_confidence=data.get("confidence"),
                         stage="transcribed")
            ok += 1
        state.save(rows, cfg["paths"]["manifest"])  # checkpoint after each call
        time.sleep(REQUEST_PAUSE_S)

    print(f"s5 done. transcribed {ok}, empty/rejected {empty}, errors {errors} "
          f"(of {len(pending)} pending).")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
