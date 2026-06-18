import json
import os
import subprocess

import librosa
import numpy as np
import requests
from openai import OpenAI

SARVAM_KEY = os.environ.get("SARVAM_KEY", "sk_v4o45pgb_UK9yjlnaKKeUBkP4GKy1fNH6")
VIDEO_URL = "https://youtu.be/pSCpQ4-9mh4"
CLIP_PATH = "clip_000.wav"
LANGUAGE_CODE = "te-IN"

# 1. DOWNLOAD audio as wav
subprocess.run([
    "yt-dlp", "-x", "--audio-format", "wav",
    "-o", "raw.%(ext)s", VIDEO_URL,
], check=True)

# 2. NORMALIZE to 16kHz mono (what Sarvam wants)
subprocess.run([
    "ffmpeg", "-y", "-i", "raw.wav",
    "-ar", "16000", "-ac", "1", "norm.wav",
], check=True)

# 3. CHOP: take ONE 30s segment starting at 60s (just to test the loop)
subprocess.run([
    "ffmpeg", "-y", "-i", "norm.wav",
    "-ss", "60", "-t", "30", CLIP_PATH,
], check=True)

# 4. ACOUSTIC FEATURES (for the emotion tagger later)
y, sr = librosa.load(CLIP_PATH, sr=16000)
f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
f0 = f0[~np.isnan(f0)]
feats = {
    "pitch_mean": round(float(np.mean(f0)), 1),
    "pitch_std": round(float(np.std(f0)), 1),
    "energy": round(float(np.sqrt(np.mean(y**2))), 4),
    "duration": round(len(y) / sr, 1),
}

# 5. SARVAM ASR (Saaras v3)
with open(CLIP_PATH, "rb") as audio_file:
    asr_resp = requests.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"api-subscription-key": SARVAM_KEY},
        files={"file": (CLIP_PATH, audio_file, "audio/wav")},
        data={
            "model": "saaras:v3",
            "mode": "transcribe",
            "language_code": LANGUAGE_CODE,
        },
        timeout=120,
    )

if not asr_resp.ok:
    raise RuntimeError(f"ASR failed ({asr_resp.status_code}): {asr_resp.text}")

asr = asr_resp.json()
if asr.get("error"):
    raise RuntimeError(f"ASR error: {asr['error']}")

transcript = (asr.get("transcript") or "").strip()
if not transcript:
    raise RuntimeError(f"ASR returned empty transcript: {asr_resp.text}")

print("\nTRANSCRIPT:", transcript)

# 6. SARVAM LLM emotion tag (OpenAI-compatible endpoint)
client = OpenAI(base_url="https://api.sarvam.ai/v1", api_key=SARVAM_KEY)
prompt = f"""You are an expert speech-emotion annotator for a TTS dataset.
Given a transcript and acoustic features, return ONLY a JSON object:
{{"emotion": one of [neutral, happy, sad, angry, excited, calm, formal, suspenseful, fearful], "style": one of [narrative, conversational, formal, instructional, dramatic], "confidence": 0.0-1.0, "reasoning": "<one short sentence>"}}

Transcript: {transcript}
Acoustic features: pitch_mean={feats['pitch_mean']}Hz, pitch_std={feats['pitch_std']}, energy={feats['energy']}
Return only the JSON, no markdown."""

resp = client.chat.completions.create(
    model="sarvam-30b",
    messages=[{"role": "user", "content": prompt}],
)
raw = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
tag = json.loads(raw)

# 7. WRITE one manifest row
row = {
    "clip_id": "clip_000",
    **feats,
    "asr_transcript": transcript,
    "language_code": asr.get("language_code", LANGUAGE_CODE),
    **tag,
    "source_url": VIDEO_URL,
    "annotator": "sarvam-30b",
}
with open("manifest.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\nFULL ROW:", json.dumps(row, ensure_ascii=False, indent=2))
print("\n✅ One clip through the whole pipeline. The rest is a loop.")
