"""Stage 4b: audio-grounded emotion (additive — does not advance the stage).

A SECOND opinion on emotion, computed straight from the waveform — the signal the text LLM
(s6) literally cannot see: HOW it was said, not WHAT. Two backends (config audio_emotion):

  backend: "categorical"  (default) — a label-matched speech-emotion classifier whose native
      labels are mapped onto OUR taxonomy via audio_emotion.label_map. Default model is
      RAVDESS-trained (angry/calm/disgust/fearful/happy/neutral/sad/surprised -> 7/10 of our
      taxonomy incl. `calm`). For the Telugu half, the multilingual `MERaLiON/MERaLiON-SER-v1`
      (covers Tamil) is the recommended upgrade — just swap `model` + set trust_remote_code.
  backend: "dimensional" — audeering MSP-dim regresses arousal/dominance/valence in [0,1],
      mapped to a coarse emotion via valence/arousal quadrants. Language-agnostic.

Why a separate model at all (and why this is allowed): Sarvam exposes no emotion API
(Bulbul V3 infers prosody from text; Saaras is ASR-only), so the required "use Sarvam for
LLM calls" path stays the s6 text tag. This stage only ADDS an acoustic cross-check; it
removes nothing. Disagreement between the two opinions is the most valuable thing to put in
front of a human reviewer, so we store it and let gold_sample prioritise it.

Writes (machine, never edited by humans): audio_emotion (on our taxonomy), audio_emotion_raw
(model's native label), audio_emotion_score, audio_emotion_model, and — dimensional backend
only — audio_arousal/audio_valence/audio_dominance. If llm_emotion already exists it also
sets emotion_agree ("agree"/"disagree", in valence/arousal quadrant space). Run AFTER s4 and
BEFORE s6 so s6 can pick up the audio vote. Idempotent: rows with audio_emotion are skipped.
"""

from __future__ import annotations

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import emotion_map  # noqa: E402
import state  # noqa: E402

CONFIG_PATH = "config.yaml"

# Lazily built once; loading torch + the model is expensive.
_MODEL = None
_PROCESSOR = None


def _build_dimensional(model_name: str, trust_remote_code: bool):
    """Load the audeering dimensional SER model (custom regression head)."""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            hidden = hidden.mean(dim=1)
            return self.classifier(hidden)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).eval()
    return processor, model


def _build_categorical(model_name: str, trust_remote_code: bool):
    """Load any HF audio-classification SER model (reads its own id2label)."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    extractor = AutoFeatureExtractor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForAudioClassification.from_pretrained(
        model_name, trust_remote_code=trust_remote_code).eval()
    return extractor, model


def _predict_dimensional(path, sample_rate, cutoffs, label_map):
    """Return a dict of audio_* fields from the dimensional model."""
    import librosa
    import numpy as np
    import torch

    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    inputs = _PROCESSOR(y, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        # audeering output order is [arousal, dominance, valence].
        out = _MODEL(inputs["input_values"]).squeeze().cpu().numpy()
    arousal, dominance, valence = (float(np.clip(v, 0.0, 1.0)) for v in out)
    emotion = emotion_map.dims_to_emotion(arousal, valence, cutoffs)
    return dict(audio_arousal=round(arousal, 4), audio_valence=round(valence, 4),
                audio_dominance=round(dominance, 4), audio_emotion=emotion,
                audio_emotion_raw=None, audio_emotion_score=None)


def _predict_categorical(path, sample_rate, cutoffs, label_map):
    """Return a dict of audio_* fields from a categorical SER classifier."""
    import librosa
    import torch

    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    inputs = _PROCESSOR(y, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = _MODEL(**inputs).logits
    probs = logits.softmax(-1).squeeze()
    idx = int(probs.argmax())
    raw = str(_MODEL.config.id2label[idx])
    score = float(probs[idx])
    # Map the model's native label onto our taxonomy (case-insensitive); fall back to the
    # label itself if it already is a taxonomy word, else neutral.
    mapped = (label_map or {}).get(raw.lower())
    if mapped is None:
        mapped = raw.lower() if raw.lower() in emotion_map.EMOTION_QUADRANT else "neutral"
    return dict(audio_emotion=mapped, audio_emotion_raw=raw,
                audio_emotion_score=round(score, 4))


def run(config_path: str = CONFIG_PATH) -> None:
    global _MODEL, _PROCESSOR
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ae = cfg.get("audio_emotion", {})
    if not ae.get("enabled", True):
        print("s4b skipped: audio_emotion.enabled is false.")
        return

    backend = ae.get("backend", "categorical")
    model_name = ae["model"]
    trust = bool(ae.get("trust_remote_code", False))
    label_map = ae.get("label_map", {})
    sample_rate = cfg.get("sample_rate", 16000)
    cutoffs = {k: ae[k] for k in
               ("arousal_high", "arousal_low", "valence_high", "valence_low") if k in ae}
    clips_dir = cfg["paths"]["clips"]
    rows = state.load(cfg["paths"]["manifest"])

    if backend == "dimensional":
        builder, predict = _build_dimensional, _predict_dimensional
    elif backend == "categorical":
        builder, predict = _build_categorical, _predict_categorical
    else:
        sys.exit(f"unknown audio_emotion.backend: {backend!r}")

    done = errors = 0
    for row in rows.values():
        if row.get("stage") == "rejected":
            continue
        if row.get("audio_emotion") is not None:
            continue  # already computed
        cid = row["clip_id"]
        clip_path = os.path.join(clips_dir, f"{cid}.wav")
        if not os.path.exists(clip_path):
            continue
        if _MODEL is None:
            print(f"  loading {backend} model {model_name} ...")
            _PROCESSOR, _MODEL = builder(model_name, trust)
        try:
            fields = predict(clip_path, sample_rate, cutoffs, label_map)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {cid}: {e}")
            errors += 1
            continue
        fields["audio_emotion_model"] = model_name
        # If the LLM already voted, record whether the two opinions agree.
        agree = emotion_map.agreement(row.get("llm_emotion"), fields["audio_emotion"])
        if agree is not None:
            fields["emotion_agree"] = agree
        state.update(rows, cid, **fields)
        state.save(rows, cfg["paths"]["manifest"])
        done += 1

    print(f"s4b done ({backend}). audio emotion for {done} clips, errors {errors}.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
