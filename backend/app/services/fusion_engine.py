"""Multimodal fusion service using weighted probabilities across modalities."""

from __future__ import annotations

from typing import Any

WEIGHTS = {"text": 0.5, "face": 0.3, "audio": 0.2}


def _norm_probs(prediction: dict[str, Any] | None) -> dict[str, float]:
    if not prediction:
        return {}
    probs = prediction.get("probabilities")
    if isinstance(probs, dict) and probs:
        normalized = {str(k).lower(): max(0.0, float(v)) for k, v in probs.items()}
        total = sum(normalized.values())
        if total > 0:
            return {k: v / total for k, v in normalized.items()}
    emotion = str(prediction.get("emotion", "neutral")).lower()
    confidence = max(0.0, min(float(prediction.get("confidence", 0.0)), 1.0))
    return {emotion: confidence}


def combine_predictions(
    text_pred: dict[str, Any] | None,
    audio_pred: dict[str, Any] | None,
    face_pred: dict[str, Any] | None,
) -> dict[str, Any]:
    predictions = {"text": text_pred, "audio": audio_pred, "face": face_pred}

    fused_scores: dict[str, float] = {}
    effective_weight = 0.0

    for modality, pred in predictions.items():
        probs = _norm_probs(pred)
        if not probs:
            continue
        weight = WEIGHTS[modality]
        effective_weight += weight
        for emotion, prob in probs.items():
            fused_scores[emotion] = fused_scores.get(emotion, 0.0) + (weight * prob)

    if not fused_scores or effective_weight == 0:
        return {"emotion": "neutral", "confidence": 0.0, "probabilities": {"neutral": 1.0}}

    renormalized = {emotion: score / effective_weight for emotion, score in fused_scores.items()}
    fused_emotion, fused_confidence = max(renormalized.items(), key=lambda kv: kv[1])
    return {
        "emotion": fused_emotion,
        "confidence": round(float(fused_confidence), 4),
        "probabilities": {k: round(float(v), 6) for k, v in sorted(renormalized.items())},
    }


def fuse_emotion_signals(
    text_result: dict[str, Any] | None = None,
    audio_result: dict[str, Any] | None = None,
    face_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fused = combine_predictions(text_result, audio_result, face_result)
    return {
        "fused_emotion": fused["emotion"],
        "fused_confidence": fused["confidence"],
        "fused_probabilities": fused["probabilities"],
        "signals": {"text": text_result, "audio": audio_result, "face": face_result},
    }
