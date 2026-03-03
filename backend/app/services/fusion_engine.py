"""Multimodal fusion service using weighted probabilities across modalities."""

from __future__ import annotations

from typing import Any

WEIGHTS = {"text": 0.5, "face": 0.3, "audio": 0.2}


def _norm_probs(prediction: dict[str, Any] | None) -> dict[str, float]:
    if not prediction:
        return {}

    confidence = max(0.0, min(float(prediction.get("confidence", 0.0)), 1.0))
    if confidence <= 0.0:
        return {}

    probs = prediction.get("probabilities")
    if isinstance(probs, dict) and probs:
        normalized = {str(k).lower(): max(0.0, float(v)) for k, v in probs.items()}
        total = sum(normalized.values())
        if total > 0:
            return {k: v / total for k, v in normalized.items()}

    emotion = str(prediction.get("emotion", "neutral")).lower()
    return {emotion: confidence}


def combine_predictions(
    text_pred: dict[str, Any] | None,
    audio_pred: dict[str, Any] | None,
    face_pred: dict[str, Any] | None,
) -> dict[str, Any]:
    predictions = {"text": text_pred, "face": face_pred, "audio": audio_pred}

    active_modalities: dict[str, dict[str, float]] = {}
    for modality, pred in predictions.items():
        probs = _norm_probs(pred)
        if probs:
            active_modalities[modality] = probs

    if not active_modalities:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 0.0},
        }

    total_weight = sum(WEIGHTS[m] for m in active_modalities)
    fused_scores: dict[str, float] = {}

    for modality, probs in active_modalities.items():
        normalized_weight = WEIGHTS[modality] / total_weight
        for emotion, prob in probs.items():
            fused_scores[emotion] = fused_scores.get(emotion, 0.0) + (normalized_weight * prob)

    fused_emotion = max(fused_scores, key=fused_scores.get)
    fused_confidence = float(fused_scores[fused_emotion])

    return {
        "emotion": fused_emotion,
        "confidence": round(fused_confidence, 4),
        "probabilities": {k: round(float(v), 6) for k, v in sorted(fused_scores.items())},
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
