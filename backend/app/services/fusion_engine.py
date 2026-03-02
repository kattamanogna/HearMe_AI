"""Multimodal fusion service placeholders."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def combine_predictions(
    text_pred: dict[str, Any] | None,
    audio_pred: dict[str, Any] | None,
    face_pred: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine text/audio/face emotion predictions with weighted scoring.

    Each modality contributes to an emotion score using these fixed weights:
    text=0.5, audio=0.3, face=0.2.
    """

    weights = {"text": 0.5, "audio": 0.3, "face": 0.2}
    predictions = {
        "text": text_pred,
        "audio": audio_pred,
        "face": face_pred,
    }

    emotion_scores: dict[str, float] = {}

    # Add weighted confidence to the predicted emotion bucket for each modality.
    # Missing predictions default to neutral confidence and do not influence scores.
    for modality, prediction in predictions.items():
        if not prediction:
            continue

        emotion = str(prediction.get("emotion", "neutral"))
        confidence = float(prediction.get("confidence", 0.0))
        weighted_confidence = weights[modality] * max(0.0, min(confidence, 1.0))
        emotion_scores[emotion] = emotion_scores.get(emotion, 0.0) + weighted_confidence

    # Fallback when no modality supplied a usable prediction.
    if not emotion_scores:
        return {"emotion": "neutral", "confidence": 0.0}

    # Pick emotion with highest combined weighted score.
    combined_emotion, combined_confidence = max(
        emotion_scores.items(), key=lambda item: item[1]
    )
    return {"emotion": combined_emotion, "confidence": round(combined_confidence, 4)}


def fuse_emotion_signals(
    text_result: dict[str, Any] | None = None,
    audio_result: dict[str, Any] | None = None,
    face_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine per-modality emotion predictions into a single result.

    Placeholder implementation: just returns inputs and a neutral fused label.
    Replace this with weighted voting or learned fusion logic.
    """
    logger.info("Fusion engine invoked")
    return {
        "fused_emotion": "neutral",
        "confidence": 0.0,
        "signals": {
            "text": text_result,
            "audio": audio_result,
            "face": face_result,
        },
    }
