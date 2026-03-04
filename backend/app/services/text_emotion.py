"""Transformer-based text emotion analysis service."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


@lru_cache(maxsize=1)
def _get_text_classifier() -> Any:
    """Load and cache the HuggingFace text-classification pipeline."""

    from transformers import pipeline  # type: ignore

    logger.info("Loading text emotion model: %s", MODEL_NAME)
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
    )


def warmup_text_model() -> None:
    """Warm model cache at startup for lower first-request latency."""

    try:
        _get_text_classifier()
    except Exception as exc:  # pragma: no cover - runtime/env dependent.
        logger.warning("Text emotion model warmup failed: %s", exc)


def analyze_text_emotion(text: str) -> dict[str, Any]:
    """Predict text emotion in a standardized ``emotion``/``confidence`` format."""

    if not text or not text.strip():
        return {
            "emotion": "neutral",
            "confidence": 0.0,
        }

    try:
        classifier = _get_text_classifier()
        result = classifier(text.strip())
        print("Text model raw output:", result)

        if not isinstance(result, list) or not result:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
            }

        top_result = result[0]

        if not isinstance(top_result, dict) or "label" not in top_result or "score" not in top_result:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
            }

        emotion = str(top_result["label"]).lower()
        confidence = float(top_result["score"])
        print("Text emotion detected:", emotion, confidence)
        return {
            "emotion": emotion,
            "confidence": confidence,
        }
    except Exception as exc:  # pragma: no cover - runtime/env dependent.
        logger.exception("Text model inference failed: %s", exc)
        return {
            "emotion": "neutral",
            "confidence": 0.0,
        }
