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
        return_all_scores=True,
    )


def warmup_text_model() -> None:
    """Warm model cache at startup for lower first-request latency."""

    try:
        _get_text_classifier()
    except Exception as exc:  # pragma: no cover - runtime/env dependent.
        logger.warning("Text emotion model warmup failed: %s", exc)


def analyze_text_emotion(text: str) -> dict[str, Any]:
    """Predict text emotion with top class, confidence, and full probabilities."""

    if not text or not text.strip():
        return {
            "modality": "text",
            "input": text,
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 0.0},
        }

    try:
        classifier = _get_text_classifier()
        results = classifier(text.strip())[0]
        probabilities: dict[str, float] = {}
        for item in results:
            if isinstance(item, dict) and "label" in item and "score" in item:
                probabilities[str(item["label"]).lower()] = float(item["score"])

        if not probabilities:
            raise ValueError("No valid emotion scores returned from text model")

        top_emotion = max(probabilities, key=probabilities.get)
        top_confidence = float(probabilities[top_emotion])
        return {
            "modality": "text",
            "input": text,
            "emotion": top_emotion,
            "confidence": top_confidence,
            "probabilities": probabilities,
        }
    except Exception as exc:  # pragma: no cover - runtime/env dependent.
        logger.exception("Text model inference failed: %s", exc)
        return {
            "modality": "text",
            "input": text,
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 0.0},
        }
