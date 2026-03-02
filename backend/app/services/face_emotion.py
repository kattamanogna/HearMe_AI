"""Facial emotion analysis service placeholders."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def analyze_face_emotion(frame_source: str) -> dict[str, Any]:
    """Analyze emotion from a face frame/image source.

    Placeholder implementation: return a static neutral response.
    Replace this with face detection and expression classification.
    """
    logger.info("Face emotion analysis requested for %s", frame_source)
    return {
        "modality": "face",
        "input": frame_source,
        "emotion": "neutral",
        "confidence": 0.0,
    }
