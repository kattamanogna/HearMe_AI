"""Multimodal fusion service placeholders."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
