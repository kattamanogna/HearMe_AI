"""Text preprocessing utilities for emotion + intent modeling."""

from __future__ import annotations

import re
from typing import Iterable


def clean_text(text: str) -> str:
    """Normalize text for model input.

    Next steps:
    - Add language-specific normalization.
    - Support emoji-to-token conversion.
    - Integrate profanity masking if needed.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_training_samples(records: Iterable[dict]) -> list[dict]:
    """Prepare labeled samples for model training.

    Next steps:
    - Validate schema keys: text, emotion_label, intent_label.
    - Split into train/validation/test.
    - Serialize to JSONL/Parquet for reproducibility.
    """
    samples = []
    for row in records:
        samples.append(
            {
                "text": clean_text(str(row.get("text", ""))),
                "emotion_label": row.get("emotion_label", "neutral"),
                "intent_label": row.get("intent_label", "general_support"),
            }
        )
    return samples
