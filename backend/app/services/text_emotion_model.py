"""Simple text emotion model used by API inference routes."""

from collections.abc import Iterable


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def predict_text_emotion(text: str) -> dict[str, str | float]:
    """Predict a text emotion label and confidence score."""
    normalized = text.lower()

    if _contains_any(normalized, ("happy", "great", "good", "excited", "love")):
        return {"emotion": "joy", "confidence": 0.91}

    if _contains_any(normalized, ("sad", "down", "depressed", "unhappy", "cry")):
        return {"emotion": "sadness", "confidence": 0.89}

    if _contains_any(normalized, ("angry", "mad", "furious", "annoyed", "hate")):
        return {"emotion": "anger", "confidence": 0.9}

    if _contains_any(normalized, ("anxious", "afraid", "scared", "worried", "nervous")):
        return {"emotion": "fear", "confidence": 0.87}

    return {"emotion": "neutral", "confidence": 0.62}
"""Real text emotion detection service using a Hugging Face transformer model.

Usage:
    from app.services.text_emotion_model import predict_text_emotion

    result = predict_text_emotion("I feel amazing today!")
    # {"emotion": "joy", "confidence": 0.98}

The model/tokenizer are cached, so they are loaded into memory only once
per process to keep inference fast for repeated calls.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


@lru_cache(maxsize=1)
def load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load and cache tokenizer + classifier.

    Returns:
        tuple: (tokenizer, model) ready for inference.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict_text_emotion(text: str) -> dict[str, float | str]:
    """Predict the dominant emotion for the provided text.

    Args:
        text: Raw text to analyze.

    Returns:
        A dictionary in the format:
            {"emotion": <predicted_label>, "confidence": <probability_float>}
    """

    if not text or not text.strip():
        return {"emotion": "neutral", "confidence": 0.0}

    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    top_score, top_index = torch.max(probabilities, dim=0)
    label = model.config.id2label[int(top_index)]

    return {
        "emotion": label,
        "confidence": float(top_score.item()),
    }
