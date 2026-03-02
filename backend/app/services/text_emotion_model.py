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
