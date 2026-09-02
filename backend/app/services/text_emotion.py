"""Transformer-based text emotion analysis and message-understanding service."""

from __future__ import annotations

from functools import lru_cache
import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

_EMOTION_ALIASES = {
    "sad": "sadness",
    "angry": "anger",
    "happy": "joy",
    "fearful": "fear",
}

_EMOTION_KEYWORDS = {
    "sadness": ("sad", "cry", "lonely", "alone", "hurt", "depressed", "upset", "grief"),
    "frustration": ("frustrated", "overwhelmed", "nobody understands", "stuck", "annoyed"),
    "anger": ("angry", "mad", "furious", "hate", "irritated"),
    "fear": ("afraid", "scared", "anxious", "worried", "panic", "terrified"),
    "joy": ("happy", "glad", "excited", "grateful", "proud", "relieved"),
    "neutral": (),
}

_TOPIC_KEYWORDS = {
    "loneliness": ("lonely", "alone", "nobody understands", "isolated"),
    "communication": ("understands", "listen", "heard", "talk", "message"),
    "work": ("work", "job", "boss", "deadline", "meeting"),
    "school": ("school", "class", "exam", "homework", "teacher"),
    "relationships": ("friend", "partner", "boyfriend", "girlfriend", "family", "mom", "dad"),
    "mental health": ("anxious", "depressed", "panic", "therapy", "stress", "overwhelmed"),
    "grief": ("loss", "died", "death", "grief", "miss them"),
}

_POSITIVE_MARKERS = ("i feel good", "i am happy", "i'm happy", "grateful", "proud", "excited", "relieved", "thank")
_NEGATIVE_MARKERS = (
    "i feel bad",
    "i feel awful",
    "i feel overwhelmed",
    "i am sad",
    "i'm sad",
    "nobody",
    "can't",
    "cry",
    "hate",
    "alone",
    "lonely",
)
_ADVICE_MARKERS = ("what should i", "how do i", "how can i", "advice", "help me", "should i", "any tips")
_LISTEN_MARKERS = ("just listen", "only listen", "vent", "no advice", "don't need advice", "do not need advice")
_URGENT_MARKERS = ("right now", "urgent", "emergency", "asap", "immediately", "can't take it", "hurt myself", "suicide")


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
    """Extract structured emotional and conversational understanding from text."""

    cleaned = text.strip() if text else ""
    if not cleaned:
        return _empty_analysis()

    emotion_scores = _predict_emotion_scores(cleaned)
    primary, secondary, confidence = _rank_emotions(emotion_scores, cleaned)

    return {
        "emotion": primary,
        "confidence": confidence,
        "primary_emotion": primary,
        "secondary_emotion": secondary,
        "stress": _detect_stress(cleaned),
        "urgency": _detect_urgency(cleaned),
        "intent": _detect_intent(cleaned),
        "topics": _detect_topics(cleaned),
        "people": _detect_people(cleaned),
        "positive_statements": _extract_statements(cleaned, _POSITIVE_MARKERS),
        "negative_statements": _extract_statements(cleaned, _NEGATIVE_MARKERS),
        "questions": _extract_questions(cleaned),
        "needs_advice": _detect_needs_advice(cleaned),
        "wants_listening_only": _detect_listening_only(cleaned),
        "probabilities": emotion_scores,
    }


def _empty_analysis() -> dict[str, Any]:
    return {
        "emotion": "neutral",
        "confidence": 0.0,
        "primary_emotion": "neutral",
        "secondary_emotion": "neutral",
        "stress": "low",
        "urgency": "low",
        "intent": "unknown",
        "topics": [],
        "people": [],
        "positive_statements": [],
        "negative_statements": [],
        "questions": [],
        "needs_advice": False,
        "wants_listening_only": False,
        "probabilities": {},
    }


def _predict_emotion_scores(text: str) -> dict[str, float]:
    """Return a numeric probability mapping from the model's raw response."""

    try:
        classifier = _get_text_classifier()
        result = classifier(text.strip())

        if not isinstance(result, list) or not result:
            logger.warning("Text emotion model returned an empty or invalid response: %r", result)
            return {"neutral": 0.0}

        rows = result[0] if isinstance(result[0], list) else result
        if not isinstance(rows, list):
            logger.warning("Text emotion model returned non-list score rows: %r", rows)
            return {"neutral": 0.0}

        scores: dict[str, float] = {}
        for item in rows:
            if not isinstance(item, dict):
                logger.warning("Ignoring non-dictionary text emotion score: %r", item)
                continue

            label = item.get("label")
            score = _coerce_score(item.get("score"))
            if not isinstance(label, str) or not label.strip() or score is None:
                logger.warning("Ignoring invalid text emotion score item: %r", item)
                continue

            scores[_normalize_emotion(label)] = score

        if not scores:
            logger.warning("Text emotion model returned no usable numeric scores")
            return {"neutral": 0.0}

        return scores
    except Exception as exc:  # pragma: no cover - runtime/env dependent.
        logger.exception("Text model inference failed: %s", exc)
        return {"neutral": 0.0}


def _rank_emotions(scores: dict[str, float], text: str) -> tuple[str, str, float]:
    """Rank valid numeric emotion scores, falling back safely to neutral."""

    if not isinstance(scores, dict):
        logger.warning("Cannot rank non-dictionary text emotion scores: %r", scores)
        scores = {}

    valid_scores = {
        str(emotion): score
        for emotion, value in scores.items()
        if (score := _coerce_score(value)) is not None
    }
    ranked = sorted(valid_scores.items(), key=lambda item: item[1], reverse=True)
    primary = ranked[0][0] if ranked else "neutral"
    confidence = float(ranked[0][1]) if ranked else 0.0
    secondary = ranked[1][0] if len(ranked) > 1 else "neutral"
    lower = text.lower()
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        if emotion != primary and any(keyword in lower for keyword in keywords):
            secondary = emotion
            break
    return primary, secondary, confidence


def _coerce_score(value: Any) -> float | None:
    """Convert a model score to a finite float, rejecting invalid values."""

    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _normalize_emotion(label: str) -> str:
    value = label.strip().lower()
    return _EMOTION_ALIASES.get(value, value)


def _detect_stress(text: str) -> str:
    lower = text.lower()
    hits = sum(marker in lower for marker in ("overwhelmed", "panic", "can't", "cry", "stress", "urgent"))
    if hits >= 2 or "overwhelmed" in lower:
        return "high"
    if hits == 1 or any(word in lower for word in ("worried", "anxious", "upset")):
        return "medium"
    return "low"


def _detect_urgency(text: str) -> str:
    lower = text.lower()
    if any(marker in lower for marker in _URGENT_MARKERS):
        return "high"
    if "soon" in lower or "today" in lower:
        return "medium"
    return "low"


def _detect_intent(text: str) -> str:
    lower = text.lower()
    if _detect_listening_only(text):
        return "vent or be heard"
    if _detect_needs_advice(text):
        return "seek advice"
    if any(marker in lower for marker in ("feel", "cry", "overwhelmed", "sad", "lonely", "anxious")):
        return "seek emotional support"
    return "share information"


def _detect_topics(text: str) -> list[str]:
    lower = text.lower()
    return [topic for topic, keywords in _TOPIC_KEYWORDS.items() if any(keyword in lower for keyword in keywords)]


def _detect_people(text: str) -> list[str]:
    patterns = (r"\b(?:mom|mother|dad|father|friend|partner|boss|teacher|doctor|therapist)\b", r"\b[A-Z][a-z]+\b")
    found: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if match not in found and match.lower() not in {"i"}:
                found.append(match)
    return found


def _extract_statements(text: str, markers: tuple[str, ...]) -> list[str]:
    sentences = _sentences(text)
    return [sentence for sentence in sentences if any(marker in sentence.lower() for marker in markers)]


def _extract_questions(text: str) -> list[str]:
    return [sentence for sentence in _sentences(text) if sentence.endswith("?")]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _detect_needs_advice(text: str) -> bool:
    lower = text.lower()
    if _detect_listening_only(text):
        return False
    return any(marker in lower for marker in _ADVICE_MARKERS) or "?" in text


def _detect_listening_only(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _LISTEN_MARKERS)
