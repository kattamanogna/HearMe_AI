"""In-memory session manager for conversational context and trend analytics."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
import logging

logger = logging.getLogger(__name__)

MAX_USER_MESSAGES = 5
MAX_EMOTION_HISTORY = 30


@dataclass
class SessionState:
    """State tracked for each chat session."""

    user_messages: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_USER_MESSAGES))
    emotion_history: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_EMOTION_HISTORY))
    confidence_history: deque[float] = field(default_factory=lambda: deque(maxlen=MAX_EMOTION_HISTORY))
    interaction_log: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=MAX_EMOTION_HISTORY))
    last_template_index: dict[str, int] = field(default_factory=dict)


_sessions: defaultdict[str, SessionState] = defaultdict(SessionState)
_session_lock = Lock()


def _normalize_session_id(session_id: str) -> str:
    return session_id.strip() or "default"


def get_or_create_session(session_id: str) -> SessionState:
    normalized = _normalize_session_id(session_id)
    with _session_lock:
        return _sessions[normalized]


def store_interaction(
    session_id: str,
    *,
    user_text: str,
    emotion: str,
    confidence: float,
    route: str,
    timestamp: str,
) -> SessionState:
    """Store user message and emotion metadata for a session."""

    normalized = _normalize_session_id(session_id)
    with _session_lock:
        state = _sessions[normalized]
        state.user_messages.append(user_text)
        state.emotion_history.append(emotion)
        state.confidence_history.append(float(confidence))
        state.interaction_log.append(
            {
                "timestamp": timestamp,
                "route": route,
                "text": user_text,
                "fused_emotion": emotion,
                "confidence": f"{float(confidence):.4f}",
            }
        )

        logger.info("Session %s emotion history: %s", normalized, list(state.emotion_history))
        logger.info("Session %s confidence trend: %s", normalized, list(state.confidence_history))
        return state


def get_chat_history(session_id: str) -> list[dict[str, str]]:
    normalized = _normalize_session_id(session_id)
    with _session_lock:
        return list(_sessions[normalized].interaction_log)


def get_recent_user_messages(session_id: str) -> list[str]:
    normalized = _normalize_session_id(session_id)
    with _session_lock:
        return list(_sessions[normalized].user_messages)


def set_last_template_index(session_id: str, emotion: str, index: int) -> None:
    normalized = _normalize_session_id(session_id)
    with _session_lock:
        _sessions[normalized].last_template_index[emotion] = index


def get_last_template_index(session_id: str, emotion: str) -> int | None:
    normalized = _normalize_session_id(session_id)
    with _session_lock:
        return _sessions[normalized].last_template_index.get(emotion)


def get_session_summary(session_id: str) -> dict[str, str | float | list[str]]:
    normalized = _normalize_session_id(session_id)
    with _session_lock:
        state = _sessions[normalized]
        emotions = list(state.emotion_history)
        confidences = list(state.confidence_history)

    if not emotions:
        return {
            "emotional_trend": "stable",
            "dominant_emotion": "neutral",
            "average_confidence": 0.0,
            "recent_emotions": [],
        }

    dominant_emotion = Counter(emotions).most_common(1)[0][0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    split = max(1, len(emotions) // 2)
    first = emotions[:split]
    second = emotions[split:]

    first_top = Counter(first).most_common(1)[0][0] if first else dominant_emotion
    second_top = Counter(second).most_common(1)[0][0] if second else dominant_emotion

    if first_top == second_top:
        trend = "stable"
    else:
        trend = f"shifting_from_{first_top}_to_{second_top}"

    return {
        "emotional_trend": trend,
        "dominant_emotion": dominant_emotion,
        "average_confidence": round(avg_confidence, 4),
        "recent_emotions": emotions[-MAX_USER_MESSAGES:],
    }
