"""Helpers for generating supportive chatbot responses from emotion + text."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from app.services.session_manager import get_last_template_index, set_last_template_index

# Lightweight safety list to avoid echoing harmful language in generated replies.
_BLOCKED_TERMS = {
    "kill",
    "suicide",
    "self-harm",
    "self harm",
    "die",
    "worthless",
    "hate",
}

_CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bsuicid(?:e|al)\b",
    r"\bself[-\s]?harm\b",
    r"\bhurt myself\b",
    r"\bdon't want to live\b",
]

EMERGENCY_SUPPORT_MESSAGE = (
    "I'm really glad you reached out. If you might hurt yourself or are in immediate danger, "
    "please call emergency services right now. You can also contact the 988 Suicide & Crisis Lifeline "
    "(US/Canada) by calling or texting 988. If you're elsewhere, please contact your local crisis hotline immediately."
)

_EMOTION_TEMPLATES: dict[str, list[str]] = {
    "happy": [
        "It's great to hear this uplift in your mood—keep noticing what is helping.",
        "That sounds like a meaningful positive moment. You're building good momentum.",
    ],
    "sad": [
        "I'm really sorry this feels so heavy. We can take this one small step at a time.",
        "Thank you for sharing this. You deserve support, and we can focus on one gentle next step.",
    ],
    "angry": [
        "I hear how intense this feels. Let's pause and pick one calming action you can take right now.",
        "Your frustration makes sense. A slow breath and short reset can help you respond from a steadier place.",
    ],
    "anxious": [
        "That sounds really stressful. Try a grounding check: name 5 things you can see and 4 you can feel.",
        "You're carrying a lot right now. Let's anchor in the present with one slow breath and one manageable task.",
    ],
    "fear": [
        "That sounds really stressful. Try a grounding check: name 5 things you can see and 4 you can feel.",
        "You're carrying a lot right now. Let's anchor in the present with one slow breath and one manageable task.",
    ],
    "neutral": [
        "Thanks for sharing. I'm here with you while we work through this.",
        "I appreciate you checking in. Let's keep exploring what would help most right now.",
    ],
}


def _build_supportive_follow_up(emotion: str) -> str:
    normalized = _normalized_emotion(emotion)
    follow_ups = {
        "sad": "Would it help to share what feels hardest right now, so we can break it into one manageable step?",
        "angry": "Would you like to name what triggered this, then choose one response you can control next?",
        "anxious": "Would you like a 30-second grounding exercise together before we continue?",
        "happy": "What do you think is helping most right now, so you can keep that support going?",
        "neutral": "Would you like to tell me a little more about what your day has been like?",
    }
    return follow_ups.get(normalized, follow_ups["neutral"])


def generate_mental_health_response(
    emotion: str,
    confidence: float | None = None,
    conversation_history: str | None = None,
) -> str:
    """Create a concise, empathetic, safety-aware message for a detected emotion.

    Args:
        emotion: Emotion label (e.g., sad, anxious, angry).
        confidence: Optional confidence score in [0, 1].
        conversation_history: Optional serialized history; currently used as an
            extensibility input for future prompt-grounded behavior.
    """

    del conversation_history  # Reserved for future context-aware tailoring.

    normalized = _normalized_emotion(emotion)
    templates = _EMOTION_TEMPLATES.get(normalized, _EMOTION_TEMPLATES["neutral"])
    base = templates[0]

    if normalized == "sad" and confidence is not None and confidence >= 0.8:
        base = (
            "I'm really sorry you're feeling this way. You don't have to carry this alone, "
            "and we can move through this one small step at a time."
        )

    return f"{base} {_build_supportive_follow_up(normalized)}"


@lru_cache(maxsize=1)
def _load_hf_generator():
    if os.getenv("ENABLE_HF_CHAT_RESPONSE", "0") != "1":
        return None

    model_name = os.getenv("HF_CHAT_MODEL", "sshleifer/tiny-gpt2")
    try:
        from transformers import pipeline  # type: ignore

        return pipeline("text-generation", model=model_name)
    except Exception:
        return None



def warmup_response_generator() -> None:
    """Warm optional response-generation model at startup."""

    _load_hf_generator()

def _sanitize_text(text: str) -> str:
    cleaned = text.strip()
    for term in _BLOCKED_TERMS:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        cleaned = pattern.sub("[redacted]", cleaned)
    return re.sub(r"\s+", " ", cleaned)


def detect_crisis_language(text: str) -> bool:
    candidate = text.strip().lower()
    return any(re.search(pattern, candidate) for pattern in _CRISIS_PATTERNS)


def _normalized_emotion(emotion: str) -> str:
    value = emotion.strip().lower() if emotion else "neutral"
    if value == "fearful":
        return "anxious"
    return value


def _template_response(session_id: str, emotion: str) -> str:
    normalized = _normalized_emotion(emotion)
    templates = _EMOTION_TEMPLATES.get(normalized, _EMOTION_TEMPLATES["neutral"])

    previous_index = get_last_template_index(session_id, normalized)
    if previous_index is None:
        next_index = 0
    else:
        next_index = (previous_index + 1) % len(templates)

    set_last_template_index(session_id, normalized, next_index)
    return templates[next_index]


def generate_response(session_id: str, emotion: str, text: str) -> dict[str, str | bool]:
    safe_text = _sanitize_text(text)

    if detect_crisis_language(text):
        return {
            "response_text": EMERGENCY_SUPPORT_MESSAGE,
            "crisis_detected": True,
            "severity": "high",
        }

    prefix = _template_response(session_id, emotion)

    generator = _load_hf_generator()
    if generator is None:
        supportive = generate_mental_health_response(emotion)
        return {
            "response_text": supportive,
            "crisis_detected": False,
            "severity": "low",
        }

    prompt = (
        f"Emotion: {emotion}. User message: {safe_text}. "
        "Write one brief, empathetic, safe response:"
    )
    try:
        output = generator(prompt, max_new_tokens=48, num_return_sequences=1)
        generated = output[0]["generated_text"].replace(prompt, "").strip()
        candidate = generated or prefix
    except Exception:
        candidate = prefix

    return {
        "response_text": _sanitize_text(candidate),
        "crisis_detected": False,
        "severity": "low",
    }
