"""Helpers for generating supportive chatbot responses from emotion + text."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from app.services.session_manager import get_chat_history, get_last_template_index, set_last_template_index

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

_NAME_PATTERNS = [
    re.compile(r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-zA-Z'-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\b(?:with|about|from|for)\s+([A-Z][a-zA-Z'-]{1,30})\b"),
]

_PROBLEM_KEYWORDS = {
    "work": "work stress",
    "job": "job stress",
    "school": "school pressure",
    "exam": "exam pressure",
    "relationship": "relationship strain",
    "friend": "friendship concern",
    "family": "family concern",
    "sleep": "sleep trouble",
    "anxious": "anxiety",
    "anxiety": "anxiety",
    "sad": "sadness",
    "angry": "anger",
    "lonely": "loneliness",
}

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


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def _extract_names(messages: list[str]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for pattern in _NAME_PATTERNS:
            names.extend(match.group(1) for match in pattern.finditer(message))
    return _unique_preserving_order(names)[-3:]


def _extract_problems(messages: list[str]) -> list[str]:
    problems: list[str] = []
    for message in messages:
        lowered = message.lower()
        for keyword, label in _PROBLEM_KEYWORDS.items():
            if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                problems.append(label)
    return _unique_preserving_order(problems)[-3:]


def _build_memory_context(history: list[dict[str, Any]]) -> str:
    """Summarize the last 10 prior exchanges for response personalization."""

    if not history:
        return ""

    recent = history[-10:]
    user_messages = [str(item.get("text", "")).strip() for item in recent if item.get("text")]
    emotions = [str(item.get("fused_emotion", "")).strip() for item in recent if item.get("fused_emotion")]
    advice = [str(item.get("response_text", "")).strip() for item in recent if item.get("response_text")]

    details: list[str] = []
    names = _extract_names(user_messages)
    if names:
        details.append(f"I remember you mentioned {', '.join(names)}.")

    problems = _extract_problems(user_messages)
    if problems:
        details.append(f"We have been talking about {', '.join(problems)}.")

    if emotions:
        unique_emotions = _unique_preserving_order([_normalized_emotion(emotion) for emotion in emotions])[-3:]
        details.append(f"Earlier emotions included {', '.join(unique_emotions)}.")

    if advice:
        details.append("Last time, we focused on taking one small, grounding step.")

    return " ".join(details)


def generate_mental_health_response(
    emotion: str,
    confidence: float | None = None,
    conversation_history: str | None = None,
) -> str:
    """Create a concise, empathetic, safety-aware message for a detected emotion."""

    normalized = _normalized_emotion(emotion)
    templates = _EMOTION_TEMPLATES.get(normalized, _EMOTION_TEMPLATES["neutral"])
    base = templates[0]

    if normalized == "sad" and confidence is not None and confidence >= 0.8:
        base = (
            "I'm really sorry you're feeling this way. You don't have to carry this alone, "
            "and we can move through this one small step at a time."
        )

    memory_prefix = f"{conversation_history} " if conversation_history else ""
    return f"{memory_prefix}{base} {_build_supportive_follow_up(normalized)}"


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
    if value in {"fearful", "fear"}:
        return "anxious"
    if value == "sadness":
        return "sad"
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
    memory_context = _build_memory_context(get_chat_history(session_id))

    if detect_crisis_language(text):
        return {
            "response_text": EMERGENCY_SUPPORT_MESSAGE,
            "crisis_detected": True,
            "severity": "high",
        }

    prefix = _template_response(session_id, emotion)

    generator = _load_hf_generator()
    if generator is None:
        supportive = generate_mental_health_response(emotion, conversation_history=memory_context)
        return {
            "response_text": supportive,
            "crisis_detected": False,
            "severity": "low",
        }

    history_prompt = f"Conversation memory: {memory_context}. " if memory_context else ""
    prompt = (
        f"{history_prompt}Emotion: {emotion}. User message: {safe_text}. "
        "Write one brief, empathetic, safe response that uses relevant prior context:"
    )
    try:
        output = generator(prompt, max_new_tokens=48, num_return_sequences=1)
        generated = output[0]["generated_text"].replace(prompt, "").strip()
        candidate = generated or f"{memory_context} {prefix}".strip()
    except Exception:
        candidate = f"{memory_context} {prefix}".strip()

    return {
        "response_text": _sanitize_text(candidate),
        "crisis_detected": False,
        "severity": "low",
    }
