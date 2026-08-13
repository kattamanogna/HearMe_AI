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

_EMOTION_DETAILS: dict[str, dict[str, str]] = {
    "happy": {
        "acknowledgement": "I'm glad there's some brightness in this moment.",
        "validation": "It makes sense to let yourself enjoy something that feels good.",
        "suggestion": "You might pause and notice what helped create this, or share the good news with someone who'd celebrate with you.",
        "encouragement": "Hold onto this; positive moments matter, even when they're small.",
        "question": "What's been the best part of it for you?",
    },
    "sad": {
        "acknowledgement": "I'm really sorry this is weighing on you.",
        "validation": "Feeling low can be exhausting, and it makes sense that this would hurt.",
        "suggestion": "If it feels doable, try one gentle next step: drink some water, step outside for a minute, or text someone safe.",
        "encouragement": "You don't have to solve everything at once; just getting through the next small piece counts.",
        "question": "What feels heaviest right now?",
    },
    "angry": {
        "acknowledgement": "I can hear how fired up and frustrated this has left you.",
        "validation": "Anger often shows up when something feels unfair, painful, or out of your control.",
        "suggestion": "Before responding, it may help to take a short pause, unclench your jaw, and write the first unfiltered version somewhere private.",
        "encouragement": "You can take care of yourself without letting this moment decide everything for you.",
        "question": "What part of this crossed the line for you?",
    },
    "anxious": {
        "acknowledgement": "That sounds like a lot for your mind and body to carry.",
        "validation": "Anxiety can feel so convincing when you're trying to handle uncertainty or pressure.",
        "suggestion": "Try planting both feet on the floor and naming five things you can see, then pick just one next task that is small enough to start.",
        "encouragement": "You can move through this one breath and one choice at a time.",
        "question": "What's the worry that keeps looping the loudest?",
    },
    "neutral": {
        "acknowledgement": "I'm here with you.",
        "validation": "Whatever you're noticing is worth taking seriously, even if it feels hard to name yet.",
        "suggestion": "You could start by checking in with your body, your energy, or the one thing you most need today.",
        "encouragement": "We can take this at your pace.",
        "question": "What would feel helpful to talk through first?",
    },
}

_EMOTION_ALIASES = {
    "sadness": "sad",
    "fear": "anxious",
    "fearful": "anxious",
    "anxiety": "anxious",
    "joy": "happy",
    "happiness": "happy",
    "anger": "angry",
}


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


def _short_reflection(text: str) -> str:
    safe_text = _sanitize_text(text)
    if not safe_text:
        return "It sounds like you're checking in and trying to make sense of what's going on."

    trimmed = safe_text.rstrip(".!?")
    if len(trimmed) > 130:
        trimmed = f"{trimmed[:127].rsplit(' ', 1)[0]}..."
    return f"From what you shared, {trimmed.lower()}."


def _compose_human_response(emotion: str, text: str) -> str:
    details = _EMOTION_DETAILS.get(_normalized_emotion(emotion), _EMOTION_DETAILS["neutral"])
    reflection = _short_reflection(text)
    return " ".join(
        [
            details["acknowledgement"],
            details["validation"],
            reflection,
            details["suggestion"],
            details["encouragement"],
            details["question"],
        ]
    )


def generate_response(
    session_id: str, emotion: str, text: str
) -> dict[str, str | bool | dict[str, bool | str | list[str]]]:
    safe_text = _sanitize_text(text)
    memory_context = _build_memory_context(get_chat_history(session_id))

    if detect_crisis_language(text):
        return {
            "response_text": EMERGENCY_SUPPORT_MESSAGE,
            "crisis_detected": True,
            "severity": "high",
            "emotional_context": context.as_dict(),
        }

    safe_text = _sanitize_text(text)
    generator = _load_hf_generator()
    if generator is None:
        supportive = generate_mental_health_response(emotion, conversation_history=memory_context)
        return {
            "response_text": supportive,
            "crisis_detected": False,
            "severity": context.intensity,
            "emotional_context": context.as_dict(),
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
        "severity": context.intensity,
        "emotional_context": context.as_dict(),
    }
