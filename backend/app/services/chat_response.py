"""Helpers for generating supportive chatbot responses from emotion + text."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from app.services.emotional_intelligence import (
    EmotionalContext,
    analyze_emotional_context,
)
from app.services.session_manager import (
    get_last_template_index,
    get_recent_user_messages,
    set_last_template_index,
)

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


def generate_mental_health_response(
    emotion: str,
    confidence: float | None = None,
    conversation_history: str | None = None,
) -> str:
    """Create a concise, empathetic, safety-aware message for a detected emotion.

    Args:
        emotion: Emotion label (e.g., sad, anxious, angry).
        confidence: Optional confidence score in [0, 1].
        conversation_history: Optional serialized history used to tune tone when supplied.
    """

    normalized = _normalized_emotion(emotion)
    templates = _EMOTION_TEMPLATES.get(normalized, _EMOTION_TEMPLATES["neutral"])
    base = templates[0]

    if normalized == "sad" and confidence is not None and confidence >= 0.8:
        base = (
            "I'm really sorry you're feeling this way. You don't have to carry this alone, "
            "and we can move through this one small step at a time."
        )

    context = analyze_emotional_context(
        conversation_history or "", primary_emotion=normalized
    )
    return _apply_emotional_tone(base, normalized, context)


def _apply_emotional_tone(base: str, emotion: str, context: EmotionalContext) -> str:
    """Adjust response wording to match nuanced emotional context."""

    normalized = _normalized_emotion(emotion)
    tone_parts: list[str] = []

    if context.hopeless:
        tone_parts.append(
            "This sounds painfully stuck, and I want to stay with you in it rather than minimize it."
        )
    elif context.intensity == "high":
        tone_parts.append(
            "This sounds really intense right now, so let's slow the pace together."
        )
    elif context.intensity == "medium":
        tone_parts.append("I can hear there is a lot of feeling behind this.")

    if context.mixed_emotions:
        readable = (
            ", ".join(context.mixed_emotions[:-1])
            + f" and {context.mixed_emotions[-1]}"
            if len(context.mixed_emotions) > 1
            else context.mixed_emotions[0]
        )
        tone_parts.append(
            f"It makes sense that this feels mixed—there may be {readable} here at the same time."
        )

    if context.confused:
        tone_parts.append(
            "If things feel confusing, we can make this simpler and take just the next clear step."
        )

    if context.sarcastic:
        tone_parts.append(
            "I may be hearing some frustration behind the sarcasm, and that frustration matters."
        )

    if context.repeated_negative_thoughts:
        tone_parts.append(
            "I also notice this thought pattern may be circling back, which can make everything feel heavier."
        )

    if context.grateful:
        tone_parts.append(
            "I'm glad you told me, and I appreciate the trust it takes to share this."
        )
    elif context.happy and normalized == "happy":
        tone_parts.append(
            "I'm happy to hear there is something positive here—let's help you hold onto it."
        )

    tone_parts.append(base)
    tone_parts.append(_build_supportive_follow_up(normalized))
    return " ".join(part for part in tone_parts if part)


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
    return _EMOTION_ALIASES.get(value, value)


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
    context = analyze_emotional_context(
        text,
        primary_emotion=emotion,
        recent_user_messages=get_recent_user_messages(session_id),
    )

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
        supportive = _apply_emotional_tone(prefix, emotion, context)
        return {
            "response_text": supportive,
            "crisis_detected": False,
            "severity": context.intensity,
            "emotional_context": context.as_dict(),
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
        "severity": context.intensity,
        "emotional_context": context.as_dict(),
    }
