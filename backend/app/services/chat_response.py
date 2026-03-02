"""Helpers for generating supportive chatbot responses from emotion + text."""

from __future__ import annotations

import os
import re
from functools import lru_cache

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

_EMOTION_TEMPLATES: dict[str, list[str]] = {
    "happy": [
        "I'm glad you're feeling upbeat. Keep building on what is working for you.",
        "That positive energy matters—hold onto the moments that are helping you feel better.",
    ],
    "sad": [
        "I'm sorry this feels heavy right now. You don't have to carry everything at once.",
        "It sounds like you're going through a tough moment. Taking one small step can help.",
    ],
    "angry": [
        "It makes sense to feel frustrated. A short pause or deep breath can create space to respond calmly.",
        "I hear the intensity in what you're sharing. Let's focus on one thing you can control next.",
    ],
    "fear": [
        "That sounds overwhelming. Grounding yourself in the present can reduce some of the pressure.",
        "Feeling anxious is hard. Try naming one immediate action that helps you feel safer.",
    ],
    "neutral": [
        "Thanks for sharing. I'm here to help you reflect on what you're experiencing.",
        "I appreciate your message. We can work through this one step at a time.",
    ],
}


@lru_cache(maxsize=1)
def _load_hf_generator():
    """Load an optional HuggingFace text-generation pipeline.

    Returns None when unavailable or disabled.
    """

    if os.getenv("ENABLE_HF_CHAT_RESPONSE", "0") != "1":
        return None

    model_name = os.getenv("HF_CHAT_MODEL", "sshleifer/tiny-gpt2")
    try:
        from transformers import pipeline  # type: ignore

        return pipeline("text-generation", model=model_name)
    except Exception:
        return None


def _sanitize_text(text: str) -> str:
    """Mask blocked terms and normalize whitespace."""

    cleaned = text.strip()
    for term in _BLOCKED_TERMS:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        cleaned = pattern.sub("[redacted]", cleaned)
    return re.sub(r"\s+", " ", cleaned)


def _template_response(emotion: str) -> str:
    """Return a deterministic emotion-based template response."""

    normalized = emotion.strip().lower() if emotion else "neutral"
    templates = _EMOTION_TEMPLATES.get(normalized, _EMOTION_TEMPLATES["neutral"])
    return templates[0]


def generate_response(emotion: str, text: str) -> str:
    """Generate a safe, supportive response using emotion and user text.

    The function always applies a language safety filter. If an optional
    HuggingFace model is enabled, it is used to produce a short continuation
    that is still post-processed by the same safety filter.
    """

    safe_text = _sanitize_text(text)
    prefix = _template_response(emotion)

    generator = _load_hf_generator()
    if generator is None:
        return f"{prefix} You said: \"{safe_text}\""

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

    return _sanitize_text(candidate)
