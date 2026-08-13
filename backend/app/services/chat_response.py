"""Helpers for generating supportive chatbot responses from emotion + text."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from app.services.emotional_intelligence import analyze_emotional_context
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
    # Self-harm and suicide signals.
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bsuicid(?:e|al)\b",
    r"\bself[-\s]?harm(?:ing)?\b",
    r"\bhurt myself\b",
    r"\bcut myself\b",
    r"\bdon't want to live\b",
    r"\bno reason to live\b",
    # Abuse and unsafe-environment signals.
    r"\babuse(?:d|s|r)?\b",
    r"\babusive\b",
    r"\bdomestic violence\b",
    r"\bsexual assault\b",
    r"\bbeing hit\b",
    r"\bthey hit me\b",
    r"\bnot safe at home\b",
    # Panic, crisis, and urgent danger signals.
    r"\bpanic(?:king|ked| attack)?\b",
    r"\bpanicky\b",
    r"\bcrisis\b",
    r"\bemergency\b",
    r"\bimmediate danger\b",
    r"\bcan't breathe\b",
]

_NAME_PATTERNS = [
    re.compile(r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-zA-Z'-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\b(?:with|about|from|for)\s+([A-Z][a-zA-Z'-]{1,30})\b"),
]

_PROBLEM_KEYWORDS = {
    "work": "work stress",
    "job": "job stress",
    "boss": "work stress",
    "meeting": "meeting pressure",
    "deadline": "deadline pressure",
    "school": "school pressure",
    "class": "class pressure",
    "exam": "exam pressure",
    "test": "test pressure",
    "homework": "school pressure",
    "relationship": "relationship strain",
    "partner": "relationship strain",
    "friend": "friendship concern",
    "family": "family concern",
    "parent": "family concern",
    "sleep": "sleep trouble",
    "anxious": "anxiety",
    "anxiety": "anxiety",
    "sad": "sadness",
    "angry": "anger",
    "lonely": "loneliness",
}

_CONTEXTUAL_ACTIVITY_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("exam", "test", "homework", "assignment", "class", "school"),
        "write the next school task on paper and spend five focused minutes on only that piece",
    ),
    (
        ("work", "job", "boss", "meeting", "deadline", "project"),
        "list the one work item that truly needs attention next, then take a two-minute breathing reset before starting it",
    ),
    (
        ("friend", "partner", "relationship"),
        "draft one honest sentence you might say to them, without sending it until you feel steadier",
    ),
    (
        ("family", "parent", "sibling", "home"),
        "step into a quieter spot and write what you need from the family situation before responding",
    ),
    (
        ("sleep", "tired", "exhausted", "bed"),
        "dim one light, put your phone down for five minutes, and let your body settle before deciding anything else",
    ),
    (
        ("lonely", "alone", "isolated"),
        "send a low-pressure check-in to one safe person, even if it is just 'could use a little company today'",
    ),
)

_EMOTION_ACTIVITY_FALLBACKS = {
    "sad": "write three lines about what hurts most, then consider sharing one line with someone you trust",
    "anxious": (
        "do a 5-4-3-2-1 grounding exercise, then break the next task into a step small enough "
        "to finish in ten minutes"
    ),
    "angry": "take a brisk two-minute walk or stretch, then write the unsent version of what you want to say",
    "happy": "capture what made this moment work so you can repeat one part of it later",
    "neutral": "choose one concrete thread from your message and take a tiny action on that first",
}

EMERGENCY_SUPPORT_MESSAGE = (
    "I'm really glad you told me. This sounds urgent, so let's focus on safety first. "
    "If you might hurt yourself, someone else might hurt you, or you are in immediate danger, "
    "call emergency services now or move toward a safer public place if you can. In the U.S. or Canada, "
    "call or text 988 for the Suicide & Crisis Lifeline; if you're elsewhere, contact your local crisis hotline. "
    "If this is panic, try to put both feet on the floor, look around, and name five things you can see while "
    "you slow your breathing. Can you contact one trusted person to stay with you or help you get support right now?"
)

_EMOTION_DETAILS: dict[str, list[dict[str, str]]] = {
    "happy": [
        {
            "acknowledgement": "That sounds like a bright spot worth savoring.",
            "validation": "Good moments can deserve room too, especially when life has been demanding.",
            "suggestion": "Maybe take a small snapshot of what made this feel good so you can return to it later.",
            "encouragement": "Let yourself enjoy it without needing to shrink it down.",
            "question": "What part of this feels most meaningful right now?",
        },
        {
            "acknowledgement": "It's lovely to hear a bit of lightness coming through.",
            "validation": "Feeling good can be grounding, and it is okay to lean into that.",
            "suggestion": "You could share it with someone kind, or simply pause and let the moment land.",
            "encouragement": "These wins count, even the quiet ones.",
            "question": "What helped bring this on?",
        },
    ],
    "sad": [
        {
            "acknowledgement": "I'm really sorry this is weighing on you.",
            "validation": "Feeling low can be exhausting, and it makes sense that this would hurt.",
            "suggestion": "If it feels doable, try one gentle next step: drink some water, step outside for a minute, or text someone safe.",
            "encouragement": "You don't have to solve everything at once; just getting through the next small piece counts.",
            "question": "What feels heaviest right now?",
        },
        {
            "acknowledgement": "That sounds tender and heavy to carry.",
            "validation": "Anyone could feel worn down when something matters this much.",
            "suggestion": "For the next few minutes, it may help to lower the pressure and do one caring thing for your body.",
            "encouragement": "Small care still matters on hard days.",
            "question": "Would talking through what happened help a little?",
        },
        {
            "acknowledgement": "Oof, that sounds painful.",
            "validation": "It is reasonable that your heart feels bruised by this.",
            "suggestion": "Try naming the feeling without arguing with it, then choose the smallest next step that feels manageable.",
            "encouragement": "You can move slowly here; there is no need to force yourself to be okay immediately.",
            "question": "What would feel like a little relief tonight?",
        },
    ],
    "angry": [
        {
            "acknowledgement": "I can hear how fired up and frustrated this has left you.",
            "validation": "Anger often shows up when something feels unfair, painful, or out of your control.",
            "suggestion": "Before responding, it may help to take a short pause, unclench your jaw, and write the first unfiltered version somewhere private.",
            "encouragement": "You can take care of yourself without letting this moment decide everything for you.",
            "question": "What part of this crossed the line for you?",
        },
        {
            "acknowledgement": "That would leave a lot of people feeling heated.",
            "validation": "Your reaction may be pointing to a boundary, a hurt, or something that needs to be addressed.",
            "suggestion": "Give yourself a little space before you choose what to say or do next.",
            "encouragement": "You can be firm and still protect your peace.",
            "question": "What outcome would feel fair from here?",
        },
    ],
    "anxious": [
        {
            "acknowledgement": "That sounds like a lot for your mind and body to carry.",
            "validation": "Anxiety can feel so convincing when you're trying to handle uncertainty or pressure.",
            "suggestion": "Try planting both feet on the floor and naming five things you can see, then pick just one next task that is small enough to start.",
            "encouragement": "You can move through this one breath and one choice at a time.",
            "question": "What's the worry that keeps looping the loudest?",
        },
        {
            "acknowledgement": "Your nervous system sounds really activated right now.",
            "validation": "When things feel uncertain, the mind can race ahead trying to protect you.",
            "suggestion": "See if you can slow the moment down: breathe out longer than you breathe in, then name what is actually in front of you.",
            "encouragement": "You do not need the whole answer before taking the next steady step.",
            "question": "What is one thing you know for sure in this moment?",
        },
    ],
    "neutral": [
        {
            "acknowledgement": "We can take this at an easy pace.",
            "validation": "Whatever you're noticing is worth taking seriously, even if it feels hard to name yet.",
            "suggestion": "You could start by checking in with your body, your energy, or the one thing you most need today.",
            "encouragement": "There is no wrong place to begin.",
            "question": "What would feel helpful to talk through first?",
        },
        {
            "acknowledgement": "Thanks for putting words to what is going on.",
            "validation": "Sometimes naming the moment is enough of a first step.",
            "suggestion": "If you want, choose one thread and we can gently untangle it together.",
            "encouragement": "No pressure to have it all figured out.",
            "question": "Where would you like to start?",
        },
    ],
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
    current_text: str = "",
    *,
    session_id: str = "default",
) -> str:
    """Create a concise, empathetic, safety-aware message for a detected emotion."""

    normalized = _normalized_emotion(emotion)
    styles = _EMOTION_DETAILS.get(normalized, _EMOTION_DETAILS["neutral"])
    last_index = get_last_template_index(session_id, normalized)
    style_index = 0 if last_index is None else (last_index + 1) % len(styles)
    set_last_template_index(session_id, normalized, style_index)

    style = styles[style_index].copy()
    if normalized == "sad" and confidence is not None and confidence >= 0.8:
        style["acknowledgement"] = "This sounds deeply painful, and you deserve gentleness right now."
        style["encouragement"] = "You do not have to carry the whole day at once."

    parts: list[str] = []
    if current_text and conversation_history:
        parts.append(conversation_history.strip())
    elif conversation_history:
        parts.append(_short_reflection(conversation_history))
    response_parts = [style["acknowledgement"], style["validation"]]
    contextual_note = _emotional_context_note(_analyze_current_context(current_text, normalized)) if current_text else ""
    if contextual_note:
        response_parts.append(contextual_note)
    if current_text:
        response_parts.append(_short_reflection(current_text))
    coping_suggestion = _personalized_coping_suggestion(normalized, current_text)
    response_parts.extend(
        [coping_suggestion or style["suggestion"], style["encouragement"], style["question"]]
    )
    parts.extend(response_parts)
    return " ".join(part for part in parts if part)


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


def _personalized_coping_suggestion(emotion: str, text: str) -> str:
    """Recommend a concrete coping activity that fits the emotion and message context."""

    lowered = text.lower()
    activity = ""
    for keywords, candidate in _CONTEXTUAL_ACTIVITY_KEYWORDS:
        if any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in keywords):
            activity = candidate
            break

    if not activity:
        activity = _EMOTION_ACTIVITY_FALLBACKS.get(emotion, _EMOTION_ACTIVITY_FALLBACKS["neutral"])

    lead_ins = {
        "sad": "Because this sounds sad and tied to what you shared, try this specific next step:",
        "anxious": "Because this sounds anxious, give your body and attention a concrete anchor:",
        "angry": "Because this sounds heated, choose an activity that creates space before action:",
        "happy": "Because this sounds like something worth protecting, try this:",
        "neutral": "For this specific situation, try one practical next step:",
    }
    return f"{lead_ins.get(emotion, lead_ins['neutral'])} {activity}."


def _short_reflection(text: str) -> str:
    safe_text = _sanitize_text(text)
    if not safe_text:
        return "It sounds like you're checking in and trying to make sense of what's going on."

    trimmed = safe_text.rstrip(".!?")
    if len(trimmed) > 130:
        trimmed = f"{trimmed[:127].rsplit(' ', 1)[0]}..."
    return f"From what you shared, {trimmed.lower()}."



def _analyze_current_context(text: str, emotion: str):
    """Analyze the current turn for tone adjustments without relying on globals."""

    return analyze_emotional_context(text, primary_emotion=emotion)


def _emotional_context_note(context) -> str:
    """Add a brief humanizing note for nuanced emotional signals."""

    notes: list[str] = []
    if context.confused:
        notes.append("If part of this feels tangled or unclear, we can slow it down together.")
    if context.sarcastic:
        notes.append("I also hear a bit of weary frustration underneath the words.")
    if context.hopeless:
        notes.append("When hope feels far away, the goal can simply be getting through this next moment safely.")
    if context.repeated_negative_thoughts:
        notes.append("Since this pattern has been showing up repeatedly, it may help to treat the thought as a signal rather than a verdict.")
    if context.mixed_emotions:
        notes.append(f"It makes sense if this feels mixed: {', '.join(context.mixed_emotions)} can overlap.")
    return " ".join(notes)


def _emotional_context_payload(context, emotion: str, *, crisis_detected: bool = False, severity: str | None = None) -> dict[str, bool | str | list[str]]:
    """Return stable emotional-context data for API consumers."""

    payload = context.as_dict()
    payload["primary_emotion"] = _normalized_emotion(emotion)
    payload["severity"] = severity or str(payload.get("intensity", "low"))
    payload["crisis_detected"] = crisis_detected
    return payload

def _fallback_response(emotion: str, text: str, memory_context: str, session_id: str) -> str:
    return generate_mental_health_response(
        emotion,
        conversation_history=memory_context,
        current_text=text,
        session_id=session_id,
    )


def generate_response(
    session_id: str, emotion: str, text: str
) -> dict[str, str | bool | dict[str, bool | str | list[str]]]:
    history = get_chat_history(session_id)
    recent_user_messages = [str(item.get("text", "")) for item in history if item.get("text")]
    context = analyze_emotional_context(text, primary_emotion=emotion, recent_user_messages=recent_user_messages)
    memory_context = _build_memory_context(history)

    if detect_crisis_language(text):
        return {
            "response_text": EMERGENCY_SUPPORT_MESSAGE,
            "crisis_detected": True,
            "severity": "high",
            "emotional_context": _emotional_context_payload(context, emotion, crisis_detected=True, severity="high"),
        }

    safe_text = _sanitize_text(text)
    fallback = _fallback_response(emotion, safe_text, memory_context, session_id)
    generator = _load_hf_generator()
    if generator is None:
        return {
            "response_text": fallback,
            "crisis_detected": False,
            "severity": context.intensity,
            "emotional_context": _emotional_context_payload(context, emotion),
        }

    history_prompt = f"Conversation memory: {memory_context}. " if memory_context else ""
    prompt = (
        f"{history_prompt}Emotion: {emotion}. User message: {safe_text}. "
        "Write one brief, warm, natural, calm, non-judgmental response. "
        "Avoid stock acknowledgement phrases and vary the wording:"
    )
    try:
        output = generator(prompt, max_new_tokens=48, num_return_sequences=1)
        generated = output[0]["generated_text"].replace(prompt, "").strip()
        candidate = generated or fallback
    except Exception:
        candidate = fallback

    return {
        "response_text": _sanitize_text(candidate),
        "crisis_detected": False,
        "severity": context.intensity,
        "emotional_context": _emotional_context_payload(context, emotion),
    }
