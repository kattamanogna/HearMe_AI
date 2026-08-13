"""Rule-based emotional context detection for adaptive chatbot tone."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re

NEGATIVE_EMOTIONS = {"sad", "sadness", "angry", "anger", "anxious", "fear", "fearful"}

_EMOTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sad": ("sad", "down", "empty", "lonely", "awful", "miserable", "hurt", "cry"),
    "angry": ("angry", "mad", "furious", "annoyed", "irritated", "unfair", "rage"),
    "anxious": (
        "anxious",
        "worried",
        "nervous",
        "panic",
        "scared",
        "afraid",
        "overwhelmed",
    ),
    "happy": (
        "happy",
        "glad",
        "excited",
        "joy",
        "great",
        "wonderful",
        "hopeful",
        "proud",
    ),
    "gratitude": ("thank", "thanks", "grateful", "appreciate", "blessed"),
}

_CONFUSION_PATTERNS = (
    r"\bconfus(?:ed|ing|ion)\b",
    r"\bi don't understand\b",
    r"\bi do not understand\b",
    r"\bi don't know\b",
    r"\bi have no idea\b",
    r"\bwhat do you mean\b",
    r"\bhow am i supposed to\b",
)

_HOPELESSNESS_PATTERNS = (
    r"\bno point\b",
    r"\bnothing (?:will|can) (?:ever )?(?:change|help|get better)\b",
    r"\bit'?s hopeless\b",
    r"\bi can't go on\b",
    r"\bi cannot go on\b",
    r"\bgive up\b",
    r"\bi'm done\b",
    r"\bi am done\b",
)

_SARCASM_PATTERNS = (
    r"\byeah right\b",
    r"\bsure,? because\b",
    r"\bthanks for nothing\b",
    r"\bjust great\b",
    r"\blove that for me\b",
    r"\bas if\b",
    r"\bwhat a surprise\b",
)

_INTENSIFIERS = {
    "really",
    "very",
    "so",
    "extremely",
    "totally",
    "completely",
    "absolutely",
    "unbearable",
}
_NEGATIVE_THOUGHT_PATTERNS = (
    r"\bi (?:always|never)\b",
    r"\bnothing (?:ever )?(?:works|helps|changes)\b",
    r"\beverything is (?:bad|wrong|awful|terrible)\b",
    r"\bi can't do (?:anything|this)\b",
)


@dataclass(frozen=True)
class EmotionalContext:
    """Nuanced emotional signals used to shape response tone."""

    mixed_emotions: list[str] = field(default_factory=list)
    confused: bool = False
    intensity: str = "low"
    sarcastic: bool = False
    repeated_negative_thoughts: bool = False
    hopeless: bool = False
    happy: bool = False
    grateful: bool = False

    def as_dict(self) -> dict[str, bool | str | list[str]]:
        return {
            "mixed_emotions": self.mixed_emotions,
            "confused": self.confused,
            "intensity": self.intensity,
            "sarcastic": self.sarcastic,
            "repeated_negative_thoughts": self.repeated_negative_thoughts,
            "hopeless": self.hopeless,
            "happy": self.happy,
            "grateful": self.grateful,
        }


def analyze_emotional_context(
    text: str,
    *,
    primary_emotion: str = "neutral",
    recent_user_messages: list[str] | None = None,
) -> EmotionalContext:
    """Infer nuanced emotional cues from current and recent messages."""

    normalized = text.lower().strip()
    emotion_hits = _detect_emotion_hits(normalized)
    primary = _normalize_emotion(primary_emotion)
    if primary in _EMOTION_KEYWORDS and primary != "neutral":
        emotion_hits[primary] += 1

    mixed = [emotion for emotion, count in emotion_hits.items() if count > 0]
    if "gratitude" in mixed and "happy" in mixed:
        mixed.remove("gratitude")
    mixed_emotions = sorted(mixed) if len(mixed) > 1 else []

    recent = recent_user_messages or []
    repeated_negative = _has_repeated_negative_thoughts(normalized, recent)

    return EmotionalContext(
        mixed_emotions=mixed_emotions,
        confused=_matches_any(normalized, _CONFUSION_PATTERNS)
        or normalized.count("?") >= 2,
        intensity=_detect_intensity(text, normalized),
        sarcastic=_matches_any(normalized, _SARCASM_PATTERNS)
        or _has_sarcastic_contrast(normalized),
        repeated_negative_thoughts=repeated_negative,
        hopeless=_matches_any(normalized, _HOPELESSNESS_PATTERNS),
        happy=emotion_hits["happy"] > 0,
        grateful=emotion_hits["gratitude"] > 0,
    )


def _detect_emotion_hits(text: str) -> Counter[str]:
    hits: Counter[str] = Counter()
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\w*\b", text):
                hits[emotion] += 1
    return hits


def _detect_intensity(original: str, normalized: str) -> str:
    score = 0
    score += min(3, len(re.findall(r"[!?]{2,}", original)))
    score += min(3, sum(1 for word in normalized.split() if word in _INTENSIFIERS))
    score += (
        2
        if re.search(
            r"\b(?:devastated|terrified|furious|unbearable|desperate)\b", normalized
        )
        else 0
    )
    score += (
        1
        if any(
            token.isupper() and len(token) > 2
            for token in re.findall(r"\b\w+\b", original)
        )
        else 0
    )
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _has_repeated_negative_thoughts(text: str, recent_messages: list[str]) -> bool:
    if _matches_any(text, _NEGATIVE_THOUGHT_PATTERNS):
        return True
    recent_negative_count = 0
    for message in recent_messages[-4:]:
        lowered = message.lower()
        if _matches_any(lowered, _NEGATIVE_THOUGHT_PATTERNS) or any(
            word in lowered for word in ("awful", "hopeless", "worthless", "never")
        ):
            recent_negative_count += 1
    return recent_negative_count >= 2


def _has_sarcastic_contrast(text: str) -> bool:
    has_positive_marker = re.search(
        r"\b(?:great|perfect|wonderful|awesome|amazing)\b", text
    )
    has_negative_context = re.search(
        r"\b(?:again|another|ruined|failed|terrible|awful|nothing works)\b", text
    )
    return bool(has_positive_marker and has_negative_context)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _normalize_emotion(emotion: str) -> str:
    value = emotion.strip().lower() if emotion else "neutral"
    return {
        "sadness": "sad",
        "anger": "angry",
        "fearful": "anxious",
        "fear": "anxious",
    }.get(value, value)
