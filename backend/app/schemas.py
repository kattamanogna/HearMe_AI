"""Pydantic request/response schemas for HearMe_AI APIs."""

from pydantic import BaseModel, Field


class TextEmotionPredictRequest(BaseModel):
    """Input payload for text emotion prediction."""

    text: str = Field(..., description="Text content to analyze for emotion.")


class TextEmotionPredictResponse(BaseModel):
    """Emotion prediction result for text input."""

    emotion: str = Field(..., description="Predicted emotion label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence.")
    probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Full probability distribution by emotion label.",
    )


class ModalityPredictResponse(BaseModel):
    """Emotion prediction result for a single non-text modality."""

    emotion: str = Field(..., description="Predicted emotion label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence.")
    probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Full probability distribution by emotion label.",
    )
    face_detected: bool | None = Field(
        default=None,
        description="Whether a face was detected before face-emotion inference.",
    )


class MultimodalRequest(BaseModel):
    session_id: str = Field(
        default="default",
        description="Session identifier used for in-memory conversation history.",
    )
    text: str = Field(..., description="User message text.")
    audio_bytes: str | None = Field(default=None, description="Optional base64-encoded audio bytes.")
    face_base64: str | None = Field(default=None, description="Optional base64-encoded image bytes (JPEG/PNG).")


class MultimodalResponse(BaseModel):
    text_emotion: str = Field(..., description="Emotion predicted from text model.")
    text_confidence: float = Field(..., ge=0.0, le=1.0)
    text_probabilities: dict[str, float] = Field(default_factory=dict)
    audio_emotion: str | None = Field(default=None)
    audio_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    audio_probabilities: dict[str, float] | None = Field(default=None)
    face_emotion: str | None = Field(default=None)
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    face_probabilities: dict[str, float] | None = Field(default=None)
    fused_emotion: str = Field(..., description="Final fused emotion across available signals.")
    fused_confidence: float = Field(..., ge=0.0, le=1.0, description="Fusion confidence score.")
    fused_probabilities: dict[str, float] = Field(default_factory=dict)
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    response_text: str = Field(...)
    crisis_detected: bool = Field(default=False)
    severity: str = Field(default="low")


class SessionSummaryResponse(BaseModel):
    emotional_trend: str = Field(...)
    dominant_emotion: str = Field(...)
    average_confidence: float = Field(..., ge=0.0, le=1.0)
    recent_emotions: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    session_id: str = Field(default="default")
    text: str = Field(...)


class ChatStreamChunk(BaseModel):
    session_id: str = Field(...)
    chunk: str = Field(...)
    done: bool = Field(default=False)
