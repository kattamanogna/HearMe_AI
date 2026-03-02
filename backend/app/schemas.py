"""Pydantic request/response schemas for HearMe_AI APIs."""

from pydantic import BaseModel, Field


class TextEmotionPredictRequest(BaseModel):
    """Input payload for text emotion prediction."""

    text: str = Field(..., description="Text content to analyze for emotion.")


class TextEmotionPredictResponse(BaseModel):
    """Emotion prediction result for text input."""

    emotion: str = Field(..., description="Predicted emotion label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence.")


class ModalityPredictResponse(BaseModel):
    """Emotion prediction result for a single non-text modality."""

    emotion: str = Field(..., description="Predicted emotion label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence.")


class MultimodalRequest(BaseModel):
    """Unified multimodal request payload.

    Example:
        {
          "text": "I feel anxious but trying to stay calm",
          "audio_bytes": "<base64-encoded-audio-bytes>",
          "face_base64": "<base64-encoded-image-bytes>"
        }
    """

    session_id: str = Field(
        default="default",
        description="Session identifier used for in-memory conversation history.",
    )
    text: str = Field(..., description="User message text.")
    audio_bytes: str | None = Field(
        default=None,
        description="Optional base64-encoded audio bytes.",
        examples=["UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."],
    )
    face_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded image bytes (JPEG/PNG).",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD..."],
    )


class MultimodalResponse(BaseModel):
    """Unified multimodal emotion response."""

    text_emotion: str = Field(..., description="Emotion predicted from text model.")
    audio_emotion: str | None = Field(
        default=None,
        description="Emotion predicted from audio model when audio is supplied.",
    )
    face_emotion: str | None = Field(
        default=None,
        description="Emotion predicted from face model when face image is supplied.",
    )
    fused_emotion: str = Field(..., description="Final fused emotion across available signals.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Fusion confidence score.")
    chat_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Last N stored interactions for the provided session.",
    )
    response_text: str = Field(
        ...,
        description="Supportive response generated from the fused emotion and text input.",
    )


class ChatMessage(BaseModel):
    """WebSocket chat input payload."""

    session_id: str = Field(
        default="default",
        description="Session identifier used for in-memory conversation history.",
    )
    text: str = Field(..., description="User message text.")


class ChatStreamChunk(BaseModel):
    """Single streamed WebSocket response chunk."""

    session_id: str = Field(..., description="Session identifier for this chat stream.")
    chunk: str = Field(..., description="Incremental piece of assistant text.")
    done: bool = Field(default=False, description="Whether this is the final chunk.")
