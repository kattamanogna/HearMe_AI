"""Pydantic request/response schemas for HearMe_AI APIs."""

from pydantic import BaseModel, Field


class TextEmotionPredictRequest(BaseModel):
    """Input payload for text emotion prediction."""

    text: str = Field(..., description="Text content to analyze for emotion.")


class TextEmotionPredictResponse(BaseModel):
    """Emotion prediction result for text input."""

    emotion: str = Field(..., description="Predicted emotion label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence.")


class MultimodalRequest(BaseModel):
    """Incoming data used for multimodal analysis."""

    text: str = Field(..., description="User message text.")
    audio_path: str | None = Field(
        default=None,
        description="Optional local/remote path to an audio clip.",
    )
    frame_path: str | None = Field(
        default=None,
        description="Optional local/remote path to a facial image frame.",
    )


class MultimodalResponse(BaseModel):
    """Predicted emotion and intent, ready for frontend consumption."""

    emotion: str
    intent: str
    confidence: float
    response_text: str
