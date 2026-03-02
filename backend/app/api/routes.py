"""API routes for health checks and model inference endpoints."""

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    MultimodalRequest,
    MultimodalResponse,
    TextEmotionPredictRequest,
    TextEmotionPredictResponse,
)
from app.services.text_emotion_model import predict_text_emotion

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Simple health endpoint for uptime monitoring."""
    return {"status": "ok"}


@router.post("/predict-text", response_model=TextEmotionPredictResponse)
def predict_text(payload: TextEmotionPredictRequest) -> TextEmotionPredictResponse:
    """Predict emotion and confidence score for the provided text."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'text' is required and cannot be empty.",
        )

    prediction = predict_text_emotion(payload.text.strip())
    return TextEmotionPredictResponse(
        emotion=prediction["emotion"],
        confidence=prediction["confidence"],
    )


@router.post("/analyze", response_model=MultimodalResponse)
def analyze_multimodal(payload: MultimodalRequest) -> MultimodalResponse:
    """Combine text, audio, and face features and return final emotion + intent."""
    # TODO: Replace placeholder response with real model outputs.
    return MultimodalResponse(
        emotion="neutral",
        intent="general_support",
        confidence=0.0,
        response_text="I am here to listen. Tell me more about how you feel.",
    )
