"""API routes for health checks and model inference endpoints."""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    MultimodalRequest,
    MultimodalResponse,
    TextEmotionPredictRequest,
    TextEmotionPredictResponse,
)
from app.services.audio_emotion_model import predict_audio_emotion
from app.services.face_emotion_model import detect_face_and_predict
from app.services.fusion_engine import combine_predictions
from app.services.text_emotion_model import predict_text_emotion

router = APIRouter(prefix="/api/v1", tags=["inference"])


def _decode_base64_payload(data: str, field_name: str) -> bytes:
    """Decode an optional base64 payload into bytes.

    Raises:
        HTTPException: If the payload is not valid base64-encoded content.
    """

    try:
        return base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{field_name}' must be valid base64-encoded bytes.",
        ) from exc


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
    """Run multimodal emotion inference and return unified response.

    Example request body:
    {
      "text": "I am stressed but hopeful",
      "audio_bytes": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
      "face_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
    }
    """

    text_value = payload.text.strip()
    if not text_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'text' is required and cannot be empty.",
        )

    text_prediction = predict_text_emotion(text_value)

    audio_prediction = None
    if payload.audio_bytes:
        audio_bytes = _decode_base64_payload(payload.audio_bytes, "audio_bytes")
        audio_prediction = predict_audio_emotion(audio_bytes)

    face_prediction = None
    if payload.face_base64:
        image_bytes = _decode_base64_payload(payload.face_base64, "face_base64")
        face_prediction = detect_face_and_predict(image_bytes)

    fused = combine_predictions(text_prediction, audio_prediction, face_prediction)

    return MultimodalResponse(
        text_emotion=str(text_prediction.get("emotion", "neutral")),
        audio_emotion=(
            str(audio_prediction.get("emotion", "neutral")) if audio_prediction else None
        ),
        face_emotion=(
            str(face_prediction.get("emotion", "neutral")) if face_prediction else None
        ),
        fused_emotion=str(fused.get("emotion", "neutral")),
        confidence=float(fused.get("confidence", 0.0)),
    )
