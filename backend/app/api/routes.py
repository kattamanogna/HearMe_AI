"""API routes for health checks and model inference endpoints."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status

from app.schemas import (
    ChatMessage,
    ChatStreamChunk,
    ModalityPredictResponse,
    MultimodalRequest,
    MultimodalResponse,
    TextEmotionPredictRequest,
    TextEmotionPredictResponse,
)
from app.services.audio_emotion import analyze_audio_emotion_bytes
from app.services.face_emotion import analyze_face_emotion_bytes
from app.services.chat_response import generate_response
from app.services.fusion_engine import combine_predictions
from app.services.history import get_chat_history, store_interaction
from app.services.text_emotion import analyze_text_emotion

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


def _stream_chunks(text: str, *, size: int = 24) -> list[str]:
    """Split generated text into deterministic chunks for streaming."""

    normalized = text.strip()
    if not normalized:
        return [""]
    return [normalized[idx : idx + size] for idx in range(0, len(normalized), size)]


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

    prediction = analyze_text_emotion(payload.text.strip())
    return TextEmotionPredictResponse(
        emotion=str(prediction.get("emotion", "neutral")),
        confidence=float(prediction.get("confidence", 0.0)),
        probabilities={str(k): float(v) for k, v in dict(prediction.get("probabilities", {})).items()},
    )


@router.post("/predict-audio", response_model=ModalityPredictResponse)
async def predict_audio(file: UploadFile = File(...)) -> ModalityPredictResponse:
    """Predict emotion from an uploaded audio file (multipart/form-data)."""

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    prediction = analyze_audio_emotion_bytes(audio_bytes)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to process uploaded audio file.",
        )

    return ModalityPredictResponse(
        emotion=str(prediction.get("emotion", "neutral")),
        confidence=float(prediction.get("confidence", 0.0)),
        probabilities={str(k): float(v) for k, v in dict(prediction.get("probabilities", {})).items()},
        face_detected=prediction.get("face_detected"),
    )


@router.post("/predict-face", response_model=ModalityPredictResponse)
async def predict_face(file: UploadFile = File(...)) -> ModalityPredictResponse:
    """Predict emotion from an uploaded face image file (multipart/form-data)."""

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    prediction = analyze_face_emotion_bytes(image_bytes)
    return ModalityPredictResponse(
        emotion=str(prediction.get("emotion", "neutral")),
        confidence=float(prediction.get("confidence", 0.0)),
        probabilities={str(k): float(v) for k, v in dict(prediction.get("probabilities", {})).items()},
        face_detected=bool(prediction.get("face_detected", False)),
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

    session_id = payload.session_id.strip() or "default"
    text_value = payload.text.strip()
    if not text_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'text' is required and cannot be empty.",
        )

    text_prediction = analyze_text_emotion(text_value)

    audio_prediction = None
    if payload.audio_bytes:
        audio_bytes = _decode_base64_payload(payload.audio_bytes, "audio_bytes")
        audio_prediction = analyze_audio_emotion_bytes(audio_bytes)

    face_prediction = None
    if payload.face_base64:
        image_bytes = _decode_base64_payload(payload.face_base64, "face_base64")
        face_prediction = analyze_face_emotion_bytes(image_bytes)

    fused = combine_predictions(text_prediction, audio_prediction, face_prediction)

    interaction = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "route": "/api/v1/analyze",
        "text": text_value,
        "fused_emotion": str(fused.get("emotion", "neutral")),
    }
    store_interaction(session_id, interaction)
    chat_history = get_chat_history(session_id)
    response_text = generate_response(str(fused.get("emotion", "neutral")), text_value)

    return MultimodalResponse(
        text_emotion=str(text_prediction.get("emotion", "neutral")),
        text_confidence=float(text_prediction.get("confidence", 0.0)),
        text_probabilities={str(k): float(v) for k, v in dict(text_prediction.get("probabilities", {})).items()},
        audio_emotion=(str(audio_prediction.get("emotion", "neutral")) if audio_prediction else None),
        audio_confidence=(float(audio_prediction.get("confidence", 0.0)) if audio_prediction else None),
        audio_probabilities=({str(k): float(v) for k, v in dict(audio_prediction.get("probabilities", {})).items()} if audio_prediction else None),
        face_emotion=(str(face_prediction.get("emotion", "neutral")) if face_prediction else None),
        face_confidence=(float(face_prediction.get("confidence", 0.0)) if face_prediction else None),
        face_probabilities=({str(k): float(v) for k, v in dict(face_prediction.get("probabilities", {})).items()} if face_prediction else None),
        fused_emotion=str(fused.get("emotion", "neutral")),
        fused_confidence=float(fused.get("confidence", 0.0)),
        fused_probabilities={str(k): float(v) for k, v in dict(fused.get("probabilities", {})).items()},
        chat_history=[{str(key): str(value) for key, value in item.items()} for item in chat_history],
        response_text=response_text,
    )


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Realtime chat endpoint that streams assistant responses over WebSocket."""

    await websocket.accept()
    try:
        while True:
            payload = ChatMessage.model_validate_json(await websocket.receive_text())
            session_id = payload.session_id.strip() or "default"
            text_value = payload.text.strip()
            if not text_value:
                error_chunk = ChatStreamChunk(
                    session_id=session_id,
                    chunk="'text' is required and cannot be empty.",
                    done=True,
                )
                await websocket.send_json(error_chunk.model_dump())
                continue

            text_prediction = analyze_text_emotion(text_value)
            emotion = str(text_prediction.get("emotion", "neutral"))
            response_text = generate_response(emotion, text_value)

            store_interaction(
                session_id,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "route": "/api/v1/ws/chat",
                    "text": text_value,
                    "fused_emotion": emotion,
                },
            )

            chunks = _stream_chunks(response_text)
            for index, chunk in enumerate(chunks):
                stream_chunk = ChatStreamChunk(
                    session_id=session_id,
                    chunk=chunk,
                    done=index == len(chunks) - 1,
                )
                await websocket.send_json(stream_chunk.model_dump())
    except WebSocketDisconnect:
        return
    except ValueError:
        await websocket.send_json(
            ChatStreamChunk(
                session_id="default",
                chunk="Invalid payload. Expected JSON with fields: session_id, text.",
                done=True,
            ).model_dump()
        )
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
