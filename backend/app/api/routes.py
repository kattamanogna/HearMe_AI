"""API routes for health checks and model inference endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect, status

from app.schemas import (
    ChatMessage,
    ChatStreamChunk,
    ModalityPredictResponse,
    SessionSummaryResponse,
    TextEmotionPredictRequest,
    TextEmotionPredictResponse,
)
from app.services.audio_emotion import analyze_audio_emotion_bytes
from app.services.face_emotion import analyze_face_emotion_bytes
from app.services.chat_response import generate_response
from app.services.session_manager import get_session_summary, store_interaction
from app.services.text_emotion import analyze_text_emotion

router = APIRouter(prefix="/api/v1", tags=["inference"])



def _stream_chunks(text: str, *, size: int = 24) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return [""]
    return [normalized[idx : idx + size] for idx in range(0, len(normalized), size)]


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/session-summary", response_model=SessionSummaryResponse)
def session_summary(session_id: str = Query(default="default")) -> SessionSummaryResponse:
    return SessionSummaryResponse.model_validate(get_session_summary(session_id))


@router.post("/predict-text", response_model=TextEmotionPredictResponse)
def predict_text(payload: TextEmotionPredictRequest) -> TextEmotionPredictResponse:
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


@router.post("/analyze")
async def analyze_multimodal(request: Request) -> dict[str, float | str]:
    content_type = request.headers.get("content-type", "")
    session_id = "frontend-session"
    text_value = ""

    if "application/json" in content_type:
        payload = await request.json()
        text_value = str(payload.get("text", "")).strip()
        session_id = str(payload.get("session_id", "frontend-session")).strip() or "frontend-session"
    else:
        form = await request.form()
        text_value = str(form.get("text", "")).strip()
        session_id = str(form.get("session_id", "frontend-session")).strip() or "frontend-session"

    if not text_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'text' is required and cannot be empty.",
        )

    print("Emotion API request received:", text_value)

    try:
        result = analyze_text_emotion(text_value)
    except Exception:
        result = {"emotion": "neutral", "confidence": 0.0}

    emotion = str(result.get("emotion", "neutral"))
    confidence = float(result.get("confidence", 0.0))

    timestamp = datetime.now(timezone.utc).isoformat()
    store_interaction(
        session_id,
        user_text=text_value,
        emotion=emotion,
        confidence=confidence,
        route="/api/v1/analyze",
        timestamp=timestamp,
    )

    return {"emotion": emotion, "confidence": confidence}


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
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

            timestamp = datetime.now(timezone.utc).isoformat()
            store_interaction(
                session_id,
                user_text=text_value,
                emotion=emotion,
                confidence=float(text_prediction.get("confidence", 0.0)),
                route="/api/v1/ws/chat",
                timestamp=timestamp,
            )

            generated = generate_response(session_id, emotion, text_value)
            chunks = _stream_chunks(str(generated["response_text"]))
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
