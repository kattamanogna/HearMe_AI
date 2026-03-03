"""FastAPI application entrypoint for HearMe_AI backend."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request

from app.api.routes import router
from app.services.audio_emotion import warmup_audio_model
from app.services.face_emotion import warmup_face_model
from app.services.text_emotion import warmup_text_model
from app.services.chat_response import warmup_response_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="HearMe_AI",
        description="Multimodal mental health AI chatbot backend.",
        version="0.1.0",
    )

    @app.middleware("http")
    async def log_api_calls(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Log each API request with timestamp and route path."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("API call | ts=%s | method=%s | route=%s", timestamp, request.method, request.url.path)
        response = await call_next(request)
        return response

    @app.on_event("startup")
    async def warmup_models() -> None:
        """Preload model caches for lower latency on first inference."""

        warmup_text_model()
        warmup_audio_model()
        warmup_face_model()
        warmup_response_generator()

    app.include_router(router)
    return app


app = create_app()
