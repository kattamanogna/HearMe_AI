"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI

from app.api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="HearMe AI Backend", version="0.1.0")
app.include_router(api_router)


@app.on_event("startup")
async def on_startup() -> None:
    """Log when the API service boots."""
    logger.info("Starting HearMe AI backend service")


@app.get("/")
async def root() -> dict[str, str]:
    """Simple root endpoint for quick checks."""
    return {"message": "HearMe AI backend is running"}
"""FastAPI entrypoint for HearMe_AI backend."""

from fastapi import FastAPI

from app.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="HearMe_AI",
        description="Multimodal mental health AI chatbot backend.",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()
