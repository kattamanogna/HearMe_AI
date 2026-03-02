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
