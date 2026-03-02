"""API route declarations."""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Basic health check endpoint."""
    logger.info("Health check requested")
    return {"status": "ok"}
