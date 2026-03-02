"""Application configuration helpers."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Settings:
    """Runtime settings for the backend service."""

    app_name: str = "HearMe AI Backend"
    app_version: str = "0.1.0"
    debug: bool = False


def get_settings() -> Settings:
    """Return static settings placeholder.

    Replace with environment-driven configuration as the project evolves.
    """
    logger.debug("Loading application settings")
    return Settings()
