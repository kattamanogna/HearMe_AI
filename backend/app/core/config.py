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
"""Configuration management for HearMe_AI backend."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application environment variables and defaults."""

    app_name: str = "HearMe_AI"
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
