"""Application settings loaded from environment variables and ``.env``."""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets are wrapped in ``SecretStr`` so they never print."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: SecretStr | None = None
    ncbi_api_key: SecretStr | None = None
    ncbi_email: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
