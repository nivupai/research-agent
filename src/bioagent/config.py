"""Application settings loaded from environment variables and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # Sandboxed code execution (see bioagent.mcp_servers.sandbox)
    sandbox_image: str = "bioagent-sandbox:latest"
    sandbox_mem: str = "512m"
    sandbox_cpus: float = 1.0
    sandbox_timeout_max: int = 120
    sandbox_pids: int = 128
    docker_host: str | None = None

    # Corpus and retrieval (see bioagent.ingest and bioagent.retrieval)
    corpus_db: Path = Path("data/corpus/corpus.sqlite")
    index_dir: Path = Path("data/indexes")
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embed_device: str | None = None  # auto: mps, then cuda, then cpu
    embed_batch_size: int = 64


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
