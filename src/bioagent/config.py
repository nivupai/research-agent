"""Application settings loaded from environment variables and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    # LLMs (see bioagent.llm). Role models are "provider:model"; unset roles use the
    # defaults for LLM_PROVIDER (planner/synth: GEMINI_MODEL, specialists: GEMINI_FAST_MODEL).
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    gemini_model: str = "gemini-3.5-flash"
    gemini_fast_model: str = "gemini-3.5-flash-lite"
    planner_model: str | None = None
    specialist_model: str | None = None
    synth_model: str | None = None
    # Per-specialist overrides of SPECIALIST_MODEL, e.g. LITERATURE_MODEL=gemini:gemini-3.5-flash
    literature_model: str | None = None
    chemistry_model: str | None = None
    analysis_model: str | None = None
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 5

    # Knowledge graph (see bioagent.kg); Neo4j runs from docker-compose.yml
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None
    kg_cache_dir: Path = Path("data/kg")
    neo4j_query_timeout_s: float = 10.0

    # Claim verification (see bioagent.verify); the NLI model needs the `models` extra
    verify_claims: bool = True
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    nli_device: str | None = None  # auto: mps, then cuda, then cpu
    nli_entail_threshold: float = 0.5
    nli_contradiction_threshold: float = 0.5
    verify_policy: Literal["remove", "flag"] = "remove"

    # Agent runs
    checkpoint_db: Path = Path("data/checkpoints.sqlite")
    runs_dir: Path = Path("results/runs")
    mcp_log_dir: Path = Path("data/logs")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
