# API and worker image for the BioResearch Agent (docker compose --profile serve).
# Includes the `models` extra (sentence-transformers/torch): the worker's retrieval server and
# the semantic cache embed with bge-small. Data, indexes, results, adapters and the Hugging
# Face cache are mounted as volumes, never baked into the image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra models --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra models

RUN useradd --create-home --uid 1000 agent && mkdir -p /app/data /app/results /app/.hf_cache \
    && chown -R agent /app
USER agent

EXPOSE 8000 9101
CMD ["uv", "run", "--no-sync", "bioagent", "serve", "--host", "0.0.0.0", "--port", "8000"]
