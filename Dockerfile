# API and worker image for the BioResearch Agent (docker compose --profile serve, Helm chart).
#   full image (default): includes the `models` extra (sentence-transformers/torch) that the
#     worker's retrieval server and the semantic cache need;
#   slim image: docker build --build-arg UV_EXTRAS= -t bioagent:slim .  (no torch; for the
#     replay pipeline, cluster smoke tests and CI-sized deployments).
# Data, indexes, results, adapters and the Hugging Face cache are mounted, never baked in.
FROM python:3.12-slim

ARG UV_EXTRAS="--extra models"
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf_cache \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev $UV_EXTRAS --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev $UV_EXTRAS && rm -rf /root/.cache

RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/data /app/results /app/models /app/.hf_cache \
    && chown -R agent /app/data /app/results /app/models /app/.hf_cache
USER 10001

EXPOSE 8000 9101
# Runs from the venv directly (no uv at runtime) so a read-only root filesystem works.
CMD ["bioagent", "serve", "--host", "0.0.0.0", "--port", "8000"]
