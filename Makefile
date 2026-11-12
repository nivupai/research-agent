.PHONY: install lint format test test-live test-docker test-models sandbox-image ingest index label-pool eval-retrieval serve-literature serve-chemistry serve-sandbox serve-retrieval smoke

install:
	uv sync --extra models
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-live:
	uv run pytest -m live

test-docker:
	uv run pytest -m docker

test-models:
	uv run pytest -m models

ingest:
	uv run python scripts/ingest_pubmed.py

index:
	uv run python scripts/build_index.py

label-pool:
	uv run python scripts/make_label_pool.py

eval-retrieval:
	uv run python scripts/eval_retrieval.py

sandbox-image:
	docker build -t bioagent-sandbox:latest docker/sandbox

serve-literature:
	uv run python -m bioagent.mcp_servers.literature

serve-chemistry:
	uv run python -m bioagent.mcp_servers.chemistry

serve-sandbox:
	uv run python -m bioagent.mcp_servers.sandbox

serve-retrieval:
	uv run python -m bioagent.mcp_servers.retrieval

smoke:
	uv run python scripts/smoke_literature.py
	uv run python scripts/smoke_chemistry.py
	uv run python scripts/smoke_sandbox.py
	uv run python scripts/smoke_retrieval.py
