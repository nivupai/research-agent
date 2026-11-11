.PHONY: install lint format test test-live test-docker sandbox-image serve-literature serve-chemistry serve-sandbox smoke

install:
	uv sync
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

sandbox-image:
	docker build -t bioagent-sandbox:latest docker/sandbox

serve-literature:
	uv run python -m bioagent.mcp_servers.literature

serve-chemistry:
	uv run python -m bioagent.mcp_servers.chemistry

serve-sandbox:
	uv run python -m bioagent.mcp_servers.sandbox

smoke:
	uv run python scripts/smoke_literature.py
	uv run python scripts/smoke_chemistry.py
	uv run python scripts/smoke_sandbox.py
