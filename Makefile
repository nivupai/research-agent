.PHONY: install lint format test test-live serve-literature serve-chemistry smoke

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

serve-literature:
	uv run python -m bioagent.mcp_servers.literature

serve-chemistry:
	uv run python -m bioagent.mcp_servers.chemistry

smoke:
	uv run python scripts/smoke_literature.py
	uv run python scripts/smoke_chemistry.py
