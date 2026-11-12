.PHONY: install lint format test test-live test-docker test-models test-llm ask sandbox-image ingest index label-pool eval-retrieval serve-literature serve-chemistry serve-sandbox serve-retrieval serve-graph graph-up graph-down kg-build test-neo4j eval-set eval-fixture eval-smoke eval-full eval-calibration judge-kappa smoke

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

test-llm:
	uv run pytest -m llm

# make ask Q="Which approved drugs act on the AMPK pathway?" [PROVIDER=ollama]
ask:
	uv run bioagent ask "$(Q)" $(if $(PROVIDER),--provider $(PROVIDER))

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

serve-graph:
	uv run python -m bioagent.mcp_servers.graph

# Neo4j 5 community on localhost (password from NEO4J_PASSWORD in .env)
graph-up:
	docker compose up -d --wait neo4j

graph-down:
	docker compose stop neo4j

# Fetch PubTator3 + gene names + ChEMBL (all cached under data/kg) and reload the graph
kg-build:
	uv run python scripts/build_kg.py --fresh

test-neo4j:
	uv run pytest -m neo4j

# Evaluation suite (see eval/ and "Evaluation suite" in the README)
eval-set:
	uv run python scripts/build_eval_set.py

eval-fixture:
	uv run python scripts/build_eval_fixture.py

# make eval-smoke [REPEATS=2]  /  make eval-full [RESUME=<run_id>]
eval-smoke: eval-fixture
	uv run bioagent eval run --suite smoke --repeats $(or $(REPEATS),2) --nli $(if $(RESUME),--resume $(RESUME))

eval-full: eval-fixture
	uv run bioagent eval run --suite full --repeats $(or $(REPEATS),1) --nli $(if $(RESUME),--resume $(RESUME))

# make eval-calibration RUNS="results/eval/full/<id> results/eval/smoke/<id>"
eval-calibration:
	uv run bioagent eval calibration $(RUNS)

judge-kappa:
	uv run bioagent eval kappa eval/labels/judge_calibration.jsonl

smoke:
	uv run python scripts/smoke_literature.py
	uv run python scripts/smoke_chemistry.py
	uv run python scripts/smoke_sandbox.py
	uv run python scripts/smoke_retrieval.py
	uv run python scripts/smoke_graph.py
