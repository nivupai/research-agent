# BioResearch Agent

A research assistant for biomedical literature, built on a free stack (Gemini free tier /
local Ollama models, NCBI E-utilities, Europe PMC) and exposed through MCP servers.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
make install            # uv sync + pre-commit hooks
cp .env.example .env    # then fill in your keys (never commit .env)
```

`.env` variables:

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Gemini API key (free tier) |
| `NCBI_API_KEY` | Optional; raises the PubMed rate limit from 3 to 10 req/s |
| `NCBI_EMAIL` | Contact email sent to NCBI with each request |
| `OLLAMA_BASE_URL` | Defaults to `http://localhost:11434` |
| `OLLAMA_MODEL` | Defaults to `qwen3:4b` |

## Literature MCP server

Tools: `search_pubmed`, `fetch_pubmed`, `search_europepmc`. Run it over stdio:

```bash
make serve-literature          # or: uv run python -m bioagent.mcp_servers.literature
```

Example MCP client config:

```json
{
  "mcpServers": {
    "literature": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/research-agent", "bioagent-literature"]
    }
  }
}
```

All paper text returned by the server is marked `untrusted_text: true`: it is retrieved data and
must never be treated as instructions by an agent.

## Development

```bash
make lint        # ruff check, ruff format --check, mypy
make test        # unit tests (network mocked with respx)
make test-live   # tests that hit the real PubMed / Europe PMC APIs
```
