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

## Chemistry MCP server

RDKit-based tools: `molecular_properties` (SMILES or a compound name, resolved via PubChem),
`drug_likeness` (Lipinski, Veber, Ghose, PAINS/Brenk alerts, CNS MPO, BBB heuristic),
`similarity` (Morgan2/MACCS fingerprints, Tanimoto/Dice), `substructure_search` (SMARTS) and
`resolve_name` (PubChem PUG-REST, 5 req/s).

```bash
make serve-chemistry           # or: uv run bioagent-chemistry
```

All scores are computed estimates from 2D structure, not experimental evidence. The CNS MPO
score approximates ClogD with cLogP and omits the pKa term (reported out of 5, not 6).

## Sandbox MCP server

`run_python(code, timeout_s=30, files=None)` runs Python (numpy, pandas, scipy, matplotlib,
rdkit) in a fresh, throwaway Docker container. Requires Docker and the image:

```bash
make sandbox-image             # docker build -t bioagent-sandbox:latest docker/sandbox
make serve-sandbox             # or: uv run bioagent-sandbox
```

Each container has no network (`network_mode=none`), a read-only root filesystem with small
tmpfs mounts at `/tmp` (noexec) and `/work`, runs as uid 10001 with all capabilities dropped
and `no-new-privileges`, and is limited to `SANDBOX_MEM` memory (no extra swap),
`SANDBOX_CPUS` CPUs and 128 processes. Code and input files are sent over stdin; nothing from
the host is mounted. Runs are killed after `timeout_s` (capped by `SANDBOX_TIMEOUT_MAX`), and
stdout/stderr are capped at 64 KB each. PNG/CSV files saved to `/work/out` are returned
base64-encoded (up to 5 files of 512 KB). Every run writes a JSON audit line (code SHA-256,
limits, outcome, duration) to the `bioagent.sandbox.audit` logger; the code itself is only
logged at DEBUG.

## Development

```bash
make lint        # ruff check, ruff format --check, mypy
make test        # unit tests (network mocked with respx)
make test-live   # tests that hit the real PubMed / Europe PMC / PubChem APIs
make test-docker # sandbox integration tests (needs Docker + make sandbox-image)
make smoke       # launch each MCP server over stdio and call every tool
```
