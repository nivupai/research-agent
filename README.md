# BioResearch Agent

A research assistant for biomedical literature, built on a free stack (Gemini free tier /
local Ollama models, NCBI E-utilities, Europe PMC) and exposed through MCP servers.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
make install            # uv sync --extra models + pre-commit hooks
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

## Corpus and hybrid retrieval

A local corpus of ~11.5k PubMed abstracts (2000+, abstracts only) on the agent's topics:
AMPK signalling, metformin/biguanides, glioblastoma treatment, oncology drug repurposing,
blood-brain barrier delivery, temozolomide, mTOR/PI3K inhibitors, neurodegeneration drug
targets and pharmacokinetics/pharmacometrics. Queries and per-query caps live in
`configs/corpus.yaml`.

```bash
make ingest          # PubMed -> data/corpus/corpus.sqlite (resumable; ~2 min)
make index           # passages + BM25 + Chroma dense index (incremental; ~2.5 min first run)
make serve-retrieval # MCP server with the search_corpus tool
uv run python scripts/search_corpus.py "temozolomide resistance MGMT" -k 5
```

* **Passages**: title + abstract; abstracts over ~350 tokens are split on sentences into
  overlapping chunks (`passage_id = pmid:chunk_idx`), each prefixed with the title.
* **BM25**: rank-bm25 Okapi with a biomedical tokenizer (keeps `ampk`, `pi3k`, `il-6`,
  `blood-brain` plus its parts), stored as gzipped JSON tokens in `data/indexes/bm25/`.
* **Dense**: sentence-transformers `EMBED_MODEL` (default `BAAI/bge-small-en-v1.5`, query
  instruction prefix applied automatically) on MPS/CUDA/CPU, persisted in Chroma
  (`data/indexes/chroma/`, one collection per model).
* **Hybrid**: reciprocal rank fusion (k=60) of the BM25 and dense top 100;
  `hybrid_rerank` reranks the fused top 50 with `RERANK_MODEL` (default
  `cross-encoder/ms-marco-MiniLM-L-6-v2`; `BAAI/bge-reranker-base` is a heavier
  alternative). Results carry scores and ranks from every stage and `untrusted_text: true`.

Model-dependent code needs the `models` extra (`uv sync --extra models`); CI runs without it.

### Retrieval evaluation

```bash
make label-pool      # pool top-20 per mode for configs/retrieval_queries.yaml (50 queries)
# grade data/labels/pool.jsonl by hand: "grade": 0 / 1 / 2 (leave null to skip)
make eval-retrieval  # recall@10, nDCG@10, MRR, precision@10, latency p50/p95 per mode
```

Results are written to `results/retrieval/` as markdown and JSON. No relevance labels are
shipped; grading is manual.

## Multi-agent pipeline (`bioagent ask`)

```bash
uv run bioagent ask "Which approved drugs act on the AMPK pathway, and is there evidence any could be repurposed for glioblastoma?"
uv run bioagent ask "..." --provider ollama      # local qwen3:4b (ollama pull qwen3:4b)
uv run bioagent ask --resume <run_id>            # continue a failed/interrupted run
uv run bioagent ask "..." --json                 # full run record on stdout
```

A LangGraph graph (`bioagent.agents`):

1. **planner** turns the question into a task DAG (structured output; `literature`,
   `chemistry`, `analysis` tasks with `depends_on`).
2. **dispatch** sends every ready task to its specialist in parallel (Send API).
3. **specialists** are ReAct-style tool loops (max 6 steps) over our MCP servers, each bound
   only to its own tools: literature (`search_corpus`, `search_pubmed`, `fetch_pubmed`,
   `search_europepmc`), chemistry (RDKit/PubChem tools), analysis (`run_python`). A
   `graph_query` scope is reserved for the knowledge graph.
4. **synthesizer** writes the answer with `[n]` citations mapped to evidence passages
   (PMIDs) or computed results; **verifier** checks that every `[n]` resolves and flags
   uncited sentences (NLI verification comes later).

All retrieved text is wrapped in `<<<UNTRUSTED_DATA>>>` blocks and the models are told never
to follow instructions inside them. Runs are checkpointed in `data/checkpoints.sqlite`
(thread per run) and saved to `results/runs/<run_id>.json` with tokens, latency and
list-price cost per call.

Models: `LLM_PROVIDER` (gemini|ollama), `GEMINI_MODEL` (planner/synthesizer, default
`gemini-3.5-flash`), `GEMINI_FAST_MODEL` (specialists, default `gemini-3.5-flash-lite`),
`OLLAMA_MODEL`, or per-role `PLANNER_MODEL` / `SPECIALIST_MODEL` / `SYNTH_MODEL` as
`provider:model`. Free-tier 429s are retried with backoff; an exhausted daily quota stops the
run with a clear message and the run can be resumed later.

## Development

```bash
make lint        # ruff check, ruff format --check, mypy
make test        # unit tests (network mocked with respx)
make test-live   # tests that hit the real PubMed / Europe PMC / PubChem APIs
make test-docker # sandbox integration tests (needs Docker + make sandbox-image)
make test-models # tests that load the real embedding / reranking models
make test-llm    # tests that call Gemini / Ollama
make smoke       # launch each MCP server over stdio and call every tool
```
