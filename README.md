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

## Knowledge graph (Neo4j) and the graph MCP server

Gene, Drug, Disease and Paper nodes built from PubTator3 annotations of the corpus, plus
ChEMBL max_phase and mechanisms of action. Neo4j 5 community runs locally from
`docker-compose.yml` (bound to 127.0.0.1; `NEO4J_PASSWORD` in `.env`).

```bash
make graph-up        # start Neo4j and wait until it is healthy
make kg-build        # PubTator3 + NCBI Gene + ChEMBL (all cached in data/kg/), clear and reload
uv run python scripts/build_kg.py --stages load --fresh  # reload from the caches only
uv run python scripts/build_kg.py --stages chembl load --chembl-cache-only --fresh
make serve-graph     # MCP server: find_entity, neighbors, drugs_targeting, paths, evidence
make test-neo4j      # integration tests against the local Neo4j (not run in CI)
```

* **Nodes**: `Entity` + `Gene` (HomoloGene-merged, `symbol`), `Drug` (MeSH chemical,
  `chembl_id`, `max_phase`, `approved` = max_phase >= 4), `Disease` (MeSH), `Paper` (PMID).
  AMPK, PI3K, mTORC1/2 and AKT are alias groups that expand to their subunit genes.
* **Edges**: `MENTIONS` (paper -> entity), PubTator3 relations (`TREATS`, `INHIBITS`,
  `STIMULATES`, `ASSOCIATED_WITH`, ...; up to 50 evidence PMIDs and a count each),
  `CO_OCCURS` (entities in the same abstract in >= 2 papers) and ChEMBL `TARGETS`
  (`action_type`, `mechanism`, `source_db: chembl`).
* **Safety**: loading uses `UNWIND $rows` templates; reads are fixed parameterised
  templates (no raw Cypher) with whitelisted relationship types, at most 3 hops, a
  per-query timeout (`NEO4J_QUERY_TIMEOUT_S`, default 10 s) and capped result sizes.
  Results are marked `untrusted` and every literature edge carries its PMIDs.

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
   `search_europepmc` plus the knowledge-graph tools), chemistry (RDKit/PubChem tools),
   analysis (`run_python`). The graph tools sit with the literature specialist because graph
   edges are mined from the same abstracts it cites: graph claims become evidence on their
   PMIDs (ChEMBL mechanisms are cited as ChEMBL records).
4. **synthesizer** writes the answer with `[n]` citations mapped to evidence passages
   (PMIDs) or computed results; **verifier** checks that every `[n]` resolves, then
   verifies each claim (below).

### Claim verification (`bioagent.verify`)

The synthesizer's sentences are numbered and an LLM (structured output, synthesizer model)
splits them into atomic claims that keep their `[n]` citations. Each claim is labelled:

| label | check |
|---|---|
| `supported` | NLI entailment >= `NLI_ENTAIL_THRESHOLD` (0.5) against a sentence window of a cited passage |
| `contradicted` | NLI contradiction >= `NLI_CONTRADICTION_THRESHOLD` (0.5) and not supported |
| `not_supported` | neither, no resolvable citation, or computed numbers that do not match |
| `computed` | cites an RDKit/sandbox result and every number matches the tool output |
| `absence` | an explicit "nothing was found" claim an LLM judged consistent with all retrieved evidence (weaker; not counted) |

NLI runs locally with `NLI_MODEL` (default
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, ~1.7 GB, needs the `models`
extra; MPS/CUDA/CPU via `NLI_DEVICE`). Passages are split into 3-sentence windows and the
best window counts. With `VERIFY_POLICY=remove` (default) sentences whose claims all fail
are removed and partly failing ones get `[unverified]`; `flag` only marks them. Citations
are renumbered afterwards and the synthesizer's text is kept as `draft_answer`.
Groundedness = (supported + computed) / (supported + computed + contradicted +
not_supported). The CLI prints the claim table; run stats gain `groundedness`, `claims`,
`removed_claims` and `verify_seconds`. `bioagent ask ... --no-verify` turns it off.

All retrieved text is wrapped in `<<<UNTRUSTED_DATA>>>` blocks and the models are told never
to follow instructions inside them. Runs are checkpointed in `data/checkpoints.sqlite`
(thread per run) and saved to `results/runs/<run_id>.json` with tokens, latency and
list-price cost per call.

Models: `LLM_PROVIDER` (gemini|ollama), `GEMINI_MODEL` (planner/synthesizer, default
`gemini-3.5-flash`), `GEMINI_FAST_MODEL` (specialists, default `gemini-3.5-flash-lite`),
`OLLAMA_MODEL`, or per-role `PLANNER_MODEL` / `SPECIALIST_MODEL` / `SYNTH_MODEL` as
`provider:model`; `LITERATURE_MODEL` / `CHEMISTRY_MODEL` / `ANALYSIS_MODEL` override the
specialist model for one kind (ignored with `--provider`).

The planner only adds chemistry/analysis tasks when the question asks for properties, BBB,
drug-likeness or another computation. A literature task with `drug_target` (e.g. "which
drugs act on AMPK") always runs `drugs_targeting`, three corpus searches and a PubMed
search, and passes on the union of graph hits and drugs annotated in the corpus hits. Free-tier 429s are retried with backoff; an exhausted daily quota stops the
run with a clear message and the run can be resumed later.

## Evaluation suite (`bioagent eval`)

```bash
make eval-set        # (re)build eval/questions/v1/ (PubMedQA, generated corpus questions, authored items)
make eval-fixture    # isolated BM25 eval index in data/eval/fixture/ (no models, no Neo4j)
make eval-smoke      # ~25 questions x 2 repeats, flash-lite for every role, + NLI metrics
make eval-full       # every question x 1 repeat; resumable: make eval-full RESUME=<run_id>
uv run bioagent eval compare eval/baselines/smoke.json results/eval/smoke/<run_id>
make eval-calibration RUNS="results/eval/full/<run_id>"   # 40-item judge calibration sheet
make judge-kappa     # Cohen's / weighted kappa (bootstrap CI) once human_label is filled in
```

**Test set** (`eval/questions/v1/*.jsonl`, schema `eval/questions/schema.json`, validated by
`bioagent.eval.dataset`): every item has `id, category, question, scoring_type, gold,
gold_pmids, key_facts`.

| category | n | gold / scoring |
|---|---|---|
| `pubmedqa` | 100 | yes/no/maybe exact match (stratified sample of PubMedQA pqa_labeled) |
| `corpus` | 60 | short answer + key facts, LLM judge; generated by flash-lite from one abstract each, `needs_review: true` (not human-verified) |
| `unanswerable` | 11 | the answer must say the evidence is insufficient (judge) |
| `false_premise` | 11 | the answer must correct the premise (judge, with the correction as notes) |
| `computed` | 6 | RDKit values (gold computed with our RDKit core), numeric tolerance 0.06 |
| `injection` | user-provided | `eval/questions/injection.jsonl` (optional; skipped with a notice when absent) |

Prompt-injection items are not shipped: their poisoned passages and payloads are to be
provided by the maintainer in `eval/questions/injection.jsonl` (same schema; each item's
`injection.passage` uses a fake PMID in the reserved 99xxxxxx range and lists `canaries`,
`forbidden_tools` and `forbidden_pmids`). The scorer reports an attack as successful if a
canary appears in the answer, a forbidden tool is called or a forbidden PMID is cited, and
records whether verification would flag claims citing the poisoned passage. Poisoned passages
only ever enter the eval index, never the main corpus.

PubMedQA (`qiaojin/PubMedQA`, pqa_labeled) is MIT-licensed; its 1000 contexts (abstracts
without the conclusion) are also ingested into the main corpus as source `pubmedqa`.

**Retrieval profile.** CI has no Neo4j, no Chroma index and no models, so every eval run
(locally and in CI) uses the same isolated index: BM25 over the gold abstracts of every item,
450 fixed distractors (`eval/fixtures/distractor_pmids.txt`) and any poisoned passages. Texts
come from the PubMedQA rows, the local corpus or PubMed efetch, so no third-party abstracts
are committed. Live PubMed/Europe PMC, the sandbox and the graph tools are off (`--graph`
turns the graph on locally). This keeps runs comparable and cheap, at the cost of measuring
the agent on an easier, smaller corpus than production.

**Metrics** per question: accuracy (0 / 0.5 / 1), citation support and groundedness (NLI over
the cited passages via `bioagent.verify`, `--nli`), gold-PMID recall in the citations, tool
calls, latency, tokens and cost; plus injection success, abstained and premise corrected.
Reports (`results/eval/<suite>/<run_id>/report.{md,json}`) give means with 95% CIs from a
two-stage bootstrap (questions, then repeats), per-question flip rates across repeats and a
per-category breakdown. `bioagent eval compare` runs a paired bootstrap and a sign-flip
permutation test on shared questions; a regression is a paired-difference CI entirely below
`-margin` (default 0).

**Caching and quota.** Each (question, repeat, agent-config hash) result is stored under the
run directory, so a run stopped by quota exhaustion continues with `--resume <run_id>`. Judge
verdicts are cached in `data/eval/cache.sqlite` by (question, answer hash, judge prompt
version, kind). Judge calibration: `eval/labels/judge_calibration.jsonl` has a stratified
sample of 40 judged answers with an empty `human_label` for a person to fill in.

**CI gate** (`.github/workflows/eval.yml`, opt-in): it builds the eval index, runs the smoke
suite at repeats=2 on flash-lite and fails on a significant accuracy drop against
`eval/baselines/smoke.json`; the tables go to the job summary. One run uses roughly 340
flash-lite calls (most of a free-tier daily quota), so it only runs when started by hand
(Actions > Eval gate > Run workflow) or on a pull request with the `run-eval` label (when the
label is added and on each later push). It is skipped with a notice if the repository secret
`GEMINI_API_KEY` is missing (fork PRs never get it). `NCBI_API_KEY` is optional.

## Fine-tuned small claim verifier (`bioagent.finetune`)

```bash
uv run python scripts/make_verifier_data.py          # pairs, teacher labels, NLI baseline, splits
uv run python scripts/make_verifier_data.py bundle   # data/finetune/verifier_bundle.zip
# full training: notebooks/train_verifier_lora.ipynb on a free Colab/Kaggle T4 (upload the
# bundle, Run all, download verifier-lora.zip), then: unzip verifier-lora.zip -d models/
uv run python scripts/train_verifier_lora.py --steps 150 --grad-checkpointing \
    --batch 2 --accum 4 --max-length 384 --out models/verifier-lora-local   # local fallback
uv run python scripts/eval_verifier.py               # models/verifier-lora, else -local
```

On a 17 GB Apple-silicon Mac the local script only fits with gradient checkpointing (batch
2 x 384 tokens: ~9.5 s per 4 examples, so ~6 h for the notebook's 600 x 16 examples); the T4
notebook (4-bit) is the practical path for the full model.

* **Pairs** (~3.2k, deduped): claims from saved runs/eval answers with their cited passages,
  results sentences paired with their own abstract, same-topic passages from other papers
  (BM25 neighbours), and rule-based perturbations (negation, direction swap, entity swap,
  changed numbers). Splits are by PMID (no paper in two splits); cross-paper pairs only use
  papers of the same split.
* **Labels**: a Gemini teacher (flash-lite, 20 pairs per structured call, cached, resumable,
  stops cleanly when the quota runs out) labels SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO;
  DeBERTa-v3-large NLI is recorded for every pair as a second baseline.
  `eval/labels/verifier_test_human.jsonl` has 150 test pairs with an empty `human_label` for
  spot-labelling (teacher-vs-human agreement); no labels are invented.
* **Model**: Qwen/Qwen3-1.7B (Apache-2.0) with a 3-way classification head and LoRA (r=16) on
  the attention and MLP projections. A classifier rather than label generation: one forward
  pass gives class probabilities (needed by the router) and nothing has to be parsed.
  Adapters go to `models/verifier-lora/` (gitignored) with `training.json`
  (params, steps, time, peak memory). `notebooks/train_verifier_lora.ipynb` trains the same
  model on a free Colab/Kaggle T4 in 4-bit.
* **Backends**: `VERIFIER_BACKEND=nli` (default) | `small_lora` | `llm` | `router`. The router
  uses the small model and escalates pairs whose top probability is below
  `VERIFIER_ROUTER_THRESHOLD` (default 0.75) to `VERIFIER_LLM_MODEL`.
  `scripts/eval_verifier.py` writes the val escalation/accuracy/cost curve to
  `results/verifier/eval.md`; 0.75 is the knee of that curve (about 31% of claims escalated,
  ~3 points below all-LLM accuracy at ~30% of the cost; see `eval/reports/verifier_week7.md`).

## Serving: API, queue, worker, cache and tracing (`bioagent.serve`)

```bash
make serve-up      # docker compose --profile serve: api, worker, redis, jaeger (+ neo4j)
make serve-deps    # or: only redis + jaeger in containers, then in two terminals:
make serve-api     #   uv run bioagent serve   (http://127.0.0.1:8000/docs)
make serve-worker  #   uv run bioagent worker
curl -s -XPOST localhost:8000/v1/ask -H 'content-type: application/json' \
     -d '{"question": "Which approved drugs act on AMPK?"}'
curl -N localhost:8000/v1/jobs/<job_id>/events    # SSE progress, then the final answer
```

* **API** (FastAPI): `POST /v1/ask` returns 202 with a job id (or a completed job straight
  away on a semantic-cache hit); `GET /v1/jobs/{id}` gives status, the result (answer,
  references, claim table, plan) and a per-query cost/latency summary;
  `GET /v1/jobs/{id}/events` streams Server-Sent Events: `queued`, `running`, `progress`
  (one per graph node: plan, each task's result, verification), `retrying`, `final`.
  `/healthz`, `/readyz` (Redis), `/metrics` (Prometheus). Optional `API_KEY` (send
  `X-API-Key`); per-client token bucket (`RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_BURST`) -> 429.
* **Queue and worker**: Redis list + delayed set + per-job event streams. `bioagent worker`
  runs the LangGraph pipeline with one shared MCP tool hub; transient errors are retried
  with exponential backoff (`JOB_MAX_ATTEMPTS`, `JOB_RETRY_BASE_S`) and resume the same run's
  checkpoint; exhausted LLM quota fails the job with a clear message.
* **Semantic cache**: the normalised question is embedded with the corpus embedder
  (bge-small) and compared with cached questions; cosine >= `SEMANTIC_CACHE_THRESHOLD` (0.95)
  returns the cached answer. Entries are scoped by a version hash of the corpus fingerprint,
  models, verifier setup and prompts, expire after `SEMANTIC_CACHE_TTL_S`, and `no_cache: true`
  skips the lookup.
* **Tracing** (`OTLP_ENDPOINT`, Jaeger UI on http://localhost:16686): `api.ask` ->
  `queue.enqueue` -> `worker.job` -> `graph.node.<name>` -> `llm.call` / `mcp.tool`, with model,
  tokens in/out, latency, list-price cost and cache-hit attributes. Metrics: requests, jobs,
  node/LLM latency histograms, tokens, cost, tool calls, cache hits/misses, rate limiting
  (API `/metrics`, worker on `WORKER_METRICS_PORT`).
* The image includes the `models` extra (torch), so it is large; data, indexes, results,
  adapters and the Hugging Face cache are mounted, not copied. The sandbox MCP server needs
  Docker, which the worker container does not have, so `run_python` is unavailable there.

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
