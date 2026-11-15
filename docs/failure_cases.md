# Failure cases and fixes

Real problems found while building and running the agent, from the commit history and the
eval reports. Each entry: what we saw, why it happened, what changed (commit), and the lesson.
Open issues are marked **open**.

## Retrieval and evidence

### 1. Evidence-budget starvation: the repurposing task's papers never reached the answer
* **Symptom**: a repurposing run called the right graph paths and found the primary
  glioblastoma study (PMID 29334602), yet the final answer did not cite it.
* **Root cause**: `select_evidence` filled the synthesizer's 40-item budget in plan order. The
  first task (drug discovery, 59 items) took 31 slots, computed results 8, and the repurposing
  task (63 items) got 1.
* **Fix**: round-robin the budget across tasks, alternating full abstracts with graph items
  within a task (`d8d5038`). The next run cited 29334602, 25867026 and two more primary GBM
  studies.
* **Lesson**: the bug only showed up in a live multi-task run; aggregate metrics looked fine.
  Look at which evidence actually reaches the final prompt, not only what was retrieved.

### 2. SQLite connection used across threads by the retrieval server
* **Symptom**: the retrieval MCP server runs `Retriever.search` in worker threads (to keep the
  event loop free); SQLite refuses a connection created in another thread by default.
* **Root cause**: one shared `sqlite3` connection plus thread offloading.
* **Fix**: `check_same_thread=False` with all access serialised by the retriever's lock
  (`e8207ac`, relied on by the retrieval server in `77a70cc`; see `CorpusStore.__init__`).
* **Lesson**: moving blocking work to threads changes the concurrency contract of every
  resource it touches; make the locking explicit next to the resource.

### 3. Lucene operator injection through entity names
* **Symptom**: a fulltext entity search for a name containing a bare `OR`/`AND`/`NOT` turned
  into a boolean query (found by a unit test of the escaping).
* **Root cause**: escaping special characters is not enough; upper-case operator words are
  syntax in Lucene.
* **Fix**: lower-case operator words before building the query (`214ac67`).
* **Lesson**: treat every query language like SQL: user (and model) text needs real
  escaping, and tests should include the language's keywords.

### 4. Duplicate keys silently merged in the knowledge graph
* **Symptom**: the first full load stored 19 fewer MENTIONS and 15 fewer TARGETS edges than
  the builder planned.
* **Root cause**: after folding NCBI-only gene ids into HomoloGene nodes, two rows shared a
  (paper, gene) key, and the same drug-gene-action from two ChEMBL targets shared an edge id;
  MERGE collapsed them.
* **Fix**: aggregate mention counts after canonicalisation and include the ChEMBL target id
  in TARGETS edge ids (`214ac67`). Planned and stored counts then matched exactly.
* **Lesson**: compare "planned" vs "in the database" counts after every bulk load.

### 5. The AMPK answer misses drug classes (sorafenib still **open**)
* **Symptom**: with the graph tools, the AMPK answer listed metformin, phenformin, berberine
  and rosiglitazone but dropped statins, aspirin/salicylate and sorafenib that a plain
  literature run had found.
* **Root cause**: the graph only knows drugs PubTator linked to AMPK subunits, and ChEMBL
  curates metformin as a complex I inhibitor, not an AMPK activator; the flash-lite
  specialist rarely ran the extra corpus searches the prompt asked for.
* **Fix**: make drug discovery deterministic: always run the graph lookup, three corpus
  searches and a PubMed search, and merge the drugs annotated in the hits (`4adb2f1`).
  TZDs, salicylates and aspirin came back; sorafenib still ranks too low to be retrieved.
* **Lesson**: do not rely on a small model to follow a multi-step search recipe; run the
  recipe in code and give the model the union.

## Verification

### 6. Same-paper chunks: correct claims marked unsupported
* **Symptom**: "metformin inhibits glioblastoma cell proliferation" scored 0.02 entailment
  against PMID 25867026, a paper that says exactly that.
* **Root cause**: the citation pointed at chunk 1 of the abstract (the chunk the synthesizer
  saw); the supporting sentence was in chunk 0, also retrieved.
* **Fix**: a citation covers every retrieved passage of the cited paper; only the 8
  best-overlapping windows are scored (`8309ae7`). AMPK groundedness went from 0.62 to 0.69.
* **Lesson**: verify at the granularity readers see (the paper), not the retrieval unit.

### 7. Computed claims sent to NLI against JSON
* **Symptom**: "both pass Lipinski's rule of five with zero violations" came out
  *contradicted*.
* **Root cause**: claims without digits fell back to NLI against the RDKit tool's JSON, which
  NLI cannot read; "zero" was not recognised as a number.
* **Fix**: rule pass/fail statements are checked against the tool's `passed` fields, number
  words count as numbers, and tool JSON never goes to NLI (`8309ae7`).
* **Lesson**: use the cheapest exact check a claim allows; NLI is for prose.

### 8. MGMT "irreversible suicide inactivation" flagged as contradicted (**open**)
* **Symptom**: a textbook-correct claim scored contradiction 1.00 against its cited abstract
  (see the example answer in the README).
* **Root cause**: NLI false positive: the abstract describes MGMT repair without saying the
  enzyme is consumed, and the model reads the extra detail as a contradiction.
* **Status**: not fixed. The verifier is deliberately strict; claims supported only by
  background knowledge get removed or flagged. Human labels on
  `eval/labels/verifier_test_human.jsonl` would measure how often this happens.
* **Lesson**: report verifier errors alongside its wins; "removed" does not mean "wrong".

### 9. Silent NOT_ENOUGH_INFO when the LLM quota ran out
* **Symptom**: a router re-score started while the Gemini quota was exhausted would have
  labelled every escalated claim NOT_ENOUGH_INFO without any error.
* **Root cause**: the LLM backend returned a default label for pairs it could not get a label
  for.
* **Fix**: raise `LLMVerifierUnavailableError` instead (`cc134db`).
* **Lesson**: never substitute a plausible default for a missing model output; fail loudly.

### 10. The LLM verifier reusing the teacher's training labels
* **Symptom** (caught in design review, before it happened): the "Gemini as verifier" row
  would have scored 100% against the teacher.
* **Root cause**: the same model and pair would hit the teacher's cache entry.
* **Fix**: inference-time LLM labels use their own cache namespace (`2c131a7`).
* **Lesson**: when a model is both the reference and a contestant, check the caches.

## Evaluation

### 11. PubMedQA accuracy 50% (**open**)
* **Symptom**: smoke-suite PubMedQA accuracy 50.0% [22.2, 77.8] vs 91.7% on corpus questions.
* **Root cause** (likely): the yes/no/maybe items need a judgement from one abstract without
  its conclusion; the agent searches broadly, answers from many papers, and often leads with
  the wrong label. Two of the first trial answers led with "Yes"/"Maybe" against gold
  "no"/"yes".
* **Status**: open; the full eval has not reached the PubMedQA items yet.
* **Lesson**: per-category breakdowns matter; the headline number hid a weak category.

## Infrastructure and operations

### 12. KEDA could not reach Redis (short service name)
* **Symptom**: on kind the ScaledObject stayed InProgress: `lookup bioagent-bioagent-redis
  ... no such host`.
* **Root cause**: the KEDA operator runs in the `keda` namespace and resolves short names
  there.
* **Fix**: use `<svc>.<namespace>.svc.cluster.local`; also pace scale-down with an explicit
  HPA stabilisation window, since `cooldownPeriod` only applies when scaling to zero
  (`ef0d6e7`).
* **Lesson**: `helm lint` and schema validation passed; only a real deploy found it.

### 13. mypy failing in CI on an optional dependency
* **Symptom**: CI's plain `uv sync` (no `models` extra) failed mypy on the `peft` import.
* **Root cause**: the import lives in an optional extra that CI does not install.
* **Fix**: a mypy override for `peft` (`9a7a266`); later serving dependencies were made main
  dependencies so CI type-checks them for real.
* **Lesson**: type-check in the same environment CI builds, or decide per dependency whether
  it is optional.

### 14. Scheduled overnight jobs never ran (Mac sleep)
* **Symptom**: two overnight jobs scheduled with `sleep` until the 08:00 UTC quota reset were
  still waiting the next evening.
* **Root cause**: macOS does not advance `sleep` timers while the system sleeps, and
  `caffeinate -i` does not stop sleep on a closed lid.
* **Fix**: none in code; the jobs were run by hand the next day. Long or timed jobs belong
  on an always-on machine (or the cloud), or under `launchd` with a wake schedule.
* **Lesson**: a laptop is not a scheduler.

### 15. Disk and swap limits on a 17 GB Mac
* **Symptom**: local LoRA training swapped to 15 GB at batch 4 x 512 tokens; later a kind
  cluster pull filled the disk to 100% and crashed Docker Desktop; model-heavy evals pushed
  swap to 8 GB.
* **Root cause**: a 1.7B model without gradient checkpointing, a torch-sized image, and a
  nearly full disk where swap and Docker's disk image grow.
* **Fix**: gradient checkpointing for local training and the free T4 notebook for the full
  run (`2c131a7`, `8cd2469`); a slim image without torch for cluster demos (`ef04728`);
  pruning the Docker build cache and a restart (clears swap) before the kind demo.
* **Lesson**: check free disk and memory before each heavy step, and abort below a floor
  rather than discovering the limit by crashing.
