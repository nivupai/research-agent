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
  Offline follow-up: the direct paper ranks 25th (the agent keeps 15; BM25 alone has it 7th,
  the reranker demotes it), and the graph edge rests on one paper, so `min_evidence=2` drops
  it ([failure_analysis_week11.md](../eval/reports/failure_analysis_week11.md)). Fusing the
  reranker order with BM25 (`RERANK_FUSION=true`) lifts the paper to rank 8 offline. Live
  (2026-11-18, after fixing a bug that kept shell settings from reaching the MCP servers) the
  fused AMPK run does retrieve the paper, but the answer still omits sorafenib, so the
  remaining gap is in the candidate selection: sorafenib ranks 41st of 50 mentioned drugs and
  is cut at the 20-candidate limit. Repeated 3x per question, fusion retrieves the paper every
  time but changes no answer measure significantly, so it stays off.
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
* **Root cause** (reproduced offline, `scripts/repro_claim.py`): the cited abstract garbles
  the mechanism; it says MGMT "irreversibly inactivates TMZ", the claim says MGMT itself is
  irreversibly inactivated. NLI reads the swapped roles as a contradiction (0.998), and one
  contradicting window of one cited paper is enough to label the claim contradicted, even
  though the other citation is neutral. The small LoRA verifier calls the same pair SUPPORTED
  (0.998), which is over-generous in the other direction: the passage never says MGMT is
  consumed, so the right label is "not enough info".
* **Status**: not fixed. "Contradicted only if NLI and the LoRA agree" was measured on the
  verifier test split: false contradictions 19 -> 10, but 9 of 45 true contradictions lost
  (accuracy -0.032, p = 0.02), so it was not adopted
  ([failure_analysis_week11.md](../eval/reports/failure_analysis_week11.md)).
* **Lesson**: a source can be wrong; "contradicted by the citation" and "false" are different
  things, and the verifier should flag rather than remove when the evidence is a single window.

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

### 16. The teacher labels some plain contradictions as SUPPORTED (**open**)
* **Symptom**: labelling the verifier spot-check independently, Claude disagreed with the
  Gemini teacher on 11 of 50 pairs, every time with the teacher saying SUPPORTED; four were
  synthetic contradictions such as "Exercise decreased cardiac AMPKalpha2 activity" against a
  passage saying it increased.
* **Root cause**: teacher leniency. Across the dataset the teacher labels 28% of synthetic
  contradiction pairs (negation, number change, role swap) SUPPORTED; DeBERTa NLI calls 16 of
  the 18 such test pairs CONTRADICTED.
* **Fix tried**: reference v2 labels the edited pairs CONTRADICTED by construction; the
  LoRA was retrained on it. Against v2 the retrained adapter catches 16 of the 18 edited
  contradictions (v1: 10) and the router 17, but on real agent claims, whose labels did not
  change, it is worse: 56 vs 65 of 80 correct (p = 0.012), with more NOT_ENOUGH_INFO claims
  called SUPPORTED (16 vs 11 of 21).
* **Status**: open. v1 stays the default adapter; v2 is selectable. Fixing the labels alone
  taught the model the synthetic edit pattern, not better judgement on real claims
  ([failure_analysis_week11.md](../eval/reports/failure_analysis_week11.md), section 8).
* **Lesson**: a model-generated reference needs an independent check before it is used as
  ground truth; a second rater found in an hour what accuracy numbers hid. And after a fix,
  score the slice that matters (real claims) separately: the overall number improved while
  that slice got worse.

## Evaluation

### 11. PubMedQA accuracy 50% (**open**)
* **Symptom**: smoke-suite PubMedQA accuracy 50.0% [22.2, 77.8] vs 91.7% on corpus questions.
* **Root cause** (from the saved records): not retrieval. Every answer cited exactly the
  source paper (gold-PMID recall 1.0), and that abstract contained the decisive results. The
  agent has a yes bias: gold "no" items were right in 1 of 8 runs, gold "yes" in 8 of 10, and
  12 of 18 answers were "Yes". One answer said "Yes" to "are margins and T-stage accurate
  predictors?" while the cited results showed 40% local failure with negative margins and
  P = ns. (An earlier version of this entry guessed "answers from many papers"; the records
  show otherwise.)
* **Status**: open. On 30 held-out items the bias reproduces (56.7% [40.0, 73.3]; gold
  "no" 5 of 12 right, 19 "yes" answers for 12 gold "yes"). A flash synthesizer and a
  results-first prompt rule were tried; neither changed accuracy (both +0.000 on the shared
  items, which were only 19 and 24 because the free quota ran out), so neither was adopted
  ([failure_analysis_week11.md](../eval/reports/failure_analysis_week11.md), section 9).
* **Lesson**: check the hypothesis against the records before writing it down; per-category
  breakdowns matter, the headline number hid a weak category.

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
