# Week 11: offline failure analysis and Claude-graded retrieval (2026-11-15)

Everything here ran locally with no Gemini calls: the small LoRA verifier on Apple MPS, the
DeBERTa NLI scores already stored on every verifier pair, the retrieval stack, and Neo4j.
Reproduce with the scripts named in each section.

## 1. Retrieval: hybrid + rerank vs embeddings-only (Claude-graded relevance)

(Rubric v1 numbers. Section 6 re-grades the sample under the owner's criterion "would this
abstract help answer the query"; the conclusion is the same.)

**Who labelled what**: relevance (0/1/2) for the 299 pooled items of an 8-query sample
(`make label-sample`: whole queries, stratified by query type) was graded by Claude, reading
each query and abstract, rubric v1 (2 = directly addresses the query's specific mechanism or
relation; 1 = related, partly useful; 0 = off-topic or the term appears only in background).
Grades are in `eval/labels/retrieval_claude.jsonl` (`labeler: "claude"`); no human grade field
was written. These are **not human labels**; the 50-item human spot-check below measures how
far they can be trusted.

`uv run python scripts/eval_retrieval.py --labels claude` (8 queries; 95% bootstrap CIs over
queries; relevant = grade >= 1). Grades: 73 x 2, 90 x 1, 136 x 0.

| mode | recall@10 | nDCG@10 | MRR | precision@10 |
|---|---|---|---|---|
| BM25 | 0.374 [0.314, 0.441] | 0.625 [0.485, 0.767] | 0.938 [0.812, 1.000] | 0.738 [0.562, 0.912] |
| dense (embeddings only) | 0.422 [0.319, 0.547] | 0.785 [0.689, 0.861] | 0.938 [0.812, 1.000] | 0.762 [0.637, 0.887] |
| hybrid (RRF) | 0.371 [0.317, 0.432] | 0.706 [0.599, 0.810] | 0.938 [0.812, 1.000] | 0.725 [0.550, 0.888] |
| hybrid + rerank | 0.440 [0.359, 0.529] | 0.753 [0.631, 0.863] | 1.000 [1.000, 1.000] | 0.812 [0.700, 0.925] |

Paired vs dense (delta [95% CI], sign-flip p): hybrid + rerank recall@10 +0.018
[-0.025, +0.060], p = 0.50; nDCG@10 -0.032 [-0.147, +0.092], p = 0.63; precision@10 +0.050
[-0.013, +0.113], p = 0.38. BM25 nDCG@10 -0.160 [-0.296, -0.039] (p = 0.06).

Reading: on these 8 queries, hybrid + rerank and embeddings-only are indistinguishable; the
reranker adds ~220 ms per query. Recall@10 is capped at 0.576 on average because several
queries have far more than 10 relevant papers (q25: 40 of 40 pooled papers relevant), so
nDCG@10 and precision@10 are the more informative metrics here. 8 queries are enough for the
paired comparison to about +/-0.10 (see `make label-plan`), not to separate modes that differ
by 0.02.

## 2. Small verifier: NOT_ENOUGH_INFO pairs called SUPPORTED

`scripts/verifier_probs.py` (caches the LoRA's class probabilities for val and test) then
`scripts/analyze_verifier_errors.py`. Reference = Gemini teacher labels. Re-run test
accuracy 0.833 [0.786, 0.877], macro-F1 0.780 (the Week 7 report has 0.837; one pair flips
between MPS runs). NOT_ENOUGH_INFO -> SUPPORTED: **13 of 38** NOT_ENOUGH_INFO test pairs.

**Where**: 11 of the 13 are `run` pairs (claims from real agent answers): the model calls 11 of
the 21 `run` NOT_ENOUGH_INFO pairs SUPPORTED, but only 2 of 16 synthetic `cross` pairs. The
synthetic negatives are learned; real near-miss claims are not. Training data: 2,716 pairs
(56% SUPPORTED, 27% NOT_ENOUGH_INFO, 17% CONTRADICTED), only 531 of them `run` pairs.

**What** (read one by one):

| category | n | example |
|---|---|---|
| multi-part claim; the passage supports the topic or one part, not the specific detail | 6 | "Phenformin ... stimulates AMPK-alpha1 and inhibits complex I" vs an abstract that only says phenformin activates AMPK |
| passage is a knowledge-graph relation summary, claim is a mechanism | 3 | "Rosiglitazone ... increases the AMP:ATP ratio" vs "Rosiglitazone positively correlates PRKAA2 (text-mined)" |
| generalisation from a study's topic | 2 | "Clinicians fail to track meconium ..." vs a cohort study of delayed ARM diagnosis |
| teacher label looks wrong (passage entails the claim; NLI entailment 0.99) | 2 | "It binds the AMPK complex ... protecting Thr-172 from dephosphorylation" vs the A-769662 abstract saying exactly that |

These errors are confident: mean top probability 0.874, and only 3 of the 13 fall below the
router threshold (0.75), so routing does not catch them.

**Fixes that need no retraining** (tuned on val only, scored once on test against the
unchanged model; paired bootstrap CI, sign-flip p):

| fix | val-tuned setting | test accuracy | test macro-F1 | NEI->SUPPORTED |
|---|---|---|---|---|
| none | - | 0.833 | 0.780 | 13 |
| class bias on NOT_ENOUGH_INFO (log-prior offset) | -0.5 | 0.841 (+0.008 [0.000, +0.020], p = 0.50) | 0.792 (+0.011 [0.000, +0.027]) | 13 |
| NLI gate (SUPPORTED with NLI entailment < t -> NEI) | t = 0 (no-op) | 0.833 | 0.780 | 13 |
| temperature scaling | T = 1.85 | unchanged | unchanged | 13 |

Val contains no NOT_ENOUGH_INFO -> SUPPORTED errors (28 NEI pairs, all right), so nothing
tuned on val can target them. Diagnostic curves on test (not used for tuning) show why: a
NEI offset of +1.0 fixes 3 and breaks 3 SUPPORTED pairs; an NLI gate at t = 0.05 fixes 8 and
breaks 28 (DeBERTa under-entails correct paraphrases). Temperature scaling does improve
calibration (test ECE 0.124 -> 0.053), but raises escalation at 0.75 from 9% to 17% of claims
(about double the LLM cost) without changing these errors. **No cheap fix is worth shipping.**

**Proposed (needs retraining; not run)**: (1) add `run`-kind NOT_ENOUGH_INFO training pairs:
real agent claims paired with a related but non-supporting passage from the same run (hard
negatives), and weight the loss by class; (2) verify graph-relation evidence against the
underlying abstract rather than the relation summary; (3) split multi-part claims before
verification; (4) the human spot-check (50 verifier pairs) will tell how many of these 13 are
teacher errors (2 look like it).

## 3. MGMT "irreversible suicide inactivation" false contradiction

`uv run python scripts/repro_claim.py 20261112-083229-fd26dc 8` (all local backends):

| backend | vs PMID 24157870 | vs PMID 24990698 | claim label |
|---|---|---|---|
| DeBERTa NLI (pipeline default) | max contradiction 0.017 | contradiction **0.998** on the window "MGMT ... repairs the DNA O(6)-methylation damage ... and irreversibly inactivates TMZ" | contradicted |
| small LoRA (whole passage) | SUPPORTED 0.920 | SUPPORTED 0.998 | supported |
| router at 0.75 | keeps LoRA (no escalation) | keeps LoRA | supported |

**Root cause (corrected)**: not a missing detail. The cited abstract itself garbles the
mechanism: it says MGMT "irreversibly inactivates TMZ", whereas the claim says MGMT is
irreversibly inactivated. NLI reads the swapped roles as a contradiction, and the pipeline
labels a claim contradicted if any window of any cited paper crosses 0.5, even when the other
citation is neutral. The LoRA calls it SUPPORTED with near certainty, which removes the false
contradiction but is the over-generous behaviour of section 2 (the passage does not say MGMT
is consumed). The right label against these passages is NOT_ENOUGH_INFO (flag, not remove).

**Tried offline**: "contradicted only if NLI and the LoRA agree" on the verifier test split
cuts NLI's false contradictions from 19 to 10 but loses 9 of 45 true ones (accuracy
-0.032 [-0.056, -0.008], p = 0.02). Not adopted. The case stays open; the verifier spot-check
labels are the way to measure how often each backend is wrong in each direction.

## 4. Prepared runs that need Gemini (after the 08:00 UTC quota reset)

### PubMedQA (smoke: 50.0% [22.2, 77.8])

What the saved smoke records show (9 items x 2 repeats, flash-lite in every role): every
answer cites exactly one paper, the source paper, and gold-PMID recall is 1.0, so retrieval
is not the problem (the earlier "answers from many papers" hypothesis in
`docs/failure_cases.md` was wrong). The errors are a **yes bias**: gold "no" items were
answered correctly in 1 of 8 runs, gold "yes" in 8 of 10; 12 of 18 answers were "Yes". In
pubmedqa-11713724 the passage the agent cited reports a 40% local failure rate even with
negative margins and "P = ns", yet the answer was "Yes, margins and T-stage are recognized
factors".

Hypotheses: (H1) flash-lite restates the question affirmatively instead of reading the
results; (H2) the synthesizer prompt asks for a lead verdict before the evidence is weighed;
(H3) "maybe" vs "no" boundary: hedged results become "maybe".

| run | command | measures | Gemini calls (approx.) |
|---|---|---|---|
| P1 baseline | `make eval-full RESUME=full-v1` (finishes the 100 PubMedQA items, flash-lite) | PubMedQA accuracy with a usable CI, confusion by gold label | ~750-1,350 for the remaining 123 items (6-11 per question) |
| P2 model | `uv run bioagent eval run --suite pubmedqa --synth gemini:gemini-3.5-flash --run-id pubmedqa-flash-synth --nli` | H1: does the larger synthesizer remove the yes bias? paired vs P1 with `eval compare` | ~600-1,100 |
| P3 prompt | same as P2 with flash-lite after a prompt change that asks for the label only after summarising the results and their significance | H2 | ~600-1,100 |

`--suite pubmedqa` and `--synth` were added for this (both offline-tested). P2/P3 can be run
on a slice first with `--limit 30`.

### AMPK / sorafenib

Offline findings (`scripts/discovery_recall.py --target AMPK --drug sorafenib`, Neo4j):

* Corpus: 3 papers mention sorafenib and AMPK. The direct one (28625738, "paradoxical
  activation of AMPK by ... sorafenib") ranks 25th for "AMPK inhibitor approved drug" with
  hybrid + rerank and is outside the top 100 for the two "activator" queries; the agent keeps
  the top 15. BM25 alone ranks it 7th (the reranker demotes it). A query that names the drug
  class finds it: "kinase inhibitor drug activates AMPK" -> rank 2; "paradoxical activation of
  AMPK by drugs" -> rank 1.
* Graph: sorafenib -> PRKAA2 POSITIVELY_CORRELATES exists but from one paper (25080865), so
  `drugs_targeting`'s default `min_evidence=2` drops it; with 1 it would rank 47th of 74
  approved drugs by evidence, past the 20-drug limit.

So sorafenib is a long-tail case on both routes. Candidate fixes and the runs to judge them:

| run | change | measures |
|---|---|---|
| A1 | add a class-agnostic query "paradoxical activation of {target} by approved drugs" and take BM25 hits before rerank into the drug-mention merge | offline first: rank of known AMPK-activating drugs' papers; then one live AMPK question (~10 calls) |
| A2 | `min_evidence=1` for drug discovery, ranked by relation type before evidence count | candidate list change offline; then one live AMPK question |
| A3 | both, 2 repeats each | does sorafenib reach the answer, and does precision of the drug list drop? |

## 5. Human spot-check (replaces labelling the full sheets)

`make label-spotcheck` serves 140 items, about 1.5 hours:

| sheet | items | agreement measured | est. time |
|---|---|---|---|
| retrieval | 50 from the Claude-graded sample (proportional by query; Claude's grade not shown or stored there) | human vs Claude, weighted kappa | ~25 min |
| verifier | 50 of the 150 test pairs (proportional by pair kind) | human vs Gemini teacher | ~21 min |
| judge | the 40 calibration answers (judge label hidden) | human vs Gemini judge (`bioagent eval kappa`) | ~40 min |

Then `make label-agreement`. Expected precision with these sizes: verifier accuracy +/-0.10,
judge kappa about +/-0.22 (`make label-plan`).

## 6. Claude as an independent second rater (inter-model agreement, not human validation)

At the project owner's request Claude labelled the spot-check items itself, blind to the
Gemini labels, reading each item. Claude's labels live in separate files
(`labeler: "claude"`): `eval/labels/verifier_test_claude.jsonl`,
`eval/labels/judge_calibration_claude.jsonl`, `eval/labels/spotcheck_retrieval_claude_v2.jsonl`,
`eval/labels/retrieval_claude_v2.jsonl`. No human label field was written. **Everything in
this section is agreement between two models; none of it is human validation**, which is
still pending (`make label-spotcheck`). `bioagent label agreement` reproduces the numbers.

| comparison | n | exact agreement | Cohen's kappa [95% CI] |
|---|---|---|---|
| verifier: Claude vs Gemini teacher | 50 | 0.78 | 0.63 [0.44, 0.81] |
| judge (answer scale): Claude vs Gemini judge | 36 | 0.94 | 0.84 [0.57, 1.00]; quadratic-weighted 0.89 [0.61, 1.00] |
| judge (abstain / false premise, yes/no): Claude vs Gemini judge | 4 | 4/4 | not informative (all "yes") |
| retrieval: Claude rubric v1 vs blind re-grade under "would it help answer the query" | 50 | 0.80 | 0.69 [0.52, 0.85]; quadratic-weighted 0.82 [0.68, 0.91] (self-consistency) |

**Judge**: the two disagreements are both Claude "partially" vs judge "correct" (an answer
that omits the direction of the survival effect, and one that reports a different error
figure than the key fact). The judge looks reliable but slightly lenient; 40 items cannot
show more than that.

**Verifier**: all 11 disagreements are teacher SUPPORTED vs Claude NOT_ENOUGH_INFO (7) or
CONTRADICTED (4). Four are synthetic contradiction pairs where the passage plainly says the
opposite ("Exercise decreased cardiac AMPKalpha2 activity" vs "increased"; a "6.4-compartment"
vs "16-compartment" model; "a significant difference also was not found" vs "was found";
"binds the CBS pockets of glucose complex" vs "of AMPK complex"); DeBERTa NLI calls all four
CONTRADICTED. Checked across the whole dataset: the teacher labels **28% of synthetic
contradiction pairs SUPPORTED** (train 171 of 623, val 18 of 59, test 18 of 63). On the 18
such test pairs NLI says CONTRADICTED for 16; the LoRA, trained on these labels, copies the
teacher on 7. Some synthetic edits may fail to change the meaning, but the four read here
did. Consequences: (a) the Week 7 verifier comparison is scored against a reference that is
lenient in a known direction, which favours teacher-like models (the LoRA, Gemini itself) over
NLI; (b) the training data teaches the LoRA to accept some contradictions. Proposed: relabel
synthetic pairs where the teacher says SUPPORTED but the edit is a negation, number change or
role swap (with NLI as a check and the human spot-check to measure the result), retrain, and
re-score every verifier. Not done here (it changes the headline numbers and needs the owner's
OK).

**Retrieval re-grade**: under the owner's criterion ("would this abstract help answer the
query": 2 = directly answers or key evidence, 1 = partially useful background, 0 = not
useful) the blind re-grade agreed with rubric v1 on 40 of 50; 7 of the 10 changes were
0 -> 1 (background that would still help), so the criterion is more lenient. The full
299-item sample was re-graded under it (`eval/labels/retrieval_claude_v2.jsonl`: the 50 blind
re-grades plus the other 249; 35 items 0 -> 1, 2 items 2 -> 1, 1 item 1 -> 2). Re-run of
`scripts/eval_retrieval.py --labels claude` (v2; 8 queries):

| mode | recall@10 v1 | recall@10 v2 | nDCG@10 v1 | nDCG@10 v2 | precision@10 v2 |
|---|---|---|---|---|---|
| BM25 | 0.374 | 0.339 [0.284, 0.407] | 0.625 | 0.651 [0.524, 0.782] | 0.800 |
| dense | 0.422 | 0.355 [0.299, 0.421] | 0.785 | 0.815 [0.713, 0.891] | 0.838 |
| hybrid | 0.371 | 0.344 [0.288, 0.412] | 0.706 | 0.746 [0.647, 0.838] | 0.825 |
| hybrid + rerank | 0.440 | 0.358 [0.304, 0.420] | 0.753 | 0.756 [0.645, 0.858] | 0.838 |

Hybrid + rerank vs dense under v2: recall@10 +0.003 [-0.026, +0.036], p = 1.0; nDCG@10
-0.059 [-0.176, +0.074], p = 0.45. Recall drops for every mode because more papers count as
relevant (best achievable recall@10 0.452, was 0.576). The conclusion does not change: no
measurable difference between hybrid + rerank and embeddings-only on these 8 queries.

## 7. Reference v2 preview: constructed contradictions labelled by construction

Prepared, not adopted: nothing has been retrained and the README headline still uses the
teacher reference. `uv run python scripts/make_verifier_data.py relabel` writes
`data/finetune/v2/{train,val,test}.jsonl` and `data/finetune/verifier_bundle_v2.zip`
(gitignored, like v1; v1 files are untouched). Rule-perturbed pairs (direction, negation,
number, entity swap; `--construction-kinds` to choose) get `labels.target = CONTRADICTED`
with `label_source = "construction"`, every other pair `labels.target = teacher`, and
`labels.teacher` keeps the original. Changes: train 171 SUPPORTED + 9 NOT_ENOUGH_INFO -> CONTRADICTED,
val 18 + 1, test 18 SUPPORTED -> CONTRADICTED. The notebook now trains on `labels.target`
when present (so it accepts either bundle) and records which reference it used.

`uv run python scripts/preview_reference_v2.py`: existing systems scored on the 252 test
pairs against both references (Gemini verifier labels read from the cache only; router at
0.75 escalates 23 pairs). "Without swap" leaves entity swaps on the teacher label, because a
swapped entity can read as NOT_ENOUGH_INFO rather than CONTRADICTED.

| system | accuracy vs v1 (teacher) | accuracy vs v2 | accuracy vs v2 without swap | macro-F1 vs v1 | macro-F1 vs v2 |
|---|---|---|---|---|---|
| DeBERTa-v3-large NLI | 0.742 [0.687, 0.794] | 0.802 [0.750, 0.849] | 0.778 [0.726, 0.825] | 0.721 | 0.780 |
| small LoRA (T4, trained on teacher labels) | 0.833 [0.786, 0.877] | 0.845 [0.802, 0.889] | 0.825 [0.778, 0.873] | 0.780 | 0.801 |
| router at 0.75 | 0.865 [0.821, 0.905] | 0.873 [0.829, 0.913] | 0.861 [0.817, 0.901] | 0.824 | 0.840 |
| Gemini verifier | 0.885 [0.845, 0.921] | 0.885 [0.845, 0.925] | 0.881 [0.837, 0.921] | 0.847 | 0.855 |
| Gemini teacher labels (as a system) | 1.000 (by definition) | 0.929 [0.893, 0.960] | 0.956 [0.929, 0.980] | 1.000 | 0.926 |

Reading: the correction mostly helps NLI (+0.060 accuracy), which had been marked wrong for
calling real contradictions CONTRADICTED; the LoRA-NLI gap shrinks from 0.091 to 0.043 and
their CIs now overlap. The Gemini verifier is unchanged and still the most accurate single
system; the ranking does not change. The LoRA here was trained on the uncorrected labels, so
the v2 numbers for it are a lower bound for a model retrained on the v2 bundle (needs the T4
notebook, about 105 minutes, owner's decision). The router's test accuracy (0.865 vs teacher)
had not been reported before; Week 7 reported its val accuracy only.
