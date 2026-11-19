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

## 8. Retrained on reference v2: better on edited contradictions, worse on real claims

The owner retrained the LoRA on `verifier_bundle_v2.zip` with the same notebook and settings
(600 steps, effective batch 16, QLoRA; this run on a Colab **L4**, the v1 adapter on a T4;
63 min; notebook val accuracy 0.936 against the v2 reference). Adapter in
`models/verifier-lora-v2/` (v1 untouched). Local scoring on Apple MPS, batch size 1:
`scripts/verifier_probs.py --adapter models/verifier-lora-v2 --out
results/verifier/small_probs_v2.json --timed`, then `scripts/compare_verifiers_v2.py`
(Gemini labels from cache only). 252 test pairs:

| system | accuracy vs v2 | macro-F1 vs v2 | F1 S / C / N (v2) | accuracy vs v1 teacher | p50 / p95 ms | $ per 1k claims |
|---|---|---|---|---|---|---|
| DeBERTa-v3-large NLI | 0.802 [0.750, 0.849] | 0.780 [0.730, 0.828] | 0.82 / 0.96 / 0.56 | 0.742 [0.687, 0.794] | 309 / 599 | 0 |
| small LoRA v1 (teacher labels) | 0.845 [0.802, 0.889] | 0.801 [0.737, 0.857] | 0.89 / 0.82 / 0.70 | 0.833 [0.786, 0.877] | 545 / 706 | 0 |
| small LoRA v2 (reference v2) | 0.861 [0.817, 0.901] | 0.801 [0.740, 0.856] | 0.90 / 0.94 / 0.57 | 0.802 [0.750, 0.849] | 616 / 824 | 0 |
| Gemini verifier (flash-lite) | 0.885 [0.845, 0.925] | 0.855 [0.799, 0.904] | 0.92 / 0.86 / 0.79 | 0.885 [0.845, 0.921] | 113 (amortised) | 0.215 |
| router v1 at 0.75 | 0.873 [0.829, 0.913] | 0.840 [0.781, 0.889] | 0.90 / 0.86 / 0.76 | 0.865 [0.821, 0.905] | - | 0.020 |
| router v2 at 0.75 | 0.901 [0.861, 0.937] | 0.852 [0.793, 0.901] | 0.92 / 0.98 / 0.66 | 0.837 [0.790, 0.881] | - | 0.019 |

(Against "v2 without swap": LoRA v1 0.825, LoRA v2 0.833, router v1 0.861, router v2 0.873,
Gemini 0.881.)

Paired tests (bootstrap CI, sign-flip p), v2 reference: LoRA v2 - LoRA v1 accuracy +0.016
[-0.024, +0.056], p = 0.57, macro-F1 +0.001; LoRA v2 - NLI +0.059 [0.000, +0.115], p = 0.053;
router v2 - router v1 +0.028 [-0.008, +0.068], p = 0.21; router v2 - Gemini +0.016
[-0.028, +0.056], p = 0.59.

**The 18 edited contradictions the teacher called SUPPORTED**, number labelled CONTRADICTED:
NLI 16, LoRA v1 10, **LoRA v2 16**, Gemini verifier 9, router v1 10, **router v2 17**. All 63
perturbed test pairs: NLI 61, LoRA v1 46, LoRA v2 59, Gemini 49, router v1 48, router v2 61.

**Real agent claims (the 80 `run` pairs, whose labels v2 did not change)**: LoRA v1 65/80,
**LoRA v2 56/80**, paired -0.113 [-0.188, -0.038], **p = 0.012**; NOT_ENOUGH_INFO run claims
called SUPPORTED: v1 11 of 21, v2 16 of 21. Router on run pairs: v1 0.813, v2 0.750 (-0.063
[-0.138, +0.013], p = 0.18), Gemini 0.838. Other kinds (v2 reference, v1 -> v2): direction
8 -> 14 of 15, number 6 -> 9 of 9, swap 11 -> 14 of 15, negation 21 -> 22 of 24, cross 15 -> 16 of 19,
positive 87 -> 86 of 90.

**Router threshold on val (v2 reference)**: v2 adapter alone 0.927 (v1 0.884, Gemini alone
0.876). v2 curve: 0.6 -> 2.6% escalated, 0.940; 0.7 -> 4.3%, 0.944; **0.75 -> 5.6%, 0.953**;
0.8 -> 6.0%, 0.953; 0.85 -> 6.9%, 0.953; 0.9 -> 10.7%, 0.944. 0.75 is still the knee (the
cheapest threshold at the plateau), at $0.012 per 1k claims. Val is only 233 pairs and is
scored against the same constructed labels the adapter learned, so it overstates v2.

**Decision: v1 stays the default adapter.** The v2 gain is concentrated on synthetic edits
produced by the same rules that defined its training labels, while it is significantly worse
on real agent claims, which is the distribution the verifier serves; NOT_ENOUGH_INFO F1 drops
from 0.70 to 0.57. v2 remains selectable with `VERIFIER_ADAPTER_DIR=models/verifier-lora-v2`.
Limits: one training run per adapter, on different GPUs (T4 vs L4), so run-to-run variance is
unmeasured; the paired tests are across test pairs, not training runs. Next: keep the
construction labels but add `run`-kind NOT_ENOUGH_INFO hard negatives (section 2), train two
seeds per variant, and select on a val split that includes run pairs.

## 9. Post-reset runs (2026-11-15): PubMedQA variants, sorafenib fixes

**Quota**: the free tier ran out after 73 agent questions (about 490 LLM calls) today:
flash-lite stopped near 500 requests and flash much earlier, so the planned runs only
partly completed. Nothing below was adopted; all defaults are unchanged.

### PubMedQA yes-bias (held-out set)

`eval/suites.yaml: pubmedqa-ab`: 30 PubMedQA items not in the smoke suite (12 gold yes, 12 no,
6 maybe; seed 20261115), flash-lite in every role, 1 repeat. Variants:
`--synth gemini:gemini-3.5-flash` and `--synth-variant results_first` (a synthesizer rule:
find the decisive result, answer Yes/No from it even against the question's framing, Maybe
only for mixed results, start with the label; opt-in, `SYNTH_VARIANT=results_first`).

| run | items done | accuracy [95% CI] | gold yes | gold no | gold maybe | answers "yes" |
|---|---|---|---|---|---|---|
| baseline (flash-lite) | 30/30 | 56.7% [40.0, 73.3] | 11/12 | 5/12 | 1/6 | 19 of 30 |
| flash synthesizer | 19/30 (flash quota) | 47.4% | 6/6 | 3/9 | 0/4 | 10 of 19 |
| results-first prompt | 24/30 (flash-lite quota) | 50.0% | 8/9 | 4/9 | 0/6 | 15 of 24 (+2 with no label) |

Paired against the baseline on the same items (`bioagent eval compare`): flash synthesizer
+0.000 [-0.158, +0.158], p = 1.0 (n = 19); results-first +0.000 [-0.167, +0.167], p = 1.0
(n = 24). On those items the gold-"no" score moved from 2 to 3 of 9 (flash) and 2 to 4 of 9
(results-first), while gold-"maybe" went from 1 to 0. Groundedness: flash +0.121
[+0.008, +0.229], p = 0.055; results-first -0.064, n.s.

Reading: the yes bias reproduces on held-out items (19 "yes" answers for 12 gold "yes"
items; gold "no" 5 of 12 right). Neither change is significantly better, so neither is
adopted. The results-first rule also made 2 answers lose their label. Resume both variant
runs after the next reset (`bioagent eval run --suite pubmedqa-ab --resume pqa-ab-flash`,
`... --resume pqa-ab-rf`); 30 items x 1 repeat can only detect large effects (CI half-width
about 0.16), so a decision needs the full 100 PubMedQA items or repeats.

### Sorafenib fixes (offline only; live runs not reached)

Three general fixes are now settings, defaults unchanged: `DISCOVERY_K` (corpus hits kept per
discovery search, default 15), `DISCOVERY_MIN_EVIDENCE` (graph evidence floor, default 2) and
`RERANK_FUSION` (hybrid_rerank fuses the reranker order with the BM25 order by RRF instead
of using the reranker alone).

* `RERANK_FUSION=true`: the sorafenib paper (28625738) moves from rank 25 to **rank 8** for
  "AMPK inhibitor approved drug", inside the 15 kept; the two "activator" queries still miss
  it. General retrieval on the Claude-graded sample (8 queries, not human labels): hybrid +
  rerank with fusion recall@10 0.347 [0.299, 0.410], nDCG@10 0.725 [0.608, 0.833], vs 0.358
  and 0.756 without fusion (small and within noise, but not an improvement).
* `DISCOVERY_K=25`: the paper sits exactly at rank 25 for one query, so k = 25 includes it by a
  margin of one; fragile.
* `DISCOVERY_MIN_EVIDENCE=1`: does not help on its own; sorafenib would rank 47th of 74 approved
  drugs with literature edges, past the drugs_targeting limit of 20.

Prepared live comparison (needs about 100 calls; not run): the AMPK question with baseline,
`DISCOVERY_K=25`, `RERANK_FUSION=true`, and both; plus mTOR/glioblastoma and PI3K/breast cancer
questions with and without fusion as an overfitting check.

### Full eval

Not resumed: `full-v1` is still at 65 of 188 (computed and corpus done; false premise,
PubMedQA and unanswerable remain). `make eval-full RESUME=full-v1` after the next reset.

## 10. Runs of 2026-11-16 (after the reset)

**PubMedQA variants, held-out 30 items** (`pubmedqa-ab`; paired against the flash-lite
baseline, 56.7% [40.0, 73.3]):

| variant | items | accuracy | paired delta [95% CI], p | gold yes / no / maybe (baseline -> variant) | other |
|---|---|---|---|---|---|
| results-first prompt (`SYNTH_VARIANT=results_first`) | 30/30 | 60.0% [40.0, 76.7] | +0.033 [-0.100, +0.167], p = 1.0 | 11 -> 11 of 12 / 5 -> 7 of 12 / 1 -> 0 of 6 | 2 answers with no yes/no/maybe label |
| flash synthesizer | 26/30 (flash quota) | 53.8% on 26 | +0.000 [-0.115, +0.115], p = 1.0 | 9 -> 9 of 9 / 4 -> 5 of 11 / 1 -> 0 of 5 | latency +21 s [+2, +40], p = 0.04 |

Neither is adopted: results-first helps on gold "no" items (+2 of 12) but loses the one
"maybe" and leaves 2 answers unlabelled, and the overall change is not significant; the
flash synthesizer changes nothing on accuracy and is slower.

**Sorafenib live comparison**: not run. Each `bioagent ask` failed at once with a flash 429:
the default agent uses flash for the planner and synthesizer, and the flash daily quota had
already been spent by the PubMedQA flash variant. Needs a day where flash is reserved for it
(about 8 questions), or a flash-lite-only rerun of the comparison.

**Full eval `full-v1`**: 110 of 188 done (45 new today: 11 false-premise, 33 PubMedQA and
one more); paused on the flash-lite quota. Partial results (110 items): accuracy 81.8%
[74.5, 88.2]; computed 100% (n = 6), corpus 90.0% [83.3, 95.8] (n = 60), false premise
corrected 100% (n = 11), PubMedQA 57.6% [39.4, 72.8] (n = 33); citation support 66.4%,
groundedness 59.1%, 27.9 s and $0.0108 (list-price equivalent) per question. Remaining: 67
PubMedQA and 11 unanswerable items (`make eval-full RESUME=full-v1`).

## 11. Runs of 2026-11-17 (after the reset)

**Full eval `full-v1`: 180 of 188** (70 more; paused on the flash-lite daily quota). Accuracy
77.8% [72.2, 83.6] over 180 items:

| category | n | accuracy [95% CI] | other |
|---|---|---|---|
| computed | 6 | 100.0% [100.0, 100.0] | |
| corpus | 60 | 90.0% [83.3, 95.8] | gold-PMID recall 100% |
| false premise | 11 | 100.0% (premise corrected) | |
| PubMedQA | 100 (complete) | 66.0% [57.0, 75.0] | gold-PMID recall 94.0% |
| unanswerable | 3 of 11 | 100.0% abstained | |

PubMedQA on all 100 items (66.0%) is higher than the 30-item held-out set (56.7%) and the
smoke subset (50.0%); the CIs overlap. Remaining: 8 unanswerable items.

**Sorafenib live comparison: 2 of 8 questions completed.**

* AMPK baseline (run 20261117-210922-ee4f6c): drugs named metformin, phenformin, berberine,
  aspirin/salicylate, rosiglitazone; no sorafenib, cites neither 28625738 nor 25080865.
* AMPK with `DISCOVERY_K=25` (run 20261117-211128-c2e262): only metformin named; no sorafenib.
* `RERANK_FUSION=true` (AMPK) failed on a Gemini flash **503 "model is currently experiencing
  high demand"** after 5 retries (transient, provider side). The other five (fusion + k25,
  mTOR and PI3K with and without fusion) stopped on the flash **daily quota (429)**. No bug in
  the fusion path; the `KeyError: '__import__'` lines in the logs are interpreter-shutdown
  noise from an un-awaited HTTP coroutine. The live effect of fusion is still unmeasured;
  offline it lifts 28625738 from rank 25 to 8.
* Flash's free daily limit is small: the planner and synthesizer use flash, and it ran out
  after about 2-3 agent questions. The comparison may need a flash-lite planner/synthesizer
  (`PLANNER_MODEL`/`SYNTH_MODEL`) to finish in one day.

## 12. Runs of 2026-11-18: full eval complete, sorafenib fusion (live)

**Full eval `full-v1`: complete, 188 of 188** (flash-lite in every role, defences on, 1 repeat).
Accuracy **78.7% [72.9, 84.6]**; citation support 63.2% [58.3, 68.3]; groundedness 54.0%
[49.0, 59.1]; 30.3 s [26.4, 34.4] and $0.0104 (list-price equivalent; free tier) per question.

| category | n | accuracy [95% CI] | other |
|---|---|---|---|
| computed | 6 | 100.0% | |
| corpus | 60 | 90.0% [83.3, 95.8] | gold-PMID recall 100% |
| false premise | 11 | 100.0% (premise corrected) | |
| PubMedQA | 100 | 66.0% [57.0, 75.0] | gold-PMID recall 94.0% |
| unanswerable | 11 | 100.0% (abstained) | |

**Bug found: shell settings did not reach the MCP servers.** The retrieval server runs as an
MCP subprocess; when a server had no per-server overrides it was started with the MCP
client's minimal default environment (PATH, HOME, ...), so a setting exported in the shell
(`RERANK_FUSION=true`) never reached it (`.env` values did, because the servers read that
file). The first fusion runs this morning (and on 2026-11-17) therefore ran *without*
fusion. Fixed: `bioagent.agents.tools.server_environment` forwards this process's bioagent
settings variables (only names of `Settings` fields) to every server; test added.

**Sorafenib, live (flash-lite in every role, one run per cell)**:

| question | run | key sorafenib paper 28625738 in evidence | sorafenib in answer | references | groundedness | drugs named |
|---|---|---|---|---|---|---|
| AMPK / glioblastoma | baseline | no | no | 7 | 0.50 | metformin, aspirin |
| AMPK / glioblastoma | fusion (fixed) | **yes** | no | 15 | 0.63 | metformin |
| mTOR / glioblastoma | baseline | no | no | 11 | 0.50 | rapamycin/sirolimus, everolimus, metformin |
| mTOR / glioblastoma | fusion (fixed) | no | no | 14 | 0.50 | + temsirolimus, alpelisib, phenformin |
| PI3K / breast cancer | baseline | no | no | 2 | 0.25 | alpelisib |
| PI3K / breast cancer | fusion (fixed) | no | no | 7 | 0.30 | + idelalisib, copanlisib, duvelisib, inavolisib, everolimus, sirolimus |

Reading: with fusion actually on, the AMPK run now retrieves the sorafenib paper, as the
offline ranking predicted (rank 25 -> 8), but the answer still does not name sorafenib, so
the next bottleneck is after retrieval (the drug-candidate merge, which ranks by paper count
and keeps 20, or the synthesizer's evidence budget). Fusion runs had more references in all
three questions, but with one run per cell and run-to-run variation as large as this (the
two invalid "fusion" runs, which were really baselines, also named more drugs for PI3K),
no effect on answer quality can be claimed. `RERANK_FUSION` stays off by default. Next:
inspect the AMPK fusion run's candidate list, and run 3+ repeats per cell before deciding.

## 13. Where sorafenib is lost after retrieval (offline, AMPK fusion run 20261118-081915-c32ab2)

Reconstructed from the saved run with the graph service (no LLM calls):

1. **Retrieved.** The discovery corpus searches return 60 papers for task t1, including
   28625738 ("Mechanisms of paradoxical activation of AMPK by the kinase inhibitors SU6656 and
   sorafenib"); the knowledge graph records `Sorafenib` as mentioned in it (2 mentions).
2. **Ranked low among mentioned drugs.** `drugs_mentioned` over the 60 papers returns 50 drugs
   ordered by the number of those papers that mention them; sorafenib is in 1 paper and comes
   **41st of 50**. Above it are frequent non-drug or endogenous entities (adenosine
   triphosphate, glucose, lipids, fatty acids, adenosine monophosphate, carbon) and the
   investigational activator A-769662 (named by its chemical name).
3. **Cut at the candidate limit.** `merge_candidates` keeps 20 candidates (drugs found by both
   the graph and the corpus first, then graph-only and corpus-only alternately, each by
   evidence count). Sorafenib is not among the 20, so neither the specialist nor the
   synthesizer is told about it; the passage itself sits among 146 evidence items and is not
   cited.

**Tried offline**: dropping corpus mentions without ChEMBL phase 4 when the question asks for
approved drugs lifts sorafenib to 18th of 50 among approved mentions, but it is still cut:
the 20 slots go to metformin, berberine, aspirin and phenformin, and also to endogenous
"approved" entries from the graph (glucose, hydrogen peroxide, calcium, adenosine,
epinephrine, nicotine) and to approved drugs that merely appear in a retrieved abstract
(sertraline, disulfiram, celecoxib). The underlying gap is that a candidate is not tied to
the target: PubTator links only STK11 and FYN in 28625738, not an AMPK subunit, so a
"co-mentioned with the target" filter would also miss it.

**No fix shipped** (no small change is clearly correct). Options, each to be measured on
several targets before adoption: (a) exclude endogenous metabolites and ions from candidates
(a curated list or a ChEBI role filter); (b) weight corpus candidates by the rank of the best
hit that mentions them, not only by paper count, so a drug named in a top-ranked hit for an
"activator/inhibitor" query survives; (c) let the specialist see the full mentioned-drug list
with paper counts instead of a fixed top 20.

## 14. Rerank fusion, repeated (2026-11-19): not adopted

AMPK, mTOR and PI3K questions, `RERANK_FUSION` off and on alternately, 3 runs per condition
(18 runs, all completed), flash-lite in every role, settings forwarded to the MCP servers
(section 12 fix), Neo4j checked healthy before every run. Per condition (mean of 3):

| question | fusion | references | groundedness | drugs named | key paper 28625738 retrieved | sorafenib in answer | seconds |
|---|---|---|---|---|---|---|---|
| AMPK / glioblastoma | off | 7.3 | 0.36 | 4.3 | 0/3 | 0/3 | 51 |
| AMPK / glioblastoma | on | 3.3 | 0.15 | 1.7 | **3/3** | 0/3 | 61 |
| mTOR / glioblastoma | off | 9.3 | 0.38 | 4.3 | - | - | 78 |
| mTOR / glioblastoma | on | 7.0 | 0.33 | 4.0 | - | - | 118 |
| PI3K / breast cancer | off | 6.0 | 0.42 | 3.7 | - | - | 112 |
| PI3K / breast cancer | on | 8.0 | 0.45 | 4.3 | - | - | 79 |

Paired over the 9 (question, repeat) pairs, on minus off (bootstrap 95% CI, sign-flip p):
references -1.4 [-3.8, +1.0], p = 0.34; groundedness -0.08 [-0.21, +0.04], p = 0.32; drugs
named -0.8 [-2.8, +1.3], p = 0.56; latency +5.6 s [-26, +38], p = 0.72.

**Verdict**: fusion does what the offline ranking predicted for retrieval (the sorafenib
paper is retrieved in 3 of 3 AMPK runs instead of 0 of 3), but it does not change any answer
measure significantly, never gets sorafenib into an answer (section 13: it is cut at the
20-candidate limit), and the AMPK runs trended worse. `RERANK_FUSION` stays off; the fix for
sorafenib has to be in the candidate selection (section 13 options), not in retrieval.
Earlier note: an aborted first launch this morning ran one question while Docker Desktop (and
so Neo4j) was down; it was discarded and the job restarted after Neo4j was back.

## 15. Ranking drug candidates by paper rank (2026-11-19): does not recover sorafenib

`DRUG_CANDIDATE_RANKING=paper_rank` (new, default `count`): corpus-only candidates are
ordered by the summed reciprocal rank, 1 / (10 + rank), of the discovery-search hits that
mention them instead of by how many hits mention them; the 20-candidate cap is unchanged.
Unit-tested (a drug named only in a top-ranked hit survives the cap; count mode unchanged).

**Endogenous entities**: the graph has no flag that separates drugs from metabolites. ChEMBL
lists glucose, calcium, oxygen, lactic acid and hydrogen peroxide as approved (max_phase 4),
and "has a curated ChEMBL mechanism" would drop phenformin, berberine, chloroquine and
temozolomide while keeping adenosine, epinephrine and nicotine. No exclusion was added.

**Offline replay** of the AMPK discovery step (the three discovery searches re-run locally,
k = 15, hybrid + rerank; same graph calls; no LLM):

| retrieval | candidates from | ranking | sorafenib paper rank | sorafenib position (kept: top 20) |
|---|---|---|---|---|
| fusion off | all mentions | count / paper_rank | not retrieved | not a candidate |
| fusion on | all mentions (56) | count | 8 | 44 |
| fusion on | all mentions (56) | paper_rank | 8 | 45 |
| fusion on | approved mentions only (38) | count | 8 | 26 |
| fusion on | approved mentions only (38) | paper_rank | 8 | 30 |

Why it fails: the highest-ranked hits are repurposing reviews that each mention many approved
drugs (niclosamide, sertraline, celecoxib, auranofin, dexamethasone, propranolol, ...), so
they outscore a drug named in a single rank-8 paper; with all mentions, unknown-phase
entities in those reviews (A-769662, AMP, lipids, AICA ribonucleotide) also come first.
Neither count nor rank says whether a paper reports the drug acting *on the target*.

**Decision**: default stays `count`; the live 18-run comparison was not run (it would test a
change that cannot recover the case it was built for, at about 130 calls). A fix needs
target-specific evidence per candidate, e.g. let the specialist read the sentence that
mentions each candidate in the top hits, or extract drug-target relations from the
retrieved passages, rather than another ranking of mention counts.
