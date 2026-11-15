# Claim verification and evaluation-suite results (Weeks 5-7)

All numbers below are copied from saved run records (`results/runs/*.json`) and eval reports
(`results/eval/<suite>/<run_id>/report.md`), which are gitignored; this file keeps them with
the code. Models: Gemini free tier (`gemini-3.5-flash` / `-flash-lite`); costs are list-price
equivalents (nothing was billed).

## Claim verification on vs off

**Off** (no claim verification; answers scored afterwards by the NLI verifier): smoke suite,
25 questions x 2 repeats, flash-lite for every role (`smoke-baseline`):

| metric | mean [95% CI] |
|---|---|
| groundedness (supported or computed / checkable claims) | 57.0% [42.5, 71.4] |
| citation support (cited claims entailed by their passage) | 72.1% [60.0, 84.0] |

So without verification about 43% of checkable claims in the final answer were not supported
by their cited passages.

**On** (verifier inside the pipeline, `VERIFY_POLICY=remove`), three live runs on 2026-11-12,
as recorded at run time (before the later same-paper and rule-outcome fixes):

| run | question | checkable claims | failed (not supported / contradicted) | removed | flagged |
|---|---|---|---|---|---|
| 20261112-082816-b56bd5 | approved AMPK drugs / glioblastoma | 26 | 10 (38%) | 5 claims | 5 sentences |
| 20261112-083229-fd26dc | MGMT and temozolomide resistance | 33 | 6 (18%) | 0 | 3 sentences |
| 20261112-084125-fb7cad | metformin vs TMZ drug-likeness (flash-lite) | 14 | 5 (36%) | 2 claims | 2 sentences |

With verification on, every failed claim is removed (when its whole sentence fails) or
marked `[unverified]` in the answer, so no unsupported claim is presented as verified.

**Router verifier vs DeBERTa NLI** (same 50 smoke answers re-scored; `smoke-router-t4` vs
`smoke-baseline`; T4 LoRA adapter, threshold 0.75; paired bootstrap + permutation test):

| metric | NLI | router | delta [95% CI] | p |
|---|---|---|---|---|
| groundedness | 0.570 | 0.677 | +0.107 [+0.007, +0.197] | 0.031 |
| citation support | 0.721 | 0.838 | +0.117 [+0.001, +0.230] | 0.067 |
| answer accuracy | 0.800 | 0.800 | 0 (same answers) | 1.0 |

## Evaluation suite

**Smoke baseline** (`smoke-baseline`, 25 questions x 2 repeats, flash-lite, isolated BM25 eval
index): accuracy 80.0% [64.0, 94.0] with an 8% flip rate across repeats; PubMedQA 50.0%
[22.2, 77.8] (n=9), corpus 91.7% [66.7, 100] (n=6), computed 100% (n=3), unanswerable abstained
100% (n=3), false premise corrected 100% (n=4); 28.4 s and $0.0083 per question.

**Full suite, partial** (`full-v1`, defences on, 65 of 188 questions done when the free quota
ran out; items run in id order, so this covers the 6 computed and 59 corpus questions only):

| metric | mean [95% CI] |
|---|---|
| accuracy | 91.5% [85.4, 96.9] (computed 100%, corpus 90.7% [84.7, 96.6]) |
| citation support | 81.2% [75.0, 87.1] |
| groundedness | 77.4% [70.5, 84.0] |
| gold-PMID recall | 100% (the eval index is small; see the README) |
| latency per question | 19.9 s [15.2, 25.2] |
| cost per question | $0.0099 [0.0087, 0.0112] |

Pending: the remaining 123 questions (PubMedQA, unanswerable, false premise), resumable with
`make eval-full RESUME=full-v1`.
