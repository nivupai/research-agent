# Week 7: fine-tuned small verifier vs large model (2026-11-13)

Test split: 252 claim/passage pairs from PMID-held-out papers; reference = Gemini teacher
(gemini-3.5-flash-lite) labels. The Gemini row is the same model used as a verifier at
inference with its own cache namespace (it never reuses the teacher's labels), so it measures
teacher self-agreement, not independent accuracy. Human spot labels
(eval/labels/verifier_test_human.jsonl) are pending.

## Headline: full T4 adapter (600 steps)

Qwen/Qwen3-1.7B + LoRA r=16 + classification head, trained by the project owner with
notebooks/train_verifier_lora.ipynb on a free Colab Tesla T4: 4-bit nf4 (QLoRA),
600 steps x effective batch 16 at max length 512,
105 min (10.46 s/step), peak GPU memory
5.23 GB; notebook val accuracy 0.884, macro-F1
0.838. Served here on the bf16 base on MPS (trained in 4-bit).

Test split (252 PMID-held-out pairs, reference = Gemini teacher labels):

| system | accuracy [95% CI] | macro-F1 [95% CI] | p50 / p95 per claim | cost per 1k claims |
|---|---|---|---|---|
| **small LoRA, T4, 600 steps** | **0.837 [0.790, 0.881]** | **0.785 [0.717, 0.843]** | 565 / 714 ms | $0 (local) |
| small LoRA, local, 150 steps | 0.821 [0.778, 0.865] | 0.772 [0.708, 0.831] | 651 / 867 ms | $0 (local) |
| DeBERTa-v3-large NLI | 0.742 [0.687, 0.794] | 0.721 [0.661, 0.776] | 309 / 599 ms | $0 (local) |
| Gemini flash-lite (verifier) | 0.885 [0.845, 0.921] | 0.847 [0.789, 0.896] | 113 ms (amortised) | $0.215 |

Per-class F1 (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO): T4 small 0.88 / 0.77 / 0.71;
local small 0.87 / 0.73 / 0.72; DeBERTa 0.78 / 0.83 / 0.56; Gemini 0.92 / 0.83 / 0.79.
Main remaining small-model error: NOT_ENOUGH_INFO pairs called SUPPORTED (13 of 38).
Gemini latency/cost were measured on 2026-11-13 when its labels were first fetched (the
labels are cached, so later runs reuse those measurements).

Router on val with the T4 adapter (escalate to Gemini when the small model's top
probability is below the threshold):

| threshold | escalated | val accuracy | cost per 1k claims |
|---|---|---|---|
| 0.6 | 3% | 0.888 | $0.006 |
| 0.7 | 7% | 0.893 | $0.015 |
| **0.75 (default)** | **8%** | **0.906** | **$0.017** |
| 0.8 | 10% | 0.914 | $0.021 |
| 0.85 | 12% | 0.918 | $0.026 |
| 0.9 | 15% | 0.914 | $0.032 |
| 0.99 | 34% | 0.910 | $0.074 |
| all Gemini (test) | 100% | 0.885 | $0.215 |

Is 0.75 still the knee? Yes, approximately: with the better-calibrated T4 adapter the
curve flattens much earlier. At 0.75 only 8% of claims are escalated and val accuracy
(0.906) is already above all-Gemini, at about 8% of its cost; 0.8-0.85 add 0.8-1.2 points
for 2-4 more points of escalation, which is within the noise of 233 val pairs. The automatic
rule (cheapest within 1 point of the best) would now pick 0.8. The default stays 0.75.

## Earlier: local fallback adapter (150 steps)

Small model: Qwen/Qwen3-1.7B + LoRA r=16 + classification head, **local fallback run**
(150 steps x effective batch 8 = 1200 examples,
max length 384, gradient checkpointing, mps:0, 40 min,
15.99 s/step, peak MPS memory 5.35 GB, 17,438,720 of
1,738,019,840 parameters trainable). The full 600-step T4 notebook run will replace it.
Gemini latency is amortised per pair over 20-pair batches; cost is list price (free tier used).

# Verifier comparison (test split, reference: teacher)

| system | n | accuracy [95% CI] | macro-F1 [95% CI] | p50 / p95 latency per claim | cost per 1k claims |
|---|---|---|---|---|---|
| fine-tuned small (LoRA, 150 steps)  | 252 | 0.821 [0.778, 0.865] | 0.772 [0.708, 0.831] | 651 / 867 ms | $0 (local) |
| DeBERTa-v3-large NLI  | 252 | 0.742 [0.687, 0.794] | 0.721 [0.661, 0.776] | 309 / 599 ms | $0 (local) |
| Gemini gemini:gemini-3.5-flash-lite (large model)  | 252 | 0.885 [0.845, 0.921] | 0.847 [0.789, 0.896] | 113 / 113 ms | $0.215 |

## Router (small model, escalate below threshold) on val

| threshold | escalated | accuracy | cost per 1k claims |
|---|---|---|---|
| 0.4 | 2% | 0.837 | $0.004 |
| 0.5 | 8% | 0.828 | $0.018 |
| 0.6 | 19% | 0.858 | $0.042 |
| 0.7 | 24% | 0.863 | $0.054 |
| 0.75 | 31% | 0.867 | $0.069 |
| 0.8 | 43% | 0.854 | $0.094 |
| 0.85 | 53% | 0.863 | $0.117 |
| 0.9 | 77% | 0.884 | $0.169 |
| 0.95 | 96% | 0.897 | $0.211 |
| 0.99 | 100% | 0.893 | $0.220 |

Automatic rule: 0.95 (cheapest within 1 point of the best val accuracy), but that escalates
96% of claims and saves almost nothing over calling the LLM for every claim.

**Chosen default: 0.75** (`VERIFIER_ROUTER_THRESHOLD`), the knee of the accuracy/cost curve:
31% of claims escalated, val accuracy 0.867 vs 0.893-0.897 for (nearly) all-LLM, at about 30%
of the all-LLM cost ($0.069 vs $0.220 per 1k claims). Set by the project owner on 2026-11-13
from this local-adapter curve; revisit once the 600-step T4 adapter is evaluated.

Confusion matrices (rows = reference, columns = prediction) are in eval.json.


## Smoke suite with the router verifier vs the NLI verifier

Same 50 saved smoke answers re-scored (agent answers unchanged; VERIFIER_BACKEND=router with
the local adapter and the earlier default threshold 0.8 vs DeBERTa NLI), so only citation support and groundedness
can differ:

### Eval comparison: smoke-router vs baseline smoke-baseline

Verdict on accuracy: **no regression** (regression = 95% CI of the paired difference entirely below -0.0)

| metric | baseline | candidate | delta [95% CI] | p (permutation) | verdict |
|---|---|---|---|---|---|
| accuracy | 0.800 | 0.800 | +0.000 [+0.000, +0.000] | 1.000 | no significant change (n=25) |
| citation_support | 0.721 | 0.789 | +0.067 [-0.032, +0.177] | 0.241 | no significant change (n=22) |
| groundedness | 0.570 | 0.635 | +0.065 [-0.004, +0.142] | 0.106 | no significant change (n=25) |
| gold_pmid_recall | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | no significant change (n=15) |
| tool_calls | 3.460 | 3.460 | +0.000 [+0.000, +0.000] | 1.000 | no significant change (n=25) |
| latency_s | 28.436 | 28.436 | +0.000 [+0.000, +0.000] | 1.000 | no significant change (n=25) |
| tokens | 23168.960 | 23168.960 | +0.000 [+0.000, +0.000] | 1.000 | no significant change (n=25) |
| cost_usd | 0.008 | 0.008 | +0.000 [+0.000, +0.000] | 1.000 | no significant change (n=25) |
| abstained | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | no significant change (n=3) |
| premise_corrected | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.000 | no significant change (n=4) |

Pending: the same smoke re-score with the T4 adapter at the 0.75 default. It needs live
Gemini calls for the escalated claims, and the free-tier flash-lite quota was exhausted on
2026-11-13 (the LLM backend now raises instead of substituting labels when that happens).
