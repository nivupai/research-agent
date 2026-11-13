# Week 7: fine-tuned small verifier vs large model (2026-11-13)

Test split: 252 claim/passage pairs from PMID-held-out papers; reference = Gemini teacher
(gemini-3.5-flash-lite) labels. The Gemini row is the same model used as a verifier at
inference with its own cache namespace (it never reuses the teacher's labels), so it measures
teacher self-agreement, not independent accuracy. Human spot labels
(eval/labels/verifier_test_human.jsonl) are pending.

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

Chosen threshold: 0.95 (cheapest within 1 point of the best val accuracy).

Confusion matrices (rows = reference, columns = prediction) are in eval.json.


## Smoke suite with the router verifier vs the NLI verifier

Same 50 saved smoke answers re-scored (agent answers unchanged; VERIFIER_BACKEND=router with
the local adapter and threshold 0.8 vs DeBERTa NLI), so only citation support and groundedness
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
