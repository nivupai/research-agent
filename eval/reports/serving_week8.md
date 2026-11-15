# Weeks 8-9 (part 1): serving latency with the semantic cache on vs off (2026-11-13)

Setup: real API (uvicorn) + real Redis + real semantic cache (bge-small, threshold 0.95) +
the real worker loop, on this Mac. To avoid Gemini calls the worker *replays* recorded
smoke-baseline answers, sleeping for each run's recorded wall time x 0.1
(`scripts/bench_serve.py`). 10 questions x 5 forms (exact, lowercase, extra spaces,
"Please answer:" prefix, one-letter typo) = 50 sequential requests per mode; latency is
client-side from POST /v1/ask until the job is completed.

| mode | requests | cache hits | p50 | p95 | mean |
|---|---|---|---|---|---|
| cache off (no_cache) | 50 | 0 | 1.22 s | 9.47 s | 2.77 s |
| cache on | 50 | 34 (68%) | 42 ms | 6.01 s | 0.89 s |
| cache hits only | 34 | - | 37 ms | 57 ms | - |

At real (1x) speed the miss path is ~10x longer (cache off: p50 ~12 s, p95
~95 s), while a hit stays ~40-60 ms (embedding + Redis scan + job record).

* p95 stays miss-dominated with the cache on: every question's first form must miss (10 of
  50 requests), so p95 only drops when repeats dominate the traffic. The win is in p50/mean
  and in LLM calls avoided (34 of 50 runs here).
* Misses: genuinely different questions scored 0.55-0.70 against cached ones (correct
  misses); 6 near-duplicate forms (polite prefix, typo) scored 0.915-0.936, just under 0.95.
  A threshold around 0.90 would have caught them with a wide margin here, but 10 questions is
  too few to lower the default; it stays 0.95 (SEMANTIC_CACHE_THRESHOLD).
* Tracing: one trace per request in Jaeger spans POST /v1/ask -> api.ask -> cache.lookup ->
  queue.enqueue -> worker.job (context carried in the job); graph.node / llm.call / mcp.tool
  spans nest under worker.job in real runs (covered by tests/test_serve.py; the replay
  pipeline skips the graph). Job polling makes many short GET traces; prefer the SSE endpoint.

Pending (needs Gemini quota, exhausted until 08:00 UTC): one live end-to-end run through
the API and worker with real LLM calls, to measure real p50/p95 and per-query cost.

## Kubernetes: kind deploy and KEDA worker autoscaling (2026-11-15)

Verified on a real single-node kind cluster (kind v0.33, KEDA from `kedacore/keda`) on this
Mac, with `make k8s-up k8s-deploy` and the slim image (`bioagent:slim`, 1.4 GB, no torch)
loaded into the node. Values: `deploy/kind/values-kind.yaml` (replay pipeline,
`replaySeconds: 20`, worker concurrency 1, KEDA `listLength: 2`, 1-4 workers, 60 s
scale-down stabilisation). No LLM calls were made.

* All pods Ready: api, worker, redis (the worker restarted once at install while Redis was
  still starting). `/healthz` ok, `/readyz` ready with the queue depth.
* One `POST /v1/ask` via `kubectl port-forward`: completed in 21 s (20 s replay), SSE stream
  `queued -> running -> progress x4 -> final`, summary with 0 LLM calls and $0.
* Chart bug found and fixed: the KEDA Redis trigger used the short service name, which the
  KEDA operator resolves in its own namespace (`no such host`); it now uses
  `<svc>.<namespace>.svc.cluster.local`. KEDA also warned that `cooldownPeriod` only applies
  with `minReplicas: 0`, so scale-down is now paced by an explicit HPA stabilisation window
  (`autoscaling.keda.scaleDownStabilizationSeconds`, default 60).

Burst of 10 replay jobs (the API's per-client burst limit), sampled every ~5 s:

| t (s) | desired workers | ready workers | jobs waiting | jobs done |
|---|---|---|---|---|
| 0 | 1 | 1 | 9 | 0 |
| 6 | 1 | 1 | 9 | 0 |
| 11 | 1 | 1 | 9 | 0 |
| 16 | 4 | 1 | 6 | 0 |
| 22 | 4 | 1 | 5 | 1 |
| 27 | 4 | 4 | 5 | 1 |
| 32 | 4 | 4 | 5 | 1 |
| 38 | 4 | 4 | 2 | 4 |
| 43 | 4 | 4 | 1 | 5 |
| 49 | 4 | 4 | 1 | 5 |
| 54 | 4 | 4 | 1 | 5 |
| 59 | 4 | 4 | 0 | 8 |
| 65 | 4 | 4 | 0 | 9 |
| 70 | 4 | 4 | 0 | 9 |
| 75 | 3 | 3 | 0 | 10 |
| 81 | 3 | 3 | 0 | 10 |
| 86 | 3 | 3 | 0 | 10 |
| 92 | 1 | 1 | 0 | 10 |

KEDA scaled the workers 1 -> 4 within 16 s of the burst (all 4 Ready at 27 s); the queue was
drained at 59 s and all 10 jobs were done at 75 s (one worker would need ~200 s); after the
60 s stabilisation window the deployment scaled 4 -> 3 -> 1 by 92 s. The cluster was then
removed with `make k8s-down`.
