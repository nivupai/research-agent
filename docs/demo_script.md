# 3-minute demo script

Target length 3:00. Record a browser window with the web UI (http://localhost:8000/ui) and a
second tab for Jaeger; keep a terminal at a large font (e.g. 18 pt) for the audit and
Kubernetes shots. Never show `.env` or any key on screen.

## Before recording (not on camera)

```bash
cd ~/research-agent
docker compose --profile serve up -d neo4j redis jaeger     # graph, queue, tracing
make serve-api        # terminal A (OTLP_ENDPOINT points at Jaeger)
make serve-worker     # terminal B
# warm-up: run the demo question once so model downloads and caches are done
curl -s -XPOST localhost:8000/v1/ask -H 'content-type: application/json' \
     -d '{"question": "What is the role of MGMT in temozolomide resistance in glioblastoma?"}'
```

Check the Gemini free quota has headroom (one live question uses ~6-11 calls; the free tier
allows roughly 500 a day). **No quota?** Run the worker as
`WORKER_PIPELINE=replay REPLAY_SECONDS=12 make serve-worker`: asking the MGMT question in the
UI then replays the saved run 20261112-083229-fd26dc with its real events, and the page says
"replayed from saved run". Either way, have
http://localhost:8000/ui?run=20261112-083229-fd26dc&claim=8 open in a spare tab as a fallback.

## Shot list

| time | shot | say (roughly) | do |
|---|---|---|---|
| 0:00-0:15 | README top: pitch + architecture diagram | "A multi-agent assistant for drug-repurposing questions that cites its evidence and checks every claim." | scroll the Mermaid diagram |
| 0:15-0:45 | Web UI: type the MGMT question, press Ask; the Progress panel fills in | "Questions go through an API and a Redis queue to a worker running a LangGraph planner and specialists; each step streams back." | point at planner, tool calls, synthesizer, verifier as they appear |
| 0:45-1:20 | Web UI: answer, stats, claim table; click claim 8 (contradicted) | "Every sentence is checked against the passage it cites. This one is flagged: the cited abstract itself says MGMT inactivates TMZ, the reverse of the claim, and the highlight shows the exact window the verifier read." | click claim 8, switch between [3] and [4], open the PubMed link |
| 1:20-1:40 | Jaeger UI (http://localhost:16686), service `bioagent-api` | "One trace per question: API, queue, worker, every graph node, LLM call and tool call, with tokens and cost." | open the latest trace, expand `worker.job` |
| 1:40-1:55 | Audit log | "Every tool call, allowed or denied, and every LLM call goes into a hash-chained audit log." | command 4 |
| 1:55-2:20 | Results table in the README | "Measured, with confidence intervals: the fine-tuned 1.7B verifier gets 0.84 accuracy vs 0.89 for Gemini at zero cost, and routing only 8 percent of claims to Gemini cuts verifier cost to about 2 cents per thousand claims." | scroll to Results |
| 2:20-2:50 | KEDA scaling on kind (pre-recorded is fine) | "On Kubernetes, workers scale on the queue length: a burst of 10 jobs scales 1 to 4 workers in 16 seconds." | command 5, or show the table in `eval/reports/serving_week8.md` |
| 2:50-3:00 | README failure cases link | "And what broke along the way is written up too." | open `docs/failure_cases.md` |

## Commands

1. Ask (terminal C):

```bash
JOB=$(curl -s -XPOST localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question": "What is the role of MGMT in temozolomide resistance in glioblastoma?", "no_cache": true}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])"); echo $JOB
```

2. Progress events:

```bash
curl -sN localhost:8000/v1/jobs/$JOB/events | grep -E '^event:|"node"'
```

3. Answer, claim table and cost:

```bash
curl -s localhost:8000/v1/jobs/$JOB | python3 -c "
import json,sys; j=json.load(sys.stdin); r=j['result']
print(r['answer'][:900]); print()
for c in r['claims'][:8]: print(f\"{c['label']:<14} {c['claim'][:80]}\")
print(); print(j['summary'])"
```

Then ask the same question again without `no_cache` to show an instant semantic-cache hit:

```bash
curl -s -XPOST localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question": "what is the role of MGMT in temozolomide resistance in glioblastoma"}'
```

4. Audit log of that job:

```bash
uv run bioagent audit verify $JOB && uv run bioagent audit show $JOB | head -20
```

5. Kubernetes scaling (about 5 minutes end to end; record separately and cut):

```bash
make k8s-up k8s-load-image k8s-deploy
kubectl port-forward svc/bioagent-bioagent-api 8000:8000 &
for i in $(seq 1 10); do curl -s -XPOST localhost:8000/v1/ask -H 'content-type: application/json' \
  -d "{\"question\": \"Replay scaling test question number $i about AMPK?\"}" >/dev/null; done
kubectl get deploy bioagent-bioagent-worker -w     # watch 1 -> 4 -> 1
make k8s-down
```

## Fallbacks

* Live Gemini call slow or quota exhausted: open the saved MGMT run in the UI
  (http://localhost:8000/ui?run=20261112-083229-fd26dc&claim=8, or pick it under "Saved runs"),
  or restart the worker with `WORKER_PIPELINE=replay` and ask again for a live-looking run.
* No disk for kind: show the scaling table in `eval/reports/serving_week8.md`.
