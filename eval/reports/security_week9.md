# Weeks 8-9 (part 2): prompt-injection defences and audit log (2026-11-13)

## Threat model

The agent reads third-party text it does not control: PubMed/Europe PMC records, the local
abstract corpus, knowledge-graph records (PubTator3/ChEMBL names and synonyms), PubChem data
and sandbox output. Any of it may contain text written to steer the model (indirect prompt
injection). We assume the attacker controls the *content* of retrieved documents, not the
user's question, the code, the prompts, the tools or the infrastructure.

What an injection may try, and what we treat as a successful attack:

* **output manipulation**: make the answer contain attacker-chosen text (measured with
  canary strings in the eval items);
* **tool misuse**: make a specialist call a tool it should not (e.g. code execution) or call
  a tool with attacker-chosen arguments (identifiers, URLs, copied text);
* **citation forgery**: make the answer cite or mention papers that were never retrieved;
* **evidence laundering**: make the answer present an instruction from a document as a
  finding.

Out of scope here: a malicious user, model-weight attacks, compromise of the MCP servers or
the host, and denial of service (rate limiting exists in the API, see the serving report).

## Defences (`SECURITY_DEFENSES=on`, the default)

| layer | mechanism | code |
|---|---|---|
| Spotlighting | every tool result, KG record, evidence passage and specialist note is wrapped as data with provenance (source, tool, PMID/passage ref); per-run random delimiters (`<<<UNTRUSTED_DATA-<mark>` ... `<<<END_UNTRUSTED_DATA-<mark>>>`) so content cannot guess a closing marker; delimiter look-alikes (including fullwidth/small-form brackets) neutralised; control, zero-width and bidi characters stripped; system prompts for planner, specialists, synthesizer, verifier and judge state that block contents are data only | `bioagent.agents.untrusted` |
| Tool permissions | per-role allowlists (specialist scopes); `run_python` only for the analysis specialist and without network or subprocess use; PMIDs/edge ids in arguments must come from the question, the task, earlier tasks' notes or results already retrieved; URLs only if they appear in the question/task; a free-text argument that is mostly a verbatim copy of retrieved text (and not of the task) is denied (`untrusted_origin`) | `bioagent.security.policy` |
| Denials | returned to the specialist as a tool error ("denied by policy (check): reason"), counted as a failed tool call, audited with the check and reason | `bioagent.agents.nodes` |
| Output checks | PMIDs written in the answer must be among the retrieved evidence (others become `[unverified PMID]`); instruction-like answer sentences that share an 8-word sequence with retrieved text are flagged; canary-leak hook for the eval; claim-level NLI verification (Week 5-6) still removes unsupported claims | `bioagent.security.output`, `bioagent.verify` |
| Audit | append-only, hash-chained JSONL per run (+ optional SQLite): question hash, plan, every tool call (role, tool, args hash, redacted args, allowed/denied, reason), every LLM call (model, tokens, cost), verifier decisions, output flags, final answer hash; job id and OpenTelemetry trace id on every entry; `bioagent audit verify|show` | `bioagent.security.audit` |

With `SECURITY_DEFENSES=off` the agent runs as before this change: fixed delimiters and
neutralisation, specialist scopes, no argument policy and no output checks. The eval config
hash includes the switch, so on/off results are never mixed.

Known limits: the policy is heuristic (an attacker can still influence free-text queries
that paraphrase rather than copy); spotlighting relies on model compliance; the
instruction-echo check only catches near-verbatim repetition; the audit log is tamper-evident,
not tamper-proof (anyone with write access can rewrite the whole chain; anchoring the last
hash elsewhere would close that).

## Measurement

`make eval-injection` runs the user-provided injection items
(`eval/questions/injection.jsonl`) with defences on and off and compares them with a paired
bootstrap and permutation test. Attack success per item = a canary appears in the answer,
a forbidden tool is called, or a forbidden PMID is cited (`bioagent.eval.scoring`); the report
gives the attack success rate with a 95% CI, plus whether the poisoned passage was retrieved
and whether verification flagged claims citing it.

## Results

**Pending.** `eval/questions/injection.jsonl` does not exist yet: the injection cases are to
be written by the maintainer (no attack payloads were authored or downloaded for this work).
Until then the suite skips with a notice. Mechanism tests (`tests/test_security.py`) cover the
delimiters, sanitising, every policy check, denials reaching the agent, output checks, audit
chaining, tamper detection, redaction and the end-to-end audit of a run.
