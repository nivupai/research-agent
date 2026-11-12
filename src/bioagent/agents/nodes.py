"""Graph nodes: planner, dispatcher, specialists, synthesizer and verifier."""

from __future__ import annotations

import asyncio
import itertools
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Send
from pydantic import BaseModel

from bioagent.agents import prompts
from bioagent.agents.state import (
    AgentState,
    Citation,
    Evidence,
    Plan,
    SpecialistInput,
    Task,
    TaskResult,
)
from bioagent.agents.tools import (
    CHEMBL_COMPOUND_URL,
    ToolProvider,
    extract_evidence,
    render_for_model,
)
from bioagent.agents.untrusted import wrap_untrusted
from bioagent.llm.providers import ModelSpec, QuotaExhaustedError, ainvoke_with_stats
from bioagent.llm.stats import RunStatsCollector
from bioagent.verify.text import (
    CITE_RE,
    ESTIMATE_RE,
    LIST_MARKER_RE,
    cite_numbers,
    is_structural,
    split_sentences,
)

if TYPE_CHECKING:
    from bioagent.verify.verifier import ClaimVerifier

MAX_SPECIALIST_STEPS = 6
MAX_EVIDENCE_FOR_SYNTH = 40
EVIDENCE_CHARS_FOR_SYNTH = 1100
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
GRAPH_SOURCES = frozenset({"graph", "chembl"})


@dataclass
class RoleModel:
    spec: ModelSpec
    model: BaseChatModel


@dataclass
class AgentDeps:
    planner: RoleModel
    specialist: RoleModel
    synth: RoleModel
    tools: ToolProvider
    max_steps: int = MAX_SPECIALIST_STEPS
    max_retries: int = 5
    # Evidence budget for the synthesizer prompt (smaller for small local models).
    synth_max_evidence: int = MAX_EVIDENCE_FOR_SYNTH
    synth_evidence_chars: int = EVIDENCE_CHARS_FOR_SYNTH
    sleep: Callable[[float], Awaitable[None]] | None = None
    # Per-specialist model overrides (e.g. a stronger model for literature only).
    specialists: dict[str, RoleModel] = field(default_factory=dict)
    # Claim-level verification (NLI); None keeps only the citation checks.
    claim_verifier: ClaimVerifier | None = None
    # Test hook: called at the start of each specialist (e.g. to inject a failure).
    on_specialist_start: Callable[[str], None] | None = field(default=None)

    def specialist_for(self, kind: str) -> RoleModel:
        return self.specialists.get(kind, self.specialist)


def message_text(message: BaseMessage) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):  # langchain-core >= 1: str-like property
        return str(text).strip()
    if callable(text):  # older langchain-core: method
        return str(text()).strip()
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
    return "".join(parts).strip()


async def _invoke(
    deps: AgentDeps,
    role: RoleModel,
    runnable: Any,
    messages: list[BaseMessage],
    role_name: str,
    stats: RunStatsCollector,
    task_id: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if deps.sleep is not None:
        kwargs["sleep"] = deps.sleep
    return await ainvoke_with_stats(
        runnable,
        messages,
        spec=role.spec,
        role=role_name,
        stats=stats,
        task_id=task_id,
        max_retries=deps.max_retries,
        **kwargs,
    )


# --- planner ---------------------------------------------------------------------------
# Chemistry/analysis tasks are only kept when the question asks for computation.
_COMPUTE_RE = re.compile(
    r"propert(?:y|ies)|drug[- ]?likeness|lipinski|veber|\bbbb\b|blood[- ]brain|"
    r"\bc?logp\b|tpsa|polar surface|molecular weight|\bsmiles\b|similar(?:ity)?\b|"
    r"substructure|physicochemical|comput|calculat|simulat|statistic|regression|"
    r"dose[- ]response|\bic50\b|half[- ]life|\bplot\b|\bcorrelat",
    re.I,
)
COMPUTE_KINDS = ("chemistry", "analysis")


def asks_for_computation(question: str) -> bool:
    return bool(_COMPUTE_RE.search(question))


def enforce_task_kinds(tasks: list[Task], question: str) -> tuple[list[Task], list[str]]:
    """Drop chemistry/analysis tasks the question did not ask for (and dangling deps)."""
    if asks_for_computation(question):
        return tasks, []
    dropped = {t.id for t in tasks if t.kind in COMPUTE_KINDS}
    if not dropped:
        return tasks, []
    kept = [
        t.model_copy(update={"depends_on": [d for d in t.depends_on if d not in dropped]})
        for t in tasks
        if t.id not in dropped
    ]
    notes = [
        f"dropped {t.kind} task {t.id} (the question asks for no properties or computation)"
        for t in tasks
        if t.id in dropped
    ]
    if not kept:
        kept = [Task(id="t1", kind="literature", description=question)]
        notes.append("no literature task left; planned a single literature task")
    return kept, notes


def make_planner(deps: AgentDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def planner(state: AgentState) -> dict[str, Any]:
        stats = RunStatsCollector()
        structured = deps.planner.model.with_structured_output(Plan, include_raw=True)
        messages: list[BaseMessage] = [
            SystemMessage(prompts.PLANNER),
            HumanMessage(f"Question: {state['question']}"),
        ]
        errors: list[str] = []
        plan: Plan | None = None
        for _attempt in range(2):
            out = await _invoke(deps, deps.planner, structured, messages, "planner", stats)
            parsed = out.get("parsed") if isinstance(out, dict) else out
            if isinstance(parsed, Plan):
                plan = parsed
                break
            error = out.get("parsing_error") if isinstance(out, dict) else "no plan returned"
            errors.append(f"planner output invalid: {str(error)[:300]}")
            messages.append(
                HumanMessage(f"Your plan was invalid ({str(error)[:300]}). Return a valid plan.")
            )
        if plan is None:
            plan = Plan(
                tasks=[Task(id="t1", kind="literature", description=state["question"])],
                rationale="fallback: planner output could not be parsed",
            )
            errors.append("planner fell back to a single literature task")
        tasks, notes = enforce_task_kinds(plan.tasks, state["question"])
        return {
            "plan": [t.model_dump() for t in tasks],
            "plan_rationale": plan.rationale,
            "plan_notes": notes,
            "errors": errors,
            **stats.update(),
        }

    return planner


# --- dispatcher ------------------------------------------------------------------------
async def dispatch(state: AgentState) -> dict[str, Any]:
    """Join point between supersteps; routing happens in ``route_ready``."""
    return {}


def ready_tasks(
    plan: list[dict[str, Any]], results: Mapping[str, TaskResult]
) -> list[dict[str, Any]]:
    done = set(results)
    return [
        t for t in plan if t["id"] not in done and all(d in done for d in t.get("depends_on", []))
    ]


def route_ready(state: AgentState) -> list[Send] | str:
    results = state.get("results", {})
    ready = ready_tasks(state.get("plan", []), results)
    if not ready:
        return "synthesizer"
    return [
        Send(
            task["kind"],
            SpecialistInput(
                question=state["question"],
                task=task,
                dependency_notes=[
                    {"task_id": d, "summary": results[d]["summary"]}
                    for d in task.get("depends_on", [])
                ],
            ),
        )
        for task in ready
    ]


# --- deterministic drug discovery --------------------------------------------------------
DISCOVERY_TOOLS = frozenset({"drugs_targeting", "search_corpus", "drugs_mentioned"})
# Live PubMed adds breadth beyond the local corpus (e.g. statins, kinase inhibitors).
DISCOVERY_PUBMED_QUERY = "{target} activation approved drugs repurposing"
DISCOVERY_QUERIES = (
    "{target} activator drug",
    "{target} inhibitor approved drug",
    "approved drugs repurposed in cancer that activate {target}",
)
DISCOVERY_K = 15
MAX_CANDIDATES = 20


@dataclass
class Discovery:
    evidence: dict[str, Evidence]
    candidates: list[dict[str, Any]]
    prompt: str
    summary: str


async def _run_tool(
    tools: Mapping[str, Any],
    name: str,
    args: dict[str, Any],
    stats: RunStatsCollector,
    task_id: str,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = await tools[name].ainvoke(args)
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        result = {"result": result}
    error = result.get("error")
    stats.add_tool(task_id, name, error is None, started, error=str(error) if error else None)
    return result


def merge_candidates(
    graph_result: Mapping[str, Any],
    mentioned_result: Mapping[str, Any],
    approved_only: bool = False,
) -> list[dict[str, Any]]:
    """Union of graph drugs_targeting hits and drugs annotated in the corpus hits."""
    merged: dict[str, dict[str, Any]] = {}
    for item in graph_result.get("results", []):
        drug = item["drug"]
        entry = merged.setdefault(
            str(drug["id"]),
            {"name": drug["name"], "max_phase": drug.get("max_phase"), "sources": set(),
             "graph_evidence": 0, "corpus_papers": 0},
        )  # fmt: skip
        entry["sources"].add("chembl" if item["source"] == "chembl" else "graph")
        entry["graph_evidence"] += int(item.get("evidence_count") or 0)
    for drug in mentioned_result.get("drugs", []):
        phase = drug.get("max_phase")
        if approved_only and phase is not None and phase < 4:
            continue  # unknown phase is kept: drug classes (e.g. "Salicylates") have none
        entry = merged.setdefault(
            str(drug["id"]),
            {"name": drug["name"], "max_phase": drug.get("max_phase"), "sources": set(),
             "graph_evidence": 0, "corpus_papers": 0},
        )  # fmt: skip
        entry["sources"].add("corpus")
        entry["corpus_papers"] = int(drug.get("papers_in_query") or 0)

    def weight(e: dict[str, Any]) -> int:
        return -(int(e["corpus_papers"]) + int(e["graph_evidence"]))

    both = sorted((e for e in merged.values() if len(e["sources"]) > 1), key=weight)
    graph_only = sorted((e for e in merged.values() if e["sources"] != {"corpus"}
                         and len(e["sources"]) == 1), key=weight)  # fmt: skip
    corpus_only = sorted((e for e in merged.values() if e["sources"] == {"corpus"}), key=weight)
    ranked = both + [
        e for pair in itertools.zip_longest(graph_only, corpus_only) for e in pair if e
    ]
    return [{**e, "sources": sorted(e["sources"])} for e in ranked[:MAX_CANDIDATES]]


async def discover_drugs(
    task: Task, tools: Mapping[str, Any], stats: RunStatsCollector
) -> Discovery:
    """Always run drugs_targeting plus the discovery corpus searches for a drug_target task.

    Graph hits and drugs annotated in the corpus hits are merged into one candidate list,
    so drug classes the graph misses (e.g. statins, salicylates) still reach the specialist.
    """
    target = str(task.drug_target)
    calls: list[tuple[str, dict[str, Any]]] = [
        ("drugs_targeting", {"target": target, "approved_only": task.approved_only, "limit": 20}),
        *(
            ("search_corpus", {"query": q.format(target=target), "k": DISCOVERY_K,
                               "mode": "hybrid_rerank"})
            for q in DISCOVERY_QUERIES
        ),
    ]  # fmt: skip
    if "search_pubmed" in tools:
        calls.append(
            ("search_pubmed", {"query": DISCOVERY_PUBMED_QUERY.format(target=target),
                               "max_results": 10})
        )  # fmt: skip
    results = await asyncio.gather(*(_run_tool(tools, n, a, stats, task.id) for n, a in calls))
    pmids = list(
        dict.fromkeys(
            str(h["pmid"]) for (n, _), r in zip(calls, results, strict=True)
            if n == "search_corpus" for h in r.get("hits", [])
        )
    )  # fmt: skip
    mentioned: dict[str, Any] = {}
    if pmids:
        # unfiltered: approval is applied in merge_candidates (keeps unphased drug classes)
        args = {"pmids": pmids, "approved_only": False, "limit": 50}
        mentioned = await _run_tool(tools, "drugs_mentioned", args, stats, task.id)
        calls.append(("drugs_mentioned", args))
        results.append(mentioned)
    evidence: dict[str, Evidence] = {}
    blocks: list[str] = []
    for (name, args), result in zip(calls, results, strict=True):
        items = extract_evidence(name, result, task.id, args=args)
        evidence.update({e["key"]: e for e in items})
        blocks.append(
            wrap_untrusted(render_for_model(name, result, items), source=name, ref=task.id)
        )
    graph_result = results[0] if "error" not in results[0] else {}
    candidates = merge_candidates(
        graph_result, mentioned if "error" not in mentioned else {}, task.approved_only
    )
    listing = "; ".join(
        f"{c['name']} ({'+'.join(c['sources'])}"
        + (
            f", max_phase {c['max_phase']}"
            if c.get("max_phase") is not None
            else ", drug class or phase unknown"
        )
        + ")"
        for c in candidates
    )
    summary = f"Candidate drugs for {target} (graph + corpus union): {listing or 'none found'}"
    # Drug names come from third-party data, so the candidate list is wrapped as well.
    prompt = (
        f"Discovery results (already run for you: drugs_targeting, "
        f"{len(DISCOVERY_QUERIES)} corpus searches and a PubMed search; do not repeat "
        "them).\n"
        f"{wrap_untrusted(summary, source='discovery', ref=task.id)}\n"
        "Cover every plausible candidate in your findings, including drug classes found only "
        "in the corpus; drop generic metabolites or reagents. Use further searches only to "
        "fill gaps.\n\n" + "\n\n".join(blocks)
    )
    return Discovery(evidence, candidates, prompt, summary)


# --- specialists -----------------------------------------------------------------------
def make_specialist(
    kind: str, deps: AgentDeps
) -> Callable[[SpecialistInput], Awaitable[dict[str, Any]]]:
    async def specialist(inp: SpecialistInput) -> dict[str, Any]:
        task = Task.model_validate(inp["task"])
        if deps.on_specialist_start is not None:
            deps.on_specialist_start(task.id)
        stats = RunStatsCollector()
        role = deps.specialist_for(kind)
        tools = {t.name: t for t in deps.tools.tools_for(kind)}
        bound = role.model.bind_tools(list(tools.values())) if tools else role.model
        notes = "\n\n".join(
            wrap_untrusted(n["summary"], source="specialist_notes", ref=n["task_id"])
            for n in inp["dependency_notes"]
        )
        evidence: dict[str, Evidence] = {}
        computed_count = 0
        discovery: Discovery | None = None
        if kind == "literature" and task.drug_target and set(tools) >= DISCOVERY_TOOLS:
            discovery = await discover_drugs(task, tools, stats)
            evidence.update(discovery.evidence)
        messages: list[BaseMessage] = [
            SystemMessage(prompts.SPECIALISTS[kind]),
            HumanMessage(
                f"Overall question: {inp['question']}\n\nYour task ({task.id}): "
                f"{task.description}"
                + (f"\n\nResults from earlier tasks:\n{notes}" if notes else "")
                + (f"\n\n{discovery.prompt}" if discovery else "")
            ),
        ]
        final: AIMessage | None = None
        steps = 0
        for steps in range(1, deps.max_steps + 1):  # noqa: B007
            ai = await _invoke(deps, role, bound, messages, f"specialist:{kind}", stats, task.id)
            messages.append(ai)
            if not ai.tool_calls:
                final = ai
                break
            for call in ai.tool_calls:
                name = call["name"]
                started = time.monotonic()
                if name not in tools:
                    stats.add_tool(task.id, name, False, started, error="not permitted")
                    content = (
                        f"Tool '{name}' is not available to the {kind} specialist. "
                        f"Available tools: {sorted(tools)}."
                    )
                else:
                    try:
                        result = await tools[name].ainvoke(call["args"])
                    except Exception as exc:  # tool crashed: report to the model
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                    if not isinstance(result, dict):
                        result = {"result": result}
                    error = result.get("error")
                    stats.add_tool(
                        task.id, name, error is None, started, error=str(error) if error else None
                    )
                    items = extract_evidence(
                        name, result, task.id, start=computed_count, args=call["args"]
                    )
                    computed_count += sum(1 for e in items if e["computed"])
                    evidence.update({e["key"]: e for e in items})
                    content = wrap_untrusted(
                        render_for_model(name, result, items), source=name, ref=task.id
                    )
                messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
        if final is None:
            messages.append(
                HumanMessage("Step limit reached. Do not call tools. Write your findings now.")
            )
            final = await _invoke(
                deps,
                role,
                role.model,
                messages,
                f"specialist:{kind}",
                stats,
                task.id,
            )
        result_entry = TaskResult(
            task_id=task.id,
            kind=kind,
            status="done",
            summary=message_text(final) + (f"\n\n{discovery.summary}" if discovery else ""),
            evidence_keys=list(evidence),
            steps=steps,
            error=None,
        )
        return {"results": {task.id: result_entry}, "evidence": evidence, **stats.update()}

    return specialist


# --- synthesizer -----------------------------------------------------------------------
def _interleave(queues: list[list[Evidence]]) -> list[Evidence]:
    out: list[Evidence] = []
    for row in itertools.zip_longest(*queues):
        out.extend(e for e in row if e is not None)
    return out


def select_evidence(state: AgentState, limit: int = MAX_EVIDENCE_FOR_SYNTH) -> list[Evidence]:
    """Share the evidence budget fairly, keep one passage per paper, cap the list.

    Tasks are interleaved round-robin (in plan order) so an early task with many results
    cannot crowd out a later one; within a task, full abstracts alternate with
    knowledge-graph/ChEMBL items so text-mined edges do not push out the papers' text.
    """
    evidence = state.get("evidence", {})
    results = state.get("results", {})
    per_task: list[list[Evidence]] = []
    for task in state.get("plan", []):
        result = results.get(task["id"])
        items = [evidence[k] for k in (result["evidence_keys"] if result else []) if k in evidence]
        abstracts = [e for e in items if e["source"] not in GRAPH_SOURCES]
        graph = [e for e in items if e["source"] in GRAPH_SOURCES]
        per_task.append(_interleave([abstracts, graph]))
    ordered: list[Evidence] = []
    seen_pmids: set[str] = set()
    for e in _interleave(per_task):
        if e["pmid"]:
            if e["pmid"] in seen_pmids:
                continue
            seen_pmids.add(e["pmid"])
        ordered.append(e)
    computed = [e for e in ordered if e["computed"]][: max(1, limit // 4)]
    papers = [e for e in ordered if not e["computed"]]
    return computed + papers[: limit - len(computed)]


def render_evidence(items: list[Evidence], chars: int = EVIDENCE_CHARS_FOR_SYNTH) -> str:
    blocks = []
    for n, e in enumerate(items, start=1):
        if e["computed"]:
            header = f"[{n}] COMPUTED ESTIMATE from {e['source']} ({e['title']})"
        elif e["source"] == "chembl":
            header = f"[{n}] CHEMBL CURATED RECORD ({e['title']})"
        elif e["source"] == "graph":
            header = (
                f"[{n}] PMID {e['pmid']} ({e['year']}) KNOWLEDGE-GRAPH RELATION from this paper"
            )
        else:
            header = f"[{n}] PMID {e['pmid'] or '-'} ({e['year']}) {e['journal'] or ''}".strip()
        text = e["text"][:chars]
        blocks.append(f"{header}\n{wrap_untrusted(text, source=e['source'], ref=e['key'])}")
    return "\n\n".join(blocks)


def _citation_url(e: Evidence) -> str | None:
    if e["pmid"]:
        return PUBMED_URL.format(pmid=e["pmid"])
    if e["source"] == "chembl" and e["passage_id"]:
        return CHEMBL_COMPOUND_URL.format(chembl_id=e["passage_id"])
    return None


def map_citations(answer: str, items: list[Evidence]) -> tuple[str, list[Citation], list[int]]:
    """Renumber [n] markers by first appearance and build the reference list.

    Returns (renumbered answer, citations, invalid numbers). Invalid numbers are rendered as
    ``[?n]`` so they stand out.
    """
    order: dict[int, int] = {}
    invalid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        parts = []
        for n in _numbers(match.group(1)):
            if 1 <= n <= len(items):
                order.setdefault(n, len(order) + 1)
                parts.append(str(order[n]))
            else:
                invalid.append(n)
                parts.append(f"?{n}")
        return "[" + ", ".join(dict.fromkeys(parts)) + "]"

    renumbered = _CITE_RE.sub(replace, answer)
    citations: list[Citation] = []
    for old, new in sorted(order.items(), key=lambda kv: kv[1]):
        e = items[old - 1]
        citations.append(
            Citation(
                n=new,
                key=e["key"],
                pmid=e["pmid"],
                title=e["title"],
                year=e["year"],
                url=_citation_url(e),
                computed=e["computed"],
                source=e["source"],
            )
        )
    return renumbered, citations, sorted(set(invalid))


def make_synthesizer(deps: AgentDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def synthesizer(state: AgentState) -> dict[str, Any]:
        stats = RunStatsCollector()
        items = select_evidence(state, deps.synth_max_evidence)
        notes = "\n\n".join(
            wrap_untrusted(r["summary"], source="specialist_notes", ref=tid)
            for tid, r in state.get("results", {}).items()
        )
        plan_text = "\n".join(
            f"- {t['id']} ({t['kind']}): {t['description']}" for t in state.get("plan", [])
        )
        messages: list[BaseMessage] = [
            SystemMessage(prompts.SYNTHESIZER),
            HumanMessage(
                f"Question: {state['question']}\n\nPlan:\n{plan_text}\n\n"
                f"Specialist notes (untrusted; use only to locate evidence):\n{notes}\n\n"
                f"Evidence ({len(items)} items):\n"
                f"{render_evidence(items, deps.synth_evidence_chars)}\n\n"
                f"Now answer this question, citing evidence numbers: {state['question']}"
            ),
        ]
        if not items:
            messages[-1] = HumanMessage(
                f"Question: {state['question']}\n\nNo evidence was retrieved. Say that the "
                "question could not be answered from the available sources and why."
            )
        ai = await _invoke(deps, deps.synth, deps.synth.model, messages, "synthesizer", stats)
        answer, citations, invalid = map_citations(message_text(ai), items)
        return {
            "answer": answer,
            "citations": citations,
            "verification": {"invalid_citations": invalid},
            **stats.update(),
        }

    return synthesizer


# --- verifier --------------------------------------------------------------------------
# Re-exported for callers that imported these from here.
_CITE_RE = CITE_RE
_numbers = cite_numbers
_is_structural = is_structural
_LIST_MARKER_RE = LIST_MARKER_RE
_ESTIMATE_RE = ESTIMATE_RE


async def verifier(state: AgentState) -> dict[str, Any]:
    """Citation checks: every [n] must map to real evidence, and sentences without a
    citation are flagged unless they are marked as computed estimates. Claim-level NLI
    verification runs on top of this in ``make_verifier``."""
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    numbers = {c["n"] for c in citations}
    sentences = [
        s.strip()
        for line in answer.splitlines()
        if not _is_structural(line)
        for s in split_sentences(_LIST_MARKER_RE.sub("", line.strip()))
        if len(s.strip()) > 40
    ]
    uncited = [s for s in sentences if not _CITE_RE.search(s) and not _ESTIMATE_RE.search(s)]
    used = {n for m in _CITE_RE.finditer(answer) for n in _numbers(m.group(1))}
    previous = state.get("verification", {})
    invalid = sorted(set(previous.get("invalid_citations", [])) | (used - numbers))
    return {
        "verification": {
            "invalid_citations": invalid,
            "all_citations_resolve": not invalid and "[?" not in answer,
            "uncited_sentences": uncited,
            "sentences_checked": len(sentences),
        }
    }


def make_verifier(deps: AgentDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def verify(state: AgentState) -> dict[str, Any]:
        base = await verifier(state)
        answer = state.get("answer", "")
        citations = state.get("citations", [])
        if deps.claim_verifier is None or not answer.strip():
            base["verification"]["claim_verification"] = "off"
            return base
        stats = RunStatsCollector()

        async def call(schema: type[BaseModel], messages: list[BaseMessage]) -> BaseModel | None:
            structured = deps.synth.model.with_structured_output(schema, include_raw=True)
            try:
                out = await _invoke(deps, deps.synth, structured, messages, "verifier", stats)
            except QuotaExhaustedError:
                raise
            except Exception:  # fall back to sentence-level claims
                return None
            parsed = out.get("parsed") if isinstance(out, dict) else out
            return parsed if isinstance(parsed, schema) else None

        result = await deps.claim_verifier.verify(
            answer, citations, state.get("evidence", {}), call=call
        )
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "draft_answer": answer,
            "verification": {
                **base["verification"],
                "claim_verification": "on",
                **result["report"],
            },
            **stats.update(),
        }

    return verify
