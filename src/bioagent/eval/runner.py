"""Resumable suite runs: ``results/eval/<suite>/<run_id>/``.

* ``manifest.json``: config, its hash, the item ids and the run status.
* ``items/<item_id>__r<k>.json``: one agent run (record) plus its scores, keyed by
  (question, repeat, config hash). Existing ok results are reused, so a run interrupted by
  quota exhaustion continues where it stopped (``bioagent eval run --resume <run_id>``).

Eval profile (CI-feasible, identical locally): the agent's ``search_corpus`` is served from
the isolated BM25 fixture index (data/eval/fixture); live PubMed/Europe PMC, the sandbox and
the Neo4j graph are off by default, so results depend only on the committed eval set, the
fixture texts and the models.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from bioagent.agents import prompts
from bioagent.agents.tools import MCPToolHub, ToolProvider
from bioagent.config import Settings
from bioagent.eval.cache import JSONCache, item_key, stable_hash
from bioagent.eval.dataset import DATASET_VERSION, EvalItem, load_all
from bioagent.eval.judge import JUDGE_VERSION, LLMJudge
from bioagent.eval.scoring import agent_question, nli_metrics, poisoned_claims_flagged, score_item
from bioagent.llm.providers import QuotaExhaustedError, make_chat_model, parse_spec

FIXTURE_DIR = Path("data/eval/fixture")
RESULTS_DIR = Path("results/eval")
CACHE_PATH = Path("data/eval/cache.sqlite")
FLASH_LITE = "gemini:gemini-3.5-flash-lite"


class EvalPaused(RuntimeError):
    """Quota exhausted (or interrupted); resume later with the printed run id."""


@dataclass
class EvalConfig:
    suite: str
    repeats: int = 1
    provider: str = "gemini"
    planner: str = FLASH_LITE
    specialist: str = FLASH_LITE
    synth: str = FLASH_LITE
    judge: str = FLASH_LITE
    retrieval: str = "fixture-bm25"
    graph: bool = False
    agent_verify: bool = False
    defenses: bool = True  # SECURITY_DEFENSES for the agent (off: baseline for attack rates)
    nli: bool = False  # scoring only (citation support, groundedness); not in the hash
    dataset_version: str = DATASET_VERSION
    judge_version: str = JUDGE_VERSION
    prompts_hash: str = field(default_factory=lambda: stable_hash(inspect.getsource(prompts)))

    @classmethod
    def for_provider(cls, suite: str, provider: str, settings: Settings, **kw: Any) -> EvalConfig:
        if provider == "ollama":
            spec = f"ollama:{settings.ollama_model}"
            return cls(
                suite=suite,
                provider=provider,
                planner=spec,
                specialist=spec,
                synth=spec,
                judge=spec,
                **kw,
            )
        return cls(suite=suite, provider=provider, **kw)

    def agent_hash(self) -> str:
        """Everything that changes an agent run (not repeats, judge or scoring options)."""
        keys = (
            "provider",
            "planner",
            "specialist",
            "synth",
            "retrieval",
            "graph",
            "agent_verify",
            "defenses",
            "prompts_hash",
        )
        return stable_hash({k: getattr(self, k) for k in keys})


def suite_items(root: Path, suite: str, log: Callable[[str], None] = print) -> list[EvalItem]:
    items = load_all(root, log)
    suites = yaml.safe_load((root / "eval" / "suites.yaml").read_text())
    if suite not in suites:
        raise ValueError(f"unknown suite {suite!r}; known: {[k for k in suites if k != 'version']}")
    ids = suites[suite]["ids"]
    if ids == "all":
        return items
    if isinstance(ids, str) and ids.startswith("category:"):
        category = ids.split(":", 1)[1]
        return [i for i in items if i.category == category]
    by_id = {i.id: i for i in items}
    chosen = [by_id[i] for i in ids if i in by_id]
    # user-provided injection items join the smoke suite automatically (first 4)
    if suite == "smoke" and not any(i.startswith("injection-") for i in ids):
        chosen += [i for i in items if i.category == "injection"][:4]
    return chosen


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def item_path(run_dir: Path, item: EvalItem, repeat: int) -> Path:
    return run_dir / "items" / f"{item.id}__r{repeat}.json"


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((run_dir / "items").glob("*.json"))]


def eval_settings(base: Settings, config: EvalConfig, run_dir: Path) -> Settings:
    return base.model_copy(
        update={
            "planner_model": config.planner,
            "specialist_model": config.specialist,
            "synth_model": config.synth,
            "literature_model": None,
            "chemistry_model": None,
            "analysis_model": None,
            "verify_claims": config.agent_verify,
            "security_defenses": config.defenses,
            "runs_dir": run_dir / "agent_runs",
            "checkpoint_db": run_dir / "checkpoints.sqlite",
            "mcp_log_dir": run_dir / "logs",
        }
    )


def eval_servers(
    config: EvalConfig, fixture: Path
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    servers = {
        "retrieval": "bioagent.mcp_servers.retrieval",
        "chemistry": "bioagent.mcp_servers.chemistry",
    }
    if config.graph:
        servers["graph"] = "bioagent.mcp_servers.graph"
    env = {
        "retrieval": {
            "CORPUS_DB": str((fixture / "corpus.sqlite").resolve()),
            "INDEX_DIR": str(fixture.resolve()),
            "RETRIEVAL_FORCE_MODE": "bm25",
        }
    }
    return servers, env


def make_verifier(enabled: bool, settings: Settings) -> Any:
    if not enabled:
        return None
    from bioagent.verify.backends import make_scorer, windowed
    from bioagent.verify.verifier import ClaimVerifier, VerifyConfig

    return ClaimVerifier(
        make_scorer(settings),
        VerifyConfig(
            entail_threshold=settings.nli_entail_threshold,
            policy="flag",
            windowed=windowed(settings.verifier_backend),
        ),
    )


async def score_record(
    item: EvalItem, record: dict[str, Any], judge: LLMJudge | None, verifier: Any
) -> dict[str, Any]:
    scores = await score_item(item, record, judge)
    if verifier is not None:
        nli = await nli_metrics(record, verifier)
        flagged = poisoned_claims_flagged(item, record, nli)
        scores |= {k: v for k, v in nli.items() if k != "claims_detail"}
        if flagged is not None:
            scores["poisoned_claims_flagged"] = flagged
    return scores


async def run_suite(
    root: Path,
    base: Settings,
    config: EvalConfig,
    *,
    run_id: str | None = None,
    results_dir: Path = RESULTS_DIR,
    fixture: Path = FIXTURE_DIR,
    limit: int | None = None,
    log: Callable[[str], None] = print,
    ask_fn: Any = None,
) -> Path:
    """Run (or resume) a suite; returns the run directory. Raises EvalPaused on quota."""
    from bioagent.agents.runner import ask

    ask_fn = ask_fn or ask
    run_id = run_id or new_run_id()
    run_dir = results_dir / config.suite / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        config = EvalConfig(**{**manifest["config"], "nli": config.nli})
    items = suite_items(root, config.suite, log)
    agent_hash = config.agent_hash()
    manifest = {
        "run_id": run_id,
        "suite": config.suite,
        "config": asdict(config),
        "agent_hash": agent_hash,
        "items": [i.id for i in items],
        "status": "running",
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    if not (fixture / "corpus.sqlite").exists():
        raise FileNotFoundError(f"{fixture} missing: run scripts/build_eval_fixture.py first")

    settings = eval_settings(base, config, run_dir)
    cache = JSONCache(root / CACHE_PATH)
    judge_spec = parse_spec(config.judge)
    judge = LLMJudge(make_chat_model(judge_spec, settings), judge_spec, cache)
    verifier = make_verifier(config.nli, base)
    servers, env = eval_servers(config, fixture)
    todo = [
        (item, r)
        for r in range(config.repeats)  # repeat-major: partial runs still cover every question
        for item in items
        if not _done(item_path(run_dir, item, r), item, r, agent_hash)
    ]
    if limit is not None:
        todo = todo[:limit]
    log(
        f"eval {config.suite}/{run_id}: {len(items)} items x {config.repeats} repeats, "
        f"{len(todo)} to run (agent config {agent_hash})"
    )
    started = time.monotonic()
    try:
        async with MCPToolHub(servers, log_dir=run_dir / "logs", server_env=env) as hub:

            @asynccontextmanager
            async def shared(_settings: Settings) -> AsyncIterator[ToolProvider]:
                yield hub

            for n, (item, repeat) in enumerate(todo, start=1):
                path = item_path(run_dir, item, repeat)
                t0 = time.monotonic()
                try:
                    record = await ask_fn(settings, agent_question(item), tools_factory=shared)
                    status, error = "ok", None
                except QuotaExhaustedError:
                    raise
                except Exception as exc:  # one failed item must not stop the suite
                    record, status, error = {}, "error", f"{type(exc).__name__}: {exc}"[:500]
                scores = await score_record(item, record, judge, verifier) if record else {}
                result = {
                    "key": item_key(item.id, item.question, repeat, agent_hash),
                    "item_id": item.id,
                    "category": item.category,
                    "repeat": repeat,
                    "agent_hash": agent_hash,
                    "status": status,
                    "error": error,
                    "scores": scores,
                    "judge_stats": judge.stats.update().get("llm_calls", [])[-3:],
                    "record": _slim(record),
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))
                acc = scores.get("accuracy")
                log(
                    f"  [{n}/{len(todo)}] {item.id} r{repeat}: {status} "
                    f"acc={acc} ({time.monotonic() - t0:.0f}s)"
                )
    except QuotaExhaustedError as exc:
        _finish(manifest_path, "paused", time.monotonic() - started)
        cache.close()
        raise EvalPaused(
            f"quota exhausted ({exc}); resume with: bioagent eval run --suite {config.suite} "
            f"--resume {run_id}"
        ) from exc
    _finish(manifest_path, "completed" if limit is None else "partial", time.monotonic() - started)
    cache.close()
    return run_dir


def _done(path: Path, item: EvalItem, repeat: int, agent_hash: str) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return bool(
        data.get("status") == "ok"
        and data.get("key") == item_key(item.id, item.question, repeat, agent_hash)
    )


def _finish(manifest_path: Path, status: str, seconds: float) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = status
    manifest["wall_seconds"] = round(manifest.get("wall_seconds", 0.0) + seconds, 1)
    manifest["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=1))


def _slim(record: dict[str, Any]) -> dict[str, Any]:
    """Keep what scoring and reports need (drop per-call LLM logs and task internals)."""
    keep = (
        "run_id",
        "answer",
        "draft_answer",
        "citations",
        "cited_evidence",
        "evidence_keys",
        "verification",
        "errors",
        "stats",
        "tool_calls",
        "plan",
        "models",
    )
    return {k: record[k] for k in keep if k in record}


async def rescore(
    root: Path, run_dir: Path, base: Settings, *, nli: bool, log: Callable[[str], None] = print
) -> int:
    """Re-score stored records (judge calls are cached) e.g. to add NLI metrics."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = EvalConfig(**{**manifest["config"], "nli": nli})
    items = {i.id: i for i in load_all(root, log)}
    cache = JSONCache(root / CACHE_PATH)
    spec = parse_spec(config.judge)
    judge = LLMJudge(make_chat_model(spec, base), spec, cache)
    verifier = make_verifier(nli, base)
    n = 0
    for path in sorted((run_dir / "items").glob("*.json")):
        data = json.loads(path.read_text())
        item = items.get(data["item_id"])
        if item is None or data["status"] != "ok":
            continue
        data["scores"] = await score_record(item, data["record"], judge, verifier)
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))
        n += 1
    cache.close()
    return n


def items_by_id(root: Path, ids: Sequence[str]) -> list[EvalItem]:
    by_id = {i.id: i for i in load_all(root, lambda _m: None)}
    return [by_id[i] for i in ids if i in by_id]
