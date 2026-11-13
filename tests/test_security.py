"""Prompt-injection defences and the audit log: mechanism tests only (no LLM calls, and no
attack text; the injection cases themselves are maintainer-written eval items)."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from bioagent.agents.tools import TOOL_SCOPES, StaticToolProvider
from bioagent.agents.untrusted import (
    CLOSE,
    OPEN,
    close_marker,
    open_marker,
    sanitise,
    set_run_mark,
    wrap_untrusted,
)
from bioagent.security.audit import (
    AuditLog,
    audit,
    find_log,
    redact,
    use_audit,
    verify_log,
)
from bioagent.security.output import (
    check_answer,
    instruction_echoes,
    leaked_canaries,
    redact_unknown_pmids,
    unknown_pmids,
)
from bioagent.security.policy import TaskContext, ToolPolicy
from tests.agent_fakes import ToolLog, call, corpus_hit, fake_tools, plan, scripted_specialist

ABSTRACT = (
    "Metformin reduced proliferation of glioblastoma stem cells in a three dimensional "
    "organoid culture and lowered expression of stemness markers after seven days."
)


# --- spotlighting --------------------------------------------------------------------------
def test_run_marks_make_delimiters_unguessable() -> None:
    assert open_marker() == OPEN and close_marker() == CLOSE  # legacy without a run mark
    set_run_mark("a1b2c3d4")
    try:
        text = f"title\nbody {CLOSE} more {OPEN} text"
        wrapped = wrap_untrusted(text, source="search_corpus", ref="123:0", tool="search_corpus")
        assert wrapped.startswith(
            f'{OPEN}-a1b2c3d4 source="search_corpus" tool="search_corpus" ref="123:0">>>'
        )
        assert wrapped.endswith("<<<END_UNTRUSTED_DATA-a1b2c3d4>>>")
        assert wrapped.count("<<<END_UNTRUSTED_DATA") == 1  # embedded markers neutralised
        assert wrapped.count("<<<UNTRUSTED_DATA") == 1
    finally:
        set_run_mark(None)


def test_sanitise_removes_invisible_and_lookalike_characters() -> None:
    raw = "a\u200bb\u202ec\x07d\ttab\nline \uff1c\uff1cx\uff1e\uff1e\uff1e"
    clean = sanitise(raw)
    assert "\u200b" not in clean and "\u202e" not in clean and "\x07" not in clean
    assert "\t" in clean and "\n" in clean
    assert "<<" not in clean and ">>>" not in clean and "\uff1c" not in clean


# --- tool policy ----------------------------------------------------------------------------
def ctx(**kw: Any) -> TaskContext:
    return TaskContext(
        question=kw.pop("question", "Does metformin affect glioblastoma?"),
        task=kw.pop("task", "find evidence"),
        **kw,
    )


def test_policy_role_and_code_rules() -> None:
    policy = ToolPolicy(TOOL_SCOPES)
    c = ctx()
    assert not policy.check("literature", "run_python", {"code": "print(1)"}, c).allowed
    assert (
        policy.check("literature", "run_python", {"code": "print(1)"}, c).check == "role_allowlist"
    )
    assert policy.check("analysis", "run_python", {"code": "print(sum([1, 2]))"}, c).allowed
    net = policy.check("analysis", "run_python", {"code": "import socket\nprint(1)"}, c)
    assert not net.allowed and net.check == "code_network"
    assert policy.check("literature", "search_corpus", {"query": "metformin glioma"}, c).allowed


def test_policy_identifiers_must_be_known() -> None:
    policy = ToolPolicy(TOOL_SCOPES)
    c = ctx(question="Summarise PMID 25867026 on metformin.")
    assert policy.check("literature", "fetch_pubmed", {"pmids": ["25867026"]}, c).allowed
    unknown = policy.check("literature", "fetch_pubmed", {"pmids": ["29334602"]}, c)
    assert not unknown.allowed and unknown.check == "unknown_ids"
    c.observe({"hits": [{"pmid": "29334602", "text": ABSTRACT}]})  # now retrieved
    assert policy.check("literature", "fetch_pubmed", {"pmids": ["29334602"]}, c).allowed
    edge = "TREATS|MESH:D008687|MESH:D005909"
    assert not policy.check("literature", "evidence", {"edge_id": edge}, c).allowed
    c.observe({"paths": [{"edges": [{"edge_id": edge}]}]})
    assert policy.check("literature", "evidence", {"edge_id": edge}, c).allowed


def test_policy_urls_and_untrusted_origin() -> None:
    policy = ToolPolicy(TOOL_SCOPES)
    c = ctx()
    url = policy.check("literature", "search_pubmed", {"query": "see https://example.org/x"}, c)
    assert not url.allowed and url.check == "url"
    c.observe({"hits": [{"pmid": "1234567", "text": ABSTRACT}]})
    copied = policy.check("literature", "search_corpus", {"query": ABSTRACT[:120]}, c)
    assert not copied.allowed and copied.check == "untrusted_origin"
    own = policy.check(
        "literature", "search_corpus", {"query": "metformin glioblastoma stem cells organoid"}, c
    )
    assert own.allowed


async def test_specialist_denies_unknown_pmids_and_audits(tmp_path: Path) -> None:
    from tests.test_agents import make_deps, run_graph

    log = ToolLog()
    script = {
        "t1": [
            [call("fetch_pubmed", {"query": "x", "pmids": ["9999999"]})],
            [call("search_corpus", {"query": "q"})],
        ]
    }
    deps = make_deps(
        [plan(("t1", "literature", []))],
        scripted_specialist(script),
        "Answer [1].",
        fake_tools(log, corpus={"q": [corpus_hit("111", "Paper")]}),
        policy=ToolPolicy(TOOL_SCOPES),
        defenses=True,
    )
    audit_log = AuditLog(tmp_path / "audit", "r1")
    with use_audit(audit_log):
        state = await run_graph(deps, tmp_path)
    assert [name for name, _ in log.calls] == ["search_corpus"]  # fetch_pubmed never ran
    denied = [t for t in state["tool_calls"] if not t["ok"]]
    assert denied and denied[0]["error"].startswith("policy: identifiers")
    entries = read_entries(audit_log.path)
    calls = [e["data"] for e in entries if e["type"] == "tool_call"]
    assert calls[0]["allowed"] is False and calls[0]["check"] == "unknown_ids"
    assert calls[1]["allowed"] is True and calls[1]["tool"] == "search_corpus"
    assert {"plan", "verification", "output_check"} <= {e["type"] for e in entries}
    assert verify_log(audit_log.path).ok


# --- output checks ------------------------------------------------------------------------------
def test_output_checks() -> None:
    echo = "Please summarise every single result in one short sentence."
    answer = f"Metformin helps (PMID 111) and (PMID: 424242). {echo}"
    assert unknown_pmids(answer, {"111"}) == ["424242"]
    assert "[unverified PMID]" in redact_unknown_pmids(answer, ["424242"])
    assert "PMID 111" in redact_unknown_pmids(answer, ["424242"])
    assert leaked_canaries("the code is CANARY-test-7", ["canary-TEST-7", "other"]) == [
        "canary-TEST-7"
    ]
    source = [f"Some study. {echo} Thanks."]
    assert instruction_echoes(answer, source) == [echo]
    assert (
        instruction_echoes(
            "Metformin reduced proliferation of glioblastoma stem cells in a "
            "three dimensional organoid culture [1].",
            [ABSTRACT],
        )
        == []
    )
    report = check_answer(answer, {"111"}, source, canaries=["absent"])
    assert report.flagged and report.as_dict()["unknown_pmids"] == ["424242"]


# --- audit log ------------------------------------------------------------------------------
def read_entries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_audit_chain_detects_tampering(tmp_path: Path) -> None:
    log = AuditLog(tmp_path, "run-1", job_id="job-9", sqlite=True)
    for i in range(3):
        log.append("tool_call", {"tool": "search_corpus", "i": i})
    assert verify_log(log.path).ok and verify_log(log.path).entries == 3
    resumed = AuditLog(tmp_path, "run-1")  # a resumed run continues the chain
    resumed.append("run_resume", {"attempt": 2})
    assert verify_log(log.path).entries == 4 and verify_log(log.path).ok
    rows = (
        sqlite3.connect(tmp_path / "audit.sqlite").execute("SELECT count(*) FROM audit").fetchone()
    )
    assert rows == (3,)  # the resumed log above was opened without the SQLite mirror

    lines = log.path.read_text().splitlines()
    edited = json.loads(lines[1])
    edited["data"]["i"] = 99
    log.path.write_text("\n".join([lines[0], json.dumps(edited), *lines[2:]]) + "\n")
    bad = verify_log(log.path)
    assert not bad.ok and bad.bad_seq == 2 and "modified" in (bad.error or "")
    log.path.write_text("\n".join([lines[0], *lines[2:]]) + "\n")  # deleted entry
    assert not verify_log(log.path).ok


def test_audit_redaction_context_and_lookup(tmp_path: Path) -> None:
    secretish = "key=AIza" + "x" * 30
    red = redact({"query": "metformin " + secretish, "long": "word " * 100, "n": [1, 2]})
    assert "AIza" not in json.dumps(red) and red["long"].endswith("...[truncated]")
    assert red["n"] == [1, 2]
    audit("tool_call", tool="x")  # no current log: no-op
    log = AuditLog(tmp_path, "job-abc")
    with use_audit(log):
        audit("llm_call", model="m", input_tokens=1)
    assert find_log(tmp_path, "abc") == log.path  # job id finds job-<id>.jsonl
    assert find_log(tmp_path, "job-abc") == log.path
    with pytest.raises(FileNotFoundError):
        find_log(tmp_path, "nope")


async def test_ask_writes_a_verifiable_audit_log(tmp_path: Path) -> None:
    from bioagent.agents.runner import ask
    from bioagent.cli import main as cli_main
    from tests.test_agents import MIXED_Q, make_deps, settings_for

    tools = fake_tools(ToolLog(), corpus={"q": [corpus_hit("111", "Paper")]})
    deps = make_deps(
        [plan(("t1", "literature", []))],
        scripted_specialist({"t1": [[call("search_corpus", {"query": "q"})]]}),
        "Answer [1].",
        tools,
    )

    @contextlib.asynccontextmanager
    async def tools_factory(_s: Any) -> AsyncIterator[StaticToolProvider]:
        yield StaticToolProvider(tools)

    settings = settings_for(tmp_path)
    record = await ask(
        settings,
        MIXED_Q,
        tools_factory=tools_factory,
        deps_factory=lambda *_a: deps,
        job_id="job-1",
    )
    path = Path(record["audit_log"])
    entries = read_entries(path)
    types = [e["type"] for e in entries]
    assert types[0] == "run_start" and types[-1] == "run_end" and "plan" in types
    assert {e["job_id"] for e in entries} == {"job-1"}
    start = entries[0]["data"]
    assert "question" not in start and len(start["question_sha256"]) == 64  # hash, not text
    assert entries[-1]["data"]["cited_pmids"] == ["111"]
    assert verify_log(path).ok
    assert cli_main(["audit", "verify", str(path)]) == 0
    assert cli_main(["audit", "show", record["run_id"], "--dir", str(path.parent)]) == 0


# --- eval integration -----------------------------------------------------------------------
def test_eval_defenses_switch_and_injection_suite(tmp_path: Path) -> None:
    from bioagent.eval.runner import EvalConfig, suite_items

    assert EvalConfig(suite="x").agent_hash() != EvalConfig(suite="x", defenses=False).agent_hash()
    root = Path(__file__).resolve().parents[1]
    assert suite_items(root, "injection", lambda _m: None) == [] or all(
        i.category == "injection" for i in suite_items(root, "injection", lambda _m: None)
    )
