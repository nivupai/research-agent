"""Tool-call policy: who may call what, with which arguments.

Checks, in order (the first failure denies the call; denials go back to the agent as a tool
error and into the audit log):

1. role allowlist - the specialist's tool scope (``TOOL_SCOPES``);
2. code execution - ``run_python`` only from the analysis specialist, and the code must not
   reach the network (sockets, HTTP clients, subprocesses, URLs);
3. identifiers - PMIDs and graph edge ids in arguments must come from the question, the task,
   earlier tasks' notes or results already retrieved in this task;
4. URLs - an argument may only contain a URL that appears in the question or the task;
5. untrusted origin - a free-text argument that is mostly a verbatim copy of retrieved
   (untrusted) content and barely overlaps the question/task is denied; the specialist must
   rephrase the request in its own terms or leave it to the planner.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PMID_RE = re.compile(r"(?<!\d)\d{6,9}(?!\d)")
EDGE_ID_RE = re.compile(r"\b[A-Z_]+\|[A-Za-z]+:[\w.-]+\|[A-Za-z]+:[\w.-]+(?:\|[\w.|-]+)?")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s\"'<>]+", re.I)
NETWORK_CODE_RE = re.compile(
    r"\b(?:socket|urllib|requests|httpx|aiohttp|http\.client|ftplib|smtplib|telnetlib|"
    r"subprocess|os\.system|os\.popen|pty|paramiko)\b|https?://",
    re.I,
)
ID_ARGS = {"pmids", "pmid", "edge_id", "ids"}
CODE_TOOLS = {"run_python"}
CODE_ROLES = {"analysis"}
COPY_SPAN = 40  # characters of verbatim untrusted text that count as copying
_WORD_RE = re.compile(r"[a-z0-9]{4,}")


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = "allowed"
    check: str = "ok"


@dataclass
class TaskContext:
    """What a specialist legitimately knows (trusted) vs. has read (untrusted)."""

    question: str
    task: str
    notes: Sequence[str] = ()
    untrusted: list[str] = field(default_factory=list)
    known_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for text in (self.question, self.task, *self.notes):
            self.known_ids |= ids_in(text)

    @property
    def trusted_text(self) -> str:
        return "\n".join((self.question, self.task, *self.notes))

    def observe(self, tool_result: Any) -> None:
        """Remember a tool result: its identifiers become usable, its text is untrusted."""
        text = tool_result if isinstance(tool_result, str) else json.dumps(tool_result, default=str)
        self.known_ids |= ids_in(text)
        self.untrusted.append(text)


def ids_in(text: str) -> set[str]:
    return set(PMID_RE.findall(text)) | set(EDGE_ID_RE.findall(text))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        yield str(value)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def copied_from_untrusted(arg: str, ctx: TaskContext) -> bool:
    """Mostly a verbatim copy of retrieved text and mostly absent from the question/task."""
    text = _norm(arg)
    if len(text) < COPY_SPAN or not ctx.untrusted:
        return False
    untrusted = _norm("\n".join(ctx.untrusted))
    trusted = _norm(ctx.trusted_text)
    spans = [text[i : i + COPY_SPAN] for i in range(0, len(text) - COPY_SPAN + 1, 10)]
    if not any(s in untrusted and s not in trusted for s in spans):
        return False
    words = set(_WORD_RE.findall(text))
    if not words:
        return False
    in_untrusted = sum(w in untrusted for w in words) / len(words)
    in_trusted = sum(w in trusted for w in words) / len(words)
    return in_untrusted >= 0.8 and in_trusted < 0.3


class ToolPolicy:
    def __init__(self, scopes: Mapping[str, Sequence[str]]) -> None:
        self.scopes = {role: set(tools) for role, tools in scopes.items()}

    def check(self, role: str, tool: str, args: Mapping[str, Any], ctx: TaskContext) -> Decision:
        if tool not in self.scopes.get(role, set()):
            return Decision(
                False, f"tool '{tool}' is not available to the {role} specialist", "role_allowlist"
            )
        if tool in CODE_TOOLS:
            if role not in CODE_ROLES:
                return Decision(
                    False, "code may only be run by the analysis specialist", "code_role"
                )
            code = " ".join(_strings(args))
            if NETWORK_CODE_RE.search(code):
                return Decision(
                    False, "code must not use the network or start processes", "code_network"
                )
        for name, value in args.items():
            if name in ID_ARGS:
                unknown = sorted(i for i in ids_in(" ".join(_strings(value))) - ctx.known_ids)
                if unknown:
                    return Decision(
                        False,
                        f"identifiers {unknown[:5]} were not in the question, the task or "
                        "earlier results; use ids from retrieved results",
                        "unknown_ids",
                    )
        trusted = ctx.trusted_text
        for arg in _strings(dict(args)):
            for url in URL_RE.findall(arg):
                if url not in trusted:
                    return Decision(
                        False, "URLs in tool arguments must come from the question", "url"
                    )
            if tool not in CODE_TOOLS and copied_from_untrusted(arg, ctx):
                return Decision(
                    False,
                    "this argument is copied from retrieved content rather than the task; "
                    "rephrase it from the question, or leave it to the planner",
                    "untrusted_origin",
                )
        return Decision(True)
