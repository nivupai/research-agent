"""``bioagent audit verify <run or job id>`` and ``bioagent audit show <run or job id>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioagent.config import get_settings
from bioagent.security.audit import find_log, verify_log

SUMMARY_FIELDS = {
    "run_start": ("question_chars", "defenses", "attempt"),
    "run_resume": ("attempt",),
    "plan": ("tasks", "notes"),
    "tool_call": ("role", "tool", "allowed", "check", "reason", "args_hash", "latency_ms"),
    "llm_call": ("role", "model", "input_tokens", "output_tokens", "cost_usd", "latency_ms"),
    "verification": ("groundedness", "claims", "removed_claims", "invalid_citations"),
    "output_check": ("flagged", "unknown_pmids", "instruction_echoes"),
    "run_end": ("cited_pmids", "llm_calls", "tool_calls", "cost_usd", "wall_seconds"),
    "run_failed": ("error_type",),
}


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    au = sub.add_parser("audit", help="inspect and verify a run's hash-chained audit log")
    asub = au.add_subparsers(dest="audit_command", required=True)
    for name, help_ in (("verify", "check the hash chain"), ("show", "print the entries")):
        p = asub.add_parser(name, help=help_)
        p.add_argument("ident", help="run id, job id or path to a .jsonl audit log")
        p.add_argument("--dir", type=Path, help="audit directory (default AUDIT_DIR)")
        if name == "show":
            p.add_argument("--json", action="store_true", help="raw JSON lines")


def main(args: argparse.Namespace) -> int:
    directory = args.dir or Path(get_settings().audit_dir)
    path = find_log(directory, args.ident)
    if args.audit_command == "verify":
        result = verify_log(path)
        status = "OK" if result.ok else "TAMPERED"
        detail = f" ({result.error})" if result.error else ""
        print(f"{status}: {path} - {result.entries} entries{detail}")
        return 0 if result.ok else 1
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if args.json:
            print(line)
            continue
        entry = json.loads(line)
        data = entry["data"]
        keys = SUMMARY_FIELDS.get(entry["type"], tuple(data))
        shown = {k: data[k] for k in keys if k in data and data[k] not in (None, [], "")}
        print(f"{entry['seq']:>4} {entry['type']:<13} {json.dumps(shown, default=str)[:180]}")
    return 0
