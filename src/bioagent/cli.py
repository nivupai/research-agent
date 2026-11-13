"""``bioagent`` command line: ``bioagent ask "question" [--provider gemini|ollama] [--json]``
and ``bioagent ask --resume RUN_ID``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import textwrap
from typing import Any

from bioagent.common.httpclient import quiet_http_logging
from bioagent.config import get_settings
from bioagent.llm.providers import LLMError

LABEL_MARKS = {
    "supported": "ok",
    "computed": "ok (computed)",
    "absence": "absence",
    "not_supported": "NOT SUPPORTED",
    "contradicted": "CONTRADICTED",
}


def format_claims(v: dict[str, Any]) -> list[str]:
    g = v.get("groundedness")
    counts = v.get("counts", {})
    lines = [
        f"Claim verification ({v.get('nli_model')}, policy={v.get('policy')}): "
        f"groundedness {'n/a' if g is None else f'{g:.0%}'} "
        f"({counts.get('supported', 0) + counts.get('computed', 0)}/"
        f"{v.get('checkable_claims', 0)} checkable claims) · "
        + " · ".join(f"{k} {n}" for k, n in counts.items() if n)
        + f" · removed {len(v.get('removed_claims', []))} claim(s), flagged "
        f"{len(v.get('flagged_sentences', []))} sentence(s) · {v.get('seconds')} s",
        "  #  check          claim  [evidence]",
    ]
    for c in v.get("claims", []):
        cites = ",".join(str(n) for n in c["citations"]) or "-"
        mark = LABEL_MARKS.get(c["label"], c["label"])
        claim = textwrap.shorten(c["claim"], 90, placeholder="...")
        lines.append(f"  {c['n']:>2} {mark:<14} {claim}  [{cites}] ({c['check']})")
    return lines


def format_report(record: dict[str, Any]) -> str:
    lines = [f"Run {record['run_id']}  ({record['status']})", "", f"Q: {record['question']}", ""]
    lines.append("Plan:")
    for t in record["plan"]:
        deps = f" (after {', '.join(t['depends_on'])})" if t.get("depends_on") else ""
        lines.append(f"  {t['id']} [{t['kind']}] {t['description']}{deps}")
    lines += ["", "Answer:", ""]
    for para in record["answer"].split("\n"):
        lines.append(textwrap.fill(para, 100) if para.strip() else "")
    lines += ["", "References:"]
    for c in record["citations"]:
        if c["computed"]:
            lines.append(f"  [{c['n']}] Computed estimate: {c['title']} ({c['source']})")
        else:
            lines.append(
                f"  [{c['n']}] {c['title']} ({c['year']}). PMID {c['pmid']}. "
                f"{c['url'] or ''}".rstrip()
            )
    v = record.get("verification", {})
    if v.get("invalid_citations"):
        lines.append(f"  ! citations with no matching evidence: {v['invalid_citations']}")
    if v.get("uncited_sentences"):
        lines.append(f"  ! {len(v['uncited_sentences'])} sentence(s) without a citation")
    if v.get("claim_verification") == "on":
        lines += ["", *format_claims(v)]
    s = record["stats"]
    cost = "n/a" if s["cost_usd"] is None else f"${s['cost_usd']:.4f}"
    if s["cost_usd"] is not None and not s["cost_complete"]:
        cost += " (partial: some model prices unknown)"
    lines += [
        "",
        f"Run stats: {s['passages']} passages · {s['papers']} papers ({s['cited_papers']} cited)"
        f" · {s['tool_calls']} tool calls · {s['wall_seconds']} s · "
        f"{s['input_tokens'] + s['output_tokens']:,} tokens "
        f"({s['input_tokens']:,} in / {s['output_tokens']:,} out) · {s['llm_calls']} LLM calls"
        f" · cost {cost} list-price equivalent",
        f"Models: {', '.join(f'{k}={v}' for k, v in record['models'].items())}",
    ]
    if record.get("errors"):
        lines.append(f"Warnings: {record['errors']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bioagent")
    sub = parser.add_subparsers(dest="command", required=True)
    ask_p = sub.add_parser("ask", help="answer a question with cited evidence")
    ask_p.add_argument("question", nargs="?")
    ask_p.add_argument("--provider", choices=["gemini", "ollama"])
    ask_p.add_argument("--resume", metavar="RUN_ID", help="resume a failed or interrupted run")
    ask_p.add_argument("--json", action="store_true", help="print the run record as JSON")
    ask_p.add_argument("--no-verify", action="store_true", help="skip claim-level NLI verification")
    from bioagent.eval.cli import add_parser as add_eval_parser
    from bioagent.serve.cli import add_parsers as add_serve_parsers

    add_eval_parser(sub)
    add_serve_parsers(sub)
    args = parser.parse_args(argv)
    if args.command in ("serve", "worker"):
        logging.basicConfig(level=logging.INFO)
        quiet_http_logging()
        from bioagent.serve.cli import run_api, run_worker

        return run_api(args) if args.command == "serve" else run_worker(args)
    if args.command == "eval":
        logging.basicConfig(level=logging.WARNING)
        quiet_http_logging()
        from bioagent.eval.cli import main as eval_main

        return eval_main(args)

    logging.basicConfig(level=logging.WARNING)
    quiet_http_logging()
    if not args.question and not args.resume:
        parser.error("ask needs a question or --resume RUN_ID")

    from bioagent.agents.runner import VerifierUnavailableError, ask

    settings = get_settings()
    if args.no_verify:
        settings = settings.model_copy(update={"verify_claims": False})
    try:
        record = asyncio.run(
            ask(settings, args.question, provider=args.provider, resume=args.resume)
        )
    except VerifierUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "The run was checkpointed; retry with: bioagent ask --resume <run_id> "
            f"(see {settings.runs_dir}/*.meta.json)",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_report(record))
        print(f"\nSaved {settings.runs_dir}/{record['run_id']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
