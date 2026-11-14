"""``bioagent eval`` subcommands: run, report, compare, calibration, kappa."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from bioagent.config import get_settings
from bioagent.eval.dataset import INJECTION_FILE

ROOT = Path.cwd()


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ev = sub.add_parser("eval", help="evaluation suite (see eval/ and bioagent.eval)")
    esub = ev.add_subparsers(dest="eval_command", required=True)

    run = esub.add_parser("run", help="run (or resume) a suite; results/eval/<suite>/<run_id>")
    run.add_argument("--suite", choices=["smoke", "full", "injection"], default="smoke")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
    run.add_argument("--resume", metavar="RUN_ID")
    run.add_argument("--run-id", help="name for a new run (default: timestamp)")
    run.add_argument("--nli", action="store_true", help="add NLI citation-support metrics")
    run.add_argument("--graph", action="store_true", help="enable the Neo4j graph tools")
    run.add_argument(
        "--defenses",
        choices=["on", "off"],
        default="on",
        help="prompt-injection defences for the agent (off: baseline for attack success)",
    )
    run.add_argument("--limit", type=int, help="run at most N (question, repeat) pairs now")
    run.add_argument("--results-dir", type=Path, default=Path("results/eval"))

    rep = esub.add_parser("report", help="aggregate a run directory (JSON + markdown)")
    rep.add_argument("run_dir", type=Path)
    rep.add_argument("--rescore", action="store_true", help="re-score stored records first")
    rep.add_argument("--nli", action="store_true")

    cmp_ = esub.add_parser("compare", help="paired comparison of two reports or run dirs")
    cmp_.add_argument("baseline", type=Path)
    cmp_.add_argument("candidate", type=Path)
    cmp_.add_argument("--margin", type=float, default=0.0)
    cmp_.add_argument("--fail-on-regression", action="store_true")
    cmp_.add_argument("--summary", type=Path, help="append the markdown table to this file")

    cal = esub.add_parser("calibration", help="write the judge calibration sheet")
    cal.add_argument("run_dirs", type=Path, nargs="+")
    cal.add_argument("--n", type=int, default=40)
    cal.add_argument("--out", type=Path, default=Path("eval/labels/judge_calibration.jsonl"))

    kap = esub.add_parser("kappa", help="Cohen's kappa: judge vs human labels")
    kap.add_argument(
        "sheet", type=Path, nargs="?", default=Path("eval/labels/judge_calibration.jsonl")
    )


def _load_report(path: Path) -> dict[str, object]:
    from bioagent.eval.report import write_report

    if path.is_dir():
        return write_report(path)
    data: dict[str, object] = json.loads(path.read_text())
    return data


def main(args: argparse.Namespace) -> int:
    from bioagent.eval import report as rep
    from bioagent.eval.runner import EvalConfig, EvalPaused, rescore, run_suite

    settings = get_settings()
    cmd = args.eval_command
    if cmd == "run":
        config = EvalConfig.for_provider(
            args.suite,
            args.provider,
            settings,
            repeats=args.repeats,
            nli=args.nli,
            graph=args.graph,
            defenses=args.defenses == "on",
        )
        if args.suite == "injection" and not (ROOT / INJECTION_FILE).exists():
            print(
                f"skipped: {INJECTION_FILE} not found. The prompt-injection suite needs "
                "maintainer-written items there (schema: bioagent.eval.dataset.EvalItem)."
            )
            return 0
        try:
            run_dir = asyncio.run(
                run_suite(
                    ROOT,
                    settings,
                    config,
                    run_id=args.resume or args.run_id,
                    results_dir=args.results_dir,
                    limit=args.limit,
                )
            )
        except EvalPaused as exc:
            print(f"paused: {exc}", file=sys.stderr)
            return 3
        report = rep.write_report(run_dir)
        print(rep.markdown(report))
        print(f"Saved {run_dir}/report.md and report.json")
        return 0
    if cmd == "report":
        if args.rescore:
            n = asyncio.run(rescore(ROOT, args.run_dir, settings, nli=args.nli))
            print(f"re-scored {n} results")
        print(rep.markdown(rep.write_report(args.run_dir)))
        return 0
    if cmd == "compare":
        result = rep.compare_reports(
            _load_report(args.baseline), _load_report(args.candidate), margin=args.margin
        )
        text = rep.compare_markdown(result)
        print(text)
        if not result["same_agent_config"]:
            print("note: baseline and candidate used different agent configurations")
        target = args.summary or (
            Path(os.environ["GITHUB_STEP_SUMMARY"]) if "GITHUB_STEP_SUMMARY" in os.environ else None
        )
        if target is not None:
            with target.open("a") as fh:
                fh.write(text + "\n")
        return 1 if (args.fail_on_regression and result["regression"]) else 0
    if cmd == "calibration":
        from bioagent.eval.dataset import load_all

        items = {i.id: i for i in load_all(ROOT, lambda _m: None)}
        rows = rep.merge_sheet(args.out, rep.calibration_sheet(args.run_dirs, items, n=args.n))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["category"]] = counts.get(r["category"], 0) + 1
        print(f"wrote {len(rows)} rows to {args.out} {counts}; fill in human_label")
        return 0
    if cmd == "kappa":
        rows = [json.loads(line) for line in args.sheet.read_text().splitlines() if line.strip()]
        print(json.dumps(rep.kappa_report(rows), indent=1))
        return 0
    raise ValueError(cmd)
