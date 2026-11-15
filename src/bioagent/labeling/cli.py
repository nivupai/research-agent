"""``bioagent label`` subcommands: serve (labelling page), sample, plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioagent.labeling.sample import (
    DEFAULT_SAMPLE_PATH,
    choose_focus_queries,
    load_sample,
    plan,
    plan_markdown,
    save_sample,
)
from bioagent.labeling.sheets import SPECS, Sheet, read_jsonl


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    lab = sub.add_parser("label", help="hand-label the retrieval, verifier and judge sheets")
    lsub = lab.add_subparsers(dest="label_command", required=True)

    srv = lsub.add_parser("serve", help="local labelling page (127.0.0.1 only)")
    srv.add_argument("--sheets", default="retrieval,verifier,judge", help="comma-separated")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument(
        "--sample",
        type=Path,
        nargs="?",
        const=DEFAULT_SAMPLE_PATH,
        help="retrieval: label the query-complete sample first (see `label sample`)",
    )
    for name, spec in SPECS.items():
        srv.add_argument(f"--{name}-file", type=Path, default=spec.default_path)

    smp = lsub.add_parser("sample", help="choose the retrieval minimum useful sample")
    smp.add_argument("--queries", type=int, default=8, help="queries to grade completely")
    smp.add_argument("--seed", type=int, default=0)
    smp.add_argument("--pool", type=Path, default=SPECS["retrieval"].default_path)
    smp.add_argument("--out", type=Path, default=DEFAULT_SAMPLE_PATH)

    pln = lsub.add_parser("plan", help="how many labels each pending metric needs")
    pln.add_argument("--json", action="store_true")


def main(args: argparse.Namespace) -> int:
    if args.label_command == "sample":
        rows = read_jsonl(args.pool)
        qids = choose_focus_queries(rows, args.queries, seed=args.seed)
        save_sample(args.out, qids, seed=args.seed, rows=rows)
        items = sum(1 for r in rows if str(r["query_id"]) in set(qids))
        print(f"wrote {args.out}: {len(qids)} queries, {items} items to grade: {', '.join(qids)}")
        return 0
    if args.label_command == "plan":
        p = plan(*(read_jsonl(SPECS[n].default_path) for n in ("retrieval", "verifier", "judge")))
        print(json.dumps(p, indent=2) if args.json else plan_markdown(p))
        return 0

    import uvicorn

    from bioagent.labeling.app import create_app

    names = [n.strip() for n in args.sheets.split(",") if n.strip()]
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        raise SystemExit(f"unknown sheet(s) {unknown}; choose from {list(SPECS)}")
    sheets: dict[str, Sheet] = {}
    for name in names:
        path: Path = getattr(args, f"{name}_file")
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        focus = load_sample(args.sample) if name == "retrieval" and args.sample else frozenset()
        sheets[name] = Sheet(SPECS[name], path, focus=focus)
    if not sheets:
        raise SystemExit("no label sheets found")
    print(f"labelling page: http://127.0.0.1:{args.port}/")
    uvicorn.run(create_app(sheets), host="127.0.0.1", port=args.port, log_level="warning")
    return 0
