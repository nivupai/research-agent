"""Aggregate a run directory into a report (JSON + markdown), compare runs, and judge
calibration (sheet + Cohen's kappa)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bioagent.eval.stats import (
    bootstrap_kappa,
    cluster_bootstrap,
    compare,
    flip_rate,
)

REPORT_VERSION = "eval-report-v1"
# metric -> higher is better
METRICS: dict[str, bool] = {
    "accuracy": True,
    "citation_support": True,
    "groundedness": True,
    "gold_pmid_recall": True,
    "tool_calls": False,
    "latency_s": False,
    "tokens": False,
    "cost_usd": False,
}
ADVERSARIAL: dict[str, bool] = {
    "injection_success": False,
    "poisoned_claims_flagged": True,
    "abstained": True,
    "premise_corrected": True,
}


def per_question(results: Sequence[Mapping[str, Any]], metric: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.get("status") != "ok":
            if metric == "accuracy":  # a crashed run counts as a wrong answer
                out[r["item_id"]].append(0.0)
            continue
        value = r["scores"].get(metric)
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, int | float):
            out[r["item_id"]].append(float(value))
    return dict(out)


def _estimate(values: Mapping[str, Sequence[float]], b: int) -> dict[str, Any]:
    est = cluster_bootstrap(values, b=b)
    return {**est.as_dict(), "flip_rate": flip_rate(values)}


def build_report(run_dir: Path, *, b: int = 2000) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    results = [json.loads(p.read_text()) for p in sorted((run_dir / "items").glob("*.json"))]
    categories = sorted({r["category"] for r in results})
    overall = {m: _estimate(per_question(results, m), b) for m in [*METRICS, *ADVERSARIAL]}
    by_category = {
        c: {
            m: _estimate(per_question([r for r in results if r["category"] == c], m), b)
            for m in ("accuracy", "citation_support", "gold_pmid_recall", *ADVERSARIAL)
        }
        for c in categories
    }
    ok = [r for r in results if r["status"] == "ok"]
    totals = {
        "items_in_suite": len(manifest["items"]),
        "repeats": manifest["config"]["repeats"],
        "runs_expected": len(manifest["items"]) * manifest["config"]["repeats"],
        "runs_done": len(ok),
        "runs_error": sum(r["status"] == "error" for r in results),
        "agent_cost_usd": round(sum(r["scores"].get("cost_usd") or 0 for r in ok), 4),
        "judge_cost_usd": round(
            sum((c.get("cost_usd") or 0) for r in ok for c in r.get("judge_stats", [])), 4
        ),
        "agent_seconds": round(sum(r["scores"].get("latency_s") or 0 for r in ok), 1),
        "tokens": sum(r["scores"].get("tokens") or 0 for r in ok),
        "wall_seconds": manifest.get("wall_seconds"),
    }
    return {
        "version": REPORT_VERSION,
        "run_id": manifest["run_id"],
        "suite": manifest["suite"],
        "status": manifest["status"],
        "config": manifest["config"],
        "agent_hash": manifest["agent_hash"],
        "totals": totals,
        "overall": overall,
        "by_category": by_category,
        "per_question": {m: per_question(results, m) for m in [*METRICS, *ADVERSARIAL]},
    }


def _fmt(est: Mapping[str, Any], pct: bool = False) -> str:
    if est.get("mean") is None:
        return "n/a"
    scale, unit = (100.0, "%") if pct else (1.0, "")

    def f(v: float) -> str:
        return f"{v * scale:.1f}{unit}" if pct else f"{v:.3g}"

    ci = "" if est.get("lo") is None else f" [{f(est['lo'])}, {f(est['hi'])}]"
    return f"{f(est['mean'])}{ci}"


PCT = {"accuracy", "citation_support", "groundedness", "gold_pmid_recall", *ADVERSARIAL}


def markdown(report: Mapping[str, Any]) -> str:
    t = report["totals"]
    c = report["config"]
    lines = [
        f"# Eval report: {report['suite']} / {report['run_id']} ({report['status']})",
        "",
        f"Models: planner {c['planner']}, specialist {c['specialist']}, synth {c['synth']}, "
        f"judge {c['judge']} · retrieval {c['retrieval']} · graph {'on' if c['graph'] else 'off'}"
        f" · NLI metrics {'on' if c.get('nli') else 'off'} · agent config {report['agent_hash']}",
        f"Runs: {t['runs_done']}/{t['runs_expected']} ok ({t['runs_error']} errors) · "
        f"{t['items_in_suite']} questions x {t['repeats']} repeats · agent time "
        f"{t['agent_seconds']} s · tokens {t['tokens']:,} · cost ${t['agent_cost_usd']} agent + "
        f"${t['judge_cost_usd']} judge (list-price equivalent; free tier)",
        "",
        "Means with 95% CIs (two-stage bootstrap over questions and repeats); flip rate = share "
        "of questions whose score differs between repeats.",
        "",
        "| metric | mean [95% CI] | n questions | flip rate |",
        "|---|---|---|---|",
    ]
    for m, est in report["overall"].items():
        if est["n"] == 0:
            continue
        flip = "" if est["flip_rate"] is None else f"{est['flip_rate']:.0%}"
        lines.append(f"| {m} | {_fmt(est, m in PCT)} | {est['n']} | {flip} |")
    lines += [
        "",
        "## By category",
        "",
        "| category | accuracy | citation support | gold-PMID recall | category metric |",
        "|---|---|---|---|---|",
    ]
    special = {
        "injection": "injection_success",
        "unanswerable": "abstained",
        "false_premise": "premise_corrected",
    }
    for cat, ms in report["by_category"].items():
        extra = special.get(cat)
        extra_txt = f"{extra}: {_fmt(ms[extra], True)}" if extra else ""
        lines.append(
            f"| {cat} (n={ms['accuracy']['n']}) | {_fmt(ms['accuracy'], True)} | "
            f"{_fmt(ms['citation_support'], True)} | {_fmt(ms['gold_pmid_recall'], True)} | "
            f"{extra_txt} |"
        )
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, *, b: int = 2000) -> dict[str, Any]:
    report = build_report(run_dir, b=b)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (run_dir / "report.md").write_text(markdown(report))
    return report


# --- comparison ---------------------------------------------------------------------------------
def compare_reports(
    base: Mapping[str, Any], cand: Mapping[str, Any], *, margin: float = 0.0
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "baseline": base["run_id"],
        "candidate": cand["run_id"],
        "same_agent_config": base.get("agent_hash") == cand.get("agent_hash"),
        "margin": margin,
        "metrics": {},
    }
    for metric, higher in {**METRICS, **ADVERSARIAL}.items():
        b = base["per_question"].get(metric, {})
        c = cand["per_question"].get(metric, {})
        if not b or not c:
            continue
        m = margin if metric == "accuracy" else 0.0
        out["metrics"][metric] = compare(b, c, margin=m, higher_is_better=higher).as_dict()
    acc = out["metrics"].get("accuracy")
    out["regression"] = bool(acc and acc["regression"])
    return out


def compare_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"### Eval comparison: {result['candidate']} vs baseline {result['baseline']}",
        "",
        f"Verdict on accuracy: **{'REGRESSION' if result['regression'] else 'no regression'}**"
        f" (regression = 95% CI of the paired difference entirely below -{result['margin']})",
        "",
        "| metric | baseline | candidate | delta [95% CI] | p (permutation) | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for metric, r in result["metrics"].items():
        if r["delta"] is None:
            continue
        ci = "" if r["lo"] is None else f" [{r['lo']:+.3f}, {r['hi']:+.3f}]"
        p = "" if r["p_value"] is None else f"{r['p_value']:.3f}"
        lines.append(
            f"| {metric} | {r['base_mean']:.3f} | {r['cand_mean']:.3f} | {r['delta']:+.3f}{ci} "
            f"| {p} | {r['verdict']} (n={r['n']}) |"
        )
    return "\n".join(lines) + "\n"


# --- judge calibration -----------------------------------------------------------------------
JUDGED = {"corpus": "answer", "unanswerable": "abstain", "false_premise": "premise"}
ANSWER_SCALE = ("incorrect", "partially", "correct")
YESNO_SCALE = ("no", "yes")


def calibration_sheet(
    run_dirs: Sequence[Path], items: Mapping[str, Any], n: int = 40, seed: int = 3
) -> list[dict[str, Any]]:
    """A stratified sample of judged answers with a blank ``human_label`` to fill in."""
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_dir in run_dirs:
        for path in sorted((run_dir / "items").glob("*__r0.json")):
            r = json.loads(path.read_text())
            judge = r.get("scores", {}).get("judge")
            if r["status"] != "ok" or not judge or r["category"] not in JUDGED or "error" in judge:
                continue
            item = items[r["item_id"]]
            kind = JUDGED[r["category"]]
            pool[r["category"]].append(
                {
                    "id": r["item_id"],
                    "run": run_dir.name,
                    "category": r["category"],
                    "judge_kind": kind,
                    "scale": list(ANSWER_SCALE if kind == "answer" else YESNO_SCALE),
                    "question": item.question,
                    "gold": item.gold,
                    "key_facts": item.key_facts,
                    "notes": item.notes,
                    "answer": r["record"].get("answer", ""),
                    "judge_label": judge.get("verdict")
                    if kind == "answer"
                    else ("yes" if judge.get("yes") else "no"),
                    "judge_rationale": judge.get("rationale"),
                    "human_label": "",
                }
            )
    rng = random.Random(seed)
    cats = sorted(pool)
    share = {
        c: max(1, round(n * len(pool[c]) / max(1, sum(map(len, pool.values()))))) for c in cats
    }
    sheet: list[dict[str, Any]] = []
    for c in cats:
        unique = list({row["id"]: row for row in pool[c]}.values())
        sheet += rng.sample(unique, min(share[c], len(unique)))
    return sorted(sheet, key=lambda r: r["id"])[:n]


def merge_sheet(path: Path, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep human labels already entered for the same id."""
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("human_label"):
                    existing[row["id"]] = row["human_label"]
    return [{**r, "human_label": existing.get(r["id"], r["human_label"])} for r in rows]


def kappa_report(rows: Sequence[Mapping[str, Any]], b: int = 2000) -> dict[str, Any]:
    labelled = [r for r in rows if str(r.get("human_label", "")).strip()]
    out: dict[str, Any] = {"rows": len(rows), "labelled": len(labelled), "by_kind": {}}
    if not labelled:
        out["message"] = "no human labels yet: fill in human_label in the calibration sheet"
        return out
    for kind, scale in (("answer", ANSWER_SCALE), ("yesno", YESNO_SCALE)):
        group = [r for r in labelled if (r["judge_kind"] == "answer") == (kind == "answer")]
        if not group:
            continue
        bad = [r["id"] for r in group if r["human_label"] not in scale]
        if bad:
            raise ValueError(f"human_label must be one of {scale}: {bad[:5]}")
        judge = [str(r["judge_label"]) for r in group]
        human = [str(r["human_label"]) for r in group]
        entry: dict[str, Any] = {
            "n": len(group),
            "agreement": sum(j == h for j, h in zip(judge, human, strict=True)) / len(group),
            "kappa": bootstrap_kappa(judge, human, scale, weights="none", b_samples=b).as_dict(),
        }
        if kind == "answer":
            entry["weighted_kappa_quadratic"] = bootstrap_kappa(
                judge, human, scale, weights="quadratic", b_samples=b
            ).as_dict()
        out["by_kind"][kind] = entry
    return out
