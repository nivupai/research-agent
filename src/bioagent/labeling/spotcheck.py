"""The human spot-check: small samples that measure agreement, not the full sheets.

* retrieval: 50 items drawn from the Claude-graded sample (``eval/labels/retrieval_claude.jsonl``),
  proportionally across its queries, written without Claude's grade to
  ``eval/labels/spotcheck_retrieval.jsonl``. Agreement: human vs Claude (weighted kappa).
* verifier: 50 of the 150 pairs in ``eval/labels/verifier_test_human.jsonl``, proportionally
  across pair kinds. Agreement: human vs the Gemini teacher label.
* judge: all 40 rows of ``eval/labels/judge_calibration.jsonl``. Agreement: human vs the Gemini
  judge (``bioagent eval kappa``).

The selection is stored in ``eval/labels/spotcheck.json`` (seeded, reproducible).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bioagent.labeling.sheets import Row, is_labelled, read_jsonl, write_jsonl_atomic

LABELS_DIR = Path("eval/labels")
SELECTION = LABELS_DIR / "spotcheck.json"
RETRIEVAL_CLAUDE = LABELS_DIR / "retrieval_claude.jsonl"
RETRIEVAL_SPOTCHECK = LABELS_DIR / "spotcheck_retrieval.jsonl"
VERIFIER_SHEET = LABELS_DIR / "verifier_test_human.jsonl"
VERIFIER_CLAUDE = LABELS_DIR / "verifier_test_claude.jsonl"
JUDGE_SHEET = LABELS_DIR / "judge_calibration.jsonl"
JUDGE_CLAUDE = LABELS_DIR / "judge_calibration_claude.jsonl"
RETRIEVAL_RETEST = LABELS_DIR / "spotcheck_retrieval_claude_v2.jsonl"
POOL = Path("data/labels/pool.jsonl")
SECONDS_PER_ITEM = {"retrieval": 30, "verifier": 25, "judge": 60}


def proportional_sample(
    rows: Sequence[Row], group: str, n: int, key: str, seed: int = 0
) -> list[str]:
    """``n`` keys spread over ``group`` values in proportion to their size (largest remainder)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[str(row[group])].append(str(row[key]))
    total = sum(len(v) for v in groups.values())
    n = min(n, total)
    exact = {g: n * len(v) / total for g, v in groups.items()}
    alloc = {g: int(x) for g, x in exact.items()}
    for g in sorted(groups, key=lambda g: (alloc[g] - exact[g], g))[: n - sum(alloc.values())]:
        alloc[g] += 1
    rng = random.Random(seed)
    chosen: list[str] = []
    for g in sorted(groups):
        chosen += rng.sample(sorted(groups[g]), alloc[g])
    return chosen


def build(
    n_retrieval: int = 50, n_verifier: int = 50, seed: int = 0, pool_path: Path = POOL
) -> dict[str, Any]:
    claude = read_jsonl(RETRIEVAL_CLAUDE)
    for r in claude:
        r["key"] = f"{r['query_id']}:{r['pmid']}"
    retrieval_keys = proportional_sample(claude, "query_id", n_retrieval, "key", seed)
    pool = {f"{r['query_id']}:{r['pmid']}": r for r in read_jsonl(pool_path)}
    existing = (
        {f"{r['query_id']}:{r['pmid']}": r for r in read_jsonl(RETRIEVAL_SPOTCHECK)}
        if RETRIEVAL_SPOTCHECK.exists()
        else {}
    )
    sheet = []
    for k in retrieval_keys:
        src = pool[k]
        old = existing.get(k, {})
        sheet.append(
            {
                "query_id": src["query_id"],
                "query": src["query"],
                "type": src["type"],
                "pmid": src["pmid"],
                "title": src["title"],
                "abstract": src["abstract"],
                "year": src.get("year"),
                "grade": old.get("grade"),  # the human grade; Claude's is not in this file
                "labeler": "human",
            }
        )
    write_jsonl_atomic(RETRIEVAL_SPOTCHECK, sheet)
    verifier = read_jsonl(VERIFIER_SHEET)
    verifier_ids = proportional_sample(verifier, "kind", n_verifier, "id", seed)
    judge_ids = [str(r["id"]) for r in read_jsonl(JUDGE_SHEET)]
    selection = {
        "seed": seed,
        "retrieval": retrieval_keys,
        "verifier": verifier_ids,
        "judge": judge_ids,
    }
    SELECTION.write_text(json.dumps(selection, indent=1) + "\n")
    return selection


def estimate_minutes(selection: dict[str, Any]) -> dict[str, float]:
    out = {k: round(len(selection[k]) * s / 60, 1) for k, s in SECONDS_PER_ITEM.items()}
    out["total"] = round(sum(out.values()), 1)
    return out


def agreement() -> dict[str, Any]:
    """Human vs Claude (retrieval), human vs teacher (verifier); judge via ``eval kappa``."""
    from bioagent.eval.stats import bootstrap_kappa

    out: dict[str, Any] = {}
    claude = {f"{r['query_id']}:{r['pmid']}": str(r["grade"]) for r in read_jsonl(RETRIEVAL_CLAUDE)}
    human = [
        (f"{r['query_id']}:{r['pmid']}", str(r["grade"]))
        for r in read_jsonl(RETRIEVAL_SPOTCHECK)
        if is_labelled(r.get("grade"))
    ]
    pairs = [(h, claude[k]) for k, h in human if k in claude]
    out["retrieval_human_vs_claude"] = _kappa(pairs, ["0", "1", "2"], bootstrap_kappa, True)

    sel = json.loads(SELECTION.read_text()) if SELECTION.exists() else {"verifier": []}
    chosen = set(sel["verifier"])
    teacher = _teacher_labels()
    rows = [
        r
        for r in read_jsonl(VERIFIER_SHEET)
        if is_labelled(r.get("human_label")) and (not chosen or r["id"] in chosen)
    ]
    cats = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    vpairs = [(r["human_label"], teacher[r["id"]]) for r in rows if r["id"] in teacher]
    out["verifier_human_vs_teacher"] = _kappa(vpairs, cats, bootstrap_kappa, False)
    if VERIFIER_CLAUDE.exists():
        cl = {r["id"]: r["label"] for r in read_jsonl(VERIFIER_CLAUDE)}
        cpairs = [(cl[i], t) for i, t in teacher.items() if i in cl]
        out["verifier_claude_vs_teacher (inter-model, not human)"] = _kappa(
            cpairs, cats, bootstrap_kappa, False
        )
        hc = [(r["human_label"], cl[r["id"]]) for r in rows if r["id"] in cl]
        out["verifier_human_vs_claude"] = _kappa(hc, cats, bootstrap_kappa, False)
    out.update(inter_model_agreement())
    return out


def inter_model_agreement() -> dict[str, Any]:
    """Claude as a second rater: vs the Gemini judge, and retrieval test-retest (not human)."""
    from bioagent.eval.stats import bootstrap_kappa

    out: dict[str, Any] = {}
    if JUDGE_CLAUDE.exists():
        claude = {r["id"]: r["label"] for r in read_jsonl(JUDGE_CLAUDE)}
        rows = [r for r in read_jsonl(JUDGE_SHEET) if r["id"] in claude]
        for kind, cats, ordinal in (
            ("answer", ["incorrect", "partially", "correct"], True),
            ("yes/no (abstain, premise)", ["no", "yes"], False),
        ):
            group = [r for r in rows if (r["judge_kind"] == "answer") == (kind == "answer")]
            pairs = [(claude[r["id"]], str(r["judge_label"])) for r in group]
            out[f"judge_claude_vs_gemini_judge ({kind}; inter-model, not human)"] = _kappa(
                pairs, cats, bootstrap_kappa, ordinal
            )
    if RETRIEVAL_RETEST.exists():
        first = {(r["query_id"], r["pmid"]): str(r["grade"]) for r in read_jsonl(RETRIEVAL_CLAUDE)}
        pairs = [
            (first[(r["query_id"], r["pmid"])], str(r["grade"]))
            for r in read_jsonl(RETRIEVAL_RETEST)
            if (r["query_id"], r["pmid"]) in first
        ]
        out["retrieval_claude_v1_vs_blind_v2 (self-consistency, not human)"] = _kappa(
            pairs, ["0", "1", "2"], bootstrap_kappa, True
        )
    return out


def _teacher_labels() -> dict[str, str]:
    path = Path("data/finetune/test.jsonl")
    if not path.exists():
        return {}
    return {
        str(r["id"]): str(r["labels"]["teacher"])
        for r in read_jsonl(path)
        if r.get("labels", {}).get("teacher")
    }


def _kappa(pairs: Sequence[tuple[str, str]], cats: list[str], fn: Any, ordinal: bool) -> Any:
    if not pairs:
        return {"n": 0, "message": "no labels yet"}
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    res: dict[str, Any] = {
        "n": len(pairs),
        "exact_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 3),
        "kappa": fn(a, b, cats, weights="none").as_dict(),
    }
    if ordinal:
        res["weighted_kappa_quadratic"] = fn(a, b, cats, weights="quadratic").as_dict()
    return res
