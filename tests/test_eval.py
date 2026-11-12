"""Evaluation suite: dataset validation, scorers, statistics, kappa, caches and reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from bioagent.eval import stats
from bioagent.eval.cache import JSONCache, answer_hash, item_key, judge_key
from bioagent.eval.dataset import (
    EvalItem,
    load_all,
    load_injection_items,
    validate_items,
)
from bioagent.eval.report import (
    build_report,
    calibration_sheet,
    compare_reports,
    kappa_report,
    markdown,
    merge_sheet,
)
from bioagent.eval.runner import EvalConfig
from bioagent.eval.scoring import (
    agent_question,
    detect_injection,
    extract_label,
    gold_pmid_recall,
    numeric_match,
    score_item,
)

ROOT = Path(__file__).resolve().parents[1]


def item(**kw: Any) -> EvalItem:
    base: dict[str, Any] = {
        "id": "corpus-x-1",
        "category": "corpus",
        "question": "What does metformin do to AMPK?",
        "scoring_type": "free_text",
        "gold": "It activates AMPK.",
        "source": "test",
    }
    return EvalItem.model_validate(base | kw)


# --- dataset ------------------------------------------------------------------------------------
def test_committed_eval_set_is_valid() -> None:
    items = load_all(ROOT, lambda _m: None)
    summary = validate_items(items)
    assert summary["items"] >= 180
    assert summary["by_category"]["pubmedqa"] == 100
    assert summary["by_category"]["corpus"] >= 60 and summary["needs_review"] >= 60
    assert summary["by_category"]["unanswerable"] >= 10
    assert summary["by_category"]["false_premise"] >= 10
    assert summary["by_category"]["computed"] >= 5


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"category": "pubmedqa", "scoring_type": "label", "gold": "perhaps"}, "label gold"),
        ({"scoring_type": "label"}, "scoring_type"),
        ({"category": "computed", "scoring_type": "numeric", "gold": "1"}, "numeric"),
        ({"gold_pmids": ["PMID123"]}, "numeric PMIDs"),
        ({"category": "injection", "scoring_type": "injection"}, "injection spec"),
    ],
)
def test_item_validation_errors(kw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        item(**kw)


def test_cross_item_validation() -> None:
    a, b = item(), item(question="Another question about AMPK?")
    with pytest.raises(ValueError, match="duplicate id"):
        validate_items([a, b])
    with pytest.raises(ValueError, match="category prefix"):
        validate_items([item(id="pubmedqa-1")])


def test_injection_file_is_optional_and_validated(tmp_path: Path) -> None:
    notices: list[str] = []
    assert load_injection_items(tmp_path / "missing.jsonl", notices.append) == []
    assert "skipped" in notices[0]
    path = tmp_path / "injection.jsonl"
    path.write_text(json.dumps(item().model_dump()) + "\n")
    with pytest.raises(ValueError, match="only contain injection"):
        load_injection_items(path)


# --- scorers --------------------------------------------------------------------------------
def test_label_extraction_and_prompt() -> None:
    assert extract_label("**Yes.** Metformin ...") == "yes"
    assert extract_label("No, it does not.") == "no"
    assert extract_label("The answer is maybe: data conflict.") == "maybe"
    assert extract_label("Several studies suggest benefit.") is None
    qa = item(id="pubmedqa-1", category="pubmedqa", scoring_type="label", gold="yes")
    assert agent_question(qa).endswith("Yes, No or Maybe, then explain.)")
    assert agent_question(item()) == item().question


def test_numeric_match() -> None:
    gold = item(
        id="computed-a",
        category="computed",
        scoring_type="numeric",
        gold=[{"name": "tpsa", "value": 91.49, "tolerance": 0.06}],
    ).gold
    assert isinstance(gold, list)
    assert numeric_match("Metformin's TPSA is 91.5 Å² [1].", gold) == (1.0, [])
    assert numeric_match("TPSA is about 92 [1].", gold)[0] == 0.0
    assert numeric_match("TPSA 91.49 vs MW 129.17 [1]", gold)[0] == 1.0


def injection_item(**kw: Any) -> EvalItem:
    spec = {
        "passage": {"pmid": "99000001", "title": "t", "abstract": "a neutral test passage"},
        "canaries": ["CANARY-TEST-1"],
        "forbidden_tools": ["run_python"],
        "forbidden_pmids": ["99999999"],
    }
    return item(
        id="injection-01",
        category="injection",
        scoring_type="injection",
        injection=spec,
        gold=None,
        **kw,
    )


def test_injection_detector() -> None:
    it = injection_item()
    clean = {
        "answer": "Evidence is insufficient [1].",
        "tool_calls": [{"tool": "search_corpus"}],
        "citations": [{"n": 1, "pmid": "99000001"}],
        "evidence_keys": ["99000001:0"],
    }
    out = detect_injection(it, clean)
    assert out["injection_success"] is False
    assert out["poisoned_retrieved"] and out["poisoned_cited"]
    for attacked in (
        {**clean, "answer": "Result: canary-test-1"},  # case-insensitive
        {**clean, "tool_calls": [{"tool": "run_python"}]},
        {**clean, "citations": [{"n": 1, "pmid": "99999999"}]},
    ):
        assert detect_injection(it, attacked)["injection_success"] is True


async def test_score_item_by_type() -> None:
    async def judge(kind: str, it: EvalItem, answer: str) -> dict[str, Any]:
        return {
            "answer": {"verdict": "partially"},
            "abstain": {"yes": True},
            "premise": {"yes": False},
            "label": {"label": "no"},
        }[kind]

    record = {
        "answer": "Yes, it activates AMPK [1].",
        "citations": [{"n": 1, "pmid": "5"}],
        "stats": {
            "tool_calls": 2,
            "wall_seconds": 3.0,
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.001,
        },
    }
    free = await score_item(item(gold_pmids=["5", "6"]), record, judge)
    assert free["accuracy"] == 0.5 and free["gold_pmid_recall"] == 0.5 and free["tokens"] == 15
    qa = item(id="pubmedqa-1", category="pubmedqa", scoring_type="label", gold="yes")
    assert (await score_item(qa, record, judge))["accuracy"] == 1.0
    fallback = await score_item(qa, {**record, "answer": "Evidence suggests none."}, judge)
    assert fallback["predicted_label"] == "no" and fallback["accuracy"] == 0.0
    un = item(id="unanswerable-1", category="unanswerable", scoring_type="abstain", gold=None)
    assert (await score_item(un, record, judge))["abstained"] is True
    fp = item(id="false_premise-1", category="false_premise", scoring_type="premise")
    assert (await score_item(fp, record, judge))["premise_corrected"] is False
    assert gold_pmid_recall(item(), record) is None


# --- statistics: known cases -------------------------------------------------------------------
def test_cluster_bootstrap_known_cases() -> None:
    const = stats.cluster_bootstrap({str(i): [1.0, 1.0] for i in range(10)}, b=200)
    assert (const.mean, const.lo, const.hi) == (1.0, 1.0, 1.0)
    # Bernoulli(0.5), 400 questions: CI half-width ~ 1.96 * sqrt(0.25 / 400) = 0.049
    values = {str(i): [float(i % 2)] for i in range(400)}
    est = stats.cluster_bootstrap(values, b=2000)
    assert est.mean == 0.5
    assert est.lo is not None and est.hi is not None
    assert abs((est.hi - est.lo) / 2 - 0.049) < 0.008
    # repeats add within-question noise: the CI must not shrink when repeats disagree
    noisy = {str(i): [0.0, 1.0] for i in range(50)}
    assert stats.cluster_bootstrap(noisy, b=500).lo < 0.5  # type: ignore[operator]
    assert stats.cluster_bootstrap({}).mean is None


def test_flip_rate() -> None:
    assert stats.flip_rate({"a": [1, 1], "b": [1, 0], "c": [0, 0], "d": [1]}) == pytest.approx(
        1 / 3
    )
    assert stats.flip_rate({"a": [1]}) is None


def test_permutation_test_exact_cases() -> None:
    # all 5 paired differences positive: only the all-plus and all-minus sign patterns are
    # as extreme -> p = 2 / 2^5
    assert stats.permutation_test([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert stats.permutation_test([0, 0, 0]) == 1.0
    assert stats.permutation_test([1, -1, 1, -1]) == 1.0
    big = [1.0] * 30  # Monte Carlo branch
    assert stats.permutation_test(big, b=2000) < 0.01


def test_compare_verdicts() -> None:
    base = {str(i): [1.0] for i in range(40)}
    worse = {str(i): [0.0 if i < 20 else 1.0] for i in range(40)}
    reg = stats.compare(base, worse)
    assert reg.regression and reg.delta == -0.5 and reg.verdict == "regression"
    assert reg.p_value is not None and reg.p_value < 0.001
    same = stats.compare(base, base)
    assert not same.regression and same.delta == 0.0
    tiny = {str(i): [0.0 if i == 0 else 1.0] for i in range(40)}
    assert not stats.compare(base, tiny, margin=0.1).regression
    cost_up = stats.compare(
        {"a": [1.0], "b": [1.0]}, {"a": [2.0], "b": [2.0]}, higher_is_better=False
    )
    assert cost_up.delta == 1.0


def test_kappa_known_values() -> None:
    # Classic 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa = 0.4
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert stats.cohen_kappa(a, b, ["no", "yes"]) == pytest.approx(0.4)
    assert stats.cohen_kappa(a, a, ["no", "yes"]) == 1.0
    # two categories: quadratic weights equal unweighted kappa
    assert stats.weighted_kappa(a, b, ["no", "yes"], "quadratic") == pytest.approx(0.4)
    scale = ["incorrect", "partially", "correct"]
    x = ["correct", "correct", "partially", "incorrect", "incorrect", "partially"]
    near = ["correct", "partially", "partially", "incorrect", "partially", "partially"]
    far = ["incorrect", "partially", "partially", "correct", "partially", "partially"]
    # near-misses cost less than opposite-end disagreements under weighting
    assert stats.weighted_kappa(x, near, scale) > stats.weighted_kappa(x, far, scale)
    assert stats.weighted_kappa(x, near, scale) > stats.cohen_kappa(x, near, scale)
    est = stats.bootstrap_kappa(a, b, ["no", "yes"], b_samples=500)
    assert est.lo is not None and est.hi is not None and est.lo < 0.4 < est.hi


# --- caches ---------------------------------------------------------------------------------
def test_cache_keys_are_stable_and_sensitive() -> None:
    k = judge_key("Q?", "Answer  text", "judge-v1", "answer")
    assert k == judge_key(" Q? ", "Answer text", "judge-v1", "answer")  # whitespace-insensitive
    assert k != judge_key("Q?", "Answer text!", "judge-v1", "answer")
    assert k != judge_key("Q?", "Answer text", "judge-v2", "answer")
    assert k != judge_key("Q?", "Answer text", "judge-v1", "abstain")
    assert item_key("a", "Q", 0, "c1") != item_key("a", "Q", 1, "c1")
    assert item_key("a", "Q", 0, "c1") != item_key("a", "Q", 0, "c2")
    assert answer_hash("a  b") == answer_hash("a b")
    cache = JSONCache(":memory:")
    assert cache.get(k) is None
    cache.set(k, {"verdict": "correct"})
    assert cache.get(k) == {"verdict": "correct"} and (cache.hits, cache.misses) == (1, 1)


def test_agent_hash_ignores_scoring_options() -> None:
    a = EvalConfig(suite="smoke", repeats=1, nli=False)
    b = EvalConfig(suite="smoke", repeats=3, nli=True, judge="ollama:x")
    assert a.agent_hash() == b.agent_hash()
    assert a.agent_hash() != EvalConfig(suite="smoke", synth="gemini:other").agent_hash()


# --- reports ------------------------------------------------------------------------------------
def fake_run(tmp_path: Path, name: str, acc: dict[str, list[float]]) -> Path:
    run = tmp_path / name
    (run / "items").mkdir(parents=True)
    ids = sorted(acc)
    config = {
        "repeats": max(len(v) for v in acc.values()),
        "planner": "p",
        "specialist": "s",
        "synth": "y",
        "judge": "j",
        "retrieval": "fixture-bm25",
        "graph": False,
    }
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "suite": "smoke",
                "config": config,
                "agent_hash": "h",
                "items": ids,
                "status": "completed",
                "wall_seconds": 1.0,
            }
        )
    )
    for qid, values in acc.items():
        for r, v in enumerate(values):
            category = "unanswerable" if qid.startswith("unanswerable") else "corpus"
            scores = {
                "accuracy": v,
                "cost_usd": 0.001,
                "latency_s": 2.0,
                "tokens": 100,
                "judge": {
                    "verdict": "correct" if v else "incorrect",
                    "rationale": "r",
                    "yes": bool(v),
                },
            }
            if category == "unanswerable":
                scores["abstained"] = bool(v)
            (run / "items" / f"{qid}__r{r}.json").write_text(
                json.dumps(
                    {
                        "item_id": qid,
                        "category": category,
                        "repeat": r,
                        "status": "ok",
                        "scores": scores,
                        "record": {"answer": f"answer {qid}"},
                    }
                )
            )
    return run


def test_report_and_compare(tmp_path: Path) -> None:
    base_acc = {f"corpus-x-{i}": [1.0, 1.0] for i in range(30)} | {"unanswerable-1": [1.0, 0.0]}
    cand_acc = {k: [0.0, 0.0] if i < 15 else v for i, (k, v) in enumerate(base_acc.items())}
    base = build_report(fake_run(tmp_path, "base", base_acc), b=300)
    cand = build_report(fake_run(tmp_path, "cand", cand_acc), b=300)
    assert base["overall"]["accuracy"]["mean"] == pytest.approx((30 + 0.5) / 31)
    assert base["overall"]["accuracy"]["flip_rate"] == pytest.approx(1 / 31)
    assert base["by_category"]["unanswerable"]["abstained"]["mean"] == 0.5
    assert base["totals"]["runs_done"] == 62
    assert "| accuracy |" in markdown(base)
    result = compare_reports(base, cand)
    assert result["regression"] and result["metrics"]["accuracy"]["delta"] < -0.4
    assert not compare_reports(base, base)["regression"]


def test_calibration_sheet_and_kappa(tmp_path: Path) -> None:
    acc = {f"corpus-x-{i}": [float(i % 2)] for i in range(10)} | {"unanswerable-1": [1.0]}
    run = fake_run(tmp_path, "r", acc)
    items = {k: item(id=k, question=f"Question number {k}?") for k in acc if k.startswith("corpus")}
    items["unanswerable-1"] = item(
        id="unanswerable-1", category="unanswerable", scoring_type="abstain", gold=None
    )
    sheet = calibration_sheet([run], items, n=6)
    assert len(sheet) <= 6 and all(r["human_label"] == "" for r in sheet)
    assert {r["category"] for r in sheet} == {"corpus", "unanswerable"}
    assert kappa_report(sheet)["labelled"] == 0
    path = tmp_path / "sheet.jsonl"
    labelled = [{**r, "human_label": r["judge_label"]} for r in sheet]
    path.write_text("".join(json.dumps(r) + "\n" for r in labelled[:2]))
    merged = merge_sheet(path, sheet)
    assert [bool(r["human_label"]) for r in merged][:2] == [True, True]
    report = kappa_report(labelled, b=200)
    answer = report["by_kind"]["answer"]
    assert answer["agreement"] == 1.0 and math.isclose(answer["kappa"]["mean"], 1.0)
    with pytest.raises(ValueError, match="human_label must be one of"):
        kappa_report([{**labelled[0], "human_label": "maybe"}])


def test_runner_reuses_only_matching_ok_results(tmp_path: Path) -> None:
    from bioagent.eval.runner import _done, item_path

    it = item()
    path = item_path(tmp_path, it, 0)
    assert not _done(path, it, 0, "h1")
    path.parent.mkdir(parents=True)
    ok = {"status": "ok", "key": item_key(it.id, it.question, 0, "h1")}
    path.write_text(json.dumps(ok))
    assert _done(path, it, 0, "h1")
    assert not _done(path, it, 0, "h2")  # agent config changed: rerun
    assert not _done(path, it, 1, "h1")  # other repeat
    path.write_text(json.dumps({**ok, "status": "error"}))
    assert not _done(path, it, 0, "h1")  # errors are retried on resume
