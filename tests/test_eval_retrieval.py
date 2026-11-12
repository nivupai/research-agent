from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bioagent.eval.retrieval import (
    dedupe,
    evaluate,
    load_qrels,
    markdown_table,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

GRADES = {"a": 2, "b": 1, "c": 0, "d": 2}
RANKED = ["x", "a", "c", "b", "y"]


def test_recall_precision_mrr() -> None:
    # relevant = {a, b, d}; top-3 has a only; top-5 has a, b
    assert recall_at_k(RANKED, GRADES, 3) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, GRADES, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, GRADES, 5) == pytest.approx(2 / 5)
    assert reciprocal_rank(RANKED, GRADES) == pytest.approx(1 / 2)
    assert reciprocal_rank(["x"], GRADES) == 0.0
    assert recall_at_k(RANKED, {"c": 0}, 5) == 0.0


def test_ndcg_hand_computed() -> None:
    # DCG@5 = (2^2-1)/log2(3) + 0 + (2^1-1)/log2(5) = 3/1.58496 + 1/2.32193
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    # ideal order: 2, 2, 1, 0 -> 3/1 + 3/log2(3) + 1/log2(4)
    idcg = 3 / 1 + 3 / math.log2(3) + 1 / 2
    assert ndcg_at_k(RANKED, GRADES, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "d", "b"], GRADES, 3) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, {}, 5) == 0.0


def test_dedupe_and_percentile() -> None:
    assert dedupe(["1", "2", "1", "3", "2"]) == ["1", "2", "3"]
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10, 20], 95) == pytest.approx(19.5)
    assert percentile([], 50) == 0.0


def test_load_qrels_skips_ungraded(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    rows = [
        {"query_id": "q1", "query": "metformin", "pmid": "1", "grade": 2},
        {"query_id": "q1", "query": "metformin", "pmid": "2", "grade": None},
        {"query_id": "q2", "query": "tmz", "pmid": "3", "grade": 0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    qrels = load_qrels(path)
    assert qrels.queries == {"q1": "metformin", "q2": "tmz"}
    assert qrels.grades == {"q1": {"1": 2}, "q2": {"3": 0}}
    path.write_text(json.dumps({"query_id": "q", "query": "x", "pmid": "1", "grade": 5}))
    with pytest.raises(ValueError, match="grade"):
        load_qrels(path)


def test_evaluate_and_table(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "query": "Q", "pmid": "a", "grade": 2})
        + "\n"
        + json.dumps({"query_id": "q1", "query": "Q", "pmid": "b", "grade": 1})
        + "\n"
    )
    qrels = load_qrels(path)
    runs = {"good": ["a", "a", "b"], "bad": ["x", "y", "b"]}
    results = evaluate(lambda q, mode, depth: runs[mode], qrels, ["good", "bad"], k=10)
    assert results["good"]["metrics"]["recall@10"] == 1.0
    assert results["good"]["metrics"]["ndcg@10"] == 1.0  # duplicates removed before scoring
    assert results["bad"]["metrics"]["mrr"] == pytest.approx(1 / 3, abs=1e-4)
    table = markdown_table(results)
    assert table.splitlines()[0].startswith("| mode | recall@10 | ndcg@10 | mrr")
    assert "| good | 1.0 | 1.0 | 1.0 |" in table
