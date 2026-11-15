"""Labelling tool: sheet ordering, atomic saves, sample selection, the local-only API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bioagent.labeling.app import create_app
from bioagent.labeling.sample import (
    choose_focus_queries,
    kappa_half_width,
    plan,
    proportion_half_width,
    queries_for_half_width,
)
from bioagent.labeling.sheets import SPECS, Sheet, read_jsonl


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _pool(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for q, qtype in (("q1", "topical"), ("q2", "topical"), ("q3", "exact_term")):
        for pmid in ("1", "2", "3"):
            rows.append(
                {
                    "query_id": q,
                    "query": f"query {q}",
                    "type": qtype,
                    "pmid": pmid,
                    "title": f"title {pmid}",
                    "abstract": "text",
                    "grade": None,
                }
            )
    return _write(tmp_path / "pool.jsonl", rows)


def _verifier(tmp_path: Path) -> Path:
    rows = [
        {"id": f"v{i}", "claim": "c", "passage": "p", "pmid": "9", "kind": "positive"}
        | {"options": ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"], "human_label": ""}
        for i in range(3)
    ]
    return _write(tmp_path / "ver.jsonl", rows)


def test_focus_groups_come_first_and_groups_stay_together(tmp_path: Path) -> None:
    sheet = Sheet(SPECS["retrieval"], _pool(tmp_path), focus=frozenset({"q3"}))
    order = sheet.order(sheet.rows())
    groups = [k.split(":")[0] for k in order]
    assert groups == ["q3"] * 3 + ["q1"] * 3 + ["q2"] * 3
    assert sheet.progress()["focus_total"] == 3


def test_set_label_saves_atomically_and_keeps_other_fields(tmp_path: Path) -> None:
    path = _pool(tmp_path)
    before = read_jsonl(path)
    sheet = Sheet(SPECS["retrieval"], path)
    assert sheet.set_label("q2:3", 2)["done"] == 1
    after = read_jsonl(path)
    assert after[5]["grade"] == 2
    assert [{k: v for k, v in r.items() if k != "grade"} for r in after] == [
        {k: v for k, v in r.items() if k != "grade"} for r in before
    ]
    assert not list(tmp_path.glob(".pool.jsonl.*"))  # no temp files left behind
    sheet.set_label("q2:3", None)
    assert read_jsonl(path)[5]["grade"] is None


def test_set_label_rejects_unknown_values_and_keys(tmp_path: Path) -> None:
    sheet = Sheet(SPECS["verifier"], _verifier(tmp_path))
    with pytest.raises(ValueError):
        sheet.set_label("v0", "MAYBE")
    with pytest.raises(KeyError):
        sheet.set_label("nope", "SUPPORTED")
    sheet.set_label("v0", "SUPPORTED")
    assert read_jsonl(sheet.path)[0]["human_label"] == "SUPPORTED"


def test_next_key_resumes_skips_and_wraps(tmp_path: Path) -> None:
    sheet = Sheet(SPECS["verifier"], _verifier(tmp_path))
    order = sheet.order(sheet.rows())
    first = sheet.next_key()
    assert first == order[0]
    sheet.set_label(order[0], "SUPPORTED")
    assert sheet.next_key(order[0]) == order[1]
    # a skipped item is offered only after everything else
    assert sheet.next_key(order[0], skipped=[order[1]]) == order[2]
    sheet.set_label(order[2], "CONTRADICTED")
    assert sheet.next_key(order[2], skipped=[order[1]]) == order[1]
    sheet.set_label(order[1], "NOT_ENOUGH_INFO")
    assert sheet.next_key() is None
    # a fresh Sheet over the same file sees the saved labels (resumable)
    assert Sheet(SPECS["verifier"], sheet.path).progress()["done"] == 3


def test_label_picks_up_a_regenerated_file(tmp_path: Path) -> None:
    path = _pool(tmp_path)
    sheet = Sheet(SPECS["retrieval"], path)
    rows = read_jsonl(path)
    rows[0]["grade"] = 1  # edited outside the server
    _write(path, rows)
    sheet.set_label("q1:2", 0)
    assert [r["grade"] for r in read_jsonl(path)[:2]] == [1, 0]


def test_judge_sheet_hides_the_judge_label(tmp_path: Path) -> None:
    row = {
        "id": "j1",
        "scale": ["incorrect", "partially", "correct"],
        "question": "q?",
        "gold": "g",
        "key_facts": ["f1"],
        "answer": "a",
        "judge_label": "correct",
        "judge_rationale": "because",
        "human_label": "",
    }
    sheet = Sheet(SPECS["judge"], _write(tmp_path / "j.jsonl", [row]))
    shown = json.dumps(sheet.item("j1"))
    assert "because" not in shown
    assert [o["key"] for o in sheet.item("j1")["options"]] == ["1", "2", "3"]


def test_choose_focus_queries_is_stratified_and_seeded() -> None:
    rows = [{"query_id": f"t{i}", "type": "topical"} for i in range(24)]
    rows += [{"query_id": f"e{i}", "type": "exact_term"} for i in range(20)]
    rows += [{"query_id": f"m{i}", "type": "multi_hop"} for i in range(6)]
    chosen = choose_focus_queries(rows, 8, seed=0)
    assert len(chosen) == 8
    assert chosen == choose_focus_queries(rows, 8, seed=0)
    kinds = [q[0] for q in chosen]
    assert kinds.count("t") == 4 and kinds.count("e") == 3 and kinds.count("m") == 1
    assert len(choose_focus_queries(rows, 2, seed=1)) == 2


def test_sample_size_formulas() -> None:
    assert queries_for_half_width(0.25, 0.10) == 25
    assert proportion_half_width(0.85, 150) == pytest.approx(0.0571, abs=1e-3)
    assert kappa_half_width(0.85, 0.5, 40) == pytest.approx(0.221, abs=1e-3)
    p = plan([{"query_id": "q1"}] * 40, [{}] * 150, [{}] * 13)
    assert p["pool"]["mean_items_per_query"] == 40
    assert p["verifier_accuracy"][-1]["n"] == 150


def _client(tmp_path: Path, host: str = "127.0.0.1") -> TestClient:
    sheets = {
        "retrieval": Sheet(SPECS["retrieval"], _pool(tmp_path)),
        "verifier": Sheet(SPECS["verifier"], _verifier(tmp_path)),
    }
    return TestClient(create_app(sheets), client=(host, 50000))


def test_api_round_trip(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert "labelling" in client.get("/").text
    names = [s["name"] for s in client.get("/api/sheets").json()]
    assert names == ["retrieval", "verifier"]
    item = client.get("/api/verifier/next").json()
    r = client.post("/api/verifier/item", params={"key": item["key"]}, json={"value": "SUPPORTED"})
    assert r.status_code == 200 and r.json()["progress"]["done"] == 1
    bad = client.post("/api/verifier/item", params={"key": item["key"]}, json={"value": "X"})
    assert bad.status_code == 422
    assert client.get("/api/nope/next").status_code == 404
    nxt = client.get("/api/verifier/next", params={"after": item["key"]}).json()
    assert nxt["key"] != item["key"]
    prev = client.get("/api/verifier/previous", params={"before": nxt["key"]}).json()
    assert prev["key"] == item["key"] and prev["label"] == "SUPPORTED"


def test_api_refuses_non_loopback_clients(tmp_path: Path) -> None:
    client = _client(tmp_path, host="192.168.1.20")
    assert client.get("/api/sheets").status_code == 403


def test_only_restricts_the_sheet_but_keeps_other_rows(tmp_path: Path) -> None:
    path = _verifier(tmp_path)
    sheet = Sheet(SPECS["verifier"], path, only=frozenset({"v1"}))
    assert [r["id"] for r in sheet.rows()] == ["v1"]
    assert sheet.progress()["total"] == 1
    with pytest.raises(KeyError):
        sheet.set_label("v0", "SUPPORTED")
    sheet.set_label("v1", "CONTRADICTED")
    rows = read_jsonl(path)
    assert [r["id"] for r in rows] == ["v0", "v1", "v2"]
    assert [r["human_label"] for r in rows] == ["", "CONTRADICTED", ""]
    assert sheet.next_key() is None


def test_proportional_sample() -> None:
    from bioagent.labeling.spotcheck import proportional_sample

    rows = [{"g": "a", "k": f"a{i}"} for i in range(30)] + [
        {"g": "b", "k": f"b{i}"} for i in range(10)
    ]
    keys = proportional_sample(rows, "g", 8, "k", seed=1)
    assert len(keys) == 8 and sum(k.startswith("a") for k in keys) == 6
    assert keys == proportional_sample(rows, "g", 8, "k", seed=1)
    assert len(proportional_sample(rows, "g", 100, "k")) == 40


def test_kappa_summary() -> None:
    from bioagent.eval.stats import bootstrap_kappa
    from bioagent.labeling.spotcheck import _kappa

    pairs = [("2", "2"), ("1", "1"), ("0", "0"), ("0", "1"), ("2", "2"), ("0", "0")]
    res = _kappa(pairs, ["0", "1", "2"], bootstrap_kappa, True)
    assert res["n"] == 6 and res["exact_agreement"] == pytest.approx(0.833, abs=1e-3)
    assert res["weighted_kappa_quadratic"]["mean"] > res["kappa"]["mean"] > 0
    assert _kappa([], ["0"], bootstrap_kappa, False)["n"] == 0
