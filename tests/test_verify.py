"""Claim verification with a fake NLI scorer (no model download)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from bioagent.agents.state import Citation, Evidence
from bioagent.verify.computed import check_numbers, numbers_in
from bioagent.verify.nli import NLIScores, passage_windows
from bioagent.verify.text import (
    answer_sentences,
    renumber_citations,
    rewrite_answer,
)
from bioagent.verify.verifier import (
    AbsenceCheck,
    ClaimList,
    ClaimVerifier,
    ExtractedClaim,
    VerifyConfig,
)

Rule = Callable[[str, str], tuple[float, float]]


class FakeNLI:
    """Scores (entailment, contradiction) with a rule; neutral is the remainder."""

    name = "fake-nli"

    def __init__(self, rule: Rule) -> None:
        self.rule = rule
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        self.pairs.extend(pairs)
        out = []
        for premise, hypothesis in pairs:
            e, c = self.rule(premise, hypothesis)
            out.append(NLIScores(e, max(0.0, 1 - e - c), c))
        return out


def keyword_rule(premise: str, hypothesis: str) -> tuple[float, float]:
    p, h = premise.lower(), hypothesis.lower()
    if "contradicts" in h and "metformin" in p:
        return 0.02, 0.95
    words = [w for w in h.replace(".", "").split() if len(w) > 4]
    hit = sum(w in p for w in words) / max(1, len(words))
    return (0.9, 0.01) if hit >= 0.8 else (0.1, 0.05)


def ev(key: str, text: str, pmid: str | None = None, computed: bool = False) -> Evidence:
    return Evidence(
        key=key,
        pmid=pmid,
        passage_id=None,
        title=None,
        year=2020,
        journal=None,
        doi=None,
        text=text,
        source="rdkit" if computed else "corpus",
        task_id="t1",
        computed=computed,
        untrusted=True,
    )


def cite(n: int, key: str, pmid: str | None = None, computed: bool = False) -> Citation:
    return Citation(
        n=n, key=key, pmid=pmid, title=key, year=2020, url=None, computed=computed, source="x"
    )


EVIDENCE = {
    "p1": ev("p1", "Metformin title.\nMetformin inhibits glioblastoma growth in mice. Other.", "1"),
    "p2": ev(
        "p2", "Statins lower cholesterol. Unrelated sentence here. A third one. And more.", "2"
    ),
    "c1": ev("c1", '{"tpsa": 91.49, "molecular_weight": 129.17, "hbd": 3}', computed=True),
}
CITATIONS = [cite(1, "p1", "1"), cite(2, "p2", "2"), cite(3, "c1", computed=True)]
ANSWER = (
    "## Findings\n"
    "Metformin inhibits glioblastoma growth in mice [1].\n"
    "* Metformin cures every cancer known to medicine [2].\n"
    "Metformin's TPSA is 91.5 Å² and its molecular weight is 129.2 (computed estimates) [3]."
)


def verifier(policy: str = "remove", threshold: float = 0.5) -> ClaimVerifier:
    return ClaimVerifier(
        FakeNLI(keyword_rule),
        VerifyConfig(entail_threshold=threshold, policy=policy),  # type: ignore[arg-type]
    )


# --- text utilities ----------------------------------------------------------------------
def test_answer_sentences_skip_structure() -> None:
    sents = answer_sentences(ANSWER + "\n---\n**Bold heading**")
    assert [s.id for s in sents] == [1, 2, 3]
    assert sents[1].text.startswith("Metformin cures")


def test_rewrite_and_renumber() -> None:
    text = rewrite_answer(ANSWER, remove={2}, flag={3})
    assert "cures" not in text and "[unverified]" in text and text.startswith("## Findings")
    renumbered, kept = renumber_citations(text, [dict(c) for c in CITATIONS])
    assert [c["key"] for c in kept] == ["p1", "c1"] and "[2]" in renumbered
    assert "[3]" not in renumbered


def test_passage_windows() -> None:
    windows = passage_windows("T.\nA one. B two. C three. D four.", size=3)
    assert windows[0] == "T. A one. B two." and len(windows) == 3
    assert passage_windows("Short.") == ["Short."] and passage_windows("") == []


# --- computed checks ---------------------------------------------------------------------
def test_check_numbers() -> None:
    out = ['{"tpsa": 91.49, "clogp": -1.24, "hbd": 3}']
    assert check_numbers("TPSA is 91.5 and cLogP is \u22121.24 with 3 donors [2]", out).ok
    bad = check_numbers("TPSA is 120.4", out)
    assert not bad.ok and bad.unmatched == [120.4]
    assert numbers_in("U87 and PRKAA1 cells, type 2 diabetes, phase 3, value 4.5 [7]") == [4.5]
    assert check_numbers("passes the Lipinski rule of 5", out).checked == []


# --- labelling, policy and groundedness ------------------------------------------------------
async def test_labels_policy_and_groundedness() -> None:
    v = verifier()
    out = await v.verify(ANSWER, CITATIONS, EVIDENCE)  # no LLM: one claim per sentence
    report = out["report"]
    labels = [c["label"] for c in report["claims"]]
    assert labels == ["supported", "not_supported", "computed"]
    assert report["claim_split"] == "sentences"
    assert report["groundedness"] == pytest.approx(2 / 3, abs=1e-3)
    assert [c["claim"] for c in report["removed_claims"]] == [
        "Metformin cures every cancer known to medicine."
    ]
    assert "cures" not in out["answer"]
    assert [c["key"] for c in out["citations"]] == ["p1", "c1"]  # renumbered 1, 2
    assert "[2]" in out["answer"] and "[3]" not in out["answer"]
    supported = report["claims"][0]["evidence"][0]
    assert supported["ref"] == "PMID 1" and supported["entailment"] == 0.9


async def test_flag_policy_keeps_text() -> None:
    out = await verifier(policy="flag").verify(ANSWER, CITATIONS, EVIDENCE)
    assert "* Metformin cures every cancer known to medicine [2] [unverified]." in out["answer"]
    assert out["report"]["removed_claims"] == [] and out["report"]["flagged_sentences"] == [2]
    assert len(out["citations"]) == 3


async def test_threshold_controls_support() -> None:
    strict = await verifier(threshold=0.95).verify(ANSWER, CITATIONS, EVIDENCE)
    assert strict["report"]["claims"][0]["label"] == "not_supported"


async def test_computed_mismatch_and_contradiction() -> None:
    answer = (
        "Metformin's TPSA is 120.4 (computed estimate) [3].\n"
        "This contradicts the metformin glioblastoma finding [1]."
    )
    out = await verifier().verify(answer, CITATIONS, EVIDENCE)
    labels = [(c["label"], c["check"]) for c in out["report"]["claims"]]
    assert labels[0] == ("not_supported", "numbers not in the tool output: 120.4")
    assert labels[1][0] == "contradicted"
    assert out["report"]["groundedness"] == 0.0


async def test_llm_claim_split_absence_and_uncited() -> None:
    claims = ClaimList(
        claims=[
            ExtractedClaim(sentence=1, text="Metformin inhibits glioblastoma growth in mice",
                           citations=[1, 9]),
            ExtractedClaim(sentence=1, text="No clinical trial of metformin in GBM was found",
                           kind="absence"),
            ExtractedClaim(sentence=2, text="Unrelated statement with no source at all"),
        ]
    )  # fmt: skip
    calls: list[type[BaseModel]] = []

    async def call(schema: type[BaseModel], messages: list[BaseMessage]) -> BaseModel | None:
        calls.append(schema)
        if schema is ClaimList:
            return claims
        return AbsenceCheck(consistent=True, reason="no trial among the evidence")

    answer = "Metformin inhibits glioblastoma growth in mice [1].\nAn uncited sentence here."
    out = await verifier().verify(answer, CITATIONS, EVIDENCE, call=call)
    rows = out["report"]["claims"]
    assert [r["label"] for r in rows] == ["supported", "absence", "not_supported"]
    assert rows[0]["citations"] == [1]  # invalid citation 9 dropped
    assert rows[2]["check"] == "no resolvable citation"
    assert calls == [ClaimList, AbsenceCheck]
    assert out["report"]["counts"]["absence"] == 1
    assert out["report"]["groundedness"] == 0.5  # absence claims are not counted


async def test_absence_contradicted_by_evidence() -> None:
    async def call(schema: type[BaseModel], messages: list[BaseMessage]) -> BaseModel | None:
        if schema is ClaimList:
            return ClaimList(
                claims=[ExtractedClaim(sentence=1, text="No GBM trial exists", kind="absence")]
            )
        return AbsenceCheck(consistent=False, reason="a phase II trial is listed")

    out = await verifier().verify("No GBM trial exists for metformin [1].", CITATIONS,
                                  EVIDENCE, call=call)  # fmt: skip
    assert out["report"]["claims"][0]["label"] == "contradicted"


# --- in the agent graph ------------------------------------------------------------------
async def test_verifier_node_rewrites_answer(tmp_path: Path) -> None:
    from tests.agent_fakes import ToolLog, call, corpus_hit, fake_tools, plan, scripted_specialist
    from tests.test_agents import make_deps, run_graph

    hits = [corpus_hit("111", "Metformin inhibits glioblastoma growth in mice")]
    deps = make_deps(
        [plan(("t1", "literature", []))],
        scripted_specialist({"t1": [[call("search_corpus", {"query": "q"})]]}),
        "Metformin inhibits glioblastoma growth in mice [1]. Metformin cures every cancer [1].",
        fake_tools(ToolLog(), corpus={"q": hits}),
    )
    deps.claim_verifier = verifier()
    state: dict[str, Any] = await run_graph(deps, tmp_path)
    v = state["verification"]
    assert v["claim_verification"] == "on" and v["claim_split"] == "sentences"
    assert [c["label"] for c in v["claims"]] == ["supported", "not_supported"]
    assert state["answer"] == "Metformin inhibits glioblastoma growth in mice [1]."
    assert "cures" in state["draft_answer"]
    assert v["all_citations_resolve"] is True


async def test_citation_covers_every_retrieved_passage_of_the_paper() -> None:
    evidence = {
        "9:0": ev("9:0", "Title.\nMetformin inhibits glioblastoma growth in mice.", "9"),
        "9:1": ev("9:1", "Title.\nA later chunk about xenograft dosing only.", "9"),
    }
    citations = [cite(1, "9:1", "9")]  # the synthesizer saw chunk 1 only
    answer = "Metformin inhibits glioblastoma growth in mice [1]."
    paper = await verifier().verify(answer, citations, evidence)
    assert paper["report"]["claims"][0]["label"] == "supported"
    strict = ClaimVerifier(FakeNLI(keyword_rule), VerifyConfig(paper_level=False))
    chunk = await strict.verify(answer, citations, evidence)
    assert chunk["report"]["claims"][0]["label"] == "not_supported"


def test_top_windows_prefilter() -> None:
    from bioagent.verify.verifier import top_windows

    windows = ["alpha beta", "metformin glioblastoma growth", "gamma", "metformin mice", "x"]
    assert top_windows(windows, "Metformin inhibits glioblastoma growth", 2) == [
        "metformin glioblastoma growth",
        "metformin mice",
    ]
    assert top_windows(["a", "a", "b"], "z", 5) == ["a", "b"]


DRUG_LIKENESS = (
    '{"lipinski": {"name": "Lipinski Ro5", "passed": true, "violations": 0}, '
    '"veber": {"name": "Veber", "passed": true, "violations": 0}, '
    '"ghose": {"name": "Ghose", "passed": false, "violations": 3, "rules": '
    '[{"name": "clogp", "criterion": ">= -0.4"}]}, '
    '"bbb": {"likely_permeable": false, "criteria": [{"name": "tpsa", "value": 91.49}]}}'
)


@pytest.mark.parametrize(
    ("claim", "ok"),
    [
        ("Both pass Lipinski's Rule of Five with zero violations.", True),
        ("Both pass Veber rules.", True),
        ("Both fail the Ghose filter due to cLogP below the lower bound of -0.4.", True),
        ("Metformin is predicted non-permeable across the blood-brain barrier.", True),
        ("Metformin is predicted to cross the blood-brain barrier.", False),
        ("Metformin fails Lipinski's rule of five.", False),
        ("Metformin has two Lipinski violations.", False),
    ],
)
def test_rule_outcomes_in_computed_claims(claim: str, ok: bool) -> None:
    result = check_numbers(claim, [DRUG_LIKENESS, DRUG_LIKENESS])
    assert result.anything_checked and result.ok is ok, result.detail


async def test_computed_claim_without_values_is_not_sent_to_nli() -> None:
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    answer = "Metformin is a small molecule with favourable properties (computed) [3]."
    out = await ClaimVerifier(nli).verify(answer, CITATIONS, EVIDENCE)
    (claim,) = out["report"]["claims"]
    assert claim["label"] == "not_supported" and nli.pairs == []
    assert claim["check"] == "no numbers or rule outcomes to check"
