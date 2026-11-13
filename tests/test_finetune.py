"""Small-verifier pipeline: pair builder, perturbations, label parsing, router, metrics."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import pytest

from bioagent.finetune.data import (
    Pair,
    Paper,
    check_no_leakage,
    claims_from_record,
    dedupe,
    pair_id,
    perturb,
    split_for,
    synthetic_pairs,
)
from bioagent.finetune.labels import PairLabel, PairLabels, nli_label, parse_labels, teacher_key
from bioagent.finetune.metrics import accuracy, bootstrap_ci, confusion, macro_f1, percentile
from bioagent.llm.providers import ModelSpec
from bioagent.verify.backends import RouterScorer, label_scores, proba_scores, route, windowed
from bioagent.verify.nli import NLIScores

ABSTRACT = (
    "Background sentence about diabetes. Metformin significantly increased AMPK "
    "phosphorylation in U87 glioblastoma cells by 45% compared with controls. "
    "Temozolomide reduced tumour growth in mice treated for 21 days."
)


# --- splits and leakage -----------------------------------------------------------------
def test_split_is_deterministic_and_roughly_80_10_10() -> None:
    assert split_for("12345") == split_for("12345")
    splits = [split_for(str(i)) for i in range(5000)]
    shares = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert abs(shares["test"] - 0.1) < 0.02 and abs(shares["val"] - 0.1) < 0.02


def papers(n: int) -> list[Paper]:
    return [Paper(str(1000 + i), f"Paper {i}", ABSTRACT.replace("45%", f"{i}%")) for i in range(n)]


def test_synthetic_pairs_never_cross_splits() -> None:
    ps = papers(80)
    ids = [p.pmid for p in ps]
    neighbours = {p: [o for o in ids if o != p] for p in ids}  # every paper is a "neighbour"
    pairs = synthetic_pairs(ps, neighbours, n_positive=40, n_cross=40, n_perturbed=40)
    kinds = {p.kind for p in pairs}
    assert {"positive", "cross"} <= kinds and kinds & {"direction", "negation", "swap", "number"}
    check_no_leakage(pairs)  # raises if a PMID is in two splits
    for p in pairs:
        if p.kind == "cross":
            assert p.claim_pmid != p.pmid and split_for(p.claim_pmid or "") == p.split


def test_leakage_check_detects_shared_pmid() -> None:
    a = Pair("claim one is here", "passage", "1", "positive").finalise()
    b = Pair("claim two is here", "passage", "2", "cross", claim_pmid="1").finalise()
    b.split = "test" if a.split != "test" else "train"
    with pytest.raises(ValueError, match="appears in"):
        check_no_leakage([a, b])


def test_dedupe_and_ids() -> None:
    a = Pair("Metformin  activates AMPK.", "P", "1", "run").finalise()
    b = Pair("metformin activates ampk.", "p", "1", "positive").finalise()
    assert a.id == b.id == pair_id("Metformin activates AMPK.", "P")
    assert len(dedupe([a, b])) == 1


# --- perturbations ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kind", "claim", "expect"),
    [
        ("direction", "Metformin increased AMPK activity.", "decreased"),
        ("negation", "The effect was associated with survival.", "was not associated"),
        ("swap", "Metformin reduced growth.", None),
        ("number", "Growth fell by 45% at 21 days.", None),
    ],
)
def test_perturbations_change_the_claim(kind: str, claim: str, expect: str | None) -> None:
    out = perturb(claim, kind, random.Random(0))
    assert out is not None
    text, detail = out
    assert text != claim and "->" in detail
    if expect:
        assert expect in text
    if kind == "swap":
        assert "Metformin" not in text
    if kind == "number":
        assert "45" not in text


def test_perturbation_not_applicable() -> None:
    rng = random.Random(0)
    assert perturb("A sentence without numbers or entities", "number", rng) is None
    assert perturb("A sentence without numbers or entities", "swap", rng) is None


def test_claims_from_record_uses_cited_passages() -> None:
    record = {
        "draft_answer": "Metformin activates AMPK in glioblastoma cells strongly [1].\n"
        "The TPSA of metformin is 91.5 (computed estimate) [2].",
        "citations": [{"n": 1, "key": "k1"}, {"n": 2, "key": "c1"}],
        "cited_evidence": {
            "k1": {"text": "Passage about AMPK", "pmid": "77", "computed": False},
            "c1": {"text": "{}", "pmid": None, "computed": True},
        },
    }
    pairs = list(claims_from_record(record))
    assert [(p.claim, p.pmid, p.kind) for p in pairs] == [
        ("Metformin activates AMPK in glioblastoma cells strongly.", "77", "run")
    ]


# --- labels ----------------------------------------------------------------------------------
def test_parse_labels_keeps_only_requested_valid_ids() -> None:
    parsed = PairLabels(
        labels=[
            PairLabel(id="a", label="SUPPORTED"),
            PairLabel(id="a", label="CONTRADICTED"),  # duplicate: first wins
            PairLabel(id="zzz", label="SUPPORTED"),  # not requested
            PairLabel(id="b", label="NOT_ENOUGH_INFO"),
        ]
    )
    assert parse_labels(parsed, ["a", "b", "c"]) == {"a": "SUPPORTED", "b": "NOT_ENOUGH_INFO"}
    assert parse_labels(None, ["a"]) == {}
    assert parse_labels({"labels": []}, ["a"]) == {}


def test_teacher_key_separates_inference_from_training_labels() -> None:
    p = Pair("claim text here", "passage", "1", "run").finalise()
    spec = ModelSpec("gemini", "x")
    assert teacher_key(p, spec) != teacher_key(p, spec, "llm-verifier-v1")
    assert teacher_key(p, spec) != teacher_key(p, ModelSpec("gemini", "y"))


def test_nli_label_mapping() -> None:
    assert nli_label(0.9, 0.05) == "SUPPORTED"
    assert nli_label(0.1, 0.8) == "CONTRADICTED"
    assert nli_label(0.3, 0.3) == "NOT_ENOUGH_INFO"
    assert nli_label(0.6, 0.7) == "CONTRADICTED"


# --- router ----------------------------------------------------------------------------------
class FakeSmall:
    def __init__(self, probs: list[list[float]]) -> None:
        self.probs = probs

    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
        return self.probs[: len(pairs)]


class FakeLLM:
    name = "llm"

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        self.seen.extend(pairs)
        return [label_scores("CONTRADICTED") for _ in pairs]


def test_router_escalates_low_confidence_only() -> None:
    probs = [[0.95, 0.03, 0.02], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]
    assert route(probs, 0.8) == [False, True, False]
    llm = FakeLLM()
    router = RouterScorer(FakeSmall(probs), llm, threshold=0.8)  # type: ignore[arg-type]
    pairs = [("p1", "c1"), ("p2", "c2"), ("p3", "c3")]
    scores = router.score(pairs)
    assert llm.seen == [("p2", "c2")] and (router.escalated, router.total) == (1, 3)
    assert scores[0] == proba_scores(probs[0]) and scores[0].entailment == 0.95
    assert scores[1].contradiction == 0.9  # from the LLM
    assert scores[2].neutral == 0.8
    assert route(probs, 0.0) == [False] * 3 and route(probs, 1.01) == [True] * 3


def test_backend_windowing() -> None:
    assert windowed("nli") and not windowed("small_lora") and not windowed("router")


# --- metrics ------------------------------------------------------------------------------------
def test_metrics_known_values() -> None:
    gold = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    pred = ["SUPPORTED", "CONTRADICTED", "CONTRADICTED", "SUPPORTED"]
    assert accuracy(gold, pred) == 0.5
    # F1: SUPPORTED p=1/2 r=1/2 -> 0.5; CONTRADICTED p=1/2 r=1 -> 2/3; NEI 0 -> macro 7/18
    assert macro_f1(gold, pred) == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    m = confusion(gold, pred)
    assert m["SUPPORTED"]["CONTRADICTED"] == 1 and m["NOT_ENOUGH_INFO"]["SUPPORTED"] == 1
    assert macro_f1(gold, gold) == 1.0
    ci = bootstrap_ci(gold * 25, gold * 25, accuracy, b=200)
    assert ci["mean"] == ci["lo"] == ci["hi"] == 1.0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5 and percentile([], 0.5) is None


@pytest.mark.models
def test_trained_adapter_loads_and_scores() -> None:
    adapter = Path("models/verifier-lora")
    if not (adapter / "training.json").exists():
        pytest.skip("no trained adapter (scripts/train_verifier_lora.py)")
    from bioagent.finetune.model import SmallVerifier

    probs = SmallVerifier(adapter).predict_proba(
        [("Metformin activates AMPK.", "Metformin activates AMPK.")]
    )
    assert len(probs) == 1 and abs(sum(probs[0]) - 1) < 1e-3


def test_llm_backend_never_invents_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from bioagent.eval.cache import JSONCache
    from bioagent.finetune import labels as labels_mod
    from bioagent.verify.backends import LLMPairScorer, LLMVerifierUnavailableError

    async def quota_out(*args: object, **kwargs: object) -> dict[str, object]:
        return {"stopped": "quota exhausted", "calls": 0}

    monkeypatch.setattr(labels_mod, "label_with_teacher", quota_out)
    scorer = LLMPairScorer(model=None, spec=ModelSpec("gemini", "x"), cache=JSONCache(":memory:"))
    with pytest.raises(LLMVerifierUnavailableError, match="quota exhausted"):
        scorer.labels([("passage", "claim")])
