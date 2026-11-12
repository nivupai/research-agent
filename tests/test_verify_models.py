"""Real NLI model on hand-made premise/hypothesis pairs (``pytest -m models``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioagent.agents.state import Citation, Evidence
from bioagent.config import get_settings
from bioagent.verify.nli import TransformersNLI
from bioagent.verify.verifier import ClaimVerifier

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def nli() -> TransformersNLI:
    pytest.importorskip("transformers")
    return TransformersNLI(get_settings().nli_model)


def test_entailment_neutral_contradiction(nli: TransformersNLI) -> None:
    premise = (
        "Metformin reduced proliferation of U87 and U251 glioblastoma cells and, combined "
        "with temozolomide, increased apoptosis in an orthotopic mouse model."
    )
    entail, neutral, contra = nli.score(
        [
            (premise, "Metformin reduced glioblastoma cell proliferation."),
            (premise, "Metformin is approved for type 2 diabetes."),
            (premise, "Metformin increased the proliferation of glioblastoma cells."),
        ]
    )
    assert entail.entailment > 0.8
    assert neutral.entailment < 0.5
    assert contra.contradiction > 0.5


def _abstract(pmid: str) -> str | None:
    from bioagent.ingest.store import CorpusStore

    db = Path(get_settings().corpus_db)
    if not db.exists():
        return None
    store = CorpusStore(db)
    try:
        rec = store.get_records([pmid]).get(pmid)
    finally:
        store.close()
    return f"{rec['title']}\n{rec['abstract']}" if rec else None


async def test_miscitation_of_gynaecological_review_is_not_supported(
    nli: TransformersNLI,
) -> None:
    """Run 20261112-065725-73c776 cited this gynaecological-cancer review for a glioma claim."""
    text = _abstract("27378194")
    if text is None:
        pytest.skip("corpus not built (make ingest)")
    evidence = {
        "27378194:0": Evidence(
            key="27378194:0", pmid="27378194", passage_id="27378194:0", title=None,
            year=2016, journal=None, doi=None, text=text, source="corpus", task_id="t1",
            computed=False, untrusted=True,
        )
    }  # fmt: skip
    citations = [
        Citation(n=1, key="27378194:0", pmid="27378194", title=None, year=2016, url=None,
                 computed=False, source="corpus")
    ]  # fmt: skip
    answer = "The PI3K/Akt/mTOR axis is frequently dysregulated in high-grade gliomas [1]."
    out = await ClaimVerifier(nli).verify(answer, citations, evidence)
    (claim,) = out["report"]["claims"]
    assert claim["label"] == "not_supported", claim
    assert out["answer"] == ""  # removed under the default policy
