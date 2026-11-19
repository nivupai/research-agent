# Design: target-specific evidence for drug candidates

Status: proposal (2026-11-19). Nothing here is implemented yet except the offline check below.

## Problem

For "which approved drugs act on target X" questions, `discover_drugs` builds a candidate list
from two sources: the knowledge graph (`drugs_targeting`: ChEMBL mechanisms and text-mined
relations) and the corpus (`drugs_mentioned`: every drug PubTator annotated in the papers
the discovery searches returned). The corpus part is ranked by mention count and cut at 20
candidates. Mention counts say nothing about whether a paper reports the drug *acting on the
target*, so frequent but irrelevant entities (ATP, glucose, lipids, drugs listed in
repurposing reviews) fill the list, and a drug like sorafenib, whose one retrieved paper is
about AMPK activation by sorafenib, is cut (failure analysis sections 13-15). Ranking by the
search rank of the mentioning papers does not fix it (section 15).

## Offline feasibility check (no LLM)

AMPK question, discovery searches with rerank fusion, 56 candidates (graph + corpus). For each
candidate, count sentences in the retrieved titles and abstracts that contain the drug (name
or KG synonym), the target (AMPK, "AMP-activated protein kinase", PRKAA1/2) and an action
word (activat-, inhibit-, stimulat-, induc-, suppress-, increas-, decreas-, phosphorylat-):

* 22 of 56 candidates have at least one such sentence.
* Ranked by (papers with such a sentence, sentences), **sorafenib comes 13th** (it was 44th by
  mention count), so it would survive the 20-candidate cap.
* Still ranked high: AICA ribonucleotide (the AMP analogue AICAR), glucose, lipids, AMP,
  adenosine, sodium: endogenous molecules or tool compounds that do co-occur with "AMPK
  activation" sentences. Graph-only drugs (berberine, phenformin) have no such sentence in
  this paper set, so the sentence score must not replace the graph source.

## Proposal

Setting `DRUG_CANDIDATE_EVIDENCE = off | sentence | sentence_llm` (default `off`).

1. **sentence (deterministic, no LLM)**: for every corpus candidate, find the best
   "drug + target + action" sentence in the discovery hits (target synonyms from
   `kg.aliases`, drug synonyms from the Drug node). Rank corpus-only candidates by
   (papers with such a sentence, best search rank of those papers, mention count); keep the
   graph/ChEMBL ordering as is; keep the 20-candidate cap. Attach the sentence and its PMID
   to each candidate in the list the specialist sees, so the model reads target-specific
   evidence instead of bare names.
2. **sentence_llm (one extra flash-lite call per discovery task)**: send the top ~30
   candidates with their evidence sentence in one structured call and ask, per candidate:
   is it a therapeutic drug (not an endogenous metabolite, ion, nutrient or lab tool
   compound)? does the sentence report it acting on the target, and in which direction?
   Drop the "no" rows, then apply the cap. The sentences are wrapped as untrusted text like
   all retrieved content. Cost: about 1 call and ~3k tokens per discovery task.

## Evaluation plan

* **Question set**: about 8 discovery questions (AMPK, mTOR, PI3K, EGFR, HDAC, PARP, VEGFR,
  COX-2), each with a gold list of approved drugs known to act on the target, taken from
  ChEMBL curated mechanisms plus a short manual list for known indirect modulators (for
  AMPK: metformin, phenformin, aspirin/salicylate, TZDs, sorafenib, ...). Gold lists are
  written before any run and stored in `eval/questions/discovery_gold.jsonl`.
* **Offline metric (no quota)**: candidate recall of gold drugs within the top 20, and the
  share of non-drug entities in the top 20, for off / sentence (and the LLM step's effect,
  which needs calls, on cached labels once run).
* **Live metric**: answer mention recall and precision of gold drugs, references,
  groundedness, latency, cost; 3 runs per (question, setting), settings interleaved,
  flash-lite, paired tests over (question, repeat).
* **Decision rule**: change the default only if gold recall in answers improves with a
  paired CI excluding zero and precision does not drop.

## Risks

* Abstracts often name the drug and the target in different sentences; a window of two
  sentences may be needed (measure both).
* Synonym coverage: PubTator/KG synonyms miss some brand names; the drug's own mention
  annotation could be used instead of string matching.
* Gold lists for indirect modulators are judgement calls; they will be reviewed before the
  runs and kept fixed afterwards.
