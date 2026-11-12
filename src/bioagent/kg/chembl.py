"""ChEMBL enrichment: drug name -> ChEMBL id, max_phase and mechanisms of action.

Mechanisms are often recorded on salt forms (e.g. metformin hydrochloride), so mechanisms
are collected across every form of the parent molecule. Target components give gene
symbols, so a mechanism on the AMPK complex group expands to its subunit genes.
Every response is cached under ``data/kg/chembl/`` and requests are rate-limited.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bioagent.common.httpclient import RateLimitedClient, UpstreamAPIError


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_HOST = "www.ebi.ac.uk"
CHEMBL_RATE = 4.0
MOLECULE_FIELDS = "molecule_chembl_id,pref_name,max_phase,molecule_hierarchy"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:120] or "_"


def parse_phase(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def pick_molecule(molecules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer the highest clinical phase; report the parent molecule id."""
    if not molecules:
        return None
    best = max(molecules, key=lambda m: parse_phase(m.get("max_phase")) or -1.0)
    parent = (best.get("molecule_hierarchy") or {}).get("parent_chembl_id")
    return {
        "chembl_id": parent or best["molecule_chembl_id"],
        "pref_name": best.get("pref_name"),
        "max_phase": parse_phase(best.get("max_phase")),
    }


def parse_target(target: dict[str, Any]) -> dict[str, Any]:
    genes: list[str] = []
    for comp in target.get("target_components", []) or []:
        for syn in comp.get("target_component_synonyms", []) or []:
            if syn.get("syn_type") == "GENE_SYMBOL" and syn.get("component_synonym"):
                genes.append(str(syn["component_synonym"]).upper())
                break
    return {
        "target_chembl_id": target.get("target_chembl_id"),
        "target_name": target.get("pref_name"),
        "target_type": target.get("target_type"),
        "organism": target.get("organism"),
        "genes": sorted(set(genes)),
    }


class ChEMBLClient:
    def __init__(
        self, http: RateLimitedClient, cache_dir: Path, base_url: str = CHEMBL_BASE
    ) -> None:
        self.http = http
        self.base = base_url.rstrip("/")
        self.mol_dir = cache_dir / "molecules"
        self.target_dir = cache_dir / "targets"
        self.mol_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self.http.get(f"{self.base}/{path}", params=params)
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def _find_molecule(self, name: str) -> dict[str, Any] | None:
        for key in ("pref_name__iexact", "molecule_synonyms__molecule_synonym__iexact"):
            data = await self._get(
                "molecule.json", {key: name, "only": MOLECULE_FIELDS, "limit": 20}
            )
            picked = pick_molecule(data.get("molecules", []))
            if picked:
                return picked
        return None

    async def target(self, target_id: str) -> dict[str, Any]:
        path = self.target_dir / f"{target_id}.json"
        cached: dict[str, Any] | None = _read_json(path)
        if cached is not None:
            return cached
        info = parse_target(await self._get(f"target/{target_id}.json"))
        _write_json(path, info)
        return info

    async def lookup(self, name: str, *, cache_only: bool = False) -> dict[str, Any]:
        """Return {query, chembl_id, pref_name, max_phase, mechanisms}; cached per name.

        With ``cache_only`` an uncached name returns an empty record marked
        ``"uncached": True`` instead of calling the API.
        """
        path = self.mol_dir / f"{_slug(name)}.json"
        cached: dict[str, Any] | None = _read_json(path)
        if cached is not None:
            return cached
        record: dict[str, Any] = {
            "query": name,
            "chembl_id": None,
            "pref_name": None,
            "max_phase": None,
            "mechanisms": [],
        }
        if cache_only:
            return {**record, "uncached": True}
        try:
            molecule = await self._find_molecule(name)
            if molecule:
                record.update(molecule)
                if (molecule["max_phase"] or 0) >= 1:
                    record["mechanisms"] = await self._mechanisms(molecule["chembl_id"])
        except UpstreamAPIError as exc:
            record["error"] = str(exc)[:300]
            return record  # not cached: retry next run
        _write_json(path, record)
        return record

    async def _mechanisms(self, parent_id: str) -> list[dict[str, Any]]:
        forms = await self._get(f"molecule_form/{parent_id}.json")
        ids = sorted(
            {f["molecule_chembl_id"] for f in forms.get("molecule_forms", [])} | {parent_id}
        )
        data = await self._get(
            "mechanism.json", {"molecule_chembl_id__in": ",".join(ids), "limit": 100}
        )
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for mech in data.get("mechanisms", []):
            tid = mech.get("target_chembl_id")
            action = str(mech.get("action_type") or "UNKNOWN")
            if not tid or (tid, action) in seen:
                continue
            seen.add((tid, action))
            target = await self.target(tid)
            out.append(
                {
                    "action_type": action,
                    "mechanism_of_action": mech.get("mechanism_of_action"),
                    **target,
                }
            )
        return out
