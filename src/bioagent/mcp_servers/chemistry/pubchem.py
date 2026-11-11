"""PubChem PUG-REST name resolution (name -> CID + SMILES)."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from rdkit import Chem

from bioagent.common.httpclient import RateLimitedClient, UpstreamAPIError, UpstreamHTTPError
from bioagent.mcp_servers.chemistry.models import ResolvedName

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_HOST = "pubchem.ncbi.nlm.nih.gov"
PUBCHEM_RATE = 5.0
MAX_NAME_LENGTH = 200
# Letters, digits, spaces and the punctuation found in drug/IUPAC names.
_NAME_RE = re.compile(r"^[\w\s\-,.'()\[\]+]+$")
# PubChem renamed CanonicalSMILES/IsomericSMILES to ConnectivitySMILES/SMILES in 2025.
_SMILES_KEYS = ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES")


class NameNotFoundError(ValueError):
    """PubChem has no compound for this name."""


def validate_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("empty compound name")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"compound name longer than {MAX_NAME_LENGTH} characters")
    if not _NAME_RE.match(name):
        raise ValueError("compound name contains unsupported characters")
    return name


class PubChemClient:
    def __init__(self, http: RateLimitedClient, *, base_url: str = PUBCHEM_BASE) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    async def resolve_name(self, name: str) -> ResolvedName:
        name = validate_name(name)
        url = (
            f"{self._base}/compound/name/{quote(name, safe='')}/property/"
            "SMILES,ConnectivitySMILES,MolecularFormula/JSON"
        )
        try:
            response = await self._http.get(url)
        except UpstreamHTTPError as exc:
            if exc.status_code == 404:
                raise NameNotFoundError(f"PubChem has no compound named {name!r}") from exc
            raise
        props: list[dict[str, Any]] = (response.json().get("PropertyTable") or {}).get(
            "Properties"
        ) or []
        if not props:
            raise NameNotFoundError(f"PubChem has no compound named {name!r}")
        first = props[0]
        smiles = next((first[k] for k in _SMILES_KEYS if first.get(k)), None)
        if not smiles:
            raise UpstreamAPIError("PubChem response had no SMILES")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise UpstreamAPIError("PubChem returned a SMILES that RDKit cannot parse")
        return ResolvedName(
            name=name,
            cid=int(first["CID"]),
            canonical_smiles=Chem.MolToSmiles(mol),
            pubchem_smiles=smiles,
            formula=first.get("MolecularFormula"),
        )
