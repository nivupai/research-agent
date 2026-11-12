"""Gene names from NCBI Gene esummary (same rate-limited E-utilities client)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bioagent.common.httpclient import RateLimitedClient, UpstreamAPIError
from bioagent.mcp_servers.literature.pubmed import EUTILS_BASE, TOOL_NAME


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


BATCH = 200


def parse_gene_summary(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    result = payload.get("result", {})
    out: dict[str, dict[str, str]] = {}
    for uid in result.get("uids", []):
        rec = result.get(uid) or {}
        out[str(uid)] = {
            "symbol": str(rec.get("name", "")),
            "name": str(rec.get("description", "")),
            "organism": str((rec.get("organism") or {}).get("scientificname", "")),
        }
    return out


async def fetch_gene_names(
    http: RateLimitedClient,
    ids: Sequence[str],
    cache_file: Path,
    *,
    api_key: str | None = None,
    email: str | None = None,
) -> dict[str, dict[str, str]]:
    cache: dict[str, dict[str, str]] = _read_json(cache_file) or {}
    todo = [i for i in dict.fromkeys(ids) if i not in cache]
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        params: dict[str, Any] = {
            "db": "gene",
            "id": ",".join(batch),
            "retmode": "json",
            "tool": TOOL_NAME,
        }
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email
        try:
            response = await http.get(f"{EUTILS_BASE}/esummary.fcgi", params=params)
        except UpstreamAPIError:
            continue
        cache.update(parse_gene_summary(response.json()))
        for gid in batch:
            cache.setdefault(gid, {"symbol": "", "name": "", "organism": ""})
        _write_json(cache_file, cache)
    return cache
