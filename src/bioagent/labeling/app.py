"""A local labelling page: one item at a time, keyboard shortcuts, saved after every answer.

Run with ``bioagent label serve`` (binds 127.0.0.1). Requests from non-loopback clients are
refused as well, in case the server is started with another host by mistake.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable, Mapping
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from bioagent.labeling.sheets import Label, Sheet


class LabelBody(BaseModel):
    value: Label | None


def _is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


PAGE = files("bioagent.labeling").joinpath("page.html").read_text(encoding="utf-8")


def create_app(sheets: Mapping[str, Sheet]) -> FastAPI:
    app = FastAPI(title="bioagent labelling", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def local_only(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not _is_loopback(request.client.host if request.client else None):
            return JSONResponse({"detail": "local access only"}, status_code=403)
        return await call_next(request)

    def get(name: str) -> Sheet:
        if name not in sheets:
            raise HTTPException(404, f"unknown sheet {name!r}")
        return sheets[name]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/sheets")
    def list_sheets() -> list[dict[str, object]]:
        return [
            {
                "name": s.spec.name,
                "title": s.spec.title,
                "path": str(s.path),
                "instructions": s.spec.instructions,
                "progress": s.progress(),
            }
            for s in sheets.values()
        ]

    @app.get("/api/{name}/next")
    def next_item(name: str, after: str | None = None, skipped: str = "") -> dict[str, object]:
        sheet = get(name)
        key = sheet.next_key(after, [k for k in skipped.split(",") if k])
        if key is None:
            return {"done": True, "progress": sheet.progress()}
        return sheet.item(key)

    @app.get("/api/{name}/previous")
    def previous_item(name: str, before: str) -> dict[str, object]:
        sheet = get(name)
        key = sheet.previous_key(before)
        if key is None:
            raise HTTPException(404, "no earlier item")
        return sheet.item(key)

    @app.get("/api/{name}/item")
    def item(name: str, key: str) -> dict[str, object]:
        try:
            return get(name).item(key)
        except KeyError as exc:
            raise HTTPException(404, f"no item {key!r}") from exc

    @app.post("/api/{name}/item")
    def label(name: str, key: str, body: LabelBody) -> dict[str, object]:
        try:
            return {"progress": get(name).set_label(key, body.value)}
        except KeyError as exc:
            raise HTTPException(404, f"no item {key!r}") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
