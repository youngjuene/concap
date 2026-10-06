"""Local researcher console for existing Phase 1/2 records; never serves via Funnel."""

from __future__ import annotations

import argparse
import csv
import io
import json
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from dpo.regen.dashboard_data import DashboardData

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PROXY_HEADERS = {"forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"}


def _local_request(request: Request) -> bool:
    if request.client is None or request.client.host not in LOCAL_HOSTS:
        return False
    if any(name in request.headers for name in PROXY_HEADERS):
        return False
    try:
        if urlsplit(f"http://{request.headers.get('host', '')}").hostname not in LOCAL_HOSTS:
            return False
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return False
    except ValueError:
        return False
    return request.headers.get("sec-fetch-site") != "cross-site"


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _csv_response(rows: list[dict[str, Any]], fields: list[str], filename: str) -> Response:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_cell(row.get(field)) for field in fields})
    return Response(
        "\ufeff" + stream.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"content-disposition": f'attachment; filename="{filename}"'},
    )


def build_dashboard_app(legacy_dir: Path, viewing_dir: Path) -> FastAPI:
    """Use a separate local port so participant publication cannot expose research data."""
    data = DashboardData(legacy_dir.resolve(), viewing_dir.resolve())
    app = FastAPI(title="Concap research console", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def local_boundary(request: Request, call_next: Any) -> Response:
        if not _local_request(request):
            return JSONResponse({"error": "Research dashboard is available on the study machine only."}, 403)
        response: Response = await call_next(request)
        response.headers["cache-control"] = "no-store"
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "no-referrer"
        response.headers["x-frame-options"] = "DENY"
        response.headers["content-security-policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("dpo.regen").joinpath("dashboard.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "local-read-only"}

    @app.get("/api/overview")
    def overview(include_qa: bool = False) -> dict[str, Any]:
        return data.overview(include_qa=include_qa)

    @app.get("/api/phase1-analysis")
    def phase1_analysis(
        include_qa: bool = False,
        participant: str = "",
        language: str = "",
        status: str = "",
        clip_id: str = "",
    ) -> dict[str, Any]:
        return data.phase1_analysis(
            include_qa, participant=participant, language=language, status=status, clip_id=clip_id
        )

    @app.get("/api/export/phase1-observations.csv")
    def phase1_observations_csv(
        include_qa: bool = False,
        participant: str = "",
        language: str = "",
        status: str = "",
        clip_id: str = "",
    ) -> Response:
        from dpo.regen.phase1_analysis import OBSERVATION_FIELDS

        result = phase1_analysis(include_qa, participant, language, status, clip_id)
        return _csv_response(result["observations"], OBSERVATION_FIELDS, "concap-phase1-observations.csv")

    @app.get("/api/export/phase1-pairs.csv")
    def phase1_pairs_csv(
        include_qa: bool = False,
        participant: str = "",
        language: str = "",
        status: str = "",
        clip_id: str = "",
    ) -> Response:
        from dpo.regen.phase1_analysis import PAIR_FIELDS

        result = phase1_analysis(include_qa, participant, language, status, clip_id)
        return _csv_response(result["pairs"], PAIR_FIELDS, "concap-phase1-pairs.csv")

    def detail(identifier: str, include_qa: bool) -> dict[str, Any]:
        try:
            return data.participant(identifier, include_qa=include_qa)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "Participant not found in this selection") from exc

    @app.get("/api/participants/{identifier}")
    def participant(identifier: str, include_qa: bool = False) -> dict[str, Any]:
        return detail(identifier, include_qa)

    @app.get("/api/export/responses.csv")
    def responses_csv(include_qa: bool = False) -> Response:
        rows = data.overview(include_qa=include_qa)["responses"]
        fields = [
            "participant",
            "phase",
            "scope",
            "video_id",
            "view_id",
            "condition",
            "flow_version",
            "instrument",
            "item_id",
            "item_text",
            "wording_available",
            "type",
            "value",
            "points",
            "min",
            "max",
            "labels",
            "options",
            "option_labels",
            "source",
            "source_row",
            "source_row_hash",
            "scale_source",
            "presented_order",
            "presentation_index",
            "client_submitted_at",
            "stimulus_assignment",
            "caption_strategy",
            "language",
            "submitted_at",
        ]
        return _csv_response(rows, fields, "concap-responses.csv")

    @app.get("/api/export/participants/{identifier}.json")
    def participant_json(identifier: str, include_qa: bool = False) -> Response:
        result = detail(identifier, include_qa)
        return Response(
            json.dumps(result, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"content-disposition": 'attachment; filename="concap-session.json"'},
        )

    @app.get("/{asset}")
    def asset_file(asset: str) -> Response:
        types = {"dashboard.css": "text/css", "dashboard.js": "text/javascript"}
        if asset not in types:
            raise HTTPException(404, "Not found")
        return Response(
            files("dpo.regen").joinpath(asset).read_text(encoding="utf-8"), media_type=types[asset]
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, default=Path("data/live/regen-responses"))
    parser.add_argument("--viewing-dir", type=Path, default=Path("data/live/regen-viewing-responses"))
    parser.add_argument("--host", choices=sorted(LOCAL_HOSTS), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    import uvicorn

    uvicorn.run(
        build_dashboard_app(args.legacy_dir, args.viewing_dir),
        host=args.host,
        port=args.port,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
