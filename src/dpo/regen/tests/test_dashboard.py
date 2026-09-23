"""Research-console HTTP boundaries and safe export behavior."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.regen.dashboard import build_dashboard_app


class FixtureData:
    def __init__(self, *_: Any) -> None:
        pass

    def overview(self, include_qa: bool = False) -> dict[str, Any]:
        return {
            "participants": [{"id": "participant-a", "label": "participant-a", "qa": False}],
            "responses": [
                {"participant": "participant-a", "phase": 1, "value": 0, "item_text": "Rating"},
                {"participant": "participant-a", "phase": 2, "value": '=HYPERLINK("bad")'},
                {"participant": "participant-a", "phase": 2, "value": "\t@danger"},
                {"participant": "participant-a", "phase": 2, "value": "줄바꿈\ntext,<script>"},
            ],
            "qa_included": include_qa,
        }

    def participant(self, identifier: str, include_qa: bool = False) -> dict[str, Any]:
        if identifier != "participant-a":
            raise KeyError(identifier)
        return {"participant": {"id": identifier}, "qa_included": include_qa}


@pytest.fixture
def app(monkeypatch: Any, tmp_path: Path) -> Any:
    monkeypatch.setattr("dpo.regen.dashboard.DashboardData", FixtureData)
    return build_dashboard_app(tmp_path / "legacy", tmp_path / "viewing")


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/api/overview",
        "/api/participants/participant-a",
        "/api/export/responses.csv",
        "/api/export/participants/participant-a.json",
    ],
)
def test_dashboard_denies_remote_clients(app: Any, path: str) -> None:
    with TestClient(app, base_url="http://localhost", client=("203.0.113.10", 40000)) as client:
        assert client.get(path).status_code == 403


@pytest.mark.parametrize(
    "headers",
    [
        {"host": "attacker.example"},
        {"host": "concap.taild3b716.ts.net"},
        {"x-forwarded-for": "127.0.0.1"},
        {"x-forwarded-host": "localhost"},
        {"forwarded": "for=127.0.0.1"},
        {"x-forwarded-proto": "https"},
        {"origin": "https://attacker.example"},
        {"sec-fetch-site": "cross-site"},
    ],
)
def test_dashboard_denies_forwarded_and_rebinding_requests(app: Any, headers: dict[str, str]) -> None:
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 40000)) as client:
        assert client.get("/api/overview", headers=headers).status_code == 403


def test_local_read_only_routes_and_headers(app: Any) -> None:
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 40000)) as client:
        result = client.get("/api/overview")
        assert result.status_code == 200 and result.json()["qa_included"] is False
        assert client.get("/api/overview?include_qa=true").json()["qa_included"] is True
        assert result.headers["cache-control"] == "no-store"
        assert result.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in result.headers["content-security-policy"]
        assert client.post("/api/overview", json={}).status_code == 405
        assert client.get("/api/participants/missing").status_code == 404
        assert client.get("/secrets.txt").status_code == 404
        assert client.get("/api/health").json()["mode"] == "local-read-only"


def test_csv_safe_roundtrip_and_session_export(app: Any) -> None:
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 40000)) as client:
        response = client.get("/api/export/responses.csv")
        assert response.status_code == 200
        rows = list(csv.DictReader(io.StringIO(response.text.lstrip("\ufeff"))))
        assert [row["value"] for row in rows] == [
            "0",
            '\'=HYPERLINK("bad")',
            "'\t@danger",
            "줄바꿈\ntext,<script>",
        ]
        export = client.get("/api/export/participants/participant-a.json?include_qa=true")
        assert export.status_code == 200 and export.json()["qa_included"] is True
        assert "attachment" in export.headers["content-disposition"]
