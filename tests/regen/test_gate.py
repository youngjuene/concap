"""What a published instrument refuses that a kiosk does not."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.regen.app import build_app
from dpo.regen.gate import Buckets, Gate
from dpo.regen.regeneration import RegenTemplateWriter


def app_with(document: dict[str, Any], media_dir: Path, tmp_path: Path, gate: Gate | None) -> Any:
    return build_app(document, media_dir, tmp_path / "out", RegenTemplateWriter(), gate=gate)


class TestBuckets:
    def test_burst_then_refill(self) -> None:
        now = [0.0]
        buckets = Buckets(rate=1.0, burst=2, clock=lambda: now[0])
        assert buckets.allow("a") and buckets.allow("a")
        assert not buckets.allow("a")
        now[0] += 1.0
        assert buckets.allow("a")
        assert not buckets.allow("a")

    def test_keys_are_independent(self) -> None:
        buckets = Buckets(rate=1.0, burst=1, clock=lambda: 0.0)
        assert buckets.allow("a")
        assert not buckets.allow("a")
        assert buckets.allow("b")

    def test_refuses_a_bucket_that_could_never_allow(self) -> None:
        with pytest.raises(ValueError):
            Buckets(rate=0, burst=1)


class TestAccessCode:
    @pytest.fixture
    def code_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "code"
        path.write_text("open-sesame\n")
        return path

    @pytest.fixture
    def client(
        self, document: dict[str, Any], media_dir: Path, tmp_path: Path, code_file: Path
    ) -> TestClient:
        return TestClient(app_with(document, media_dir, tmp_path, Gate(code_file=code_file)))

    def test_no_code_is_closed(self, client: TestClient) -> None:
        response = client.post("/api/session", json={})
        assert response.status_code == 403
        assert response.json()["closed"] is True

    def test_wrong_code_is_closed(self, client: TestClient) -> None:
        assert client.post("/api/session", json={"code": "nope"}).status_code == 403

    def test_right_code_enrols(self, client: TestClient) -> None:
        body = client.post("/api/session", json={"code": "open-sesame"}).json()
        assert body["participant"].startswith("p")

    def test_resume_needs_no_code(self, client: TestClient) -> None:
        first = client.post("/api/session", json={"code": "open-sesame"}).json()
        again = client.post("/api/session", json={"participant": first["participant"]}).json()
        assert again["participant"] == first["participant"]

    def test_changing_the_file_closes_at_once(self, client: TestClient, code_file: Path) -> None:
        assert client.post("/api/session", json={"code": "open-sesame"}).status_code == 200
        code_file.write_text("something-else\n")
        assert client.post("/api/session", json={"code": "open-sesame"}).status_code == 403
        assert client.post("/api/session", json={"code": "something-else"}).status_code == 200

    def test_empty_or_missing_file_is_closed(self, client: TestClient, code_file: Path) -> None:
        code_file.write_text("  \n")
        response = client.post("/api/session", json={"code": ""})
        assert response.status_code == 503 and response.json()["closed"] is True
        code_file.unlink()
        assert client.post("/api/session", json={"code": "open-sesame"}).status_code == 503

    def test_a_kiosk_needs_none(self, document: dict[str, Any], media_dir: Path, tmp_path: Path) -> None:
        client = TestClient(app_with(document, media_dir, tmp_path, None))
        body = client.post("/api/session", json={}).json()
        assert body["participant"].startswith("p") and body["download"] is True


class TestLogStaysHome:
    @pytest.fixture
    def app(self, document: dict[str, Any], media_dir: Path, tmp_path: Path) -> Any:
        return app_with(document, media_dir, tmp_path, Gate(log_local_only=True))

    def test_from_elsewhere_refused_and_not_offered(self, app: Any) -> None:
        client = TestClient(app)  # peer address "testclient"
        body = client.post("/api/session", json={}).json()
        assert body["download"] is False
        response = client.get("/api/log", params={"participant": body["participant"]})
        assert response.status_code == 403

    def test_from_this_machine_served(self, app: Any) -> None:
        client = TestClient(app, client=("127.0.0.1", 40000))
        body = client.post("/api/session", json={}).json()
        assert body["download"] is True
        response = client.get("/api/log", params={"participant": body["participant"]})
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]

    def test_a_proxy_in_front_is_not_local(self, app: Any) -> None:
        client = TestClient(app, client=("127.0.0.1", 40000))
        headers = {"x-forwarded-for": "203.0.113.9"}
        body = client.post("/api/session", json={}, headers=headers).json()
        assert body["download"] is False
        response = client.get("/api/log", params={"participant": body["participant"]}, headers=headers)
        assert response.status_code == 403


class TestRateLimit:
    def test_requests_by_address(self, document: dict[str, Any], media_dir: Path, tmp_path: Path) -> None:
        # A refill so slow that nothing comes back during the test: only the
        # burst is spent, and the third request is the first one over it.
        gate = Gate(requests_per_second=0.001, burst=2)
        client = TestClient(app_with(document, media_dir, tmp_path, gate))
        assert client.get("/api/strings").status_code == 200
        assert client.get("/api/strings").status_code == 200
        assert client.get("/api/strings").status_code == 429

    def test_enrolments_by_address(self, document: dict[str, Any], media_dir: Path, tmp_path: Path) -> None:
        gate = Gate(enrolments_per_minute=2.0)
        client = TestClient(app_with(document, media_dir, tmp_path, gate))
        first = client.post("/api/session", json={}).json()
        assert client.post("/api/session", json={}).status_code == 200
        assert client.post("/api/session", json={}).status_code == 429
        # A resume is not an enrolment.
        assert client.post("/api/session", json={"participant": first["participant"]}).status_code == 200

    def test_public_profile(self, tmp_path: Path) -> None:
        gate = Gate.public(tmp_path / "code")
        assert gate.log_local_only and gate.requires_code
        assert gate.record() == {
            "access_code": "required",
            "log_download": "this machine only",
            "rate_limit": "20/s, burst 60",
        }
