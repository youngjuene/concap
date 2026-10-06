"""Persistent-cookie jar restart and renewal over the real questionnaire routes."""

from __future__ import annotations

import json
from http.cookiejar import MozillaCookieJar
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.routing import APIRoute
from starlette.requests import Request

from dpo.regen.tests.test_survey_hierarchy import _client

LIFETIME = 30 * 24 * 60 * 60


class BrowserJar:
    """Simulate cookie persistence without an ASGI test portal or network listener."""

    def __init__(self, app: Any, path: Path, *, scheme: str = "http", restore: bool = False) -> None:
        self.app, self.path, self.scheme = app, path, scheme
        self.jar = MozillaCookieJar(str(path))
        if restore:
            self.jar.load(ignore_discard=False, ignore_expires=False)
        self.cookies = httpx.Cookies(self.jar)

    def call(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        address: str = "127.0.0.1",
    ) -> tuple[dict[str, Any], Any]:
        outgoing = httpx.Request(
            method, f"{self.scheme}://testserver{path}", headers={"x-study-request": "1"}
        )
        self.cookies.set_cookie_header(outgoing)
        request = Request(
            {
                "type": "http",
                "scheme": self.scheme,
                "method": method,
                "path": path,
                "headers": [(key.lower(), value) for key, value in outgoing.headers.raw],
                "client": (address, 41111),
                "server": ("testserver", 80),
            }
        )
        exact = next(
            (
                route
                for route in self.app.routes
                if isinstance(route, APIRoute) and route.path == path and method in (route.methods or set())
            ),
            None,
        )
        if exact:
            kwargs: dict[str, Any] = {"request": request}
            if method == "POST":
                kwargs["payload"] = payload or {}
            response = exact.endpoint(**kwargs)
        else:
            route = next(
                route
                for route in self.app.routes
                if isinstance(route, APIRoute) and route.path == "/api/questionnaire/{action}"
            )
            response = route.endpoint(request=request, action=path.rsplit("/", 1)[-1], payload=payload or {})
        incoming = httpx.Response(
            response.status_code, headers=response.raw_headers, content=response.body, request=outgoing
        )
        self.cookies.extract_cookies(incoming)
        return json.loads(response.body), incoming

    def restart(self) -> BrowserJar:
        self.jar.save(ignore_discard=False, ignore_expires=False)
        return BrowserJar(self.app, self.path, scheme=self.scheme, restore=True)


def cookies_from(response: httpx.Response) -> SimpleCookie:
    result = SimpleCookie()
    for header in response.headers.get_list("set-cookie"):
        result.load(header)
    return result


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_questionnaire_cookie_has_30_day_expiry_and_security_attributes(tmp_path: Path, scheme: str) -> None:
    _, _, app = _client(tmp_path)
    browser = BrowserJar(app, tmp_path / "cookies.txt", scheme=scheme)
    _, response = browser.call("/api/questionnaire/session", method="POST")
    assert response.status_code == 200
    cookie = cookies_from(response)[app.state.sheet_questionnaire.cookie]
    assert cookie["max-age"] == str(LIFETIME)
    persisted = next(cookie for cookie in browser.jar if cookie.name == app.state.sheet_questionnaire.cookie)
    assert persisted.expires is not None and persisted.discard is False
    assert cookie["path"] == "/"
    assert cookie["httponly"]
    assert cookie["samesite"] == "strict"
    assert bool(cookie["secure"]) is (scheme == "https")


def test_restart_retains_same_participant_and_valid_reads_renew_cookie(tmp_path: Path) -> None:
    _, _, app = _client(tmp_path)
    browser = BrowserJar(app, tmp_path / "cookies.txt")
    original, _ = browser.call("/api/questionnaire/session", method="POST")
    protocol = app.state.sheet_questionnaire
    resumed = browser.restart()
    state, response = resumed.call("/api/questionnaire/session", method="POST")
    assert response.status_code == 200
    assert state["session_id"] == original["session_id"]
    for path in ("/api/questionnaire/state", "/api/questionnaire/export"):
        body, response = resumed.call(path)
        assert response.status_code == 200
        assert cookies_from(response)[protocol.cookie]["max-age"] == str(LIFETIME)
        assert (body.get("state", body))["session_id"] == original["session_id"]
    with protocol.store.connection() as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1


def test_handoff_keeps_both_persistent_cookies_and_exports_after_restart(tmp_path: Path) -> None:
    _, _, app = _client(tmp_path)
    browser = BrowserJar(app, tmp_path / "cookies.txt")
    original, _ = browser.call("/api/questionnaire/session", method="POST")
    protocol = app.state.sheet_questionnaire
    token = next(cookie.value for cookie in browser.jar if cookie.name == protocol.cookie)
    continuation_cookie = "persistent_viewing_cookie"
    protocol.continuation = SimpleNamespace(
        store=protocol.store, app=SimpleNamespace(state=SimpleNamespace(cookie_name=continuation_cookie))
    )
    protocol.store.mutate(
        token,
        "handoff-fixture",
        original["revision"],
        "fixture",
        lambda state: state.update(stage="ready", sheet_page="handoff"),
    )
    handed, response = browser.call("/api/questionnaire/handoff", method="POST", payload={})
    assert response.status_code == 200
    issued = cookies_from(response)
    assert {protocol.cookie, continuation_cookie} <= set(issued)
    for name in (protocol.cookie, continuation_cookie):
        assert issued[name]["max-age"] == str(LIFETIME)
        assert issued[name].value == token
        assert issued[name]["httponly"] and issued[name]["samesite"] == "strict"
    assert issued[continuation_cookie]["path"] == handed["next_url"]
    assert response.headers["cache-control"] == "no-store"
    restarted = browser.restart()
    assert {cookie.name for cookie in restarted.jar} >= {protocol.cookie, continuation_cookie}
    exported, response = restarted.call("/api/questionnaire/export")
    assert response.status_code == 200
    assert exported["state"]["session_id"] == original["session_id"]
    assert exported["state"]["stage"] == "ready"
    with protocol.store.connection() as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1


def test_invalid_credentials_are_never_renewed_or_reenrolled(tmp_path: Path) -> None:
    _, _, app = _client(tmp_path)
    browser = BrowserJar(app, tmp_path / "cookies.txt")
    browser.call("/api/questionnaire/session", method="POST")
    protocol = app.state.sheet_questionnaire
    browser.cookies.clear()
    browser.cookies.set(protocol.cookie, "unrecognized-credential", domain="testserver.local", path="/")
    for path, method in (
        ("/api/questionnaire/state", "GET"),
        ("/api/questionnaire/session", "POST"),
        ("/api/questionnaire/export", "GET"),
    ):
        _, response = browser.call(path, method=method)
        assert response.status_code == 401
        assert not response.headers.get_list("set-cookie")
    with protocol.store.connection() as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1


def test_persistent_cookie_does_not_relax_local_export_restriction(tmp_path: Path) -> None:
    _, _, app = _client(tmp_path)
    browser = BrowserJar(app, tmp_path / "cookies.txt")
    original, _ = browser.call("/api/questionnaire/session", method="POST")
    restarted = browser.restart()
    _, denied = restarted.call("/api/questionnaire/export", address="203.0.113.7")
    assert denied.status_code == 401
    assert not denied.headers.get_list("set-cookie")
    exported, allowed = restarted.call("/api/questionnaire/export")
    assert allowed.status_code == 200
    assert exported["state"]["session_id"] == original["session_id"]
