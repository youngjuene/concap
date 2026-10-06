"""Recovery must preserve frozen responses and let unconsented setup be repaired."""

from __future__ import annotations

from copy import deepcopy
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from tests.regen.test_sheet_protocol import setup, study_config, submit


def test_completed_handoff_reissues_continuation_cookie_after_refresh(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    app, client, state = setup(document, media_dir, tmp_path)
    protocol = app.state.sheet_questionnaire
    token = client.cookies[protocol.cookie]
    cookie_name = "sheet_recovery_viewing"
    protocol.continuation = SimpleNamespace(
        store=protocol.store,
        app=SimpleNamespace(state=SimpleNamespace(cookie_name=cookie_name)),
    )
    frozen_profile = {"hash": "already-frozen-profile", "defaults": {"texture": 0.5, "context": 0.5}}

    def completed(saved: dict[str, Any]) -> None:
        saved.update(stage="ready", sheet_page="handoff", profile=deepcopy(frozen_profile))

    protocol.store.mutate(token, "completed-handoff-fixture", state["revision"], "fixture", completed)
    # Reloading the original URL knows only the questionnaire cookie. The
    # handoff reply (including the viewing cookie) may previously have been lost.
    assert cookie_name not in client.cookies
    refreshed = client.get("/api/questionnaire/state")
    assert refreshed.status_code == 200
    target = f"/viewing/{state['session_id']}/"
    for attempt in range(2):
        current = client.get("/api/questionnaire/state").json()
        response = client.post(
            "/api/questionnaire/handoff",
            json={
                "key": f"fresh-key-after-reload-{attempt}",
                "revision": current["revision"],
                "data": {"page": "handoff"},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["next_url"] == target
        cookies = SimpleCookie()
        cookies.load(response.headers["set-cookie"])
        assert cookies[cookie_name].value == token
        assert cookies[cookie_name]["path"] == target
        assert cookies[cookie_name]["httponly"]
        assert cookies[cookie_name]["samesite"] == "strict"
        saved = protocol.store.state(token)
        assert saved["stage"] == "ready"
        assert saved["profile"] == frozen_profile
        assert saved["session_id"] == state["session_id"]
        assert saved["sheet_responses"] == []

    stranger = TestClient(app, client=("127.0.0.1", 40001))
    stranger.headers["x-study-request"] = "1"
    denied = stranger.post(
        "/api/questionnaire/handoff",
        json={"key": "unauthenticated-resume", "revision": saved["revision"], "data": {"page": "handoff"}},
    )
    assert denied.status_code == 401
    assert "set-cookie" not in denied.headers


def test_preconsent_configuration_refresh_updates_snapshot_and_revision_once(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    app, client, state = setup(document, media_dir, tmp_path, configured=False)
    protocol = app.state.sheet_questionnaire
    token = client.cookies[protocol.cookie]
    assert state["setup_missing"]
    initial = protocol.store.state(token)
    configured = study_config(document)
    protocol.config = deepcopy(configured)

    refreshed = protocol.checked(token)

    assert refreshed["revision"] == initial["revision"] + 1
    assert refreshed["sheet_config"] == configured
    assert refreshed["poststudy"]["debrief"]["text"] == configured["debrief_text"]
    assert refreshed["sheet_page"] == "P0"
    assert "sheet_consent" not in refreshed
    assert protocol.setup_missing(refreshed) == []
    assert protocol.checked(token)["revision"] == refreshed["revision"]
    reloaded = client.get("/api/questionnaire/state")
    assert reloaded.status_code == 200
    assert reloaded.json()["revision"] == refreshed["revision"]
    assert reloaded.json()["consent_text"] == configured["consent_text"]
    assert reloaded.json()["setup_missing"] == []


def test_consented_configuration_stays_frozen_when_operator_config_changes(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    app, client, state = setup(document, media_dir, tmp_path)
    protocol = app.state.sheet_questionnaire
    token = client.cookies[protocol.cookie]
    submit(client, state, "intro-complete", consented=True)
    consented = protocol.store.state(token)
    changed = study_config(document)
    changed["consent_text"] = "다음 참가자를 위한 다른 동의서"
    changed["debrief_text"] = "다음 참가자를 위한 다른 디브리핑"
    first_clip = next(iter(changed["clips"]))
    changed["clips"][first_clip]["options"]["A1s"][0]["label"] = "변경된 선택지"
    protocol.config = changed

    resumed = protocol.checked(token)

    assert resumed["revision"] == consented["revision"]
    assert resumed["sheet_config"] == consented["sheet_config"]
    assert resumed["sheet_consent"] == consented["sheet_consent"]
    assert resumed["poststudy"] == consented["poststudy"]
    assert client.get("/api/questionnaire/state").status_code == 200
    assert protocol.store.state(token)["revision"] == consented["revision"]
