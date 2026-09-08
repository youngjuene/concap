"""The page-six handoff carries real calibration into a private viewing session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_regen_app import POINTS, answer, hear

from dpo.regen.app import build_app
from dpo.regen.continuation import ViewingConfig
from dpo.regen.regeneration import RegenTemplateWriter
from dpo.regen.tests.test_study import make_media


@pytest.fixture(scope="module")
def viewing_media(tmp_path_factory: Any) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("continuation-media")
    manifest = make_media(root)
    manifest["languages"] = ["en", "ko"]
    for video in manifest["viewing_videos"]:
        for cue in video["cues"]:
            cue["fallback"]["ko"] = "버스 엔진 소리가 울린다."
    return root, manifest


@pytest.fixture
def integrated(document: Any, media_dir: Path, tmp_path: Path, viewing_media: Any) -> Any:
    root, raw = viewing_media
    manifest = tmp_path / "viewing.json"
    manifest.write_text(json.dumps(raw))
    app = build_app(
        document,
        media_dir,
        tmp_path / "legacy",
        RegenTemplateWriter(),
        viewing=ViewingConfig(manifest, root, tmp_path / "viewing"),
    )
    return TestClient(app), app.state.continuation, manifest


def complete(client: TestClient, entry: dict[str, Any], language: str = "en") -> None:
    participant = entry["participant"]
    assert (
        client.post("/api/language", json={"participant": participant, "language": language}).status_code
        == 200
    )
    for step in ("view_prepared", "view_regenerated"):
        assert (
            client.post(
                "/api/viewing",
                json={"participant": participant, "step": step, "started_at": "t0", "ended_at": "t1"},
            ).status_code
            == 200
        )
        assert answer(client, participant, "art" if step == "view_prepared" else "survey").status_code == 200
        if step == "view_prepared":
            assert (
                client.post("/api/visual", json={"participant": participant, "points": POINTS}).status_code
                == 200
            )
            assert hear(client, participant).status_code == 200
            assert client.post("/api/regenerate", json={"participant": participant}).status_code == 200


def handoff(client: TestClient, entry: dict[str, Any]) -> Any:
    return client.post(
        "/api/continuation",
        headers={"x-study-request": "1"},
        json={"participant": entry["participant"], "token": entry["viewing_token"]},
    )


@pytest.mark.parametrize("language", ["en", "ko"])
def test_page_six_freezes_calibration_and_continues_without_recollecting(
    integrated: Any, language: str
) -> None:
    client, continuation, _ = integrated
    entry = client.post("/api/session", json={}).json()
    assert entry["viewing_enabled"] is True
    complete(client, entry, language)
    response = handoff(client, entry)
    assert response.status_code == 200, response.text
    url = response.json()["url"]
    assert url.startswith("/viewing/") and url.endswith("/")
    assert f"Path={url}" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]
    headers = {"x-study-request": "1"}
    current = client.post(url + "api/study/session", headers=headers, json={}).json()
    assert current["stage"] == "ready"
    assert current["language"] == language
    frozen = continuation.store.state(entry["viewing_token"])
    assert frozen["profile"]["sample_count"] == 1
    assert frozen["profile"]["heard_observations"] == {"things": 1}
    assert frozen["profile"]["visual_observations"] == {"Building": 1, "Person": 1}
    assert frozen["profile"]["defaults"] == {"texture": 0.5, "context": 0.5}
    assert frozen["profile"]["preference_source"] == "not_collected"
    assert [len(r["responses"]) for r in frozen["calibration_source"]["responses"]] == [8, 22]
    assert len(frozen["calibration_source"]["viewings"]) == 2
    profile_hash = frozen["profile"]["hash"]
    assert handoff(client, entry).json()["url"] == url
    assert continuation.store.state(entry["viewing_token"])["profile"]["hash"] == profile_hash
    watch = client.post(
        url + "api/study/start-viewing",
        headers=headers,
        json={"key": "start", "revision": current["revision"], "data": {}},
    )
    assert watch.status_code == 200, watch.text
    assert watch.json()["stage"] == "watch"
    assert watch.json()["axes"] == {"texture": 0.5, "context": 0.5}
    assert client.get(url + "study/media/watch/0", headers={"Range": "bytes=0-31"}).status_code == 206
    assert client.get(url + "study.js").status_code == 200
    assert client.get(url + "api/study/state").json()["stage"] == "watch"
    assert handoff(client, entry).json()["url"] == url
    assert continuation.store.state(entry["viewing_token"])["stage"] == "watch"


def test_handoff_refuses_incomplete_and_unowned_calibration(integrated: Any) -> None:
    client, _, _ = integrated
    entry = client.post("/api/session", json={}).json()
    assert handoff(client, entry).status_code == 409
    complete(client, entry)
    assert handoff(client, {**entry, "viewing_token": "wrong"}).status_code == 403
    assert (
        client.post(
            "/api/continuation", json={"participant": entry["participant"], "token": entry["viewing_token"]}
        ).status_code
        == 403
    )
    stranger = TestClient(client.app)
    assert (
        stranger.post(
            "/viewing/unknown/api/study/session", headers={"x-study-request": "1"}, json={}
        ).status_code
        == 401
    )
    resumed = stranger.post("/api/session", json={"participant": entry["participant"]}).json()
    assert "viewing_token" not in resumed


def test_two_participants_keep_independent_path_scoped_sessions(integrated: Any) -> None:
    client, _, _ = integrated
    entries = [client.post("/api/session", json={}).json() for _ in range(2)]
    urls = []
    for entry in entries:
        complete(client, entry)
        urls.append(handoff(client, entry).json()["url"])
    assert urls[0] != urls[1]
    states = [client.get(url + "api/study/state").json() for url in urls]
    assert states[0]["session_id"] != states[1]["session_id"]
    assert all(s["stage"] == "ready" for s in states)


def test_pending_media_preserves_profile_and_never_reports_study_complete(integrated: Any) -> None:
    client, continuation, path = integrated
    entry = client.post("/api/session", json={}).json()
    complete(client, entry)
    url = handoff(client, entry).json()["url"]
    raw = json.loads(path.read_text())
    raw["viewing_videos"][0]["status"] = "pending"
    path.write_text(json.dumps(raw))
    current = client.get(url + "api/study/state").json()
    result = client.post(
        url + "api/study/start-viewing",
        headers={"x-study-request": "1"},
        json={"key": "start", "revision": current["revision"], "data": {}},
    )
    assert result.status_code == 400
    assert continuation.store.state(entry["viewing_token"])["stage"] == "ready"
    assert "saved" in result.json()["error"]


def test_telemetry_cannot_supply_authoritative_calibration(integrated: Any) -> None:
    client, continuation, _ = integrated
    entry = client.post("/api/session", json={}).json()
    # A record from before this upgrade is untrusted even if it has a reserved type.
    continuation.log.append(
        entry["participant"], [{"type": "visual.submitted", "points": [{"x": 0.99}]}], None
    )
    assert (
        client.post(
            "/api/events",
            json={"participant": entry["participant"], "events": [{"type": "visual.submitted"}]},
        ).status_code
        == 400
    )
    complete(client, entry)
    assert handoff(client, entry).status_code == 200
    observed = continuation.store.state(entry["viewing_token"])["observations"][0]
    assert observed["points_source"] == "server_snapshot"
    assert [p["x"] for p in observed["points"]] == [p["x"] for p in POINTS]


def test_preupgrade_participant_uses_researcher_recovery(
    document: Any, media_dir: Path, tmp_path: Path, viewing_media: Any
) -> None:
    from urllib.parse import parse_qs, urlsplit

    old = TestClient(build_app(document, media_dir, tmp_path / "legacy", RegenTemplateWriter()))
    entry = old.post("/api/session", json={}).json()
    complete(old, entry)
    media, raw = viewing_media
    path = tmp_path / "viewing.json"
    path.write_text(json.dumps(raw))
    app = build_app(
        document,
        media_dir,
        tmp_path / "legacy",
        RegenTemplateWriter(),
        viewing=ViewingConfig(path, media, tmp_path / "viewing"),
    )
    remote = TestClient(app)
    headers = {"x-study-request": "1"}
    payload = {"participant": entry["participant"]}
    assert "viewing_token" not in remote.post("/api/session", json=payload).json()
    assert remote.post("/api/continuation/recover", headers=headers, json=payload).status_code == 403
    operator = TestClient(app, client=("127.0.0.1", 1234))
    recovery = operator.post("/api/continuation/recover", headers=headers, json=payload)
    assert recovery.status_code == 200
    supplied = {k: v[0] for k, v in parse_qs(urlsplit(recovery.json()["url"]).fragment).items()}
    resumed = remote.post("/api/session", json=supplied).json()
    assert resumed["participant"] == entry["participant"]
    assert resumed["viewing_token"] == supplied["viewing_token"]
    assert handoff(remote, resumed).status_code == 200
