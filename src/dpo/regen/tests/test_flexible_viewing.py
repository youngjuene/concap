"""Focused regressions for flexible long-video viewing counts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from dpo.regen.study_api import build_study_app
from dpo.regen.study_prepare import stage_viewing_videos
from dpo.regen.study_schema import digest, load_manifest
from dpo.regen.tests.test_study import Session, form_values
from dpo.regen.tests.test_study import assets as _assets
from dpo.regen.tests.test_viewing_prepare import viewing_sources as _viewing_sources

assets = _assets
viewing_sources = _viewing_sources


def _two_video_manifest(source: dict[str, Any]) -> dict[str, Any]:
    manifest = deepcopy(source)
    manifest["viewing_videos"] = manifest["viewing_videos"][:2]
    return manifest


def test_manifest_and_flow_accept_two_long_videos(assets: Any, tmp_path: Path, monkeypatch: Any) -> None:
    root, source = assets
    manifest = _two_video_manifest(source)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_manifest(path, root)
    assert [video["id"] for video in loaded["viewing_videos"]] == ["long-0", "long-1"]

    session = Session(TestClient(build_study_app(path, root, tmp_path / "out")))
    session.calibrate()
    session.submit("start-viewing")
    assert session.state["viewing_total"] == 2

    now = [1000.0]
    monkeypatch.setattr("dpo.regen.study_api.time.time", lambda: now[0])
    for index in range(2):
        video_id = f"long-{index}"
        for sequence, ms in enumerate(range(0, 300001, 5000)):
            now[0] += 5
            session.submit(
                "playback",
                {"video_id": video_id, "position_ms": ms, "sequence": sequence, "playing": True},
            )
        session.submit("video-ended", {"video_id": video_id})
        assert session.state["stage"] == "video-survey"
        answers = form_values(session.state["items"])
        session.submit("draft", {"video_id": video_id, "answers": answers})
        assert session.state["draft"] == {"video_id": video_id, "answers": answers}
        session.submit("video-survey", {"video_id": video_id, "answers": answers})
        if index == 0:
            assert session.state["stage"] == "break"
            session.submit("continue")

    assert session.state["stage"] == "final"


def test_manifest_accepts_one_calibration_and_one_long_video(assets: Any, tmp_path: Path) -> None:
    root, source = assets
    manifest = _two_video_manifest(source)
    manifest["calibration_clips"] = manifest["calibration_clips"][:1]
    manifest["viewing_videos"] = manifest["viewing_videos"][:1]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_manifest(path, root)

    assert len(loaded["calibration_clips"]) == 1
    assert len(loaded["viewing_videos"]) == 1


def test_settings_noop_keeps_revision_jobs_and_logs_scope(assets: Any, tmp_path: Path) -> None:
    root, source = assets
    path = tmp_path / "study.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    app = build_study_app(path, root, tmp_path / "out")
    session = Session(TestClient(app))
    session.calibrate()
    session.submit("start-viewing")

    before = app.state.store.state(session.token)
    jobs_before = app.state.store.captions(session.token, before)
    session.submit(
        "settings",
        {
            "video_id": "long-0",
            "texture": 0.75,
            "context": 0.25,
            "position_hint_ms": 45000,
            "origin": "slider:texture",
        },
    )
    after = app.state.store.state(session.token)

    assert after["settings_revision"] == before["settings_revision"]
    assert app.state.store.captions(session.token, after) == jobs_before
    event = json.loads(app.state.store.export(session.token)["events"][-1]["body"])
    assert event["data"]["settings_revision"] == before["settings_revision"]
    assert event["data"]["phase"] == 2
    assert event["data"]["stage"] == "watch"
    assert event["data"]["flow_version"] == 2
    assert event["data"]["video_id"] == "long-0"
    assert event["data"]["origin"] == "slider:texture"


def test_settings_retry_idempotency_and_newest_revision_wins(assets: Any, tmp_path: Path) -> None:
    root, source = assets
    path = tmp_path / "study.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    app = build_study_app(path, root, tmp_path / "out")
    session = Session(TestClient(app))
    session.calibrate()
    session.submit("start-viewing")

    first = {
        "revision": session.state["revision"],
        "key": "settings-retry",
        "data": {"video_id": "long-0", "texture": 1, "context": 1, "position_hint_ms": 0, "origin": "pad"},
    }
    first_response = session.client.post("/api/study/settings", json=first)
    retry_response = session.client.post("/api/study/settings", json=first)
    assert first_response.status_code == retry_response.status_code == 200
    session.state = retry_response.json()
    assert session.state["settings_revision"] == 1

    session.submit(
        "settings",
        {"video_id": "long-0", "texture": 0, "context": 1, "position_hint_ms": 45000, "origin": "reset"},
    )
    state = app.state.store.state(session.token)

    assert state["axes"] == {"texture": 0.0, "context": 1.0}
    assert state["settings_revision"] == 2
    assert {job["revision"] for job in app.state.store.captions(session.token, state)} == {2}
    exported = app.state.store.export(session.token)
    queued = [job for job in exported["jobs"] if job["state"] == "queued"]
    assert all(job["revision"] == 2 for job in queued)


def test_settings_origin_must_be_specific(assets: Any, tmp_path: Path) -> None:
    root, source = assets
    path = tmp_path / "study.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    session = Session(TestClient(build_study_app(path, root, tmp_path / "out")))
    session.calibrate()
    session.submit("start-viewing")

    session.submit(
        "settings",
        {
            "video_id": "long-0",
            "texture": 1,
            "context": 1,
            "position_hint_ms": 0,
            "origin": "participant",
        },
        expected=400,
    )


def test_legacy_settings_same_value_still_advances_revision(assets: Any, tmp_path: Path) -> None:
    root, source = assets
    path = tmp_path / "study.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    app = build_study_app(path, root, tmp_path / "out")
    session = Session(TestClient(app))
    session.calibrate()
    session.submit("start-viewing")

    def legacy(state: dict[str, Any]) -> None:
        state.pop("protocol_version", None)

    app.state.store.mutate(session.token, "legacy-fixture", session.state["revision"], "fixture", legacy)
    session.state = session.client.get("/api/study/state").json()
    before = app.state.store.state(session.token)
    session.submit(
        "settings",
        {"video_id": "long-0", "texture": 0.75, "context": 0.25, "position_hint_ms": 45000},
    )

    assert app.state.store.state(session.token)["settings_revision"] == before["settings_revision"] + 1


def test_stage_viewing_accepts_manifest_count(viewing_sources: Any) -> None:
    root, sources, manifest, durations = viewing_sources
    manifest = _two_video_manifest(manifest)
    cues = {
        f"viewing-{index + 1}": [
            {
                "start_ms": start,
                "end_ms": min(start + 5000, durations[index]),
                "evidence": f"Operator evidence for video {index + 1}.",
                "fallback": {"en": f"Caption for video {index + 1} at {start} ms."},
            }
            for start in range(0, durations[index], 5000)
        ]
        for index in range(2)
    }

    prepared = stage_viewing_videos(
        manifest,
        root,
        sources[:2],
        cues,
        provenance="operator-authored-flexible-count-test",
    )

    assert [video["id"] for video in prepared["viewing_videos"]] == ["viewing-1", "viewing-2"]
    assert digest(prepared["viewing_videos"]) != digest(manifest["viewing_videos"])
