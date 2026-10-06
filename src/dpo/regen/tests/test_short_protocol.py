from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpo.regen.tests.test_survey_hierarchy import _client


def test_new_browser_requires_played_coverage(tmp_path: Path, monkeypatch: Any) -> None:
    client, out, _ = _client(tmp_path)
    session = client.post("/api/session", json={"client_protocol": 3}).json()
    person = session["participant"]
    payload = {
        "participant": person,
        "clip_index": 0,
        "step": "view_prepared",
        "started_at": "2026-09-22T01:00:00Z",
        "ended_at": "2026-09-22T01:00:10Z",
    }
    assert client.post("/api/viewing", json=payload).status_code == 409
    detail = client.get(f"/api/step/view_prepared?participant={person}").json()
    assert detail["playback"]["coverage"] == []
    clock = [1000.0]
    monkeypatch.setattr("dpo.regen.study_api.time.time", lambda: clock[0])
    for seq, position in enumerate(range(0, detail["duration_ms"] + 1, 1000)):
        clock[0] = 1000.0 + position / 1000
        response = client.post(
            "/api/playback",
            json={
                "participant": person,
                "clip_index": 0,
                "step": "view_prepared",
                "context_id": detail["event_context"]["context_id"],
                "sequence": seq,
                "position_ms": position,
                "playing": True,
                "hidden": False,
            },
        )
        assert response.status_code == 200, response.text
    assert client.post("/api/viewing", json=payload).status_code == 200
    viewing = json.loads((out / f"viewings-{person}.jsonl").read_text().splitlines()[0])
    assert viewing["coverage"]


def test_delayed_events_keep_original_scope_and_deduplicate(tmp_path: Path) -> None:
    client, out, _ = _client(tmp_path)
    session = client.post("/api/session", json={"client_protocol": 3}).json()
    person = session["participant"]
    event = {
        **session["event_context"],
        "event_id": "event-1",
        "type": "caption.shown",
        "at": "2026-09-22T01:00:00Z",
        "text": "Traffic.",
    }
    payload = {"participant": person, "events": [event]}
    assert client.post("/api/events", json=payload).json()["acknowledged"] == ["event-1"]
    assert client.post("/api/events", json=payload).json()["acknowledged"] == ["event-1"]
    rows = [json.loads(row) for row in (out / f"events-{person}.jsonl").read_text().splitlines()]
    assert sum(row.get("event_id") == "event-1" for row in rows) == 1
    event["clip_index"] = 1
    assert client.post("/api/events", json=payload).status_code == 400


def test_new_session_rejects_changed_stimulus_configuration(tmp_path: Path) -> None:
    client, out, _ = _client(tmp_path)
    session = client.post("/api/session", json={"client_protocol": 3}).json()
    path = out / f"snapshot-{session['participant']}.json"
    snapshot = json.loads(path.read_text())
    snapshot["document_hash"] = "changed"
    path.write_text(json.dumps(snapshot))
    response = client.get(f"/api/step/view_prepared?participant={session['participant']}")
    assert response.status_code == 409


@pytest.mark.parametrize("count", [1, 4])
def test_frozen_short_order_can_address_all_clips(tmp_path: Path, count: int) -> None:
    from fastapi.testclient import TestClient
    from PIL import Image

    from dpo.regen.app import build_app
    from dpo.regen.config import Calibration, Configuration
    from dpo.regen.regeneration import RegenTemplateWriter
    from dpo.regen.tests.test_qa_validation import _segment

    ids = [f"clip{i}" for i in range(count)]
    doc: dict[str, Any] = {
        "schema": "dpo.caption-regen/v4",
        "session_id": "flexible",
        "clip_order": ids,
        "config": Configuration(
            study_id="flex", corpus_id="clips", calibration=Calibration(cue_slots=4, minimum_points=1)
        ).artifact(),
        "segments": {key: _segment(key) for key in ids},
    }
    media = tmp_path / "media"
    for segment in doc["segments"].values():
        for frame in segment["frames"]:
            path = media / frame["objects"][0]["mask"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), 255).save(path)
    client = TestClient(build_app(doc, media, tmp_path / "out", RegenTemplateWriter(), derive=False))
    session = client.post("/api/session", json={}).json()
    assert session["clip_count"] == count
    assert session["clip_order"] == ids
    assert (
        client.get(f"/api/step/view_prepared?participant={session['participant']}").json()["segment"]
        == ids[0]
    )


def test_historical_session_cannot_be_reinterpreted_as_v4(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from dpo.regen.app import build_app
    from dpo.regen.config import Calibration, Configuration
    from dpo.regen.regeneration import RegenTemplateWriter
    from dpo.regen.tests.test_qa_validation import _segment

    client, out, _ = _client(tmp_path)
    session = client.post("/api/session", json={}).json()
    document = {
        "schema": "dpo.caption-regen/v4",
        "session_id": "hierarchy",
        "clip_order": ["A", "B"],
        "config": Configuration(
            study_id="hierarchy",
            corpus_id="short-clips",
            calibration=Calibration(cue_slots=4, minimum_points=1),
        ).artifact(),
        "segments": {"A": _segment("A"), "B": _segment("B")},
    }
    changed = TestClient(build_app(document, tmp_path / "media", out, RegenTemplateWriter(), derive=False))
    assert changed.post("/api/session", json={"participant": session["participant"]}).status_code == 409
