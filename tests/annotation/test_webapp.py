"""Collection web-app endpoint tests over the in-process FastAPI app."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from dpo.annotation.collection_tasks import RESPONSES_SCHEMA, build_collection_tasks
from dpo.annotation.webapp import build_app
from dpo.contracts.study_contract import CHOICES, REASON_TAGS, TIE_SUBTYPES
from tests.conftest import PreferenceWorld


def _client(world: PreferenceWorld, tmp_path: Path) -> tuple[TestClient, dict[str, object], Path]:
    tasks_document, _ = build_collection_tasks(world.contract, world.pool, audio_presentations={})
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    out_path = tmp_path / "responses"
    app = build_app(tasks_document, media_dir, out_path)
    return TestClient(app), tasks_document, out_path


def test_page_meta_and_tasks_roundtrip(world: PreferenceWorld, tmp_path: Path) -> None:
    client, tasks_document, _ = _client(world, tmp_path)
    page = client.get("/")
    assert page.status_code == 200
    assert "caption" in page.text.lower()
    meta = client.get("/api/meta")
    assert meta.status_code == 200
    assert meta.json() == {
        "choices": list(CHOICES),
        "tie_subtypes": list(TIE_SUBTYPES),
        "reason_tags": list(REASON_TAGS),
    }
    served = client.get("/api/tasks")
    assert served.status_code == 200
    assert served.json() == tasks_document


def test_media_serves_by_presentation_and_fails_closed(world: PreferenceWorld, tmp_path: Path) -> None:
    client, tasks_document, _ = _client(world, tmp_path)
    tasks = tasks_document["tasks"]
    assert isinstance(tasks, list)
    clip_id = str(tasks[0]["clip_id"])
    video = tmp_path / "media" / f"{clip_id}.mp4"
    video.write_bytes(b"\x00\x00fake-video")
    audio = tmp_path / "media" / f"{clip_id}.wav"
    audio.write_bytes(b"RIFFfake-audio")
    served = client.get(f"/media/{clip_id}", params={"presentation": "muted_video"})
    assert served.status_code == 200
    assert served.content == video.read_bytes()
    served = client.get(f"/media/{clip_id}", params={"presentation": "audio_only"})
    assert served.status_code == 200
    assert served.content == audio.read_bytes()
    # Traversal-shaped clip ids are rejected before any filesystem access:
    # encoded slashes never reach the handler, and dotted ids fail the
    # [A-Za-z0-9_-]+ screen inside it.
    assert client.get("/media/..%2Fsecret", params={"presentation": "muted_video"}).status_code == 404
    denied = client.get("/media/....", params={"presentation": "muted_video"})
    assert denied.status_code == 400
    assert "invalid clip id" in denied.json()["error"]
    # Absent media is a clear JSON 404, not a server error.
    missing = client.get("/media/absent-clip", params={"presentation": "audio_only"})
    assert missing.status_code == 404
    assert "absent-clip" in missing.json()["error"]


def test_responses_post_writes_the_document_and_fails_closed(world: PreferenceWorld, tmp_path: Path) -> None:
    client, tasks_document, out_path = _client(world, tmp_path)
    tasks = tasks_document["tasks"]
    assert isinstance(tasks, list)
    document = {
        "schema": RESPONSES_SCHEMA,
        "annotator": "Jamie Q. Annotator",
        "responses": [
            {
                "task_id": str(task["task_id"]),
                "choice": "a_better",
                "tie_subtype": None,
                "preference_strength": 4,
                "confidence": 4,
                "reason_tags": ["coverage"],
                "response_time_ms": 4000,
                "replay_count": 0,
            }
            for task in tasks[:3]
        ],
    }
    reply = client.post("/api/responses", json=document)
    assert reply.status_code == 200
    saved = reply.json()
    assert saved["responses"] == 3
    destination = Path(str(saved["saved"]))
    assert destination.parent == out_path
    assert destination.name == "responses-jamie-q-annotator.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == document
    # Unknown task ids, wrong schemas, and anonymous saves are rejected.
    unknown = {
        **document,
        "responses": [{**document["responses"][0], "task_id": "task-does-not-exist"}],
    }
    assert client.post("/api/responses", json=unknown).status_code == 400
    assert client.post("/api/responses", json={**document, "schema": "other/v1"}).status_code == 400
    assert client.post("/api/responses", json={**document, "annotator": "  "}).status_code == 400
