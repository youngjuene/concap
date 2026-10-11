"""The live phase-2 app: the page flow, capacity, ratings, the event log and blindness (§6.5)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from diff.caption_sources import CaptionSource, parse_sources
from diff.job_plan import Job, live_job
from diff.live_calibration import calibrate
from diff.live_generation import LiveService, LiveStream, Spec
from diff.live_server import LivePool, LiveVideo, build_live_app, rating_items
from diff.model_profile import DREAM_7B

VIDEO = "live_test_video"
ITEMS = {
    "items": [
        {
            "id": "L1",
            "text": "자막이 소리를 잘 나타냈습니까?",
            "min": 0,
            "max": 6,
            "anchors": ["전혀", "완전히"],
        }
    ]
}


def stream(cues: int = 3) -> LiveStream:
    document = {
        "schema": "diff.caption-sources/v1",
        "sources": [
            {
                "schema": "diff.caption-source/v1",
                "phase": 2,
                "unit": {"kind": "video", "key": VIDEO},
                "cue_index": index,
                "start_ms": index * 5000,
                "end_ms": (index + 1) * 5000,
                "text": f"차가 지나간다 {index}.",
                "evidence": {"events": []},
                "provenance": {
                    "text": "synthetic",
                    "evidence": "synthetic",
                    "reviewed_by": [],
                    "reviewed_at": None,
                    "origin": "test",
                },
                "status": "draft",
                "material": "synthetic",
            }
            for index in range(cues)
        ],
    }
    return LiveStream.from_sources(parse_sources(document).sources, VIDEO)


def job_for(source: CaptionSource, spec: Spec) -> Job:
    return live_job(source, spec.texture, spec.context, spec.guidance, "smoke")


class FakeAttempt:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append(payload["job_id"])
        cue = payload["cue_index"]
        return {
            "status": "completed",
            "output": {"text": f"다시 쓴 자막 {cue}번"},
            "latency_ms": 8000,
            "seed": 7,
        }


@pytest.fixture
def app_parts(tmp_path: Path) -> dict[str, Any]:
    media = tmp_path / "secret-name-viewing.webm"
    media.write_bytes(b"\x1a\x45\xdf\xa3fake-webm")
    now = [0.0]
    service = LiveService(FakeAttempt(), tmp_path / "logs", clock=lambda: now[0])
    observations = [{"heard": {"human": True}, "labels": ["person"]}]
    app = build_live_app(
        [LiveVideo(VIDEO, media, stream())],
        rating_items(ITEMS),
        LivePool([service]),
        tmp_path / "logs",
        job_for=job_for,
        calibrate=lambda code: calibrate(observations),
        secret=b"test-secret",
    )
    return {"app": app, "service": service, "logs": tmp_path / "logs", "media": media}


def drain(service: LiveService) -> None:
    while service.step():
        pass


def test_a_whole_watch_without_leaking_internals(app_parts: dict[str, Any]) -> None:
    client, service = TestClient(app_parts["app"]), app_parts["service"]
    bodies: list[str] = []

    def keep(response: Any) -> Any:
        bodies.append(response.text)
        bodies.extend(f"{k}={v}" for k, v in response.headers.items())
        return response

    started = keep(client.post("/api/session", json={}))
    assert started.status_code == 200 and started.json()["stage"] == "ready"
    assert keep(client.post("/api/start")).status_code == 409  # cue 0 is not written yet
    drain(service)
    ready = keep(client.get("/api/captions")).json()["captions"]
    assert sorted(ready) == ["0", "1", "2"] and ready["0"]["text"] == "다시 쓴 자막 0번"
    media_url = started.json()["video"]["media"]
    assert keep(client.get("/" + media_url)).status_code == 200
    assert client.get("/media/wrong-token").status_code == 404
    assert keep(client.post("/api/start")).json()["stage"] == "watch"
    assert keep(
        client.post(
            "/api/settings",
            json={"texture": 1.0, "context": 1.0, "position_ms": 0, "origin": "slider:texture"},
        )
    ).json() == {"ok": True}
    assert keep(client.post("/api/playback", json={"position_ms": 1000, "playing": True})).status_code == 200
    drain(service)
    token = ready["0"]["token"]
    keep(
        client.post(
            "/api/exposures", json={"entries": [{"cue": 0, "token": token, "from_ms": 0, "to_ms": 5000}]}
        )
    )
    assert keep(client.post("/api/video-ended")).json()["stage"] == "ratings"
    assert keep(client.get("/api/items")).json()["items"][0]["id"] == "L1"
    assert keep(client.post("/api/ratings", json={"answers": {"L1": 9}})).status_code == 422
    assert keep(client.post("/api/ratings", json={"answers": {"L1": 5}})).json() == {"stage": "done"}

    everything = "\n".join(bodies)
    for secret in (
        "job-",
        "dream",
        DREAM_7B.profile_id,
        "seed",
        "k2",
        "smoke",
        VIDEO,
        "secret-name",
        "relation",
        "broad",
    ):
        assert secret not in everything, secret

    session_log = next(app_parts["logs"].glob("session-*.jsonl"))
    events = [json.loads(line) for line in session_log.read_text(encoding="utf-8").splitlines()]
    kinds = [event["event"] for event in events if event["event"] != "playback"]
    assert kinds == ["session_started", "watch_started", "settings", "exposures", "video_ended", "ratings"]
    assert events[0]["calibration"]["heard_observations"] == {"human": 1}
    assert next(e for e in events if e["event"] == "exposures")["entries"][0]["job_id"].startswith("job-")
    assert next(e for e in events if e["event"] == "settings")["steps"] == {
        "texture": 4,
        "context": "relation",
    }
    live_log = next(app_parts["logs"].glob("live-*.jsonl"))
    attempts = [json.loads(line) for line in live_log.read_text(encoding="utf-8").splitlines()]
    assert attempts[0]["guidance"].startswith("참고로") and attempts[-1]["spec"] == {
        "texture": 4,
        "context": "relation",
    }


def test_a_second_participant_is_refused_while_the_gpu_is_taken(app_parts: dict[str, Any]) -> None:
    first, second = TestClient(app_parts["app"]), TestClient(app_parts["app"])
    assert first.post("/api/session", json={}).status_code == 200
    busy = second.post("/api/session", json={})
    assert busy.status_code == 503 and busy.json()["stage"] == "busy" and "잠시 후" in busy.json()["message"]


def test_requests_without_a_session_are_refused(app_parts: dict[str, Any]) -> None:
    client = TestClient(app_parts["app"])
    assert client.get("/api/captions").status_code == 401
    assert client.post("/api/settings", json={}).status_code == 401


def test_bad_settings_are_refused(app_parts: dict[str, Any]) -> None:
    client = TestClient(app_parts["app"])
    client.post("/api/session", json={})
    for body in (
        {"texture": 2, "context": 0.5, "position_ms": 0, "origin": "slider:texture"},
        {"texture": 0.5, "context": 0.5, "position_ms": 0, "origin": "keyboard"},
        {"texture": 0.5},
    ):
        assert client.post("/api/settings", json=body).status_code == 422


def test_rating_items_are_validated() -> None:
    assert rating_items(ITEMS)[0].scale_max == 6
    for bad in (
        {"items": []},
        {"items": [{"id": "x", "text": "t", "min": 3, "max": 3, "anchors": ["a", "b"]}]},
    ):
        with pytest.raises(ValueError):
            rating_items(bad)
