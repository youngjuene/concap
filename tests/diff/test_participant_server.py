"""The whole session through HTTP: entry, phase 1, the handoff, the watch, resume and blindness."""

from __future__ import annotations

import json
import uuid
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from phase1_fixtures import CLIPS, DURATION_MS, approved_setup
from test_live_server import ITEMS, VIDEO, FakeAttempt, drain, job_for, stream

from diff.live_generation import LiveService
from diff.live_server import LivePool, LiveVideo, rating_items
from diff.model_profile import TRIDA_7B
from diff.participant_server import COOKIE, build_participant_app
from diff.phase1_flow import HANDOFF_TEXT, Phase1
from diff.phase1_tweak import check_record, tweak_job_for
from diff.session_store import SessionStore

HEADERS = {"x-study-request": "1"}
ANSWERS: dict[str, dict[str, Any]] = {
    "P3": {"A1s": "s1"},
    "P4": {item: 2 for item in ("F1", "F2", "F3", "BF1", "BF2", "BF3", "E1", "E2", "E3", "P1", "P2")},
    "P5": {"FA1": "unsure", "FA2": 1, "FA3": 4},
    "P8": {"A2s": "s2"},
    "P9": {f"{item}_2": 3 for item in ("F1", "F2", "F3", "BF1", "BF2", "BF3", "E1", "E2", "E3", "P1", "P2")},
    "P10": {"S1": 4, "S2": 3, "S3": 5},
    "P11": {"R1": 5, "R2": ["m1"]},
}


class Lab:
    """The app over one store, rebuildable as after a server restart."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.setup = approved_setup(root)
        self.media = root / "secret-name-viewing.webm"
        self.media.write_bytes(b"\x1a\x45\xdf\xa3fake-webm")
        self.now = [1000.0]
        self.bodies: list[str] = []
        self.start()

    def start(self) -> None:
        self.service = LiveService(
            FakeAttempt(), self.root / "logs", tweak_check=check_record, clock=lambda: 0.0
        )
        self.app = build_participant_app(
            store=SessionStore(self.root / "sessions.sqlite"),
            pool=LivePool([self.service]),
            videos=[LiveVideo(VIDEO, self.media, stream())],
            items=rating_items(ITEMS),
            log_dir=self.root / "logs",
            job_for=job_for,
            phase1=Phase1(self.setup, root=self.root),
            tweak_for=partial(tweak_job_for, purpose="smoke"),
            secret=b"test-secret",
            clock=lambda: self.now[0],
        )

    def client(self) -> TestClient:
        return TestClient(self.app)

    def keep(self, response: Any) -> Any:
        self.bodies.append(response.text)
        self.bodies.extend(f"{k}={v}" for k, v in response.headers.items())
        return response

    def enter(self, client: TestClient, participant: str = "P1") -> Any:
        return self.keep(client.post("/api/enter", json={"participant": participant}, headers=HEADERS))

    def state(self, client: TestClient) -> dict[str, Any]:
        return dict(self.keep(client.get("/api/p1/state")).json())

    def act(self, client: TestClient, detail: dict[str, Any], action: str, **data: Any) -> Any:
        body = {
            "key": str(uuid.uuid4()),
            "revision": detail["revision"],
            "data": {"page": detail["page"], **data},
        }
        return self.keep(client.post(f"/api/p1/{action}", json=body, headers=HEADERS))


def watch_clip(lab: Lab, client: TestClient, detail: dict[str, Any]) -> dict[str, Any]:
    assert lab.keep(client.get(detail["media"]["url"])).status_code == 200
    for sequence, position in enumerate(range(0, DURATION_MS + 1, 1000)):
        lab.now[0] += 1.0
        tick = {"position_ms": position, "playing": True, "hidden": False, "sequence": sequence}
        detail = lab.act(client, detail, "playback", **tick).json()
    return dict(lab.act(client, detail, "view-ended").json())


def through_stage_a(lab: Lab, client: TestClient) -> dict[str, Any]:
    detail = lab.state(client)
    assert detail["page"] == "P0" and detail["title"] == "도입"
    detail = lab.act(client, detail, "intro-complete").json()
    assert detail["page"] == "practice"
    assert lab.keep(client.get(detail["frames"][0]["url"])).status_code == 200
    practice = {
        "practice_rating": 3,
        "practice_sound": "s1",
        "practice_visual": [{"frame_id": "0", "x": 0.5, "y": 0.5}],
    }
    detail = lab.act(client, detail, "practice-complete", answers=practice).json()
    for _ in CLIPS:
        detail = clip_step(lab, client, detail, "V1", ("P3", "P4", "P5"))
    # The stage-B interstitial, with the long videos' count and length from the videos served.
    assert (detail["page"], detail["kind"], detail["step"]) == ("interlude", "interlude", "B")
    assert detail["progress"] == {"segment": 3, "fraction": 0.0}
    assert detail["remaining"] == [
        {"label": f"두 번째 시청 · 영상 {len(CLIPS)}편", "time": None},
        {"label": "긴 영상 1편 · 자막 조절", "time": "약 1분"},
        {"label": "짧은 인터뷰", "time": "2~3분"},
    ]
    detail = lab.act(client, detail, "interlude-complete").json()
    assert (detail["page"], detail["clip_index"]) == ("P6", 0)
    assert detail["progress"] == {"segment": 3, "fraction": 0.0}
    return detail


def clip_step(
    lab: Lab, client: TestClient, detail: dict[str, Any], item: str, pages: tuple[str, ...]
) -> dict[str, Any]:
    detail = watch_clip(lab, client, detail)
    assert lab.keep(client.get(detail["frames"][1]["url"])).status_code == 200
    point = [{"frame_id": "1", "x": 0.4, "y": 0.2}]
    detail = lab.act(client, detail, "submit", answers={item: point}).json()
    for page in pages:
        assert detail["page"] == page
        detail = lab.act(client, detail, "submit", answers=ANSWERS[page]).json()
    return detail


def through_stage_b(lab: Lab, client: TestClient, detail: dict[str, Any]) -> dict[str, Any]:
    for _ in CLIPS:
        assert detail["kind"] == "view", detail["kind"]
        detail = clip_step(lab, client, detail, "V2", ("P8", "P9", "P10", "P11"))
    assert detail["page"] == "handoff" and detail["title"] == "이제 긴 영상 1편을 봅니다"
    assert detail["instruction"] == f"한 편은 약 1분입니다. {HANDOFF_TEXT}"
    return detail


def through_phase_one(lab: Lab, client: TestClient) -> dict[str, Any]:
    detail = through_stage_a(lab, client)
    assert detail["kind"] == "preparing"
    drain(lab.service)  # Trida writes the three second captions
    return through_stage_b(lab, client, lab.state(client))


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    return Lab(tmp_path)


def test_a_whole_session_hands_phase_one_answers_to_the_tuner(lab: Lab) -> None:
    client = lab.client()
    assert lab.enter(client).json() == {"stage": "phase1"}
    assert lab.service.session_id is not None  # the GPU is claimed at entry
    detail = through_phase_one(lab, client)
    assert lab.act(client, detail, "handoff").json() == {"next_url": "watch/"}
    assert lab.state(client) == {"kind": "moved", "next_url": "watch/"}
    started = lab.keep(client.post("/watch/api/session", json={}))
    assert started.json()["stage"] == "ready"
    drain(lab.service)
    assert lab.keep(client.post("/watch/api/start")).json()["stage"] == "watch"
    assert lab.keep(client.post("/watch/api/video-ended")).json()["stage"] == "ratings"
    assert lab.keep(client.post("/watch/api/ratings", json={"answers": {"L1": 4}})).json() == {
        "stage": "done"
    }

    events = [
        json.loads(line)
        for line in next((lab.root / "logs").glob("session-*.jsonl")).read_text().splitlines()
    ]
    calibration = events[0]["calibration"]
    assert events[0]["event"] == "session_started"
    assert calibration["caption_rating_means"] == {"S1": 4.0, "S2": 3.0, "S3": 5.0}
    families = {lab.setup["clips"][i]["options"]["A1s"][0]["family"] for i in range(3)}
    assert set(calibration["heard_observations"]) == families
    attempts = [
        json.loads(line) for line in next((lab.root / "logs").glob("live-*.jsonl")).read_text().splitlines()
    ]
    live = [line for line in attempts if line.get("kind") != "tweak"]
    assert "소리 표현 4" in live[0]["system_message"]
    assert [line["kind"] for line in attempts if line.get("kind") == "tweak"] == ["tweak"] * 3

    store = SessionStore(lab.root / "sessions.sqlite")
    (session,) = store.sessions()
    assert session.participant == "P1" and session.finished_at is not None
    assert len(session.state["responses"]) == 27 and session.state["live"]["answers"] == {"L1": 4}
    seconds = session.state["second_captions"]
    assert {entry["status"] for entry in seconds.values()} == {"ready"}
    assert {entry["text"] for entry in seconds.values()} == {"다시 쓴 두 번째 자막"}
    assert lab.enter(lab.client()).status_code == 409


def test_nothing_a_browser_fetches_names_clips_families_answers_or_the_model(lab: Lab) -> None:
    client = lab.client()
    lab.enter(client)
    detail = through_phase_one(lab, client)
    lab.act(client, detail, "handoff")
    lab.keep(client.post("/watch/api/session", json={}))
    pages = [client.get(path).text for path in ("/", "/phase1_pages.js", "/participant_theme.css", "/watch/")]
    everything = "\n".join(lab.bodies + pages)
    for secret in (
        *CLIPS,
        "mentioned",
        "audioset",
        "orig_phrase",
        "orig_text",
        "a bus idles",
        "trida",
        TRIDA_7B.profile_id,
        "smoke",
        VIDEO,
        "secret-name",
        "slots",
        "fascination",
        "job-",
    ):
        assert secret not in everything, secret
    api = "\n".join(lab.bodies)
    for secret in ("family", "things", "human", "natural", ".png", ".mp4", "data/media"):
        assert secret not in api, secret


def test_a_restart_resumes_phase_one_and_restarts_an_interrupted_video(lab: Lab) -> None:
    client = lab.client()
    lab.enter(client)
    detail = lab.act(client, lab.state(client), "intro-complete").json()
    assert detail["page"] == "practice"
    lab.start()  # the server restarts; its memory and GPU claims are gone
    resumed = lab.client()
    assert lab.enter(resumed).json() == {"stage": "phase1"}
    assert lab.state(resumed)["page"] == "practice"
    assert lab.state(client).get("error")  # the old browser's cookie was voided by the new entry
    detail = lab.act(resumed, lab.state(resumed), "practice-complete", answers={"practice_rating": 3})
    assert detail.status_code == 400


def test_second_captions_pending_at_a_restart_are_queued_again(lab: Lab) -> None:
    client = lab.client()
    lab.enter(client)
    assert through_stage_a(lab, client)["kind"] == "preparing"
    assert len(lab.service.tweaks) == 3
    lab.start()  # the restart drops the queue with the process
    resumed = lab.client()
    assert lab.enter(resumed).json() == {"stage": "phase1"}
    assert len(lab.service.tweaks) == 3  # entry queues the pending ones again
    drain(lab.service)
    assert lab.state(resumed)["kind"] == "view"
    attempts = [
        json.loads(line) for line in next((lab.root / "logs").glob("live-*.jsonl")).read_text().splitlines()
    ]
    assert [line["kind"] for line in attempts] == ["tweak"] * 3
    assert {line["attempt"] for line in attempts} == {1}


def test_a_failed_second_caption_skips_its_step_through_http(lab: Lab) -> None:
    client = lab.client()
    lab.enter(client)
    through_stage_a(lab, client)
    lab.service.attempt = lambda payload, timeout: {"status": "timeout", "error": "30 s"}
    drain(lab.service)
    detail = lab.state(client)
    assert detail["kind"] == "skipped" and "media" not in detail
    for _ in CLIPS:
        detail = lab.act(client, detail, "skip-second").json()
    assert detail["page"] == "handoff"
    (session,) = SessionStore(lab.root / "sessions.sqlite").sessions()
    assert [entry["reason"] for entry in session.state["skipped"]] == ["timeout: 30 s"] * 3
    events = [
        event["event"] for event in SessionStore(lab.root / "sessions.sqlite").events(session.session_id)
    ]
    assert events.count("tweak_failed") == 3


def test_a_watch_interrupted_by_a_restart_restarts_its_video(lab: Lab) -> None:
    client = lab.client()
    lab.enter(client)
    lab.act(client, through_phase_one(lab, client), "handoff")
    client.post("/watch/api/session", json={})
    client.post(
        "/watch/api/settings",
        json={"texture": 1.0, "context": 0.0, "position_ms": 0, "origin": "slider:texture"},
    )
    drain(lab.service)
    assert client.post("/watch/api/start").json()["stage"] == "watch"
    lab.start()
    resumed = lab.client()
    assert lab.enter(resumed).json() == {"next_url": "watch/"}
    view = resumed.post("/watch/api/session", json={}).json()
    assert view["stage"] == "ready"
    assert view["sliders"] == {"texture": 1.0, "context": 0.0, "stops": {"texture": 2, "context": 0}}
    assert view["progress"] == {"segment": 4, "fraction": 0.0}
    log = next((lab.root / "logs").glob("session-*.jsonl")).read_text().splitlines()
    assert json.loads(log[-1]) | {"at": 0} == {
        "schema": "diff.live-event/v1",
        "at": 0,
        "session_id": json.loads(log[0])["session_id"],
        "event": "session_resumed",
        "video_restarted": True,
    }


def test_entry_refusals(lab: Lab) -> None:
    client = lab.client()
    assert client.post("/api/enter", json={"participant": "P1"}).status_code == 401  # no request header
    assert lab.enter(client, "p1").status_code == 400
    assert lab.enter(client, "P1").status_code == 200
    busy = lab.enter(lab.client(), "P2")
    assert busy.status_code == 503 and "잠시 후" in busy.json()["error"]
    assert client.get("/api/p1/state").status_code == 200
    assert lab.client().get("/api/p1/state").status_code == 401


def test_answers_are_applied_once_and_against_the_current_revision(lab: Lab) -> None:
    client = lab.client()
    lab.enter(client)
    detail = lab.state(client)
    body = {"key": "k-intro", "revision": detail["revision"], "data": {"page": "P0"}}
    first = client.post("/api/p1/intro-complete", json=body, headers=HEADERS)
    again = client.post("/api/p1/intro-complete", json=body, headers=HEADERS)
    assert first.json() == again.json() and first.json()["page"] == "practice"
    stale = client.post("/api/p1/intro-complete", json={**body, "key": "k-2"}, headers=HEADERS)
    assert stale.status_code == 409
    assert client.post("/api/p1/generate", json=body, headers=HEADERS).status_code == 404
    assert (
        client.post(
            "/api/p1/handoff", json={**body, "key": "k-3", "revision": 1}, headers=HEADERS
        ).status_code
        == 409
    )


def test_only_the_current_page_s_media_are_served(lab: Lab) -> None:
    client = lab.client()
    lab.enter(client)
    detail = lab.act(client, lab.state(client), "intro-complete").json()
    frame = detail["frames"][0]["url"]
    assert client.get(frame).status_code == 200 and client.get(frame).headers["content-type"] == "image/png"
    practice = {
        "practice_rating": 3,
        "practice_sound": "s1",
        "practice_visual": [{"frame_id": "0", "x": 0.5, "y": 0.5}],
    }
    lab.act(client, detail, "practice-complete", answers=practice)
    assert client.get(frame).status_code == 404
    assert client.get("/api/p1/media/not-a-token").status_code == 404


def test_security_headers(lab: Lab) -> None:
    client = lab.client()
    page = client.get("/")
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert lab.enter(client).headers["cache-control"] == "no-store"
    assert client.cookies.get(COOKIE)
