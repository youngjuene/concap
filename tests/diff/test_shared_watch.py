"""A watch served by two servers over one store, as on Vercel: nothing restarts, the same captions
are served everywhere, none is written twice, and the time watched is counted once."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_live_server import ITEMS, enter, job_for, stream

from diff.live_generation import STALE_PLAY_S, LiveService
from diff.live_server import LivePool, LiveVideo, rating_items
from diff.participant_server import build_participant_app
from diff.session_store import SessionStore

VIDEOS = ("secret-a", "secret-b")


class Counting:
    """A writer that counts the calls it gets for each job."""

    def __init__(self, name: str) -> None:
        self.name, self.calls = name, []

    def __call__(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append(payload["job_id"])
        return {
            "status": "completed",
            "output": {"text": f"{self.name}가 쓴 자막 {len(self.calls)}번"},
            "latency_ms": 500,
        }


class Server:
    """One instance: its own writer, pool and memory, over the store and logs every instance shares."""

    def __init__(self, name: str, root: Path, wall: list[float], **pool: Any) -> None:
        ticks = itertools.count()
        self.writer = Counting(name)
        # A local clock that moves, so the service re-reads the shared record on each step.
        self.service = LiveService(self.writer, root / "logs", clock=lambda: next(ticks) * 0.3)
        self.pool = LivePool([self.service], **pool)
        self.app = build_participant_app(
            store=SessionStore(root / "sessions.sqlite", clock=lambda: wall[0]),
            pool=self.pool,
            videos=[LiveVideo(video, root / f"{video}.webm", stream(6, video)) for video in VIDEOS],
            items=rating_items(ITEMS),
            log_dir=root / "logs",
            job_for=job_for,
            phase1=None,
            secret=b"one-secret-for-every-instance",
            clock=lambda: wall[0],
        )
        self.client = TestClient(self.app)

    def drain(self) -> None:
        for service in self.pool.services:
            while service.step():
                pass


@pytest.fixture
def two(tmp_path: Path) -> dict[str, Any]:
    for video in VIDEOS:
        (tmp_path / f"{video}.webm").write_bytes(video.encode())
    wall = [1000.0]
    a, b = Server("가", tmp_path, wall), Server("나", tmp_path, wall)
    assert enter(a.client).status_code == 200
    b.client.cookies.update(dict(a.client.cookies))  # one browser, whichever server answers
    return {
        "a": a,
        "b": b,
        "wall": wall,
        "root": tmp_path,
        "store": SessionStore(tmp_path / "sessions.sqlite"),
    }


def events(root: Path) -> list[dict[str, Any]]:
    log = next((root / "logs").glob("session-*.jsonl")).read_text().splitlines()
    return [json.loads(line) for line in log]


def test_a_second_server_serves_the_same_watch_without_restarting_it(two: dict[str, Any]) -> None:
    a, b, wall = two["a"], two["b"], two["wall"]
    assert a.client.post("/watch/api/session", json={}).json()["stage"] == "ready"
    a.drain()
    first = a.client.get("/watch/api/captions").json()["captions"]
    assert first["0"]["current"] and first["0"]["text"].startswith("가가 쓴")
    assert a.client.post("/watch/api/start").json()["stage"] == "watch"
    a.client.post("/watch/api/playback", json={"position_ms": 4000, "playing": True})
    wall[0] += 1
    # The next requests reach a server that has never seen this session.
    assert b.client.get("/watch/api/state").json()["stage"] == "watch"
    served = b.client.get("/watch/api/captions").json()["captions"]
    assert served["0"] == first["0"]  # the same text under the same token
    b.drain()
    assert set(b.writer.calls).isdisjoint(a.writer.calls)  # nothing written twice
    assert b.service.scheduler.position_ms == 4000 and b.service.scheduler.playing
    names = [event["event"] for event in events(two["root"])]
    assert names.count("session_started") == 1 and "server_joined" in names
    assert "video_restarted" not in names
    assert two["store"].sessions()[0].state["live"]["stage"] == "watch"


def test_a_slider_move_on_one_server_steers_the_other_s_writer(two: dict[str, Any]) -> None:
    a, b, wall = two["a"], two["b"], two["wall"]
    a.client.post("/watch/api/session", json={})
    a.drain()
    a.client.post("/watch/api/start")
    b.client.get("/watch/api/state")
    b.drain()
    wall[0] += 1
    move = {"texture": 1.0, "context": 0.0, "position_ms": 1000, "origin": "slider:texture"}
    assert a.client.post("/watch/api/settings", json=move).status_code == 200
    before = len(b.writer.calls)
    b.drain()  # no request reached b: its writer follows the shared record
    assert b.service.scheduler.spec.texture == 2 and b.service.scheduler.spec.context == "broad"
    assert len(b.writer.calls) > before
    a.drain()
    assert set(b.writer.calls).isdisjoint(a.writer.calls)
    assert a.client.get("/watch/api/captions").json() == b.client.get("/watch/api/captions").json()
    # A move b took is in the store, so a's copy of the session follows it at its next request.
    back = {"texture": 0.0, "context": 1.0, "position_ms": 2000, "origin": "slider:context"}
    b.client.post("/watch/api/settings", json=back)
    assert a.client.get("/watch/api/state").json()["sliders"]["stops"] == {"texture": 0, "context": 2}
    assert a.service.scheduler.spec.context == "relation"


def test_the_next_video_and_the_exposures_work_across_servers(two: dict[str, Any]) -> None:
    a, b = two["a"], two["b"]
    a.client.post("/watch/api/session", json={})
    a.drain()
    a.client.post("/watch/api/start")
    token = a.client.get("/watch/api/captions").json()["captions"]["0"]["token"]
    # b never served that token, and still names its caption in the exposure log.
    shown = {"entries": [{"cue": 0, "token": token, "from_ms": 0, "to_ms": 9000}]}
    assert b.client.post("/watch/api/exposures", json=shown).status_code == 200
    logged = [event for event in events(two["root"]) if event["event"] == "exposures"]
    assert logged[-1]["entries"][0]["job_id"] is not None
    # b ends the video; a opens the next one at its next request, and its old writer stops.
    view = b.client.post("/watch/api/video-ended").json()
    assert view["stage"] == "ready" and view["video"]["number"] == 2
    a.drain()
    assert a.service.scheduler.spec is None  # the first video is closed in the record
    assert a.client.get("/watch/api/state").json()["video"]["number"] == 2
    assert a.service.video_id() == VIDEOS[1]
    b.drain()
    assert a.client.post("/watch/api/start").json()["stage"] == "watch"
    assert b.client.post("/watch/api/video-ended").json()["stage"] == "ratings"
    assert a.pool.holder(a.service.session_id or "") is not None  # a has not heard yet
    assert a.client.get("/watch/api/state").json()["stage"] == "ratings"
    assert a.service.session_id is None and b.service.session_id is None  # no writer is held after the videos
    assert a.client.post("/watch/api/ratings", json={"answers": {"L1": 3}}).json() == {"stage": "done"}
    assert b.client.get("/watch/api/state").json()["stage"] == "done"
    assert b.client.post("/watch/api/ratings", json={"answers": {"L1": 3}}).json() == {"stage": "done"}
    assert "video_restarted" not in [event["event"] for event in events(two["root"])]


def test_the_time_watched_is_counted_once_whichever_server_hears_it(two: dict[str, Any]) -> None:
    a, b, wall, store = two["a"], two["b"], two["wall"], two["store"]
    a.client.post("/watch/api/session", json={})
    a.drain()
    a.client.post("/watch/api/start")
    for second in range(0, 31):  # a report a second, to alternating servers
        server = a if second % 2 else b
        server.client.post("/watch/api/playback", json={"position_ms": second * 1000, "playing": second < 30})
        wall[0] += 1
    b.client.post("/watch/api/video-ended")
    (session,) = store.sessions()
    video, opened = session.state["live"]["coverage"]  # the one watched, and the next, just opened
    assert opened["video_id"] == VIDEOS[1] and opened["watched_ms"] == 0
    assert video == {
        "video_id": VIDEOS[0],
        "duration_ms": 30000,
        "watched_ms": 30000,
        "watched_share": 1.0,
        "furthest_ms": 30000,
        "restarts": 0,
    }


def test_a_video_ended_unwatched_is_recorded_as_unwatched(two: dict[str, Any]) -> None:
    a, store = two["a"], two["store"]
    a.client.post("/watch/api/session", json={})
    a.drain()
    a.client.post("/watch/api/start")
    a.client.post("/watch/api/playback", json={"position_ms": 0, "playing": True})
    a.client.post("/watch/api/playback", json={"position_ms": 29000, "playing": True})  # no time has passed
    a.client.post("/watch/api/video-ended")
    video = store.sessions()[0].state["live"]["coverage"][0]
    assert video["watched_ms"] == 300 and video["watched_share"] == 0.01 and video["furthest_ms"] == 29000


def test_a_browser_that_goes_quiet_stops_the_writers(two: dict[str, Any]) -> None:
    a, wall = two["a"], two["wall"]
    a.client.post("/watch/api/session", json={})
    a.drain()
    a.client.post("/watch/api/start")
    a.client.post("/watch/api/playback", json={"position_ms": 1000, "playing": True})
    a.drain()
    assert a.service.scheduler.playing
    wall[0] += STALE_PLAY_S + 1  # the tab is gone: no report, and no "paused" either
    a.drain()
    assert not a.service.scheduler.playing
    written = len(a.writer.calls)
    wall[0] += 60
    a.drain()
    assert len(a.writer.calls) == written  # it does not write on to the end of the video


def test_settings_and_reports_after_the_videos_change_nothing(two: dict[str, Any]) -> None:
    a, store = two["a"], two["store"]
    a.client.post("/watch/api/session", json={})
    for _ in VIDEOS:
        a.drain()
        a.client.post("/watch/api/start")
        a.client.post("/watch/api/video-ended")
    move = {"texture": 1.0, "context": 1.0, "position_ms": 0, "origin": "reset"}
    assert a.client.post("/watch/api/settings", json=move).status_code == 409
    assert a.client.post("/watch/api/playback", json={"position_ms": 5, "playing": True}).json() == {
        "ok": True
    }
    live = store.sessions()[0].state["live"]
    assert (live["texture"], live["context"], live["stage"]) == (0.5, 0.5, "ratings")
    off_stop = {"texture": 0.37, "context": 0.5, "position_ms": 0, "origin": "reset"}
    assert a.client.post("/watch/api/settings", json=off_stop).status_code in (409, 422)


def test_a_grown_pool_never_refuses_a_session_in_the_middle_of_a_video(tmp_path: Path) -> None:
    for video in VIDEOS:
        (tmp_path / f"{video}.webm").write_bytes(video.encode())
    wall = [1000.0]
    local = [0.0]
    made: list[LiveService] = []

    def spawn() -> LiveService:
        made.append(LiveService(Counting("다"), tmp_path / "logs", clock=lambda: 0.0))
        return made[-1]

    server = Server("가", tmp_path, wall, spawn=spawn, idle_s=600.0, clock=lambda: local[0])
    first, second = TestClient(server.app), TestClient(server.app)
    assert enter(first, "P1").status_code == 200
    assert enter(second, "P2").status_code == 200  # the lab's one-writer pool would answer 503
    assert len(server.pool.services) == 2 and made
    # P1 sends nothing for a long while; P3 arrives and takes over P1's writer.
    local[0] += 601
    third = TestClient(server.app)
    assert enter(third, "P3").status_code == 200
    assert len(server.pool.services) == 2
    # P1 comes back: its state is in the store, so it simply claims a writer again (P2's, idle too).
    assert first.post("/watch/api/session", json={}).json()["stage"] == "ready"
    assert len(server.pool.services) == 2
    assert second.post("/watch/api/session", json={}).json()["stage"] == "ready"  # and so does P2
    assert len(server.pool.services) == 3
