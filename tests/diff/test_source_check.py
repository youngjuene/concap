"""The perceived-source block: the draw, its routes, what the browser sees, the export."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_live_server import ITEMS, FakeAttempt, drain, enter, job_for

from diff.caption_sources import parse_sources
from diff.export_bundle import SOURCE_CHECK_COLUMNS, source_check_rows
from diff.live_generation import LiveService, LiveStream
from diff.live_server import LivePool, LiveVideo, rating_items
from diff.participant_server import build_participant_app
from diff.session_store import SessionStore
from diff.source_check import (
    OFF_SCREEN_ANSWER,
    ON_SCREEN_ANSWER,
    TRIALS_OFF_SCREEN,
    TRIALS_ON_SCREEN,
    draw,
    pool_problems,
)

CUES = 15  # two videos, 30 cues: 15 on screen, 14 off, and one with no mask
TRIALS = TRIALS_ON_SCREEN + TRIALS_OFF_SCREEN


def stream(video_id: str) -> LiveStream:
    document = {
        "schema": "diff.caption-sources/v1",
        "sources": [
            {
                "schema": "diff.caption-source/v1",
                "phase": 2,
                "unit": {"kind": "video", "key": video_id},
                "cue_index": index,
                "start_ms": index * 10000,
                "end_ms": (index + 1) * 10000,
                "text": f"{video_id} 소리 {index}",
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
            for index in range(CUES)
        ],
    }
    return LiveStream.from_sources(parse_sources(document).sources, video_id)


def videos_at(root: Path) -> list[LiveVideo]:
    made = []
    for video_id in ("secret-a", "secret-b"):
        media = root / f"{video_id}.webm"
        media.write_bytes(video_id.encode())
        made.append(LiveVideo(video_id, media, stream(video_id)))
    return made


def scenes_for(videos: list[LiveVideo]) -> dict[str, Any]:
    cues: dict[str, Any] = {}
    number = 0
    for video in videos:
        for cue in video.stream.cues:
            on_screen = number % 2 == 0
            cues[cue.source_id] = {
                "sounds": [
                    {
                        "sound": f"소리 {number}",
                        "mask_match": "none" if number == 29 else "label",
                        "visibility": {"mean": 0.9 if on_screen else 0.1, "on_screen": on_screen},
                    }
                ]
            }
            number += 1
    return {"cues": cues}


@pytest.fixture
def lab(tmp_path: Path) -> dict[str, Any]:
    videos = videos_at(tmp_path)
    scenes = scenes_for(videos)
    service = LiveService(FakeAttempt(), tmp_path / "logs", clock=lambda: 0.0)
    store = SessionStore(tmp_path / "sessions.sqlite")
    app = build_participant_app(
        store=store,
        pool=LivePool([service]),
        videos=videos,
        items=rating_items(ITEMS),
        log_dir=tmp_path / "logs",
        job_for=job_for,
        phase1=None,
        secret=b"test-secret",
        source_draw=lambda session_id: draw(session_id, scenes, videos),
    )
    return {
        "app": app,
        "service": service,
        "store": store,
        "videos": videos,
        "scenes": scenes,
        "root": tmp_path,
    }


def through_ratings(lab: dict[str, Any]) -> TestClient:
    client = TestClient(lab["app"])
    enter(client)
    client.post("/watch/api/session", json={})
    for _ in lab["videos"]:
        drain(lab["service"])
        assert client.post("/watch/api/start").status_code == 200
        client.post("/watch/api/video-ended")
    reply = client.post("/watch/api/ratings", json={"answers": {"L1": 4}})
    assert reply.json() == {"stage": "source"}
    return client


def answer(client: TestClient, index: int, choice: str = ON_SCREEN_ANSWER, played: int = 10000) -> Any:
    body = {
        "index": index,
        "answer": choice,
        "played_ms": played,
        "waited_ms": 80,
        "stalls": 0,
        "restarts": 0,
    }
    return client.post("/watch/api/source-answer", json=body)


def test_the_draw_is_stratified_grouped_by_video_and_repeatable(tmp_path: Path) -> None:
    videos = videos_at(tmp_path)
    scenes = scenes_for(videos)
    first = draw("session-1", scenes, videos)
    assert first == draw("session-1", scenes, videos)
    assert first["trials"] != draw("session-2", scenes, videos)["trials"]
    trials = first["trials"]
    assert len(trials) == TRIALS and len({t["source_id"] for t in trials}) == TRIALS
    assert Counter(t["on_screen"] for t in trials) == {True: TRIALS_ON_SCREEN, False: TRIALS_OFF_SCREEN}
    order = [t["video_id"] for t in trials]
    assert sum(a != b for a, b in zip(order, order[1:], strict=False)) <= 1  # one video, then the other
    assert sorted(first["answer_order"]) == sorted([ON_SCREEN_ANSWER, OFF_SCREEN_ANSWER])


def test_a_cue_with_no_mask_is_never_drawn(tmp_path: Path) -> None:
    videos = videos_at(tmp_path)
    scenes = scenes_for(videos)
    unmasked = next(k for k, c in scenes["cues"].items() if c["sounds"][0]["mask_match"] == "none")
    drawn = {t["source_id"] for seed in range(200) for t in draw(f"s{seed}", scenes, videos)["trials"]}
    assert unmasked not in drawn and len(drawn) == 29


def test_a_pool_too_small_is_a_problem_before_anyone_starts(tmp_path: Path) -> None:
    videos = videos_at(tmp_path)
    scenes = scenes_for(videos)
    assert pool_problems(scenes, videos) == []
    assert "no scene parts" in pool_problems(None, videos)[0]
    for cue in scenes["cues"].values():
        cue["sounds"][0]["visibility"]["on_screen"] = True
    assert pool_problems(scenes, videos) == ["0 off-screen cues, the block draws 11"]


def test_the_block_follows_the_ratings_and_ends_the_session(lab: dict[str, Any]) -> None:
    client = through_ratings(lab)
    page = client.get("/watch/api/source").json()
    assert page["next"] == 0 and len(page["trials"]) == TRIALS and len(page["answers"]) == 2
    for index in range(TRIALS - 1):
        assert answer(client, index).json() == {"stage": "source", "next": index + 1}
    assert answer(client, TRIALS - 1, OFF_SCREEN_ANSWER).json()["stage"] == "done"
    session = lab["store"].by_participant("P1") if hasattr(lab["store"], "by_participant") else None
    assert client.get("/watch/api/source").status_code == 409  # the block is over
    # The finished session's browser keeps its end screen: a reload, or the last answer sent again
    # because its reply was lost, never meets "no session".
    assert client.get("/watch/api/state").json()["stage"] == "done"
    assert answer(client, TRIALS - 1, OFF_SCREEN_ANSWER).json() == {"stage": "done", "next": TRIALS}
    assert answer(client, TRIALS - 1, ON_SCREEN_ANSWER).status_code == 409  # a different answer is not kept
    assert client.get("/api/p1/state").json() == {"kind": "moved", "next_url": "watch/"}
    assert len(lab["store"].sessions()[0].state["live"]["source_answers"]) == TRIALS
    assert session is None or session.finished_at is not None


def test_answers_keep_their_order_and_a_retry_counts_once(lab: dict[str, Any]) -> None:
    client = through_ratings(lab)
    assert answer(client, 1).status_code == 409
    assert answer(client, 0).json()["next"] == 1
    assert answer(client, 0).json()["next"] == 1  # the same request again
    assert answer(client, 0, OFF_SCREEN_ANSWER).status_code == 409  # a change of mind is not a retry
    assert answer(client, 1, "somewhere").status_code == 422
    body = {"index": 1, "answer": ON_SCREEN_ANSWER}
    assert client.post("/watch/api/source-answer", json=body).status_code == 422


def test_the_page_never_sees_the_masks_the_stratum_or_a_video_s_name(lab: dict[str, Any]) -> None:
    client = through_ratings(lab)
    text = client.get("/watch/api/source").text
    for secret in ("secret-a", "secret-b", "visibility", "source_id", "video_id"):
        assert secret not in text, secret


def test_the_block_serves_every_video_and_only_during_the_block(lab: dict[str, Any]) -> None:
    client = TestClient(lab["app"])
    enter(client)
    watching = client.post("/watch/api/session", json={}).json()["video"]["media"]
    client2 = through_ratings_from(lab, client)
    trials = client2.get("/watch/api/source").json()["trials"]
    tokens = {trial["media"] for trial in trials}
    assert len(tokens) == 2 and watching in tokens
    for media in tokens:
        assert client2.get(f"/watch/{media}").status_code == 200
    assert client2.get("/watch/media/wrong").status_code == 404


def through_ratings_from(lab: dict[str, Any], client: TestClient) -> TestClient:
    for _ in lab["videos"]:
        drain(lab["service"])
        client.post("/watch/api/start")
        client.post("/watch/api/video-ended")
    client.post("/watch/api/ratings", json={"answers": {"L1": 4}})
    return client


def test_a_restart_resumes_at_the_next_cue_with_the_same_draw(lab: dict[str, Any], tmp_path: Path) -> None:
    client = through_ratings(lab)
    before = client.get("/watch/api/source").json()
    for index in range(3):
        answer(client, index)
    # A new app on the same store is another Vercel instance, or a restart.
    restarted = build_participant_app(
        store=lab["store"],
        pool=LivePool([LiveService(FakeAttempt(), tmp_path / "logs2", clock=lambda: 0.0)]),
        videos=lab["videos"],
        items=rating_items(ITEMS),
        log_dir=tmp_path / "logs2",
        job_for=job_for,
        phase1=None,
        secret=b"test-secret",
        source_draw=lambda session_id: draw(session_id, lab["scenes"], lab["videos"]),
    )
    other = TestClient(restarted)
    other.cookies.update(client.cookies)
    after = other.get("/watch/api/source").json()
    assert after["next"] == 3 and after["trials"] == before["trials"]
    assert answer(other, 3).json()["next"] == 4
    # The first instance still holds its copy, and takes the next answer from the store's count.
    assert answer(client, 4).json()["next"] == 5


def test_without_a_draw_the_items_still_end_the_session(tmp_path: Path) -> None:
    videos = videos_at(tmp_path)
    service = LiveService(FakeAttempt(), tmp_path / "logs", clock=lambda: 0.0)
    app = build_participant_app(
        store=SessionStore(tmp_path / "s.sqlite"),
        pool=LivePool([service]),
        videos=videos,
        items=rating_items(ITEMS),
        log_dir=tmp_path / "logs",
        job_for=job_for,
        phase1=None,
        secret=b"test-secret",
    )
    client = TestClient(app)
    enter(client)
    client.post("/watch/api/session", json={})
    for _ in videos:
        drain(service)
        client.post("/watch/api/start")
        client.post("/watch/api/video-ended")
    assert client.post("/watch/api/ratings", json={"answers": {"L1": 4}}).json() == {"stage": "done"}


def test_the_export_row_joins_each_answer_to_its_cue(tmp_path: Path) -> None:
    videos = videos_at(tmp_path)
    plan = draw("s", scenes_for(videos), videos)
    answers = [
        {
            "index": 0,
            "source_id": plan["trials"][0]["source_id"],
            "answer": ON_SCREEN_ANSWER,
            "played_ms": 10000,
            "waited_ms": 50,
            "stalls": 0,
            "restarts": 0,
            "answered_at": 1.0,
        }
    ]
    rows = source_check_rows("P1", "s", True, {"source": plan, "source_answers": answers})
    assert len(rows) == 1 and set(rows[0]) == set(SOURCE_CHECK_COLUMNS)
    assert rows[0]["visibility"] == plan["trials"][0]["visibility"] and rows[0]["answer"] == ON_SCREEN_ANSWER
    assert source_check_rows("P1", "s", True, {}) == []
    out = tmp_path / "out.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, SOURCE_CHECK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert out.read_text(encoding="utf-8").count("\n") == 2
