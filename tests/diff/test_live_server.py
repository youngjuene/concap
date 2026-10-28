"""The live watch: the page flow, capacity, ratings, the event log and blindness (§6.5).

These run the participant app without phase 1 (a smoke run), so an entry goes
straight to the watch; tests/diff/test_participant_server.py covers phase 1.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from diff.caption_sources import CaptionSource, parse_sources
from diff.job_plan import Job, live_job
from diff.live_generation import LiveService, LiveStream, Spec, slider_steps
from diff.live_server import LivePool, LiveVideo, rating_items
from diff.model_profile import TRIDA_7B
from diff.participant_server import build_participant_app
from diff.session_store import SessionStore
from diff.study_protocol import CONTEXT_STEPS, TWEAK_MAX_CHARS

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
    return live_job(source, spec.texture, spec.context, spec.system_message, "smoke")


def full_output(text: str, terminator: str | None = "endoftext") -> dict[str, Any]:
    """An output record as the worker sends it, which the checks read."""
    return {
        "raw_text": text,
        "text": text,
        "terminator": terminator,
        "terminator_index": len(text) if terminator else None,
        "postprocess": {"tail_token_ids": [], "tail_text": ""},
        "leftover_mask_count": 0,
        "special_ids_in_text": [],
        "new_token_count": 40,
    }


class FakeAttempt:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append(payload["job_id"])
        if payload.get("cell") is None:  # phase 1's second caption, naming the attended sound
            attended = re.search(r"그중 (.*?)에 가장 주의를 기울였습니다", payload["prompt"]["text"])
            text = f"{attended.group(1) if attended else ''}가 또렷하게 들린다."
            return {"status": "completed", "output": full_output(text), "latency_ms": 400}
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
    store = SessionStore(tmp_path / "sessions.sqlite")
    app = build_participant_app(
        store=store,
        pool=LivePool([service]),
        videos=[LiveVideo(VIDEO, media, stream())],
        items=rating_items(ITEMS),
        log_dir=tmp_path / "logs",
        job_for=job_for,
        phase1=None,
        secret=b"test-secret",
    )
    return {"app": app, "service": service, "logs": tmp_path / "logs", "media": media, "store": store}


def enter(client: TestClient, participant: str = "P1") -> Any:
    return client.post("/api/enter", json={"participant": participant}, headers={"x-study-request": "1"})


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

    assert keep(enter(client)).json() == {"next_url": "watch/"}
    started = keep(client.post("/watch/api/session", json={}))
    assert started.status_code == 200 and started.json()["stage"] == "ready"
    # The calibrated start (0.5, 0.5) is the middle stop of each slider; no phase 1, no session progress.
    assert started.json()["sliders"]["stops"] == {"texture": 1, "context": 1}
    assert started.json()["video"]["duration_ms"] == 15000 and "progress" not in started.json()
    assert keep(client.post("/watch/api/start")).status_code == 409  # cue 0 is not written yet
    drain(service)
    ready = keep(client.get("/watch/api/captions")).json()["captions"]
    assert sorted(ready) == ["0", "1", "2"] and ready["0"]["text"] == "다시 쓴 자막 0번"
    assert ready["0"]["current"] is True and ready["0"]["settled"] is True
    media_url = started.json()["video"]["media"]
    assert keep(client.get("/watch/" + media_url)).status_code == 200
    assert client.get("/watch/media/wrong-token").status_code == 404
    assert keep(client.post("/watch/api/start")).json()["stage"] == "watch"
    assert keep(
        client.post(
            "/watch/api/settings",
            json={"texture": 1.0, "context": 1.0, "position_ms": 0, "origin": "slider:texture"},
        )
    ).json() == {"ok": True}
    playback = {"position_ms": 1000, "playing": True}
    assert keep(client.post("/watch/api/playback", json=playback)).status_code == 200
    hidden = {"position_ms": 1200, "playing": False, "hidden": True}  # the page paused for a hidden tab
    assert keep(client.post("/watch/api/playback", json=hidden)).status_code == 200
    assert keep(client.post("/watch/api/playback", json=playback)).status_code == 200
    drain(service)
    token = ready["0"]["token"]
    keep(
        client.post(
            "/watch/api/exposures",
            json={"entries": [{"cue": 0, "token": token, "from_ms": 0, "to_ms": 5000}]},
        )
    )
    assert keep(client.post("/watch/api/video-ended")).json()["stage"] == "ratings"
    assert service.free  # the last video releases the GPU before the items
    assert keep(client.get("/watch/api/items")).json()["items"][0]["id"] == "L1"
    assert keep(client.post("/watch/api/ratings", json={"answers": {"L1": 9}})).status_code == 422
    assert keep(client.post("/watch/api/ratings", json={"answers": {"L1": 5}})).json() == {"stage": "done"}
    assert enter(TestClient(app_parts["app"])).status_code == 409  # a finished ID cannot be entered again

    everything = "\n".join(bodies)
    for secret in (
        "job-",
        "trida",
        TRIDA_7B.profile_id,
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
    on_video = [event for event in events if "video_index" in event]
    assert on_video and all(event["video_id"] == VIDEO for event in on_video)  # the log names the video
    assert events[0]["calibration"]["heard_observations"] == {}  # no phase 1 in a smoke run
    assert next(e for e in events if e["event"] == "exposures")["entries"][0]["job_id"].startswith("job-")
    pauses = [(e["position_ms"], e["hidden"]) for e in events if e["event"] == "playback"]
    assert pauses == [(1000, False), (1200, True), (1000, False)]
    settings_event = next(e for e in events if e["event"] == "settings")
    assert settings_event["steps"] == {"texture": 2, "context": "relation"}  # texture capped at step 2
    live_log = next(app_parts["logs"].glob("live-*.jsonl"))
    attempts = [json.loads(line) for line in live_log.read_text(encoding="utf-8").splitlines()]
    assert attempts[0]["system_message"].startswith("당신은") and "없음" in attempts[0]["system_message"]
    assert attempts[-1]["spec"] == {"texture": 2, "context": "relation"}


def test_a_second_participant_is_refused_while_the_gpu_is_taken(app_parts: dict[str, Any]) -> None:
    first, second = TestClient(app_parts["app"]), TestClient(app_parts["app"])
    assert enter(first, "P1").status_code == 200
    busy = enter(second, "P2")
    assert busy.status_code == 503 and "잠시 후" in busy.json()["error"]
    assert "diff_session" not in busy.cookies


def test_requests_without_a_session_are_refused(app_parts: dict[str, Any]) -> None:
    client = TestClient(app_parts["app"])
    assert client.get("/watch/api/captions").status_code == 401
    assert client.post("/watch/api/settings", json={}).status_code == 401
    assert client.post("/watch/api/session", json={}).status_code == 401


def test_bad_settings_are_refused(app_parts: dict[str, Any]) -> None:
    client = TestClient(app_parts["app"])
    enter(client)
    client.post("/watch/api/session", json={})
    for body in (
        {"texture": 2, "context": 0.5, "position_ms": 0, "origin": "slider:texture"},
        {"texture": 0.5, "context": 0.5, "position_ms": 0, "origin": "keyboard"},
        {"texture": 0.5},
    ):
        assert client.post("/watch/api/settings", json=body).status_code == 422


def test_the_browser_s_three_stops_are_the_prompt_s_steps() -> None:
    """live.js sends a stop as stop / 2; each lands on its own step, in order."""
    assert [slider_steps(stop / 2, stop / 2) for stop in (0, 1, 2)] == [
        (step, CONTEXT_STEPS[step]) for step in (0, 1, 2)
    ]


def test_rating_items_are_validated() -> None:
    assert rating_items(ITEMS)[0].scale_max == 6
    for bad in (
        {"items": []},
        {"items": [{"id": "x", "text": "t", "min": 3, "max": 3, "anchors": ["a", "b"]}]},
    ):
        with pytest.raises(ValueError):
            rating_items(bad)


@pytest.mark.parametrize(
    "approved",
    ["", "TUNER_PROVISIONAL", "LIVE_PROMPTS_PROVISIONAL", "BINDING_PROVISIONAL", "PHASE2_SCENE_PROVISIONAL"],
)
def test_participant_sessions_wait_for_the_researcher_s_tuner_text_and_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved: str
) -> None:
    from diff.__main__ import main

    # All provisional, then one approved, whatever the code's current approvals are.
    modules = {"BINDING_PROVISIONAL": "phase1_binding", "PHASE2_SCENE_PROVISIONAL": "phase2_scene"}
    for flag in (
        "TUNER_PROVISIONAL",
        "LIVE_PROMPTS_PROVISIONAL",
        "BINDING_PROVISIONAL",
        "PHASE2_SCENE_PROVISIONAL",
    ):
        monkeypatch.setattr(f"diff.{modules.get(flag, 'rewrite_prompts')}.{flag}", flag != approved)

    sources = tmp_path / "sources.json"
    document = {"schema": "diff.caption-sources/v1", "sources": [cue.record() for cue in stream().cues]}
    sources.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    items = tmp_path / "items.json"
    items.write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    media = tmp_path / "v.webm"
    media.write_bytes(b"x")
    argv = ["live", "serve", "--sources", str(sources), "--video", f"{VIDEO}={media}", "--items", str(items)]
    argv += ["--log-dir", str(tmp_path / "logs"), "--fake-worker", "--purpose", "participant"]
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    for blocker, flag in (
        ("tuner system message", "TUNER_PROVISIONAL"),
        ("nine live prompts are still provisional", "LIVE_PROMPTS_PROVISIONAL"),
        ("binding model's parameters and plan are still provisional", "BINDING_PROVISIONAL"),
        ("scene parts and scene-aware live prompts are still provisional", "PHASE2_SCENE_PROVISIONAL"),
    ):
        assert (blocker in str(stopped.value)) is (flag != approved), blocker


def test_participant_sessions_need_reviewed_starting_captions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from diff.__main__ import main

    monkeypatch.setattr("diff.rewrite_prompts.TUNER_PROVISIONAL", False)
    monkeypatch.setattr("diff.rewrite_prompts.LIVE_PROMPTS_PROVISIONAL", False)
    monkeypatch.setattr("diff.phase1_binding.BINDING_PROVISIONAL", False)
    sources = tmp_path / "sources.json"  # the test stream's cues are drafts
    document = {"schema": "diff.caption-sources/v1", "sources": [cue.record() for cue in stream().cues]}
    sources.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    items = tmp_path / "items.json"
    items.write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    argv = ["live", "serve", "--sources", str(sources), "--video", f"{VIDEO}={tmp_path / 'v.webm'}"]
    argv += ["--items", str(items), "--log-dir", str(tmp_path / "logs"), "--fake-worker"]
    argv += ["--phase1-setup", str(tmp_path / "setup.json")]
    with pytest.raises(SystemExit, match="every phase-2 starting caption reviewed"):
        main([*argv, "--purpose", "participant"])


def test_preflight_lists_every_blocker_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from diff.__main__ import main

    def no_key(self: object) -> None:  # never a real API call from a test
        raise TypeError("Could not resolve authentication method")

    monkeypatch.setattr("diff.claude_writer.ClaudeWriter.warm_up", no_key)
    monkeypatch.setattr("diff.phase1_binding.BINDING_PROVISIONAL", True)  # every gate closed
    sources = tmp_path / "sources.json"  # the test stream's cues are drafts
    document = {"schema": "diff.caption-sources/v1", "sources": [cue.record() for cue in stream().cues]}
    sources.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    items = tmp_path / "items.json"
    items.write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    argv = ["live", "preflight", "--sources", str(sources), "--video", f"{VIDEO}={tmp_path / 'missing.mp4'}"]
    argv += ["--items", str(items), "--store", str(tmp_path / "s.sqlite")]
    assert main(argv) == 1
    printed = capsys.readouterr().out
    for blocker in (
        "no video at",
        "tuner system message is still provisional",
        "nine live prompts are still provisional",
        "binding model's parameters and plan are still provisional",
        "need phase 1",
        "3 cues are not",
        "the writer cannot reach the Claude API: TypeError",
    ):
        assert blocker in printed, blocker
    assert "post-watch items: 1" in printed and "not created yet" in printed


def test_participant_sessions_need_phase_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from diff.__main__ import main

    monkeypatch.setattr("diff.rewrite_prompts.TUNER_PROVISIONAL", False)
    monkeypatch.setattr("diff.rewrite_prompts.LIVE_PROMPTS_PROVISIONAL", False)
    monkeypatch.setattr("diff.phase1_binding.BINDING_PROVISIONAL", False)
    sources = tmp_path / "sources.json"
    document = {"schema": "diff.caption-sources/v1", "sources": [cue.record() for cue in stream().cues]}
    sources.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    items = tmp_path / "items.json"
    items.write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    argv = ["live", "serve", "--sources", str(sources), "--video", f"{VIDEO}={tmp_path / 'v.webm'}"]
    argv += ["--items", str(items), "--log-dir", str(tmp_path / "logs"), "--fake-worker"]
    with pytest.raises(SystemExit, match="--phase1-setup"):
        main([*argv, "--purpose", "participant"])


def test_the_fake_worker_quotes_the_cue_of_a_live_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # live-p2 (v4 on) ends with the cue on a "원래 자막:" line; the examples before it must not be taken.
    from diff.__main__ import _fake_live_attempt

    monkeypatch.setattr("time.sleep", lambda seconds: None)
    cue = stream().cues[0]
    job = live_job(cue, 1, "broad", "튜너", "smoke")
    text = _fake_live_attempt(job.record(), 1.0)["output"]["text"]
    assert text == f"{cue.text} (시험용 · 질감 1 · broad)"


def test_the_fake_worker_s_second_caption_fits_the_tweak_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from diff.__main__ import _fake_live_attempt

    monkeypatch.setattr("time.sleep", lambda seconds: None)
    long = "가" * 120
    payload = {"prompt": {"text": f"자막입니다: 「{long}」\n고쳐 쓰세요."}, "cell": None}
    text = _fake_live_attempt(payload, 1.0)["output"]["text"]
    assert len(text) <= TWEAK_MAX_CHARS and text.endswith("(시험용 · 두 번째 자막)")
