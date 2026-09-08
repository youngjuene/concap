"""Regression coverage for the two-stage protocol and delivery boundaries."""

from __future__ import annotations

import json
import subprocess
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dpo.regen.config import SOUND_FAMILIES
from dpo.regen.study_api import build_study_app
from dpo.regen.study_schema import FINAL_ITEMS, PREFERENCES, answers, caption_instruction, compile_profile
from dpo.regen.study_store import Conflict, StudyStore
from dpo.regen.study_worker import excerpt


def make_media(root: Path) -> dict[str, Any]:
    """Real tiny videos support duration/range checks as well as browser fixtures."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x425f78:s=160x90:r=2",
            "-t",
            "300",
            "-c:v",
            "libvpx",
            "-b:v",
            "30k",
            "-y",
            str(root / "long.webm"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(root / "long.webm"),
            "-t",
            "2",
            "-c",
            "copy",
            "-y",
            str(root / "short.webm"),
        ],
        check=True,
    )
    Image.new("RGB", (160, 90), "#425f78").save(root / "frame.png")
    for index, colour in enumerate(("0x425f78", "0x506b42", "0x805344")):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={colour}:s=160x90:r=2",
                "-t",
                "300",
                "-c:v",
                "libvpx",
                "-b:v",
                "30k",
                "-y",
                str(root / f"long-{index}.webm"),
            ],
            check=True,
        )
    Image.new("L", (160, 90), 255).save(root / "mask.png")
    with wave.open(str(root / "audio.wav"), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 8000 * 300)
    return {
        "schema": "dpo.caption-study/v2",
        "language": "en",
        "calibration_clips": [
            {
                "id": f"clip-{i}",
                "title": f"Calibration clip {i + 1}",
                "video": "short.webm",
                "duration_ms": 2000,
                "present_families": ["things"],
                "frames": [
                    {
                        "at_ms": 1000,
                        "still": "frame.png",
                        "objects": [{"id": "bus", "label": "Bus", "mask": "mask.png"}],
                    }
                ],
            }
            for i in range(3)
        ],
        "viewing_videos": [
            {
                "id": f"long-{i}",
                "title": f"Long video {i + 1}",
                "status": "ready",
                "video": f"long-{i}.webm",
                "audio": "audio.wav",
                "duration_ms": 300000,
                "cues": [
                    {
                        "start_ms": start,
                        "end_ms": start + 5000,
                        "evidence": "An audible bus engine has a steady low rumble.",
                        "fallback": {"en": "A bus engine rumbles."},
                    }
                    for start in range(0, 300000, 5000)
                ],
            }
            for i in range(3)
        ],
    }


@pytest.fixture(scope="module")
def assets(tmp_path_factory: Any) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("study-media")
    return root, make_media(root)


class Session:
    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.client.headers["x-study-request"] = "1"
        response = client.post("/api/study/session", json={})
        self.cookie_name = response.headers["set-cookie"].split("=", 1)[0]
        self.state = response.json()

    @property
    def token(self) -> str:
        return str(self.client.cookies[self.cookie_name])

    def submit(self, action: str, data: dict[str, Any] | None = None, expected: int = 200) -> Any:
        result = self.client.post(
            f"/api/study/{action}",
            json={
                "revision": self.state["revision"],
                "key": uuid.uuid4().hex,
                "data": data or {},
            },
        )
        assert result.status_code == expected, result.text
        if expected == 200:
            self.state = result.json()
        return result

    def calibrate(self) -> None:
        self.submit("preferences", {"texture": 4, "context": 2})
        now = [1000.0]
        with patch("dpo.regen.study_api.time.time", lambda: now[0]):
            for index in range(3):
                for sequence, ms in enumerate((0, 1000, 2000)):
                    now[0] += 1
                    self.submit(
                        "clip-playback",
                        {
                            "clip_id": f"clip-{index}",
                            "position_ms": ms,
                            "sequence": sequence,
                            "playing": True,
                        },
                    )
                self.submit("clip-ended")
                self.submit(
                    "observation",
                    {
                        "heard": {f: f == "things" for f in SOUND_FAMILIES},
                        "points": [{"frame": 0, "x": 0.5, "y": 0.5}],
                    },
                )


@pytest.fixture
def session(assets: Any, tmp_path: Path) -> Any:
    root, manifest = assets
    path = tmp_path / "study.json"
    path.write_text(json.dumps(manifest))
    app = build_study_app(path, root, tmp_path / "out")
    # No lifespan: unit/integration tests control completion rather than load a model.
    client = TestClient(app)
    return Session(client), app.state.store, path


def test_multiclip_calibration_and_pending_handoff(session: Any) -> None:
    s, store, path = session
    s.submit("start-viewing", expected=409)
    s.calibrate()
    assert s.state["stage"] == "ready"
    token = s.token
    before = store.state(token)["profile"]
    assert before["sample_count"] == 3
    assert before["defaults"] == {"texture": 0.75, "context": 0.25}
    document = json.loads(path.read_text())
    document["viewing_videos"][0]["status"] = "pending"
    path.write_text(json.dumps(document))
    s.submit("start-viewing", expected=400)
    assert store.state(token)["profile"] == before
    document["viewing_videos"][0]["status"] = "ready"
    path.write_text(json.dumps(document))
    s.submit("start-viewing")
    assert s.state["video"]["id"] == "long-0"
    assert store.state(token)["profile"] == before
    assert s.client.get("/study/media/watch/0", headers={"Range": "bytes=0-31"}).status_code == 206


def test_three_videos_one_final_survey_and_completion_coverage(session: Any, monkeypatch: Any) -> None:
    s, store, _ = session
    s.calibrate()
    s.submit("start-viewing")
    s.submit("final-survey", expected=409)
    now = [1000.0]
    monkeypatch.setattr("dpo.regen.study_api.time.time", lambda: now[0])
    profile = store.state(s.token)["profile"]["hash"]
    for index in range(3):
        video_id = f"long-{index}"
        s.submit("video-ended", {"video_id": video_id}, expected=400)
        for sequence, ms in enumerate(range(0, 300001, 5000)):
            now[0] += 5
            s.submit(
                "playback", {"video_id": video_id, "position_ms": ms, "sequence": sequence, "playing": True}
            )
        s.submit("video-ended", {"video_id": video_id})
        if index < 2:
            assert s.state["stage"] == "break"
            s.submit("continue")
    assert s.state["stage"] == "final"
    response: dict[str, Any] = {item["id"]: 4 for item in FINAL_ITEMS if item["type"] == "rating"}
    response.update(timing="Fast enough", control_texture="na")
    receipt = {"revision": s.state["revision"], "key": "final-once", "data": response}
    first = s.client.post("/api/study/final-survey", json=receipt)
    second = s.client.post("/api/study/final-survey", json=receipt)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    stored = store.state(s.token)
    assert stored["profile"]["hash"] == profile
    assert len(stored["completions"]) == 3
    assert stored["final_survey"]["answers"]["comments"] == ""


def test_stale_playback_and_nonfinite_controls(session: Any) -> None:
    s, _, _ = session
    s.calibrate()
    s.submit("start-viewing")
    s.submit("settings", {"video_id": "long-0", "texture": True, "context": 0.5}, expected=400)
    data = {"video_id": "long-0", "position_ms": 250000, "sequence": 1, "playing": False, "seek": True}
    s.submit("playback", data)
    assert s.state["epoch"] == 1
    s.submit("playback", data, expected=409)
    s.submit("video-ended", {"video_id": "long-0"}, expected=400)
    s.submit("settings", {"video_id": "long-1", "texture": 0.5, "context": 0.5}, expected=409)


def test_survey_types_and_na_are_distinct() -> None:
    with pytest.raises(ValueError):
        answers({"texture": "na", "context": 3}, PREFERENCES)
    with pytest.raises(ValueError):
        answers({"texture": True, "context": 3}, PREFERENCES)
    assert answers({"texture": 1, "context": 5}, PREFERENCES) == {"texture": 1, "context": 5}


def test_prompt_calibration_axes_and_relative_time() -> None:
    obs = {"clip": "one", "heard": {f: f == "things" for f in SOUND_FAMILIES}, "labels": ["Bus"]}
    profile = compile_profile({"texture": 5, "context": 1}, [obs], "en")
    other = compile_profile(
        {"texture": 5, "context": 1},
        [{**obs, "labels": ["Tree"], "heard": {f: f == "animal" for f in SOUND_FAMILIES}}],
        "en",
    )
    cue = {"start_ms": 240000, "end_ms": 245000, "evidence": "A bus idles."}
    low = caption_instruction(profile, {"texture": 0, "context": 0}, cue)
    high = caption_instruction(profile, {"texture": 1, "context": 1}, cue)
    assert low != high and profile["template"] != other["template"]
    assert "starts at 0 and lasts 5.000" in high
    assert "240000" not in high


def test_atomic_mutation_and_idempotency(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "state.sqlite3")
    token = store.create("hash", "en")

    def mutate(i: int) -> str:
        try:
            store.mutate(token, f"key-{i}", 0, "request", lambda s: s.update(stage="clip"))
            return "ok"
        except Conflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(mutate, [1, 2])) == ["conflict", "ok"]
    assert store.state(token)["revision"] == 1


def test_lookahead_does_not_supersede_current_job(session: Any) -> None:
    s, store, _ = session
    s.calibrate()
    s.submit("start-viewing")
    token = s.token
    jobs = store.captions(token, store.state(token))
    assert [job["state"] for job in jobs] == ["queued", "queued"]


def test_excerpt_uses_correct_absolute_audio_window(tmp_path: Path) -> None:
    source, target = tmp_path / "source.wav", tmp_path / "slice.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes([42]) * 8000 + bytes([96]) * 8000)
    excerpt(str(source), str(target), 1000, 2000)
    with wave.open(str(target), "rb") as audio:
        assert audio.getnframes() == 8000
        assert set(audio.readframes(8000)) == {96}


def test_calibration_playback_cannot_be_skipped(session: Any) -> None:
    s, _, _ = session
    s.submit("preferences", {"texture": 3, "context": 3})
    s.submit("clip-ended", expected=400)
    s.submit(
        "clip-playback",
        {"clip_id": "clip-0", "position_ms": 2000, "sequence": 0, "playing": False, "seek": True},
    )
    s.submit("clip-ended", expected=400)


def test_cache_survives_revision_but_delivery_does_not(session: Any) -> None:
    s, store, _ = session
    s.calibrate()
    s.submit("start-viewing")
    token = s.token
    job = store.claim()
    assert job is not None
    store.finish(job, {"text": "Generated test caption.", "fallback": False})
    s.submit("settings", {"video_id": "long-0", "texture": 1, "context": 1})
    assert not any(j["result"] for j in s.client.get("/api/study/captions").json()["jobs"])
    s.submit("settings", {"video_id": "long-0", "texture": 0.75, "context": 0.25})
    current = store.captions(token, store.state(token))
    assert current[0]["result"]["text"] == "Generated test caption."
    assert current[0]["revision"] == 2


def test_exposure_deduplication_and_window_bounds(session: Any) -> None:
    s, store, _ = session
    s.calibrate()
    s.submit("start-viewing")
    event = {"id": "event-1", "cue": 0, "start_ms": 100, "end_ms": 1000, "fallback": True}
    s.submit("exposures", {"video_id": "long-0", "entries": [event]})
    s.submit("exposures", {"video_id": "long-0", "entries": [event]})
    exported = store.export(s.token)
    assert len(exported["exposures"]) == 1
    assert "exposures" not in exported["state"]
    s.submit("exposures", {"video_id": "long-0", "entries": [{**event, "end_ms": 7000}]}, expected=400)
    s.submit("exposures", {"video_id": "long-0", "entries": [{**event, "end_ms": 2000}]}, expected=409)


def test_receipt_replay_does_not_rewind_state(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "db")
    token = store.create("hash", "en")
    store.mutate(token, "one", 0, "a", lambda s: s.update(stage="clip"))
    store.mutate(token, "two", 1, "b", lambda s: s.update(stage="observe"))
    result = store.mutate(token, "one", 0, "a", lambda s: pytest.fail("replayed mutation"))
    assert result["stage"] == "observe"
    assert result["revision"] == 2


def test_invalid_manifest_does_not_admit_wrong_duration_or_duplicate_video(
    assets: Any, tmp_path: Path
) -> None:
    from copy import deepcopy

    from dpo.regen.study_schema import load_manifest

    root, source = assets
    document = deepcopy(source)
    document["viewing_videos"][1]["video"] = document["viewing_videos"][0]["video"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="distinct"):
        load_manifest(path, root)
    document = deepcopy(source)
    document["viewing_videos"][0]["video"] = "short.webm"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duration"):
        load_manifest(path, root)


def test_media_change_after_binding_is_rejected(tmp_path: Path) -> None:
    from dpo.regen.study_schema import bound_media, fingerprint

    path = tmp_path / "asset"
    path.write_bytes(b"initial")
    entry = {"video": "asset", "media_hashes": {"video": fingerprint(path)}}
    assert bound_media(tmp_path, entry, "video") == path
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed"):
        bound_media(tmp_path, entry, "video")


def test_studies_on_same_host_do_not_overwrite_session_cookies(
    session: Any, assets: Any, tmp_path: Path
) -> None:
    first, store, path = session
    first.submit("preferences", {"texture": 4, "context": 2})
    other = TestClient(build_study_app(path, assets[0], tmp_path / "other-output"))
    other.cookies.update(first.client.cookies)
    second = Session(other)
    assert second.cookie_name != first.cookie_name
    assert second.state["stage"] == "preferences"
    assert store.state(first.token)["stage"] == "clip"


def stalled_worker(pipe: Any, settings: Any) -> None:
    import time

    pipe.recv()
    time.sleep(10)


def test_worker_timeout_is_bounded_and_recorded(session: Any, monkeypatch: Any) -> None:
    import time

    from dpo.regen import study_worker

    s, store, _ = session
    s.calibrate()
    s.submit("start-viewing")
    monkeypatch.setattr(study_worker, "worker", stalled_worker)
    supervisor = study_worker.Supervisor(store, {}, timeout=0.2)
    supervisor.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            jobs = store.captions(
                s.token, store.state(s.token)
            )
            if any(j["result"] for j in jobs):
                break
            time.sleep(0.05)
        assert jobs[0]["result"]["fallback"] is True
        assert jobs[0]["result"]["reason"] == "inference_deadline"
    finally:
        supervisor.stop()
    assert not supervisor.thread.is_alive()
