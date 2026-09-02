"""The HTTP surface: what reaches the browser, when, and what the log keeps."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.caption.media import MediaError
from dpo.session import app as session_app
from dpo.session.app import CAPTION_FAILED, STATIC_FILES, build_app
from dpo.session.writer import CaptionRequest, TemplateWriter, WriterError

FIXTURE = Path(__file__).parent / "fixtures" / "session.json"
PAGES = {"kiosk.html": "<title>Caption session</title>", "followup.html": "<title>Caption session</title>"}


def _document() -> dict[str, Any]:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _fake_media(root: Path, document: dict[str, Any]) -> Path:
    media = root / "media"
    (media / "unmuted_video").mkdir(parents=True, exist_ok=True)
    for clip in document["clips"]:
        (media / "unmuted_video" / f"{clip['clip_id']}.mp4").write_bytes(b"video-with-sound")
    return media


@pytest.fixture
def package_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frontend files are another builder's; the routes only need a loader."""

    def loader(name: str) -> str:
        if name in PAGES:
            return PAGES[name]
        if name in STATIC_FILES:
            return f"/* {name} */"
        raise FileNotFoundError(name)

    monkeypatch.setattr(session_app, "_package_file", loader)


@pytest.fixture
def client(tmp_path: Path, package_files: None) -> TestClient:
    document = _document()
    return TestClient(
        build_app(document, _fake_media(tmp_path, document), tmp_path / "out", TemplateWriter())
    )


def _submit_listing(client: TestClient, clip_id: str, participant: str = "P01") -> None:
    response = client.post(
        "/api/events",
        json={
            "participant": participant,
            "events": [{"t": 1200, "type": "listing.submit", "clip_id": clip_id, "sounds": ["BUS BRAKES"]}],
            "snapshot": {"screen": "shape", "clip_index": 0},
        },
    )
    assert response.status_code == 200 and response.json() == {"appended": 1}


def test_pages_and_only_the_five_static_files_are_served(client: TestClient) -> None:
    assert client.get("/").status_code == 200
    assert client.get("/followup").status_code == 200
    for name, media_type in STATIC_FILES.items():
        response = client.get(f"/static/{name}")
        assert response.status_code == 200 and response.headers["content-type"] == media_type
    assert client.get("/static/page.html").status_code == 404
    assert client.get("/static/kiosk.html").status_code == 404
    assert client.get("/static/..%2Fuserstudy%2Fpage.html").status_code == 404
    assert client.get("/docs").status_code == 404


def test_the_session_document_leaks_no_weights_prose_or_answers(client: TestClient) -> None:
    document = client.get("/api/session").json()
    serialized = json.dumps(document)
    for leaked in ("weights", "prose", "sources", "scene", "atmosphere", '"a":'):
        assert leaked not in serialized, leaked
    assert [clip["clip_id"] for clip in document["clips"]][:2] == ["demo_tram_stop", "demo_market"]
    assert set(document["followup"]["sound_only"][0]) == {"excerpt_id"}
    assert set(document["followup"]["check"][0]) == {"clip_id"}


def test_state_and_events_round_trip_with_the_participant_id_checked(
    client: TestClient, tmp_path: Path
) -> None:
    assert client.get("/api/state").status_code == 400
    assert client.get("/api/state?participant=P%2001").status_code == 400
    assert client.get("/api/state?participant=P01").json() == {"snapshot": None}
    response = client.post(
        "/api/events",
        json={
            "participant": "P01",
            "events": [
                {"t": 0, "type": "session.begin", "participant": "P01", "session_id": "demo-session"},
                {"t": 3, "type": "screen.enter", "screen": "intro"},
            ],
            "snapshot": {"schema": "dpo.caption-session-snapshot/v1", "screen": "intro"},
        },
    )
    assert response.json() == {"appended": 2}
    assert client.get("/api/state?participant=P01").json()["snapshot"]["screen"] == "intro"
    assert (tmp_path / "out" / "events-P01.jsonl").read_text().count("\n") == 2
    assert client.post("/api/events", json={"participant": "bad id", "events": []}).status_code == 400
    assert client.post("/api/events", json={"participant": "P01", "events": "no"}).status_code == 400
    assert client.post("/api/events", json={"participant": "P01", "events": [{"t": 1}]}).status_code == 400


def test_the_inventory_is_gated_on_the_listing_and_carries_orderings_not_weights(client: TestClient) -> None:
    before = client.get("/api/inventory/demo_tram_stop?participant=P01")
    assert before.status_code == 403 and before.json() == {"error": "listing not submitted"}
    assert client.get("/api/inventory/demo_tram_stop").status_code == 400
    assert client.get("/api/inventory/demo_zoo?participant=P01").status_code == 404
    _submit_listing(client, "demo_tram_stop")
    # Another participant's listing opens nothing for this one.
    assert client.get("/api/inventory/demo_tram_stop?participant=P02").status_code == 403
    inventory = client.get("/api/inventory/demo_tram_stop?participant=P01").json()
    assert inventory["task"] == "shaped" and inventory["poles"] == ["Eye", "Ear"]
    assert inventory["role_heads"] == {
        "underneath": "Underneath",
        "stands_out": "Stands out",
        "only_here": "Only here",
    }
    serialized = json.dumps(inventory)
    assert "weights" not in serialized and "prose" not in serialized
    shot = inventory["shots"][0]
    assert shot["sources"][0] == {
        "id": "tram",
        "token": "TRAM BRAKING",
        "phrases": ["in frame", "once, briefly"],
        "role": "stands_out",
    }
    assert shot["scene"] == {"token": "TRAM STOP ON A WIDE STREET"}
    assert shot["atmosphere"] == {"phrase": "steady, with one rise"}
    assert shot["opening"] == {"level": "itemized", "balance": 0.5}
    full = shot["orderings"]["itemized"]["chatter+footsteps+pigeons+siren+tram"]
    assert (
        full[0]["order"] == ["tram", "pigeons", "footsteps", "chatter", "siren"] and full[0]["span"][0] == 0.0
    )
    assert full[-1]["span"][1] == 1.0
    assert set(shot["auditions"]) == {
        "tram",
        "siren",
        "footsteps",
        "chatter",
        "pigeons",
        "role:underneath",
        "role:stands_out",
        "role:only_here",
    }
    assert shot["auditions"]["siren"] == "A distant siren rises and fades."
    # The control clip: its visual roles and poles, the same shape otherwise.
    _submit_listing(client, "demo_market")
    control = client.get("/api/inventory/demo_market?participant=P01").json()
    assert control["poles"] == ["Near", "Far"]
    assert control["role_heads"] == {
        "backdrop": "Backdrop",
        "passing": "Passing through",
        "fixed": "Fixed here",
    }
    assert "role:backdrop" in control["shots"][0]["auditions"]


def test_caption_round_trip_cached_flag_and_refusals(client: TestClient, tmp_path: Path) -> None:
    body: dict[str, Any] = {
        "participant": "P01",
        "clip_id": "demo_tram_stop",
        "shot_id": "s1",
        "settings": {"level": "itemized", "admitted": ["tram", "siren"], "order": ["tram", "siren"]},
    }
    gated = client.post("/api/caption", json=body)
    assert gated.status_code == 403 and gated.json() == {"error": "listing not submitted"}
    _submit_listing(client, "demo_tram_stop")
    first = client.post("/api/caption", json=body)
    assert first.status_code == 200
    assert first.json() == {
        "caption": "A tram braking once, briefly and a distant siren rises and fades.",
        "key": "itemized|tram,siren",
        "cached": False,
        "prefetched": False,
        "writer": "template",
        "names_excluded": [],
    }
    again = client.post("/api/caption", json=body)
    assert again.json()["cached"] is True and again.json()["caption"] == first.json()["caption"]
    assert json.loads((tmp_path / "out" / "captions.json").read_text())["captions"][
        "demo_tram_stop/s1/itemized|tram,siren"
    ]
    # The other ordering of the same pair is reachable too (they cross).
    flipped = client.post(
        "/api/caption", json={**body, "settings": {**body["settings"], "order": ["siren", "tram"]}}
    )
    assert flipped.status_code == 200 and flipped.json()["key"] == "itemized|siren,tram"
    # tram (0.9, 0.7) always outranks pigeons (0.8, 0.15): pigeons-first is unreachable.
    forged = client.post(
        "/api/caption",
        json={
            **body,
            "settings": {"level": "itemized", "admitted": ["tram", "pigeons"], "order": ["pigeons", "tram"]},
        },
    )
    assert forged.status_code == 400 and "settings.order" in forged.json()["error"]
    unknown = client.post(
        "/api/caption",
        json={**body, "settings": {"level": "itemized", "admitted": ["ghost"], "order": ["ghost"]}},
    )
    assert unknown.status_code == 400 and "settings.admitted" in unknown.json()["error"]
    assert client.post("/api/caption", json={**body, "settings": "itemized"}).status_code == 400
    assert client.post("/api/caption", json={**body, "shot_id": "s9"}).status_code == 404
    assert client.post("/api/caption", json={**body, "clip_id": "demo_zoo"}).status_code == 404
    assert client.post("/api/caption", json={**body, "participant": "P 01"}).status_code == 400
    scene = client.post(
        "/api/caption", json={**body, "settings": {"level": "scene", "admitted": [], "order": []}}
    )
    assert scene.json() == {
        "caption": "A tram stop on a wide street.",
        "key": "scene",
        "cached": False,
        "prefetched": False,
        # Provenance travels with every caption for the log; the page shows neither.
        "writer": "template",
        "names_excluded": [],
    }
    grouped = client.post(
        "/api/caption",
        json={
            **body,
            "settings": {
                "level": "grouped",
                "admitted": ["footsteps", "chatter", "pigeons"],
                "order": ["underneath", "only_here"],
            },
        },
    )
    assert (
        grouped.status_code == 200
        and grouped.json()["key"] == "grouped|underneath,only_here|chatter,footsteps,pigeons"
    )


class SceneFailsWriter(TemplateWriter):
    """Auditions (itemized, grouped) succeed so the app builds; a scene request fails."""

    def write(self, request: CaptionRequest) -> str:
        if request.level == "scene":
            raise WriterError("backend down")
        return super().write(request)


def test_a_writer_failure_is_the_table_6_error(tmp_path: Path, package_files: None) -> None:
    document = _document()
    client = TestClient(
        build_app(document, _fake_media(tmp_path, document), tmp_path / "out", SceneFailsWriter())
    )
    _submit_listing(client, "demo_tram_stop")
    body = {
        "participant": "P01",
        "clip_id": "demo_tram_stop",
        "shot_id": "s1",
        "settings": {"level": "scene"},
    }
    failed = client.post("/api/caption", json=body)
    assert failed.status_code == 502 and failed.json() == {"error": CAPTION_FAILED}
    assert CAPTION_FAILED == "Caption request failed. Check the connection and try again."


def test_a_failed_shot_audio_cut_is_the_same_502(tmp_path: Path, package_files: None) -> None:
    # In Gemma mode the shot's audio is cut before the writer runs; ffmpeg
    # failing there is a writer failure to the participant (contract §7), not
    # an unhandled 500 with a bare status.
    document = _document()

    # Warm-up cuts every shot once at build time; make only the request-time cut fail.
    class CountsPerShot:
        def __init__(self) -> None:
            self.seen: set[tuple[str, str]] = set()

        def __call__(self, clip_id: str, shot: dict[str, Any]) -> Path:
            key = (clip_id, str(shot["shot_id"]))
            if key in self.seen:
                raise MediaError("ffmpeg: no audio stream")
            self.seen.add(key)
            return tmp_path / f"{clip_id}-{shot['shot_id']}.wav"

    client = TestClient(
        build_app(
            document,
            _fake_media(tmp_path, document),
            tmp_path / "out",
            TemplateWriter(),
            shot_media=CountsPerShot(),
        )
    )
    _submit_listing(client, "demo_tram_stop")
    failed = client.post(
        "/api/caption",
        json={
            "participant": "P01",
            "clip_id": "demo_tram_stop",
            "shot_id": "s1",
            "settings": {"level": "scene"},
        },
    )
    assert failed.status_code == 502 and failed.json() == {"error": CAPTION_FAILED}


def test_the_log_downloads_as_an_attachment(client: TestClient) -> None:
    _submit_listing(client, "demo_tram_stop")
    response = client.get("/api/log?participant=P01")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="session-log-P01.json"'
    exported = response.json()
    assert exported["schema"] == "dpo.caption-session-log/v1" and exported["session_id"] == "demo-session"
    assert [event["type"] for event in exported["events"]] == ["listing.submit"]
    assert exported["snapshot"]["screen"] == "shape"
    assert client.get("/api/log?participant=%2E%2E").status_code == 400


def test_clip_media_is_served_with_sound_or_not_at_all(client: TestClient) -> None:
    assert client.get("/media/demo_tram_stop").content == b"video-with-sound"
    assert client.get("/media/demo_zoo").status_code == 404
    assert client.get("/media/still/demo_zoo").status_code == 404
    assert client.get("/media/excerpt/x9").status_code == 404


def _kept_snapshot(document: dict[str, Any]) -> dict[str, Any]:
    kept = {
        clip["clip_id"]: {
            shot["shot_id"]: {
                "caption": f"Own caption for {clip['clip_id']} {shot['shot_id']}.",
                "key": "scene",
            }
            for shot in clip["shots"]
        }
        for clip in document["clips"]
        if clip["task"] == "shaped"
    }
    return {"schema": "dpo.caption-session-snapshot/v1", "kept": kept, "done": True}


def test_the_followup_needs_kept_captions_and_honours_the_a_b_assignment(client: TestClient) -> None:
    assert client.get("/api/followup").status_code == 400
    missing = client.get("/api/followup?participant=P01")
    assert missing.status_code == 409 and missing.json() == {"error": "no kiosk session for participant"}
    document = _document()
    snapshot = _kept_snapshot(document)
    partial = json.loads(json.dumps(snapshot))
    del partial["kept"]["demo_canal"]["s3"]
    client.post("/api/events", json={"participant": "P01", "events": [], "snapshot": partial})
    unfinished = client.get("/api/followup?participant=P01")
    # The participant-facing string is the same; the body names the gap so
    # an unfinished kiosk session can be told from none.
    assert unfinished.status_code == 409
    assert unfinished.json() == {
        "error": "no kiosk session for participant",
        "missing": {"clip_id": "demo_canal", "shot_id": "s3"},
    }
    client.post("/api/events", json={"participant": "P01", "events": [], "snapshot": snapshot})
    followup = client.get("/api/followup?participant=P01").json()
    assert [clip["clip_id"] for clip in followup["clips"]] == [clip["clip_id"] for clip in document["clips"]]
    assert followup["clips"][0]["shots"][0] == {"shot_id": "s1", "start_ms": 0, "end_ms": 4200}
    assert followup["recognition"][0]["sounds"][:2] == ["TRAM BRAKING", "DISTANT SIREN"]
    assert followup["sound_only"] == [{"excerpt_id": "x1"}, {"excerpt_id": "x2"}, {"excerpt_id": "x3"}]
    serialized = json.dumps(followup)
    assert '"own"' not in serialized and '"automatic"' not in serialized and '"a":' not in serialized
    checks = {entry["clip_id"]: entry["captions"] for entry in followup["check"]}
    # demo_tram_stop: A = own; demo_canal: A = automatic (fixture followup.check).
    assert checks["demo_tram_stop"]["A"][0]["caption"] == "Own caption for demo_tram_stop s1."
    assert (
        checks["demo_tram_stop"]["B"][0]["caption"] == document["clips"][0]["shots"][0]["automatic_caption"]
    )
    assert checks["demo_canal"]["A"][2]["caption"] == document["clips"][2]["shots"][2]["automatic_caption"]
    assert checks["demo_canal"]["B"][2]["caption"] == "Own caption for demo_canal s3."
    assert set(checks) == {"demo_tram_stop", "demo_canal", "demo_crossing"}


def _complete_responses(document: dict[str, Any]) -> dict[str, Any]:
    """A follow-up document that answers every task once, as the page sends it."""
    return {
        "schema": "dpo.caption-session-followup/v1",
        "participant": "P01",
        "recognition": [
            {"clip_id": entry["clip_id"], "checked": list(entry["sounds"][:1])}
            for entry in document["followup"]["recognition"]
        ],
        "sound_only": [
            {"excerpt_id": entry["excerpt_id"], "chosen_clip_id": "demo_canal", "played": 2}
            for entry in document["followup"]["sound_only"]
        ],
        "check": [
            {"clip_id": entry["clip_id"], "chosen": "B", "toggles": 3}
            for entry in document["followup"]["check"]
        ],
    }


def test_followup_responses_are_saved_once_and_foreign_ids_refused(
    client: TestClient, tmp_path: Path
) -> None:
    document = _document()
    body = _complete_responses(document)

    def refused(**changes: Any) -> bool:
        return client.post("/api/followup/responses", json={**body, **changes}).status_code == 400

    assert refused(schema="dpo.userstudy-responses/v2")
    assert refused(participant="")
    # Coverage: every clip, excerpt, and checked clip answered exactly once.
    assert refused(recognition=body["recognition"][1:])
    assert refused(sound_only=body["sound_only"] + body["sound_only"][:1])
    assert refused(check=body["check"][:1])
    assert refused(recognition=[{**body["recognition"][0], "clip_id": "demo_zoo"}] + body["recognition"][1:])
    assert refused(sound_only=[{**body["sound_only"][0], "excerpt_id": "x9"}] + body["sound_only"][1:])
    assert refused(check=[{**body["check"][0], "clip_id": "demo_market"}] + body["check"][1:])
    # Shape: answers drawn from what the page offered, counters that count.
    assert refused(
        recognition=[{**body["recognition"][0], "checked": ["NOT A SOUND"]}] + body["recognition"][1:]
    )
    assert refused(
        recognition=[{**body["recognition"][0], "checked": "TRAM BRAKING"}] + body["recognition"][1:]
    )
    assert refused(sound_only=[{**body["sound_only"][0], "chosen_clip_id": None}] + body["sound_only"][1:])
    assert refused(sound_only=[{**body["sound_only"][0], "played": -1}] + body["sound_only"][1:])
    assert refused(sound_only=[{**body["sound_only"][0], "played": "2"}] + body["sound_only"][1:])
    assert refused(check=[{**body["check"][0], "chosen": "C"}] + body["check"][1:])
    assert refused(check=[{**body["check"][0], "toggles": True}] + body["check"][1:])
    assert not (tmp_path / "out" / "followup-P01.json").exists()
    # Saved: each entry rebuilt from the contract's keys, stamped on receipt.
    extra = json.loads(json.dumps(body))
    extra["check"][0]["note"] = "not part of the contract"
    saved = client.post("/api/followup/responses", json=extra)
    assert saved.status_code == 200
    written = json.loads((tmp_path / "out" / "followup-P01.json").read_text())
    assert written["check"] == body["check"] and written["session_id"] == "demo-session"
    assert written["recognition"] == body["recognition"] and written["sound_only"] == body["sound_only"]
    assert written["received_at"].endswith("Z")
    # A second submission (another device, a cleared browser) keeps the first.
    again = client.post(
        "/api/followup/responses",
        json={**body, "check": [{**body["check"][0], "chosen": "A"}] + body["check"][1:]},
    )
    assert again.status_code == 409
    assert json.loads((tmp_path / "out" / "followup-P01.json").read_text())["check"] == body["check"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")
def test_stills_and_excerpts_are_cut_from_the_clip(tmp_path: Path, package_files: None) -> None:
    document = _document()
    media = tmp_path / "media" / "unmuted_video"
    media.mkdir(parents=True)
    # One real one-second clip serves every clip id: the cut is what is tested.
    real = tmp_path / "one-second.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x36:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(real),
        ],
        check=True,
    )
    for clip in document["clips"]:
        shutil.copy(real, media / f"{clip['clip_id']}.mp4")
    client = TestClient(build_app(document, tmp_path / "media", tmp_path / "out", TemplateWriter()))
    still = client.get("/media/still/demo_tram_stop")
    assert still.status_code == 200 and still.headers["content-type"] == "image/jpeg"
    assert still.content[:2] == b"\xff\xd8"
    assert (tmp_path / "out" / "media-cache" / "stills" / "demo_tram_stop.jpg").is_file()
    # x2 is the first 3 s of demo_crossing; the one-second source just makes it short.
    excerpt = client.get("/media/excerpt/x2")
    assert excerpt.status_code == 200 and excerpt.headers["content-type"] == "audio/wav"
    assert excerpt.content[:4] == b"RIFF"
    assert (tmp_path / "out" / "media-cache" / "excerpts" / "x2.wav").is_file()
    assert client.get("/media/excerpt/x2").status_code == 200  # cached, no second cut
    # The control task's writer looks at a still from the MIDDLE of the shot
    # (spec 4.6), cut once per shot and cached by its span.
    from dpo.caption.media import shot_still

    frame = shot_still(tmp_path / "media", tmp_path / "out" / "media-cache", "demo_market", 0, 1000)
    assert frame == tmp_path / "out" / "media-cache" / "stills" / "demo_market-0-1000.jpg"
    assert frame.read_bytes()[:2] == b"\xff\xd8"
    stamp = frame.stat().st_mtime_ns
    assert shot_still(tmp_path / "media", tmp_path / "out" / "media-cache", "demo_market", 0, 1000) == frame
    assert frame.stat().st_mtime_ns == stamp
