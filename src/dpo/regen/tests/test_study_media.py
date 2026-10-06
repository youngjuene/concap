"""Public video delivery preserves encoded stimuli and supports cached ranges."""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dpo.regen.study_media import PacedVideoResponse, streamable_video
from dpo.regen.study_schema import fingerprint
from dpo.regen.tests.test_viewing_prepare import _video


def _streams(path: Path) -> bytes:
    return subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_packets",
            "-show_data_hash",
            "sha256",
            "-show_entries",
            "packet=stream_index,pts_time,dts_time,duration_time,data_hash",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
    ).stdout


def test_faststart_keeps_original_and_encoded_streams(tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    _video(source, 2)
    original_hash = fingerprint(source)
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: streamable_video(source, cache), range(2)))
    assert results[0] == results[1] != source
    target = results[0]
    content = target.read_bytes()
    assert content.index(b"moov") < content.index(b"mdat")
    assert _streams(source) == _streams(target)
    assert fingerprint(source) == original_hash
    modified = target.stat().st_mtime_ns
    assert streamable_video(source, cache).stat().st_mtime_ns == modified


def test_non_mp4_keeps_existing_delivery(tmp_path: Path) -> None:
    source = tmp_path / "clip.webm"
    assert streamable_video(source, tmp_path / "cache") == source
    assert not (tmp_path / "cache").exists()


def test_web_copy_keeps_audio_resolution_and_nominal_cadence(tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    _video(source, 2)
    original_hash = fingerprint(source)
    target = streamable_video(source, tmp_path / "cache", video_rate=1_600_000)

    def audio(path: Path) -> list[dict[str, object]]:
        return [p for p in json.loads(_streams(path))["packets"] if p["stream_index"] == 1]

    assert audio(source) == audio(target)

    def frames(path: Path) -> list[dict[str, object]]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time,width,height",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        return [
            {key: value for key, value in frame.items() if key != "side_data_list"}
            for frame in json.loads(result.stdout)["frames"]
        ]

    assert frames(source) == frames(target)
    assert fingerprint(source) == original_hash
    assert target.read_bytes().index(b"moov") < target.read_bytes().index(b"mdat")


def test_paced_video_keeps_range_requests_and_control_routes_responsive(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    content = bytes(range(256)) * 2048
    path.write_bytes(content)
    app = FastAPI()

    @app.get("/video")
    def video() -> PacedVideoResponse:
        return PacedVideoResponse(path, headers={"cache-control": "private, max-age=86400"})

    @app.get("/control")
    def control() -> dict[str, bool]:
        return {"ready": True}

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        transfer = pool.submit(client.get, "/video", headers={"range": "bytes=32768-294911"})
        time.sleep(0.1)
        assert not transfer.done()
        assert client.get("/control").json() == {"ready": True}
        assert not transfer.done()
        response = transfer.result(timeout=5)
    assert response.status_code == 206
    assert response.content == content[32768:294912]
    assert response.headers["content-range"] == "bytes 32768-294911/524288"
    assert response.headers["cache-control"].startswith("private")
