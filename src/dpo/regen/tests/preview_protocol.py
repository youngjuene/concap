"""Synthetic protocol browser-QA server; never use for participant studies."""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse

from dpo.regen.app import build_app, warm_derivatives
from dpo.regen.config import DEFAULT_SCALE_ANCHORS_BY_LANGUAGE, Calibration, Configuration, Scale
from dpo.regen.continuation import Continuation, ViewingConfig
from dpo.regen.document import REGEN_SCHEMA_V4
from dpo.regen.study_store import Conflict
from dpo.regen.tests.preview_readiness import ROOT, SyntheticCaptionWriter, _segment, _viewing_manifest

COLOURS = ("0x425f78", "0x506b42", "0x805344", "0x705080", "0x807044")
STEMS = (
    [("traffic", "Traffic noise", "Sounds of things"), ("bird", "Birds", "Animal")],
    [("siren", "Siren", "Sounds of things"), ("footsteps", "Footsteps", "Human sounds")],
    [("rain", "Rain", "Natural sounds"), ("guitar", "Guitar", "Music")],
    [("speech", "Speech", "Human sounds"), ("wind", "Wind", "Natural sounds")],
    [("dog", "Dog", "Animal"), ("bell", "Bell", "Sounds of things")],
)


def _count(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _document(media: Path, count: int) -> dict[str, Any]:
    configuration = Configuration(
        study_id="protocol-browser",
        corpus_id="synthetic",
        calibration=Calibration(
            cue_slots=4,
            minimum_points=2,
            languages=("en", "ko"),
            scale=Scale(anchors_by_language=DEFAULT_SCALE_ANCHORS_BY_LANGUAGE),
        ),
    )
    names = [f"clip{i + 1}" for i in range(count)]
    return {
        "schema": REGEN_SCHEMA_V4,
        "session_id": "regen-protocol",
        "config": configuration.artifact(),
        "clip_order": names,
        "segments": {
            name: _segment(media, name, COLOURS[index % len(COLOURS)], STEMS[index % len(STEMS)])
            for index, name in enumerate(names)
        },
    }


def _long_manifest(root: Path, count: int) -> dict[str, Any]:
    manifest = _viewing_manifest(root)
    source_videos = deepcopy(manifest["viewing_videos"])
    videos: list[dict[str, Any]] = []
    for index in range(count):
        source = deepcopy(source_videos[index % len(source_videos)])
        source["id"] = f"long-{index}"
        source["title"] = f"Long protocol video {index + 1}"
        if index >= len(source_videos):
            original = root / source_videos[index % len(source_videos)]["video"]
            target = root / f"long-{index}.webm"
            shutil.copyfile(original, target)
            source["video"] = target.name
        videos.append(source)
    manifest["viewing_videos"] = videos
    return manifest


def _loopback(request: Request) -> bool:
    return request.client is not None and request.client.host in {"127.0.0.1", "testclient"}


def main() -> None:
    short_count = _count("REGEN_PROTOCOL_SHORT_COUNT", 1)
    long_count = _count("REGEN_PROTOCOL_LONG_COUNT", 2)
    media = ROOT / "protocol-short-media"
    viewing_media = ROOT / "protocol-long-media"
    out = ROOT / "protocol-out"
    viewing_out = ROOT / "protocol-viewing-out"
    document = _document(media, short_count)
    manifest = _long_manifest(viewing_media, long_count)
    manifest_path = ROOT / "protocol-viewing-study.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    warm_derivatives(document, media)
    app = build_app(
        document,
        media,
        out,
        SyntheticCaptionWriter(),
        derive=True,
        viewing=ViewingConfig(manifest_path, viewing_media, viewing_out, inference_timeout=5),
    )
    continuation: Continuation = app.state.continuation

    @app.get("/qa/protocol-log/{participant}")
    def protocol_log(request: Request, participant: str) -> JSONResponse:
        if not _loopback(request):
            return JSONResponse({"error": "QA endpoint is loopback-only"}, status_code=403)
        result: dict[str, Any] = {}
        for prefix in ("snapshot", "events", "viewings", "responses"):
            path = out / f"{prefix}-{participant}.jsonl"
            if prefix == "snapshot":
                path = out / f"snapshot-{participant}.json"
            if not path.exists():
                result[prefix] = [] if path.suffix == ".jsonl" else None
                continue
            if path.suffix == ".jsonl":
                result[prefix] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            else:
                result[prefix] = json.loads(path.read_text(encoding="utf-8"))
        return JSONResponse(result)

    @continuation.app.get("/api/study/export")
    def export(request: Request) -> JSONResponse:
        token = request.cookies.get(continuation.app.state.cookie_name, "")
        return JSONResponse(continuation.store.export(token))

    @continuation.app.post("/qa/complete-current-video")
    def complete_current_video(request: Request) -> JSONResponse:
        if not _loopback(request):
            return JSONResponse({"error": "QA endpoint is loopback-only"}, status_code=403)
        token = request.cookies.get(continuation.app.state.cookie_name, "")
        updated: dict[str, Any] | None = None
        for _ in range(5):
            state = continuation.store.state(token)
            if state["stage"] != "watch":
                raise Conflict(f"QA completion belongs to watch; session is at {state['stage']}")
            video = state["viewing"][state["video_index"]]
            duration = video["duration_ms"]
            request_key = f"qa-complete-video-{state['video_index']}-{state['revision']}"

            def complete(draft: dict[str, Any], video_duration: int = duration) -> None:
                if draft["stage"] != "watch":
                    raise Conflict(f"QA completion belongs to watch; session is at {draft['stage']}")
                draft["coverage"] = [[0, video_duration]]
                draft["position_ms"] = video_duration
                draft["sequence"] = max(int(draft.get("sequence", -1)), 0) + 1
                now = time.time()
                draft["last_tick"] = {
                    "position": video_duration,
                    "at": now,
                    "playing": False,
                    "clock": {"at": now, "covered": video_duration},
                }

            try:
                updated = continuation.store.mutate(
                    token, request_key, state["revision"], request_key, complete
                )
                break
            except Conflict as exc:
                if str(exc) != "Session changed; reload its current state":
                    raise
        if updated is None:
            raise Conflict("Session changed; reload its current state")
        return JSONResponse(
            {
                "stage": updated["stage"],
                "revision": updated["revision"],
                "video_index": updated["video_index"],
                "position_ms": updated["position_ms"],
                "coverage_ms": sum(end - start for start, end in updated["coverage"]),
            }
        )

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    print(
        json.dumps(
            {
                "root": str(ROOT),
                "url": f"http://127.0.0.1:{port}",
                "short_count": short_count,
                "long_count": long_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        if not bool(int(os.environ.get("REGEN_PROTOCOL_KEEP", "0"))):
            shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
