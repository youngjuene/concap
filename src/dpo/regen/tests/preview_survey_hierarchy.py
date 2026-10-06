"""Synthetic survey-hierarchy browser QA server; never use for participant studies."""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse

from dpo.regen import progress
from dpo.regen.app import build_app, warm_derivatives
from dpo.regen.continuation import Continuation, ViewingConfig
from dpo.regen.log import EventLog
from dpo.regen.study_store import Conflict
from dpo.regen.tests.preview_readiness import (
    ROOT,
    SyntheticCaptionWriter,
    _legacy_document,
    _viewing_manifest,
)


def main() -> None:
    media = ROOT / "legacy-media"
    viewing_media = ROOT / "viewing-media"
    out = ROOT / "out"
    viewing_out = ROOT / "viewing-out"
    document = _legacy_document(media)
    manifest = _viewing_manifest(viewing_media)
    manifest_path = ROOT / "viewing-study.json"
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
    short_log = EventLog(out, "")

    @continuation.app.get("/api/study/export")
    def export(request: Request) -> JSONResponse:
        token = request.cookies.get(continuation.app.state.cookie_name, "")
        return JSONResponse(continuation.store.export(token))

    @app.post("/qa/complete-short-viewing")
    def complete_short_viewing(request: Request, payload: dict[str, Any]) -> JSONResponse:
        if request.client is None or request.client.host not in {"127.0.0.1", "testclient"}:
            return JSONResponse({"error": "QA endpoint is loopback-only"}, status_code=403)
        participant = payload.get("participant")
        step = payload.get("step")
        clip_index = payload.get("clip_index")
        if (
            not isinstance(participant, str)
            or step not in {"view_prepared", "view_regenerated"}
            or not isinstance(clip_index, int)
            or isinstance(clip_index, bool)
        ):
            return JSONResponse({"error": "participant and viewing step are required"}, status_code=400)
        snapshot = short_log.snapshot(participant)
        if snapshot is None or snapshot.get("protocol_version") != 3:
            return JSONResponse(
                {"error": "QA completion requires a protocol-3 short session"}, status_code=409
            )
        if progress.current(snapshot) != step:
            raise Conflict(f"QA completion belongs to {step}; session is at {progress.current(snapshot)}")
        if clip_index != progress.clip_index(snapshot):
            raise Conflict(
                f"QA completion belongs to clip {clip_index}; session is at {progress.clip_index(snapshot)}"
            )
        order = snapshot["clip_order"]
        segment = order[clip_index]
        duration = document["segments"][segment]["duration_ms"]
        playback = dict(snapshot.get("playback") or {})
        key = f"{clip_index}:{step}"
        previous = dict(playback.get(key) or {})
        sequence = int(previous.get("sequence", -1))
        now = time.time()
        playback[key] = {
            "sequence": sequence,
            "position_ms": duration,
            "coverage": [[0, duration]],
            "epoch": int(previous.get("epoch", 0)),
            "started_at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "last_tick": {
                "position": duration,
                "at": now,
                "playing": False,
                "clock": {"at": now, "covered": duration},
            },
        }
        short_log.write_snapshot(participant, {**snapshot, "playback": playback})
        return JSONResponse(
            {
                "participant": participant,
                "step": step,
                "clip_index": clip_index,
                "sequence": sequence,
                "position_ms": duration,
                "coverage_ms": duration,
            }
        )

    @continuation.app.post("/qa/complete-current-video")
    def complete_current_video(request: Request) -> JSONResponse:
        if request.client is None or request.client.host not in {"127.0.0.1", "testclient"}:
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

    print(json.dumps({"root": str(ROOT), "url": "http://127.0.0.1:18781"}, sort_keys=True), flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=18781, log_level="warning")
    finally:
        if not bool(int(__import__("os").environ.get("REGEN_READINESS_KEEP", "0"))):
            shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
