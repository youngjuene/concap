"""Serve the v2 study: python -m dpo.regen.study_api --help."""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from dpo.regen.config import SOUND_FAMILIES
from dpo.regen.points import MaskObject, match_points, matched_labels, parse_points
from dpo.regen.study_schema import (
    FINAL_ITEMS,
    PREFERENCES,
    answers,
    bound_media,
    caption_instruction,
    compile_profile,
    digest,
    fingerprint,
    load_manifest,
    number,
)
from dpo.regen.study_store import Conflict, StudyStore
from dpo.regen.study_worker import InferenceProcess, Supervisor


def require(state: dict[str, Any], stage: str) -> None:
    if state["stage"] != stage:
        raise Conflict(f"This action belongs to {stage}; session is at {state['stage']}")


def merged(intervals: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def record_playback(state: dict[str, Any], data: dict[str, Any], duration: float) -> None:
    sequence = data.get("sequence")
    if type(sequence) is not int or sequence <= state["sequence"]:
        raise Conflict("Playback update is out of order")
    position = number(data.get("position_ms"), 0, duration)
    now, previous = time.time(), state.get("last_tick")
    if data.get("seek") is True:
        state["epoch"] += 1
    elif previous and previous["playing"] and not data.get("hidden", False):
        delta = position - previous["position"]
        elapsed = (now - previous["at"]) * 1000
        if 0 <= delta <= min(7000, elapsed * 1.25 + 300):
            state["coverage"] = merged(state["coverage"] + [[previous["position"], position]])
    state["position_ms"], state["sequence"] = position, sequence
    state["last_tick"] = {
        "position": position,
        "at": now,
        "playing": data.get("playing") is True and not data.get("hidden", False),
    }


def build_study_app(
    manifest_path: Path,
    media_dir: Path,
    out_dir: Path,
    model: dict[str, Any] | None = None,
    access_code: str | None = None,
    inference_timeout: float = 30,
    *,
    linked_only: bool = False,
    engine: InferenceProcess | None = None,
) -> FastAPI:
    manifest = load_manifest(manifest_path, media_dir)
    calibration_hash = digest(
        {"clips": manifest["calibration_clips"], "language": manifest["language"], "items": PREFERENCES}
    )
    store = StudyStore(out_dir / "study.sqlite3")
    cookie_name = f"caption_study_{digest(str(out_dir.resolve()))[:12]}"
    (out_dir / "excerpts").mkdir(exist_ok=True)
    model = model or {}
    number(inference_timeout, 0.1, 600)
    model_identity: dict[str, Any] = {
        "settings": model,
        "implementation": digest(
            [files("dpo.regen").joinpath(name).read_text() for name in ("study_schema.py", "study_worker.py")]
        ),
    }
    for field in ("backend_config", "contract"):
        if model.get(field):
            model_identity[field] = fingerprint(Path(model[field]))
    if model.get("checkpoint"):
        checkpoint = Path(model["checkpoint"])
        model_identity["checkpoint_files"] = {
            str(path.relative_to(checkpoint)): fingerprint(path)
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file()
        }
    supervisor = Supervisor(store, model, inference_timeout, engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        with (out_dir / "study.lock").open("a") as ownership:
            try:
                fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This study output is already served by another process") from exc
            supervisor.start()
            try:
                yield
            finally:
                supervisor.stop()

    app = FastAPI(
        title="Sound captions — two-stage study",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store
    app.state.calibration_hash = calibration_hash
    app.state.cookie_name = cookie_name

    @app.middleware("http")
    async def boundary(request: Request, call_next: Any) -> Response:
        if request.method == "POST" and request.headers.get("x-study-request") != "1":
            return JSONResponse({"error": "Same-origin study request required"}, status_code=403)
        response: Response = await call_next(request)
        response.headers["cache-control"] = "no-store"
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "same-origin"
        return response

    @app.exception_handler(ValueError)
    async def invalid(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=409 if isinstance(exc, Conflict) else 400)

    @app.exception_handler(PermissionError)
    async def unauthenticated(_: Request, exc: PermissionError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=401)

    def person(request: Request) -> tuple[str, dict[str, Any]]:
        token = request.cookies.get(cookie_name, "")
        state = store.state(token)
        if linked_only and request.path_params.get("continuation_id") != state["session_id"]:
            raise PermissionError("Continue from your calibration survey.")
        if state["calibration_hash"] != calibration_hash:
            raise Conflict("The calibration configuration changed; contact the researcher")
        return token, state

    def present(state: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: state[key]
            for key in (
                "revision",
                "session_id",
                "stage",
                "clip_index",
                "video_index",
                "axes",
                "settings_revision",
                "epoch",
                "position_ms",
                "sequence",
                "draft",
                "language",
            )
        }
        result["calibration_count"] = len(manifest["calibration_clips"])
        result["families"] = list(SOUND_FAMILIES)
        result["items"] = PREFERENCES if state["stage"] == "preferences" else FINAL_ITEMS
        result["completed_videos"] = len(state["completions"])
        if state["stage"] in ("clip", "observe"):
            clip = manifest["calibration_clips"][state["clip_index"]]
            result["clip"] = {
                "id": clip["id"],
                "title": clip.get("title", "Calibration clip"),
                "duration_ms": clip["duration_ms"],
                "video": f"/study/media/calibration/{state['clip_index']}",
                "frames": [
                    {"at_ms": frame["at_ms"], "url": f"/study/frame/{state['clip_index']}/{i}"}
                    for i, frame in enumerate(clip["frames"])
                ],
            }
        if "profile" in state:
            result["defaults"] = state["profile"]["defaults"]
        if state["stage"] == "watch":
            video = state["viewing"][state["video_index"]]
            result["video"] = {
                "id": video["id"],
                "title": video.get("title", "Your viewing experience"),
                "duration_ms": video["duration_ms"],
                "url": f"/study/media/watch/{state['video_index']}",
                "cues": [
                    {
                        "start_ms": cue["start_ms"],
                        "end_ms": cue["end_ms"],
                        "fallback": cue["fallback"][state["language"]],
                    }
                    for cue in video["cues"]
                ],
            }
        return result

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return files("dpo.regen").joinpath("study.html").read_text()

    @app.get("/api/study/strings")
    def strings() -> dict[str, str]:
        return dict(json.loads(files("dpo.regen").joinpath("study-ko.json").read_text()))

    @app.post("/api/study/session")
    def session(request: Request, payload: dict[str, Any]) -> Response:
        token = request.cookies.get(cookie_name)
        if token:
            _, state = person(request)
        else:
            if linked_only:
                raise PermissionError("Continue from your calibration survey.")
            if access_code and payload.get("code") != access_code:
                return JSONResponse(
                    {"error": "Enter the study access code", "code_required": True}, status_code=403
                )
            token = store.create(calibration_hash, manifest["language"])
            state = store.state(token)
        response = JSONResponse(present(state))
        response.set_cookie(
            cookie_name,
            token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=60 * 60 * 24 * 30,
            path=(request.scope.get("root_path", "") + "/") if linked_only else "/",
        )
        return response

    @app.get("/api/study/state")
    def current(request: Request) -> dict[str, Any]:
        return present(person(request)[1])

    def schedule(token: str, state: dict[str, Any]) -> None:
        if state["stage"] != "watch":
            return
        video = state["viewing"][state["video_index"]]
        current_index = next(
            (i for i, c in enumerate(video["cues"]) if c["start_ms"] <= state["position_ms"] < c["end_ms"]),
            None,
        )
        if current_index is None:
            return
        for index in range(current_index, min(current_index + 2, len(video["cues"]))):
            cue = video["cues"][index]
            instruction = caption_instruction(state["profile"], state["axes"], cue)
            content = {
                "video": digest(video),
                "model": model_identity,
                "instruction": instruction,
                "start": cue["start_ms"],
                "end": cue["end_ms"],
                "version": 1,
            }
            key = digest(content)
            spec = {
                "model_identity": model_identity,
                "profile_hash": state["profile"]["hash"],
                "video_hash": digest(video),
                "instruction": instruction,
                "audio": str(bound_media(media_dir, video, "audio")),
                "excerpt": str(out_dir / "excerpts" / f"{digest([digest(video), index])}.wav"),
                "start_ms": cue["start_ms"],
                "end_ms": cue["end_ms"],
                "language": state["language"],
                "axes": state["axes"],
                "fallback": cue["fallback"][state["language"]],
                "queue_timeout": 15,
                "current_index": current_index,
            }
            store.enqueue(token, state, index, key, spec, limit=32)

    @app.post("/api/study/{action}")
    def action(request: Request, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        token, _ = person(request)
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("Action data must be an object")
        if len(json.dumps(data)) > 128000:
            raise ValueError("Study action is too large")
        prepared = load_manifest(manifest_path, media_dir) if action == "start-viewing" else None

        def apply(state: dict[str, Any]) -> None:
            if action == "preferences":
                require(state, "preferences")
                state["preferences"] = answers(data, PREFERENCES)
                state["stage"], state["draft"] = "clip", {}
            elif action == "clip-ended":
                require(state, "clip")
                clip = manifest["calibration_clips"][state["clip_index"]]
                covered = sum(end - start for start, end in state["coverage"])
                if covered < clip["duration_ms"] - 500 or state["position_ms"] < clip["duration_ms"] - 250:
                    raise ValueError("Watch the calibration clip before continuing")
                state.setdefault("calibration_viewings", []).append(
                    {"clip": clip["id"], "coverage": state["coverage"]}
                )
                state["position_ms"], state["sequence"], state["coverage"], state["last_tick"] = (
                    0,
                    -1,
                    [],
                    None,
                )
                state["stage"] = "observe"
            elif action == "clip-playback":
                require(state, "clip")
                clip = manifest["calibration_clips"][state["clip_index"]]
                if data.get("clip_id") != clip["id"]:
                    raise Conflict("Playback belongs to another calibration clip")
                record_playback(state, data, clip["duration_ms"])
            elif action == "observation":
                require(state, "observe")
                clip = manifest["calibration_clips"][state["clip_index"]]
                heard = data.get("heard")
                if (
                    not isinstance(heard, dict)
                    or set(heard) != set(SOUND_FAMILIES)
                    or any(type(value) is not bool for value in heard.values())
                ):
                    raise ValueError("Answer heard or did not hear for every sound family")
                points = parse_points(data.get("points"), len(clip["frames"]))
                if not 1 <= len(points) <= 100:
                    raise ValueError("Mark between 1 and 100 points")
                objects = {
                    i: tuple(
                        MaskObject(obj["id"], obj["label"], bound_media(media_dir, obj, "mask"))
                        for obj in frame.get("objects", [])
                    )
                    for i, frame in enumerate(clip["frames"])
                }
                matches = match_points(points, objects)
                state["observations"].append(
                    {
                        "clip": clip["id"],
                        "heard": heard,
                        "points": [match.record() for match in matches],
                        "labels": list(matched_labels(matches)),
                        "present": clip.get("present_families"),
                    }
                )
                state["clip_index"] += 1
                state["draft"] = {}
                if state["clip_index"] == len(manifest["calibration_clips"]):
                    state["profile"] = compile_profile(
                        state["preferences"], state["observations"], state["language"]
                    )
                    state["axes"] = state["profile"]["defaults"].copy()
                    state["stage"] = "ready"
                else:
                    state["stage"] = "clip"
            elif action == "start-viewing":
                require(state, "ready")
                assert prepared is not None
                if any(video["status"] != "ready" for video in prepared["viewing_videos"]):
                    raise ValueError("Your calibration is saved. The three viewing videos are not ready yet.")
                if (
                    digest(
                        {
                            "clips": prepared["calibration_clips"],
                            "language": prepared["language"],
                            "items": PREFERENCES,
                        }
                    )
                    != calibration_hash
                ):
                    raise Conflict("Calibration changed while preparing viewing media")
                if state["language"] not in prepared.get("languages", [prepared["language"]]):
                    raise ValueError("Viewing captions are not prepared in your language yet.")
                state["viewing"] = prepared["viewing_videos"]
                state["viewing_hash"] = digest(state["viewing"])
                state["stage"] = "watch"
                state["last_tick"] = None
            elif action in ("settings", "playback", "exposures", "video-ended"):
                require(state, "watch")
                video = state["viewing"][state["video_index"]]
                if data.get("video_id") != video["id"]:
                    raise Conflict("This update belongs to a different video")
                if action == "settings":
                    state["axes"] = {
                        key: round(number(data.get(key), 0, 1), 2) for key in ("texture", "context")
                    }
                    state["settings_revision"] += 1
                elif action == "playback":
                    record_playback(state, data, video["duration_ms"])
                elif action == "exposures":
                    entries = data.get("entries")
                    if not isinstance(entries, list) or len(entries) > 100:
                        raise ValueError("Exposure batches contain at most 100 entries")
                    batch = []
                    for entry in entries:
                        if not isinstance(entry, dict):
                            raise ValueError("Each exposure must be an object")
                        key = entry.get("id")
                        if not isinstance(key, str) or not 1 <= len(key) <= 100:
                            raise ValueError("Exposure ID required")
                        start = number(entry.get("start_ms"), 0, video["duration_ms"])
                        end = number(entry.get("end_ms"), start, video["duration_ms"])
                        cue_index = entry.get("cue")
                        if type(cue_index) is not int or not 0 <= cue_index < len(video["cues"]):
                            raise ValueError("Exposure must name a video cue")
                        cue = video["cues"][cue_index]
                        if not cue["start_ms"] <= start <= end <= cue["end_ms"]:
                            raise ValueError("Exposure crosses its cue boundary")
                        if len(json.dumps(entry)) > 3000:
                            raise ValueError("Exposure record too large")
                        batch.append({**entry, "start_ms": start, "end_ms": end, "video_id": video["id"]})
                    state["_exposure_batch"] = batch
                else:
                    covered = sum(end - start for start, end in state["coverage"])
                    if (
                        covered < video["duration_ms"] - 1500
                        or state["position_ms"] < video["duration_ms"] - 500
                    ):
                        raise ValueError(
                            "Please watch the full video before continuing. Replay any skipped sections."
                        )
                    state["completions"].append({"video_id": video["id"], "coverage": state["coverage"]})
                    state["video_index"] += 1
                    state["stage"] = "final" if state["video_index"] == 3 else "break"
                    state["epoch"] += 1
                    state["position_ms"], state["sequence"], state["coverage"], state["last_tick"] = (
                        0,
                        -1,
                        [],
                        None,
                    )
            elif action == "continue":
                require(state, "break")
                state["stage"] = "watch"
            elif action == "draft":
                if state["stage"] not in ("preferences", "observe", "final"):
                    raise Conflict("No form is active")
                if len(json.dumps(data)) > 20000:
                    raise ValueError("Draft too large")
                state["draft"] = data
            elif action == "final-survey":
                require(state, "final")
                state["final_survey"] = {
                    "answers": answers(data, FINAL_ITEMS),
                    "items_hash": digest(FINAL_ITEMS),
                    "profile_hash": state["profile"]["hash"],
                    "viewing_hash": state["viewing_hash"],
                }
                state["stage"], state["draft"] = "done", {}
            else:
                raise ValueError("Unknown study action")

        state = store.mutate(
            token,
            payload.get("key", ""),
            payload.get("revision", -1),
            json.dumps({"action": action, "data": data}, sort_keys=True),
            apply,
        )
        schedule(token, store.state(token))
        return present(state)

    @app.get("/api/study/captions")
    def captions(request: Request) -> dict[str, Any]:
        token, state = person(request)
        jobs = store.captions(token, state)
        for job in jobs:
            if job["result"]:
                job["result"] = {key: job["result"].get(key) for key in ("text", "fallback", "reason")}
        return {
            "epoch": state["epoch"],
            "revision": state["settings_revision"],
            "jobs": jobs,
        }

    @app.get("/study/media/{kind}/{index}")
    def media(request: Request, kind: str, index: int) -> FileResponse:
        _, state = person(request)
        entries = (
            manifest["calibration_clips"]
            if kind == "calibration"
            else state.get("viewing", [])
            if kind == "watch"
            else []
        )
        if not 0 <= index < len(entries):
            raise ValueError("Unknown media")
        return FileResponse(bound_media(media_dir, entries[index], "video"))

    @app.get("/study/frame/{clip}/{index}")
    def frame(request: Request, clip: int, index: int) -> FileResponse:
        person(request)
        entries = manifest["calibration_clips"]
        if not 0 <= clip < len(entries) or not 0 <= index < len(entries[clip]["frames"]):
            raise ValueError("Unknown frame")
        return FileResponse(bound_media(media_dir, entries[clip]["frames"][index], "still"))

    @app.get("/{asset}")
    def asset(asset: str) -> Response:
        allowed = {
            "identity.css": "text/css",
            "regen.css": "text/css",
            "study.css": "text/css",
            "study.js": "text/javascript",
        }
        if asset not in allowed:
            return Response(status_code=404)
        return Response(files("dpo.regen").joinpath(asset).read_text(), media_type=allowed[asset])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--backend-config")
    parser.add_argument("--contract")
    parser.add_argument("--checkpoint")
    parser.add_argument("--access-code")
    parser.add_argument("--inference-timeout", type=float, default=30)
    args = parser.parse_args()
    if args.backend_config and not args.contract:
        parser.error("--backend-config requires --contract")
    import uvicorn

    app = build_study_app(
        args.manifest.resolve(),
        args.media.resolve(),
        args.out.resolve(),
        {key: getattr(args, key) for key in ("backend_config", "contract", "checkpoint")},
        args.access_code,
        args.inference_timeout,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
