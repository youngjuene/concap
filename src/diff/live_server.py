"""The live phase-2 participant app (plan "Live phase 2", researcher's design of 2026-10-11).

Pages, in order:
1. ready: the calibrated settings are written as Trida's rewrite of the first
   cues before playback, and Start waits for cue 0's caption;
2. watch: the video plays, the two sliders can be moved at any time, and the
   caption changes only at cue boundaries (the previous caption stays when
   nothing new is ready);
3. the next video (settings carry over), or the researcher's rating items once
   the last video ends;
4. done.

Every response the browser can fetch carries opaque tokens, caption text and
timing only. Job IDs, the model, the decoding condition, seeds, step names and
media file names never leave the server (plan §6.5). Participant-facing text is
Korean.

FastAPI is imported inside :func:`build_live_app`, so importing this module
pulls in no web stack and no model code. Session state lives in memory, with an
append-only event log per session. A server restart ends the sessions in
progress; their logs remain.
"""

# No ``from __future__ import annotations`` here: FastAPI reads the handlers'
# annotations at runtime, and ``Request`` is imported inside build_live_app so
# that importing this module pulls in no web stack.
import json
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from diff.caption_sources import CaptionSource
from diff.job_plan import Job
from diff.live_calibration import Calibration
from diff.live_generation import LiveScheduler, LiveService, LiveStream, Spec, opaque_token, slider_steps

EVENT_SCHEMA: Final = "diff.live-event/v1"
COOKIE: Final = "diff_live"
STATIC: Final = Path(__file__).with_name("live_static")
BUSY_MESSAGE: Final = "지금은 다른 참가자가 참여하고 있습니다. 잠시 후 다시 시도해 주세요."
ORIGINS: Final = ("slider:texture", "slider:context", "reset")


@dataclass(frozen=True)
class LiveVideo:
    video_id: str
    media: Path
    stream: LiveStream


@dataclass(frozen=True)
class RatingItem:
    """One researcher-authored item, answered after the watch (Korean text)."""

    item_id: str
    text: str
    scale_min: int
    scale_max: int
    anchors: tuple[str, str]

    def record(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "text": self.text,
            "min": self.scale_min,
            "max": self.scale_max,
            "anchors": list(self.anchors),
        }


def rating_items(document: Mapping[str, Any]) -> tuple[RatingItem, ...]:
    """``{"items": [{"id", "text", "min", "max", "anchors": [low, high]}]}``, every field required."""
    items = []
    for index, raw in enumerate(document.get("items", [])):
        try:
            item = RatingItem(
                str(raw["id"]),
                str(raw["text"]),
                int(raw["min"]),
                int(raw["max"]),
                (str(raw["anchors"][0]), str(raw["anchors"][1])),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"items[{index}]: {exc}") from exc
        if not item.item_id or not item.text or item.scale_min >= item.scale_max:
            raise ValueError(f"items[{index}]: needs an id, a text and min < max")
        items.append(item)
    if not items or len({item.item_id for item in items}) != len(items):
        raise ValueError("the rating items need at least one item and unique ids")
    return tuple(items)


class LivePool:
    """The live services, one per GPU; a session takes a free one or is refused."""

    def __init__(self, services: Sequence[LiveService]) -> None:
        self.services = list(services)
        self.lock = threading.Lock()

    def acquire(self) -> LiveService | None:
        with self.lock:
            return next((service for service in self.services if service.free), None)


@dataclass
class LiveSession:
    session_id: str
    service: LiveService
    calibration: Calibration
    stage: str = "ready"
    video_index: int = 0
    texture: float = 0.5
    context: float = 0.5
    answers: dict[str, int] = field(default_factory=dict)
    tokens: dict[str, str] = field(default_factory=dict)  # every token served, to its job ID


JobFor = Callable[[CaptionSource, Spec], Job]


def build_live_app(
    videos: Sequence[LiveVideo],
    items: Sequence[RatingItem],
    pool: LivePool,
    log_dir: Path,
    *,
    job_for: JobFor,
    calibrate: Callable[[str | None], Calibration],
    secret: bytes | None = None,
    clock: Callable[[], float] = time.time,
) -> Any:
    """The FastAPI app; ``calibrate(code)`` returns a session's starting point."""
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    if not videos:
        raise ValueError("the live app needs at least one video")
    key = secret if secret is not None else secrets.token_bytes(32)
    sessions: dict[str, LiveSession] = {}
    lock = threading.Lock()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def log(session: LiveSession, event: str, **data: Any) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": EVENT_SCHEMA,
            "at": clock(),
            "session_id": session.session_id,
            "event": event,
            **data,
        }
        with (log_dir / f"session-{session.session_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def current(request: Request) -> LiveSession:
        session = sessions.get(request.cookies.get(COOKIE, ""))
        if session is None:
            raise HTTPException(status_code=401, detail="세션이 없습니다. 처음부터 다시 시작해 주세요.")
        return session

    def video_of(session: LiveSession) -> LiveVideo:
        return videos[session.video_index]

    def spec_of(session: LiveSession) -> Spec:
        texture, context = slider_steps(session.texture, session.context)
        return Spec(texture, context, session.calibration.guidance)

    def open_video(session: LiveSession) -> None:
        """Attach the video's scheduler and write its first cues at the current settings."""
        scheduler = LiveScheduler(video_of(session).stream, job_for)
        session.service.attach(session.session_id, scheduler)
        session.service.settings(spec_of(session), 0)

    def view(session: LiveSession) -> dict[str, Any]:
        video = video_of(session)
        return {
            "stage": session.stage,
            "video": {
                "number": session.video_index + 1,
                "count": len(videos),
                "media": f"media/{opaque_token(key, session.session_id, video.video_id)}",
                "cues": [{"start_ms": cue.start_ms, "end_ms": cue.end_ms} for cue in video.stream.cues],
            },
            "sliders": {"texture": session.texture, "context": session.context},
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "live.html").read_text(encoding="utf-8")

    @app.get("/live.js")
    def script() -> Response:
        return Response((STATIC / "live.js").read_text(encoding="utf-8"), media_type="text/javascript")

    @app.post("/api/session")
    async def start_session(request: Request) -> Response:
        body = await request.json() if await request.body() else {}
        existing = sessions.get(request.cookies.get(COOKIE, ""))
        if existing is not None:
            return JSONResponse(view(existing))
        service = pool.acquire()
        if service is None:
            return JSONResponse({"stage": "busy", "message": BUSY_MESSAGE}, status_code=503)
        calibration = calibrate(body.get("code") if isinstance(body, Mapping) else None)
        session = LiveSession(
            secrets.token_urlsafe(18),
            service,
            calibration,
            texture=calibration.texture,
            context=calibration.context,
        )
        with lock:
            sessions[session.session_id] = session
        log(session, "session_started", calibration=dict(calibration.record))
        open_video(session)
        response = JSONResponse(view(session))
        response.set_cookie(COOKIE, session.session_id, httponly=True, samesite="strict")
        return response

    @app.get("/api/state")
    def state(request: Request) -> dict[str, Any]:
        return view(current(request))

    @app.get("/media/{token}")
    def media(token: str, request: Request) -> FileResponse:
        session = current(request)
        video = video_of(session)
        if not secrets.compare_digest(token, opaque_token(key, session.session_id, video.video_id)):
            raise HTTPException(status_code=404)
        return FileResponse(
            video.media, media_type="video/webm" if video.media.suffix == ".webm" else "video/mp4"
        )

    @app.post("/api/start")
    def start_watch(request: Request) -> dict[str, Any]:
        session = current(request)
        if session.stage != "ready":
            raise HTTPException(status_code=409, detail="지금은 시청을 시작할 수 없습니다.")
        if 0 not in session.service.ready():
            raise HTTPException(status_code=409, detail="첫 자막을 준비하고 있습니다.")
        session.stage = "watch"
        log(session, "watch_started", video_index=session.video_index)
        return view(session)

    @app.post("/api/settings")
    async def settings(request: Request) -> dict[str, Any]:
        session = current(request)
        body = await request.json()
        try:
            texture, context = float(body["texture"]), float(body["context"])
            position, origin = int(body["position_ms"]), str(body["origin"])
            slider_steps(texture, context)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="설정 값을 읽을 수 없습니다.") from exc
        if origin not in ORIGINS:
            raise HTTPException(status_code=422, detail="설정 값을 읽을 수 없습니다.")
        session.texture, session.context = texture, context
        spec = spec_of(session)
        session.service.settings(spec, position)
        log(
            session,
            "settings",
            video_index=session.video_index,
            texture=texture,
            context=context,
            steps={"texture": spec.texture, "context": spec.context},
            position_ms=position,
            origin=origin,
        )
        return {"ok": True}

    @app.post("/api/playback")
    async def playback(request: Request) -> dict[str, Any]:
        session = current(request)
        body = await request.json()
        try:
            position, playing = int(body["position_ms"]), bool(body["playing"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="재생 정보를 읽을 수 없습니다.") from exc
        session.service.playback(position, playing)
        log(session, "playback", video_index=session.video_index, position_ms=position, playing=playing)
        return {"ok": True}

    @app.get("/api/captions")
    def captions(request: Request) -> dict[str, Any]:
        session = current(request)
        ready = session.service.ready() if session.stage in ("ready", "watch") else {}
        served = {}
        for cue, entry in ready.items():
            token = opaque_token(key, session.session_id, entry["job_id"])
            session.tokens[token] = entry["job_id"]
            served[str(cue)] = {"token": token, "text": entry["text"]}
        return {"captions": served}

    @app.post("/api/exposures")
    async def exposures(request: Request) -> dict[str, Any]:
        session = current(request)
        body = await request.json()
        entries = body.get("entries") if isinstance(body, Mapping) else None
        if not isinstance(entries, list):
            raise HTTPException(status_code=422, detail="표시 기록을 읽을 수 없습니다.")
        resolved = [
            {**entry, "job_id": session.tokens.get(str(entry.get("token")))}
            for entry in entries
            if isinstance(entry, Mapping)
        ]
        log(session, "exposures", video_index=session.video_index, entries=resolved)
        return {"ok": True}

    @app.post("/api/video-ended")
    def video_ended(request: Request) -> dict[str, Any]:
        session = current(request)
        if session.stage != "watch":
            raise HTTPException(status_code=409, detail="지금은 영상을 마칠 수 없습니다.")
        log(session, "video_ended", video_index=session.video_index)
        session.service.detach(session.session_id)
        if session.video_index + 1 < len(videos):
            session.video_index += 1
            session.stage = "ready"
            open_video(session)
        else:
            session.stage = "ratings"
        return view(session)

    @app.get("/api/items")
    def get_items(request: Request) -> dict[str, Any]:
        current(request)
        return {"items": [item.record() for item in items]}

    @app.post("/api/ratings")
    async def ratings(request: Request) -> dict[str, Any]:
        session = current(request)
        if session.stage != "ratings":
            raise HTTPException(status_code=409, detail="지금은 응답할 수 없습니다.")
        body = await request.json()
        answers = body.get("answers") if isinstance(body, Mapping) else None
        if not isinstance(answers, Mapping) or set(answers) != {item.item_id for item in items}:
            raise HTTPException(status_code=422, detail="모든 문항에 답해 주세요.")
        for item in items:
            value = answers[item.item_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not item.scale_min <= value <= item.scale_max
            ):
                raise HTTPException(status_code=422, detail="응답 범위를 확인해 주세요.")
        session.answers = {item.item_id: int(answers[item.item_id]) for item in items}
        session.stage = "done"
        log(session, "ratings", answers=session.answers)
        return {"stage": "done"}

    return app
