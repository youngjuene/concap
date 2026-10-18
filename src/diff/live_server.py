"""The live phase-2 routes (plan "Live phase 2", researcher's design of 2026-10-11).

Pages, in order:
1. ready: the calibrated settings are written as Trida's rewrite of the first
   cues before playback, and Start waits for cue 0's caption;
2. watch: the video plays, the two sliders can be moved at any time, and the
   caption changes only at cue boundaries (the previous caption stays when
   nothing new is ready);
3. the next video (settings carry over), or the researcher's rating items once
   the last video ends;
4. done.

Since 2026-10-18 the participant app (:mod:`diff.participant_server`) owns the
session:
- the experimenter's entry claims a GPU;
- phase 1 runs;
- the handoff builds the calibration.

These routes serve the watch under ``/watch/`` for the session that app hands
them through ``current``, and report each change through ``changed`` so that
it is stored.

Every response the browser can fetch carries opaque tokens, caption text and
timing only. Job IDs, the model, the decoding condition, step names and media
file names never leave the server (plan §6.5). Participant-facing text is
Korean.

FastAPI is imported inside :func:`add_live_routes`, so importing this module
pulls in no web stack and no model code.
"""

# No ``from __future__ import annotations`` here: FastAPI reads the handlers'
# annotations at runtime, and ``Request`` is imported inside add_live_routes so
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
STATIC: Final = Path(__file__).with_name("live_static")
WATCH: Final = "/watch"
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
    """The live services, one per GPU; a session claims a free one at entry or is refused."""

    def __init__(self, services: Sequence[LiveService]) -> None:
        self.services = list(services)
        self.lock = threading.Lock()

    def holder(self, session_id: str) -> LiveService | None:
        return next((service for service in self.services if service.session_id == session_id), None)

    def acquire(self, session_id: str) -> LiveService | None:
        """The GPU this session holds, else a free one now claimed for it, else ``None``."""
        with self.lock:
            held = self.holder(session_id)
            if held is not None:
                return held
            return next((service for service in self.services if service.claim(session_id)), None)

    def release(self, session_id: str) -> None:
        with self.lock:
            for service in self.services:
                service.detach(session_id)


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


def live_event(log_dir: Path, at: float, session_id: str, event: str, **data: Any) -> None:
    """Append one ``diff.live-event/v1`` record to the session's log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {"schema": EVENT_SCHEMA, "at": at, "session_id": session_id, "event": event, **data}
    with (log_dir / f"session-{session_id}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def spec_of(session: LiveSession) -> Spec:
    texture, context = slider_steps(session.texture, session.context)
    return Spec(texture, context, session.calibration.system_message)


def open_video(session: LiveSession, videos: Sequence[LiveVideo], job_for: JobFor) -> None:
    """Attach the video's scheduler and write its first cues at the current settings."""
    scheduler = LiveScheduler(videos[session.video_index].stream, job_for)
    session.service.attach(session.session_id, scheduler)
    session.service.settings(spec_of(session), 0)


def add_live_routes(
    app: Any,
    *,
    videos: Sequence[LiveVideo],
    items: Sequence[RatingItem],
    log_dir: Path,
    job_for: JobFor,
    key: bytes,
    current: Callable[[Any], LiveSession],
    changed: Callable[[LiveSession, str], None],
    clock: Callable[[], float] = time.time,
) -> None:
    """The watch's routes under ``/watch/``; ``current(request)`` raises HTTPException without a session."""
    from fastapi import APIRouter, HTTPException, Request, Response
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

    if not videos:
        raise ValueError("the live app needs at least one video")
    router = APIRouter(prefix=WATCH)

    def log(session: LiveSession, event: str, **data: Any) -> None:
        live_event(log_dir, clock(), session.session_id, event, **data)

    def view(session: LiveSession) -> dict[str, Any]:
        video = videos[session.video_index]
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

    def watch_root() -> RedirectResponse:
        return RedirectResponse("watch/")

    app.add_api_route(WATCH, watch_root, methods=["GET"])

    @router.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "live.html").read_text(encoding="utf-8")

    @router.get("/live.js")
    def script() -> Response:
        return Response((STATIC / "live.js").read_text(encoding="utf-8"), media_type="text/javascript")

    @router.post("/api/session")
    def session_view(request: Request) -> dict[str, Any]:
        return view(current(request))

    @router.get("/api/state")
    def state(request: Request) -> dict[str, Any]:
        return view(current(request))

    @router.get("/media/{token}")
    def media(token: str, request: Request) -> FileResponse:
        session = current(request)
        video = videos[session.video_index]
        if not secrets.compare_digest(token, opaque_token(key, session.session_id, video.video_id)):
            raise HTTPException(status_code=404)
        return FileResponse(
            video.media, media_type="video/webm" if video.media.suffix == ".webm" else "video/mp4"
        )

    @router.post("/api/start")
    def start_watch(request: Request) -> dict[str, Any]:
        session = current(request)
        if session.stage != "ready":
            raise HTTPException(status_code=409, detail="지금은 시청을 시작할 수 없습니다.")
        if 0 not in session.service.ready():
            raise HTTPException(status_code=409, detail="첫 자막을 준비하고 있습니다.")
        session.stage = "watch"
        log(session, "watch_started", video_index=session.video_index)
        changed(session, "watch_started")
        return view(session)

    @router.post("/api/settings")
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
        changed(session, "settings")
        return {"ok": True}

    @router.post("/api/playback")
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

    @router.get("/api/captions")
    def captions(request: Request) -> dict[str, Any]:
        session = current(request)
        ready = session.service.ready() if session.stage in ("ready", "watch") else {}
        served = {}
        for cue, entry in ready.items():
            token = opaque_token(key, session.session_id, entry["job_id"])
            session.tokens[token] = entry["job_id"]
            served[str(cue)] = {"token": token, "text": entry["text"]}
        return {"captions": served}

    @router.post("/api/exposures")
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

    @router.post("/api/video-ended")
    def video_ended(request: Request) -> dict[str, Any]:
        session = current(request)
        if session.stage != "watch":
            raise HTTPException(status_code=409, detail="지금은 영상을 마칠 수 없습니다.")
        log(session, "video_ended", video_index=session.video_index)
        if session.video_index + 1 < len(videos):
            session.service.stop(session.session_id)
            session.video_index += 1
            session.stage = "ready"
            open_video(session, videos, job_for)
        else:
            # The last video frees the GPU for the next participant while this one rates.
            session.service.detach(session.session_id)
            session.stage = "ratings"
        changed(session, "video_ended")
        return view(session)

    @router.get("/api/items")
    def get_items(request: Request) -> dict[str, Any]:
        current(request)
        return {"items": [item.record() for item in items]}

    @router.post("/api/ratings")
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
        changed(session, "ratings")
        return {"stage": "done"}

    app.include_router(router)
