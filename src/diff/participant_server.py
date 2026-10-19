"""One participant app: the experimenter's entry, phase 1 and the live watch.

Researcher's decisions of 2026-10-18 (plan "Phase 1 — 2026-10-18"):
- **Entry:** the experimenter types the participant ID on the start screen.
  The same ID resumes an unfinished session; a finished one is refused.
- **GPU:** each session claims a GPU when the ID is entered, so a full machine
  shows before the participant starts, not at the handoff.
- **Phases:** phase 1 (:class:`diff.phase1_flow.Phase1`) runs the
  questionnaire spreadsheet's pages, two viewings per clip. Each clip's second
  caption is written by Trida on the GPU the session claimed, queued when the
  clip's first step closes (P5) and written back into the session when it
  finishes. At the handoff, the tuner's calibration is built from the phase-1
  answers and the live watch (``/watch/``) opens with it, with no researcher
  step in between.
- **Lab only:** the server listens on loopback or the lab network.

Routes:
- ``/``, ``/phase1_pages.js``, ``/participant_theme.css``: the start screen
  and phase 1. Open ``/#enter`` to switch participants.
- ``POST /api/enter``: creates or resumes a session and sets the
  ``diff_session`` cookie.
- ``/api/p1/state``, ``POST /api/p1/<action>``, ``/api/p1/media/<token>``:
  phase 1. Every change goes through the store's revision and idempotency key.
- ``/watch/…``: the live routes (:func:`diff.live_server.add_live_routes`).

Sessions are durable (:mod:`diff.session_store`). A server restart keeps
phase-1 progress exactly. A live watch resumes at the start of the video it
was in, with its settings, and the restart is logged.

**Blindness:** nothing a browser can fetch carries a clip key, a sound family,
an R2 answer key, the model or the decoding condition. Media go through
per-session opaque tokens, and research data leave only through
``python -m diff export``.

FastAPI is imported inside :func:`build_participant_app`.
"""

# No ``from __future__ import annotations``: FastAPI reads the handlers' annotations at runtime.
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from diff.caption_sources import CaptionSource
from diff.job_plan import Job
from diff.live_calibration import Calibration, calibrate
from diff.live_generation import TweakRequest, opaque_token
from diff.live_server import (
    BUSY_MESSAGE,
    JobFor,
    LivePool,
    LiveSession,
    LiveVideo,
    RatingItem,
    add_live_routes,
    live_event,
    open_video,
)
from diff.phase1_flow import Phase1
from diff.session_store import Conflict, Session, SessionFinished, SessionStore

COOKIE: Final = "diff_session"
STATIC: Final = Path(__file__).with_name("participant_static")
ASSETS: Final = {"phase1_pages.js": "text/javascript", "participant_theme.css": "text/css"}
ACTIONS: Final = frozenset(
    {
        "intro-complete",
        "practice-complete",
        "playback",
        "view-ended",
        "submit",
        "skip-second",
        "interlude-complete",
        "handoff",
    }
)
TweakFor = Callable[[Mapping[str, Any], Mapping[str, str]], tuple[CaptionSource, Job, dict[str, Any]]]
WATCH_URL: Final = "watch/"
PHASE2_ONLY: Final = "diff.phase2-only/v1"
MEDIA_TYPES: Final = {".png": "image/png", ".mp4": "video/mp4", ".webm": "video/webm"}
HEADERS: Final = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "frame-ancestors 'none'",
}
NO_SESSION: Final = "세션이 없습니다. 처음부터 다시 시작해 주세요."
FINISHED: Final = "이미 완료된 참가자 번호입니다. 다른 번호를 입력해 주세요."
BAD_ID: Final = "참가자 번호는 P1, P2와 같은 형식입니다."
NOT_YET: Final = "아직 이 단계가 아닙니다."
UNREQUESTED: Final = "요청을 확인할 수 없습니다. 화면을 새로고침해 주세요."


def calibration_record(calibration: Calibration) -> dict[str, Any]:
    return {
        "texture": calibration.texture,
        "context": calibration.context,
        "system_message": calibration.system_message,
        "provisional": calibration.provisional,
        "record": dict(calibration.record),
    }


def calibration_of(record: Mapping[str, Any]) -> Calibration:
    return Calibration(
        record["texture"],
        record["context"],
        record["system_message"],
        record["provisional"],
        record["record"],
    )


def live_progress(calibration: Calibration) -> dict[str, Any]:
    return {
        "stage": "ready",
        "video_index": 0,
        "texture": calibration.texture,
        "context": calibration.context,
        "answers": {},
        "started": False,
    }


def build_participant_app(
    *,
    store: SessionStore,
    pool: LivePool,
    videos: Sequence[LiveVideo],
    items: Sequence[RatingItem],
    log_dir: Path,
    job_for: JobFor,
    phase1: Phase1 | None,
    tweak_for: TweakFor | None = None,
    secret: bytes | None = None,
    clock: Callable[[], float] = time.time,
) -> Any:
    """The whole session's app; without ``phase1`` (smoke runs) an entry goes straight to the watch.

    ``tweak_for(clip, slots)`` builds a clip's second-caption job; phase 1 needs it.
    """
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    key = secret if secret is not None else secrets.token_bytes(32)
    # Each long video's length in whole minutes, for phase 1's interstitial.
    live_minutes = tuple(max(1, round(video.stream.duration_ms / 60000)) for video in videos)
    live_sessions: dict[str, LiveSession] = {}
    lock = threading.Lock()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        for name, value in HEADERS.items():
            response.headers.setdefault(name, value)
        if "/api/" in request.url.path:
            response.headers["Cache-Control"] = "no-store"
        return response

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    def stored(request: Request) -> Session | None:
        session = store.by_token(request.cookies.get(COOKIE))
        return None if session is None or session.finished_at is not None else session

    def media_token(state: Mapping[str, Any], name: str) -> str:
        return opaque_token(key, state["session_id"], f"p1:{state['clip_index']}:{state['page']}:{name}")

    def page(state: Mapping[str, Any]) -> dict[str, Any]:
        assert phase1 is not None
        payload = phase1.present(state, lambda name: f"api/p1/media/{media_token(state, name)}", live_minutes)
        payload["session"] = opaque_token(key, state["session_id"], "drafts")
        return payload

    def tweak_done(request: TweakRequest, outcome: dict[str, Any]) -> None:
        assert phase1 is not None
        note = {
            "event": f"tweak_{outcome['status']}",
            "clip": request.clip,
            "attempt": outcome["attempt"],
            "reason": outcome.get("reason"),
        }
        store.update(
            request.session_id,
            lambda state: phase1.record_tweak(state, request.clip, outcome, clock()),
            note,
        )

    def submit_pending(session_id: str) -> None:
        """Queue every pending second caption on the session's GPU (already queued ones are skipped)."""
        if phase1 is None or tweak_for is None:
            return
        session = store.get(session_id)
        if session.state.get("stage") != "phase1":
            return
        pending = phase1.pending_tweaks(session.state)
        if not pending:
            return
        service = pool.holder(session_id) or pool.acquire(session_id)
        if service is None:
            return
        for clip_key, slots, attempt in pending:
            source, job, steer = tweak_for(phase1.clips[clip_key], slots)
            service.submit_tweak(TweakRequest(session_id, clip_key, source, job, steer, attempt), tweak_done)

    def initial(participant: str) -> Callable[[str], dict[str, Any]]:
        def make(session_id: str) -> dict[str, Any]:
            if phase1 is not None:
                return phase1.new_state(session_id, participant)
            calibration = calibrate([], [])
            return {
                "protocol": PHASE2_ONLY,
                "session_id": session_id,
                "participant": participant,
                "stage": "phase2",
                "calibration": calibration_record(calibration),
                "live": live_progress(calibration),
                "revision": 0,
            }

        return make

    # -- pages -----------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "diff_participant.html").read_text(encoding="utf-8")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)  # no icon; keeps browsers' console free of a 404

    for asset, media_type in ASSETS.items():

        def serve(asset: str = asset, media_type: str = media_type) -> Response:
            return Response((STATIC / asset).read_text(encoding="utf-8"), media_type=media_type)

        app.add_api_route(f"/{asset}", serve, methods=["GET"])

    # -- entry -----------------------------------------------------------------
    @app.post("/api/enter")
    async def enter(request: Request) -> Response:
        if request.headers.get("x-study-request") != "1":
            return error(401, UNREQUESTED)
        body = await request.json() if await request.body() else {}
        participant = body.get("participant") if isinstance(body, Mapping) else None
        if not isinstance(participant, str):
            return error(400, BAD_ID)
        try:
            session, _ = store.enter(participant, initial(participant))
        except SessionFinished:
            return error(409, FINISHED)
        except ValueError:
            return error(400, BAD_ID)
        if pool.acquire(session.session_id) is None:
            return error(503, BUSY_MESSAGE)
        submit_pending(session.session_id)
        reply = {"next_url": WATCH_URL} if session.state["stage"] == "phase2" else {"stage": "phase1"}
        response = JSONResponse(reply)
        response.set_cookie(COOKIE, session.token, httponly=True, samesite="strict", path="/")
        return response

    # -- phase 1 ---------------------------------------------------------------
    @app.get("/api/p1/state")
    def phase1_state(request: Request) -> Response:
        session = stored(request)
        if session is None:
            return error(401, NO_SESSION)
        if session.state["stage"] == "phase2":
            return JSONResponse({"kind": "moved", "next_url": WATCH_URL})
        submit_pending(session.session_id)
        return JSONResponse(page(session.state))

    @app.get("/api/p1/media/{token}")
    def phase1_media(token: str, request: Request) -> Response:
        session = stored(request)
        if session is None or phase1 is None:
            raise HTTPException(status_code=404)
        for name, path in phase1.media_files(session.state).items():
            if secrets.compare_digest(token, media_token(session.state, name)):
                return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"))
        raise HTTPException(status_code=404)

    @app.post("/api/p1/{action}")
    async def phase1_action(action: str, request: Request) -> Response:
        if request.headers.get("x-study-request") != "1":
            return error(401, UNREQUESTED)
        session = stored(request)
        if session is None or phase1 is None:
            return error(401, NO_SESSION)
        if action not in ACTIONS:
            return error(404, UNREQUESTED)
        body = await request.json()
        data = body.get("data", {}) if isinstance(body, Mapping) else None
        if not isinstance(data, dict):
            return error(400, "Action data must be an object")
        flow = phase1

        def apply(state: dict[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
            now = clock()
            if action != "handoff":
                event = flow.apply(state, action, data, now)
                return page(state), event
            if data.get("page") != "handoff":
                raise Conflict("This answer belongs to another page")
            calibration = calibrate(*flow.profile_inputs(state))
            event = flow.hand_off(state, calibration_record(calibration), now)
            state["live"] = live_progress(calibration)
            return {"next_url": WATCH_URL}, event

        try:
            reply = store.mutate(
                session.session_id,
                body.get("key"),
                body.get("revision"),
                {"action": action, "data": data},
                apply,
            )
        except Conflict as exc:
            return error(409, str(exc))
        except (ValueError, KeyError, TypeError) as exc:
            return error(400, str(exc))
        submit_pending(session.session_id)
        return JSONResponse(reply)

    # -- live phase 2 ----------------------------------------------------------
    def current(request: Request) -> LiveSession:
        session = stored(request)
        if session is None:
            raise HTTPException(status_code=401, detail=NO_SESSION)
        if session.state["stage"] != "phase2":
            raise HTTPException(status_code=409, detail=NOT_YET)
        with lock:
            live = live_sessions.get(session.session_id)
            if live is not None:
                return live
            progress = session.state["live"]
            if progress["stage"] == "ratings":
                # The last video already released the GPU; the items need none.
                service = pool.holder(session.session_id) or pool.services[0]
            else:
                acquired = pool.acquire(session.session_id)
                if acquired is None:
                    raise HTTPException(status_code=503, detail=BUSY_MESSAGE)
                service = acquired
            interrupted = progress["stage"] == "watch"
            live = LiveSession(
                session.session_id,
                service,
                calibration_of(session.state["calibration"]),
                stage="ready" if interrupted else progress["stage"],
                video_index=progress["video_index"],
                texture=progress["texture"],
                context=progress["context"],
                answers=dict(progress["answers"]),
            )
            live_sessions[session.session_id] = live
        if not progress["started"]:
            live_event(
                log_dir, clock(), live.session_id, "session_started", calibration=live.calibration.record
            )
        else:
            live_event(log_dir, clock(), live.session_id, "session_resumed", video_restarted=interrupted)
        changed(live, "session_started" if not progress["started"] else "session_resumed")
        if live.stage == "ready":
            open_video(live, videos, job_for)
        return live

    def changed(live: LiveSession, event: str) -> None:
        progress = {
            "stage": live.stage,
            "video_index": live.video_index,
            "texture": live.texture,
            "context": live.context,
            "answers": dict(live.answers),
            "started": True,
        }
        note = None if event == "settings" else {"event": f"live_{event}", "video_index": live.video_index}
        store.update(live.session_id, lambda state: state.update(live=progress), note)
        if live.stage == "done":
            store.finish(live.session_id)
            pool.release(live.session_id)
            with lock:
                live_sessions.pop(live.session_id, None)

    add_live_routes(
        app,
        videos=videos,
        items=items,
        log_dir=log_dir,
        job_for=job_for,
        key=key,
        current=current,
        changed=changed,
        clock=clock,
    )
    return app
