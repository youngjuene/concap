"""One participant app: the experimenter's entry, phase 1 and the live watch.

Researcher's decisions of 2026-10-18 (plan "Phase 1 — 2026-10-18"):
- **Entry:** the experimenter types the participant ID on the start screen.
  The same ID resumes an unfinished session; a finished one is refused.
  Since 2026-11-12 a remote participant opens a personal link instead
  (``/?p=<token>``; ``entry="link"``): the token stands for their ID
  (``SessionStore.link_issue``), nobody types an ID, and an ID sent by hand is
  refused, so one participant cannot open another's session by changing a
  number. Opening the link again resumes the session, or shows the end screen
  once it is finished.
- **Writer:** each session claims a writer service when the ID is entered, so a
  full machine shows before the participant starts, not at the handoff (a GPU
  with Trida until 2026-10-27; ``live serve --sessions`` of them since).
- **Phases:** phase 1 (:class:`diff.phase1_flow.Phase1`) runs the
  questionnaire spreadsheet's pages, two viewings per clip. Each clip's second
  caption is written by the session's writer service, queued when the
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
phase-1 progress exactly. Since 2026-11-12 a live watch survives one too, and a
request that reaches another server (Vercel): the session is rebuilt from the
store on every request, and the video's playback, sliders and written captions
are in the store's shared watch record, so the video carries on where it is.
Only a browser that lost its video (a reload) starts that video again
(``/watch/api/restart``), which is logged as ``video_restarted``. A finished
session's browser still gets its end screen, and nothing else.

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
    WATCHING,
    JobFor,
    LivePool,
    LiveSession,
    LiveVideo,
    RatingItem,
    SourceDraw,
    add_live_routes,
    live_event,
    open_video,
    spec_of,
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
TweakFor = Callable[[Mapping[str, Any], Mapping[str, str], int], tuple[CaptionSource, Job, dict[str, Any]]]
WATCH_URL: Final = "watch/"
# What each live event may change in the stored progress. An update writes only these, so two servers
# handling two requests of one session cannot overwrite each other's part.
PROGRESS_KEYS: Final = {
    "session_started": (),
    "settings": ("texture", "context"),
    "watch_started": ("stage",),
    "video_restarted": ("stage",),
    "video_ended": ("stage", "video_index", "coverage"),
    "ratings": ("answers", "stage", "source"),
    "source_answer": ("source_answers", "stage"),
}
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
BAD_LINK: Final = "링크를 확인할 수 없습니다. 연구자에게 받은 링크로 다시 접속해 주세요."
ENTRIES: Final = ("id", "link")
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


class SessionShare:
    """One session's shared watch record: the store's ``watch_*`` tables bound to the session
    (``live_generation.SharedWatch``)."""

    def __init__(self, store: SessionStore, session_id: str) -> None:
        self.store, self.session_id = store, session_id

    def now(self, video_id: str) -> dict[str, Any] | None:
        return self.store.watch_now(self.session_id, video_id)

    def rows(self, after: int) -> list[dict[str, Any]]:
        return self.store.watch_rows(self.session_id, after)

    def put(self, video_id: str, cue: int, job_id: str, origin: str, result: Mapping[str, Any]) -> None:
        self.store.watch_put(self.session_id, video_id, cue, job_id, origin, result)

    def claim(self, key: str, ttl_s: float) -> bool:
        return self.store.watch_claim(self.session_id, key, ttl_s)

    def release(self, key: str) -> None:
        self.store.watch_release(self.session_id, key)


def live_progress(calibration: Calibration) -> dict[str, Any]:
    return {
        "stage": "ready",
        "video_index": 0,
        "texture": calibration.texture,
        "context": calibration.context,
        "answers": {},
        "started": False,
        "source": None,
        "source_answers": [],
        "coverage": [],
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
    source_draw: SourceDraw | None = None,
    entry: str = "id",
) -> Any:
    """The whole session's app; without ``phase1`` (smoke runs) an entry goes straight to the watch.

    ``tweak_for(clip, slots, rung)`` builds a clip's second-caption job at a rung of its ladder;
    phase 1 needs it. The session's live service writes them: Claude through the API since
    2026-10-27, which also writes phase 2's live captions (on 2026-10-26 to 27 phase 1 had its own
    writer pool, while phase 2 stayed on Trida). ``source_draw`` is the perceived-source block's draw.

    ``entry``: ``"id"`` (the lab: the experimenter types the participant ID) or ``"link"`` (remote
    participants: a personal link's token is the only way in).
    """
    if entry not in ENTRIES:
        raise ValueError(f"entry is one of {ENTRIES}, not {entry!r}")
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    key = secret if secret is not None else secrets.token_bytes(32)
    # Each long video's length in whole minutes, for the handoff's briefing.
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
        if outcome["status"] == "next":  # the gate refused it: queue the next rung now
            submit_pending(request.session_id)

    def submit_pending(session_id: str, state: Mapping[str, Any] | None = None) -> None:
        """Queue every pending second caption on the session's writer (already queued ones are skipped).

        ``state``: the session's state when the caller has just read it, so it is not read again.
        """
        if phase1 is None or tweak_for is None:
            return
        if state is None:
            state = store.get(session_id).state
        if state.get("stage") != "phase1":
            return
        pending = phase1.pending_tweaks(state)
        if not pending:
            return
        service = pool.acquire(session_id)  # also tells the writer this server is being asked
        if service is None:
            return
        service.share(session_id, SessionShare(store, session_id))
        for clip_key, slots, attempt, rung in pending:
            source, job, steer = tweak_for(phase1.clips[clip_key], slots, rung)
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
        link = body.get("link") if isinstance(body, Mapping) else None
        if isinstance(link, str):
            participant = store.link_participant(link)
            if participant is None:
                return error(404, BAD_LINK)
        elif entry == "link":
            return error(403, BAD_LINK)  # no typed IDs: only a link's token names a participant
        else:
            participant = body.get("participant") if isinstance(body, Mapping) else None
        if not isinstance(participant, str):
            return error(400, BAD_ID)
        try:
            session, _ = store.enter(participant, initial(participant))
        except SessionFinished:
            # A personal link opened again after the end: its own end screen, not an error.
            return JSONResponse({"error": FINISHED, "finished": True}, status_code=409)
        except ValueError:
            return error(400, BAD_ID)
        if pool.acquire(session.session_id) is None:
            return error(503, BUSY_MESSAGE)
        submit_pending(session.session_id, session.state)
        reply = {"next_url": WATCH_URL} if session.state["stage"] == "phase2" else {"stage": "phase1"}
        response = JSONResponse(reply)
        response.set_cookie(COOKIE, session.token, httponly=True, samesite="strict", path="/")
        return response

    # -- phase 1 ---------------------------------------------------------------
    @app.get("/api/p1/state")
    def phase1_state(request: Request) -> Response:
        session = store.by_token(request.cookies.get(COOKIE))
        if session is None or (session.finished_at is not None and session.state["stage"] != "phase2"):
            return JSONResponse({"error": NO_SESSION, "entry": entry}, status_code=401)
        if session.state["stage"] == "phase2":  # a finished session's browser lands on its end screen
            return JSONResponse({"kind": "moved", "next_url": WATCH_URL})
        submit_pending(session.session_id, session.state)
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
        # The cookie's session, without its state: the store reads that once, inside the change.
        session = store.watch_session(request.cookies.get(COOKIE))
        if session is None or session["finished"] or phase1 is None:
            return error(401, NO_SESSION)
        session_id = str(session["session_id"])
        if action not in ACTIONS:
            return error(404, UNREQUESTED)
        body = await request.json()
        data = body.get("data", {}) if isinstance(body, Mapping) else None
        if not isinstance(data, dict):
            return error(400, "Action data must be an object")
        flow = phase1
        waiting: list[bool] = []  # whether the action left a second caption to write

        def apply(state: dict[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
            now = clock()
            if action != "handoff":
                event = flow.apply(state, action, data, now)
                waiting.append(bool(flow.pending_tweaks(state)))
                return page(state), event
            if data.get("page") != "handoff":
                raise Conflict("This answer belongs to another page")
            calibration = calibrate(*flow.profile_inputs(state))
            event = flow.hand_off(state, calibration_record(calibration), now)
            state["live"] = live_progress(calibration)
            return {"next_url": WATCH_URL}, event

        try:
            reply = store.mutate(
                session_id,
                body.get("key"),
                body.get("revision"),
                {"action": action, "data": data},
                apply,
            )
        except Conflict as exc:
            return error(409, str(exc))
        except (ValueError, KeyError, TypeError) as exc:
            return error(400, str(exc))
        # Only then is the state read again to queue them: most actions leave none, and the state
        # is hundreds of kilobytes by stage B. A retried request (no apply) is covered by the
        # page's own polling while it waits.
        if waiting and waiting[-1]:
            submit_pending(session_id)
        return JSONResponse(reply)

    # -- live phase 2 ----------------------------------------------------------
    durations = {video.video_id: video.stream.duration_ms for video in videos}

    def reopen(live: LiveSession) -> None:
        open_video(live, videos, job_for, store, SessionShare(store, live.session_id))

    def current(request: Request) -> LiveSession:
        """The request's live session, as the store has it now.

        Nothing but a cache lives in this server's memory: any server may take any request (Vercel),
        and each brings its copy and its writer up to the store before it answers.
        """
        # Only the session's live part is read: the whole state, with phase 1's record, is hundreds
        # of kilobytes, and a watch page asks several times a second.
        found = store.watch_session(request.cookies.get(COOKIE))
        if found is None:
            raise HTTPException(status_code=401, detail=NO_SESSION)
        if found["stage"] != "phase2":
            raise HTTPException(status_code=409, detail=NOT_YET)
        session_id, finished, progress = found["session_id"], found["finished"], found["live"]
        with lock:
            live = live_sessions.get(session_id)
            joined = live is None
            if live is None:
                live = LiveSession(session_id, pool.services[0], calibration_of(found["calibration"]))
                if not finished:
                    live_sessions[session_id] = live
            live.stage, live.video_index = progress["stage"], progress["video_index"]
            live.texture, live.context = progress["texture"], progress["context"]
            live.answers = dict(progress["answers"])
            live.source = progress.get("source")
            live.source_answers = list(progress.get("source_answers", []))
        if finished:
            return live  # the end screen, for a reload or a retried last answer: nothing is stored
        if live.stage in WATCHING:
            service = pool.acquire(session_id)
            if service is None:
                raise HTTPException(status_code=503, detail=BUSY_MESSAGE)
            live.service = service
            video = videos[live.video_index]
            if service.video_id() != video.video_id or service.spec() is None:
                reopen(live)
            elif service.spec() != spec_of(live):
                # Another server took a slider move: the record first, so every writer follows it.
                store.watch_open(live.session_id, video.video_id, live.texture, live.context)
                service.settings(spec_of(live))
        else:
            pool.release(session_id)  # the last video released the writer; these pages need none
        if joined and not progress["started"]:
            live_event(
                log_dir, clock(), live.session_id, "session_started", calibration=live.calibration.record
            )
            changed(live, "session_started")
        elif joined:
            # A server restart, or another server's first request of this session: nothing restarts.
            live_event(
                log_dir,
                clock(),
                live.session_id,
                "server_joined",
                stage=live.stage,
                video_index=live.video_index,
                video_id=videos[live.video_index].video_id,
            )
        return live

    def coverage(session_id: str) -> list[dict[str, Any]]:
        """How much of each video was played, from the shared record's playback reports."""
        return [
            {
                "video_id": row["video_id"],
                "duration_ms": durations[row["video_id"]],
                "watched_ms": row["watched_ms"],
                "watched_share": round(min(1.0, row["watched_ms"] / durations[row["video_id"]]), 4),
                "furthest_ms": row["max_position_ms"],
                "restarts": row["restarts"],
            }
            for row in store.watch_videos(session_id)
            if row["video_id"] in durations
        ]

    def changed(live: LiveSession, event: str) -> None:
        progress = {
            "stage": live.stage,
            "video_index": live.video_index,
            "texture": live.texture,
            "context": live.context,
            "answers": dict(live.answers),
            "source": live.source,
            "source_answers": list(live.source_answers),
        }
        if event == "video_ended":
            progress["coverage"] = coverage(live.session_id)
        kept = {name: progress[name] for name in PROGRESS_KEYS[event]} | {"started": True}
        note = None if event == "settings" else {"event": f"live_{event}", "video_index": live.video_index}
        store.update(live.session_id, lambda state: state["live"].update(kept), note)
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
        after_phase1=phase1 is not None,
        source_draw=source_draw,
        watch=store,
        reopen=reopen,
    )
    return app
