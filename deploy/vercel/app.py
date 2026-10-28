"""Vercel's entrypoint for the diff participant app: both phases, written by Claude through the API.

``deploy/vercel/stage.py`` copies this file to the root of a staged site, beside the ``diff``
and ``dpo`` packages, the setup's stills and masks, and ``deploy.json``, which says what to serve.
The app is the lab's (:func:`diff.participant_server.build_participant_app`), built the way
``python -m diff live serve`` builds it, with these differences forced by Vercel:

- **Videos come from Vercel Blob.** A function's response is at most 4.5 MB and each clip is
  more, so ``FileResponse`` is replaced, before the app is built, by a 307 to the clip's Blob
  URL. Blob names are content hashes, so no URL carries a clip key. Stills and masks stay in
  the bundle and are served as before.
- **One media secret for every instance.** ``DIFF_SECRET`` keys the opaque media tokens, so a
  token one instance issued opens on another.
- **The store is Postgres** (Neon, ``DATABASE_URL``), shared by every instance: the lab's
  store with its connection layer swapped (:mod:`pg_store`). A store in each instance's /tmp
  lost a session on the 2026-10-27 walk as soon as a request reached a second instance.
- **An access code.** The production address is public, so only a browser that opened the
  study link (``/?code=…``) gets past :func:`_gate`.
- **Logs go to Postgres too** (:class:`pg_store.PgLogDir`), one row per line, since /tmp dies
  with the instance. ``pull.py`` writes them back out as the lab's files for ``diff export``.

Writer threads (``LiveService``) start at import. The browser polls ``/api/p1/state`` while it
waits for a second caption, which keeps the instance running while the thread writes.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEPLOY = json.loads((ROOT / "deploy.json").read_text(encoding="utf-8"))


def _redirect_videos(media: dict[str, str]) -> None:
    """Make every ``FileResponse`` of a staged video a redirect to its Blob copy."""
    import fastapi.responses as responses

    file_response = responses.FileResponse

    def respond(path: Any, *args: Any, **kwargs: Any) -> Any:
        url = media.get(Path(path).resolve().relative_to(ROOT).as_posix())
        if url is None:
            return file_response(path, *args, **kwargs)
        return responses.RedirectResponse(url, status_code=307)

    responses.FileResponse = respond  # type: ignore[assignment,misc]


def _gate(app: Any, code: str, *, open: bool = True) -> None:
    """Admit a browser only after it opened the study link (``/?code=…``); the rest get CLOSED.

    The production address is public, so the link's code is the only thing that stops a stranger
    entering a participant ID and spending API calls. The code is in ``deploy.json`` (in the
    bundle, not served); a re-stage with a new code closes every old link. A site staged with
    ``--closed`` (``open: false``) shows the closed page to everyone, code or not.
    """
    import hashlib
    import hmac

    from fastapi import Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    admitted = hashlib.sha256(code.encode()).hexdigest()

    def closed() -> Any:
        return HTMLResponse(CLOSED, status_code=403, headers={"Cache-Control": "no-store"})

    @app.middleware("http")
    async def access(request: Request, call_next: Any) -> Any:
        if not open:
            return closed()
        given = request.query_params.get("code")
        if given is not None:
            if not hmac.compare_digest(given.strip().lower().encode(), code.encode()):
                return closed()
            response = RedirectResponse(request.url.path, status_code=303)
            # Lax, not strict: the link is opened from another site (a message, a mail).
            response.set_cookie(
                ACCESS_COOKIE, admitted, max_age=ACCESS_SECONDS, httponly=True, secure=True, samesite="lax"
            )
            return response
        if hmac.compare_digest(request.cookies.get(ACCESS_COOKIE, "").encode(), admitted.encode()):
            return await call_next(request)
        return closed()


ACCESS_COOKIE = "diff_access"
ACCESS_SECONDS = 12 * 3600
CLOSED = (
    '<!doctype html><html lang="ko"><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1"><title>ConCap</title>'
    '<p style="font: 18px/1.6 sans-serif; margin: 20vh auto; max-width: 32em; padding: 0 16px">'
    "지금은 이 주소로 참여할 수 없습니다. 연구자에게 받은 링크로 접속해 주세요.</p></html>"
)


def _build() -> Any:
    from pg_store import PgSessionStore, on_database

    import diff.workbook_instrument as workbook_instrument
    from diff.caption_sources import parse_sources
    from diff.claude_writer import ClaudeWriter
    from diff.job_plan import live_job
    from diff.live_generation import LiveService, LiveStream
    from diff.live_server import LivePool, LiveVideo, rating_items
    from diff.output_checks import check_output, displayed, output_from_record
    from diff.participant_server import build_participant_app
    from diff.phase1_flow import Phase1
    from diff.phase1_setup import read_setup
    from diff.phase1_tweak import tweak_job_for
    from diff.phase1_usable import check_record
    from diff.phase2_scene import cue_steer
    from diff.study_protocol import LIVE_MAX_CHARS

    purpose = DEPLOY["purpose"]
    sources = parse_sources(json.loads((ROOT / DEPLOY["sources"]).read_text(encoding="utf-8"))).sources
    videos = [
        LiveVideo(video_id, ROOT / path, LiveStream.from_sources(sources, video_id))
        for video_id, path in DEPLOY["videos"]
    ]
    items = rating_items(json.loads((ROOT / DEPLOY["items"]).read_text(encoding="utf-8")))
    # stage.py ran setup_problems against the full media before staging; the videos are not here.
    phase1 = Phase1(read_setup(ROOT / DEPLOY["phase1_setup"]), root=ROOT, instrument=workbook_instrument)

    scenes = (
        json.loads((ROOT / DEPLOY["scenes"]).read_text(encoding="utf-8")) if DEPLOY.get("scenes") else None
    )

    def job_for(source: Any, spec: Any) -> Any:
        scene = scenes["cues"].get(source.source_id) if scenes is not None else None
        steer = cue_steer(scene, spec.scene) if scenes is not None else None
        return live_job(source, spec.texture, spec.context, spec.system_message, purpose, steer)

    def tweak_for(clip: Any, slots: Any, rung: int) -> Any:
        return tweak_job_for(clip, slots, purpose, rung)

    def check(source: Any, job: Any, result: Any) -> Any:
        output = displayed(output_from_record(result["output"]))
        return check_output(
            source, output, template_phase=2, cell=job.cell, max_chars=LIVE_MAX_CHARS
        ).record()

    store = PgSessionStore(on_database(os.environ["DATABASE_URL"], DEPLOY["database"]))
    logs = store.log_dir() / "sessions"
    writer = ClaudeWriter()
    services = [
        LiveService(writer.attempt, logs, check=check, tweak_check=check_record)
        for _ in range(DEPLOY["sessions"])
    ]
    for service in services:
        service.start()
    _redirect_videos(DEPLOY["media"])
    app = build_participant_app(
        store=store,
        pool=LivePool(services),
        videos=videos,
        items=items,
        log_dir=logs,
        job_for=job_for,
        phase1=phase1,
        tweak_for=tweak_for,
        secret=os.environ["DIFF_SECRET"].encode(),
    )
    _gate(app, DEPLOY["access_code"], open=DEPLOY.get("open", True))
    return app


app = _build()
