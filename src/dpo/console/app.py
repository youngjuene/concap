"""The console's HTTP surface: what the page may ask for, and what it may never see.

One FastAPI app serves the page and everything it fetches. Four rules shape it,
all consequences of the specification rather than of HTTP.

*The deterministic layer is served, not sent.* Everything through the prompt is
deterministic and recomputes instantly on any parameter change (§2), but the
quantities it recomputes from are measurements, and no measured value may reach
a screen as a number (§4.4). So the browser never holds ``r_g``, ``p_g``,
``c_g``, ``e_g``, ``v_g`` or ``ŵ_g``. ``/api/shot`` sends the sources with their
band and register *as phrases*, and the regimes of every admitted subset as
orders and spans. The page can then respond to a mute or a segment click with
no round trip, which is what "anything that looks live must respond" (§2)
requires, while the numbers stay on the server.

*A caption request is validated against the console's own math.* A regime index
past what the admitted set reaches is a 400, not a caption: the browser reaches
a regime only by clicking a segment, so anything else is forged. What passes is
cached by its settings key, so identical settings always return the identical
prose within a session and a revisit costs nothing (§8). There is no reroll
route, because there is no reroll.

*The check follows the endpoint.* ``/api/check`` answers only once every shot
of the clip has a committed caption, and it labels the two captions A and B
with a side assignment derived from the participant, the shot and the
configuration hash — stable across a reload, unpredictable to the participant,
and reproducible by the analysis without a stored table.

*The measurements are written, never returned.* ``C_anch``, ``D`` and ``ρ_g``
go to the log when a shot is first opened (§9). No route returns them.

Section numbers cite ``spec-system.md`` except §4.4 and §2's
instant-response rule, which are ``spec-uiux.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from dpo.caption import media as media_tools
from dpo.caption.media import MediaError
from dpo.caption.writer import CachedWriter, CaptionWriter, ShotMedia, WriterError
from dpo.console.copy import CARDS, STRINGS
from dpo.console.document import (
    clip_by_id,
    configuration_of,
    field_of,
    load_console_document,
    participant_document,
    shot_by_id,
    validate_console_document,
)
from dpo.console.log import ConsoleLogError, EventLog, committed_in, validate_participant
from dpo.console.quantities import Field, subsets_of
from dpo.console.requests import (
    ConsoleTemplateWriter,
    RequestBuilder,
    SettingsError,
    every_audition,
    validate_settings,
)

CAPTION_FAILED = STRINGS["error"]
STATIC_FILES = {
    "identity.css": "text/css; charset=utf-8",
    "console.css": "text/css; charset=utf-8",
    "console.js": "text/javascript; charset=utf-8",
}
CACHE_FILE = "captions.json"
MEDIA_CACHE_DIR = "media-cache"


def subset_key(ids: Any) -> str:
    """The identity of an admitted set: sorted ids, so order never keys a lookup."""
    return "+".join(sorted(ids))


def _package_file(name: str) -> str:
    """Read one file shipped inside ``dpo.console``; patched by tests."""
    return files("dpo.console").joinpath(name).read_text(encoding="utf-8")


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


def side_for(participant: str, clip_id: str, shot_id: str, config_hash: str) -> str:
    """Which caption goes on side A of the check for this shot.

    Derived rather than stored. A random draw would have to be persisted to
    survive a reload, and a fixed table in the document would let a participant
    who saw one shot's layout predict the next. A hash of the participant, the
    shot and the configuration is stable for one person, unpredictable to them,
    balanced across a sample, and recomputable by the analysis from the log's
    stamp alone.
    """
    digest = hashlib.sha256(f"{config_hash}/{participant}/{clip_id}/{shot_id}".encode()).digest()
    return "own" if digest[0] % 2 == 0 else "default"


def build_app(
    document: Mapping[str, Any],
    media_dir: Path,
    out_dir: Path,
    writer: CaptionWriter,
    *,
    shot_media: ShotMedia | None = None,
) -> FastAPI:
    """The app over one validated document.

    Every audition — the caption for one source alone, which a held token shows
    — is written before the first request, so solo is instant (§2: the
    deterministic layer responds instantly; only generation is slow, and an
    audition must not feel like generation).
    """
    validate_console_document(document)
    configuration = configuration_of(document)
    app = FastAPI(title="dpo caption console", docs_url=None, redoc_url=None, openapi_url=None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = Path(media_dir)
    cache_dir = out_dir / MEDIA_CACHE_DIR
    log = EventLog(out_dir, configuration.hash)
    cached = writer if isinstance(writer, CachedWriter) else CachedWriter(writer, out_dir / CACHE_FILE)
    builder = RequestBuilder(configuration)
    narrowed = participant_document(document)

    fields: dict[tuple[str, str], Field] = {}
    for clip in document["clips"]:
        for shot in clip["shots"]:
            fields[(clip["clip_id"], shot["shot_id"])] = field_of(shot, configuration)

    def _media_for(clip_id: str, shot: Mapping[str, Any]) -> Path | None:
        return shot_media(clip_id, shot) if shot_media is not None else None

    auditions: dict[tuple[str, str], dict[str, str]] = {}
    for clip in document["clips"]:
        for shot in clip["shots"]:
            field = fields[(clip["clip_id"], shot["shot_id"])]
            media = _media_for(clip["clip_id"], shot)
            auditions[(clip["clip_id"], shot["shot_id"])] = {
                source_id: cached.write(builder.build(clip["clip_id"], shot, field, settings, media))
                for source_id, settings in every_audition(field).items()
            }

    def _participant(value: str | None) -> str | JSONResponse:
        try:
            return validate_participant(value)
        except ConsoleLogError as exc:
            return _error(400, str(exc))

    def _shot(clip_id: str, shot_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]] | JSONResponse:
        clip = clip_by_id(document, clip_id)
        if clip is None:
            return _error(404, f"unknown clip {clip_id!r}")
        shot = shot_by_id(clip, shot_id)
        if shot is None:
            return _error(404, f"unknown shot {shot_id!r}")
        return clip, shot

    # ---- the page ----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        return HTMLResponse(_package_file("console.html"))

    @app.get("/{filename}")
    def static_file(filename: str) -> Response:
        # Looked up whole. Matching a stem would serve console.css for a
        # request for console.js, since both start "console." — with the
        # wrong media type, and only the extension to tell them apart.
        media_type = STATIC_FILES.get(filename)
        if media_type is None:
            return _error(404, f"no such file {filename!r}")
        return Response(_package_file(filename), media_type=media_type)

    # ---- the session -------------------------------------------------------

    @app.get("/api/session")
    def session(participant: str | None = None) -> Any:
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        return {**narrowed, "strings": STRINGS, "cards": CARDS}

    @app.get("/api/shot/{clip_id}/{shot_id}")
    def shot_detail(clip_id: str, shot_id: str, participant: str | None = None) -> Any:
        """The console's whole deterministic surface for one shot, as words and spans."""
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        found = _shot(clip_id, shot_id)
        if isinstance(found, JSONResponse):
            return found
        _, shot = found
        field = fields[(clip_id, shot_id)]
        # §9, written on first open and never returned below.
        log.measure_shot(checked, clip_id, shot_id, field.measurements())
        return configuration.stamped(
            {
                "clip_id": clip_id,
                "shot_id": shot_id,
                "sources": [
                    {
                        "id": source.id,
                        "prose": source.prose,
                        "token": source.prose.upper(),
                        "band": configuration.visibility_phrase(field.visibility[source.id]),
                        "register": configuration.register_phrase(source.presence),
                    }
                    for source in field.sources
                ],
                # One entry per non-empty admitted subset, so muting a source
                # redraws the crossfader with no round trip (§2, §4.4).
                "regimes": {
                    subset_key(subset): field.regimes(subset)
                    for subset in subsets_of([source.id for source in field.sources])
                },
                "auditions": auditions[(clip_id, shot_id)],
            }
        )

    @app.post("/api/caption")
    def caption(payload: Mapping[str, Any]) -> Any:
        checked = _participant(payload.get("participant"))
        if isinstance(checked, JSONResponse):
            return checked
        clip_id, shot_id = str(payload.get("clip_id")), str(payload.get("shot_id"))
        found = _shot(clip_id, shot_id)
        if isinstance(found, JSONResponse):
            return found
        _, shot = found
        field = fields[(clip_id, shot_id)]
        raw = payload.get("settings")
        if not isinstance(raw, Mapping):
            return _error(400, "settings must be an object")
        try:
            settings = validate_settings(field, raw)
        except SettingsError as exc:
            return _error(400, str(exc))
        request = builder.build(clip_id, shot, field, settings, _media_for(clip_id, shot))
        try:
            text, from_cache = cached.write_cached(request)
        except WriterError:
            return _error(502, CAPTION_FAILED)
        return configuration.stamped(
            {
                "clip_id": clip_id,
                "shot_id": shot_id,
                "settings": settings.as_json(),
                "key": settings.key,
                "caption": text,
                "cached": from_cache,
            }
        )

    # ---- the check ---------------------------------------------------------

    @app.get("/api/check/{clip_id}")
    def check(clip_id: str, participant: str | None = None) -> Any:
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        clip = clip_by_id(document, clip_id)
        if clip is None:
            return _error(404, f"unknown clip {clip_id!r}")
        snapshot = log.snapshot(checked)
        shots: list[dict[str, Any]] = []
        for shot in clip["shots"]:
            own = committed_in(snapshot, clip_id, shot["shot_id"])
            if own is None:
                # The check follows the endpoint (§11). The body names the
                # first shot with nothing committed, so an operator can see
                # where the session stopped.
                return _error(
                    409,
                    "the check follows the endpoint; this shot has no committed caption",
                    clip_id=clip_id,
                    shot_id=shot["shot_id"],
                )
            side = side_for(checked, clip_id, str(shot["shot_id"]), configuration.hash)
            pair = (own, shot["default_caption"]) if side == "own" else (shot["default_caption"], own)
            shots.append({"shot_id": shot["shot_id"], "a": pair[0], "b": pair[1]})
        return configuration.stamped({"clip_id": clip_id, "shots": shots})

    # ---- the log -----------------------------------------------------------

    @app.post("/api/events")
    def events(payload: Mapping[str, Any]) -> Any:
        checked = _participant(payload.get("participant"))
        if isinstance(checked, JSONResponse):
            return checked
        batch = payload.get("events")
        if not isinstance(batch, list):
            return _error(400, "events must be a list")
        snapshot = payload.get("snapshot")
        if snapshot is not None and not isinstance(snapshot, Mapping):
            return _error(400, "snapshot must be an object")
        try:
            written = log.append(checked, batch, snapshot)
        except ConsoleLogError as exc:
            return _error(400, str(exc))
        return {"appended": written}

    @app.get("/api/state")
    def state(participant: str | None = None) -> Any:
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        return {"snapshot": log.snapshot(checked)}

    @app.get("/api/log")
    def download(participant: str | None = None) -> Any:
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        return log.export(checked, str(document["session_id"]))

    # ---- media -------------------------------------------------------------

    @app.get("/media/{clip_id}")
    def clip_media(clip_id: str) -> Any:
        if clip_by_id(document, clip_id) is None:
            return _error(404, f"unknown clip {clip_id!r}")
        try:
            path = media_tools.find_clip_video(media_dir, clip_id)
        except MediaError as exc:
            return _error(404, str(exc))
        if path is None:
            return _error(404, f"no media for clip {clip_id!r}")
        return FileResponse(path)

    @app.get("/media/still/{clip_id}")
    def clip_still(clip_id: str) -> Any:
        clip = clip_by_id(document, clip_id)
        if clip is None:
            return _error(404, f"unknown clip {clip_id!r}")
        try:
            path = media_tools.clip_still(media_dir, cache_dir, clip_id)
        except MediaError as exc:
            return _error(404, str(exc))
        return FileResponse(path)

    return app


def run_console_app(
    session: str | Path,
    media_dir: str | Path,
    out_dir: str | Path,
    *,
    writer: CaptionWriter | None = None,
    shot_media: ShotMedia | None = None,
    host: str = "127.0.0.1",
    port: int = 8778,
) -> None:
    """Serve one document. Port 8778, one past the skeleton instrument's 8777."""
    document = load_console_document(session)
    chosen = ConsoleTemplateWriter() if writer is None else writer
    app = build_app(document, Path(media_dir), Path(out_dir), chosen, shot_media=shot_media)
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = [
    "CACHE_FILE",
    "CAPTION_FAILED",
    "MEDIA_CACHE_DIR",
    "STATIC_FILES",
    "build_app",
    "run_console_app",
    "side_for",
    "subset_key",
    "validate_console_document",
]
