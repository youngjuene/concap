"""The caption session's HTTP surface: kiosk, follow-up, and what each may ask for.

One FastAPI app serves both pages and everything they fetch (contract §7).
Three rules shape it, all consequences of the spec rather than of HTTP:

* The participant document (``/api/session``) is narrowed (document.py): no
  weights, no prose, no answers, and no sources. A shot's inventory — its
  sources, phrases, roles, the precomputed orderings, the audition captions —
  comes only from ``/api/inventory/{clip}``, and that route reads the
  append-only event log for ``listing.submit`` on that clip before answering
  (spec 2, 7: the skeleton is unreachable before the listing is submitted, by
  any path). The same gate guards ``/api/caption``.
* A caption request is validated against the skeleton's own math
  (``validate_settings``): an order the columns could not have produced is a
  400, not a caption. What passes is cached by its settings key, so identical
  settings return the identical caption (spec 7) and the response says whether
  it was cached.
* The follow-up page gets the check captions already assigned to A and B per
  the document, never labelled own/automatic; its sound-only excerpts are cut
  server-side and named by excerpt id, so the clip they came from — the
  answer — never reaches the browser.

Page and static files are read lazily per request through ``_package_file``
so the server can be built and tested while the frontend is still being
written; only the five names the contract lists are served.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from dpo.caption import media as media_tools
from dpo.caption.background import BackgroundWriter
from dpo.caption.media import MediaError
from dpo.caption.writer import written_by
from dpo.session.document import (
    clip_by_id,
    load_session_document,
    participant_document,
    poles_for,
    roles_for,
    shot_by_id,
    validate_session_document,
)
from dpo.session.log import (
    FOLLOWUP_SCHEMA,
    EventLog,
    FollowupExistsError,
    SessionLogError,
    kept_caption_in,
    validate_participant,
)
from dpo.session.skeleton import SettingsError, all_orderings, neighbours, validate_settings
from dpo.session.writer import (
    CachedWriter,
    CaptionWriter,
    ShotMedia,
    WriterError,
    audition_requests,
    build_request,
)

# Spec Table 6 `error`, the one string the server itself puts in front of a participant.
CAPTION_FAILED = "Caption request failed. Check the connection and try again."
STATIC_FILES = {
    "identity.css": "text/css; charset=utf-8",
    "kiosk.css": "text/css; charset=utf-8",
    "followup.css": "text/css; charset=utf-8",
    "kiosk.js": "text/javascript; charset=utf-8",
    "followup.js": "text/javascript; charset=utf-8",
}
CACHE_FILE = "captions.json"
MEDIA_CACHE_DIR = "media-cache"


def _package_file(name: str) -> str:
    """Read one file shipped inside ``dpo.session``; patched by tests."""
    return files("dpo.session").joinpath(name).read_text(encoding="utf-8")


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


def _count(value: object) -> bool:
    """A non-negative integer (bool is an int to Python, not to a counter)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _covering(
    name: str, entries: list[Any], key: str, expected: Sequence[str]
) -> dict[str, dict[str, Any]] | JSONResponse:
    """Every entry an object whose ``key`` covers ``expected`` exactly once."""
    by_id: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get(key), str):
            return _error(400, f"{name}[{index}].{key} must be a string")
        if entry[key] not in expected:
            return _error(400, f"{name}[{index}].{key} is not part of this session")
        if entry[key] in by_id:
            return _error(400, f"{name}[{index}].{key} is listed twice")
        by_id[entry[key]] = entry
    missing = [identifier for identifier in expected if identifier not in by_id]
    if missing:
        return _error(400, f"{name} is missing {missing[0]!r}: every item must be answered once")
    return by_id


def build_app(
    document: Mapping[str, Any],
    media_dir: Path,
    out_dir: Path,
    writer: CaptionWriter,
    *,
    shot_media: ShotMedia | None = None,
    prefetch: bool = True,
) -> FastAPI:
    """The app over one validated document.

    ``shot_media`` maps (clip_id, shot) to the shot's audio on a shaped clip or a
    still on a control clip, for a writer
    that listens (the Gemma writer); the template writer needs none. The writer
    is wrapped in the cache unless it already is one, and every audition is
    written before the first request so holding a row is instant (spec 7).
    """
    validate_session_document(document)
    app = FastAPI(title="dpo caption session", docs_url=None, redoc_url=None, openapi_url=None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = Path(media_dir)
    cache_dir = out_dir / MEDIA_CACHE_DIR
    log = EventLog(out_dir)
    cached = writer if isinstance(writer, CachedWriter) else CachedWriter(writer, out_dir / CACHE_FILE)
    narrowed = participant_document(document)
    # Every audition is written off the request path, in document order; a
    # clip's own auditions jump the queue when its inventory is requested and
    # the route waits for them (spec 7: the audition is instant). After every
    # caption, the settings one gesture away are written the same way.
    background = BackgroundWriter(cached, enabled=prefetch)
    audition_table: dict[tuple[str, str], dict[str, Any]] = {}
    for clip in document["clips"]:
        for shot in clip["shots"]:
            key = (str(clip["clip_id"]), str(shot["shot_id"]))
            media = shot_media(str(clip["clip_id"]), shot) if shot_media is not None else None
            audition_table[key] = audition_requests(document, clip, shot, media)
            background.warm(audition_table[key].values(), key=key)
    # One caption is written here, synchronously, before anything is served: a
    # writer that cannot write must stop the server at start-up, not the
    # participant at the first shot. The background then owns the rest.
    probe = next((requests for requests in audition_table.values() if requests), {})
    if probe:
        request = next(iter(probe.values()))
        if cached.lookup(request) is None:
            cached.write(request)
        else:
            # A cache hit proves nothing about the writer: over a warm cache
            # the model would otherwise load on the first miss, mid-session,
            # on a participant's press. One caption from the writer itself.
            written_by(cached.inner, request)

    def _auditions(key: tuple[str, str]) -> dict[str, str] | None:
        background.urgent(audition_table[key].values(), key=key)
        background.wait_for(key)
        found = {row_id: cached.lookup(request) for row_id, request in audition_table[key].items()}
        if any(caption is None for caption in found.values()):
            return None
        return {row_id: caption for row_id, caption in found.items() if caption is not None}

    excerpts = {entry["excerpt_id"]: entry for entry in document["followup"]["sound_only"]}

    def _participant(value: str | None) -> str | JSONResponse:
        try:
            return validate_participant(value)
        except SessionLogError as exc:
            return _error(400, str(exc))

    def _page(name: str) -> Response:
        try:
            return HTMLResponse(_package_file(name))
        except FileNotFoundError:
            return _error(404, f"{name} is not part of this build")

    @app.get("/", response_class=HTMLResponse)
    def kiosk() -> Any:
        return _page("kiosk.html")

    @app.get("/followup", response_class=HTMLResponse)
    def followup_page() -> Any:
        return _page("followup.html")

    @app.get("/static/{name}")
    def static(name: str) -> Any:
        media_type = STATIC_FILES.get(name)
        if media_type is None:
            return _error(404, f"no static file {name!r}")
        try:
            return Response(_package_file(name), media_type=media_type)
        except FileNotFoundError:
            return _error(404, f"{name} is not part of this build")

    @app.get("/api/session")
    def session_document() -> Any:
        return narrowed

    @app.get("/api/state")
    def state(participant: str | None = None) -> Any:
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        return {"snapshot": log.snapshot(checked)}

    @app.post("/api/events")
    def events(body: dict[str, Any]) -> Any:
        checked = _participant(body.get("participant"))
        if isinstance(checked, JSONResponse):
            return checked
        batch = body.get("events")
        if not isinstance(batch, list):
            return _error(400, "events must be a list")
        snapshot = body.get("snapshot")
        if snapshot is not None and not isinstance(snapshot, dict):
            return _error(400, "snapshot must be an object or null")
        try:
            appended = log.append(checked, batch, snapshot)
        except SessionLogError as exc:
            return _error(400, str(exc))
        return {"appended": appended}

    def _gated_clip(participant: str | None, clip_id: str) -> Mapping[str, Any] | JSONResponse:
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        clip = clip_by_id(document, clip_id)
        if clip is None:
            return _error(404, f"unknown clip {clip_id!r}")
        if not log.listing_submitted(checked, clip_id):
            return _error(403, "listing not submitted")
        return clip

    @app.get("/api/inventory/{clip_id}")
    def inventory(clip_id: str, participant: str | None = None) -> Any:
        clip = _gated_clip(participant, clip_id)
        if isinstance(clip, JSONResponse):
            return clip
        roles = roles_for(document, clip)
        # An audition that could not be written is the writer failing, and the
        # participant is told so (contract §7), not handed a shot whose held
        # tokens ghost an empty line.
        auditions = {
            str(shot["shot_id"]): _auditions((str(clip["clip_id"]), str(shot["shot_id"])))
            for shot in clip["shots"]
        }
        if any(found is None for found in auditions.values()):
            return _error(502, CAPTION_FAILED)
        return {
            "clip_id": clip["clip_id"],
            "task": clip["task"],
            "poles": poles_for(document, clip),
            "role_heads": roles,
            "shots": [
                {
                    "shot_id": shot["shot_id"],
                    "start_ms": shot["start_ms"],
                    "end_ms": shot["end_ms"],
                    # No weights, no prose: the phrases are the only measured
                    # thing a participant sees, and only as words (spec 2).
                    "sources": [
                        {
                            "id": source["id"],
                            "token": source["token"],
                            "phrases": list(source["phrases"]),
                            "role": source["role"],
                        }
                        for source in shot["sources"]
                    ],
                    "scene": {"token": shot["scene"]["token"]},
                    "atmosphere": {"phrase": shot["atmosphere"]["phrase"]},
                    "orderings": all_orderings(shot, roles),
                    "opening": dict(clip["opening"]),
                    "auditions": auditions[str(shot["shot_id"])],
                }
                for shot in clip["shots"]
            ],
        }

    @app.post("/api/caption")
    def caption(body: dict[str, Any]) -> Any:
        clip_id = str(body.get("clip_id", ""))
        clip = _gated_clip(body.get("participant"), clip_id)
        if isinstance(clip, JSONResponse):
            return clip
        shot = shot_by_id(clip, str(body.get("shot_id", "")))
        if shot is None:
            return _error(404, f"unknown shot {body.get('shot_id')!r} in clip {clip_id!r}")
        settings = body.get("settings")
        if not isinstance(settings, dict):
            return _error(400, "settings must be an object with level, admitted, and order")
        try:
            validated = validate_settings(shot, roles_for(document, clip), settings)
        except SettingsError as exc:
            return _error(400, str(exc))
        try:
            # Cutting the shot's audio or still (Gemma mode) is part of writing the
            # caption: its failure is the same 502 with the Table 6 string, not
            # a bare 500 (contract §7 "On writer failure").
            media = shot_media(clip_id, shot) if shot_media is not None else None
            request = build_request(document, clip, shot, validated, media)
            with background.foreground():
                written, hit = cached.write_attributed_cached(request)
        except (WriterError, MediaError):
            return _error(502, CAPTION_FAILED)
        # What the participant is likeliest to press next, written while they
        # read this one.
        background.urgent(
            build_request(document, clip, shot, near, media)
            for near in neighbours(shot, roles_for(document, clip), validated)
        )
        # ``writer`` and ``names_excluded`` are for the log, not the screen:
        # the page records them on caption.written and shows neither.
        return {
            "caption": written.caption,
            "key": validated.key,
            "cached": hit,
            # A hit is not a revisit if the prefetcher wrote it first; revisits
            # are read from the participant's own request events.
            "prefetched": hit and background.wrote(request),
            "writer": written.writer,
            "names_excluded": list(written.names_excluded),
        }

    @app.get("/api/log")
    def download_log(participant: str | None = None) -> Any:
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        payload = json.dumps(log.export(checked, str(document["session_id"])), ensure_ascii=False, indent=2)
        return Response(
            payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="session-log-{checked}.json"'},
        )

    @app.get("/media/{clip_id}")
    def clip_media(clip_id: str) -> Any:
        # Always the clip WITH its soundtrack: the participant lists and
        # shapes what they hear (same lookup as dpo.userstudy.app).
        if clip_by_id(document, clip_id) is None:
            return _error(404, f"unknown clip {clip_id!r}")
        path = media_tools.find_clip_video(media_dir, clip_id)
        if path is None:
            return _error(404, f"no video with sound for clip {clip_id!r} under {media_dir}")
        return FileResponse(path)

    @app.get("/media/still/{clip_id}")
    def still(clip_id: str) -> Any:
        if clip_by_id(document, clip_id) is None:
            return _error(404, f"unknown clip {clip_id!r}")
        try:
            return FileResponse(
                media_tools.clip_still(media_dir, cache_dir, clip_id), media_type="image/jpeg"
            )
        except MediaError as exc:
            return _error(404 if "no video" in str(exc) else 502, str(exc))

    @app.get("/media/excerpt/{excerpt_id}")
    def excerpt(excerpt_id: str) -> Any:
        entry = excerpts.get(excerpt_id)
        if entry is None:
            return _error(404, f"unknown excerpt {excerpt_id!r}")
        try:
            path = media_tools.excerpt_audio(
                media_dir,
                cache_dir,
                excerpt_id,
                str(entry["clip_id"]),
                int(entry["start_ms"]),
                int(entry["end_ms"]),
            )
        except MediaError as exc:
            return _error(404 if "no video" in str(exc) else 502, str(exc))
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/followup")
    def followup(participant: str | None = None) -> Any:
        checked = _participant(participant)
        if isinstance(checked, JSONResponse):
            return checked
        check: list[dict[str, Any]] = []
        # One read of the snapshot for every shot: the kiosk replacing it
        # mid-request cannot mix two snapshots into one set of own captions.
        snapshot = log.snapshot(checked)
        for entry in document["followup"]["check"]:
            clip = clip_by_id(document, str(entry["clip_id"]))
            assert clip is not None  # validated: every check clip exists
            own: list[dict[str, str]] = []
            automatic: list[dict[str, str]] = []
            for shot in clip["shots"]:
                kept = kept_caption_in(snapshot, clip["clip_id"], shot["shot_id"])
                if kept is None:
                    # The participant-facing string stays (Table 6 voice); the
                    # body names the first shot with no kept caption so the
                    # researcher can tell an unfinished kiosk session from none.
                    if snapshot is None:
                        return _error(409, "no kiosk session for participant")
                    return _error(
                        409,
                        "no kiosk session for participant",
                        missing={"clip_id": clip["clip_id"], "shot_id": shot["shot_id"]},
                    )
                own.append({"shot_id": shot["shot_id"], "caption": kept})
                automatic.append({"shot_id": shot["shot_id"], "caption": shot["automatic_caption"]})
            sides = (own, automatic) if entry["a"] == "own" else (automatic, own)
            check.append({"clip_id": clip["clip_id"], "captions": {"A": sides[0], "B": sides[1]}})
        return {
            "clips": [
                {
                    "clip_id": clip["clip_id"],
                    "shots": [
                        {"shot_id": s["shot_id"], "start_ms": s["start_ms"], "end_ms": s["end_ms"]}
                        for s in clip["shots"]
                    ],
                }
                for clip in document["clips"]
            ],
            "recognition": [
                {"clip_id": entry["clip_id"], "sounds": list(entry["sounds"])}
                for entry in document["followup"]["recognition"]
            ],
            "sound_only": [
                {"excerpt_id": entry["excerpt_id"]} for entry in document["followup"]["sound_only"]
            ],
            "check": check,
        }

    @app.post("/api/followup/responses")
    def followup_responses(body: dict[str, Any]) -> Any:
        if body.get("schema") != FOLLOWUP_SCHEMA:
            return _error(400, f"schema must be {FOLLOWUP_SCHEMA!r}")
        checked = _participant(body.get("participant"))
        if isinstance(checked, JSONResponse):
            return checked
        known_clips = [str(clip["clip_id"]) for clip in document["clips"]]
        recognition = body.get("recognition")
        sound_only = body.get("sound_only")
        check = body.get("check")
        if not all(isinstance(part, list) for part in (recognition, sound_only, check)):
            return _error(400, "recognition, sound_only, and check must be lists")
        assert isinstance(recognition, list) and isinstance(sound_only, list) and isinstance(check, list)
        # A document is complete or refused: every task answered exactly once,
        # each answer drawn from what the page offered, and each entry rebuilt
        # from the contract's keys so nothing else is written as a response.
        sounds_of = {
            str(entry["clip_id"]): [str(sound) for sound in entry["sounds"]]
            for entry in document["followup"]["recognition"]
        }
        recognized = _covering("recognition", recognition, "clip_id", list(sounds_of))
        if isinstance(recognized, JSONResponse):
            return recognized
        recognition_rows: list[dict[str, Any]] = []
        for clip_id, entry in recognized.items():
            checked_sounds = entry.get("checked")
            if not isinstance(checked_sounds, list) or len(set(map(str, checked_sounds))) != len(
                checked_sounds
            ):
                return _error(400, f"recognition[{clip_id}].checked must be a list without repeats")
            if any(sound not in sounds_of[clip_id] for sound in checked_sounds):
                return _error(400, f"recognition[{clip_id}].checked names a sound the clip's list does not")
            recognition_rows.append({"clip_id": clip_id, "checked": [str(sound) for sound in checked_sounds]})
        heard = _covering("sound_only", sound_only, "excerpt_id", list(excerpts))
        if isinstance(heard, JSONResponse):
            return heard
        sound_only_rows: list[dict[str, Any]] = []
        for excerpt_id, entry in heard.items():
            if entry.get("chosen_clip_id") not in known_clips:
                return _error(400, f"sound_only[{excerpt_id}].chosen_clip_id is not a clip of this session")
            if not _count(entry.get("played")):
                return _error(400, f"sound_only[{excerpt_id}].played must be a non-negative integer")
            sound_only_rows.append(
                {
                    "excerpt_id": excerpt_id,
                    "chosen_clip_id": entry["chosen_clip_id"],
                    "played": entry["played"],
                }
            )
        shaped = [str(entry["clip_id"]) for entry in document["followup"]["check"]]
        compared = _covering("check", check, "clip_id", shaped)
        if isinstance(compared, JSONResponse):
            return compared
        check_rows: list[dict[str, Any]] = []
        for clip_id, entry in compared.items():
            if entry.get("chosen") not in ("A", "B"):
                return _error(400, f"check[{clip_id}].chosen must be 'A' or 'B'")
            if not _count(entry.get("toggles")):
                return _error(400, f"check[{clip_id}].toggles must be a non-negative integer")
            check_rows.append({"clip_id": clip_id, "chosen": entry["chosen"], "toggles": entry["toggles"]})
        try:
            saved = log.write_followup(
                checked,
                {
                    "schema": FOLLOWUP_SCHEMA,
                    "session_id": document["session_id"],
                    "participant": checked,
                    "recognition": recognition_rows,
                    "sound_only": sound_only_rows,
                    "check": check_rows,
                },
            )
        except FollowupExistsError as exc:
            # The first submission is the record (contract §6's append-only
            # rule, applied to the one document the follow-up writes).
            return _error(409, str(exc))
        return {"saved": str(saved)}

    return app


def run_session_app(
    *,
    session_path: Path,
    media_dir: Path,
    out_dir: Path,
    writer: CaptionWriter,
    shot_media: ShotMedia | None = None,
    host: str = "127.0.0.1",
    port: int = 8777,
) -> None:
    document = load_session_document(session_path)
    app = build_app(document, media_dir, out_dir, writer, shot_media=shot_media)
    uvicorn.run(app, host=host, port=port, log_level="warning")
