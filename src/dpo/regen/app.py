"""The instrument's HTTP surface: what the page may ask for, and when.

One FastAPI app serves the page and everything it fetches. Four rules shape it,
all consequences of the specification rather than of HTTP.

*The order is enforced here.* Every route that advances the session names its
step and calls :func:`dpo.regen.progress.require` first, so §9.1's "back
navigation and re-entry to completed steps blocked" holds against a reload, a
second tab and a typed URL, none of which a page can prevent. A submission out
of order is a 409 — the state is wrong, not the request.

*The masks never reach the browser.* §4 matches once, on the coordinates
submitted with the page. A browser holding the masks could match on every
click, which would put the segmentation's answer in front of a participant who
is still deciding. So ``/api/visual`` takes coordinates and returns nothing
about them; the matched labels go to the log and to §6.

*The regenerated track is written on the server and returned once.* §6 runs
inside ``POST /api/regenerate``, which is the only route that can be slow. It
is idempotent per participant: a reload during the wait returns the track that
was already written rather than spending the model a second time on the same
report — and rather than giving the participant a second, different track that
their §8 answers would then be about.

*Which segment is which is never in the document.* The assignment comes from
the participant's sequence number (§1) and is applied here, so one document
serves everyone and the page is told only "your clip for this step".

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import formatdate, parsedate_to_datetime
from functools import wraps
from importlib.resources import files
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from dpo.caption.writer import CachedWriter, CaptionWriter
from dpo.regen import progress
from dpo.regen.assignment import PREPARED, REGENERATED, Assignment
from dpo.regen.captions import Cue, cues_of, record_of
from dpo.regen.config import FAMILY_OF_PARENT, SOUND_FAMILIES
from dpo.regen.copy import strings_for
from dpo.regen.derive import Derivatives
from dpo.regen.document import (
    configuration_of,
    frames_of,
    objects_of,
    segment_of,
    track_of,
    validate_regen_document,
)
from dpo.regen.enrolment import Roster
from dpo.regen.gate import Buckets, Gate, client_address, is_local
from dpo.regen.items import ItemsError, ItemSet, load_items
from dpo.regen.log import EventLog, RegenLogError, validate_participant, view_id
from dpo.regen.points import PointError, match_points, matched_labels, parse_points, summary_of
from dpo.regen.progress import ProgressError
from dpo.regen.regeneration import Report, regenerate

STATIC_FILES = {
    "identity.css": "text/css; charset=utf-8",
    "regen.css": "text/css; charset=utf-8",
    "regen.js": "text/javascript; charset=utf-8",
}
CACHE_FILE = "captions.json"
# What a browser may keep, and for how long.
#
# The media of a running study does not change under the participant looking at
# it, and §4 is a screen they move back and forth across: five moments, each a
# picture, revisited as they decide where the marks go. Re-fetching one on every
# return is the difference between a strip that responds and a strip that waits,
# so the media carries a day and an entity tag. `private` because a participant's
# clip is nobody else's to hold; there is no shared cache on the published path
# anyway, and this says so rather than relying on it.
MEDIA_CACHE = "private, max-age=86400"
# The page's own code is the opposite case: a fix has to reach the next reload,
# not the reload after the cache expires. `no-cache` is not "do not store" — it
# stores and revalidates, so an unchanged file costs a 304 and no body.
SHELL_CACHE = "no-cache"
# The step a viewing belongs to, and the index it is filed under (§9.5).
VIEWS = {
    progress.VIEW_PREPARED: (0, PREPARED),
    progress.VIEW_REGENERATED: (1, REGENERATED),
}
# Which survey page joins to which viewing (§9.5): §3 is about the first
# viewing, §8 about the second.
SURVEY_PAGES = {"art": 0, "survey": 1}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _package_file(name: str) -> str:
    """Read one file shipped inside ``dpo.regen``; patched by tests."""
    return files("dpo.regen").joinpath(name).read_text(encoding="utf-8")


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


def _tag(material: str) -> str:
    """An entity tag over whatever identifies a body."""
    return f'"{hashlib.md5(material.encode(), usedforsecurity=False).hexdigest()}"'


def _unchanged(request: Request, etag: str, modified: float | None = None) -> bool:
    """Whether the browser already holds this exact body.

    ``FileResponse`` sends an entity tag but reads none: a conditional request
    for a still that has not changed came back as the whole PNG again, which is
    the same as having no cache at all. This is the half that was missing.
    ``If-None-Match`` wins over ``If-Modified-Since`` where both are sent (RFC
    9110 §13.1.3), and a weak tag matches its strong twin because the
    comparison a conditional GET calls for is the weak one.
    """
    matches = request.headers.get("if-none-match")
    if matches is not None:
        held = {candidate.strip() for candidate in matches.split(",")}
        weak = etag[2:] if etag.startswith("W/") else etag
        return "*" in held or any((tag[2:] if tag.startswith("W/") else tag) == weak for tag in held)
    since = request.headers.get("if-modified-since")
    if since is not None and modified is not None:
        try:
            asked = parsedate_to_datetime(since)
        except (TypeError, ValueError):
            return False
        # HTTP dates have a second's resolution, so a file written within the
        # same second as the one held would compare as newer forever.
        return asked is not None and int(modified) <= int(asked.timestamp())
    return False


def build_app(
    document: Mapping[str, Any],
    media_dir: Path,
    out_dir: Path,
    writer: CaptionWriter,
    *,
    items: ItemSet | None = None,
    watch: Callable[[str, int, int], None] | None = None,
    gate: Gate | None = None,
    derive: bool = True,
) -> FastAPI:
    """The app over one validated document.

    ``watch`` is told ``(participant, slots written, slots in the track)`` as
    §6 runs, for whatever the operator is looking at — the CLI hands it a
    terminal bar. It is the same report the waiting screen polls for, so the
    console and the participant cannot disagree about where the model is.

    ``gate`` is what stands in front of the instrument when it is published
    (:mod:`dpo.regen.gate`): an access code on enrolment, the log download
    kept to this machine, a rate limit by address. None of it is on by
    default, so a kiosk run is unchanged.

    ``derive`` is whether the browser is served the web-sized copies of
    :mod:`dpo.regen.derive` rather than the archival files themselves. On by
    default because the published link is the case that needs it; off restores
    the byte-for-byte staging to every response.
    """
    validate_regen_document(document)
    configuration = configuration_of(document)
    item_set = items if items is not None else load_items()
    app = FastAPI(title="dpo caption regen", docs_url=None, redoc_url=None, openapi_url=None)
    gate = gate or Gate()
    requests = Buckets(gate.requests_per_second, gate.burst) if gate.requests_per_second else None
    enrolments = (
        Buckets(gate.enrolments_per_minute / 60.0, max(1, round(gate.enrolments_per_minute)))
        if gate.enrolments_per_minute
        else None
    )

    if requests is not None:

        @app.middleware("http")
        async def rate_limit(request: Request, call_next: Callable[..., Any]) -> Any:
            if not requests.allow(client_address(request)):
                return _error(429, "too many requests from this address; slow down")
            return await call_next(request)

    media_dir = Path(media_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    derivatives = Derivatives(media_dir, enabled=derive)
    log = EventLog(out_dir, configuration.hash)
    roster = Roster(out_dir)
    cached = writer if isinstance(writer, CachedWriter) else CachedWriter(writer, out_dir / CACHE_FILE)
    # How far §6 has got, per participant, while it is running. In memory and
    # per process on purpose: this is an observation of a call in flight, not
    # a measure, and a restart that loses it loses nothing an analysis wanted.
    # Everything §6 actually records goes into the log with the regeneration.
    writing: dict[str, dict[str, int]] = {}
    session_locks: dict[str, Any] = {}
    session_locks_guard = threading.Lock()

    def session_write(endpoint: Callable[..., Any]) -> Callable[..., Any]:
        """Keep legacy read/check/write transitions whole within the supported process."""

        @wraps(endpoint)
        def locked(*args: Any, **kwargs: Any) -> Any:
            payload = kwargs.get("payload") or {}
            identifier = payload.get("participant", "") if isinstance(payload, Mapping) else ""
            # Invalid input stays on one bounded lock and is rejected by the endpoint.
            if not isinstance(identifier, str) or len(identifier) > 64:
                identifier = ""
            with session_locks_guard:
                lock = session_locks.setdefault(identifier, threading.RLock())
            with lock:
                return endpoint(*args, **kwargs)

        return locked

    def _assignment(participant: str) -> Assignment | JSONResponse:
        sequence = roster.sequence_of(participant)
        if sequence is None:
            return _error(404, f"no participant {participant!r} has been enrolled")
        _, assignment = roster.enrol(participant)
        return assignment

    def _participant(value: str | None) -> str | JSONResponse:
        try:
            return validate_participant(value)
        except RegenLogError as exc:
            return _error(400, str(exc))

    def _media(reference: str) -> Path | JSONResponse:
        """Resolve a document reference under the media directory, never above it."""
        resolved = (media_dir / reference).resolve()
        root = media_dir.resolve()
        if not resolved.is_relative_to(root):
            return _error(400, "media reference escapes the media directory")
        if not resolved.is_file():
            return _error(404, f"{reference} is not staged under {media_dir}")
        return resolved

    def _gate(participant: str, step: str) -> Mapping[str, Any] | JSONResponse:
        snapshot = log.snapshot(participant) or {"step": progress.VIEW_PREPARED}
        try:
            progress.require(snapshot, step)
        except ProgressError as exc:
            return _error(409, str(exc), step=progress.current(snapshot))
        return snapshot

    def _language(participant: str) -> str:
        """The language this session is being read in."""
        chosen = (log.snapshot(participant) or {}).get("language")
        return str(chosen) if configuration.calibration.offers(chosen) else configuration.language

    def _started_viewing(participant: str) -> bool:
        """Whether a clip has already played, which is what locks the language."""
        return bool(log.viewings(participant))

    def _slots(segment: str) -> tuple[Cue, ...]:
        """The fixed slots §6 writes into: the segment's own prepared timings."""
        return track_of(document, segment, "prepared_track")

    @app.get("/", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        return HTMLResponse(_package_file("regen.html"))

    @app.get("/api/strings")
    def strings() -> Any:
        """The chrome, in every language this study offers.

        All of them at once, keyed by tag, rather than the one the participant
        is currently reading. The page switches language without a round trip
        — §9.3 lets them switch until the first clip plays, and a fetch between
        the press and the redraw is a stutter on a control whose whole job is
        to be reversible — and boot still asks for the copy and the enrolment
        together, which it could not do if the copy depended on the enrolment's
        answer. Two languages of chrome is a few kilobytes.
        """
        return {
            "strings": {tag: strings_for(tag) for tag in configuration.languages},
            "scale": {
                "points": configuration.scale.points,
                "anchors": list(configuration.scale.anchors),
            },
            "minimum_points": configuration.calibration.minimum_points,
            # §6 names its worst case on the waiting screen. An indeterminate
            # bar is right — nothing on the page can predict the model — but
            # indeterminate had been implemented as silent, and a bounded wait
            # the participant is told about is a different wait from one they
            # are not.
            "latency_ceiling_ms": configuration.calibration.latency_ceiling_ms,
            # §9.4's slot count, so §6 can draw one cell per cue before its
            # first progress report arrives rather than growing a bar.
            "cue_slots": configuration.cue_slots,
            "steps": list(progress.PARTICIPANT_STEPS),
            "languages": list(configuration.languages),
        }

    @app.post("/api/session")
    @session_write
    def session(request: Request, payload: Mapping[str, Any] | None = None) -> Any:
        """§1: issue an identifier and fix the assignment, once, at Page 1 load."""
        asked = (payload or {}).get("participant")
        if asked is not None and not isinstance(asked, str):
            return _error(400, "participant must be a string if it is given")
        # A reload resumes without a code; only a new enrolment is gated,
        # because that is the request that spends a sequence number.
        try:
            resuming = asked is not None and roster.sequence_of(asked) is not None
        except RegenLogError as exc:
            return _error(400, str(exc))
        if not resuming:
            if gate.requires_code:
                expected = gate.code()
                given = (payload or {}).get("code")
                if expected is None:
                    return _error(503, "the study is not open right now", closed=True)
                if not isinstance(given, str) or given != expected:
                    return _error(403, "this link is not active", closed=True)
            if enrolments is not None and not enrolments.allow(client_address(request)):
                return _error(429, "too many new sessions from this address; try again in a minute")
        try:
            participant, assignment = roster.enrol(asked)
        except (RegenLogError, ValueError) as exc:
            return _error(400, str(exc))
        snapshot = log.snapshot(participant)
        if snapshot is None:
            snapshot = {"schema": "dpo.caption-regen-snapshot/v1", "step": progress.VIEW_PREPARED}
            log.write_snapshot(participant, snapshot)
            log.append(participant, [{"type": "session.enrolled", **assignment.record()}], snapshot)
        return {
            "participant": participant,
            "assignment": assignment.record(),
            "step": progress.current(snapshot),
            "session_id": document["session_id"],
            "config_hash": configuration.hash,
            "items_provenance": item_set.provenance,
            "language": _language(participant),
            "languages": list(configuration.languages),
            "language_locked": _started_viewing(participant),
            # Whether the done screen may offer the log. Off the study machine
            # the route below refuses, and a button that leads to a refusal
            # would replace the kiosk's last screen with an error body.
            "download": not gate.log_local_only or is_local(request),
        }

    @app.post("/api/language")
    @session_write
    def language(payload: Mapping[str, Any]) -> Any:
        """Which of the study's languages this participant reads (§9.3).

        Refused once a clip has played. The measure is the change in the ART
        answers between §3 and §8, and a session read half in one language and
        half in the other has moved something else as well.
        """
        person = _participant(payload.get("participant"))
        if isinstance(person, JSONResponse):
            return person
        wanted = payload.get("language")
        if not configuration.calibration.offers(wanted):
            return _error(400, f"this study offers {list(configuration.languages)}", asked=wanted)
        if _started_viewing(person):
            return _error(
                409,
                "the language is fixed once the first clip has played; both viewings are read "
                "in one language (§9.3)",
                language=_language(person),
            )
        snapshot = {**(log.snapshot(person) or {"step": progress.VIEW_PREPARED}), "language": wanted}
        log.append(person, [{"type": "language.chosen", "language": wanted}], snapshot)
        return {"language": str(wanted), "language_locked": False}

    @app.get("/api/step/{step}")
    def step_detail(step: str, participant: str | None = None) -> Any:
        """What one step needs to render, and nothing the step after it needs."""
        person = _participant(participant)
        if isinstance(person, JSONResponse):
            return person
        assignment = _assignment(person)
        if isinstance(assignment, JSONResponse):
            return assignment
        gated = _gate(person, step)
        if isinstance(gated, JSONResponse):
            return gated
        if step in VIEWS:
            _, condition = VIEWS[step]
            segment = assignment.segment_of(condition)
            captions = _prepared_or_regenerated(person, condition, segment)
            if isinstance(captions, JSONResponse):
                return captions
            return {
                "step": step,
                "condition": condition,
                "segment": segment,
                "clip_id": segment_of(document, segment)["clip_id"],
                "duration_ms": segment_of(document, segment)["duration_ms"],
                "captions": record_of(captions, _language(person)),
            }
        if step == progress.VISUAL:
            segment = assignment.prepared_segment
            # The strip, and when each frame is from. The masks stay here: §4
            # matches once, on submit, and a page holding them could match on
            # every click.
            return {
                "step": step,
                "segment": segment,
                "minimum": configuration.calibration.minimum_points,
                "frames": [
                    {"index": index, "at_ms": frame["at_ms"]}
                    for index, frame in enumerate(frames_of(document, segment))
                ],
            }
        if step == progress.AUDITORY:
            segment = assignment.prepared_segment
            # The five families, in one fixed order, rather than the sources
            # this clip happens to carry: §5 asks the same question of every
            # participant about every family, so a family that is not in the
            # clip is answerable and a false alarm is a measure. The labels are
            # not sent — they are chrome, the page already holds both languages
            # of them, and sending them here would be a second copy to drift.
            entry = segment_of(document, segment)
            return {
                "step": step,
                "segment": segment,
                "duration_ms": entry["duration_ms"],
                "families": list(SOUND_FAMILIES),
            }
        if step in SURVEY_PAGES:
            return {
                "step": step,
                "blocks": [block.record(_language(person)) for block in item_set.page_blocks(step)],
                "scale": {
                    "points": configuration.scale.points,
                    "anchors": list(configuration.scale.anchors),
                },
            }
        return _error(404, f"no step named {step!r}")

    def _prepared_or_regenerated(
        participant: str, condition: str, segment: str
    ) -> tuple[Cue, ...] | JSONResponse:
        if condition == PREPARED:
            return track_of(document, segment, "prepared_track")
        snapshot = log.snapshot(participant) or {}
        stored = snapshot.get("regenerated")
        if not isinstance(stored, list):
            return _error(409, "the regenerated track has not been written yet")
        return cues_of(stored, configuration.language)

    @app.post("/api/viewing")
    @session_write
    def viewing(payload: Mapping[str, Any]) -> Any:
        """§2 and §7: the viewing row, written when playback ends (§9.5)."""
        person = _participant(payload.get("participant"))
        if isinstance(person, JSONResponse):
            return person
        step = payload.get("step")
        if step not in VIEWS:
            return _error(400, f"step must be one of {sorted(VIEWS)}")
        gated = _gate(person, str(step))
        if isinstance(gated, JSONResponse):
            return gated
        assignment = _assignment(person)
        if isinstance(assignment, JSONResponse):
            return assignment
        index, condition = VIEWS[str(step)]
        reading = _language(person)
        segment = assignment.segment_of(condition)
        captions = _prepared_or_regenerated(person, condition, segment)
        if isinstance(captions, JSONResponse):
            return captions
        started, ended = payload.get("started_at"), payload.get("ended_at")
        if not isinstance(started, str) or not isinstance(ended, str):
            return _error(400, "started_at and ended_at are required timestamps")
        key = log.record_viewing(
            person,
            index=index,
            condition=condition,
            segment=segment,
            clip_id=str(segment_of(document, segment)["clip_id"]),
            captions=record_of(captions, reading),
            started_at=started,
            ended_at=ended,
            # The language is on the viewing, not only in the caption text: a
            # row should say what a participant read without anyone having to
            # look at the characters to work it out.
            extra={**assignment.record(), "language": reading},
        )
        log.append(
            person,
            [{"type": "viewing.ended", "view_id": key, "step": step}],
            progress.advance(gated, str(step)),
        )
        return {"view_id": key, "step": progress.current(log.snapshot(person))}

    @app.post("/api/survey")
    @session_write
    def survey(payload: Mapping[str, Any]) -> Any:
        """§3 and §8: per-item responses, joined to a viewing by key (§9.5)."""
        person = _participant(payload.get("participant"))
        if isinstance(person, JSONResponse):
            return person
        page_name = payload.get("page")
        if page_name not in SURVEY_PAGES:
            return _error(400, f"page must be one of {sorted(SURVEY_PAGES)}")
        gated = _gate(person, str(page_name))
        if isinstance(gated, JSONResponse):
            return gated
        responses = payload.get("responses")
        if not isinstance(responses, Mapping):
            return _error(400, "responses must be an object of item id to answer")
        try:
            expected = item_set.item_ids(str(page_name))
        except ItemsError as exc:
            return _error(500, str(exc))
        missing = [item for item in expected if item not in responses]
        if missing:
            return _error(400, "every item must be answered before submit (§3, §8)", missing=missing)
        unknown = [item for item in responses if item not in expected]
        if unknown:
            return _error(400, "the submission answers items this page does not ask", unknown=unknown)
        bad = [item for item, value in responses.items() if not configuration.scale.accepts(value)]
        if bad:
            return _error(400, f"answers are integers in 1..{configuration.scale.points}", invalid=bad)
        entered, submitted = payload.get("entered_at"), payload.get("submitted_at")
        if not isinstance(entered, str) or not isinstance(submitted, str):
            return _error(400, "entered_at and submitted_at are required timestamps")
        key = view_id(person, SURVEY_PAGES[str(page_name)])
        log.record_responses(
            person,
            page=str(page_name),
            key=key,
            responses=dict(responses),
            entered_at=entered,
            submitted_at=submitted,
            items_digest=item_set.digest,
            items_provenance=item_set.provenance,
        )
        log.append(
            person,
            [{"type": "survey.submitted", "page": page_name, "view_id": key}],
            progress.advance(gated, str(page_name)),
        )
        return {"view_id": key, "step": progress.current(log.snapshot(person))}

    @app.post("/api/visual")
    @session_write
    def visual(payload: Mapping[str, Any]) -> Any:
        """§4: match the submitted points once, and keep what matched nothing."""
        person = _participant(payload.get("participant"))
        if isinstance(person, JSONResponse):
            return person
        gated = _gate(person, progress.VISUAL)
        if isinstance(gated, JSONResponse):
            return gated
        assignment = _assignment(person)
        if isinstance(assignment, JSONResponse):
            return assignment
        segment = assignment.prepared_segment
        try:
            points = parse_points(payload.get("points"), len(frames_of(document, segment)))
        except PointError as exc:
            return _error(400, str(exc))
        minimum = configuration.calibration.minimum_points
        if len(points) < minimum:
            return _error(400, f"§4 requires at least {minimum} points; {len(points)} were submitted")
        try:
            matches = match_points(points, objects_of(document, media_dir, segment))
        except PointError as exc:
            return _error(500, str(exc))
        summary = summary_of(matches)
        snapshot = {**gated, "visual_labels": list(matched_labels(matches))}
        log.append(
            person,
            [
                {
                    "type": "visual.submitted",
                    "segment": segment,
                    "points": [match.record() for match in matches],
                    # Nested, not spread: the summary's own "points" is a count
                    # and would take the key the records are under.
                    "summary": summary,
                }
            ],
            progress.advance(snapshot, progress.VISUAL),
        )
        return {"step": progress.current(log.snapshot(person)), **summary}

    @app.post("/api/auditory")
    @session_write
    def auditory(payload: Mapping[str, Any]) -> Any:
        """§5: heard or did not hear, on each of the five fixed sound families.

        Every family is required. A blank is not "did not hear" — it is a
        participant who did not answer — and the two have to stay distinct or
        §6 is conditioned on the difference between a denial and a shrug. The
        page cannot submit until all five are answered; this refuses the
        request that gets past it anyway.
        """
        person = _participant(payload.get("participant"))
        if isinstance(person, JSONResponse):
            return person
        gated = _gate(person, progress.AUDITORY)
        if isinstance(gated, JSONResponse):
            return gated
        assignment = _assignment(person)
        if isinstance(assignment, JSONResponse):
            return assignment
        segment = assignment.prepared_segment
        raw = payload.get("heard")
        if not isinstance(raw, Mapping):
            return _error(400, "heard must be an object of sound family to true or false")
        unknown = sorted(key for key in raw if key not in SOUND_FAMILIES)
        if unknown:
            return _error(400, "heard names families the study does not ask about", unknown=unknown)
        malformed = sorted(key for key, value in raw.items() if not isinstance(value, bool))
        if malformed:
            return _error(400, "each family is answered true or false", invalid=malformed)
        missing = [family for family in SOUND_FAMILIES if family not in raw]
        if missing:
            return _error(400, "every sound family has to be answered", missing=missing)
        # The order is the study's, not the page's: a report is a set of
        # judgments made at once, and there is no selection order to preserve
        # now that the screen is not a sequence of choices.
        heard = [family for family in SOUND_FAMILIES if raw[family]]
        # What §6 is conditioned on is the family's prose form, not the label
        # the participant happened to read: the display label is chrome and
        # changes with the interface language, and two participants reporting
        # the same families must hand the writer the same input whichever
        # language they read the screen in. It is also the form that makes a
        # sentence — "sounds of things" is a taxonomy node, not a caption.
        snapshot = {
            **gated,
            "auditory_labels": [SOUND_FAMILIES[family] for family in heard],
            "auditory_ids": list(heard),
        }
        log.append(
            person,
            [
                {
                    "type": "auditory.submitted",
                    "segment": segment,
                    "heard": heard,
                    "not_heard": [family for family in SOUND_FAMILIES if not raw[family]],
                    # What the clip actually carries, in the same vocabulary
                    # as the two fields above, so a false alarm is readable in
                    # the log without joining to the document — which is the
                    # whole reason this field is here, and which it could not
                    # do while it held the document's AudioSet names against
                    # §5's family keys.
                    "present": sorted(
                        {
                            FAMILY_OF_PARENT[parent]
                            for stem in segment_of(document, segment)["stems"]
                            if (parent := stem.get("parent")) in FAMILY_OF_PARENT
                        }
                    ),
                    # A family the clip carries that §5 does not ask about is
                    # not a false alarm the participant could have made, and
                    # dropping it silently would leave the log looking as
                    # though the clip held nothing else.
                    "present_unasked": sorted(
                        {
                            str(parent)
                            for stem in segment_of(document, segment)["stems"]
                            if (parent := stem.get("parent")) and parent not in FAMILY_OF_PARENT
                        }
                    ),
                }
            ],
            progress.advance(snapshot, progress.AUDITORY),
        )
        return {"step": progress.current(log.snapshot(person)), "heard": heard}

    @app.post("/api/regenerate")
    @session_write
    def regenerate_track(payload: Mapping[str, Any]) -> Any:
        """§6: write the second viewing's track, once per participant."""
        person = _participant(payload.get("participant"))
        if isinstance(person, JSONResponse):
            return person
        snapshot = log.snapshot(person) or {}
        stored = snapshot.get("regenerated")
        if isinstance(stored, list):
            # A reload during the wait: return what was written rather than
            # spending the model again on the same report, and rather than
            # handing the participant a second, different track.
            return {
                "track": record_of(cues_of(stored, configuration.language), _language(person)),
                "fallback": bool(snapshot.get("regeneration_fallback")),
                "cached": True,
            }
        gated = _gate(person, progress.REGENERATING)
        if isinstance(gated, JSONResponse):
            return gated
        assignment = _assignment(person)
        if isinstance(assignment, JSONResponse):
            return assignment
        segment = assignment.regenerated_segment
        report = Report(
            visual_labels=tuple(snapshot.get("visual_labels") or ()),
            auditory_labels=tuple(snapshot.get("auditory_labels") or ()),
            auditory_ids=tuple(snapshot.get("auditory_ids") or ()),
        )
        reading = _language(person)
        # The clip's sound, staged as a wav beside it. Handing the writer the
        # mp4 instead put the container in front of the model's audio stack,
        # which decodes wav and little else without torchcodec — and none of
        # torchcodec's wheels link against this torch. Every slot failed, and
        # the participant got a fallback track with nothing on screen to say so.
        sound = _media(str(segment_of(document, segment)["audio"]))

        def _wrote(done: int, total: int, *, person: str = person) -> None:
            writing[person] = {"done": done, "total": total}
            if watch is not None:
                watch(person, done, total)

        try:
            result = regenerate(
                cached,
                configuration,
                clip_id=str(segment_of(document, segment)["clip_id"]),
                slots=_slots(segment),
                fallback=track_of(document, segment, "fallback_track"),
                report=report,
                language=reading,
                media=sound if isinstance(sound, Path) else None,
                settings={"segment": segment, "config_hash": configuration.hash, "language": reading},
                on_slot=_wrote,
            )
        finally:
            # Whether it finished, fell back or raised: nothing is in flight
            # for this participant now, and a stale count on the waiting screen
            # would outlive the thing it was counting.
            writing.pop(person, None)
        # Stored under the document's own shape — text keyed by language — so
        # reading it back needs nothing but the track itself.
        track = [
            {"index": cue.index, "start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": dict(cue.text)}
            for cue in result.cues
        ]
        advanced = {
            **progress.advance(gated, progress.REGENERATING),
            "regenerated": track,
            "regeneration_fallback": result.fallback,
        }
        log.append(person, [{"type": "regeneration.written", **result.record()}], advanced)
        return {
            "track": record_of(result.cues, reading),
            "fallback": result.fallback,
            "cached": False,
        }

    @app.get("/api/regenerate/progress")
    def regenerate_progress(participant: str | None = None) -> Any:
        """How far §6 has got. Read by the screen that is waiting on it (§6).

        Slots written out of slots in the track, and nothing else. The waiting
        screen is deliberately indeterminate about *time* — nothing here can
        predict the model — but a cue that has been written is an event that
        happened, and a screen that can say four of them are coming and two
        have arrived is not predicting anything.

        Best-effort, like the event stream: a participant whose regeneration
        has not started yet, or has just finished, reads as not writing, and
        the page falls back to what it already shows.

        It also names the clip the next viewing will play. §6 is the one wait
        in the session that is already a wait — the participant is watching an
        indeterminate bar while the model writes — and the second clip is 5–7
        MB that would otherwise be fetched afterwards, from a standing start,
        behind *Preparing the clip…*. Fetching it here costs the participant
        nothing they are not already spending. This is the same fact
        ``/api/step/view_regenerated`` returns a moment later, so nothing is
        disclosed earlier than the assignment already decides — and the media
        routes were never gated in the first place. Omitted rather than an
        error where no assignment exists yet, because a progress display may
        not be the thing that ends a session.
        """
        person = _participant(participant)
        if isinstance(person, JSONResponse):
            return person
        at = writing.get(person)
        report: dict[str, Any] = {
            "writing": at is not None,
            "done": at["done"] if at else 0,
            "total": at["total"] if at else configuration.cue_slots,
        }
        assignment = _assignment(person)
        if not isinstance(assignment, JSONResponse):
            report["next_segment"] = assignment.regenerated_segment
        return report

    @app.post("/api/events")
    @session_write
    def events(payload: Mapping[str, Any]) -> Any:
        """The stream the page keeps: points moved, lanes played, steps entered."""
        person = _participant(payload.get("participant"))
        if isinstance(person, JSONResponse):
            return person
        # Every other route that writes goes through the roster first; this one
        # did not, so a well-formed identifier nobody enrolled opened its own
        # `events-<p>.jsonl`. A file in --out with no roster entry and no
        # viewing beside it is a record of a session that never happened, and
        # an analyst reading the directory has no way to know that.
        if roster.sequence_of(person) is None:
            return _error(404, f"no participant {person!r} has been enrolled")
        batch = payload.get("events")
        if not isinstance(batch, Sequence) or isinstance(batch, str):
            return _error(400, "events must be a list")
        try:
            # The snapshot is the server's; a page cannot move its own step by
            # posting one, so only the events are taken from the payload.
            written = log.append(person, list(batch), None)
        except RegenLogError as exc:
            return _error(400, str(exc))
        return {"written": written}

    @app.get("/api/state")
    def state(participant: str | None = None) -> Any:
        person = _participant(participant)
        if isinstance(person, JSONResponse):
            return person
        snapshot = log.snapshot(person)
        return {"step": progress.current(snapshot), "participant": person}

    @app.get("/api/log")
    def download(request: Request, participant: str | None = None) -> Any:
        if gate.log_local_only and not is_local(request):
            return _error(403, "the session log is downloaded on the study machine, not from here")
        person = _participant(participant)
        if isinstance(person, JSONResponse):
            return person
        # The done screen navigates here; as an attachment the browser saves
        # the file and stays on the screen, where a bare JSON body would
        # replace the kiosk with the log's text — which is the state a
        # fullscreen kiosk cannot be brought back from.
        return JSONResponse(
            log.export(person, str(document["session_id"])),
            headers={"Content-Disposition": f'attachment; filename="regen-log-{person}.json"'},
        )

    def _serve(request: Request, path: Path, media_type: str | None = None) -> Response:
        """A media file, with the two headers that decide whether it is fetched twice."""
        try:
            stat = path.stat()
        except OSError:
            return _error(404, f"{path.name} is no longer readable")
        etag = _tag(f"{stat.st_mtime_ns}-{stat.st_size}")
        headers = {
            "cache-control": MEDIA_CACHE,
            "etag": etag,
            "last-modified": formatdate(stat.st_mtime, usegmt=True),
        }
        if _unchanged(request, etag, stat.st_mtime):
            return Response(status_code=304, headers=headers)
        return FileResponse(path, media_type=media_type, headers=headers)

    @app.get("/media/video/{segment}")
    def video(request: Request, segment: str) -> Any:
        return _file_response(request, segment, "video")

    @app.get("/media/frame/{segment}/{index}")
    def frame(request: Request, segment: str, index: int) -> Any:
        """One frame of §4's strip. The masks for it never leave the server.

        What leaves is a WebP of the staged still at its own size — the picture
        the page draws, not the archival PNG it was cut from. §4's coordinates
        are normalised to the delivered image and the matching reads the
        originals here, so the record is the same either way.
        """
        try:
            strip = frames_of(document, segment)
        except ValueError as exc:
            return _error(404, str(exc))
        if not 0 <= index < len(strip):
            return _error(404, f"segment {segment} has no frame {index}")
        resolved = _media(str(strip[index]["still"]))
        if isinstance(resolved, JSONResponse):
            return resolved
        path, media_type = derivatives.still(resolved)
        return _serve(request, path, media_type)

    @app.get("/media/stem/{segment}/{stem_id}")
    def stem(request: Request, segment: str, stem_id: str) -> Any:
        try:
            entry = segment_of(document, segment)
        except ValueError as exc:
            return _error(404, str(exc))
        for candidate in entry["stems"]:
            if candidate["id"] == stem_id:
                resolved = _media(str(candidate["audio"]))
                return resolved if isinstance(resolved, JSONResponse) else _serve(request, resolved)
        return _error(404, f"segment {segment} has no stem {stem_id!r}")

    def _file_response(request: Request, segment: str, key: str) -> Any:
        try:
            entry = segment_of(document, segment)
        except ValueError as exc:
            return _error(404, str(exc))
        resolved = _media(str(entry[key]))
        return resolved if isinstance(resolved, JSONResponse) else _serve(request, resolved)

    @app.get("/{filename}")
    def static_file(request: Request, filename: str) -> Response:
        media_type = STATIC_FILES.get(filename)
        if media_type is None:
            return _error(404, f"no file {filename!r}")
        body = _package_file(filename)
        etag = _tag(body)
        headers = {"cache-control": SHELL_CACHE, "etag": etag}
        if _unchanged(request, etag):
            return Response(status_code=304, headers=headers)
        return Response(body, media_type=media_type, headers=headers)

    return app


def warm_derivatives(document: Mapping[str, Any], media_dir: Path) -> tuple[int, int]:
    """Make every web-sized copy the document will ask for, before anyone asks.

    Encoding on first request would put the cost on a participant — the first
    one through §4 would wait out five WebP encodes that nobody after them
    waits for. Doing it at startup costs the operator a second and makes the
    instrument's behaviour the same for the first session as for the tenth.
    """
    derivatives = Derivatives(Path(media_dir))
    stills = [
        still
        for name in document["segments"]
        for moment in frames_of(document, name)
        if (still := Path(media_dir) / str(moment["still"])).is_file()
    ]
    return derivatives.warm(stills)


def run_regen_app(
    document: Mapping[str, Any],
    media_dir: Path | str,
    out_dir: Path | str,
    *,
    writer: CaptionWriter,
    items: ItemSet | None = None,
    host: str = "127.0.0.1",
    port: int = 8779,
    watch: Callable[[str, int, int], None] | None = None,
    gate: Gate | None = None,
) -> None:
    """Serve the instrument. Port 8779, one past the console's 8778."""
    app = build_app(document, Path(media_dir), Path(out_dir), writer, items=items, watch=watch, gate=gate)
    stills, whole = warm_derivatives(document, Path(media_dir))
    print(f"web copies ready: {stills} stills", end="")
    print(f"; {whole} served whole" if whole else "", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
