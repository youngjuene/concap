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

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from dpo.caption.writer import CachedWriter, CaptionWriter
from dpo.regen import progress
from dpo.regen.assignment import PREPARED, REGENERATED, Assignment
from dpo.regen.captions import Cue, cues_of, record_of
from dpo.regen.copy import STRINGS
from dpo.regen.document import (
    configuration_of,
    frames_of,
    objects_of,
    participant_document,
    segment_of,
    track_of,
    validate_regen_document,
)
from dpo.regen.enrolment import Roster
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


def _played(lane: object) -> bool:
    """Whether §5's lane statistics report this lane as having been played.

    Total over anything the page could send, and false for a lane it did not
    report or reported without a count: "no count" is not evidence that a
    playback happened, and the flag this feeds says a selection was made
    without one. The shape is refused separately, so a malformed batch is a
    400 rather than a lane silently reading as unplayed.
    """
    plays = lane.get("plays") if isinstance(lane, Mapping) else None
    return isinstance(plays, int) and not isinstance(plays, bool) and plays > 0


def build_app(
    document: Mapping[str, Any],
    media_dir: Path,
    out_dir: Path,
    writer: CaptionWriter,
    *,
    items: ItemSet | None = None,
) -> FastAPI:
    """The app over one validated document."""
    validate_regen_document(document)
    configuration = configuration_of(document)
    item_set = items if items is not None else load_items()
    app = FastAPI(title="dpo caption regen", docs_url=None, redoc_url=None, openapi_url=None)
    media_dir = Path(media_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(out_dir, configuration.hash)
    roster = Roster(out_dir)
    cached = writer if isinstance(writer, CachedWriter) else CachedWriter(writer, out_dir / CACHE_FILE)

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
        return {
            "strings": STRINGS,
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
            "steps": list(progress.PARTICIPANT_STEPS),
            "languages": list(configuration.languages),
        }

    @app.post("/api/session")
    def session(payload: Mapping[str, Any] | None = None) -> Any:
        """§1: issue an identifier and fix the assignment, once, at Page 1 load."""
        asked = (payload or {}).get("participant")
        if asked is not None and not isinstance(asked, str):
            return _error(400, "participant must be a string if it is given")
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
        }

    @app.post("/api/language")
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
            return {"step": step, **participant_document(document, segment)}
        if step in SURVEY_PAGES:
            return {
                "step": step,
                "blocks": [block.record() for block in item_set.page_blocks(step)],
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
    def auditory(payload: Mapping[str, Any]) -> Any:
        """§5: the selected sources, and what the participant played to choose them."""
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
        stems = {str(stem["id"]): str(stem["label"]) for stem in segment_of(document, segment)["stems"]}
        selected = payload.get("selected")
        if not isinstance(selected, Sequence) or isinstance(selected, str):
            return _error(400, "selected must be a list of source ids in selection order")
        unknown = [source for source in selected if source not in stems]
        if unknown:
            return _error(400, "selected names sources this segment does not have", unknown=unknown)
        raw_lanes = payload.get("lanes")
        if raw_lanes is not None and not isinstance(raw_lanes, Mapping):
            return _error(400, "lanes must be an object of source id to that lane's statistics")
        lanes = dict(raw_lanes or {})
        malformed = sorted(key for key, value in lanes.items() if not isinstance(value, Mapping))
        if malformed:
            return _error(400, "each lane carries its own playback statistics", invalid=malformed)
        # §5 logs "whether a selection was made without playback" — computed
        # here rather than trusted from the page, from the same lane statistics
        # the page reports, so the flag and the counts cannot disagree.
        unheard = [source for source in selected if not _played(lanes.get(source))]
        snapshot = {
            **gated,
            "auditory_labels": [stems[source] for source in selected],
            "auditory_ids": list(selected),
        }
        log.append(
            person,
            [
                {
                    "type": "auditory.submitted",
                    "segment": segment,
                    "selected": list(selected),
                    "labels": [stems[source] for source in selected],
                    "lanes": lanes,
                    "selected_without_playback": unheard,
                }
            ],
            progress.advance(snapshot, progress.AUDITORY),
        )
        return {"step": progress.current(log.snapshot(person)), "selected_without_playback": unheard}

    @app.post("/api/regenerate")
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
        video = _media(str(segment_of(document, segment)["video"]))
        result = regenerate(
            cached,
            configuration,
            clip_id=str(segment_of(document, segment)["clip_id"]),
            slots=_slots(segment),
            fallback=track_of(document, segment, "fallback_track"),
            report=report,
            language=reading,
            media=video if isinstance(video, Path) else None,
            settings={"segment": segment, "config_hash": configuration.hash, "language": reading},
        )
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

    @app.post("/api/events")
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
    def download(participant: str | None = None) -> Any:
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

    @app.get("/media/video/{segment}")
    def video(segment: str) -> Any:
        return _file_response(segment, "video")

    @app.get("/media/frame/{segment}/{index}")
    def frame(segment: str, index: int) -> Any:
        """One frame of §4's strip. The masks for it never leave the server."""
        try:
            strip = frames_of(document, segment)
        except ValueError as exc:
            return _error(404, str(exc))
        if not 0 <= index < len(strip):
            return _error(404, f"segment {segment} has no frame {index}")
        resolved = _media(str(strip[index]["still"]))
        return resolved if isinstance(resolved, JSONResponse) else FileResponse(resolved)

    @app.get("/media/stem/{segment}/{stem_id}")
    def stem(segment: str, stem_id: str) -> Any:
        try:
            entry = segment_of(document, segment)
        except ValueError as exc:
            return _error(404, str(exc))
        for candidate in entry["stems"]:
            if candidate["id"] == stem_id:
                resolved = _media(str(candidate["audio"]))
                return resolved if isinstance(resolved, JSONResponse) else FileResponse(resolved)
        return _error(404, f"segment {segment} has no stem {stem_id!r}")

    def _file_response(segment: str, key: str) -> Any:
        try:
            entry = segment_of(document, segment)
        except ValueError as exc:
            return _error(404, str(exc))
        resolved = _media(str(entry[key]))
        return resolved if isinstance(resolved, JSONResponse) else FileResponse(resolved)

    @app.get("/{filename}")
    def static_file(filename: str) -> Response:
        media_type = STATIC_FILES.get(filename)
        if media_type is None:
            return _error(404, f"no file {filename!r}")
        return Response(_package_file(filename), media_type=media_type)

    return app


def run_regen_app(
    document: Mapping[str, Any],
    media_dir: Path | str,
    out_dir: Path | str,
    *,
    writer: CaptionWriter,
    items: ItemSet | None = None,
    host: str = "127.0.0.1",
    port: int = 8779,
) -> None:
    """Serve the instrument. Port 8779, one past the console's 8778."""
    app = build_app(document, Path(media_dir), Path(out_dir), writer, items=items)
    uvicorn.run(app, host=host, port=port, log_level="warning")
