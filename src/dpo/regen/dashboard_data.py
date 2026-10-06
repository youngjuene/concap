"""Read-only, allowlisted research views over both persisted study protocols.

The database session key is a bearer credential. It is used only for joins inside
this module; neither it nor raw source objects belong in a dashboard response.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from dpo.regen.study_schema import digest

MAX_ROWS = 100_000
MAX_FILE_BYTES = 64 * 1024 * 1024
LABEL = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
QA_LABEL = re.compile(r"^(?:qa|test|demo|smoke|preview)(?:[-_]|$)", re.IGNORECASE)
SHEET_PROTOCOL = "dpo.sheet-questionnaire/v1"
INTERACTION_ONLY = "interaction-only-sheet-v1"
SHEET_PAGES = {*(f"P{index}" for index in range(12)), "generate", "handoff"}
EVENT_FIELDS = (
    "origin",
    "step",
    "page",
    "frame",
    "index",
    "x",
    "y",
    "count",
    "sequence",
    "playing",
    "hidden",
    "texture",
    "context",
    "position_hint_ms",
    "settings_revision",
    "revision",
    "epoch",
    "fallback",
    "cached",
    "duration_ms",
    "language",
    "done",
    "total",
    "family",
    "heard",
)


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _number(value: Any) -> int | float | None:
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _text(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _identifier(value: Any) -> str | None:
    """Identifiers are labels, never media paths or credentials from other fields."""
    return value if isinstance(value, str) and LABEL.fullmatch(value) else None


def _scalar_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if _number(value) is not None:
        return str(value)
    return None


def _stimulus_assignment(value: Any) -> dict[str, str] | str | None:
    if not isinstance(value, dict):
        return _scalar_text(value)  # Historical scalar assignment labels remain readable.
    return {
        key: value[key]
        for key in ("audiovisual_congruence", "description_depth")
        if isinstance(value.get(key), str)
    }


def _at(value: Any) -> str | None:
    try:
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return None  # Do not invent a timezone for historical client timestamps.
        elif _number(value) is not None:
            parsed = datetime.fromtimestamp(value, UTC)
        else:
            return None
        return parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (ValueError, OverflowError, OSError):
        return None


def _timestamp(row: dict[str, Any]) -> str | None:
    return next(
        (
            stamp
            for key in ("submitted_at", "received_at", "at", "created_at")
            if (stamp := _at(row.get(key))) is not None
        ),
        None,
    )


def _selector(label: str) -> str:
    return "p-" + hashlib.sha256(label.encode()).hexdigest()[:24]


def _warn(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _json(value: str | bytes, warnings: list[str], context: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    _warn(warnings, f"{context}: invalid JSON object; available records are shown.")
    return {}


def _new_person(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "id": _selector(label),
        "snapshot": {},
        "viewings": [],
        "submissions": [],
        "legacy_events": [],
        "sessions": [],
        "warnings": [],
    }


def _unique(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for row in rows:
        identity = [row.get(field) for field in fields]
        # Rows with no original key can only be deduplicated by their full content.
        key = json.dumps(
            identity if any(value is not None for value in identity) else row,
            sort_keys=True,
            ensure_ascii=False,
        )
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


class DashboardData:
    def __init__(self, legacy_dir: Path, viewing_dir: Path) -> None:
        self.legacy_dir = Path(legacy_dir)
        self.viewing_dir = Path(viewing_dir)

    def _legacy(self, warnings: list[str]) -> dict[str, dict[str, Any]]:
        people: dict[str, dict[str, Any]] = {}
        try:
            paths = sorted(self.legacy_dir.iterdir())
        except OSError:
            _warn(warnings, "Phase 1 source directory is unavailable.")
            return people
        pattern = re.compile(r"^(snapshot|viewings|responses|events)-([A-Za-z0-9_-]{1,64})\.(jsonl|json)$")
        for path in paths:
            match = pattern.fullmatch(path.name)
            if not match or path.is_symlink() or not path.is_file():
                continue
            kind, label, suffix = match.groups()
            if suffix != ("json" if kind == "snapshot" else "jsonl"):
                continue
            person = people.setdefault(label, _new_person(label))
            notices = person["warnings"]
            context = f"Phase 1 {kind} for {label}"
            try:
                with path.open("rb") as stream:
                    data = stream.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    _warn(
                        notices,
                        f"{context}: source exceeds the read limit; only the first records are shown.",
                    )
                    data = data[:MAX_FILE_BYTES]
            except OSError:
                _warn(notices, f"{context}: source could not be read.")
                continue
            if kind == "snapshot":
                person["snapshot"] = _json(data, notices, context)
                continue
            lines = data.splitlines()
            if len(lines) > MAX_ROWS:
                _warn(notices, f"{context}: row limit reached; counts are partial.")
            key = {"events": "legacy_events", "responses": "submissions", "viewings": "viewings"}[kind]
            for index, line in enumerate(lines[:MAX_ROWS], 1):
                if line.strip():
                    row = _json(line, notices, f"{context}, line {index}")
                    if row:
                        person[key].append(row)
        return people

    def _viewing(self, warnings: list[str]) -> list[dict[str, Any]]:
        seen_files: set[tuple[int, int]] = set()
        sessions: dict[str, dict[str, Any]] = {}
        for directory in dict.fromkeys((self.legacy_dir, self.viewing_dir)):
            for name in ("questionnaire.sqlite3", "study.sqlite3"):
                path = directory / name
                try:
                    if not path.is_file() or path.is_symlink():
                        continue
                    stat = path.stat()
                except OSError:
                    continue
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen_files:
                    continue
                seen_files.add(identity)
                for session in self._database(path, warnings):
                    key = session["_identity"]
                    previous = sessions.get(key)
                    # A migrated/copied database may retain an older snapshot of
                    # the same session. Use one committed revision, not two histories.
                    if previous is None or (
                        _number(session["state"].get("revision")) or 0,
                        len(session["events"]),
                    ) > (
                        _number(previous["state"].get("revision")) or 0,
                        len(previous["events"]),
                    ):
                        sessions[key] = session
        if not seen_files:
            _warn(warnings, "Phase 2 database is unavailable; no source was created.")
        return list(sessions.values())

    def _database(self, path: Path, warnings: list[str]) -> list[dict[str, Any]]:
        tables = {
            "sessions": "id,state",
            "calibration_links": "source,token",
            "events": "session,kind,body,at",
            "jobs": "id,session,epoch,revision,cue,state,spec,result,created",
            "exposures": "session,id,body",
        }
        loaded: dict[str, list[dict[str, Any]]] = {}
        try:
            db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
            try:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA query_only=ON")
                db.execute("BEGIN")  # All tables are from the same consistent read snapshot.
                for table, columns in tables.items():
                    try:
                        rows = db.execute(
                            f"SELECT {columns} FROM {table} LIMIT ?", (MAX_ROWS + 1,)
                        ).fetchall()
                        if len(rows) > MAX_ROWS:
                            _warn(warnings, f"Phase 2 {table}: row limit reached; counts are partial.")
                        loaded[table] = [dict(row) for row in rows[:MAX_ROWS]]
                    except sqlite3.Error:
                        loaded[table] = []
                        _warn(warnings, f"Phase 2 {table} table is unavailable or unreadable.")
                db.rollback()
            finally:
                db.close()
        except (sqlite3.Error, OSError):
            _warn(warnings, "Phase 2 database could not be read; available Phase 1 records are shown.")
            return []
        linked = {
            row["token"]: _identifier(_text(row["source"]).rsplit(":", 1)[-1])
            for row in loaded.get("calibration_links", [])
        }
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for table in ("events", "jobs", "exposures"):
            grouped[table] = defaultdict(list)
            for row in loaded.get(table, []):
                grouped[table][row["session"]].append(row)
        sessions = []
        for row in loaded.get("sessions", []):
            notices: list[str] = []
            state = _json(row["state"], notices, "Phase 2 session state")
            source = _object(state.get("calibration_source"))
            label = _identifier(source.get("participant")) or linked.get(row["id"])
            if state.get("sheet_protocol") == SHEET_PROTOCOL:
                label = _identifier(state.get("session_id")) or label
            if label is None:
                explicit = _identifier(state.get("participant"))
                label = explicit or "viewing-" + hashlib.sha256(str(row["id"]).encode()).hexdigest()[:12]
            session: dict[str, Any] = {
                "state": state,
                "label": label,
                "warnings": notices,
                "_identity": _identifier(state.get("session_id"))
                or hashlib.sha256(str(row["id"]).encode()).hexdigest(),
            }
            for table in ("events", "jobs", "exposures"):
                decoded = []
                for item in grouped[table].get(row["id"], []):
                    safe = {key: value for key, value in item.items() if key != "session"}
                    for field in ("spec", "result") if table == "jobs" else ("body",):
                        raw = item[field]
                        # Calibration handoff receipts historically store an opaque
                        # source digest instead of an action JSON document.
                        if table == "events" and raw == SHEET_PROTOCOL:
                            safe[field] = {"action": "sheet-enrollment"}
                        elif (
                            table == "events"
                            and isinstance(raw, str)
                            and re.fullmatch(r"[0-9a-f]{12,64}", raw)
                        ):
                            safe[field] = {"action": "calibration.handoff"}
                        else:
                            safe[field] = _json(raw, notices, f"Phase 2 {table} record") if raw else {}
                    decoded.append(safe)
                session[table] = decoded
            sessions.append(session)
        return sessions

    def _collect(self, include_qa: bool) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        people = self._legacy(warnings)
        for session in self._viewing(warnings):
            label = session["label"]
            person = people.setdefault(label, _new_person(label))
            person["sessions"].append(session)
            state = session["state"]
            source = _object(state.get("calibration_source"))
            if state.get("sheet_protocol") == SHEET_PROTOCOL:
                person["submissions"].extend(_rows(state.get("sheet_responses")))
                person["viewings"].extend(_rows(state.get("sheet_viewings")))
            person["submissions"].extend(_rows(source.get("responses")))
            person["viewings"].extend(_rows(source.get("viewings")))
        details = []
        for person in people.values():
            person["qa"] = bool(QA_LABEL.match(person["label"])) or any(
                _object(session["state"].get("sheet_config")).get("qa_only") is True
                for session in person["sessions"]
            )
            if person["qa"] and not include_qa:
                continue
            detail = _normalize(person)
            details.append(detail)
            for notice in detail["warnings"]:
                _warn(warnings, notice)
        details.sort(
            key=lambda item: (item["participant"]["last_activity"] or "", item["participant"]["label"]),
            reverse=True,
        )
        return details, warnings

    def overview(self, include_qa: bool = False) -> dict[str, Any]:
        details, warnings = self._collect(include_qa)
        participants = [detail["participant"] for detail in details]
        responses = [row for detail in details for row in detail["responses"]]
        durations = [value for detail in details for value in detail["_durations"]]
        captions = [row for detail in details for row in detail["captions"] if row["displayed"]]
        activity: dict[str, dict[str, Any]] = {}
        for detail in details:
            for phase, stamp in detail["_submissions"]:
                if stamp:
                    day = stamp[:10]
                    bucket = activity.setdefault(day, {"date": day, "phase1": 0, "phase2": 0})
                    bucket[f"phase{phase}"] += 1
        return {
            "updated_at": _at(datetime.now(UTC).timestamp()),
            "warnings": warnings,
            "metrics": {
                "participants": len(participants),
                "phase1_complete": sum(row["phase1_status"] == "done" for row in participants),
                "phase2_started": sum(row["phase2_started"] for row in participants),
                "phase2_complete": sum(row["phase2_status"] == "done" for row in participants),
                "survey_submissions": sum(len(detail["_submissions"]) for detail in details),
                "response_values": len(responses),
                "control_changes": sum(row["control_changes"] for row in participants),
                "generated_exposures": sum(row["kind"] == "generated" for row in captions),
                "fallback_exposures": sum(row["kind"] == "fallback" for row in captions),
                "generation_median_ms": median(durations) if durations else None,
            },
            "participants": participants,
            "responses": responses,
            "activity": [activity[day] for day in sorted(activity)],
        }

    def phase1_analysis(
        self,
        include_qa: bool = False,
        *,
        participant: str = "",
        language: str = "",
        status: str = "",
        clip_id: str = "",
    ) -> dict[str, Any]:
        """One filtered snapshot supplies metrics, denominators and both exports."""
        from dpo.regen.phase1_analysis import analyze_sessions

        details, warnings = self._collect(include_qa)
        sessions = []
        query = participant.strip().casefold()
        for detail in details:
            person = detail["participant"]
            if query and query not in f"{person['label']} {person['id']}".casefold():
                continue
            if language and language != person["language"]:
                continue
            complete = person["phase2_status"] in {"done", "complete", "submitted"}
            phase1 = person["phase1_status"] in {"done", "complete", "submitted"}
            if status and not {
                "complete": complete,
                "active": not complete,
                "phase1": phase1,
                "no_phase2": person["phase2_status"] in {None, "", "missing"},
            }.get(status, False):
                continue
            sessions.extend(detail["_phase1_sessions"])
        available_clips = sorted(
            {
                row["clip_id"]
                for session in sessions
                for row in session["sheet_responses"]
                if isinstance(row.get("clip_id"), str) and not row.get("analysis_excluded")
            }
        )
        result = analyze_sessions(sessions, clip_id=clip_id or None)
        result.update(
            available_clips=available_clips,
            filters={
                "include_qa": include_qa,
                "participant": participant,
                "language": language,
                "status": status,
                "clip_id": clip_id,
            },
            updated_at=_at(datetime.now(UTC).timestamp()),
        )
        result["warnings"] = list(dict.fromkeys([*warnings, *result.get("warnings", [])]))
        return result

    def participant(self, id: str, include_qa: bool = False) -> dict[str, Any]:
        details, warnings = self._collect(include_qa)
        for detail in details:
            if detail["participant"]["id"] == id:
                result = {key: value for key, value in detail.items() if not key.startswith("_")}
                result["warnings"] = list(dict.fromkeys([*warnings, *detail["warnings"]]))
                return result
        raise KeyError("Participant not found")


def _phase1_sessions(person: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep credentials and unrelated setup text out of the analysis boundary."""
    payloads = []
    for session in person["sessions"]:
        state = session["state"]
        source = _object(state.get("calibration_source"))
        if state.get("sheet_protocol") != SHEET_PROTOCOL and source.get("schema") != SHEET_PROTOCOL:
            continue
        session_id = _identifier(state.get("session_id"))
        if session_id is None:
            continue
        clips = _object(_object(state.get("sheet_config")).get("clips"))
        payloads.append(
            {
                "session_id": session_id,
                "participant": person["id"],
                "sheet_responses": _rows(
                    state["sheet_responses"] if "sheet_responses" in state else source.get("responses")
                ),
                "sheet_clip_order": state.get("sheet_clip_order", []),
                "sheet_config": {
                    "clips": {
                        key: {
                            field: value[field]
                            for field in ("source_families", "analysis_codebook")
                            if field in value
                        }
                        for key, value in clips.items()
                        if isinstance(key, str) and isinstance(value, dict)
                    }
                },
            }
        )
    return payloads


def _response_rows(
    participant: str,
    phase: int,
    scope: str,
    answers: dict[str, Any],
    *,
    language: str,
    instrument: str,
    items: list[dict[str, Any]],
    submitted_at: str | None,
    video_id: str | None = None,
    flow_version: str = "legacy",
    view_id: str | None = None,
    condition: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    indexed = {item.get("id"): item for item in items}
    rows = []
    for item_id, value in answers.items():
        item = indexed.get(item_id, {})
        if item.get("type") == "multi" and isinstance(value, list):
            if not all(isinstance(option, str) for option in value):
                continue
            value = list(value)
        elif item.get("type") == "visual" and isinstance(value, list):
            value = [
                {
                    "frame_id": _text(point.get("frame_id")),
                    "x": _number(point.get("x")),
                    "y": _number(point.get("y")),
                }
                for point in value
                if isinstance(point, dict)
            ]
        elif not isinstance(value, (str, int, float, bool, type(None))):
            continue
        wording = item.get("text")
        if isinstance(wording, dict):
            wording = wording.get(language)
        available = isinstance(wording, str) and bool(wording)
        raw_options = item.get("options")
        options = raw_options if isinstance(raw_options, list) else []
        option_values = [option.get("value") if isinstance(option, dict) else option for option in options]
        option_labels = [
            _text(option.get("label")) if isinstance(option, dict) else str(option) for option in options
        ]
        row = {
            "participant": participant,
            "phase": phase,
            "scope": scope,
            "video_id": video_id,
            "instrument": instrument,
            "flow_version": flow_version,
            "view_id": view_id,
            "condition": condition,
            "item_id": item_id,
            "item_text": wording if available else item_id,
            "wording_available": available,
            "type": _text(item.get("type"), "rating" if phase == 1 else "unknown"),
            "value": value if not isinstance(value, float) or math.isfinite(value) else None,
            "points": _number(item.get("points")),
            "options": [
                option for option in option_values if isinstance(option, str) or _number(option) is not None
            ],
            "language": language,
            "submitted_at": submitted_at,
        }
        if metadata is not None:
            raw_order = metadata.get("presented_order")
            order = [key for key in raw_order if isinstance(key, str)] if isinstance(raw_order, list) else []
            row.update(
                min=_number(item.get("min")),
                max=_number(item.get("max")),
                labels=[_text(label) for label in item.get("labels", [])]
                if isinstance(item.get("labels"), list)
                else [],
                option_labels=option_labels,
                presented_order=order,
                presentation_index=order.index(item_id) if item_id in order else None,
                client_submitted_at=_at(metadata.get("client_submitted_at")),
                clip_index=_number(metadata.get("clip_index")),
                stimulus_assignment=_stimulus_assignment(metadata.get("stimulus_assignment")),
                caption_strategy=_scalar_text(metadata.get("caption_strategy")),
                analysis_excluded=False,
            )
            for key in ("source", "source_row", "source_row_hash", "scale_source", "subscale"):
                row[key] = _scalar_text(item.get(key))
            for key in ("neutral_value", "min_selections", "max_selections"):
                if key in item:
                    row[key] = _number(item[key])
            if isinstance(item.get("reverse"), bool):
                row["reverse"] = item["reverse"]
        rows.append(row)
    return rows


def _event(phase: int, kind: str, data: dict[str, Any], at: str | None) -> dict[str, Any]:
    clip_id = _identifier(data.get("clip_id"))
    return {
        "phase": phase,
        "kind": kind,
        "at": at,
        "stage": _scalar_text(data.get("stage")),
        "clip_id": clip_id,
        "clip_index": data.get("clip_index") if type(data.get("clip_index")) is int else None,
        "view_id": _identifier(data.get("view_id")),
        "condition": _scalar_text(data.get("condition")),
        "protocol": _scalar_text(data.get("protocol")),
        "flow_version": _scalar_text(data.get("flow_version")) or _scalar_text(data.get("protocol")),
        "stimulus_assignment": _stimulus_assignment(data.get("stimulus_assignment")),
        "caption_strategy": _scalar_text(data.get("caption_strategy")),
        "session_id": _scalar_text(data.get("session_id")),
        "config_hash": _scalar_text(data.get("config_hash")),
        "video_id": _identifier(data.get("video_id")) or clip_id,
        "position_ms": next(
            (
                number
                for field in ("position_hint_ms", "position_ms", "at_ms")
                if (number := _number(data.get(field))) is not None
            ),
            None,
        ),
        "detail": {
            key: data[key]
            for key in EVENT_FIELDS
            if key in data
            and isinstance(data[key], (str, bool, int, float, type(None)))
            and not (isinstance(data[key], float) and not math.isfinite(data[key]))
        },
    }


def _caption(phase: int, video_id: str | None, text: Any, **values: Any) -> dict[str, Any]:
    return {
        "phase": phase,
        "video_id": video_id,
        "text": _text(text),
        "start_ms": None,
        "end_ms": None,
        "kind": "job",
        "displayed": False,
        "job_id": None,
        "revision": None,
        "texture": None,
        "context": None,
        "generation_ms": None,
        "created_at": None,
        "status": "unknown",
        **values,
    }


def _normalize(person: dict[str, Any]) -> dict[str, Any]:
    sessions, snapshot = person["sessions"], person["snapshot"]
    warnings = list(person["warnings"])
    for session in sessions:
        warnings.extend(session["warnings"])
    submissions = _unique(
        [row for row in person["submissions"] if not row.get("analysis_excluded")], ("page", "view_id")
    )
    viewings = _unique(
        [row for row in person["viewings"] if not row.get("analysis_excluded")], ("view_id", "config_hash")
    )
    by_view = {row.get("view_id"): row for row in viewings}
    language = _text(snapshot.get("language")) or next(
        (_text(session["state"].get("language")) for session in sessions if session["state"].get("language")),
        next((_text(row.get("language")) for row in viewings if row.get("language")), "unknown"),
    )
    responses: list[dict[str, Any]] = []
    survey_dates: list[tuple[int, str | None]] = []
    events = [
        _event(1, _text(row.get("type"), "unknown"), row, _timestamp(row)) for row in person["legacy_events"]
    ]
    captions: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    durations: list[int | float] = []
    for row in submissions:
        sheet = isinstance(row.get("answers"), dict) and isinstance(row.get("items"), list)
        answers = row.get("answers") if sheet else row.get("responses")
        if not isinstance(answers, dict):
            _warn(warnings, "Phase 1 submission has no readable answers.")
            continue
        viewing = by_view.get(row.get("view_id"), {})
        reading = _text(row.get("language")) or _text(viewing.get("language")) or language
        items = _rows(row.get("items"))
        # A matched frozen PRSS instrument is historical evidence; current package
        # defaults are deliberately never substituted for missing old wording.
        for session in sessions:
            instrument = _object(session["state"].get("instrument"))
            questionnaire = _object(_object(session["state"].get("profile")).get("questionnaire"))
            if (
                questionnaire.get("items_digest") == row.get("items_digest")
                and row.get("items_digest")
                and questionnaire.get("language") == reading
            ):
                points = _number(_object(questionnaire.get("scale")).get("points"))
                items.extend(
                    {**_object(item), "id": key, "type": "rating", "points": points}
                    for key, item in _object(questionnaire.get("items")).items()
                )
            if (
                _object(instrument.get("prss")).get("items_digest") == row.get("items_digest")
                and row.get("items_digest")
                and instrument.get("language") == reading
            ):
                items.extend(
                    item
                    for item in _rows(_object(instrument.get("items")).get("final"))
                    if item.get("block") == "prss"
                )
        stamp = _timestamp(row)
        survey_dates.append((1, stamp))
        responses.extend(
            _response_rows(
                person["id"],
                1,
                _text(row.get("page"), "survey"),
                answers,
                language=reading,
                instrument=_text(row.get("instrument_hash")) or _text(row.get("items_digest"), "unknown"),
                items=items,
                submitted_at=stamp,
                video_id=_identifier(row.get("clip_id")) or _identifier(viewing.get("clip_id")),
                flow_version=_text(row.get("flow_version"), SHEET_PROTOCOL if sheet else "legacy"),
                view_id=_identifier(row.get("view_id")),
                condition=_text(row.get("condition")) or _text(viewing.get("condition")) or None,
                metadata={
                    **row,
                    "stimulus_assignment": row.get("stimulus_assignment", viewing.get("stimulus_assignment")),
                    "caption_strategy": row.get("caption_strategy", viewing.get("caption_strategy")),
                }
                if sheet
                else None,
            )
        )
    for viewing in viewings:
        generated = viewing.get("condition") == "regenerated"
        fallback: bool | None = None
        if generated:
            for session in sessions:
                state = session["state"]
                order = state.get("sheet_clip_order", [])
                index = viewing.get("clip_index")
                if (
                    state.get("sheet_protocol") == SHEET_PROTOCOL
                    and type(index) is int
                    and 0 <= index < len(order)
                ):
                    result = _object(_object(state.get("sheet_regenerations")).get(order[index]))
                    if isinstance(result.get("fallback"), bool):
                        fallback = result["fallback"]
            if fallback is None:
                fallback = _object(snapshot.get("regeneration_fallback_by_segment")).get(
                    _text(viewing.get("segment"))
                )
            for event in person["legacy_events"]:
                if event.get("type") == "regeneration.written" and event.get("segment") == viewing.get(
                    "segment"
                ):
                    fallback = event.get("fallback")
            if fallback is None and len(viewings) <= 2:
                fallback = snapshot.get("regeneration_fallback")
            if fallback is None:
                _warn(warnings, "Phase 1 regenerated track lacks stored fallback provenance.")
        kind = "prepared" if not generated else "fallback" if fallback else "generated"
        if generated and fallback is None:
            kind = "unknown"
        for cue in _rows(viewing.get("captions")):
            captions.append(
                _caption(
                    1,
                    _identifier(viewing.get("clip_id")),
                    cue.get("text"),
                    kind=kind,
                    displayed=True,
                    start_ms=_number(cue.get("start_ms")),
                    end_ms=_number(cue.get("end_ms")),
                    status="recorded",
                    created_at=_at(viewing.get("playback_started_at", viewing.get("started_at"))),
                )
            )
    phase2_surveys = 0
    completed: set[str] = set()
    for session in sessions:
        state = session["state"]
        instrument = _object(state.get("instrument"))
        mutation_dates: dict[tuple[str, str | None], str | None] = {}
        for row in sorted(session["events"], key=lambda entry: _at(entry.get("at")) or ""):
            body = row["body"]
            data = _object(body.get("data")) if row.get("kind") == "mutation" else body
            kind = _text(body.get("action")) or _text(row.get("kind"), "unknown")
            sheet = state.get("sheet_protocol") == SHEET_PROTOCOL
            if sheet and (
                data.get("page") == "practice" or kind == "practice-complete" or data.get("analysis_excluded")
            ):
                continue
            phase = (
                data["phase"]
                if type(data.get("phase")) is int and data["phase"] in (1, 2)
                else 1
                if sheet and (kind == "sheet-enrollment" or data.get("page") in SHEET_PAGES)
                else 2
            )
            event = _event(phase, kind, data, _at(row.get("at")))
            events.append(event)
            mutation_dates[(kind, event["video_id"])] = event["at"]
            if kind == "settings":
                interactions.append(
                    {
                        "phase": 2,
                        "video_id": event["video_id"],
                        "at": event["at"],
                        "position_ms": event["position_ms"],
                        "texture": _number(data.get("texture")),
                        "context": _number(data.get("context")),
                        "revision": _number(data.get("settings_revision", data.get("revision"))),
                        "kind": kind,
                        "origin": _scalar_text(data.get("origin")),
                    }
                )
        submitted: list[tuple[str, dict[str, Any], str]] = []
        if state.get("survey_policy") != INTERACTION_ONLY:
            if isinstance(state.get("preferences"), dict) and state["preferences"]:
                submitted.append(("preferences", {"answers": state["preferences"]}, "preferences"))
            submitted.extend(("video", row, "video-survey") for row in _rows(state.get("video_surveys")))
            if isinstance(state.get("final_survey"), dict):
                submitted.append(("overall", state["final_survey"], "final-survey"))
        for scope, row, action in submitted:
            if not isinstance(row.get("answers"), dict):
                _warn(warnings, "Phase 2 submission has no readable answers.")
                continue
            frozen = _object(row.get("instrument")) or instrument
            items_key = "final" if action == "final-survey" else scope
            items = _rows(_object(frozen.get("items")).get(items_key))
            video_id = _identifier(row.get("video_id"))
            stamp = _timestamp(row) or mutation_dates.get((action, video_id))
            survey_dates.append((2, stamp))
            phase2_surveys += 1
            responses.extend(
                _response_rows(
                    person["id"],
                    2,
                    scope,
                    row["answers"],
                    language=_text(frozen.get("language")) or language,
                    instrument=_text(row.get("instrument_hash"))
                    or _text(frozen.get("hash"))
                    or _text(row.get("items_hash"), "unknown"),
                    items=items,
                    submitted_at=stamp,
                    video_id=video_id,
                    flow_version=_text(frozen.get("flow")) or str(state.get("flow_version", "legacy")),
                )
            )
        completed.update(_text(row.get("video_id")) for row in _rows(state.get("completions")))
        jobs = {row["id"]: row for row in session["jobs"]}
        videos = {}
        for video in _rows(state.get("viewing")):
            try:
                videos[digest(video)] = video
            except ValueError:
                _warn(warnings, "Phase 2 video metadata is invalid; job video assignment may be unavailable.")
        shown: set[str] = set()
        for stored in session["exposures"]:
            exposure = stored["body"]
            if not exposure:
                continue
            job_id = _identifier(exposure.get("job_id"))
            job = jobs.get(job_id, {})
            if job_id:
                shown.add(job_id)
            result = _object(job.get("result"))
            axes = _object(exposure.get("axes"))
            fallback = exposure.get("fallback")
            kind = "fallback" if fallback is True else "generated" if fallback is False else "unknown"
            if kind == "unknown":
                _warn(warnings, "Phase 2 exposure lacks stored generated/fallback provenance.")
            captions.append(
                _caption(
                    2,
                    _identifier(exposure.get("video_id")),
                    exposure.get("text"),
                    kind=kind,
                    displayed=True,
                    job_id=job_id,
                    revision=_number(exposure.get("settings_revision")),
                    start_ms=_number(exposure.get("start_ms")),
                    end_ms=_number(exposure.get("end_ms")),
                    texture=_number(axes.get("texture")),
                    context=_number(axes.get("context")),
                    generation_ms=_number(result.get("duration_ms")),
                    created_at=_at(job.get("created")),
                    status=_text(job.get("state"), "prepared" if fallback else "unknown"),
                )
            )
        for job_id, job in jobs.items():
            result, spec = job["result"], job["spec"]
            duration = _number(result.get("duration_ms"))
            if job.get("state") in ("succeeded", "failed") and duration is not None and duration >= 0:
                durations.append(duration)
            if job_id in shown:
                continue
            # Epoch also changes on seek/recovery and cannot identify a video.
            video = videos.get(_text(spec.get("video_hash")), {})
            axes = _object(spec.get("axes"))
            captions.append(
                _caption(
                    2,
                    _identifier(video.get("id")),
                    result.get("text"),
                    job_id=_identifier(job_id),
                    revision=_number(job.get("revision")),
                    start_ms=_number(spec.get("start_ms")),
                    end_ms=_number(spec.get("end_ms")),
                    generation_ms=duration,
                    texture=_number(axes.get("texture")),
                    context=_number(axes.get("context")),
                    created_at=_at(job.get("created")),
                    status=_text(job.get("state"), "unknown"),
                )
            )
    sheet_states = [
        session["state"] for session in sessions if session["state"].get("sheet_protocol") == SHEET_PROTOCOL
    ]
    phase1_exists = bool(snapshot or submissions or viewings or person["legacy_events"] or sheet_states)
    phase1_done = (
        snapshot.get("step") == "done"
        or any(_object(session["state"].get("calibration_source")) for session in sessions)
        or any(state.get("sheet_page") == "handoff" for state in sheet_states)
    )
    states = [session["state"] for session in sessions]
    stage = next((_text(state.get("stage")) for state in reversed(states) if state.get("stage")), "missing")
    if any(state.get("stage") == "done" for state in states):
        stage = "done"
    if stage == "sheet-questionnaire":
        stage = "awaiting-calibration"
    videos_total = next(
        (
            len(_rows(state.get("viewing")))
            for state in reversed(states)
            if isinstance(state.get("viewing"), list)
        ),
        None,
    )
    activity = [event["at"] for event in events] + [stamp for _, stamp in survey_dates]
    activity.extend(_at(row.get("playback_ended_at", row.get("ended_at"))) for row in viewings)
    activity.extend(caption["created_at"] for caption in captions)
    summary = {
        "id": person["id"],
        "label": person["label"],
        "language": language,
        "qa": person["qa"],
        "phase1_status": "done" if phase1_done else "in_progress" if phase1_exists else "missing",
        "phase2_status": stage,
        "phase2_started": any(state.get("viewing") for state in states)
        or stage in ("watch", "break", "video-survey", "final", "poststudy", "debrief", "done"),
        "phase1_surveys": len(survey_dates) - phase2_surveys,
        "phase2_surveys": phase2_surveys,
        "videos_completed": len(completed - {""}),
        "videos_total": videos_total,
        "control_changes": len(interactions),
        "last_activity": max((stamp for stamp in activity if stamp), default=None),
    }
    return {
        "participant": summary,
        "responses": responses,
        "interactions": interactions,
        "captions": captions,
        "events": sorted(events, key=lambda event: event["at"] or ""),
        "warnings": list(dict.fromkeys(warnings)),
        "_durations": durations,
        "_submissions": survey_dates,
        "_phase1_sessions": _phase1_sessions(person),
    }
