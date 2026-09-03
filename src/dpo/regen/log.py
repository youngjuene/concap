"""§9.5: one row per viewing, with the survey responses joined to it by key.

Four files per participant under ``--out``, and the split is the storage unit
§9.5 asks for rather than a filing convenience.

``viewings-<p>.jsonl``
    The analysis unit. One row per viewing: its key, its index, its condition,
    its segment and clip, the full text of the captions that were displayed,
    and the playback start and end timestamps. Two rows per completed session.

``responses-<p>.jsonl``
    One row per survey submission, carrying the ``view_id`` of the viewing it
    is about. §3 joins to the first viewing and §8 to the second. Stored beside
    the viewings rather than inside them because a viewing is written when the
    clip ends and the responses arrive minutes later; merging would mean
    rewriting a row that is already on disk, and an append-only file that is
    never rewritten is the property that makes a crashed session readable.

``events-<p>.jsonl``
    The full stream: every step entered, every point placed and moved, every
    lane played, the regeneration and its prompt. Nothing here summarises, and
    nothing in the other two files is absent from this one.

``snapshot-<p>.json``
    The resumable state, replaced atomically on every write, so a reload during
    a session returns to the step it was on rather than to the start (§9.1).

Every line carries the configuration hash, applied here on receipt from the
configuration the server is running under — never from anything the browser
sends, which could be a stale tab holding an older session's stamp.

Captions are written into the viewing row under one schema for both conditions
(§9.6), so nothing downstream can distinguish a prepared row from a regenerated
one by shape. The only field that says which is ``condition``.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dpo.core.atomic import replace_atomically

PARTICIPANT_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
LOG_SCHEMA = "dpo.caption-regen-log/v1"
SNAPSHOT_SCHEMA = "dpo.caption-regen-snapshot/v1"
VIEWING_SCHEMA = "dpo.caption-regen-viewing/v1"
RESPONSE_SCHEMA = "dpo.caption-regen-response/v1"


class RegenLogError(ValueError):
    """A participant id, an event batch, or a row the log refuses."""


def validate_participant(value: object) -> str:
    if not isinstance(value, str) or not PARTICIPANT_RE.fullmatch(value):
        raise RegenLogError("participant must match [A-Za-z0-9_-]{1,64}")
    return value


def _received_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def view_id(participant: str, index: int) -> str:
    """The key a survey row joins on. Derived, so nothing has to allocate it."""
    return f"{validate_participant(participant)}-v{index}"


class EventLog:
    """One log per ``--out``; every append is serialised across the threadpool.

    FastAPI runs the sync routes in a threadpool, so two writes can be in
    flight at once — the debounced event POST and a step submit. The lock makes
    each append whole with respect to every other, including the snapshot
    replacement that follows it.
    """

    def __init__(self, out_dir: Path, config_hash: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self._lock = threading.Lock()

    def _path(self, kind: str, participant: str) -> Path:
        return (
            self.out_dir
            / f"{kind}-{validate_participant(participant)}.{'json' if kind == 'snapshot' else 'jsonl'}"
        )

    def events_path(self, participant: str) -> Path:
        return self._path("events", participant)

    def viewings_path(self, participant: str) -> Path:
        return self._path("viewings", participant)

    def responses_path(self, participant: str) -> Path:
        return self._path("responses", participant)

    def snapshot_path(self, participant: str) -> Path:
        return self._path("snapshot", participant)

    def _append(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
        received_at = _received_at()
        lines = [
            json.dumps(
                {**row, "received_at": received_at, "config_hash": self.config_hash},
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in rows
        ]
        if lines:
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return len(lines)

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
        rows: list[dict[str, Any]] = []
        for index, stripped in enumerate(lines):
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                # A torn last line is a write that did not finish — a crash, a
                # full disk — and the lines before it are whole; reading them
                # keeps the session answering. A torn line anywhere else is
                # corruption, and stays an error.
                if index == len(lines) - 1:
                    break
                raise
        return rows

    def append(
        self, participant: str, events: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any] | None
    ) -> int:
        """Append the events, then replace the snapshot; return the count appended."""
        for index, event in enumerate(events):
            if not isinstance(event, Mapping) or not isinstance(event.get("type"), str):
                raise RegenLogError(f"events[{index}] must be an object with a string 'type'")
        with self._lock:
            written = self._append(self.events_path(participant), events)
            if snapshot is not None:
                replace_atomically(self.snapshot_path(participant), _json_bytes(snapshot))
        return written

    def events(self, participant: str) -> list[dict[str, Any]]:
        return self._read(self.events_path(participant))

    def viewings(self, participant: str) -> list[dict[str, Any]]:
        return self._read(self.viewings_path(participant))

    def responses(self, participant: str) -> list[dict[str, Any]]:
        return self._read(self.responses_path(participant))

    def snapshot(self, participant: str) -> dict[str, Any] | None:
        path = self.snapshot_path(participant)
        if not path.is_file():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None

    def write_snapshot(self, participant: str, snapshot: Mapping[str, Any]) -> None:
        with self._lock:
            replace_atomically(self.snapshot_path(participant), _json_bytes(snapshot))

    def record_viewing(
        self,
        participant: str,
        *,
        index: int,
        condition: str,
        segment: str,
        clip_id: str,
        captions: Sequence[Mapping[str, Any]],
        started_at: str,
        ended_at: str,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """§2 and §7's row. Returns the key the survey rows join on.

        Idempotent per view index: a page that re-posts its end — a keepalive
        flush racing the auto-advance — must not produce two rows for one
        viewing, because the analysis unit is the viewing and a duplicate would
        be a second observation that never happened.
        """
        key = view_id(participant, index)
        with self._lock:
            for row in self._read(self.viewings_path(participant)):
                if row.get("view_id") == key:
                    return key
            self._append(
                self.viewings_path(participant),
                [
                    {
                        "schema": VIEWING_SCHEMA,
                        "view_id": key,
                        "participant": participant,
                        "view_index": index,
                        "condition": condition,
                        "segment": segment,
                        "clip_id": clip_id,
                        "captions": [dict(cue) for cue in captions],
                        "playback_started_at": started_at,
                        "playback_ended_at": ended_at,
                        **dict(extra or {}),
                    }
                ],
            )
        return key

    def record_responses(
        self,
        participant: str,
        *,
        page: str,
        key: str,
        responses: Mapping[str, Any],
        entered_at: str,
        submitted_at: str,
        items_digest: str,
        items_provenance: str,
    ) -> bool:
        """§3 and §8's row, joined to a viewing by ``key``. False if already there.

        The item digest and its provenance travel with every submission, so a
        study that edited its wording mid-run, or piloted on the placeholder
        set, can be separated in the analysis instead of averaged through.
        """
        with self._lock:
            for row in self._read(self.responses_path(participant)):
                if row.get("page") == page and row.get("view_id") == key:
                    return False
            self._append(
                self.responses_path(participant),
                [
                    {
                        "schema": RESPONSE_SCHEMA,
                        "view_id": key,
                        "participant": participant,
                        "page": page,
                        "responses": dict(responses),
                        "entered_at": entered_at,
                        "submitted_at": submitted_at,
                        "items_digest": items_digest,
                        "items_provenance": items_provenance,
                    }
                ],
            )
        return True

    def export(self, participant: str, session_id: str) -> dict[str, Any]:
        """Everything one participant produced, in one document."""
        return {
            "schema": LOG_SCHEMA,
            "session_id": session_id,
            "config_hash": self.config_hash,
            "participant": validate_participant(participant),
            "viewings": self.viewings(participant),
            "responses": self.responses(participant),
            "events": self.events(participant),
            "snapshot": self.snapshot(participant),
        }
