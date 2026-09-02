"""The event log: what a session leaves behind, and the two things the server reads back.

Per participant under ``--out``: ``events-<p>.jsonl`` is append-only and never
rewritten — every gesture the browser emits (contract §6) lands as one line
with the server's receipt time, so the record of a session is the record and
not a reconstruction. ``snapshot-<p>.json`` is the browser's whole resumable
state, replaced atomically on every POST, so a crash resumes from the start of
the interrupted step (spec 7: "autosaved to an event log, resuming after a
crash"). The server treats the snapshot as opaque except for the kept captions
the follow-up needs.

The one thing the server decides from the log is the gate: a clip's inventory
is served only once ``listing.submit`` for that clip is on record (spec 2:
"The skeleton never appears, and is never reachable, until the listing step is
submitted"). Reading it from the append-only file rather than the snapshot
means a browser cannot claim to have listed by editing its own state.

``dpo.core.atomic.atomic_write_bytes`` publishes immutable bytes and refuses to
overwrite, which is right for artifacts and wrong for a snapshot that changes
every few seconds; ``replace_atomically`` reuses its fsynced temporary and
finishes with ``os.replace`` so a reader sees either the old file or the new.
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
LOG_SCHEMA = "dpo.caption-session-log/v1"
FOLLOWUP_SCHEMA = "dpo.caption-session-followup/v1"
GATING_EVENT = "listing.submit"


class SessionLogError(ValueError):
    """A participant id or an event batch the log refuses."""


class FollowupExistsError(SessionLogError):
    """A second follow-up submission for a participant whose first is on disk."""


def kept_caption_in(snapshot: Mapping[str, Any] | None, clip_id: str, shot_id: str) -> str | None:
    """``snapshot.kept[clip_id][shot_id].caption`` from an already loaded snapshot.

    The follow-up reads one snapshot for every check shot (app.followup), so
    the kiosk replacing the file mid-request can never mix two snapshots into
    one participant's captions.
    """
    if snapshot is None:
        return None
    kept = snapshot.get("kept")
    if not isinstance(kept, Mapping):
        return None
    clip = kept.get(clip_id)
    if not isinstance(clip, Mapping):
        return None
    shot = clip.get(shot_id)
    if not isinstance(shot, Mapping):
        return None
    caption = shot.get("caption")
    return caption if isinstance(caption, str) and caption.strip() else None


def _received_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_participant(value: object) -> str:
    if not isinstance(value, str) or not PARTICIPANT_RE.fullmatch(value):
        raise SessionLogError("participant must match [A-Za-z0-9_-]{1,64}")
    return value


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


class EventLog:
    """One log per ``--out``; ``append`` is serialised across the app's threadpool.

    FastAPI runs the sync ``/api/events`` route in a threadpool, so two batches
    can be in flight at once (the debounced POST and the keepalive flush on
    ``visibilitychange``). The lock makes each append — events line, then
    snapshot replace — whole with respect to every other, so the snapshot on
    disk is always one batch's snapshot, written after that batch's events.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def events_path(self, participant: str) -> Path:
        return self.out_dir / f"events-{validate_participant(participant)}.jsonl"

    def snapshot_path(self, participant: str) -> Path:
        return self.out_dir / f"snapshot-{validate_participant(participant)}.json"

    def followup_path(self, participant: str) -> Path:
        return self.out_dir / f"followup-{validate_participant(participant)}.json"

    def append(
        self, participant: str, events: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any] | None
    ) -> int:
        """Append the events, then replace the snapshot; return the count appended.

        The events go first: the gate reads the events file, and a snapshot
        that says "listed" with no ``listing.submit`` on record must never
        open anything.
        """
        for index, event in enumerate(events):
            if not isinstance(event, Mapping) or not isinstance(event.get("type"), str):
                raise SessionLogError(f"events[{index}] must be an object with a string 'type'")
        with self._lock:
            received_at = _received_at()
            lines = [
                json.dumps({**event, "received_at": received_at}, ensure_ascii=False, sort_keys=True)
                for event in events
            ]
            if lines:
                with self.events_path(participant).open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            if snapshot is not None:
                replace_atomically(self.snapshot_path(participant), _json_bytes(snapshot))
        return len(lines)

    def events(self, participant: str) -> list[dict[str, Any]]:
        path = self.events_path(participant)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
        for index, stripped in enumerate(lines):
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                # A torn last line is a write that did not finish — a crash, a
                # full disk — and the events before it are whole; reading them
                # keeps the gate and the log answering. A torn line anywhere
                # else is corruption, and stays an error.
                if index == len(lines) - 1:
                    break
                raise
        return rows

    def snapshot(self, participant: str) -> dict[str, Any] | None:
        path = self.snapshot_path(participant)
        if not path.is_file():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None

    def listing_submitted(self, participant: str, clip_id: str) -> bool:
        """The gate (contract §6): scan the append-only file for the event."""
        return any(
            event.get("type") == GATING_EVENT and event.get("clip_id") == clip_id
            for event in self.events(participant)
        )

    def kept_caption(self, participant: str, clip_id: str, shot_id: str) -> str | None:
        """``snapshot.kept[clip_id][shot_id].caption`` — the participant's own caption."""
        return kept_caption_in(self.snapshot(participant), clip_id, shot_id)

    def write_followup(self, participant: str, document: Mapping[str, Any]) -> Path:
        """Write the follow-up responses once, stamped with the receipt time.

        The first submission is the record: a second one (another device, a
        cleared browser) is refused rather than silently replacing it, the way
        the events file is never rewritten.
        """
        path = self.followup_path(participant)
        with self._lock:
            if path.exists():
                raise FollowupExistsError(f"follow-up responses for {participant!r} already recorded")
            replace_atomically(path, _json_bytes({**document, "received_at": _received_at()}))
        return path

    def export(self, participant: str, session_id: str) -> dict[str, Any]:
        """The download on the final screen (contract §7 ``/api/log``)."""
        return {
            "schema": LOG_SCHEMA,
            "session_id": session_id,
            "participant": validate_participant(participant),
            "events": self.events(participant),
            "snapshot": self.snapshot(participant),
        }
