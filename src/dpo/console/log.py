"""The console's event log: the full event stream, stamped with the config hash.

Per participant under ``--out``: ``events-<p>.jsonl`` is append-only and never
rewritten, and ``snapshot-<p>.json`` is the browser's whole resumable state,
replaced atomically on every POST so a crash resumes from the interrupted step
(§7). The server treats the snapshot as opaque except for the committed
captions the check needs.

Two things are particular to this instrument.

Every line carries the configuration hash (§10: "Every caption, endpoint, and
log entry carries that hash, so any result is reproducible from its stamp").
The stamp is applied here, on receipt, from the configuration the server is
running under — never from anything the browser sends, which could be a stale
tab holding an older session's hash.

The §9 measurements are written by the server, not by the browser. They are
"computed for analysis and never feed generation" and "hidden from participants
entirely", so ``C_anch``, ``D`` and ``ρ_g`` reach the record without ever
having been in a page. :meth:`measure_shot` writes one such line the first time
a shot is opened, and is idempotent, so re-entering a shot does not restate
what has not changed.

The full stream is kept because "trajectories in adjustment paradigms routinely
carry more information than endpoints" (§11). Nothing here summarizes.

Section numbers cite ``spec-system.md`` except §7, which is ``spec-uiux.md``.
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
LOG_SCHEMA = "dpo.caption-console-log/v1"
SNAPSHOT_SCHEMA = "dpo.caption-console-snapshot/v1"
MEASUREMENT_EVENT = "shot.measured"


class ConsoleLogError(ValueError):
    """A participant id or an event batch the log refuses."""


def validate_participant(value: str | None) -> str:
    if value is None or not PARTICIPANT_RE.fullmatch(value):
        raise ConsoleLogError("participant must match [A-Za-z0-9_-]{1,64}")
    return value


def _received_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def committed_in(snapshot: Mapping[str, Any] | None, clip_id: str, shot_id: str) -> str | None:
    """``snapshot.committed[clip_id][shot_id].caption`` from a loaded snapshot.

    The check reads one snapshot for every shot it compares, so the kiosk
    replacing the file mid-request cannot mix two snapshots into one
    participant's captions.
    """
    if snapshot is None:
        return None
    committed = snapshot.get("committed")
    if not isinstance(committed, Mapping):
        return None
    clip = committed.get(clip_id)
    if not isinstance(clip, Mapping):
        return None
    shot = clip.get(shot_id)
    if not isinstance(shot, Mapping):
        return None
    caption = shot.get("caption")
    return caption if isinstance(caption, str) else None


class EventLog:
    """One log per ``--out``; every append is serialised across the threadpool.

    FastAPI runs the sync ``/api/events`` route in a threadpool, so two batches
    can be in flight at once (the debounced POST and the keepalive flush on
    ``visibilitychange``). The lock makes each append — events line, then
    snapshot replace — whole with respect to every other.
    """

    def __init__(self, out_dir: Path, config_hash: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self._lock = threading.Lock()

    def events_path(self, participant: str) -> Path:
        return self.out_dir / f"events-{validate_participant(participant)}.jsonl"

    def snapshot_path(self, participant: str) -> Path:
        return self.out_dir / f"snapshot-{validate_participant(participant)}.json"

    def _write(self, participant: str, events: Sequence[Mapping[str, Any]]) -> int:
        received_at = _received_at()
        lines = [
            json.dumps(
                {**event, "received_at": received_at, "config_hash": self.config_hash},
                ensure_ascii=False,
                sort_keys=True,
            )
            for event in events
        ]
        if lines:
            with self.events_path(participant).open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return len(lines)

    def append(
        self, participant: str, events: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any] | None
    ) -> int:
        """Append the events, then replace the snapshot; return the count appended."""
        for index, event in enumerate(events):
            if not isinstance(event, Mapping) or not isinstance(event.get("type"), str):
                raise ConsoleLogError(f"events[{index}] must be an object with a string 'type'")
        with self._lock:
            written = self._write(participant, events)
            if snapshot is not None:
                replace_atomically(self.snapshot_path(participant), _json_bytes(snapshot))
        return written

    def events(self, participant: str) -> list[dict[str, Any]]:
        path = self.events_path(participant)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    def snapshot(self, participant: str) -> dict[str, Any] | None:
        path = self.snapshot_path(participant)
        if not path.is_file():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None

    def measure_shot(
        self, participant: str, clip_id: str, shot_id: str, measurements: Mapping[str, Any]
    ) -> bool:
        """Record §9 for one shot, once. True if this call wrote the line.

        Written server-side and never sent to the browser. Idempotent per
        (participant, clip, shot, config hash): the measurements are a function
        of the document and the calibration, so a second line would restate a
        constant. A session re-served under a changed calibration writes a new
        line, because then it is not a constant any more.
        """
        with self._lock:
            for event in self.events(participant):
                if (
                    event.get("type") == MEASUREMENT_EVENT
                    and event.get("clip_id") == clip_id
                    and event.get("shot_id") == shot_id
                    and event.get("config_hash") == self.config_hash
                ):
                    return False
            self._write(
                participant,
                [
                    {
                        "type": MEASUREMENT_EVENT,
                        "clip_id": clip_id,
                        "shot_id": shot_id,
                        "measurements": dict(measurements),
                    }
                ],
            )
        return True

    def committed_caption(self, participant: str, clip_id: str, shot_id: str) -> str | None:
        return committed_in(self.snapshot(participant), clip_id, shot_id)

    def export(self, participant: str, session_id: str) -> dict[str, Any]:
        """The download on the final screen."""
        return {
            "schema": LOG_SCHEMA,
            "session_id": session_id,
            "config_hash": self.config_hash,
            "participant": validate_participant(participant),
            "events": self.events(participant),
            "snapshot": self.snapshot(participant),
        }
