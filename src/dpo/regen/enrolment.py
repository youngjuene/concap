"""§1: issuing a participant identifier and the sequence number that assigns them.

The assignment alternates by sequence number, so the sequence number has to be
allocated once per participant and then never move. Two failures are worth
guarding against, and both have happened to studies before.

*Re-entry must not re-roll.* A participant who reloads before finishing gets
their existing sequence number back, or they would change condition mid-session
and their two viewings would no longer be the pair the design calls for. The
roster is therefore keyed by participant identifier and written before it is
used.

*Two allocations must not share a number.* The next number is read and written
under the same lock the roster is written under, so two requests arriving at
once take two numbers rather than one twice.

What re-entry rests on is the identifier the page sends, and the page keeps it
in ``sessionStorage`` — per tab, by design. A reload resumes; a genuinely new
tab sends nothing, enrols anew, and takes the next number. That is the right
trade for a kiosk, where the alternative — an identifier that outlives the tab
— would sit participant two down inside participant one's session. The cost is
that a tab opened by mistake spends a sequence number, and §1's alternation
runs one step out for everyone after it. Such an entry is identifiable rather
than invisible: it is a roster entry with no viewing rows against it.

The roster is a plain file the operator can read. A study that loses it loses
the mapping from identifier to sequence number, but not the assignment: every
viewing row carries the sequence and both segment assignments, so the analysis
reconstructs it from the logs alone.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import json
import secrets
import threading
from pathlib import Path
from typing import Any

from dpo.core.atomic import replace_atomically
from dpo.regen.assignment import Assignment, assign
from dpo.regen.log import validate_participant

ROSTER_SCHEMA = "dpo.caption-regen-roster/v1"
ROSTER_FILE = "roster.json"
# Six hex characters: enough that a researcher reading two logs side by side is
# not comparing p1 with p2, short enough to read aloud in a lab.
ISSUED_BYTES = 3


class Roster:
    """Participant identifiers and their sequence numbers, on disk."""

    def __init__(self, out_dir: Path) -> None:
        self.path = Path(out_dir) / ROSTER_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema": ROSTER_SCHEMA, "participants": {}}
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("participants"), dict):
            raise ValueError(f"{self.path}: not a {ROSTER_SCHEMA} roster")
        return loaded

    def sequence_of(self, participant: str) -> int | None:
        entry = self._read()["participants"].get(validate_participant(participant))
        return int(entry["sequence"]) if isinstance(entry, dict) else None

    def enrol(self, participant: str | None = None) -> tuple[str, Assignment]:
        """The identifier and assignment for a participant, allocating if new.

        Passing an identifier returns that participant's existing assignment
        unchanged; passing none issues a fresh identifier and the next sequence
        number. Both paths go through the same lock, so a reload and a second
        tab cannot allocate twice.
        """
        with self._lock:
            roster = self._read()
            people: dict[str, Any] = roster["participants"]
            if participant is not None:
                validate_participant(participant)
                existing = people.get(participant)
                if isinstance(existing, dict):
                    return participant, assign(int(existing["sequence"]))
            else:
                participant = f"p{secrets.token_hex(ISSUED_BYTES)}"
                while participant in people:
                    participant = f"p{secrets.token_hex(ISSUED_BYTES)}"
            sequence = len(people)
            people[participant] = {"sequence": sequence}
            replace_atomically(
                self.path, json.dumps(roster, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
            )
        return participant, assign(sequence)
