"""Delete one participant's records from Neon, for a withdrawal (docs/v4-diff/consent-data-flow.md §5).

    uv run --with "psycopg[binary]==3.3.6" python deploy/vercel/forget.py P12 [--purpose smoke] [--yes]

Without ``--yes`` it only counts what would go. With it, the participant's sessions, receipts,
events and live logs (``logs/sessions/session-<id>.jsonl`` and ``live-<id>.jsonl``) are deleted in
one transaction. The lab's copies are not touched: it lists every pull and export that still holds
the participant, to delete by hand (then pull and export again).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from pull import REPO, VERCEL, database_url

sys.path.insert(0, str(REPO / "src"))

ROWS = {
    "logs": "DELETE FROM logs WHERE name = ANY(%s)",
    "events": "DELETE FROM events WHERE session_id = ANY(%s)",
    "receipts": "DELETE FROM receipts WHERE session_id = ANY(%s)",
    "sessions": "DELETE FROM sessions WHERE session_id = ANY(%s)",
}


def lab_copies(participant: str) -> list[Path]:
    """Every pull and export directory on this machine that holds the participant."""
    found = []
    for store in sorted((VERCEL / "pulls").glob("*/sessions.sqlite")):
        with sqlite3.connect(store) as lite:
            if lite.execute("SELECT 1 FROM sessions WHERE participant = ?", (participant,)).fetchone():
                found.append(store.parent)
    for listing in sorted((REPO / "artifacts/diff").glob("export-*/participants.jsonl")):
        if f'"participant": "{participant}"' in listing.read_text(encoding="utf-8"):
            found.append(listing.parent)
    return found


def main() -> int:
    import psycopg
    from pg_store import DATABASES, on_database

    from diff.session_store import PARTICIPANT_ID

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("participant")
    parser.add_argument("--purpose", choices=tuple(DATABASES), default="participant")
    parser.add_argument("--yes", action="store_true", help="delete; without it, only count")
    args = parser.parse_args()
    if not PARTICIPANT_ID.fullmatch(args.participant):
        raise SystemExit(f"{args.participant!r} is not a participant ID (P1 to P9999)")
    url = on_database(database_url(), DATABASES[args.purpose])
    with psycopg.connect(url, prepare_threshold=None) as pg:
        ids = [
            row[0]
            for row in pg.execute(
                "SELECT session_id FROM sessions WHERE participant = %s", (args.participant,)
            )
        ]
        names = [f"sessions/{kind}-{session_id}.jsonl" for session_id in ids for kind in ("session", "live")]
        counts = {}
        for table, statement in ROWS.items():
            keys = names if table == "logs" else ids
            count = statement.replace("DELETE", "SELECT count(*)", 1)
            counts[table] = pg.execute(count, (keys,)).fetchone()[0]
            if args.yes:
                pg.execute(statement, (keys,))
        # psycopg commits on leaving the block; without --yes nothing was written.
    verb = "deleted" if args.yes else "would delete"
    print(f"{verb} {args.participant} ({args.purpose}): " + ", ".join(f"{n} {t}" for t, n in counts.items()))
    for copy in lab_copies(args.participant):
        print(f"  still in the lab: {copy.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
