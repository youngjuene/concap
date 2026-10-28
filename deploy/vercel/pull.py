"""Pull the Vercel study's sessions and live logs from Neon into a lab store on this machine.

    uv run --with "psycopg[binary]==3.3.6" python deploy/vercel/pull.py [--purpose smoke]
    uv run python -m diff export --store PULL/sessions.sqlite --log-dir PULL/logs \\
        --phase1-setup artifacts/diff/phase1/setup-draft.json --out artifacts/diff/export-YYYYMMDD

Each pull is a new directory under ``artifacts/diff/vercel/pulls/`` holding:
- ``sessions.sqlite``, a ``diff.session-store/v1`` store with Neon's sessions, receipts and
  events copied row for row (event ids and order kept), which ``diff export`` reads as it reads
  the lab's own;
- ``logs/``, the live logs (``pg_store.PgLogDir`` rows) written back as the files the lab server
  writes: ``logs/sessions/session-<id>.jsonl`` and ``logs/sessions/live-<id>.jsonl``;
- ``pull.json``: when, from which database, and the row counts.

``--purpose`` picks the database (:data:`pg_store.DATABASES`): the participants' by default, the
smoke tests' with ``smoke``. The pull ends with a health line: sessions started and finished, and
every live watch that resumed with its video restarted, which happens when a request reaches
another Vercel instance than the one that ran the watch (README, "One caveat for phase 2").

The connection string is ``DATABASE_URL``, or the line of that name in
``artifacts/diff/vercel/.env.production`` (``vercel env pull`` there, kept out of git). The pull
only reads Neon; the study keeps running.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
VERCEL = REPO / "artifacts/diff/vercel"
sys.path.insert(0, str(REPO / "src"))


def database_url() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env = VERCEL / ".env.production"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name == "DATABASE_URL":
                return value.strip().strip('"')
    raise SystemExit(f"set DATABASE_URL, or run `vercel env pull {env} --environment production` in the site")


def restarted_watches(files: dict[str, list[str]]) -> list[str]:
    """``session (video n)`` for every live watch that resumed with its video restarted."""
    found = []
    for lines in files.values():
        for line in lines:
            event = json.loads(line)
            if event.get("event") == "session_resumed" and event.get("video_restarted"):
                found.append(f"{event.get('session_id')} at {event.get('at')}")
    return found


def main() -> int:
    import psycopg
    from pg_store import DATABASES, on_database

    from diff.session_store import SessionStore

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--purpose", choices=tuple(DATABASES), default="participant")
    parser.add_argument("--out", type=Path, help="a new directory (default: pulls/<time>-<purpose>)")
    args = parser.parse_args()
    out: Path = args.out or VERCEL / "pulls" / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.purpose}"
    if out.exists():
        raise SystemExit(f"{out} exists; a pull is always a new directory")
    url = on_database(database_url(), DATABASES[args.purpose])
    final, out = out, out.with_name(out.name + ".partial")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    store_path = out / "sessions.sqlite"
    SessionStore(store_path)  # the lab's own schema
    counts = {}
    with psycopg.connect(url) as pg, sqlite3.connect(store_path) as lite:
        # One snapshot for every table, so a session's events and receipts agree with its state.
        pg.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
        tables = {
            "sessions": "SELECT session_id, participant, token, state, revision, created_at, finished_at"
            " FROM sessions ORDER BY created_at",
            "receipts": "SELECT session_id, key, request_sha256, reply, at FROM receipts ORDER BY at",
            "events": "SELECT id, session_id, at, body FROM events ORDER BY id",
        }
        # A database no deploy has served yet has no tables: an empty pull.
        (has_store,) = pg.execute("SELECT to_regclass('sessions') IS NOT NULL").fetchone()
        for table, query in tables.items():
            rows = pg.execute(query).fetchall() if has_store else []
            if rows:
                marks = ", ".join("?" * len(rows[0]))
                lite.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
            counts[table] = len(rows)
        # Deploys before 2026-10-27 16:30 wrote their logs to /tmp only: no table, no lines.
        (has_logs,) = pg.execute("SELECT to_regclass('logs') IS NOT NULL").fetchone()
        logs = pg.execute("SELECT name, line FROM logs ORDER BY id").fetchall() if has_logs else []
    counts["log_lines"] = len(logs)
    files: dict[str, list[str]] = {}
    for name, line in logs:
        files.setdefault(name, []).append(line)
    for name, lines in files.items():
        target = (out / "logs" / name).resolve()
        if not target.is_relative_to((out / "logs").resolve()):
            raise SystemExit(f"a log row names a path outside the pull: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    counts["log_files"] = len(files)
    finished = lite_count(store_path, "SELECT count(*) FROM sessions WHERE finished_at IS NOT NULL")
    restarted = restarted_watches(files)
    (out / "logs").mkdir(exist_ok=True)

    record = {
        "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": args.purpose,
        "host": urlsplit(url).hostname,
        "database": DATABASES[args.purpose],
        "counts": counts,
        "finished": finished,
        "restarted_watches": restarted,
    }
    (out / "pull.json").write_text(json.dumps(record, indent=2) + "\n")
    out.rename(final)
    out = final
    print(
        f"pulled {counts['sessions']} {args.purpose} sessions, {counts['events']} events, "
        f"{counts['log_lines']} log lines into {out}"
    )
    print(f"health: {finished} of {counts['sessions']} sessions finished; {len(restarted)} restarted watches")
    for watch in restarted:
        print(f"  !! a live watch restarted its video: {watch}")
    where = out.resolve().relative_to(REPO) if out.resolve().is_relative_to(REPO) else out.resolve()
    print(
        f"next: uv run python -m diff export --store {where}/sessions.sqlite "
        f"--log-dir {where}/logs --phase1-setup artifacts/diff/phase1/setup-draft.json "
        f"--out artifacts/diff/export-{time.strftime('%Y%m%d')}"
    )
    return 0


def lite_count(path: Path, query: str) -> int:
    with sqlite3.connect(path) as lite:
        return int(lite.execute(query).fetchone()[0])


if __name__ == "__main__":
    raise SystemExit(main())
