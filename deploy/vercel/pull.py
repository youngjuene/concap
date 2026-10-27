"""Pull the Vercel study's sessions and live logs from Neon into a lab store on this machine.

    uv run --with "psycopg[binary]==3.3.6" python deploy/vercel/pull.py
    uv run python -m diff export --store PULL/sessions.sqlite --log-dir PULL/logs \\
        --phase1-setup artifacts/diff/phase1/setup-draft.json --out artifacts/diff/export-YYYYMMDD

Each pull is a new directory under ``artifacts/diff/vercel/pulls/`` holding:
- ``sessions.sqlite``, a ``diff.session-store/v1`` store with Neon's sessions, receipts and
  events copied row for row (event ids and order kept), which ``diff export`` reads as it reads
  the lab's own;
- ``logs/``, the live logs (``pg_store.PgLogDir`` rows) written back as the files the lab server
  writes: ``logs/sessions/session-<id>.jsonl`` and ``logs/sessions/live-<id>.jsonl``;
- ``pull.json``: when, from which database host, and the row counts.

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


def main() -> int:
    import psycopg

    from diff.session_store import SessionStore

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="a new directory (default: pulls/<time>)")
    args = parser.parse_args()
    out: Path = args.out or VERCEL / "pulls" / time.strftime("%Y%m%d-%H%M%S")
    if out.exists():
        raise SystemExit(f"{out} exists; a pull is always a new directory")
    url = database_url()
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
        for table, query in tables.items():
            rows = pg.execute(query).fetchall()
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
    (out / "logs").mkdir(exist_ok=True)

    record = {
        "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": urlsplit(url).hostname,
        "counts": counts,
    }
    (out / "pull.json").write_text(json.dumps(record, indent=2) + "\n")
    out.rename(final)
    out = final
    print(
        f"pulled {counts['sessions']} sessions, {counts['events']} events, "
        f"{counts['log_lines']} log lines into {out}"
    )
    where = out.resolve().relative_to(REPO) if out.resolve().is_relative_to(REPO) else out.resolve()
    print(
        f"next: uv run python -m diff export --store {where}/sessions.sqlite "
        f"--log-dir {where}/logs --phase1-setup artifacts/diff/phase1/setup-draft.json "
        f"--out artifacts/diff/export-{time.strftime('%Y%m%d')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
