"""Dashboard scope normalization for scoped event delivery."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dpo.regen.dashboard_data import DashboardData

STAMP = "2026-09-22T01:02:03.456Z"
EPOCH = datetime(2026, 9, 22, 1, 2, 3, tzinfo=UTC).timestamp()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    legacy, viewing = tmp_path / "legacy", tmp_path / "viewing"
    legacy.mkdir()
    viewing.mkdir()
    return legacy, viewing


def test_phase1_scoped_events_keep_explicit_identity_without_context_id(tmp_path: Path) -> None:
    legacy, viewing = make_dirs(tmp_path)
    write_jsonl(
        legacy / "events-p1.jsonl",
        [
            {
                "event_id": "p1-event-clip",
                "type": "playback.sample",
                "phase": 1,
                "stage": "clip_original",
                "clip_id": "clip-A",
                "clip_index": 0,
                "view_id": "p1-v0",
                "condition": "prepared",
                "flow_version": "flow-v2",
                "context_id": "do-not-export-context-hash",
                "session_id": "public-session-1",
                "config_hash": "config-a",
                "at": STAMP,
                "position_ms": 1200,
            },
            {
                "event_id": "p1-event-overall",
                "type": "overall.opened",
                "phase": 1,
                "stage": "overall",
                "clip_id": None,
                "clip_index": None,
                "view_id": None,
                "condition": None,
                "flow_version": "flow-v2",
                "session_id": "public-session-1",
                "config_hash": "config-a",
                "at": STAMP,
            },
            {"type": "step.entered", "step": "legacy-art", "received_at": STAMP},
        ],
    )

    data = DashboardData(legacy, viewing)
    participant = data.overview()["participants"][0]
    detail = data.participant(participant["id"])
    clip = next(row for row in detail["events"] if row["kind"] == "playback.sample")
    overall = next(row for row in detail["events"] if row["kind"] == "overall.opened")
    historical = next(row for row in detail["events"] if row["kind"] == "step.entered")

    assert clip["stage"] == "clip_original"
    assert clip["clip_id"] == "clip-A"
    assert clip["clip_index"] == 0
    assert clip["view_id"] == "p1-v0"
    assert clip["condition"] == "prepared"
    assert clip["flow_version"] == "flow-v2"
    assert clip["session_id"] == "public-session-1"
    assert clip["config_hash"] == "config-a"
    assert clip["video_id"] == "clip-A"
    assert "context_id" not in json.dumps(detail)
    assert detail["participant"]["videos_total"] is None

    assert overall["clip_id"] is None
    assert overall["view_id"] is None
    assert overall["condition"] is None
    assert overall["video_id"] is None

    assert historical["clip_id"] is None
    assert historical["view_id"] is None
    assert historical["condition"] is None
    assert historical["flow_version"] is None


def test_phase2_settings_revision_remains_joined_to_event_and_interaction(tmp_path: Path) -> None:
    legacy, viewing = make_dirs(tmp_path)
    with sqlite3.connect(viewing / "study.sqlite3") as db:
        db.executescript("""
            CREATE TABLE sessions(id TEXT PRIMARY KEY, state TEXT);
            CREATE TABLE calibration_links(source TEXT PRIMARY KEY, token TEXT);
            CREATE TABLE events(session TEXT, kind TEXT, body TEXT, at REAL);
            CREATE TABLE jobs(id TEXT, session TEXT, epoch INTEGER, revision INTEGER, cue INTEGER,
                              state TEXT, spec TEXT, result TEXT, created REAL);
            CREATE TABLE exposures(session TEXT, id TEXT, body TEXT);
        """)
        db.execute(
            "INSERT INTO sessions VALUES(?,?)",
            (
                "private-session-token",
                json.dumps(
                    {
                        "participant": "p2",
                        "stage": "watch",
                        "language": "en",
                        "viewing": [{"id": "viewing-1"}, {"id": "viewing-2"}],
                    }
                ),
            ),
        )
        db.execute(
            "INSERT INTO events VALUES(?,?,?,?)",
            (
                "private-session-token",
                "mutation",
                json.dumps(
                    {
                        "action": "settings",
                        "data": {
                            "stage": "watch",
                            "video_id": "viewing-1",
                            "position_hint_ms": 5000,
                            "texture": 0.25,
                            "context": 0.75,
                            "settings_revision": 7,
                        },
                    }
                ),
                EPOCH,
            ),
        )

    data = DashboardData(legacy, viewing)
    participant = data.overview()["participants"][0]
    detail = data.participant(participant["id"])
    event = next(row for row in detail["events"] if row["kind"] == "settings")
    interaction = detail["interactions"][0]

    assert event["stage"] == "watch"
    assert event["video_id"] == "viewing-1"
    assert event["detail"]["settings_revision"] == 7
    assert interaction["video_id"] == "viewing-1"
    assert interaction["position_ms"] == 5000
    assert interaction["revision"] == 7
    assert detail["participant"]["videos_total"] == 2
    assert "private-session-token" not in json.dumps(detail)
