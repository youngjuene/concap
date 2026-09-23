"""Contract fixtures contain no real participant records or credentials."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from dpo.regen.dashboard_data import DashboardData
from dpo.regen.study_schema import digest

SECRET = "fixture-bearer-do-not-export"
PROMPT = "fixture-private-model-instruction"
MEDIA_PATH = "/private/research/media/secret.wav"
XSS = '<img src=x onerror="window.dashboardXss=1">'
STAMP = "2026-09-13T12:00:00Z"
EPOCH = datetime(2026, 9, 13, 12, tzinfo=UTC).timestamp()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_records(root: Path) -> tuple[Path, Path]:
    """Create isolated old/new/partial/QA study fixtures for backend and browser QA."""
    legacy, viewing = root / "legacy", root / "viewing"
    legacy.mkdir(parents=True)
    viewing.mkdir(parents=True)
    old_responses: list[dict[str, Any]] = [
        {
            "page": page,
            "view_id": f"p-old-v{index}",
            "responses": {"art_1": 3},
            "items_digest": "old-phase1",
            "submitted_at": STAMP,
            "config_hash": "old-config",
        }
        for index, page in enumerate(("art", "survey"))
    ]
    new_responses: list[dict[str, Any]] = [
        {
            "page": "art" if index % 2 == 0 else "survey",
            "view_id": f"p-new-v{index}",
            "responses": {"art_1": index + 1},
            "items_digest": "new-phase1",
            "submitted_at": STAMP,
            "scope": "clip_original" if index % 2 == 0 else "clip_updated",
            "condition": "prepared" if index % 2 == 0 else "regenerated",
            "clip_id": f"clip-{index // 2}",
            "flow_version": "clip-caption-prss-v2",
            "config_hash": "new-config",
        }
        for index in range(4)
    ] + [
        {
            "page": "overall",
            "view_id": "p-new-overall",
            "responses": {"prss_1": 5},
            "items_digest": "new-phase1",
            "submitted_at": STAMP,
            "scope": "overall",
            "flow_version": "clip-caption-prss-v2",
            "config_hash": "new-config",
        }
    ]
    for label, response_rows in (("p-old", old_responses), ("p-new", new_responses)):
        language = "en" if label == "p-old" else "ko"
        records = [
            {
                "view_id": row["view_id"],
                "clip_id": row.get("clip_id", f"old-clip-{index}"),
                "segment": f"segment-{index // 2}",
                "condition": "prepared" if index % 2 == 0 else "regenerated",
                "language": language,
                "config_hash": row["config_hash"],
                "playback_ended_at": STAMP,
                "captions": [{"text": f"Recorded caption {index}", "start_ms": 0, "end_ms": 5000}],
            }
            for index, row in enumerate(response_rows)
            if row["page"] != "overall"
        ]
        _write_jsonl(legacy / f"responses-{label}.jsonl", response_rows)
        _write_jsonl(legacy / f"viewings-{label}.jsonl", records)
        (legacy / f"snapshot-{label}.json").write_text(
            json.dumps(
                {
                    "step": "done",
                    "language": language,
                    "regeneration_fallback": True,
                    "regeneration_fallback_by_segment": {"segment-0": False, "segment-1": True},
                    "draft": {"must_not_export": SECRET},
                }
            ),
            encoding="utf-8",
        )
    _write_jsonl(
        legacy / "events-p-new.jsonl",
        [
            {
                "type": "step.entered",
                "step": "art",
                "received_at": STAMP,
                "token": SECRET,
                "path": MEDIA_PATH,
            },
            {
                "type": "regeneration.written",
                "segment": "segment-0",
                "fallback": False,
                "prompt": PROMPT,
                "duration_ms": 88888,
                "received_at": STAMP,
                "settings": {"secret": SECRET},
            },
        ],
    )
    for label in ("p-incomplete", "qa-browser"):
        (legacy / f"snapshot-{label}.json").write_text(
            json.dumps({"step": "art", "language": "en", "draft": {"art_1": 7}}), encoding="utf-8"
        )
    old_instrument = {
        "hash": "old-instrument",
        "language": "en",
        "flow": "long-video-surveys/v1",
        "items": {"final": [{"id": "useful", "text": "The captions helped.", "type": "rating", "points": 5}]},
    }
    new_instrument = {
        "hash": "new-instrument",
        "language": "ko",
        "flow": "long-video-surveys/v2",
        "prss": {"items_digest": "new-phase1"},
        "items": {
            "video": [
                {"id": "useful", "text": "자막이 도움이 되었습니다.", "type": "rating", "points": 5},
                {"id": "na_item", "text": "해당 없음 응답", "type": "rating", "points": 5},
            ],
            "final": [
                {
                    "id": "prss_1",
                    "text": "이 장소는 편안했습니다.",
                    "type": "rating",
                    "points": 7,
                    "block": "prss",
                },
                {"id": "comment", "text": "기타 의견", "type": "text"},
            ],
        },
    }
    new_state: dict[str, Any] = {
        "session_id": "public-new-session",
        "stage": "done",
        "language": "ko",
        "flow_version": 2,
        "instrument": new_instrument,
        "calibration_source": {
            "participant": "p-new",
            "responses": new_responses,
            "viewings": [],
            "session_id": "not-a-participant-token",
        },
        "profile": {
            "instruction": PROMPT,
            "questionnaire": {
                "items_digest": "new-phase1",
                "language": "ko",
                "scale": {"points": 7},
                "items": {"art_1": {"id": "art_1", "text": "집중하기 쉬웠습니다.", "reverse": False}},
            },
        },
        "viewing": [{"id": "viewing-1", "audio": MEDIA_PATH}, {"id": "viewing-2", "audio": MEDIA_PATH}],
        "completions": [{"video_id": "viewing-1"}, {"video_id": "viewing-2"}],
        "video_surveys": [
            {
                "video_id": "viewing-1",
                "answers": {"useful": 0, "na_item": "na"},
                "instrument_hash": "new-instrument",
            },
            {
                "video_id": "viewing-2",
                "answers": {"useful": 4, "na_item": None},
                "instrument_hash": "new-instrument",
            },
        ],
        "final_survey": {"answers": {"prss_1": 6, "comment": XSS}, "instrument": new_instrument},
        "draft": {"comment": "do not export draft"},
    }
    old_state = {
        "stage": "done",
        "language": "en",
        "flow_version": 1,
        "instrument": old_instrument,
        "final_survey": {"answers": {"useful": 5}, "items_hash": "old-final"},
        "completions": [{"video_id": "old-viewing-1"}],
    }
    with sqlite3.connect(viewing / "study.sqlite3") as db:
        db.executescript("""
            CREATE TABLE sessions(id TEXT PRIMARY KEY, state TEXT);
            CREATE TABLE calibration_links(source TEXT PRIMARY KEY, token TEXT);
            CREATE TABLE events(session TEXT, kind TEXT, body TEXT, at REAL);
            CREATE TABLE jobs(id TEXT, session TEXT, epoch INTEGER, revision INTEGER, cue INTEGER,
                              state TEXT, spec TEXT, result TEXT, created REAL);
            CREATE TABLE exposures(session TEXT, id TEXT, body TEXT);
        """)
        db.executemany(
            "INSERT INTO sessions VALUES(?,?)",
            [
                (SECRET, json.dumps(new_state)),
                ("old-secret", json.dumps(old_state)),
                ("pending-secret", json.dumps({"stage": "awaiting-calibration", "language": "en"})),
            ],
        )
        db.executemany(
            "INSERT INTO calibration_links VALUES(?,?)",
            [
                ("private-namespace:p-old", "old-secret"),
                ("private-namespace:p-incomplete", "pending-secret"),
            ],
        )
        mutations = [
            ("playback", {"video_id": "viewing-1", "position_ms": 99}),
            ("settings", {"video_id": "viewing-1", "texture": 0, "context": 1}),
            ("settings", {"video_id": "viewing-1", "position_hint_ms": 0, "texture": 0.5, "context": 0}),
            ("settings", {"video_id": "viewing-1", "position_hint_ms": 5000, "texture": 0.9, "context": 0.8}),
            ("video-survey", {"video_id": "viewing-1", "answers": {"secret": SECRET}}),
            ("video-survey", {"video_id": "viewing-2"}),
            ("final-survey", {"comment": XSS}),
        ]
        db.executemany(
            "INSERT INTO events VALUES(?,?,?,?)",
            [
                (SECRET, "mutation", json.dumps({"action": action, "data": data}), EPOCH + index)
                for index, (action, data) in enumerate(mutations)
            ],
        )
        db.execute(
            "INSERT INTO events VALUES(?,?,?,?)",
            ("old-secret", "mutation", json.dumps({"action": "final-survey", "data": {"useful": 5}}), EPOCH),
        )
        for index, (job_id, status, duration) in enumerate(
            [
                ("shown-job", "succeeded", 0),
                ("unshown-job", "succeeded", 100),
                ("failed-job", "failed", 300),
                ("queued-job", "queued", 99999),
            ]
        ):
            db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    SECRET,
                    9,
                    index,
                    index,
                    status,
                    json.dumps(
                        {
                            "start_ms": index * 5000,
                            "end_ms": (index + 1) * 5000,
                            "axes": {"texture": 0, "context": 0.5},
                            "instruction": PROMPT,
                            "audio": MEDIA_PATH,
                            "video_hash": digest(new_state["viewing"][0]),
                        }
                    ),
                    json.dumps(
                        {
                            "text": f"Model caption {index}",
                            "duration_ms": duration,
                            "fallback": status == "failed",
                            "prompt": PROMPT,
                        }
                    ),
                    EPOCH + index,
                ),
            )
        for index in range(3):
            db.execute(
                "INSERT INTO exposures VALUES(?,?,?)",
                (
                    SECRET,
                    f"exposure-{index}",
                    json.dumps(
                        {
                            "video_id": "viewing-1",
                            "text": "Shown caption" if index < 2 else "Prepared fallback",
                            "job_id": "shown-job" if index < 2 else None,
                            "settings_revision": 0 if index < 2 else None,
                            "start_ms": index * 1000,
                            "end_ms": (index + 1) * 1000,
                            "fallback": index == 2,
                            "axes": {"texture": 0, "context": 0.5} if index < 2 else None,
                            "token": SECRET,
                        }
                    ),
                ),
            )
    return legacy, viewing


def test_protocol_versions_join_deduplicate_and_preserve_wording(tmp_path: Path) -> None:
    data = DashboardData(*make_records(tmp_path))
    overview = data.overview()
    assert overview["metrics"]["participants"] == 3
    assert overview["metrics"]["phase1_complete"] == 2
    assert overview["metrics"]["phase2_started"] == 2
    assert overview["metrics"]["phase2_complete"] == 2
    assert overview["metrics"]["survey_submissions"] == 11
    assert overview["metrics"]["response_values"] == 14
    assert overview["activity"] == [{"date": "2026-09-13", "phase1": 7, "phase2": 4}]
    newer = next(row for row in overview["participants"] if row["label"] == "p-new")
    assert newer["phase1_surveys"] == 5
    detail = data.participant(newer["id"])
    phase1 = [row for row in detail["responses"] if row["phase"] == 1]
    assert len(phase1) == 5
    assert all(row["wording_available"] for row in phase1)
    assert all(row["points"] == 7 for row in phase1)
    assert phase1[0]["item_text"] == "집중하기 쉬웠습니다."
    assert {row["flow_version"] for row in overview["responses"]} == {
        "legacy",
        "clip-caption-prss-v2",
        "long-video-surveys/v1",
        "long-video-surveys/v2",
    }
    old_phase1 = [row for row in overview["responses"] if row["instrument"] == "old-phase1"]
    assert all(not row["wording_available"] and row["points"] is None for row in old_phase1)
    assert not overview["warnings"]


def test_values_control_positions_and_job_durations_are_not_inferred(tmp_path: Path) -> None:
    data = DashboardData(*make_records(tmp_path))
    overview = data.overview()
    person = next(row for row in overview["participants"] if row["label"] == "p-new")
    detail = data.participant(person["id"])
    assert [row["position_ms"] for row in detail["interactions"]] == [None, 0, 5000]
    assert detail["interactions"][0]["texture"] == 0
    assert all(row["revision"] is None for row in detail["interactions"])
    values = [row["value"] for row in detail["responses"]]
    assert 0 in values and "na" in values and None in values and XSS in values
    assert overview["metrics"]["generation_median_ms"] == 100
    assert overview["metrics"]["control_changes"] == 3
    shown = [row for row in detail["captions"] if row["phase"] == 2 and row["displayed"]]
    assert len(shown) == 3
    assert shown[0]["generation_ms"] == 0
    assert shown[-1]["generation_ms"] is None
    unshown = [row for row in detail["captions"] if not row["displayed"]]
    assert len(unshown) == 3
    assert {row["status"] for row in unshown} == {"queued", "succeeded", "failed"}
    assert all(row["kind"] == "job" and row["video_id"] == "viewing-1" for row in unshown)
    assert overview["metrics"]["generated_exposures"] == 4
    assert overview["metrics"]["fallback_exposures"] == 2


def test_no_credentials_prompts_paths_or_drafts_and_sources_unchanged(tmp_path: Path) -> None:
    paths = make_records(tmp_path)
    before = {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    data = DashboardData(*paths)
    overview = data.overview(include_qa=True)
    exported = json.dumps(
        [overview, *[data.participant(row["id"], include_qa=True) for row in overview["participants"]]]
    )
    for forbidden in (
        SECRET,
        "old-secret",
        "pending-secret",
        PROMPT,
        MEDIA_PATH,
        "private-namespace",
        "do not export draft",
    ):
        assert forbidden not in exported
    after = {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_qa_only_uses_explicit_labels_and_selectors_are_required(tmp_path: Path) -> None:
    paths = make_records(tmp_path)
    with sqlite3.connect(paths[1] / "study.sqlite3") as db:
        db.execute(
            "INSERT INTO sessions VALUES(?,?)", ("test-looking-credential", json.dumps({"stage": "watch"}))
        )
    data = DashboardData(*paths)
    assert data.overview()["metrics"]["participants"] == 4
    all_rows = data.overview(include_qa=True)["participants"]
    assert len(all_rows) == 5
    qa = next(row for row in all_rows if row["qa"])
    with pytest.raises(KeyError):
        data.participant(qa["id"])
    assert data.participant(qa["id"], include_qa=True)["participant"]["label"] == "qa-browser"
    for selector in ("p-new", SECRET, "../../study.sqlite3"):
        with pytest.raises(KeyError):
            data.participant(selector)


def test_partial_jsonl_and_missing_or_corrupt_sources_are_visible(tmp_path: Path) -> None:
    legacy, viewing = make_records(tmp_path)
    path = legacy / "responses-p-new.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + '\n{"torn":\n' + lines[1] + '\n{"partial"', encoding="utf-8")
    (legacy / "snapshot-p-old.json").write_text("{", encoding="utf-8")
    data = DashboardData(legacy, viewing)
    overview = data.overview()
    assert overview["metrics"]["survey_submissions"] == 11  # Embedded rows supplement partial JSONL.
    assert any("line 2" in warning for warning in overview["warnings"])
    assert any("snapshot" in warning for warning in overview["warnings"])
    assert str(tmp_path) not in json.dumps(overview)
    missing = DashboardData(tmp_path / "absent-phase1", tmp_path / "absent-phase2").overview()
    assert missing["metrics"]["participants"] == 0
    assert missing["metrics"]["generation_median_ms"] is None
    assert len(missing["warnings"]) == 2
    assert not (tmp_path / "absent-phase2").exists()


def test_partial_sqlite_schema_does_not_hide_readable_sessions(tmp_path: Path) -> None:
    legacy, viewing = tmp_path / "legacy", tmp_path / "viewing"
    legacy.mkdir()
    viewing.mkdir()
    with sqlite3.connect(viewing / "study.sqlite3") as db:
        db.execute("CREATE TABLE sessions(id TEXT, state TEXT)")
        db.execute("INSERT INTO sessions VALUES(?,?)", (SECRET, json.dumps({"stage": "ready"})))
        db.execute("INSERT INTO sessions VALUES(?,?)", ("other-secret", "{"))
    overview = DashboardData(legacy, viewing).overview()
    assert overview["metrics"]["participants"] == 2
    assert overview["metrics"]["phase2_started"] == 0
    assert any("events table" in warning for warning in overview["warnings"])
    assert any("invalid JSON" in warning for warning in overview["warnings"])
    assert SECRET not in json.dumps(overview)


def test_read_snapshot_stays_consistent_during_a_wal_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, viewing = make_records(tmp_path)
    original_connect = sqlite3.connect
    writer = original_connect(viewing / "study.sqlite3")
    writer.execute("PRAGMA journal_mode=WAL")
    committed = False

    def trace(statement: str) -> None:
        nonlocal committed
        if statement.startswith("SELECT source,token") and not committed:
            writer.execute(
                "UPDATE sessions SET state=? WHERE id=?",
                (json.dumps({"stage": "done", "participant": "changed"}), SECRET),
            )
            writer.execute("DELETE FROM events WHERE session=?", (SECRET,))
            writer.commit()
            committed = True

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        assert "mode=ro" in args[0]
        assert kwargs.get("uri") is True
        db: sqlite3.Connection = original_connect(*args, **kwargs)
        db.set_trace_callback(trace)
        return db

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        overview = DashboardData(legacy, viewing).overview()
        assert committed
        assert overview["metrics"]["control_changes"] == 3
        assert overview["metrics"]["survey_submissions"] == 11
        assert not any(row["label"] == "changed" for row in overview["participants"])
    finally:
        writer.close()


def test_opaque_handoff_unknown_exposure_and_pilot_label(tmp_path: Path) -> None:
    legacy, viewing = make_records(tmp_path)
    (legacy / "snapshot-pilot-participant.json").write_text('{"step":"art"}', encoding="utf-8")
    with sqlite3.connect(viewing / "study.sqlite3") as db:
        db.execute("INSERT INTO events VALUES(?,?,?,?)", (SECRET, "mutation", "a" * 64, EPOCH))
        db.execute(
            "INSERT INTO exposures VALUES(?,?,?)",
            (SECRET, "unknown", json.dumps({"text": "Historical caption", "video_id": "viewing-1"})),
        )
    data = DashboardData(legacy, viewing)
    overview = data.overview()
    assert overview["metrics"]["participants"] == 4
    assert overview["metrics"]["generated_exposures"] == 4
    assert any("provenance" in warning for warning in overview["warnings"])
    assert not any("invalid JSON" in warning for warning in overview["warnings"])


def test_wal_database_records_remain_unchanged(tmp_path: Path) -> None:
    legacy, viewing = make_records(tmp_path)
    writer = sqlite3.connect(viewing / "study.sqlite3")
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("UPDATE sessions SET state=state")
        writer.commit()
        before = {path.name: path.read_bytes() for path in viewing.iterdir() if path.is_file()}
        DashboardData(legacy, viewing).overview()
        after = {path.name: path.read_bytes() for path in viewing.iterdir() if path.is_file()}
        assert before == after
    finally:
        writer.close()


def test_torn_utf8_tail_does_not_discard_valid_submissions(tmp_path: Path) -> None:
    legacy, viewing = tmp_path / "legacy", tmp_path / "viewing"
    legacy.mkdir()
    viewing.mkdir()
    valid = json.dumps({"page": "art", "view_id": "real-v0", "responses": {"rating": 4}}).encode()
    (legacy / "responses-real.jsonl").write_bytes(valid + b'\n{"comment":"\xed\x95')
    overview = DashboardData(legacy, viewing).overview()
    assert overview["metrics"]["survey_submissions"] == 1
    assert overview["responses"][0]["value"] == 4
    assert any("line 2" in warning for warning in overview["warnings"])
