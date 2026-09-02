"""The event log: appended and never rewritten, a snapshot that is whole or absent, and the gate."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from dpo.session.log import (
    EventLog,
    FollowupExistsError,
    SessionLogError,
    kept_caption_in,
    validate_participant,
)


def test_events_append_and_the_snapshot_is_replaced_whole(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "out")
    first = log.append("P01", [{"t": 0, "type": "session.begin", "participant": "P01"}], {"screen": "intro"})
    assert first == 1
    lines = log.events_path("P01").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["type"] == "session.begin" and row["received_at"].endswith("Z")
    second = log.append("P01", [{"t": 5, "type": "screen.enter", "screen": "watch1"}], {"screen": "watch1"})
    assert second == 1
    after = log.events_path("P01").read_text(encoding="utf-8").splitlines()
    assert after[0] == lines[0], "the first line is never rewritten"
    assert json.loads(after[1])["screen"] == "watch1"
    assert log.snapshot("P01") == {"screen": "watch1"}
    assert log.events("P01")[1]["t"] == 5
    assert not list((tmp_path / "out").glob(".*.tmp")), "no temporary file survives a replace"
    # An empty batch with a snapshot only refreshes the snapshot.
    assert log.append("P01", [], {"screen": "list"}) == 0
    assert log.snapshot("P01") == {"screen": "list"}
    assert len(log.events("P01")) == 2


def test_concurrent_appends_land_whole_and_the_snapshot_is_one_batchs(tmp_path: Path) -> None:
    # The events route runs in a threadpool, so the debounced POST and the
    # keepalive flush can overlap; every event must still land as its own
    # line and the snapshot must be one whole batch's, never a torn write.
    log = EventLog(tmp_path)
    batches = 16

    def post(index: int) -> None:
        events = [{"t": index, "type": "skeleton.strike", "id": f"s{index}"}] * 3
        log.append("P01", events, {"screen": "shape", "batch": index, "filler": "x" * 4096})

    workers = [threading.Thread(target=post, args=(index,)) for index in range(batches)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    events = log.events("P01")
    assert len(events) == batches * 3
    assert sorted({event["t"] for event in events}) == list(range(batches))
    # Each batch's three lines are contiguous: no batch was interleaved.
    for start in range(0, len(events), 3):
        assert len({event["t"] for event in events[start : start + 3]}) == 1
    snapshot = log.snapshot("P01")
    assert snapshot is not None and snapshot["batch"] in range(batches)
    assert not list(tmp_path.glob(".*.tmp"))


def test_the_gate_reads_the_append_only_file_not_the_snapshot(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    assert log.listing_submitted("P01", "clip-a") is False
    # A snapshot claiming the listing is done opens nothing.
    log.append("P01", [], {"listings": {"clip-a": ["bus"]}, "screen": "shape"})
    assert log.listing_submitted("P01", "clip-a") is False
    log.append("P01", [{"t": 9, "type": "listing.submit", "clip_id": "clip-a", "sounds": ["bus"]}], None)
    assert log.listing_submitted("P01", "clip-a") is True
    assert log.listing_submitted("P01", "clip-b") is False
    assert log.listing_submitted("P02", "clip-a") is False


def test_kept_captions_are_read_from_the_snapshot(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    assert log.kept_caption("P01", "clip-a", "s1") is None
    log.append("P01", [], {"kept": {"clip-a": {"s1": {"caption": "A tram brakes.", "key": "itemized|tram"}}}})
    assert log.kept_caption("P01", "clip-a", "s1") == "A tram brakes."
    assert log.kept_caption("P01", "clip-a", "s2") is None
    log.append("P01", [], {"kept": {"clip-a": {"s1": {"caption": "   "}}}})
    assert log.kept_caption("P01", "clip-a", "s1") is None
    # The same reading over an already loaded snapshot (the follow-up reads once).
    assert (
        kept_caption_in({"kept": {"clip-a": {"s1": {"caption": "A tram brakes."}}}}, "clip-a", "s1")
        == "A tram brakes."
    )
    assert kept_caption_in({"kept": "no"}, "clip-a", "s1") is None
    assert kept_caption_in(None, "clip-a", "s1") is None


@pytest.mark.parametrize("bad", ["", "P 01", "p/01", "../P01", "P" * 65, "P01\n"])
def test_participant_ids_are_refused_before_any_file_is_named(tmp_path: Path, bad: str) -> None:
    log = EventLog(tmp_path)
    with pytest.raises(SessionLogError):
        validate_participant(bad)
    with pytest.raises(SessionLogError):
        log.append(bad, [{"type": "session.begin"}], None)
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(("events-", "snapshot-"))]


def test_a_torn_last_line_does_not_take_the_log_down(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append("P01", [{"type": "session.begin"}, {"type": "listing.submit", "clip_id": "c1"}], None)
    with log.events_path("P01").open("a", encoding="utf-8") as handle:
        handle.write('{"type": "screen.enter", "scr')  # the process died mid-write
    # The whole lines still read, the gate still answers, and the next append
    # starts on its own line.
    assert [event["type"] for event in log.events("P01")] == ["session.begin", "listing.submit"]
    assert log.listing_submitted("P01", "c1")


def test_a_torn_line_that_is_not_the_last_is_corruption(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append("P01", [{"type": "session.begin"}], None)
    path = log.events_path("P01")
    path.write_text('{"type": "session.begin"\n' + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        log.events("P01")


def test_a_malformed_event_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    with pytest.raises(SessionLogError, match=r"events\[1\]"):
        log.append("P01", [{"type": "session.begin"}, {"no": "type"}], {"screen": "intro"})
    assert not log.events_path("P01").exists()
    assert log.snapshot("P01") is None


def test_the_export_carries_the_events_and_the_snapshot(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append("P01", [{"t": 1, "type": "probe.submit", "text": "nothing"}], {"done": True})
    exported = log.export("P01", "demo-session")
    assert exported["schema"] == "dpo.caption-session-log/v1"
    assert exported["session_id"] == "demo-session" and exported["participant"] == "P01"
    assert exported["events"][0]["type"] == "probe.submit" and exported["snapshot"] == {"done": True}
    saved = log.write_followup("P01", {"schema": "dpo.caption-session-followup/v1", "participant": "P01"})
    assert saved == tmp_path / "followup-P01.json"
    written = json.loads(saved.read_text())
    assert written["participant"] == "P01" and written["received_at"].endswith("Z")


def test_the_followup_is_written_once(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    first = {"schema": "dpo.caption-session-followup/v1", "participant": "P01", "check": [{"chosen": "A"}]}
    log.write_followup("P01", first)
    with pytest.raises(FollowupExistsError):
        log.write_followup("P01", {**first, "check": [{"chosen": "B"}]})
    assert json.loads((tmp_path / "followup-P01.json").read_text())["check"] == [{"chosen": "A"}]
