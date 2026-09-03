"""§9.5: one row per viewing, and the survey rows that join to it by key."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpo.regen.log import EventLog, RegenLogError, view_id

PARTICIPANT = "p01"
CAPTIONS = [{"index": 0, "start_ms": 0, "end_ms": 2500, "text": "A car passes."}]


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "out", "abc123abc123")


class TestViewings:
    def test_a_viewing_is_one_row(self, log: EventLog) -> None:
        log.record_viewing(
            PARTICIPANT,
            index=0,
            condition="prepared",
            segment="A",
            clip_id="amsterdam_006",
            captions=CAPTIONS,
            started_at="t0",
            ended_at="t1",
        )
        rows = log.viewings(PARTICIPANT)
        assert len(rows) == 1
        assert rows[0]["condition"] == "prepared"
        assert rows[0]["captions"] == CAPTIONS
        assert rows[0]["config_hash"] == "abc123abc123"

    def test_a_re_posted_end_does_not_produce_a_second_observation(self, log: EventLog) -> None:
        for _ in range(3):
            log.record_viewing(
                PARTICIPANT,
                index=0,
                condition="prepared",
                segment="A",
                clip_id="amsterdam_006",
                captions=CAPTIONS,
                started_at="t0",
                ended_at="t1",
            )
        assert len(log.viewings(PARTICIPANT)) == 1

    def test_both_conditions_are_stored_under_one_shape(self, log: EventLog) -> None:
        # §9.6: nothing downstream may tell them apart except `condition`.
        for index, condition in enumerate(("prepared", "regenerated")):
            log.record_viewing(
                PARTICIPANT,
                index=index,
                condition=condition,
                segment="AB"[index],
                clip_id=f"clip_{index}",
                captions=CAPTIONS,
                started_at="t0",
                ended_at="t1",
            )
        first, second = log.viewings(PARTICIPANT)
        assert set(first) == set(second)


class TestResponses:
    def test_responses_join_to_a_viewing_by_key(self, log: EventLog) -> None:
        key = log.record_viewing(
            PARTICIPANT,
            index=0,
            condition="prepared",
            segment="A",
            clip_id="c",
            captions=CAPTIONS,
            started_at="t0",
            ended_at="t1",
        )
        log.record_responses(
            PARTICIPANT,
            page="art",
            key=key,
            responses={"art_1": 5},
            entered_at="t2",
            submitted_at="t3",
            items_digest="d",
            items_provenance="authored",
        )
        assert log.responses(PARTICIPANT)[0]["view_id"] == key == view_id(PARTICIPANT, 0)

    def test_the_item_provenance_travels_with_every_submission(self, log: EventLog) -> None:
        log.record_responses(
            PARTICIPANT,
            page="art",
            key=view_id(PARTICIPANT, 0),
            responses={"art_1": 5},
            entered_at="t2",
            submitted_at="t3",
            items_digest="d",
            items_provenance="placeholder",
        )
        assert log.responses(PARTICIPANT)[0]["items_provenance"] == "placeholder"

    def test_a_second_submission_of_one_page_is_refused(self, log: EventLog) -> None:
        for expected in (True, False):
            wrote = log.record_responses(
                PARTICIPANT,
                page="art",
                key=view_id(PARTICIPANT, 0),
                responses={"art_1": 5},
                entered_at="t2",
                submitted_at="t3",
                items_digest="d",
                items_provenance="authored",
            )
            assert wrote is expected


class TestStream:
    def test_events_are_stamped_on_receipt(self, log: EventLog) -> None:
        log.append(PARTICIPANT, [{"type": "step.entered", "step": "art"}], {"step": "art"})
        event = log.events(PARTICIPANT)[0]
        assert event["config_hash"] == "abc123abc123"
        assert event["received_at"]

    def test_the_snapshot_is_replaced_not_appended(self, log: EventLog) -> None:
        log.append(PARTICIPANT, [], {"step": "art"})
        log.append(PARTICIPANT, [], {"step": "visual"})
        assert log.snapshot(PARTICIPANT) == {"step": "visual"}

    def test_an_event_without_a_type_is_refused(self, log: EventLog) -> None:
        with pytest.raises(RegenLogError, match="string 'type'"):
            log.append(PARTICIPANT, [{"step": "art"}], None)

    def test_a_torn_last_line_still_reads(self, log: EventLog) -> None:
        # A crash mid-write leaves the lines before it whole; reading them
        # keeps the session answering.
        log.append(PARTICIPANT, [{"type": "a"}], None)
        with log.events_path(PARTICIPANT).open("a", encoding="utf-8") as handle:
            handle.write('{"type": "b"')
        assert [row["type"] for row in log.events(PARTICIPANT)] == ["a"]


class TestExport:
    def test_the_export_carries_all_three_files(self, log: EventLog) -> None:
        log.append(PARTICIPANT, [{"type": "a"}], {"step": "art"})
        export = log.export(PARTICIPANT, "street-regen")
        assert set(export) >= {"viewings", "responses", "events", "snapshot", "config_hash"}

    @pytest.mark.parametrize("bad", ["../escape", "p" * 65, "", "p 1"])
    def test_a_participant_id_that_could_name_another_path_is_refused(self, log: EventLog, bad: str) -> None:
        with pytest.raises(RegenLogError):
            log.events_path(bad)
