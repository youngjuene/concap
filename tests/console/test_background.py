"""Captions written off the request path: auditions ahead of need, neighbours ahead of the press."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from dpo.caption.background import BackgroundWriter
from dpo.caption.writer import CachedWriter, CaptionRequest, SourceSpec, TemplateWriter, WriterError

TRAM = SourceSpec("tram", "TRAM BRAKING", "a tram braking", ("in frame", "once, briefly"))
SIREN = SourceSpec("siren", "DISTANT SIREN", "a distant siren", ("out of frame", "rises and fades"))
STEPS = SourceSpec("footsteps", "FOOTSTEPS", "footsteps", ("at the edge of the frame", "comes and goes"))


def _request(*sources: SourceSpec, shot: str = "s1") -> CaptionRequest:
    return CaptionRequest(
        clip_id="clip",
        shot_id=shot,
        task="shaped",
        level="itemized",
        settings_key="itemized|" + ",".join(s.id for s in sources),
        sources=sources,
        heads=(),
        scene_prose="",
        atmosphere_prose="",
        media_path=None,
    )


class Slow:
    """A writer that takes a beat per caption and counts what it wrote, in order."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.order: list[str] = []
        self.lock = threading.Lock()

    def write(self, request: CaptionRequest) -> str:
        time.sleep(self.delay)
        with self.lock:
            self.order.append(request.settings_key)
        return "caption for " + request.settings_key


class Failing:
    def write(self, request: CaptionRequest) -> str:
        raise WriterError("no model")


class TestQueueing:
    def test_warm_work_is_written_and_wait_for_returns_when_it_is(self, tmp_path: Path) -> None:
        cached = CachedWriter(Slow(), tmp_path / "captions.json")
        background = BackgroundWriter(cached)
        try:
            background.warm([_request(TRAM), _request(SIREN)], key="s1")
            assert background.wait_for("s1", timeout=5)
            assert cached.lookup(_request(TRAM)) == "caption for itemized|tram"
            assert cached.lookup(_request(SIREN)) == "caption for itemized|siren"
        finally:
            background.close()

    def test_urgent_work_goes_ahead_of_the_warm_up(self, tmp_path: Path) -> None:
        inner = Slow(delay=0.03)
        cached = CachedWriter(inner, tmp_path / "captions.json")
        background = BackgroundWriter(cached)
        try:
            # Hold the worker on one item, queue a warm one behind it, then an
            # urgent one: the urgent one must be written before the warm one.
            background.warm([_request(TRAM)], key="a")
            time.sleep(0.005)
            background.warm([_request(STEPS)], key="b")
            background.urgent([_request(SIREN)], key="c")
            assert background.wait_for("b", timeout=5)
            assert inner.order.index("itemized|siren") < inner.order.index("itemized|footsteps")
        finally:
            background.close()

    def test_a_key_with_nothing_left_to_write_is_ready_at_once(self, tmp_path: Path) -> None:
        cached = CachedWriter(TemplateWriter(), tmp_path / "captions.json")
        cached.write(_request(TRAM))
        background = BackgroundWriter(cached)
        try:
            background.warm([_request(TRAM)], key="s1")
            assert background.wait_for("s1", timeout=1)
        finally:
            background.close()

    def test_disabled_writes_synchronously(self, tmp_path: Path) -> None:
        cached = CachedWriter(TemplateWriter(), tmp_path / "captions.json")
        background = BackgroundWriter(cached, enabled=False)
        background.warm([_request(TRAM)], key="s1")
        assert cached.lookup(_request(TRAM)) is not None
        assert background.wait_for("s1", timeout=0) is True


class TestYielding:
    def test_the_worker_does_not_start_a_caption_while_a_participant_is_waiting(self, tmp_path: Path) -> None:
        inner = Slow(delay=0.02)
        cached = CachedWriter(inner, tmp_path / "captions.json")
        background = BackgroundWriter(cached)
        try:
            with background.foreground():
                background.warm([_request(TRAM), _request(SIREN)], key="s1")
                time.sleep(0.1)
                # Nothing was written while the participant held the floor.
                assert inner.order == []
            assert background.wait_for("s1", timeout=5)
            assert len(inner.order) == 2
        finally:
            background.close()


class Flaky:
    """A writer whose first request fails outside the writer's own error type."""

    def __init__(self) -> None:
        self.calls = 0

    def write(self, request: CaptionRequest) -> str:
        self.calls += 1
        if self.calls == 1:
            raise OSError("no space left on device")
        return "caption for " + request.settings_key


class TestResilience:
    def test_the_worker_survives_an_error_that_is_not_the_writers_own(self, tmp_path: Path) -> None:
        cached = CachedWriter(Flaky(), tmp_path / "captions.json")
        background = BackgroundWriter(cached)
        try:
            background.warm([_request(TRAM), _request(SIREN)], key="s1")
            # The key still comes ready, the failure is on record, and the
            # thread that would otherwise have died wrote the next caption.
            assert background.wait_for("s1", timeout=5)
            assert background.failures and background.failures[0][1].startswith("OSError")
            assert cached.lookup(_request(SIREN)) is not None
            assert background._thread.is_alive()
        finally:
            background.close()

    def test_urgent_moves_queued_warm_work_ahead_instead_of_queuing_it_twice(self, tmp_path: Path) -> None:
        inner = Slow(delay=0.05)
        cached = CachedWriter(inner, tmp_path / "captions.json")
        background = BackgroundWriter(cached)
        try:
            with background.foreground():  # hold the worker while the queues fill
                for shot in ("s1", "s2", "s3"):
                    background.warm([_request(TRAM, shot=shot), _request(SIREN, shot=shot)], key=shot)
                background.urgent([_request(TRAM, shot="s3"), _request(SIREN, shot="s3")], key="s3")
                assert background._pending["s3"] == 2
            # s3 is written first, and its key is ready as soon as its own two
            # are — before the sweep has reached the last shot it was queued
            # behind, which is two captions further on.
            assert background.wait_for("s3", timeout=0.4)
            assert cached.lookup(_request(TRAM, shot="s3")) is not None
            assert cached.lookup(_request(SIREN, shot="s3")) is not None
            assert cached.lookup(_request(SIREN, shot="s2")) is None
            assert len(inner.order) <= 3
            assert background.wait_for("s2", timeout=5)
        finally:
            background.close()

    def test_a_finished_key_takes_new_work_with_a_fresh_event(self, tmp_path: Path) -> None:
        cached = CachedWriter(Slow(delay=0.05), tmp_path / "captions.json")
        background = BackgroundWriter(cached)
        try:
            background.warm([_request(TRAM)], key="s1")
            assert background.wait_for("s1", timeout=5)
            background.warm([_request(SIREN)], key="s1")
            assert background.wait_for("s1", timeout=0.001) is False
            assert background.wait_for("s1", timeout=5)
            assert cached.lookup(_request(SIREN)) is not None
        finally:
            background.close()


class TestProvenance:
    def test_wrote_tells_a_prefetched_hit_from_a_revisit(self, tmp_path: Path) -> None:
        cached = CachedWriter(TemplateWriter(), tmp_path / "captions.json")
        background = BackgroundWriter(cached, enabled=False)
        background.urgent([_request(TRAM)])
        assert background.wrote(_request(TRAM)) is True
        # A caption the participant asked for first is not the prefetcher's.
        cached.write(_request(SIREN))
        background.urgent([_request(SIREN)])
        assert background.wrote(_request(SIREN)) is False

    def test_a_failure_in_the_background_is_recorded_not_raised(self, tmp_path: Path) -> None:
        cached = CachedWriter(Failing(), tmp_path / "captions.json")
        background = BackgroundWriter(cached, enabled=False)
        background.warm([_request(TRAM)], key="s1")
        assert background.failures and background.failures[0][1] == "no model"


# ---- through the apps ---------------------------------------------------------


class TestConsoleApp:
    def test_a_caption_prefetches_its_neighbours(self, document: dict[str, Any], tmp_path: Path) -> None:
        from dpo.console.app import build_app
        from dpo.console.requests import ConsoleTemplateWriter

        app = build_app(document, tmp_path / "m", tmp_path / "o", ConsoleTemplateWriter(), prefetch=False)
        client = TestClient(app)
        clip = document["clips"][0]
        shot = clip["shots"][0]
        answer = client.get(f"/api/shot/{clip['clip_id']}/{shot['shot_id']}?participant=P01").json()
        ids = [source["id"] for source in answer["sources"]]
        first = client.post(
            "/api/caption",
            json={
                "participant": "P01",
                "clip_id": clip["clip_id"],
                "shot_id": shot["shot_id"],
                "settings": {"grain": "itemized", "admitted": ids, "regime": 0},
            },
        ).json()
        assert first["cached"] is False and first["prefetched"] is False
        # One mute away was written while the first caption was "read".
        muted = client.post(
            "/api/caption",
            json={
                "participant": "P01",
                "clip_id": clip["clip_id"],
                "shot_id": shot["shot_id"],
                "settings": {"grain": "itemized", "admitted": ids[1:], "regime": 0},
            },
        ).json()
        assert muted["cached"] is True and muted["prefetched"] is True
        # Asking again for the first is a hit, and not a prefetched one.
        again = client.post(
            "/api/caption",
            json={
                "participant": "P01",
                "clip_id": clip["clip_id"],
                "shot_id": shot["shot_id"],
                "settings": {"grain": "itemized", "admitted": ids, "regime": 0},
            },
        ).json()
        assert again["cached"] is True and again["prefetched"] is False


class TestConsoleNeighbours:
    def test_one_gesture_away_and_nothing_further(self, document: dict[str, Any]) -> None:
        from dpo.console.document import configuration_of, field_of
        from dpo.console.requests import Settings, neighbours

        field = field_of(document["clips"][0]["shots"][0], configuration_of(document))
        ids = tuple(source.id for source in field.sources)
        near = neighbours(field, Settings("grouped", ids, 0))
        keys = {candidate.key for candidate in near}
        assert len(keys) == len(near)  # no duplicates
        assert Settings("grouped", ids, 0).key not in keys  # not itself
        # Every single mute is there, at the same grain.
        for source_id in ids:
            rest = tuple(item for item in ids if item != source_id)
            assert any(c.grain == "grouped" and set(c.admitted) == set(rest) for c in near)
        # Both grains either side.
        assert any(c.grain == "itemized" for c in near)
        assert any(c.grain == "scene" for c in near)

    def test_the_unnamed_grain_has_one_way_out(self, document: dict[str, Any]) -> None:
        from dpo.console.document import configuration_of, field_of
        from dpo.console.requests import Settings, neighbours

        field = field_of(document["clips"][0]["shots"][0], configuration_of(document))
        near = neighbours(field, Settings("atmospheric", (), 0))
        assert [c.grain for c in near] == ["scene"]
        assert set(near[0].admitted) == {source.id for source in field.sources}
