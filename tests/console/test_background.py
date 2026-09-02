"""What the console does with the background writer it shares with the skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from dpo.caption.writer import CaptionRequest, SourceSpec

TRAM = SourceSpec("tram", "TRAM BRAKING", "a tram braking", ("in frame", "once, briefly"))
SIREN = SourceSpec("siren", "DISTANT SIREN", "a distant siren", ("out of frame", "rises and fades"))


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
