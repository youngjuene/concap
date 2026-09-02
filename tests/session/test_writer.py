"""Writers: deterministic, in order, cached, and the Gemma prompt says exactly what to mention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpo.session.document import clip_by_id
from dpo.session.skeleton import Settings, orderings
from dpo.session.writer import (
    CAPTION_MAX_CHARS,
    GEMMA_INSTRUCTION,
    GEMMA_VISUAL_INSTRUCTION,
    CaptionRequest,
    GemmaWriter,
    HeadSpec,
    SourceSpec,
    TemplateWriter,
    WriterError,
    audition_requests,
    build_request,
    gemma_instruction,
)

FIXTURE = Path(__file__).parent / "fixtures" / "session.json"
TRAM = SourceSpec("tram", "TRAM BRAKING", "a tram braking", ("in frame", "once, briefly"))
SIREN = SourceSpec("siren", "DISTANT SIREN", "a distant siren", ("out of frame", "rises and fades"))
STEPS = SourceSpec("footsteps", "FOOTSTEPS", "footsteps", ("at the edge of the frame", "comes and goes"))
VAN = SourceSpec("van", "PARKED VAN", "a parked van", ("a third of the frame", "still"))
CYCLIST = SourceSpec("cyclist", "CYCLIST", "a cyclist", ("small", "crosses left to right"))


def _request(
    level: str, sources: tuple[SourceSpec, ...] = (), heads: tuple[HeadSpec, ...] = (), **extra: Any
) -> CaptionRequest:
    values: dict[str, Any] = {
        "clip_id": "demo_tram_stop",
        "shot_id": "s1",
        "task": "shaped",
        "level": level,
        "settings_key": f"{level}|" + ",".join(s.id for s in sources),
        "sources": sources,
        "heads": heads,
        "scene_prose": "A tram stop on a wide street.",
        "atmosphere_prose": "Steady, with one rise.",
        "media_path": None,
    }
    values.update(extra)
    return CaptionRequest(**values)


def _document() -> dict[str, Any]:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


class TestTemplateWriter:
    def test_itemized_is_deterministic_sentence_prose_in_the_given_order(self) -> None:
        writer = TemplateWriter()
        request = _request("itemized", (TRAM, SIREN, STEPS))
        caption = writer.write(request)
        assert (
            caption
            == "A tram braking once, briefly, a distant siren rises and fades, and footsteps comes and goes."
        )
        assert writer.write(request) == caption
        lowered = caption.lower()
        assert lowered.index(TRAM.prose) < lowered.index(SIREN.prose) < lowered.index(STEPS.prose)
        reversed_request = _request("itemized", (STEPS, SIREN, TRAM))
        flipped = writer.write(reversed_request)
        flipped = flipped.lower()
        assert flipped.index(STEPS.prose) < flipped.index(SIREN.prose) < flipped.index(TRAM.prose)

    def test_one_and_two_sources_read_as_sentences(self) -> None:
        writer = TemplateWriter()
        assert writer.write(_request("itemized", (SIREN,))) == "A distant siren rises and fades."
        assert (
            writer.write(_request("itemized", (TRAM, STEPS)))
            == "A tram braking once, briefly and footsteps comes and goes."
        )

    def test_a_sentence_that_would_clip_drops_the_phrases_and_keeps_the_order(self) -> None:
        # Five phrased clauses overflow the two-line box; the sources stay, in order.
        sources = (TRAM, SIREN, STEPS, VAN, CYCLIST)
        caption = TemplateWriter().write(_request("itemized", sources))
        assert len(caption) <= CAPTION_MAX_CHARS
        lowered = caption.lower()
        positions = [lowered.index(source.prose) for source in sources]
        assert positions == sorted(positions)
        assert "once, briefly" not in caption and "rises and fades" not in caption
        # Under the budget the phrases stay.
        assert "once, briefly" in TemplateWriter().write(_request("itemized", (TRAM, SIREN)))

    def test_control_uses_the_motion_phrase(self) -> None:
        caption = TemplateWriter().write(_request("itemized", (VAN, CYCLIST), task="control"))
        assert caption == "A parked van still and a cyclist crosses left to right."

    def test_grouped_writes_one_clause_per_head_in_order(self) -> None:
        heads = (
            HeadSpec("stands_out", "Stands out", (TRAM, SIREN)),
            HeadSpec("underneath", "Underneath", (STEPS,)),
        )
        caption = TemplateWriter().write(_request("grouped", (TRAM, SIREN, STEPS), heads))
        assert caption == "Stands out, a tram braking and a distant siren; underneath, footsteps."

    def test_a_grouped_sentence_that_would_clip_loses_its_leads_not_its_members(self) -> None:
        heads = (
            HeadSpec("underneath", "Underneath", (STEPS, TRAM)),
            HeadSpec("only_here", "Only here", (SIREN, VAN)),
            HeadSpec("stands_out", "Stands out", (CYCLIST,)),
        )
        caption = TemplateWriter().write(_request("grouped", (STEPS, TRAM, SIREN, VAN, CYCLIST), heads))
        assert len(caption) <= CAPTION_MAX_CHARS
        assert caption == "Footsteps and a tram braking; a distant siren and a parked van; a cyclist."

    def test_scene_and_atmospheric_are_the_rows_prose(self) -> None:
        writer = TemplateWriter()
        assert writer.write(_request("scene")) == "A tram stop on a wide street."
        assert writer.write(_request("atmospheric")) == "Steady, with one rise."


class FakeAdapter:
    """Records what the writer sends and returns a fixed completion; no torch."""

    def __init__(self, completion: str = "  A tram brakes as a siren fades.  ") -> None:
        self.completion = completion
        self.messages: list[dict[str, Any]] = []
        self.kwargs: dict[str, Any] = {}

    def generate_stimulus(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> str:
        self.messages = messages
        self.kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        }
        return self.completion


class TestGemmaWriter:
    def test_the_prompt_names_the_sources_in_order_over_the_shots_audio(self, tmp_path: Path) -> None:
        adapter = FakeAdapter()
        audio = tmp_path / "shot.wav"
        request = _request("itemized", (TRAM, SIREN, STEPS), media_path=audio)
        caption = GemmaWriter(adapter).write(request)
        assert caption == "A tram brakes as a siren fades."
        system, user = adapter.messages
        assert system["role"] == "system" and system["content"].startswith(GEMMA_INSTRUCTION)
        instruction = system["content"]
        assert "ONLY these sound sources, in THIS order" in instruction
        assert instruction.index(TRAM.prose) < instruction.index(SIREN.prose) < instruction.index(STEPS.prose)
        assert "once, briefly" in instruction
        assert user["role"] == "user" and user["content"] == [{"type": "audio", "audio": str(audio)}]
        assert adapter.kwargs == {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 60, "seed": 0}

    def test_each_level_has_its_own_rule(self, tmp_path: Path) -> None:
        heads = (
            HeadSpec("stands_out", "Stands out", (TRAM, SIREN)),
            HeadSpec("underneath", "Underneath", (STEPS,)),
        )
        grouped = gemma_instruction(_request("grouped", (TRAM, SIREN, STEPS), heads, media_path=tmp_path))
        assert "one clause per group" in grouped
        assert grouped.index("Stands out: a tram braking, a distant siren") < grouped.index(
            "Underneath: footsteps"
        )
        scene = gemma_instruction(_request("scene", media_path=tmp_path))
        assert "names the scene as a whole" in scene and "A tram stop on a wide street." in scene
        atmospheric = gemma_instruction(_request("atmospheric", media_path=tmp_path))
        assert "no nouns" in atmospheric and "Steady, with one rise." in atmospheric
        assert "a tram braking" not in atmospheric

    def test_failures_become_writer_errors(self, tmp_path: Path) -> None:
        with pytest.raises(WriterError, match="media"):
            GemmaWriter(FakeAdapter()).write(_request("scene"))
        with pytest.raises(WriterError, match="empty"):
            GemmaWriter(FakeAdapter("   ")).write(_request("scene", media_path=tmp_path))

        class Broken(FakeAdapter):
            def generate_stimulus(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
                raise RuntimeError("CUDA out of memory")

        with pytest.raises(WriterError, match="CUDA out of memory"):
            GemmaWriter(Broken()).write(_request("scene", media_path=tmp_path))


def test_build_request_flattens_grouped_members_in_head_order() -> None:
    document = _document()
    clip = clip_by_id(document, "demo_tram_stop")
    assert clip is not None
    shot = clip["shots"][0]
    settings = Settings("grouped", ("tram", "siren", "footsteps", "chatter"), ("stands_out", "underneath"))
    request = build_request(document, clip, shot, settings, None)
    assert [head.text for head in request.heads] == ["Stands out", "Underneath"]
    assert [[m.id for m in head.members] for head in request.heads] == [
        ["tram", "siren"],
        ["footsteps", "chatter"],
    ]
    assert [s.id for s in request.sources] == ["tram", "siren", "footsteps", "chatter"]
    assert request.settings_key == "grouped|stands_out,underneath|chatter,footsteps,siren,tram"
    itemized = orderings(shot["sources"])[0]["order"]
    request = build_request(
        document, clip, shot, Settings("itemized", tuple(itemized), tuple(itemized)), None
    )
    assert [s.id for s in request.sources] == itemized and request.heads == ()


def test_auditions_cover_every_source_and_every_present_head() -> None:
    document = _document()
    clip = clip_by_id(document, "demo_tram_stop")
    assert clip is not None
    requests = audition_requests(document, clip, clip["shots"][0], None)
    assert set(requests) == {
        "tram",
        "siren",
        "footsteps",
        "chatter",
        "pigeons",
        "role:underneath",
        "role:stands_out",
        "role:only_here",
    }
    assert requests["siren"].level == "itemized" and [s.id for s in requests["siren"].sources] == ["siren"]
    head = requests["role:underneath"]
    assert head.level == "grouped" and {m.id for m in head.heads[0].members} == {"footsteps", "chatter"}
    assert TemplateWriter().write(requests["siren"]) == "A distant siren rises and fades."


class TestGemmaControl:
    """Spec 4.6: the control caption is a plain visual description, so the writer looks, not listens."""

    def test_a_control_request_is_asked_for_what_is_seen_over_one_frame(self, tmp_path: Path) -> None:
        adapter = FakeAdapter("A parked van fills a third of the frame as a cyclist crosses.")
        still = tmp_path / "shot.jpg"
        request = _request("itemized", (VAN, CYCLIST), task="control", media_path=still)
        GemmaWriter(adapter).write(request)
        system, user = adapter.messages
        assert system["content"].startswith(GEMMA_VISUAL_INSTRUCTION)
        assert "heard" not in system["content"] and "visible things" in system["content"]
        assert "a parked van — a third of the frame, still" in system["content"]
        assert user["content"] == [{"type": "image", "image": str(still)}]

    def test_a_shaped_request_still_listens(self, tmp_path: Path) -> None:
        adapter = FakeAdapter()
        GemmaWriter(adapter).write(_request("itemized", (TRAM,), media_path=tmp_path / "shot.wav"))
        _, user = adapter.messages
        assert [part["type"] for part in user["content"]] == ["audio"]

    def test_the_control_scene_and_atmosphere_rules_name_no_sound(self, tmp_path: Path) -> None:
        for level in ("scene", "atmospheric"):
            adapter = FakeAdapter("A shopping street with a tram crossing.")
            GemmaWriter(adapter).write(_request(level, task="control", media_path=tmp_path / "s.jpg"))
            assert "sound" not in adapter.messages[0]["content"]


class TestExcludedInRequests:
    def test_v1_carries_the_muted_sources_at_the_levels_with_rows(self) -> None:
        document = _document()
        clip, shot = document["clips"][0], document["clips"][0]["shots"][0]
        request = build_request(document, clip, shot, Settings("itemized", ("tram",), ("tram",)), None)
        assert {s.id for s in request.excluded} == {s["id"] for s in shot["sources"]} - {"tram"}

    def test_v1_excludes_nothing_where_admission_has_no_surface(self) -> None:
        document = _document()
        clip, shot = document["clips"][0], document["clips"][0]["shots"][0]
        for level in ("scene", "atmospheric"):
            request = build_request(document, clip, shot, Settings(level, (), ()), None)
            assert request.excluded == ()


class TestSceneAnchoring:
    def test_the_scene_request_carries_the_shots_sources(self) -> None:
        document = _document()
        clip, shot = document["clips"][0], document["clips"][0]["shots"][0]
        request = build_request(document, clip, shot, Settings("scene", (), ()), None)
        assert [s.id for s in request.sources] == [s["id"] for s in shot["sources"]]

    def test_the_scene_instruction_lists_the_sounds_and_forbids_adding_one(self) -> None:
        request = _request("scene", (TRAM, SIREN))
        text = gemma_instruction(request)
        assert "The scene: A tram stop on a wide street. The sounds in it:" in text
        assert "a tram braking, a distant siren" in text
        assert "add no sound that is not listed" in text
        assert "do not enumerate" in text
