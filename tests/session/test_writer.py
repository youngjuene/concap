"""Writers: deterministic, in order, cached, and the Gemma prompt says exactly what to mention."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from dpo.session.document import clip_by_id
from dpo.session.skeleton import Settings, orderings
from dpo.session.writer import (
    CAPTION_MAX_CHARS,
    GEMMA_INSTRUCTION,
    GEMMA_VISUAL_INSTRUCTION,
    CachedWriter,
    CaptionRequest,
    GemmaWriter,
    HeadSpec,
    SourceSpec,
    TemplateWriter,
    WriterError,
    audition_requests,
    build_request,
    cut_at_sentence,
    gemma_instruction,
    warm_auditions,
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


class CountingWriter:
    def __init__(self) -> None:
        self.calls = 0

    def write(self, request: CaptionRequest) -> str:
        self.calls += 1
        return f"caption {self.calls} for {request.settings_key}"


def test_the_cache_returns_the_same_string_without_calling_the_inner_writer(tmp_path: Path) -> None:
    inner = CountingWriter()
    cache_path = tmp_path / "captions.json"
    cached = CachedWriter(inner, cache_path)
    request = _request("itemized", (TRAM, SIREN))
    first, hit = cached.write_cached(request)
    assert hit is False and inner.calls == 1
    second, hit = cached.write_cached(request)
    assert hit is True and second == first and inner.calls == 1
    other, _ = cached.write_cached(_request("itemized", (SIREN, TRAM)))
    assert other != first and inner.calls == 2
    # A rerun over the same file costs nothing: the cache is loaded at start.
    fresh_inner = CountingWriter()
    reloaded = CachedWriter(fresh_inner, cache_path)
    assert reloaded.write(request) == first
    assert fresh_inner.calls == 0
    assert not list(tmp_path.glob(".*.tmp"))


class SlowCountingWriter(CountingWriter):
    """Holds every call open until released, so concurrent misses overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Event()

    def write(self, request: CaptionRequest) -> str:
        self.entered.set()
        self.release.wait(5)
        return super().write(request)


def test_concurrent_misses_on_one_key_call_the_inner_writer_once(tmp_path: Path) -> None:
    # FastAPI runs the sync caption route in a threadpool: two Show caption
    # presses for one settings key must not become two generations (spec 7:
    # identical settings return the identical cached caption).
    inner = SlowCountingWriter()
    cached = CachedWriter(inner, tmp_path / "captions.json")
    request = _request("itemized", (TRAM, SIREN))
    results: list[tuple[str, bool]] = []
    workers = [
        threading.Thread(target=lambda: results.append(cached.write_cached(request))) for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    assert inner.entered.wait(5)
    inner.release.set()
    for worker in workers:
        worker.join(5)
    assert inner.calls == 1
    assert {caption for caption, _ in results} == {"caption 1 for itemized|tram,siren"}
    assert sorted(hit for _, hit in results) == [False, True, True, True]
    assert json.loads((tmp_path / "captions.json").read_text(encoding="utf-8")) == {
        "demo_tram_stop/s1/itemized|tram,siren": "caption 1 for itemized|tram,siren"
    }


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
    table = warm_auditions(TemplateWriter(), document)
    assert set(table) == {(c["clip_id"], s["shot_id"]) for c in document["clips"] for s in c["shots"]}
    assert table[("demo_tram_stop", "s1")]["siren"] == "A distant siren rises and fades."
    control = table[("demo_market", "s1")]
    assert "role:backdrop" in control and control["van"] == "A parked van still."


class TestGemmaBudget:
    """The model was measured overrunning the two-line budget it is asked for."""

    class Drafts(FakeAdapter):
        def __init__(self, *drafts: str) -> None:
            super().__init__()
            self.drafts = list(drafts)
            self.calls: list[str] = []

        def generate_stimulus(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            self.calls.append(str(messages[0]["content"]))
            return self.drafts.pop(0)

    LONG = (
        "Distant siren rises and fades. Tram brakes once. Chatter is steady. Footsteps come and go."
        " Pigeons are steady."
    )
    SHORT = "Siren rises and fades. Tram brakes once. Chatter, footsteps, and pigeons steady."

    def test_a_caption_within_budget_costs_one_call(self, tmp_path: Path) -> None:
        adapter = self.Drafts(self.SHORT)
        assert GemmaWriter(adapter).write(_request("scene", media_path=tmp_path)) == self.SHORT
        assert len(adapter.calls) == 1

    def test_an_overrun_is_retried_once_naming_the_overrun(self, tmp_path: Path) -> None:
        assert len(self.LONG) > CAPTION_MAX_CHARS >= len(self.SHORT)
        adapter = self.Drafts(self.LONG, self.SHORT)
        assert GemmaWriter(adapter).write(_request("scene", media_path=tmp_path)) == self.SHORT
        assert len(adapter.calls) == 2
        assert f"Too long: {len(self.LONG)} characters" in adapter.calls[1] or (
            f"ran to {len(self.LONG)} characters" in adapter.calls[1]
        )
        assert adapter.calls[1].startswith(adapter.calls[0])
        # A scene request has no entries to name, so the retry only asks for less.
        assert "separated by commas" not in adapter.calls[1]

    def test_the_retry_names_every_entry_in_order(self, tmp_path: Path) -> None:
        adapter = self.Drafts(self.LONG, self.SHORT)
        request = _request("itemized", (TRAM, SIREN, STEPS), media_path=tmp_path)
        assert GemmaWriter(adapter).write(request) == self.SHORT
        assert (
            "lists a tram braking, a distant siren, footsteps, in that order, separated by commas"
            in adapter.calls[1]
        )

    def test_four_or_more_entries_carry_the_budget_hint_in_the_first_draft(self, tmp_path: Path) -> None:
        few = FakeAdapter()
        GemmaWriter(few).write(_request("itemized", (TRAM, SIREN, STEPS), media_path=tmp_path))
        assert "four or more entries" not in few.messages[0]["content"]
        many = FakeAdapter()
        GemmaWriter(many).write(_request("itemized", (TRAM, SIREN, STEPS, VAN), media_path=tmp_path))
        assert f"stay under {CAPTION_MAX_CHARS} characters" in many.messages[0]["content"]

    def test_a_second_overrun_falls_back_to_the_template_for_the_same_list(self, tmp_path: Path) -> None:
        # The template keeps every entry in order; a sentence cut would not.
        adapter = self.Drafts(self.LONG, self.LONG)
        request = _request("itemized", (TRAM, SIREN, STEPS), media_path=tmp_path)
        caption = GemmaWriter(adapter).write(request)
        assert caption == TemplateWriter().write(request)
        assert len(caption) <= CAPTION_MAX_CHARS
        for source in (TRAM, SIREN, STEPS):
            assert source.prose in caption.lower()
        # A longer retry never replaces the draft it was meant to shorten.
        adapter = self.Drafts(self.LONG, self.LONG + " And more.")
        assert GemmaWriter(adapter).write(request) == caption

    def test_when_even_the_template_overruns_the_caption_ends_at_a_whole_sentence(
        self, tmp_path: Path
    ) -> None:
        class Overrunning:
            def write(self, request: CaptionRequest) -> str:
                return "x" * (CAPTION_MAX_CHARS + 1)

        adapter = self.Drafts(self.LONG, self.LONG)
        caption = GemmaWriter(adapter, fallback=Overrunning()).write(_request("scene", media_path=tmp_path))
        assert (
            caption
            == "Distant siren rises and fades. Tram brakes once. Chatter is steady. Footsteps come and go."
        )
        assert len(caption) <= CAPTION_MAX_CHARS and caption.endswith(".")

    def test_the_cut_leaves_a_caption_with_no_sentence_end_alone(self) -> None:
        run_on = "x" * (CAPTION_MAX_CHARS + 10)
        assert cut_at_sentence(run_on, CAPTION_MAX_CHARS) == run_on
        assert cut_at_sentence("One. Two.", 6) == "One."


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
