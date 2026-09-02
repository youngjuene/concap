"""Settings become a caption request: §8, generation.

The prompt assembles the ordered admitted sources, each carrying a visibility
band on ``v_g`` and a temporal register from ``p_g``, followed by the grain
instruction. The generating model is told to honour the order and is never told
what produced it — not the weights, not α, not that a person moved anything.
Visible sources are described as seen producing their sound; non-visible ones
as arriving from outside the frame.

Grain is a granularity of description over the same admitted, ordered set, not
a different set. This is where the console departs most sharply from the
skeleton instrument, which folds the list into role heads and then into a
single authored row. Here only the last step removes the surface: "at the
atmospheric step nothing is named, so the admitted set and α have no surface to
act on" (§7). Scene-level still ranks and still filters; it describes the
moment more broadly.

The settings key is the cache key of §8 — shot, regime, admitted set, grain —
and it is also the identity a commit is recorded under. At the atmospheric
grain the admitted set and the regime drop out of the key, because nothing
distinguishes two atmospheric captions that differ only in them, and a cache
that pretended otherwise would generate one sentence many times and log
distinctions that do not exist.

Everything that turns a measured number into words goes through
:class:`RequestBuilder`, which holds the configuration. Bands and registers are
per-corpus calibrations (§10), so no module-level function may format one: a
phrase that did not come from the configuration in force would be a phrase
outside the hash.

Section numbers cite ``spec-system.md`` except §4.1, which is ``spec-uiux.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dpo.caption.writer import (
    CAPTION_MAX_CHARS,
    CaptionRequest,
    SourceSpec,
    as_sentence,
    join_clauses,
)
from dpo.console.config import Configuration
from dpo.console.quantities import GRAINS, UNNAMED_GRAIN, Field


class SettingsError(ValueError):
    """Settings name something this shot cannot produce."""


@dataclass(frozen=True)
class Settings:
    """One point in the reachable space: a grain, an admitted set, a regime."""

    grain: str
    admitted: tuple[str, ...]
    regime: int

    @property
    def key(self) -> str:
        if self.grain == UNNAMED_GRAIN:
            return UNNAMED_GRAIN
        return f"{self.grain}|{self.regime}|{','.join(sorted(self.admitted))}"

    def as_json(self) -> dict[str, object]:
        return {"grain": self.grain, "admitted": list(self.admitted), "regime": self.regime}


def validate_settings(field: Field, raw: Mapping[str, object]) -> Settings:
    """Refuse settings the console could not have produced.

    The browser reaches a regime only by clicking a segment, so an index past
    the ones this admitted set yields is a forged request, not a choice.
    """
    grain = raw.get("grain")
    if not isinstance(grain, str) or grain not in GRAINS:
        raise SettingsError(f"grain must be one of {list(GRAINS)}")
    known = {source.id for source in field.sources}
    admitted_raw = raw.get("admitted")
    if not isinstance(admitted_raw, list) or not all(isinstance(item, str) for item in admitted_raw):
        raise SettingsError("admitted must be a list of source ids")
    admitted = tuple(str(item) for item in admitted_raw)
    if len(set(admitted)) != len(admitted):
        raise SettingsError("admitted repeats a source")
    for source_id in admitted:
        if source_id not in known:
            raise SettingsError(f"admitted names unknown source {source_id!r}")
    if grain == UNNAMED_GRAIN:
        # Nothing is named, so nothing about the set or the regime reaches the
        # caption. Normalized away so the key cannot record a phantom choice.
        return Settings(grain=grain, admitted=(), regime=0)
    if not admitted:
        raise SettingsError(f"admitted must not be empty at the {grain} grain")
    regime = raw.get("regime")
    if isinstance(regime, bool) or not isinstance(regime, int):
        raise SettingsError("regime must be an integer index")
    reachable = field.regimes(admitted)
    if not 0 <= regime < len(reachable):
        raise SettingsError(f"regime {regime} is not one of the {len(reachable)} this set reaches")
    return Settings(grain=grain, admitted=admitted, regime=regime)


# ---- the grain instruction --------------------------------------------------


CONSOLE_PREAMBLE = (
    "You write the caption for one shot of street footage from its sound. Write one or two short"
    f" sentences of plain sentence prose, at most {CAPTION_MAX_CHARS} characters in total, in the"
    " register of a subtitle for the deaf and hard of hearing: no brackets, no lists, no numerals,"
    " no headings, and nothing about yourself. Describe only what is heard.\n"
)

# §8: a source that can be seen is described as seen producing its sound, and
# one that cannot as arriving from outside the frame. The rule is stated to the
# model with the out-of-frame phrase quoted from the configuration, so the two
# cannot drift apart when a corpus recalibrates its bands.
CONSOLE_FRAMING = (
    " Each entry carries two notes: where its source sits in the frame, and how it sits in time."
    " A source noted {absent!r} is heard but cannot be seen, so write it as arriving from outside"
    " the frame; any other source is seen producing its sound. Write the notes into prose, never"
    " copied word for word.\n"
)

GRAIN_RULES = {
    "itemized": ("Mention ONLY these sources, in THIS order, and no other source: {sources}. Name each one."),
    "grouped": (
        "Mention ONLY these sources, in THIS order, and no other source: {sources}."
        " Gather them into a few broader phrases rather than naming each one separately,"
        " keeping the order they are given in."
    ),
    "scene": (
        "These are the sources present, in order of importance: {sources}."
        " Write one sentence about the moment as a whole that is true of them, led by the first."
        " Do not enumerate them."
    ),
    UNNAMED_GRAIN: (
        "Describe only the temporal character of the moment as a whole — how the sound moves,"
        " steadies, rises or thins — with no noun naming any source or thing."
    ),
}


@dataclass(frozen=True)
class RequestBuilder:
    """Turns settings into a request, under one configuration's bands.

    Also the instrument's instruction and its template writer, because all
    three need the same calibrated phrases and none of them may invent one.
    """

    configuration: Configuration

    @property
    def absent_phrase(self) -> str:
        """The band phrase for a source with no visible area at all.

        Read from the configuration rather than written down here, so the
        instruction quotes the phrase the entries actually carry.
        """
        return self.configuration.visibility_phrase(0.0)

    def specs(self, field: Field, settings: Settings) -> tuple[SourceSpec, ...]:
        """The admitted sources in the regime's order, each banded and registered."""
        if settings.grain == UNNAMED_GRAIN:
            return ()
        order = field.regimes(settings.admitted)[settings.regime]["order"]
        by_id = {source.id: source for source in field.sources}
        return tuple(
            SourceSpec(
                id=source_id,
                token=by_id[source_id].prose.upper(),
                prose=by_id[source_id].prose,
                phrases=(
                    self.configuration.visibility_phrase(field.visibility[source_id]),
                    self.configuration.register_phrase(by_id[source_id].presence),
                ),
            )
            for source_id in order
        )

    def build(
        self,
        clip_id: str,
        shot: Mapping[str, object],
        field: Field,
        settings: Settings,
        media_path: Path | None = None,
    ) -> CaptionRequest:
        return CaptionRequest(
            clip_id=clip_id,
            shot_id=str(shot["shot_id"]),
            # One task: the console captions sound. The shared writer keeps a
            # visual branch for the other instrument's control clips.
            task="shaped",
            level=settings.grain,
            settings_key=settings.key,
            sources=self.specs(field, settings),
            heads=(),
            scene_prose="",
            atmosphere_prose="",
            media_path=media_path,
            excluded=self.excluded(field, settings),
        )

    def excluded(self, field: Field, settings: Settings) -> tuple[SourceSpec, ...]:
        """The shot's sources the admitted set left out; none at the unnamed grain."""
        if settings.grain == UNNAMED_GRAIN:
            return ()
        admitted = set(settings.admitted)
        return tuple(
            SourceSpec(
                id=source.id,
                token=source.prose.upper(),
                prose=source.prose,
                phrases=(
                    self.configuration.visibility_phrase(field.visibility[source.id]),
                    self.configuration.register_phrase(source.presence),
                ),
            )
            for source in field.sources
            if source.id not in admitted
        )

    def instruction(self, request: CaptionRequest) -> str:
        """The exact system text for a request, assembled from the constants above."""
        rule = GRAIN_RULES[request.level]
        if request.level == UNNAMED_GRAIN:
            return CONSOLE_PREAMBLE + rule
        listed = "; ".join(f"{spec.prose} — {spec.phrases[0]}, {spec.phrases[1]}" for spec in request.sources)
        return (
            CONSOLE_PREAMBLE + CONSOLE_FRAMING.format(absent=self.absent_phrase) + rule.format(sources=listed)
        )


class ConsoleTemplateWriter:
    """Deterministic prose for the four grains; no model, no randomness.

    What the instrument is built, tested and demonstrated on. The grains differ
    in how much of the list survives into the sentence, which is the same
    distinction the model is asked for, so a template caption and a Gemma
    caption move in the same direction when the grain changes.
    """

    identity = "template"

    def write(self, request: CaptionRequest) -> str:
        if request.level == UNNAMED_GRAIN:
            return "Steady, with the sound moving through."
        names = [spec.prose for spec in request.sources]
        if not names:
            raise SettingsError("a named grain needs at least one admitted source")
        if request.level == "scene":
            return as_sentence(f"{names[0]} over a street that carries the rest")
        if request.level == "grouped":
            lead, *rest = names
            if not rest:
                return as_sentence(lead)
            grouped = as_sentence(f"{lead}, with {join_clauses(rest)} under it")
            return grouped if len(grouped) <= CAPTION_MAX_CHARS else as_sentence(join_clauses(names))
        phrased = as_sentence(join_clauses([f"{spec.prose} {spec.phrases[1]}" for spec in request.sources]))
        return phrased if len(phrased) <= CAPTION_MAX_CHARS else as_sentence(join_clauses(names))


def audition_settings(source_id: str) -> Settings:
    """The settings a held token auditions: that source alone, itemized.

    Solo is a momentary inspection gesture (§4.1), so its caption is a real
    point in the reachable space, written up front like any other and read from
    the cache while the token is held.
    """
    return Settings(grain="itemized", admitted=(source_id,), regime=0)


def every_audition(field: Field) -> dict[str, Settings]:
    return {source.id: audition_settings(source.id) for source in field.sources}


def neighbours(field: Field, settings: Settings) -> list[Settings]:
    """The settings one gesture away: what the participant most likely asks for next.

    One mute or restore per source, one segment of balance either way, one
    detent of grain either way. These are written in the background while the
    current caption is read (see ``dpo.caption.background``), so the next Show
    caption is usually a cache hit. The list is what the console can reach in
    one gesture and nothing more — a prefetch that wandered further would spend
    the GPU on settings nobody is about to choose.

    At the unnamed grain there is nothing to mute and no balance to move, so
    its only neighbour is the grain below it.
    """
    grains = list(GRAINS)
    position = grains.index(settings.grain)
    found: list[Settings] = []

    def add(candidate: Settings) -> None:
        if candidate.key != settings.key and all(candidate.key != other.key for other in found):
            found.append(candidate)

    if settings.grain == UNNAMED_GRAIN:
        # Down from atmospheric, the console reopens with every source admitted
        # at the middle of the axis — the same state the page restores.
        ids = tuple(source.id for source in field.sources)
        regimes = field.regimes(ids)
        add(Settings(grains[position - 1], ids, len(regimes) // 2))
        return found

    admitted = list(settings.admitted)
    regimes = field.regimes(admitted)
    alpha = (regimes[settings.regime]["span"][0] + regimes[settings.regime]["span"][1]) / 2.0

    def regime_at(subset: tuple[str, ...]) -> int:
        # Carry α across an admission change, as the page does.
        for index, regime in enumerate(field.regimes(subset)):
            low, high = regime["span"]
            if low <= alpha <= high:
                return index
        return max(0, len(field.regimes(subset)) - 1)

    for source in field.sources:
        if source.id in admitted:
            if len(admitted) > 1:
                subset = tuple(item for item in admitted if item != source.id)
                add(Settings(settings.grain, subset, regime_at(subset)))
        else:
            subset = tuple(item for item in (*admitted, source.id))
            add(Settings(settings.grain, subset, regime_at(subset)))

    for step in (-1, 1):
        index = settings.regime + step
        if 0 <= index < len(regimes):
            add(Settings(settings.grain, tuple(admitted), index))

    for step in (-1, 1):
        index = position + step
        if 0 <= index < len(grains):
            grain = grains[index]
            if grain == UNNAMED_GRAIN:
                add(Settings(grain, (), 0))
            else:
                add(Settings(grain, tuple(admitted), settings.regime))
    return found
