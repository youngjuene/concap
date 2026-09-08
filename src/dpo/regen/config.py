"""The versioned configuration artifact, and the hash every result carries.

Sorted by the same criterion the other instruments use. If varying it within a
session is the point of the session, it is an experimental variable; if varying
it between sessions would invalidate comparison, it is a method constant; if it
must be fixed to the study once and then hold still, it is a calibration.

    | Quantity                          | Tier                  | Cadence      |
    |-----------------------------------|-----------------------|--------------|
    | condition-segment assignment rule | method constant       | per study    |
    | caption track schema (§9.6)       | method constant       | per study    |
    | ART block reused across 2a and 4  | method constant       | per study    |
    | cue slot count (§9.4)             | calibration           | per study    |
    | minimum point count (§4)          | calibration           | per study    |
    | latency ceiling (§6)              | calibration           | per study    |
    | per-slot character and line caps  | calibration           | per study    |
    | scale points and anchors (§9.3)   | calibration           | per study    |
    | the languages offered (§9.3)      | calibration           | per study    |
    | which language a participant reads| experimental variable | per person   |
    | which segment a participant gets  | experimental variable | per person   |

The condition-segment assignment is the variable the design turns: fixed on
entry from the sequence number (§1), never moved after. A study offering two
languages adds a second thing that varies per person, and it is the only one a
participant chooses — before the first clip, and then locked, because the
study's main measure is the change in the ART answers between §3 and §8 and a
participant who read one viewing in Korean and the other in English would have
changed two things. Everything else is frozen here, hashed, and stamped onto
every caption, every endpoint and every log line, so a result is reproducible
from its stamp.

``cue_slots`` is the single value §9.4 requires. The prepared track is
validated against it and so is the regenerated one, which is what makes the two
conditions comparable at all: a prepared track of six cues against a
regenerated track of four would confound provenance with density. It lives here
rather than in the document so that a document cannot disagree with the rules
its captions were validated under.

The scale (§9.3) is here for the same reason. Two survey pages carry the same
items and must carry the same response format; holding the format beside the
items would let one page's copy drift from the other's.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

CONFIG_SCHEMA = "dpo.caption-regen-config/v1"
HASH_LENGTH = 12

# §5's vocabulary: AudioSet's top-level classes, in AudioSet's own order,
# restricted to the five that occur in this corpus. Fixed rather than drawn
# from the clip, which is the point of the redesign — a participant shown only
# the families a clip contains can report a true positive and nothing else, so
# the screen could not tell "heard it" from "had the chance to say so". Asked
# about all five, they can claim a family that is not there, and a false alarm
# is a measure rather than an impossibility.
#
# `prose` is what §6 puts in a caption, and is deliberately not the display
# label: "the sound of sounds of things" is not a sentence. The label is what
# the participant reads and what the log records; the prose is what the writer
# is handed.
SOUND_FAMILIES: Mapping[str, str] = {
    "human": "people",
    "animal": "animals",
    "things": "vehicles and machinery",
    "music": "music",
    "natural": "wind and water",
}

# Declared, hashed, never dispatched on. Each line states what some module in
# this package does, so a study's stamp changes when the method changes.
# Changing a formula without changing the declaration beside it leaves two
# studies sharing one stamp, which is the failure the stamp exists to prevent.
# The AudioSet class name a staged document stores in `stem.parent`, mapped to
# the family key §5 answers in. The two vocabularies exist because the document
# records the ontology's own words and §5 records its own ids, and for a while
# the log wrote one field in each: `heard: ["things"]` beside
# `present: ["Sounds of things"]`, two lists with no value in common, in a row
# whose whole purpose was letting the two be compared. Mapped here, at the one
# boundary they meet, rather than by whoever reads the log later.
FAMILY_OF_PARENT: Mapping[str, str] = {
    "Human sounds": "human",
    "Animal": "animal",
    "Sounds of things": "things",
    "Music": "music",
    "Natural sounds": "natural",
}

METHOD_CONSTANTS: Mapping[str, Any] = {
    "assignment": "prepared on segment A when the sequence number is even, on B when it is odd",
    "assignment_from": "participant sequence number alone; no stored table",
    "caption_schema": "identical for prepared and regenerated tracks",
    "cue_timings": "pre-set per slot; regeneration writes text only",
    "art_block": "one definition, referenced by both survey pages",
    "point_matching": "once, on the submitted coordinates; smallest containing mask wins",
    "unclassified": "points outside every mask are kept with their coordinates",
    "regeneration_inputs": "matched visual labels excluding unclassified, plus the sound "
    "families reported heard",
    # §5 stopped being a selection among the sources a clip happens to carry
    # and became a fixed judgment on all five families, so what the screen
    # measures — and what §6 is conditioned on — is a different thing. Named
    # here because it is hashed: a session run before this constant changed is
    # not comparable with one run after, and the config hash is what says so.
    "auditory_report": "heard / did not hear, on each of five fixed sound families",
}


class ConfigError(ValueError):
    """The configuration is not one a study could be run under."""


@dataclass(frozen=True)
class Scale:
    """The response format both survey pages share (§9.3).

    ``anchors`` are the words at the ends. They are copy, they are hashed, and
    they are served to the page from here, so the two survey pages cannot come
    to disagree about what a 1 or a 7 means. ``points`` is the range: responses
    are integers in ``1..points`` and nothing else is accepted.
    """

    points: int = 7
    anchors: tuple[str, str] = ("Not at all", "Very much")

    def __post_init__(self) -> None:
        if self.points < 2:
            raise ConfigError("a scale needs at least two points")
        if len(self.anchors) != 2 or not all(anchor.strip() for anchor in self.anchors):
            raise ConfigError("a scale needs a low and a high anchor, both non-empty")

    def accepts(self, value: object) -> bool:
        """True for an answer this scale could have produced."""
        return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= self.points


@dataclass(frozen=True)
class Calibration:
    """What is fixed to the study once and then holds still.

    ``cue_slots`` is §9.4's single value. ``slot_max_chars`` and
    ``slot_max_lines`` are what §6 validates a generated slot against, and what
    an authored prepared track is validated against on load — one rule, applied
    to both tracks, because §9.6 stores them under one schema.

    ``minimum_points`` is §4's floor on the visual selection. One, because the
    floor is there to stop an empty submission and nothing more: a participant
    who found one thing worth marking has answered the question, and a higher
    floor makes them invent marks to get past the screen — which is the one
    failure that would corrupt the measure rather than thin it.
    ``latency_ceiling_ms``
    is §6's: past it the regeneration is abandoned for the fallback track, and
    the participant waits a bounded time rather than an unbounded one.

    ``languages`` are the language tags the study offers, the first being the
    one a session opens in. They are tags, not locales to format with: the
    check is that generated text is in the language on screen, which is the
    failure mode a model that answers in English to a Korean prompt produces.

    A study offering two puts the choice in front of the participant and fixes
    it before the first clip plays (§2). Both are in the hash, because a
    bilingual study and a monolingual one over the same footage are two
    studies, and because which language came first is part of what was run.
    """

    cue_slots: int = 4
    slot_max_chars: int = 96
    slot_max_lines: int = 2
    minimum_points: int = 1
    latency_ceiling_ms: int = 20000
    languages: tuple[str, ...] = ("en",)
    scale: Scale = field(default_factory=Scale)

    def __post_init__(self) -> None:
        if self.cue_slots < 1:
            raise ConfigError("a caption track needs at least one cue slot")
        if self.slot_max_chars < 1:
            raise ConfigError("the per-slot character cap is a positive length")
        if self.slot_max_lines < 1:
            raise ConfigError("the per-slot line cap is a positive count")
        if self.minimum_points < 1:
            raise ConfigError("§4 requires at least one point before Next enables")
        if self.latency_ceiling_ms < 1:
            raise ConfigError("the latency ceiling is a positive duration")
        object.__setattr__(self, "languages", tuple(self.languages))
        if not self.languages:
            raise ConfigError("a study runs in at least one language")
        if any(not tag.strip() for tag in self.languages):
            raise ConfigError("every study language must be named")
        if len(set(self.languages)) != len(self.languages):
            raise ConfigError(f"a language is offered twice: {list(self.languages)}")

    @property
    def language(self) -> str:
        """The one a session opens in, and the only one a study that offers one has."""
        return self.languages[0]

    def offers(self, tag: object) -> bool:
        return isinstance(tag, str) and tag in self.languages


@dataclass(frozen=True)
class Configuration:
    """The frozen artifact, and its hash.

    ``study_id`` and ``corpus_id`` are in the hash on purpose: two studies
    calibrated to identical numbers over different footage are still two
    studies, and neither result should claim to be reproducible from the
    other's clips.
    """

    study_id: str
    corpus_id: str
    calibration: Calibration = field(default_factory=Calibration)

    def artifact(self) -> dict[str, Any]:
        """The hashable document: everything frozen, nothing computed."""
        return {
            "schema": CONFIG_SCHEMA,
            "study_id": self.study_id,
            "corpus_id": self.corpus_id,
            "method_constants": dict(METHOD_CONSTANTS),
            # §5's vocabulary is hashed because it is a study input, not a
            # presentation detail. The prose beside each family is handed
            # straight to §6's writer, so editing "wind and water" — or adding
            # a sixth family — changes the stimulus a participant is shown.
            # `method_constants` declares only the *shape* of §5's question;
            # without this, two studies asking about different families would
            # share one stamp, which is the failure the stamp exists to
            # prevent.
            "sound_families": dict(SOUND_FAMILIES),
            "calibration": asdict(self.calibration),
        }

    @property
    def hash(self) -> str:
        """The stamp: sha256 over the canonical artifact, truncated.

        Canonical means sorted keys and no incidental whitespace, so the stamp
        depends on the values and not on how the file was written.
        """
        payload = json.dumps(self.artifact(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LENGTH]

    def stamped(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """``document`` with the stamp on it. Every result leaves through here."""
        return {**document, "config_hash": self.hash}

    @property
    def cue_slots(self) -> int:
        return self.calibration.cue_slots

    @property
    def languages(self) -> tuple[str, ...]:
        return self.calibration.languages

    @property
    def language(self) -> str:
        """The one a session opens in."""
        return self.calibration.language

    @property
    def scale(self) -> Scale:
        return self.calibration.scale


def load_configuration(raw: Mapping[str, Any]) -> Configuration:
    """Rebuild a configuration from an artifact, refusing a changed method.

    A stored artifact whose ``method_constants`` differ from this build's was
    written by a different instrument. Loading it would produce results stamped
    with a hash that no longer describes how they were computed, so it is an
    error rather than a warning.
    """
    if raw.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"not a {CONFIG_SCHEMA} artifact")
    if raw.get("method_constants") != dict(METHOD_CONSTANTS):
        raise ConfigError(
            "the artifact's method constants are not this build's; "
            "results computed here could not carry its hash honestly"
        )
    if raw.get("sound_families") != dict(SOUND_FAMILIES):
        raise ConfigError(
            "the artifact's sound families are not this build's; §5 would ask a "
            "different question and §6 would be given different words for it"
        )
    calibration = dict(raw.get("calibration") or {})
    if "languages" in calibration:
        # JSON has no tuples; the frozen dataclass wants one.
        calibration["languages"] = tuple(calibration["languages"])
    try:
        scale = calibration.pop("scale", None)
        if scale is not None:
            anchors = scale.get("anchors")
            if not isinstance(anchors, Sequence) or isinstance(anchors, str):
                raise ConfigError("calibration.scale.anchors must be a pair of strings")
            calibration["scale"] = Scale(
                points=int(scale["points"]), anchors=(str(anchors[0]), str(anchors[1]))
            )
        return Configuration(
            study_id=str(raw["study_id"]),
            corpus_id=str(raw["corpus_id"]),
            calibration=Calibration(**calibration),
        )
    except (IndexError, KeyError, TypeError) as exc:
        raise ConfigError(f"the artifact is missing or misnames a field: {exc}") from exc
