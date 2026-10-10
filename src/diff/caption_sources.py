"""Caption sources: one cue's authored text, structured evidence and provenance.

A source is what a phase-1 rewrite is measured against (the cue's 원문) and the
only grounding a phase-2 lattice cell may use. Dream is text-only (MC-08), so a
sound that is not an event here cannot legitimately appear in an output; the
evidence is therefore structured — source, category, count, action,
closed-vocabulary qualities, location or relation, certainty — rather than free
text, and the output checks can compare against it.

Provenance travels with every source. Development runs may generate from
drafts, but :func:`bundle_problems` refuses any source whose text or evidence is
neither human-authored nor human-reviewed, so the unreviewed Gemma drafts in the
live viewing manifest can never reach a participant (plan §0.4).

The importers read the regen document and viewing manifest as JSON data; they
never import ``dpo.regen`` code and never write next to their input.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from diff.study_protocol import MAX_CHARS, MAX_LINES, sha256_of

SOURCES_SCHEMA: Final = "diff.caption-sources/v1"
SOURCE_SCHEMA: Final = "diff.caption-source/v1"

# Copied verbatim from dpo.regen.caption_controls.QUALITIES at 3842302 (final
# adjective, connective adjective, English). It is diff's own protocol data from
# here on; QUALITIES_SHA256 pins the copy and a test compares it with regen's.
QUALITIES: Final[dict[str, tuple[str, str, str]]] = {
    "soft": ("작은", "작고", "soft"),
    "loud": ("큰", "크고", "loud"),
    "low_pitched": ("음이 낮은", "음이 낮고", "low-pitched"),
    "high_pitched": ("음이 높은", "음이 높고", "high-pitched"),
    "steady": ("일정한", "일정하고", "steady"),
    "intermittent": ("간헐적인", "간헐적이고", "intermittent"),
    "rhythmic": ("규칙적인", "규칙적이고", "rhythmic"),
    "irregular": ("불규칙적인", "불규칙적이고", "irregular"),
    "rough": ("거친", "거칠고", "rough"),
    "smooth": ("부드러운", "부드럽고", "smooth"),
    "sharp": ("날카로운", "날카롭고", "sharp"),
    "dull": ("둔탁한", "둔탁하고", "dull"),
    "metallic": ("금속성인", "금속성이고", "metallic"),
    "muffled": ("먹먹한", "먹먹하고", "muffled"),
    "rising": ("점점 커지는", "점점 커지고", "growing louder"),
    "fading": ("점점 작아지는", "점점 작아지고", "fading"),
}
QUALITIES_SHA256: Final = "24d5386b370e83b59794c8c9048cd10611e8d0853ca493296473e1ab142dccd8"

# Category labels as regen's caption_controls.CATEGORIES_KO renders them.
CATEGORIES_KO: Final[dict[str, str]] = {
    "human": "사람 소리",
    "animal": "동물 소리",
    "things": "교통·기계 소리",
    "music": "음악 소리",
    "natural": "자연 소리",
    "other": "소리",
}
COUNTS_KO: Final[dict[str, str]] = {
    "one": "하나",
    "two": "둘",
    "several": "여러 개",
    "many": "많음",
    "unknown": "알 수 없음",
}
CERTAINTY_KO: Final[dict[str, str]] = {"clear": "분명함", "uncertain": "불분명함"}
UNIT_KINDS: Final = {1: "clip", 2: "video"}
STATUSES: Final = ("draft", "reviewed", "rejected")
# Which clips a source may serve. G0 and G1 read development material only and
# never participant clips; synthetic material is for tests and smoke runs and
# can never stand in for either.
MATERIALS: Final = ("development", "participant", "synthetic")
CONGRUENCE: Final = ("congruent", "incongruent")
DEPTH: Final = ("shallow", "deep")


class SourceError(ValueError):
    """A sources file that cannot be used as written; every problem is listed."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def provenance_eligible(value: str) -> bool:
    """Human-authored, or human-reviewed: the only provenance a bundle accepts."""
    return value == "human" or (value.endswith("-reviewed") and "unreviewed" not in value)


@dataclass(frozen=True)
class SoundEvent:
    event_id: str
    order: int
    source: str
    category: str
    count: str
    action: str
    qualities: tuple[str, ...]
    location: str | None
    relation: str | None
    certainty: str

    def record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "order": self.order,
            "source": self.source,
            "category": self.category,
            "count": self.count,
            "action": self.action,
            "qualities": list(self.qualities),
            "location": self.location,
            "relation": self.relation,
            "certainty": self.certainty,
        }


@dataclass(frozen=True)
class CaptionSource:
    phase: int
    unit_key: str
    cue_index: int
    start_ms: int
    end_ms: int
    text: str
    events: tuple[SoundEvent, ...]
    provenance_text: str
    provenance_evidence: str
    status: str
    material: str = "participant"
    properties: Mapping[str, str] = field(default_factory=dict)
    reviewed_by: tuple[str, ...] = ()
    reviewed_at: str | None = None
    origin: str | None = None
    note: str | None = None

    @property
    def unit_kind(self) -> str:
        return UNIT_KINDS[self.phase]

    def identity(self) -> dict[str, Any]:
        """What generation reads: provenance and review state are not part of it."""
        return {
            "schema": SOURCE_SCHEMA,
            "phase": self.phase,
            "unit": {"kind": self.unit_kind, "key": self.unit_key},
            "cue_index": self.cue_index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "events": [event.record() for event in self.ordered_events()],
        }

    @property
    def source_sha256(self) -> str:
        return sha256_of(self.identity())

    @property
    def source_id(self) -> str:
        return "src-" + self.source_sha256[:16]

    def ordered_events(self) -> tuple[SoundEvent, ...]:
        return tuple(sorted(self.events, key=lambda event: event.order))

    def evidence_terms(self) -> dict[str, list[str]]:
        """Korean surface forms the evidence licenses, by kind, for the output checks."""
        events = self.ordered_events()
        return {
            "sources": [event.source for event in events if event.source],
            "relations": [term for event in events for term in (event.location, event.relation) if term],
            "qualities": list(dict.fromkeys(q for event in events for q in event.qualities)),
        }

    def record(self) -> dict[str, Any]:
        result = self.identity()
        result.update(
            source_id=self.source_id,
            evidence={"events": result.pop("events")},
            provenance={
                "text": self.provenance_text,
                "evidence": self.provenance_evidence,
                "reviewed_by": list(self.reviewed_by),
                "reviewed_at": self.reviewed_at,
                "origin": self.origin,
            },
            status=self.status,
            material=self.material,
            properties=dict(self.properties),
            note=self.note,
        )
        return result


def bundle_problems(source: CaptionSource) -> list[str]:
    """Why this source may not enter a stimulus bundle (empty when it may)."""
    problems = []
    if source.status != "reviewed":
        problems.append(f"{source.source_id}: status is {source.status}, not reviewed")
    for kind, value in (("text", source.provenance_text), ("evidence", source.provenance_evidence)):
        if not provenance_eligible(value):
            problems.append(f"{source.source_id}: {kind} provenance {value!r} is not human or reviewed")
    if not source.events:
        problems.append(f"{source.source_id}: no evidence events")
    return problems


def _string(raw: Any, where: str, problems: list[str], *, allow_empty: bool = False) -> str:
    if not isinstance(raw, str) or (not allow_empty and not raw.strip()):
        problems.append(f"{where}: expected a non-empty string")
        return ""
    return raw


def _optional_string(raw: Any, where: str, problems: list[str]) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        problems.append(f"{where}: expected a non-empty string or null")
        return None
    return raw


def _integer(raw: Any, where: str, problems: list[str], *, minimum: int = 0) -> int:
    if type(raw) is not int or raw < minimum:
        problems.append(f"{where}: expected an integer ≥ {minimum}")
        return minimum
    return raw


def _choice(raw: Any, allowed: Iterable[str], where: str, problems: list[str]) -> str:
    options = list(allowed)
    if raw not in options:
        problems.append(f"{where}: expected one of {options}")
        return options[0]
    return str(raw)


def _event(raw: Any, where: str, problems: list[str]) -> SoundEvent | None:
    if not isinstance(raw, Mapping):
        problems.append(f"{where}: expected an object")
        return None
    unknown = set(raw) - {
        "event_id",
        "order",
        "source",
        "category",
        "count",
        "action",
        "qualities",
        "location",
        "relation",
        "certainty",
    }
    if unknown:
        problems.append(f"{where}: unknown fields {sorted(unknown)}")
    qualities = raw.get("qualities", [])
    if not isinstance(qualities, list) or not all(isinstance(q, str) for q in qualities):
        problems.append(f"{where}.qualities: expected a list of closed-vocabulary keys")
        qualities = []
    outside = [q for q in qualities if q not in QUALITIES]
    if outside:
        problems.append(f"{where}.qualities: {outside} are not in the closed vocabulary")
    if len(set(qualities)) != len(qualities):
        problems.append(f"{where}.qualities: duplicated quality")
    action = raw.get("action", "")
    if not isinstance(action, str):
        problems.append(f"{where}.action: expected a string")
        action = ""
    return SoundEvent(
        event_id=_string(raw.get("event_id"), f"{where}.event_id", problems),
        order=_integer(raw.get("order"), f"{where}.order", problems),
        source=_string(raw.get("source"), f"{where}.source", problems),
        category=_choice(raw.get("category"), CATEGORIES_KO, f"{where}.category", problems),
        count=_choice(raw.get("count"), COUNTS_KO, f"{where}.count", problems),
        action=action.strip(),
        qualities=tuple(q for q in qualities if q in QUALITIES),
        location=_optional_string(raw.get("location"), f"{where}.location", problems),
        relation=_optional_string(raw.get("relation"), f"{where}.relation", problems),
        certainty=_choice(raw.get("certainty"), CERTAINTY_KO, f"{where}.certainty", problems),
    )


def parse_source(raw: Any, where: str, problems: list[str]) -> CaptionSource | None:
    if not isinstance(raw, Mapping):
        problems.append(f"{where}: expected an object")
        return None
    if raw.get("schema") != SOURCE_SCHEMA:
        problems.append(f"{where}.schema: expected {SOURCE_SCHEMA}")
    phase = raw.get("phase")
    if phase not in UNIT_KINDS:
        problems.append(f"{where}.phase: expected 1 or 2")
        return None
    unit = raw.get("unit")
    if not isinstance(unit, Mapping) or unit.get("kind") != UNIT_KINDS[phase]:
        problems.append(f"{where}.unit: expected {{kind: {UNIT_KINDS[phase]!r}, key}}")
        unit = {}
    text = _string(raw.get("text"), f"{where}.text", problems)
    limit = MAX_CHARS[phase]
    if len(text) > limit:
        problems.append(f"{where}.text: {len(text)} characters exceeds {limit}")
    if phase in MAX_LINES and len(text.split("\n")) > MAX_LINES[phase]:
        problems.append(f"{where}.text: more than {MAX_LINES[phase]} lines")
    evidence = raw.get("evidence", {})
    events_raw = evidence.get("events", []) if isinstance(evidence, Mapping) else None
    if not isinstance(events_raw, list):
        problems.append(f"{where}.evidence.events: expected a list")
        events_raw = []
    events = [_event(item, f"{where}.evidence.events[{i}]", problems) for i, item in enumerate(events_raw)]
    parsed = tuple(event for event in events if event is not None)
    if len({e.event_id for e in parsed}) != len(parsed) or len({e.order for e in parsed}) != len(parsed):
        problems.append(f"{where}.evidence.events: event_id and order must be unique")
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        problems.append(f"{where}.provenance: expected an object")
        provenance = {}
    reviewed_by = provenance.get("reviewed_by", [])
    if not isinstance(reviewed_by, list) or not all(isinstance(r, str) and r for r in reviewed_by):
        problems.append(f"{where}.provenance.reviewed_by: expected a list of reviewer IDs")
        reviewed_by = []
    properties = raw.get("properties", {})
    if not isinstance(properties, Mapping):
        problems.append(f"{where}.properties: expected an object")
        properties = {}
    if "audiovisual_congruence" in properties and properties["audiovisual_congruence"] not in CONGRUENCE:
        problems.append(f"{where}.properties.audiovisual_congruence: expected one of {list(CONGRUENCE)}")
    if "description_depth" in properties and properties["description_depth"] not in DEPTH:
        problems.append(f"{where}.properties.description_depth: expected one of {list(DEPTH)}")
    source = CaptionSource(
        phase=int(phase),
        unit_key=_string(unit.get("key"), f"{where}.unit.key", problems),
        cue_index=_integer(raw.get("cue_index"), f"{where}.cue_index", problems),
        start_ms=_integer(raw.get("start_ms"), f"{where}.start_ms", problems),
        end_ms=_integer(raw.get("end_ms"), f"{where}.end_ms", problems, minimum=1),
        text=text,
        events=parsed,
        provenance_text=_string(provenance.get("text"), f"{where}.provenance.text", problems),
        provenance_evidence=_string(provenance.get("evidence"), f"{where}.provenance.evidence", problems),
        status=_choice(raw.get("status"), STATUSES, f"{where}.status", problems),
        material=_choice(raw.get("material"), MATERIALS, f"{where}.material", problems),
        properties={str(k): str(v) for k, v in properties.items()},
        reviewed_by=tuple(reviewed_by),
        reviewed_at=_optional_string(
            provenance.get("reviewed_at"), f"{where}.provenance.reviewed_at", problems
        ),
        origin=_optional_string(provenance.get("origin"), f"{where}.provenance.origin", problems),
        note=_optional_string(raw.get("note"), f"{where}.note", problems),
    )
    if source.end_ms <= source.start_ms:
        problems.append(f"{where}: end_ms must be after start_ms")
    claimed = raw.get("source_id")
    if claimed is not None and claimed != source.source_id:
        problems.append(f"{where}.source_id: {claimed} does not match its content ({source.source_id})")
    return source


def _check_units(sources: Sequence[CaptionSource], problems: list[str]) -> None:
    by_unit: dict[tuple[int, str], list[CaptionSource]] = {}
    for source in sources:
        by_unit.setdefault((source.phase, source.unit_key), []).append(source)
    for (phase, key), cues in by_unit.items():
        cues.sort(key=lambda cue: cue.cue_index)
        indexes = [cue.cue_index for cue in cues]
        if indexes != list(range(len(cues))):
            problems.append(
                f"{UNIT_KINDS[phase]} {key}: cue indexes must be 0..{len(cues) - 1}, got {indexes}"
            )
            continue
        if cues[0].start_ms != 0 and phase == 2:
            problems.append(f"video {key}: cues must start at 0 ms")
        for before, after in zip(cues, cues[1:], strict=False):
            if phase == 2 and after.start_ms != before.end_ms:
                problems.append(
                    f"video {key}: cue {after.cue_index} must start where cue {before.cue_index} ends"
                )
            if phase == 1 and after.start_ms < before.end_ms:
                problems.append(f"clip {key}: cue {after.cue_index} overlaps cue {before.cue_index}")


@dataclass(frozen=True)
class SourceSet:
    sources: tuple[CaptionSource, ...]

    def by_id(self) -> dict[str, CaptionSource]:
        return {source.source_id: source for source in self.sources}

    def phase(self, phase: int) -> list[CaptionSource]:
        return sorted((s for s in self.sources if s.phase == phase), key=lambda s: (s.unit_key, s.cue_index))

    def record(self) -> dict[str, Any]:
        return {"schema": SOURCES_SCHEMA, "sources": [source.record() for source in self.sources]}


def parse_sources(document: Any) -> SourceSet:
    """Validate a ``diff.caption-sources/v1`` document; raise with every problem."""
    problems: list[str] = []
    if not isinstance(document, Mapping) or document.get("schema") != SOURCES_SCHEMA:
        raise SourceError([f"expected a {SOURCES_SCHEMA} document"])
    raw_sources = document.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise SourceError(["sources: expected a non-empty list"])
    parsed = [parse_source(raw, f"sources[{i}]", problems) for i, raw in enumerate(raw_sources)]
    sources = tuple(source for source in parsed if source is not None)
    keys = [(s.phase, s.unit_key, s.cue_index) for s in sources]
    if len(set(keys)) != len(keys):
        problems.append("sources: a (phase, unit, cue_index) appears twice")
    _check_units(sources, problems)
    if problems:
        raise SourceError(problems)
    return SourceSet(sources)


def import_viewing_manifest(manifest: Mapping[str, Any], material: str = "participant") -> SourceSet:
    """Phase-2 drafts from regen's viewing manifest, marked unreviewed.

    The live manifest's evidence is visual context only — it states that no
    audio-event labels are supplied — and its fallbacks are Gemma audio drafts.
    So every imported source has no events, unreviewed provenance and the
    original evidence string in ``note``: material for authors, never a bundle.
    """
    sources: list[CaptionSource] = []
    for video in manifest.get("viewing_videos", []):
        if video.get("status") != "ready":
            continue
        for index, cue in enumerate(video.get("cues", [])):
            fallback = cue.get("fallback", {})
            text = fallback.get("ko", "") if isinstance(fallback, Mapping) else ""
            sources.append(
                CaptionSource(
                    phase=2,
                    unit_key=str(video["id"]),
                    cue_index=index,
                    start_ms=int(cue["start_ms"]),
                    end_ms=int(cue["end_ms"]),
                    text=str(text).strip(),
                    events=(),
                    provenance_text="gemma-draft-unreviewed",
                    provenance_evidence="assistant-visual-context-unreviewed",
                    status="draft",
                    material=material,
                    origin=f"viewing-manifest:{video['id']}:cue:{index}",
                    note=f"manifest evidence: {cue.get('evidence', '')}",
                )
            )
    return SourceSet(tuple(sources))


def import_regen_document(document: Mapping[str, Any], material: str = "participant") -> SourceSet:
    """Phase-1 drafts from a regen session document's prepared tracks.

    The prepared text becomes the draft 원문 with its provenance marked
    unreviewed; evidence is left empty for authors to fill in.
    """
    sources: list[CaptionSource] = []
    for segment in document.get("segments", {}).values():
        for cue in segment.get("prepared_track", []):
            raw = cue.get("text")
            text = raw.get("ko", "") if isinstance(raw, Mapping) else str(raw or "")
            english = raw.get("en") if isinstance(raw, Mapping) else None
            sources.append(
                CaptionSource(
                    phase=1,
                    unit_key=str(segment["clip_id"]),
                    cue_index=int(cue["index"]),
                    start_ms=int(cue["start_ms"]),
                    end_ms=int(cue["end_ms"]),
                    text=text.strip(),
                    events=(),
                    provenance_text="regen-prepared-track-unreviewed",
                    provenance_evidence="none",
                    status="draft",
                    material=material,
                    origin=f"regen-document:{segment.get('segment')}:cue:{cue['index']}",
                    note=f"english prepared text: {english}" if english else None,
                )
            )
    return SourceSet(tuple(sources))
