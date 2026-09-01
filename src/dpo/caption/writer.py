"""Prose from settings: the one slow thing in the interface, and its cache.

The skeleton is deterministic and instant; the sentence is "requested
explicitly and is the only slow thing in the interface" (spec 2), always behind
Show caption. A writer turns a ``CaptionRequest`` — the admitted sources in the
order the skeleton shows them, or the heads with their members, or the scene or
atmosphere row — into one or two sentences of sentence prose.

``CaptionRequest`` is the seam between the instrument and this module:
``dpo.session.writer`` builds one from a skeleton ordering, and from the
request onward nothing here knows what a skeleton is.

Two writers. ``TemplateWriter`` is deterministic string assembly, so the
instrument can be built, tested, and demonstrated without a GPU and so every
behavioral test has a caption it can predict. ``GemmaWriter`` conditions the
study's own model on the shot's audio with an instruction that names exactly
the sources to mention in exactly this order; the instruction is a module
constant so a reader can inspect what the model was told.

``CachedWriter`` is what makes identical settings return the identical caption,
and what makes an audition instant: single-source and single-head captions are
written up front and read from the cache while a row is held.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dpo.core.atomic import replace_atomically

# The two reserved lines of the caption box, in characters. Both instruments
# typeset the caption at the same measure, so both hold to one budget: the
# template writer stays inside it by dropping phrases or leads, the Gemma
# instruction states it, and ``GemmaWriter`` enforces it after the fact
# (docs/v1-session/runbook.md §3).
CAPTION_MAX_CHARS = 96


@dataclass(frozen=True)
class SourceSpec:
    id: str
    token: str
    prose: str
    phrases: tuple[str, str]


@dataclass(frozen=True)
class HeadSpec:
    role_id: str
    text: str
    members: tuple[SourceSpec, ...]


@dataclass(frozen=True)
class CaptionRequest:
    clip_id: str
    shot_id: str
    task: str  # "shaped" | "control"
    level: str
    settings_key: str
    sources: tuple[SourceSpec, ...]  # admitted, in order (itemized) / members in head order (grouped)
    heads: tuple[HeadSpec, ...]  # grouped only, in order
    scene_prose: str
    atmosphere_prose: str
    # Gemma only: the shot's audio excerpt on a shaped clip, a still from the
    # middle of the shot on a control clip (spec 4.6: a visual description).
    media_path: Path | None


class CaptionWriter(Protocol):
    def write(self, request: CaptionRequest) -> str: ...


class WriterError(RuntimeError):
    """The writer could not produce a caption; the route turns this into 502."""


# ---- the template writer --------------------------------------------------


def join_clauses(clauses: Sequence[str]) -> str:
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def as_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


class TemplateWriter:
    """Deterministic sentence prose from the request; no model, no randomness.

    Rules (contract §4): sentence case, one sentence, sources mentioned in the
    request's order and nothing else.

    * itemized — each source becomes ``"{prose} {right phrase lowercased}"``
      (the temporal phrase on the audio skeleton, the motion phrase on the
      control one); clauses join with commas and a final "and". When that
      sentence would overflow the two-line box (``CAPTION_MAX_CHARS``) the
      phrases are dropped and the sources alone are joined, still in order.
    * grouped — one clause per head in order, led by the head text lowercased:
      ``"underneath, footsteps and chatter; standing out, ..."`` — members
      joined the same way as itemized clauses, never truncated. Over the
      budget, the clauses keep their order and lose their leads.
    * scene — the shot's scene prose. atmospheric — its atmosphere prose.
    """

    def write(self, request: CaptionRequest) -> str:
        if request.level == "scene":
            return as_sentence(request.scene_prose)
        if request.level == "atmospheric":
            return as_sentence(request.atmosphere_prose)
        if request.level == "grouped":
            filled = [head for head in request.heads if head.members]
            groups = [join_clauses([member.prose for member in head.members]) for head in filled]
            leads = [head.text.lower() for head in filled]
            led = as_sentence(
                "; ".join(f"{lead}, {group}" for lead, group in zip(leads, groups, strict=True))
            )
            if len(led) <= CAPTION_MAX_CHARS:
                return led
            return as_sentence("; ".join(groups))
        phrased = as_sentence(
            join_clauses([f"{source.prose} {source.phrases[1].lower()}" for source in request.sources])
        )
        if len(phrased) <= CAPTION_MAX_CHARS:
            return phrased
        return as_sentence(join_clauses([source.prose for source in request.sources]))


# ---- the Gemma writer -----------------------------------------------------


class StimulusAdapter(Protocol):
    def generate_stimulus(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> str: ...


GEMMA_INSTRUCTION = (
    "You write the caption for one shot of street footage from its sound. Write one or two short"
    f" sentences of plain sentence prose, at most {CAPTION_MAX_CHARS} characters in total, in the"
    " register of a subtitle for the deaf and hard of hearing: no brackets, no lists, no numerals,"
    " no headings, and nothing about yourself. Describe only what is heard.\n"
)
GEMMA_LEVEL_RULES = {
    "itemized": (
        "Mention ONLY these sound sources, in THIS order, and no other source: {sources}."
        " Each entry is a source with two notes, how it sits in the frame and how it moves in"
        " time. Write the notes into prose — complete sentences with full stops, never a list, and"
        " never the notes copied word for word."
    ),
    "grouped": (
        "Write one clause per group, in THIS order, mentioning only the members listed under it"
        " and no other source: {groups}."
    ),
    "scene": (
        "Write one sentence that names the scene as a whole, and nothing else — no individual sound"
        " source, nothing added. The scene: {scene}"
    ),
    "atmospheric": (
        "Describe only the temporal character of the moment as a whole, with no nouns naming any"
        " source or thing. The character: {atmosphere}"
    ),
}
# The control task (spec 4.6): the same skeleton over visible things, and the
# caption is a plain visual description. The writer looks at a still from the
# middle of the shot instead of listening to it.
GEMMA_VISUAL_INSTRUCTION = (
    "You write the caption for one shot of street footage from this frame of it. Write one or two"
    f" short sentences of plain sentence prose, at most {CAPTION_MAX_CHARS} characters in total, as a"
    " plain visual description: no brackets, no lists, no numerals, no headings, and nothing about"
    " yourself. Describe only what is seen.\n"
)
GEMMA_VISUAL_LEVEL_RULES = {
    "itemized": (
        "Mention ONLY these visible things, in THIS order, and nothing else: {sources}."
        " Each entry is a thing with two notes, its size in the frame and how it moves. Write the"
        " notes into prose — complete sentences with full stops, never a list, and never the notes"
        " copied word for word."
    ),
    "grouped": (
        "Write one clause per group, in THIS order, mentioning only the things listed under it"
        " and nothing else: {groups}."
    ),
    "scene": (
        "Write one sentence that names the scene as a whole, and nothing else — no individual"
        " thing, nothing added. The scene: {scene}"
    ),
    "atmospheric": (
        "Describe only the character of the motion in the moment as a whole, with no nouns naming"
        " any thing. The character: {atmosphere}"
    ),
}
GEMMA_MAX_NEW_TOKENS = 60
# Appended to the system text for the one retry an over-budget caption gets.
# The budget policy, measured on the base E4B over the worst cases (four or five
# entries, real and synthetic audio, a control still): (1) with four or more
# entries the first draft carries the hint below, which fit two of six cases
# outright and shortened the rest; (2) an over-budget draft gets one retry
# asking for a comma list with at most two words of notes per entry, which fit
# every remaining case but one while keeping every entry in order — a retry
# asking for prose "in this shape" dropped entries instead; (3) what still
# overruns becomes the template's sentence for the same list (``write``).
GEMMA_MANY_ENTRIES_HINT = (
    " The whole caption must stay under {budget} characters, so with four or more entries most of"
    " them get no note at all — name them, and keep the notes for one or two."
)
GEMMA_TIGHTEN = (
    "\nToo long: {length} characters against a limit of {budget}. Now write a single sentence under"
    " {budget} characters that lists {names}, in that order, separated by commas, each with at most"
    " two words of its notes, and nothing else."
)
MANY_ENTRIES = 4
GEMMA_TIGHTEN_UNNAMED = (
    "\nA draft of this caption ran to {length} characters, over the {budget}-character limit."
    " Write it again in at most {budget} characters."
)


def tighten(messages: list[dict[str, Any]], length: int, names: Sequence[str]) -> list[dict[str, Any]]:
    """The same messages with the overrun named in the system text."""
    system = dict(messages[0])
    note = (
        GEMMA_TIGHTEN.format(length=length, budget=CAPTION_MAX_CHARS, names=", ".join(names))
        if names
        else GEMMA_TIGHTEN_UNNAMED.format(length=length, budget=CAPTION_MAX_CHARS)
    )
    system["content"] = str(system["content"]) + note
    return [system, *messages[1:]]


def cut_at_sentence(caption: str, budget: int) -> str:
    """The caption up to its last sentence end inside the budget, or unchanged if none fits."""
    head = caption[:budget]
    end = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
    return caption[: end + 1] if end > 0 else caption


def _entry(source: SourceSpec) -> str:
    """One itemized entry as the instruction lists it: the thing, then its two notes."""
    return f"{source.prose} — {source.phrases[0]}, {source.phrases[1]}"


def gemma_instruction(request: CaptionRequest) -> str:
    """The exact system text for a request, assembled from the module constants."""
    control = request.task == "control"
    preamble = GEMMA_VISUAL_INSTRUCTION if control else GEMMA_INSTRUCTION
    rule = (GEMMA_VISUAL_LEVEL_RULES if control else GEMMA_LEVEL_RULES)[request.level]
    if request.level == "itemized":
        listed = "; ".join(_entry(s) for s in request.sources)
        text = preamble + rule.format(sources=listed)
        if len(request.sources) >= MANY_ENTRIES:
            text += GEMMA_MANY_ENTRIES_HINT.format(budget=CAPTION_MAX_CHARS)
        return text
    if request.level == "grouped":
        groups = "; ".join(
            f"{head.text}: " + ", ".join(member.prose for member in head.members) for head in request.heads
        )
        return preamble + rule.format(groups=groups)
    if request.level == "scene":
        return preamble + rule.format(scene=request.scene_prose)
    return preamble + rule.format(atmosphere=request.atmosphere_prose)


def visual_messages(instruction: str, image_reference: str) -> list[dict[str, Any]]:
    """Messages for a control caption: the instruction and one frame of the shot.

    ``stimulus_messages`` (prompt.py) always carries audio, which a visual
    description must not hear; this is the image-only counterpart, built here
    because it exists for the control task alone.
    """
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": [{"type": "image", "image": image_reference}]},
    ]


class GemmaWriter:
    """The study's model, conditioned on the shot's audio, told exactly what to mention.

    Decoding is frozen (temperature 0, seed 0) so the writer is a function of
    its request, which is what lets the cache stand in for it. Measured on one
    RTX 3090 with the unquantized base E4B (2026-08-28): 15.5 GiB resident,
    0.6-1.6 s per caption after the first call, and the model does follow the
    skeleton's order — but it does not keep to the character budget the
    instruction asks for (a five-source itemized caption came back at 110
    against 96), so ``write`` measures the caption and enforces the budget.
    """

    def __init__(
        self,
        adapter: StimulusAdapter,
        *,
        max_new_tokens: int = GEMMA_MAX_NEW_TOKENS,
        fallback: CaptionWriter | None = None,
    ) -> None:
        self.adapter = adapter
        self.max_new_tokens = max_new_tokens
        # What a caption becomes when the model will not fit the box in two
        # tries: the template's sentence for the same list, which keeps every
        # entry in order and drops notes to fit. A caption that silently lost
        # an admitted source would misreport the skeleton it was written for.
        self.fallback: CaptionWriter = TemplateWriter() if fallback is None else fallback

    def messages(self, request: CaptionRequest) -> list[dict[str, Any]]:
        from dpo.models.gemma4.prompt import stimulus_messages

        if request.media_path is None:
            raise WriterError("the Gemma writer needs the shot's media; no media_path on the request")
        if request.task == "control":
            return visual_messages(gemma_instruction(request), str(request.media_path))
        return stimulus_messages(gemma_instruction(request), audio_reference=str(request.media_path))

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        try:
            text = self.adapter.generate_stimulus(
                messages,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
                seed=0,
            )
        except WriterError:
            raise
        except Exception as exc:
            # Any backend failure is one 502 to the participant (spec Table 6 `error`).
            raise WriterError(f"Gemma generation failed: {exc}") from exc
        return " ".join(text.split()).strip()

    def write(self, request: CaptionRequest) -> str:
        messages = self.messages(request)
        caption = self._generate(messages)
        if not caption:
            raise WriterError("Gemma returned an empty caption")
        if len(caption) > CAPTION_MAX_CHARS:
            # The two-line box clips a longer caption mid-word behind an
            # ellipsis, which the participant would read as the caption. One
            # deterministic retry names the overrun; if the model still overruns,
            # the caption ends at its last full sentence inside the budget, so
            # what is read is whole sentences in the skeleton's order.
            shorter = self._generate(tighten(messages, len(caption), [s.prose for s in request.sources]))
            if shorter and len(shorter) < len(caption):
                caption = shorter
            if len(caption) > CAPTION_MAX_CHARS:
                templated = self.fallback.write(request)
                caption = (
                    templated
                    if len(templated) <= CAPTION_MAX_CHARS
                    else cut_at_sentence(caption, CAPTION_MAX_CHARS)
                )
        return caption


# ---- the cache --------------------------------------------------------------


class CachedWriter:
    """Identical settings ⇒ identical caption; a rerun costs nothing.

    Keyed ``"{clip_id}/{shot_id}/{settings_key}"`` in one JSON file under
    ``--out``, loaded at start and replaced atomically after every miss.

    One lock serialises ``write_cached``: FastAPI runs the sync caption route
    in a threadpool, so two requests for one key could otherwise both miss,
    both call the inner writer (two GPU generations that need not agree), and
    write the file over each other. With the lock the guarantee that identical
    settings return the identical caption (spec 1, 7) rests on the cache, not
    on the model's determinism.
    """

    def __init__(self, inner: CaptionWriter, cache_path: Path) -> None:
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.entries: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.cache_path.is_file():
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.entries = {str(key): str(value) for key, value in loaded.items()}

    @staticmethod
    def key_for(request: CaptionRequest) -> str:
        return f"{request.clip_id}/{request.shot_id}/{request.settings_key}"

    def lookup(self, request: CaptionRequest) -> str | None:
        return self.entries.get(self.key_for(request))

    def write_cached(self, request: CaptionRequest) -> tuple[str, bool]:
        """(caption, whether it came from the cache)."""
        key = self.key_for(request)
        with self._lock:
            cached = self.entries.get(key)
            if cached is not None:
                return cached, True
            caption = self.inner.write(request)
            self.entries[key] = caption
            payload = (
                json.dumps(self.entries, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            replace_atomically(self.cache_path, payload)
        return caption, False

    def write(self, request: CaptionRequest) -> str:
        return self.write_cached(request)[0]


# (clip_id, shot) -> the media the Gemma writer listens to or looks at for that
# shot: audio on a shot captioned from sound, a still on one captioned from the
# image. None for writers that need no media.
ShotMedia = Callable[[str, Mapping[str, Any]], Path | None]


__all__ = [
    "CAPTION_MAX_CHARS",
    "GEMMA_INSTRUCTION",
    "GEMMA_LEVEL_RULES",
    "GEMMA_MANY_ENTRIES_HINT",
    "GEMMA_TIGHTEN",
    "GEMMA_VISUAL_INSTRUCTION",
    "GEMMA_VISUAL_LEVEL_RULES",
    "CachedWriter",
    "CaptionRequest",
    "CaptionWriter",
    "GemmaWriter",
    "HeadSpec",
    "ShotMedia",
    "SourceSpec",
    "StimulusAdapter",
    "TemplateWriter",
    "WriterError",
    "as_sentence",
    "cut_at_sentence",
    "join_clauses",
    "gemma_instruction",
    "tighten",
    "visual_messages",
]
