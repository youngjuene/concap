"""Caption requests for the skeleton instrument (``docs/v1-session/``).

The writers, the cache, and the budget live in :mod:`dpo.caption.writer` and
are shared with the console instrument. What belongs to *this* instrument is
how a request is built: from a settings triple the skeleton could have
produced — a level, an admitted set, and one of the orderings that set can
take — with the role heads and member order the skeleton draws.

The shared writer names are re-exported here so this instrument's modules have
one import for everything caption-shaped.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dpo.caption.writer import (
    CAPTION_MAX_CHARS,
    GEMMA_INSTRUCTION,
    GEMMA_LEVEL_RULES,
    GEMMA_MANY_ENTRIES_HINT,
    GEMMA_TIGHTEN,
    GEMMA_VISUAL_INSTRUCTION,
    GEMMA_VISUAL_LEVEL_RULES,
    CachedWriter,
    CaptionRequest,
    CaptionWriter,
    GemmaWriter,
    HeadSpec,
    ShotMedia,
    SourceSpec,
    StimulusAdapter,
    TemplateWriter,
    WriterError,
    cut_at_sentence,
    gemma_instruction,
    tighten,
    visual_messages,
)
from dpo.session.document import roles_for
from dpo.session.skeleton import (
    Settings,
    balance_of,
    heads_present,
    members_in_order,
    settings_key,
)

# ---- building a request from settings ---------------------------------------


def _source_spec(source: Mapping[str, Any]) -> SourceSpec:
    return SourceSpec(
        id=str(source["id"]),
        token=str(source["token"]),
        prose=str(source["prose"]),
        phrases=(str(source["phrases"][0]), str(source["phrases"][1])),
    )


def build_request(
    document: Mapping[str, Any],
    clip: Mapping[str, Any],
    shot: Mapping[str, Any],
    settings: Settings,
    media_path: Path | None,
) -> CaptionRequest:
    """The request for validated settings: the skeleton's list, as prose inputs.

    Itemized carries the admitted sources in the chosen order. Grouped carries
    the present heads in the chosen order, each with its members in the head's
    itemized order at the ordering's balance (contract §3), and ``sources``
    flattened in that same reading order so every writer mentions things in
    the order the participant sees them.
    """
    roles = roles_for(document, clip)
    by_id = {str(source["id"]): source for source in shot["sources"]}
    heads: tuple[HeadSpec, ...] = ()
    if settings.level == "itemized":
        sources = tuple(_source_spec(by_id[source_id]) for source_id in settings.order)
    elif settings.level == "grouped":
        admitted = [by_id[source_id] for source_id in settings.admitted]
        balance = balance_of(shot, roles, settings)
        heads = tuple(
            HeadSpec(
                role_id=role,
                text=roles[role],
                members=tuple(_source_spec(member) for member in members_in_order(admitted, role, balance)),
            )
            for role in settings.order
        )
        sources = tuple(member for head in heads for member in head.members)
    else:
        sources = ()
    # What admission left out, where admission has a surface. At scene and
    # atmospheric there are no rows to strike (spec 4.3), so nothing is excluded.
    excluded: tuple[SourceSpec, ...] = ()
    if settings.level in ("itemized", "grouped"):
        admitted_ids = set(settings.admitted)
        excluded = tuple(_source_spec(s) for s in shot["sources"] if str(s["id"]) not in admitted_ids)
    return CaptionRequest(
        clip_id=str(clip["clip_id"]),
        shot_id=str(shot["shot_id"]),
        task=str(clip["task"]),
        level=settings.level,
        settings_key=settings.key,
        sources=sources,
        heads=heads,
        scene_prose=str(shot["scene"]["prose"]),
        atmosphere_prose=str(shot["atmosphere"]["prose"]),
        media_path=media_path,
        excluded=excluded,
    )


# ---- auditions ------------------------------------------------------------


def audition_requests(
    document: Mapping[str, Any], clip: Mapping[str, Any], shot: Mapping[str, Any], media_path: Path | None
) -> dict[str, CaptionRequest]:
    """Every audition for a shot: one per source, one per present head.

    Keys are the skeleton's row ids — the source id, or ``role:<role_id>`` for
    a head — which is how the inventory names them for the browser.
    """
    roles = roles_for(document, clip)
    requests: dict[str, CaptionRequest] = {}
    for source in shot["sources"]:
        source_id = str(source["id"])
        settings = Settings(level="itemized", admitted=(source_id,), order=(source_id,))
        requests[source_id] = build_request(document, clip, shot, settings, media_path)
    for role in heads_present(shot["sources"], roles):
        members = tuple(str(source["id"]) for source in shot["sources"] if source["role"] == role)
        settings = Settings(level="grouped", admitted=members, order=(role,))
        requests[f"role:{role}"] = build_request(document, clip, shot, settings, media_path)
    return requests


def warm_auditions(
    writer: CaptionWriter, document: Mapping[str, Any], shot_media: ShotMedia | None = None
) -> dict[tuple[str, str], dict[str, str]]:
    """Write every audition caption up front so holding a row is instant (spec 7)."""
    table: dict[tuple[str, str], dict[str, str]] = {}
    for clip in document["clips"]:
        for shot in clip["shots"]:
            media = shot_media(str(clip["clip_id"]), shot) if shot_media is not None else None
            requests = audition_requests(document, clip, shot, media)
            table[(str(clip["clip_id"]), str(shot["shot_id"]))] = {
                row_id: writer.write(request) for row_id, request in requests.items()
            }
    return table


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
    "audition_requests",
    "build_request",
    "cut_at_sentence",
    "gemma_instruction",
    "settings_key",
    "tighten",
    "visual_messages",
    "warm_auditions",
]
