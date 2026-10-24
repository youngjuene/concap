"""Phase 1's second captions as the researcher approved them, one per caption a participant can reach.

Researcher's decisions of 2026-10-24: every second caption is reviewed before the study, and a
participant session needs every caption it could show approved. A caption's prompt depends on the
participant only through the clip's mode and the attended sound (``diff.phase1-tweak/v5``), so each
clip has one per sound in each mode (:func:`diff.phase1_tweak.reachable_slots`), keyed by a hash of
its slots (:func:`diff.phase1_tweak.variant_id`).

The researcher fills the review sheet that ``scripts/diff_second_captions.py`` writes, and
``python -m diff phase1 import-second-captions`` checks it (:func:`import_sheet`) and writes the
approved texts into the setup's ``second_captions``, replacing any earlier import:
- each row's variant must be one the setup can reach: a changed setup changes the IDs;
- an approved text must have something readable and fit :data:`TWEAK_MAX_CHARS`;
- ``approved_from`` names the draft taken as it is (whose text it must then be), or
  :data:`EDITED` or :data:`WRITTEN`.
A row left without ``approved_text`` stays unapproved. The import records who reviewed and when,
and a hash of the captions, so a later hand edit is refused as an edited clip is
(:func:`approval_problems`). Phase 1 shows the approved caption without calling Trida
(:mod:`diff.phase1_flow`); a smoke session falls back to Trida where none is approved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from diff.output_checks import readable
from diff.phase1_tweak import reachable_slots, variant_id
from diff.study_protocol import TWEAK_MAX_CHARS, sha256_of

CAPTIONS_SCHEMA: Final = "diff.phase1-second-captions/v1"
EDITED, WRITTEN = "edited", "written"
DRAFT_PREFIX: Final = "draft:"
SHEET_COLUMNS: Final = ("variant_id", "clip", "mode", "approved_text", "approved_from", "note")


def reachable(setup: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every second caption the setup's clips can show: variant ID → clip key and slots."""
    return {
        variant_id(slots): {"clip": clip["key"], "slots": slots}
        for clip in setup["clips"]
        for slots in reachable_slots(clip)
    }


def import_sheet(
    setup: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    *,
    reviewed_by: str,
    at: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """The setup with the sheet's approved captions, the problems that refuse the import, and each
    reachable caption the sheet leaves unapproved."""
    missing = [name for name in SHEET_COLUMNS if name not in columns]
    if missing:
        return dict(setup), [f"the sheet has no column {missing}"], []
    if not reviewed_by.strip():
        return dict(setup), ["name the reviewer"], []
    drafts = [name.removeprefix(DRAFT_PREFIX) for name in columns if name.startswith(DRAFT_PREFIX)]
    known = reachable(setup)
    problems: list[str] = []
    captions: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for number, row in enumerate(rows, 2):  # spreadsheet rows, after the header
        key = row["variant_id"].strip()
        where = f"row {number} ({key})"
        if key in seen:
            problems.append(f"{where}: the variant appears twice")
        seen.add(key)
        variant = known.get(key)
        if variant is None:
            problems.append(f"{where}: not a caption this setup can show (redraft after a setup change)")
            continue
        if (row["clip"], row["mode"]) != (variant["clip"], variant["slots"]["mode"]):
            problems.append(f"{where}: its clip or mode does not match the variant")
        text = row["approved_text"].strip()
        source = row["approved_from"].strip()
        if not text:
            if source:
                problems.append(f"{where}: approved_from without an approved_text")
            continue
        if not readable(text):
            problems.append(f"{where}: the approved text has nothing readable")
        if len(text) > TWEAK_MAX_CHARS:
            problems.append(f"{where}: the approved text has {len(text)} characters, over {TWEAK_MAX_CHARS}")
        if source in drafts:
            if text != row[f"{DRAFT_PREFIX}{source}"].strip():
                problems.append(f"{where}: the text is not {source}'s draft as written; say {EDITED!r}")
        elif source not in (EDITED, WRITTEN):
            problems.append(f"{where}: approved_from must be one of {[*drafts, EDITED, WRITTEN]}")
        captions[key] = {
            "clip": variant["clip"],
            "mode": variant["slots"]["mode"],
            "attended_sound": variant["slots"]["attended_sound"],
            "text": text,
            "approved_from": source,
            "note": row["note"].strip(),
        }
    unapproved = [
        f"{v['clip']} {v['slots']['mode']} {v['slots']['attended_sound']}"
        for k, v in known.items()
        if k not in captions
    ]
    updated = dict(setup)
    updated["second_captions"] = {
        "schema": CAPTIONS_SCHEMA,
        "reviewed_by": reviewed_by.strip(),
        "reviewed_at": at,
        "captions": captions,
        "sha256": sha256_of(captions),
    }
    return updated, problems, unapproved


def approved_caption(setup: Mapping[str, Any], slots: Mapping[str, str]) -> dict[str, Any] | None:
    """The approved caption for these slots, with who reviewed it and when; ``None`` if none."""
    block = setup.get("second_captions") or {}
    caption = block.get("captions", {}).get(variant_id(slots))
    if caption is None:
        return None
    return {**caption, "reviewed_by": block["reviewed_by"], "reviewed_at": block["reviewed_at"]}


def approval_problems(setup: Mapping[str, Any]) -> list[str]:
    """Why participant sessions cannot show only approved second captions; empty when they can."""
    block = setup.get("second_captions")
    run = "run `python -m diff phase1 import-second-captions`"
    if not block:
        return [f"second captions: none approved; {run}"]
    if block.get("schema") != CAPTIONS_SCHEMA or block.get("sha256") != sha256_of(block.get("captions")):
        return [f"second captions: edited since they were imported; {run} again"]
    gaps: dict[str, int] = {}
    totals: dict[str, int] = {}
    for key, variant in reachable(setup).items():
        totals[variant["clip"]] = totals.get(variant["clip"], 0) + 1
        if key not in block["captions"]:
            gaps[variant["clip"]] = gaps.get(variant["clip"], 0) + 1
    return [f"{clip}: {count} of {totals[clip]} second captions not approved" for clip, count in gaps.items()]
