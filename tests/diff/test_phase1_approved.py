"""The researcher's approved second captions: the sheet's import, its checks, and the participant gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from phase1_fixtures import approved_setup

import diff.workbook_instrument as workbook_instrument
from diff.phase1_approved import (
    EDITED,
    SHEET_COLUMNS,
    WRITTEN,
    approval_problems,
    approved_caption,
    import_sheet,
    reachable,
)
from diff.phase1_setup import setup_problems

COLUMNS = (*SHEET_COLUMNS, "draft:B-words")


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Any]:
    return approved_setup(tmp_path)


def sheet(setup: dict[str, Any]) -> list[dict[str, str]]:
    """Every reachable caption, approved as B's draft."""
    return [
        {
            "variant_id": key,
            "clip": variant["clip"],
            "mode": variant["slots"]["mode"],
            "draft:B-words": f"{variant['slots']['attended_sound']}가 들린다.",
            "approved_text": f"{variant['slots']['attended_sound']}가 들린다.",
            "approved_from": "B-words",
            "note": "",
        }
        for key, variant in reachable(setup).items()
    ]


def imported(
    setup: dict[str, Any], rows: list[dict[str, str]]
) -> tuple[dict[str, Any], list[str], list[str]]:
    return import_sheet(setup, rows, COLUMNS, reviewed_by="researcher", at="2026-10-24T23:00:00+0900")


def test_a_full_sheet_approves_every_caption_and_opens_the_gate(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    assert len(reachable(setup)) == 2 * sum(len(clip["options"]["A1s"]) for clip in setup["clips"])
    assert approval_problems(setup) == [
        "second captions: none approved; run `python -m diff phase1 import-second-captions`"
    ]
    updated, problems, unapproved = imported(setup, sheet(setup))
    assert problems == [] and unapproved == []
    assert approval_problems(updated) == []
    key, variant = next(iter(reachable(updated).items()))
    caption = approved_caption(updated, variant["slots"])
    assert caption is not None and caption["text"].endswith("가 들린다.")
    assert (caption["approved_from"], caption["reviewed_by"]) == ("B-words", "researcher")
    # Clips and practice approved, captions approved: only the provisional model's gates remain.
    assert setup_problems(updated, "participant", root=tmp_path, instrument=workbook_instrument) == []


def test_unapproved_rows_and_hand_edits_keep_participants_out(setup: dict[str, Any], tmp_path: Path) -> None:
    rows = sheet(setup)
    rows[0].update(approved_text="", approved_from="")
    updated, problems, unapproved = imported(setup, rows)
    assert problems == [] and len(unapproved) == 1
    assert approval_problems(updated) == [f"{rows[0]['clip']}: 1 of 4 second captions not approved"]
    gate = setup_problems(updated, "participant", root=tmp_path, instrument=workbook_instrument)
    assert gate == approval_problems(updated)
    complete, _, _ = imported(setup, sheet(setup))
    complete["second_captions"]["captions"][rows[1]["variant_id"]]["text"] = "손으로 고친 자막."
    assert "edited since they were imported" in approval_problems(complete)[0]


def test_a_sheet_that_does_not_hold_is_refused_whole(setup: dict[str, Any]) -> None:
    rows = sheet(setup)
    rows[0].update(approved_text="가" * 81, approved_from=EDITED)
    rows[1].update(approved_text="…!", approved_from=WRITTEN)
    rows[2]["approved_text"] = "고쳐 쓴 자막이 들린다."  # B's draft changed, but still called B's
    rows[3]["approved_from"] = "C-words"
    rows[4]["mode"] = "acousmatic" if rows[4]["mode"] == "scene" else "scene"
    stale = {**rows[5], "variant_id": "000000000000"}
    rows[6]["approved_text"], rows[6]["approved_from"] = "", EDITED
    _, problems, _ = imported(setup, [*rows, rows[7], stale])
    assert [problem.split(": ", 1)[1] for problem in problems] == [
        "the approved text has 81 characters, over 80",
        "the approved text has nothing readable",
        f"the text is not B-words's draft as written; say {EDITED!r}",
        f"approved_from must be one of ['B-words', {EDITED!r}, {WRITTEN!r}]",
        "its clip or mode does not match the variant",
        "approved_from without an approved_text",
        "the variant appears twice",
        "not a caption this setup can show (redraft after a setup change)",
    ]
    assert imported(setup, [])[1] == []
    assert import_sheet(setup, [], SHEET_COLUMNS[:-1], reviewed_by="r", at="t")[1] == [
        "the sheet has no column ['note']"
    ]
    assert import_sheet(setup, [], COLUMNS, reviewed_by=" ", at="t")[1] == ["name the reviewer"]
