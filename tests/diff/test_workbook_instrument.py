"""diff's copy of the questionnaire: regen's instrument over the spreadsheet as of 2026-10-18."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import diff.workbook_instrument as workbook_instrument
import dpo.regen.sheet_instrument as regen_instrument

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = {"A1s": [{"value": "s1", "label": "버스 소리"}], "R2": [{"value": "m1", "label": "버스 소리"}]}
R1_NOW = "영상을 보는 동안 자막을 어디까지 읽으셨습니까?"
R1_REGEN = "영상을 보는 동안 자막을 얼마나 읽으셨습니까?"


def test_the_snapshot_is_regen_s_with_only_r1_and_the_read_date_changed() -> None:
    ours = json.loads((ROOT / "src/diff/items/workbook-20261018.json").read_text(encoding="utf-8"))
    regen = json.loads((ROOT / "src/dpo/regen/items/sheet-workbook.json").read_text(encoding="utf-8"))
    assert (ours["read_date"], regen["read_date"]) == ("2026-10-18", "2026-10-04")
    changed = [
        (sheet["properties"]["title"], row, column, before, after)
        for sheet, old in zip(ours["sheets"], regen["sheets"], strict=True)
        for row, (new_cells, old_cells) in enumerate(zip(sheet["values"], old["values"], strict=True))
        for column, (after, before) in enumerate(zip(new_cells, old_cells, strict=True))
        if after != before
    ]
    assert changed == [("문항표", 39, 9, R1_REGEN, R1_NOW)]
    assert {key: value for key, value in ours.items() if key not in ("read_date", "sheets")} == {
        key: value for key, value in regen.items() if key not in ("read_date", "sheets")
    }


@pytest.mark.parametrize("page", sorted(workbook_instrument.PAGE_KINDS))
def test_every_page_is_regen_s_but_for_r1(page: str) -> None:
    ours = workbook_instrument.page_spec(page, options=OPTIONS)
    regen = regen_instrument.page_spec(page, options=OPTIONS)
    for item in regen["items"]:
        if item["id"] == "R1":
            assert item["text"] == R1_REGEN
            item["text"] = R1_NOW
        item.pop("source_row_hash", None)
        item.get("source_cells", {}).pop("한국어 (수정안)", None)
    for item in ours["items"]:
        item.pop("source_row_hash", None)
        item.get("source_cells", {}).pop("한국어 (수정안)", None)
    assert ours == regen


def test_answers_validate_as_in_regen() -> None:
    for page, answers in (
        ("P4", {item["id"]: 3 for item in workbook_instrument.page_spec("P4")["items"]}),
        ("P5", {"FA1": "unsure", "FA2": 0, "FA3": 6}),
        ("P11", {"R1": 6, "R2": ["m1"]}),
    ):
        assert workbook_instrument.validate_answers(page, answers, options=OPTIONS) == (
            regen_instrument.validate_answers(page, answers, options=OPTIONS)
        )
    with pytest.raises(ValueError, match="exclusive"):
        workbook_instrument.validate_answers(
            "P11", {"R1": 6, "R2": ["m1", "memory_unknown"]}, options=OPTIONS
        )


def test_the_snapshot_names_its_own_provenance() -> None:
    snapshot = workbook_instrument.snapshot()
    assert snapshot["schema"] == "diff.workbook-instrument/v1"
    assert snapshot["provenance"].endswith("-2026-10-18")
    assert snapshot["hash"] != regen_instrument.snapshot()["hash"]
