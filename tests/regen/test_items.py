"""§9.2: items as data, and the ART block referenced rather than copied."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpo.regen.items import ITEMS_SCHEMA, ItemsError, load_items, parse_items


def _document(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": ITEMS_SCHEMA,
        "provenance": "authored",
        "blocks": [
            {"id": "art", "title": "Scene", "items": [{"id": "art_1", "text": "a"}]},
            {"id": "caption", "title": "Captions", "items": [{"id": "cap_1", "text": "b"}]},
            {"id": "prss", "title": "Sounds", "items": [{"id": "prss_1", "text": "c"}]},
        ],
    }
    return {**base, **overrides}


class TestShippedSet:
    def test_the_default_set_loads_and_says_it_is_a_placeholder(self) -> None:
        items = load_items()
        assert items.is_placeholder
        assert items.provenance == "placeholder"

    def test_it_defines_every_block_the_pages_ask_for(self) -> None:
        items = load_items()
        assert set(items.blocks) >= {"art", "caption", "prss"}


class TestOneArtBlock:
    def test_both_survey_pages_ask_the_identical_art_items(self) -> None:
        # The study's main measure is the change between §3 and §8 on these
        # items. Two copies that drifted would look like an effect.
        items = load_items()
        first = items.page_blocks("art")[0]
        second = next(block for block in items.page_blocks("survey") if block.id == "art")
        assert first is second

    def test_the_survey_page_asks_art_first_then_the_caption_and_prss_blocks(self) -> None:
        assert [block.id for block in load_items().page_blocks("survey")] == ["art", "caption", "prss"]

    def test_item_ids_are_the_keys_a_submission_must_fill(self) -> None:
        items = load_items()
        assert set(items.item_ids("art")) <= set(items.item_ids("survey"))


class TestDigest:
    def test_the_same_wording_digests_the_same_way(self) -> None:
        assert parse_items(_document()).digest == parse_items(_document()).digest

    def test_edited_wording_changes_the_digest(self) -> None:
        edited = _document()
        edited["blocks"][0]["items"][0]["text"] = "a different question"
        assert parse_items(edited).digest != parse_items(_document()).digest

    def test_placeholder_and_authored_sets_never_share_a_digest(self) -> None:
        assert parse_items(_document(provenance="placeholder")).digest != parse_items(_document()).digest


class TestRefusals:
    def test_a_missing_provenance_is_refused(self) -> None:
        with pytest.raises(ItemsError, match="provenance"):
            parse_items(_document(provenance="pilot"))

    def test_a_document_missing_a_block_a_page_asks_is_refused(self) -> None:
        document = _document()
        document["blocks"] = document["blocks"][:1]
        with pytest.raises(ItemsError, match="prss"):
            parse_items(document)

    def test_a_repeated_item_id_is_refused(self) -> None:
        document = _document()
        document["blocks"][1]["items"][0]["id"] = "art_1"
        with pytest.raises(ItemsError, match="used by another item"):
            parse_items(document)

    def test_a_page_that_does_not_exist_is_refused(self) -> None:
        with pytest.raises(ItemsError, match="no page named"):
            parse_items(_document()).page_blocks("debrief")


class TestFromDisk:
    def test_a_study_can_supply_its_own_file(self, tmp_path: Path) -> None:
        path = tmp_path / "items.json"
        path.write_text(json.dumps(_document()), encoding="utf-8")
        assert load_items(path).provenance == "authored"

    def test_a_file_that_is_not_json_names_itself(self, tmp_path: Path) -> None:
        path = tmp_path / "items.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(ItemsError, match="not JSON"):
            load_items(path)
