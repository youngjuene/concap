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


class TestWording:
    """An item is text in every language the study has wording for."""

    def test_a_bare_string_is_read_as_english(self) -> None:
        # Items files written before the study had a second language are still
        # valid and still mean what they said.
        items = parse_items(_document())
        block = items.page_blocks("art")[0]
        assert block.record("en")["items"][0]["text"] == block.record("ko")["items"][0]["text"]

    def test_the_shipped_set_is_asked_in_both_languages(self) -> None:
        block = load_items().page_blocks("art")[0]
        assert block.record("ko")["title"] != block.record("en")["title"]
        assert block.record("ko")["items"][0]["text"] != block.record("en")["items"][0]["text"]

    def test_a_language_with_no_wording_falls_back_to_english(self) -> None:
        block = load_items().page_blocks("art")[0]
        assert block.record("fr") == block.record("en")

    def test_wording_without_english_is_refused(self) -> None:
        # English is what every other language falls back to, so a set without
        # it cannot be served to a participant whose language is missing.
        document = _document()
        document["blocks"][0]["items"][0]["text"] = {"ko": "질문"}
        with pytest.raises(ItemsError, match="en"):
            parse_items(document)

    def test_wording_that_is_neither_text_nor_a_mapping_is_refused(self) -> None:
        document = _document()
        document["blocks"][0]["items"][0]["text"] = 7
        with pytest.raises(ItemsError, match="text"):
            parse_items(document)

    def test_the_digest_covers_every_language(self) -> None:
        """The digest says which wording was asked, so a Korean edit moves it.

        Responses are comparable only across participants who answered the
        same items; an edit to the Korean is an edit to the instrument even
        for the arm that never reads it, and the seam has to be visible.
        """
        document = _document()
        before = parse_items(document).digest
        document["blocks"][0]["items"][0]["text"] = {"en": "Original.", "ko": "\ubc88\uc5ed."}
        assert parse_items(document).digest != before

    def test_the_digest_does_not_change_with_the_reader(self) -> None:
        items = load_items()
        digest = items.digest
        items.page_blocks("art")[0].record("ko")
        assert items.digest == digest
