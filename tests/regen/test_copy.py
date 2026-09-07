"""The chrome in both languages: one shape, one set of placeholders.

The Korean is a parallel tree rather than a Korean form beside every English
leaf, which keeps :mod:`dpo.regen.copy` readable but puts the burden of not
drifting here. These tests are that burden: the two trees must have the same
keys in the same places, and every string must carry the same placeholders, so
a line added in English and forgotten in Korean fails a test rather than
reaching a participant as a blank or a crash.

The survey items are deliberately not covered. §9.2 holds those in
:mod:`dpo.regen.items`; they are the ART and PRSS scales, and their Korean is a
measurement question rather than a wording one.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from dpo.regen.copy import KOREAN, STRINGS, TRANSLATIONS, strings_for

PLACEHOLDER = re.compile(r"\{(\w+)\}")


def shape(node: Any, path: str = "") -> dict[str, str]:
    """Every leaf's path mapped to its kind, so two trees can be compared."""
    if isinstance(node, dict):
        found: dict[str, str] = {}
        for key, value in node.items():
            found.update(shape(value, f"{path}.{key}" if path else key))
        return found
    if isinstance(node, list):
        return {f"{path}[{index}]": "str" for index, _ in enumerate(node)}
    return {path: type(node).__name__}


def strings(node: Any, path: str = "") -> dict[str, str]:
    if isinstance(node, dict):
        found: dict[str, str] = {}
        for key, value in node.items():
            found.update(strings(value, f"{path}.{key}" if path else key))
        return found
    if isinstance(node, list):
        return {f"{path}[{index}]": item for index, item in enumerate(node)}
    return {path: node}


class TestTheTwoTreesAgree:
    def test_korean_has_every_string_english_has(self) -> None:
        missing = sorted(set(shape(STRINGS)) - set(shape(KOREAN)))
        assert not missing, f"untranslated: {missing}"

    def test_korean_has_no_string_english_lacks(self) -> None:
        # A key only Korean has is a key nothing reads: the page indexes the
        # tree by the names the English one established.
        extra = sorted(set(shape(KOREAN)) - set(shape(STRINGS)))
        assert not extra, f"only in Korean: {extra}"

    def test_the_lists_are_the_same_length(self) -> None:
        # `steps` is indexed by position — the rail asks for step 3, not for
        # "What you saw" — so a Korean list one short renames the wrong screen.
        assert len(KOREAN["steps"]) == len(STRINGS["steps"])

    @pytest.mark.parametrize("path", sorted(strings(STRINGS)))
    def test_every_string_carries_the_same_placeholders(self, path: str) -> None:
        """A translation that drops {seconds} silently loses the number.

        Order may differ — Korean puts the total before the count in several of
        these — so the comparison is on the set, not the sequence.
        """
        english = strings(STRINGS)[path]
        korean = strings(KOREAN).get(path)
        assert korean is not None
        assert set(PLACEHOLDER.findall(english)) == set(PLACEHOLDER.findall(korean)), path


class TestWhatIsNotTranslated:
    def test_each_language_is_named_in_its_own_language(self) -> None:
        # The point of the toggle: a participant who cannot read the other
        # language can still find theirs, so these two are not translated.
        assert KOREAN["languages"]["names"] == STRINGS["languages"]["names"]

    def test_the_korean_tree_is_actually_korean(self) -> None:
        # Guards against a tree copied from English and left to be filled in.
        hangul = re.compile(r"[가-힣]")
        english = strings(STRINGS)
        untouched = [
            path
            for path, text in strings(KOREAN).items()
            if text == english.get(path) and not hangul.search(str(text)) and str(text).strip("—{}")
        ]
        # `names` is deliberately shared, and `tally` is a bare placeholder.
        assert set(untouched) <= {
            "languages.names.en",
            "languages.names.ko",
            "visual.tally",
            "visual.tally_none",
        }, untouched


class TestResolving:
    def test_english_is_returned_as_it_is(self) -> None:
        assert strings_for("en") is STRINGS

    def test_a_language_nobody_wrote_falls_back_to_english(self) -> None:
        assert strings_for("fr") is STRINGS

    def test_korean_resolves_to_korean(self) -> None:
        assert strings_for("ko")["app_title"] == KOREAN["app_title"]
        assert strings_for("ko")["steps"] == KOREAN["steps"]

    def test_a_gap_in_a_translation_shows_that_line_in_english(self) -> None:
        """Per string, not per tree.

        A language missing one line should show that line in English and the
        rest in its own; reverting the whole interface over one gap gives the
        participant less of what they can read, not more.
        """
        partial = {"app_title": "부분", "done": {"heading": "끝"}}
        try:
            TRANSLATIONS["xx"] = partial
            resolved = strings_for("xx")
        finally:
            TRANSLATIONS.pop("xx", None)
        assert resolved["app_title"] == "부분"
        assert resolved["done"]["heading"] == "끝"
        assert resolved["done"]["body"] == STRINGS["done"]["body"]
        assert resolved["actions"] == STRINGS["actions"]

    def test_resolving_does_not_mutate_the_english_tree(self) -> None:
        before = strings(STRINGS)
        strings_for("ko")
        assert strings(STRINGS) == before
