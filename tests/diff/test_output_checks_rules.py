"""Output checks fail, flag and measure; the cut at the first terminator never shortens text."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fake_dream_model import ENDOFTEXT, IM_END, MASK, FakeTokenizer

from diff.caption_sources import CaptionSource, SourceSet
from diff.output_checks import (
    GeneratedOutput,
    check_output,
    ending_class,
    extract_output,
    non_hangul_runs,
    output_from_record,
    repeated_word_ngrams,
    script_passes,
    sentence_count,
)
from diff.study_protocol import TERMINATOR_IDS

TOKENIZER = FakeTokenizer()
SPECIAL = frozenset(TOKENIZER.added_tokens_decoder)


def output(text: str, *, terminator: str | None = "endoftext", masks: int = 0) -> GeneratedOutput:
    return GeneratedOutput(
        raw_text=text,
        text=text.strip(),
        terminator=terminator,
        terminator_index=len(text) if terminator else None,
        tail_token_ids=(),
        tail_text="",
        leftover_mask_count=masks,
        special_ids_in_text=(),
        new_token_count=128,
    )


def codes(result: object, kind: str = "failures") -> set[str]:
    return {finding.code for finding in getattr(result, kind)}


@pytest.fixture
def bus(development_sources: SourceSet) -> CaptionSource:
    return development_sources.phase(1)[0]  # 버스가 오른쪽 차로로 지나간다.


def test_cut_at_the_first_terminator_keeps_the_tail() -> None:
    prompt = [1, 2, 3]
    new = [*TOKENIZER.encode("버스"), IM_END, *TOKENIZER.encode("꼬리"), ENDOFTEXT]
    result = extract_output([*prompt, *new], 3, TOKENIZER.decode, SPECIAL, TERMINATOR_IDS, mask_id=MASK)
    assert result.text == "버스" and result.terminator == "im_end" and result.terminator_index == 2
    assert result.tail_text.startswith("꼬리") and result.completion == "terminated"
    assert output_from_record(result.record()) == result


def test_no_terminator_is_incomplete_and_masks_are_counted() -> None:
    new = [*TOKENIZER.encode("버스"), MASK, MASK]
    result = extract_output(new, 0, TOKENIZER.decode, SPECIAL, TERMINATOR_IDS, mask_id=MASK)
    assert result.completion == "incomplete" and result.leftover_mask_count == 2
    assert result.special_ids_in_text == (MASK, MASK)


def test_endoftext_before_im_end_wins() -> None:
    new = [*TOKENIZER.encode("가"), ENDOFTEXT, IM_END]
    assert extract_output(new, 0, TOKENIZER.decode, SPECIAL, TERMINATOR_IDS).terminator == "endoftext"


def test_a_clean_rewrite_passes(bus: CaptionSource) -> None:
    result = check_output(bus, output("큰 버스가 오른쪽 차로로 천천히 지나간다."))
    assert not result.failures, result.record()
    assert result.measures["specific_source_count"] == 1 and result.measures["quality_term_count"] == 1


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("버스가 지나간다. 경적이 울린다.", "multiple_sentences"),
        ("버스が 지나간다.", "non_korean_script"),
        ("公共汽车가 지나간다.", "non_korean_script"),
        ("bus가 지나간다.", "non_korean_script"),
        ("버스가 오른쪽 차로로 지나감", "sentence_ending_mismatch"),
        ("가" * 97, "length_over_limit"),
        ("버스가\n지나가고\n또 지나간다.", "too_many_lines"),
    ],
)
def test_fail_rules(bus: CaptionSource, text: str, code: str) -> None:
    assert code in codes(check_output(bus, output(text)))


def test_incomplete_and_leftover_masks_fail(bus: CaptionSource) -> None:
    assert "incomplete" in codes(check_output(bus, output("버스가 지나간다.", terminator=None)))
    assert "leftover_mask" in codes(check_output(bus, output("버스가 지나간다.", masks=1)))


def test_a_latin_word_from_the_evidence_is_allowed(bus: CaptionSource) -> None:
    event = replace(bus.events[0], source="KTX")
    source = replace(bus, events=(event,))
    assert "non_korean_script" not in codes(check_output(source, output("KTX가 지나간다.")))


def test_flags_route_to_review(bus: CaptionSource) -> None:
    copied = check_output(bus, output(bus.text))
    assert "identical_to_source" in codes(copied, "flags") and copied.measures["exact_copy_of_source"]
    repeated = check_output(bus, output("버스가 지나간다 버스가 지나간다 버스가 지나간다."))
    assert "repetition" in codes(repeated, "flags")
    outside = check_output(bus, output("날카로운 버스가 지나간다."))
    assert "quality_outside_evidence" in codes(outside, "flags")
    added = check_output(bus, output("버스와 새가 지나간다."), lexicon=["버스", "새"])
    assert "sound_term_absent_from_evidence" in codes(added, "flags")


def test_phase2_cell_flags(development_sources: SourceSet) -> None:
    source = development_sources.phase(2)[1]  # 오토바이, loud/rough/fading, 왼쪽 차로, 버스 뒤
    broad = {"quality_limit": 0, "context": "broad"}
    result = check_output(source, output("큰 오토바이가 왼쪽 차로로 지나간다."), cell=broad)
    assert {"quality_count_over_limit", "source_name_at_broad", "relation_below_relation_step"} <= codes(
        result, "flags"
    )
    with pytest.raises(ValueError, match="cell"):
        check_output(source, output("소리"))


def test_a_split_multibyte_character_fails(bus: CaptionSource) -> None:
    result = check_output(bus, output("\ufffd리서 버스가 지나간다."))
    assert "undecodable_bytes" in codes(result)
    assert not script_passes(result)


def test_g0_script_check_counts_incomplete_as_failure(bus: CaptionSource) -> None:
    assert script_passes(check_output(bus, output("버스가 지나간다.")))
    assert not script_passes(check_output(bus, output("버스가 지나간다.", terminator=None)))
    assert not script_passes(check_output(bus, output("バスが通る。")))


def test_helpers() -> None:
    assert non_hangul_runs("KTX가 3.5초 지난다") == ["KTX"]
    assert sentence_count("3.5초 동안 들린다.") == 1
    assert ending_class("버스가 지나간다.") == ("da", "period")
    assert ending_class("바람 소리") == ("noun", "none")
    assert ending_class("들림") == ("nominal_m", "none")
    assert repeated_word_ngrams("a b c a b c") == ["a b c"]
