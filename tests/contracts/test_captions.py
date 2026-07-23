"""Caption compliance and cross-modal lexicon screens."""

from __future__ import annotations

from dpo.contracts.audio_caption import check_audio_caption
from dpo.contracts.captions import is_single_sentence, word_count
from dpo.contracts.study_contract import StudyContract
from dpo.contracts.visual_caption import check_visual_caption


def test_word_count_and_sentence_rules() -> None:
    assert word_count("A cyclist crosses the road.") == 5
    assert is_single_sentence("A cyclist crosses the road.")
    assert not is_single_sentence("Two sentences. Right here.")
    assert not is_single_sentence("A line\nbreak.")
    assert is_single_sentence("No terminal punctuation at all")


def test_visual_captions_flag_sound_claims(contract: StudyContract) -> None:
    visual = contract.tracks["visual"]
    report = check_visual_caption("A dog is barking near the gate.", visual)
    assert "barking" in report.modality_flags
    assert not report.compliant
    clean = check_visual_caption("A dog runs along the gate toward the yard.", visual)
    assert clean.compliant


def test_audio_captions_flag_visual_claims(contract: StudyContract) -> None:
    audio = contract.tracks["audio"]
    report = check_audio_caption("A red car appears while an engine hums.", audio)
    assert "red" in report.modality_flags
    clean = check_audio_caption("An engine hums steadily and then fades out.", audio)
    assert clean.compliant


def test_length_bounds_come_from_the_contract(contract: StudyContract) -> None:
    visual = contract.tracks["visual"]
    too_short = check_visual_caption("Too short.", visual)
    assert not too_short.within_length
    words = " ".join(["word"] * (visual.max_words + 1)) + "."
    too_long = check_visual_caption(words, visual)
    assert not too_long.within_length
