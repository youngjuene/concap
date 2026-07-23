"""Shared automated checks over generated captions (PRD section 26.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dpo.contracts.audio_caption import check_audio_caption
from dpo.contracts.captions import caption_words
from dpo.contracts.study_contract import CaptionContract
from dpo.contracts.visual_caption import check_visual_caption


@dataclass(frozen=True)
class GeneratedCaption:
    clip_id: str
    text: str


@dataclass(frozen=True)
class ComplianceMetrics:
    caption_count: int
    empty_rate: float
    mean_word_count: float
    length_violation_rate: float
    single_sentence_rate: float
    modality_violation_rate: float
    duplicate_output_rate: float
    repetition_rate: float
    distinct_2: float
    grammar: str

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.compliance-metrics/v1",
            "caption_count": self.caption_count,
            "empty_rate": self.empty_rate,
            "mean_word_count": self.mean_word_count,
            "length_violation_rate": self.length_violation_rate,
            "single_sentence_rate": self.single_sentence_rate,
            "modality_violation_rate": self.modality_violation_rate,
            "duplicate_output_rate": self.duplicate_output_rate,
            "repetition_rate": self.repetition_rate,
            "distinct_2": self.distinct_2,
            "grammar": self.grammar,
        }


def _bigrams(tokens: Sequence[str]) -> list[tuple[str, str]]:
    return [(tokens[index], tokens[index + 1]) for index in range(len(tokens) - 1)]


def compute_compliance(captions: Sequence[GeneratedCaption], contract: CaptionContract) -> ComplianceMetrics:
    if not captions:
        raise ValueError("compliance metrics require at least one caption")
    check = check_visual_caption if contract.track == "visual" else check_audio_caption
    empty = 0
    length_violations = 0
    single_sentences = 0
    modality_violations = 0
    repetitions = 0
    word_counts = []
    all_bigrams: list[tuple[str, str]] = []
    texts_seen: dict[str, int] = {}
    for caption in captions:
        report = check(caption.text, contract)
        if report.empty:
            empty += 1
            continue
        word_counts.append(report.word_count)
        if not report.within_length:
            length_violations += 1
        if report.single_sentence:
            single_sentences += 1
        if report.modality_flags:
            modality_violations += 1
        tokens = [token.casefold() for token in caption_words(caption.text)]
        bigrams = _bigrams(tokens)
        all_bigrams.extend(bigrams)
        if bigrams and len(set(bigrams)) < len(bigrams):
            repetitions += 1
        texts_seen[caption.text.strip().casefold()] = texts_seen.get(caption.text.strip().casefold(), 0) + 1
    total = len(captions)
    duplicates = sum(count - 1 for count in texts_seen.values() if count > 1)
    return ComplianceMetrics(
        caption_count=total,
        empty_rate=empty / total,
        mean_word_count=sum(word_counts) / len(word_counts) if word_counts else 0.0,
        length_violation_rate=length_violations / total,
        single_sentence_rate=single_sentences / total,
        modality_violation_rate=modality_violations / total,
        duplicate_output_rate=duplicates / total,
        repetition_rate=repetitions / total,
        distinct_2=(len(set(all_bigrams)) / len(all_bigrams)) if all_bigrams else 0.0,
        grammar="blocked_pending_external_operation",
    )
