"""Annotator reliability, quality controls, and preregistered exclusions.

Exclusion rules are parameters of the study contract and are applied to raw
annotations BEFORE aggregation. Excluded annotations are never deleted from
the raw store — the report lists them, and derived views simply do not read
them. That preserves the append-only property while making the exclusion
decision reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dpo.annotation.raw_annotations import AnnotationError, RawAnnotation


@dataclass(frozen=True)
class AnnotatorReport:
    annotator_id_hash: str
    judgment_count: int
    attention_check_count: int
    attention_pass_rate: float
    repeat_count: int
    repeat_consistency: float
    left_choice_rate: float
    position_bias: float
    fast_response_count: int
    excluded: bool
    exclusion_reasons: tuple[str, ...]

    def document(self) -> dict[str, object]:
        return {
            "annotator_id_hash": self.annotator_id_hash,
            "judgment_count": self.judgment_count,
            "attention_check_count": self.attention_check_count,
            "attention_pass_rate": self.attention_pass_rate,
            "repeat_count": self.repeat_count,
            "repeat_consistency": self.repeat_consistency,
            "left_choice_rate": self.left_choice_rate,
            "position_bias": self.position_bias,
            "fast_response_count": self.fast_response_count,
            "excluded": self.excluded,
            "exclusion_reasons": list(self.exclusion_reasons),
        }


@dataclass(frozen=True)
class ReliabilityReport:
    annotators: tuple[AnnotatorReport, ...]
    excluded_annotation_ids: tuple[str, ...]
    retained_count: int
    excluded_count: int

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.reliability-report/v1",
            "annotators": [report.document() for report in self.annotators],
            "excluded_annotation_ids": list(self.excluded_annotation_ids),
            "retained_count": self.retained_count,
            "excluded_count": self.excluded_count,
        }


def _repeat_consistency(
    annotations: Sequence[RawAnnotation], primary_by_id: dict[str, RawAnnotation]
) -> tuple[int, float]:
    """How often an annotator's repeat judgment matches their own primary one."""
    matches = 0
    repeats = 0
    for annotation in annotations:
        if annotation.repeat_of is None:
            continue
        primary = primary_by_id.get(annotation.repeat_of)
        if primary is None:
            raise AnnotationError(
                f"repeat annotation {annotation.annotation_id!r} references unknown primary"
                f" {annotation.repeat_of!r}"
            )
        if primary.annotator_id_hash != annotation.annotator_id_hash:
            raise AnnotationError(
                f"repeat annotation {annotation.annotation_id!r} crosses annotators;"
                " repeats measure within-annotator consistency"
            )
        repeats += 1
        if primary.canonical_choice() == annotation.canonical_choice():
            matches += 1
    return repeats, (matches / repeats if repeats else 1.0)


def build_reliability_report(
    annotations: Sequence[RawAnnotation],
    *,
    attention_expected: dict[str, str],
    min_response_ms: int,
    max_position_bias: float,
    min_attention_pass: float,
) -> ReliabilityReport:
    """Score annotators and apply the preregistered exclusion rules.

    ``attention_expected`` maps attention-check pair ids to the displayed
    choice a diligent annotator must make.
    """
    primary_by_id: dict[str, RawAnnotation] = {
        annotation.annotation_id: annotation
        for annotation in annotations
        if annotation.repeat_of is None and not annotation.is_attention_check
    }
    by_annotator: dict[str, list[RawAnnotation]] = {}
    for annotation in annotations:
        by_annotator.setdefault(annotation.annotator_id_hash, []).append(annotation)
    reports = []
    excluded_ids: list[str] = []
    for annotator in sorted(by_annotator):
        rows = by_annotator[annotator]
        attention_rows = [row for row in rows if row.is_attention_check]
        attention_passes = 0
        for row in attention_rows:
            expected = attention_expected.get(row.pair_id)
            if expected is None:
                raise AnnotationError(f"attention check {row.pair_id!r} has no expected outcome registered")
            if row.choice == expected:
                attention_passes += 1
        attention_pass_rate = attention_passes / len(attention_rows) if attention_rows else 1.0
        repeat_count, repeat_consistency = _repeat_consistency(rows, primary_by_id)
        decisive = [row for row in rows if row.choice in {"a_better", "b_better"}]
        left_choices = sum(1 for row in decisive if row.choice == "a_better")
        left_choice_rate = left_choices / len(decisive) if decisive else 0.5
        position_bias = abs(left_choice_rate - 0.5)
        fast = sum(1 for row in rows if row.response_time_ms < min_response_ms)
        reasons: list[str] = []
        if attention_rows and attention_pass_rate < min_attention_pass:
            reasons.append("attention_check_failure")
        if len(decisive) >= 10 and position_bias > max_position_bias:
            reasons.append("position_bias")
        if fast and fast >= max(1, len(rows) // 2):
            reasons.append("response_time")
        excluded = bool(reasons)
        if excluded:
            excluded_ids.extend(sorted(row.annotation_id for row in rows))
        reports.append(
            AnnotatorReport(
                annotator_id_hash=annotator,
                judgment_count=len(rows),
                attention_check_count=len(attention_rows),
                attention_pass_rate=attention_pass_rate,
                repeat_count=repeat_count,
                repeat_consistency=repeat_consistency,
                left_choice_rate=left_choice_rate,
                position_bias=position_bias,
                fast_response_count=fast,
                excluded=excluded,
                exclusion_reasons=tuple(reasons),
            )
        )
    excluded_set = set(excluded_ids)
    retained = sum(1 for annotation in annotations if annotation.annotation_id not in excluded_set)
    return ReliabilityReport(
        annotators=tuple(reports),
        excluded_annotation_ids=tuple(sorted(excluded_set)),
        retained_count=retained,
        excluded_count=len(excluded_set),
    )


def retained_annotations(
    annotations: Sequence[RawAnnotation], report: ReliabilityReport
) -> tuple[RawAnnotation, ...]:
    excluded = set(report.excluded_annotation_ids)
    return tuple(annotation for annotation in annotations if annotation.annotation_id not in excluded)
