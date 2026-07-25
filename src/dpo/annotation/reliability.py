"""Annotator reliability, quality controls, and preregistered exclusions.

Exclusion rules are parameters of the study contract and are applied to raw
annotations BEFORE aggregation. Excluded annotations are never deleted from
the raw store — the report lists them, and derived views simply do not read
them. That preserves the append-only property while making the exclusion
decision reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


def parse_reliability_report(payload: bytes | str | Mapping[str, Any]) -> ReliabilityReport:
    """Rebuild a published screening decision; the inverse of ``document()``.

    A later stage must retain exactly the annotations this report excluded, and
    the exclusion rules read attention-check expectations that no published
    artifact carries. Reusing the decision is therefore the only way to keep the
    derived views identical to the ones the annotation stage implied.
    """
    if isinstance(payload, (bytes, str)):
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AnnotationError("reliability report payload is not valid JSON") from exc
    else:
        document = dict(payload)
    expected = {"schema", "annotators", "excluded_annotation_ids", "retained_count", "excluded_count"}
    if not isinstance(document, dict) or set(document) != expected:
        raise AnnotationError("reliability report payload has an unexpected field set")
    if document["schema"] != "dpo.reliability-report/v1":
        raise AnnotationError("reliability report schema is unknown")
    rows = document["annotators"]
    excluded_ids = document["excluded_annotation_ids"]
    if not isinstance(rows, list) or not isinstance(excluded_ids, list):
        raise AnnotationError("reliability report annotators/exclusions must be arrays")
    annotators = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AnnotationError("reliability report annotator rows must be objects")
        reasons = row["exclusion_reasons"]
        if not isinstance(reasons, list):
            raise AnnotationError("reliability report exclusion_reasons must be an array")
        annotators.append(
            AnnotatorReport(
                annotator_id_hash=str(row["annotator_id_hash"]),
                judgment_count=int(row["judgment_count"]),
                attention_check_count=int(row["attention_check_count"]),
                attention_pass_rate=float(row["attention_pass_rate"]),
                repeat_count=int(row["repeat_count"]),
                repeat_consistency=float(row["repeat_consistency"]),
                left_choice_rate=float(row["left_choice_rate"]),
                position_bias=float(row["position_bias"]),
                fast_response_count=int(row["fast_response_count"]),
                excluded=bool(row["excluded"]),
                exclusion_reasons=tuple(str(reason) for reason in reasons),
            )
        )
    return ReliabilityReport(
        annotators=tuple(annotators),
        excluded_annotation_ids=tuple(str(value) for value in excluded_ids),
        retained_count=int(document["retained_count"]),
        excluded_count=int(document["excluded_count"]),
    )


def retained_annotations(
    annotations: Sequence[RawAnnotation], report: ReliabilityReport
) -> tuple[RawAnnotation, ...]:
    excluded = set(report.excluded_annotation_ids)
    return tuple(annotation for annotation in annotations if annotation.annotation_id not in excluded)
