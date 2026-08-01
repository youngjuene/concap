"""Annotator reliability, quality controls, and preregistered exclusions.

Exclusion rules are parameters of the study contract and are applied to raw
annotations BEFORE aggregation. Excluded annotations are never deleted from
the raw store — the report lists them, and derived views simply do not read
them. That preserves the append-only property while making the exclusion
decision reproducible.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dpo.annotation.raw_annotations import AnnotationError, RawAnnotation


def two_sided_binomial_p(successes: int, trials: int) -> float:
    """P(a fair coin lands at least this far from even in ``trials`` flips).

    Exact rather than normal-approximated: an annotator contributes judgments in
    the dozens, which is exactly where the approximation is worst, and
    ``math.comb`` makes the exact sum free at this size. Integer arithmetic up
    to a single final division, so the value is byte-identical across runs.
    """
    if trials <= 0:
        return 1.0
    # Halves of an integer are exact in binary, so this comparison is not a
    # float-tolerance question: a count is either at least as far from even as
    # the observed one or it is not.
    distance = abs(successes - trials / 2.0)
    tail = sum(
        math.comb(trials, count) for count in range(trials + 1) if abs(count - trials / 2.0) >= distance
    )
    outcomes: int = 2**trials  # every sequence of `trials` flips, all equally likely
    return tail / outcomes


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
    position_bias_p_value: float
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
            "position_bias_p_value": self.position_bias_p_value,
            "fast_response_count": self.fast_response_count,
            "excluded": self.excluded,
            "exclusion_reasons": list(self.exclusion_reasons),
        }


def krippendorff_alpha_nominal(annotations: Sequence[RawAnnotation]) -> float | None:
    """Chance-corrected agreement across annotators, nominal metric.

    The one agreement statistic this study can actually report. ``agreement`` on
    an aggregated pair is the modal share, and at two judgments per pair that is
    0.5 or 1.0 with nothing credited to chance — a coin would score 0.5 there.
    Alpha is defined against expected disagreement, so it answers the question a
    reader is really asking.

    Judgments are read through ``canonical_choice()``, never the raw ``choice``:
    display order is randomized per annotator, so two annotators who both
    answered "a_better" may have endorsed opposite candidates.

    Two exclusions, both deliberate. Attention checks have a right answer, so
    concurring on one is compliance rather than agreement about a preference.
    Repeats are a second judgment from the SAME annotator, which measures
    within-rater consistency — ``repeat_consistency`` already reports that, and
    folding it in here would inflate alpha with self-agreement.

    Computed before exclusions, because with two annotators the post-exclusion
    number carries no information: dropping either one leaves no pair with two
    judgments at all.

    Returns None when nothing could have disagreed — no pair judged twice, or
    every judgment in one category — since 0.0 there would read as total
    disagreement rather than as an undefined quantity.
    """
    by_unit: dict[str, list[str]] = {}
    for annotation in annotations:
        if annotation.is_attention_check or annotation.repeat_of is not None:
            continue
        by_unit.setdefault(annotation.pair_id, []).append(annotation.canonical_choice())
    units = [choices for choices in by_unit.values() if len(choices) >= 2]
    if not units:
        return None
    categories = sorted({choice for choices in units for choice in choices})
    # Coincidence matrix: every ordered pair of judgments within a unit, each
    # weighted 1/(m-1) so a pair judged many times cannot outvote one judged
    # twice. Iteration order is fixed by `categories` and by insertion, so the
    # float sums are reproducible.
    coincidence: dict[tuple[str, str], float] = {}
    for choices in units:
        counts = {category: choices.count(category) for category in categories}
        weight = float(len(choices) - 1)
        for first in categories:
            for second in categories:
                pairs = counts[first] * (counts[second] - (1 if first == second else 0))
                if pairs:
                    key = (first, second)
                    coincidence[key] = coincidence.get(key, 0.0) + pairs / weight
    totals = {
        category: sum(value for (first, _), value in coincidence.items() if first == category)
        for category in categories
    }
    grand = sum(totals[category] for category in categories)
    observed = sum(value for (first, second), value in coincidence.items() if first != second)
    expected = sum(
        totals[first] * totals[second] for first in categories for second in categories if first != second
    )
    if grand <= 1.0 or expected <= 0.0:
        return None
    return 1.0 - (grand - 1.0) * observed / expected


@dataclass(frozen=True)
class ReliabilityReport:
    annotators: tuple[AnnotatorReport, ...]
    excluded_annotation_ids: tuple[str, ...]
    retained_count: int
    excluded_count: int
    krippendorff_alpha: float | None

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.reliability-report/v2",
            "annotators": [report.document() for report in self.annotators],
            "excluded_annotation_ids": list(self.excluded_annotation_ids),
            "retained_count": self.retained_count,
            "excluded_count": self.excluded_count,
            "krippendorff_alpha": self.krippendorff_alpha,
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
    position_bias_alpha: float,
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
        position_bias_p = two_sided_binomial_p(left_choices, len(decisive))
        fast = sum(1 for row in rows if row.response_time_ms < min_response_ms)
        reasons: list[str] = []
        if attention_rows and attention_pass_rate < min_attention_pass:
            reasons.append("attention_check_failure")
        # A side lean has to be both big enough to distort a preference view and
        # too big to be chance before it costs an annotator every judgment they
        # made. Either test alone is wrong in a way the other is not: the effect
        # size alone convicts eight-of-eleven, which a fair coin produces about
        # a fifth of the time, while the p-value alone convicts a 0.55 lean once
        # there are enough judgments to resolve it, and a 0.55 lean distorts
        # nothing. Requiring both also retires an arbitrary `n >= 10` floor —
        # too few judgments now fails the test on its own, as it should.
        if position_bias > max_position_bias and position_bias_p < position_bias_alpha:
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
                position_bias_p_value=position_bias_p,
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
        krippendorff_alpha=krippendorff_alpha_nominal(annotations),
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
    expected = {
        "schema",
        "annotators",
        "excluded_annotation_ids",
        "retained_count",
        "excluded_count",
        "krippendorff_alpha",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise AnnotationError("reliability report payload has an unexpected field set")
    if document["schema"] != "dpo.reliability-report/v2":
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
                position_bias_p_value=float(row["position_bias_p_value"]),
                fast_response_count=int(row["fast_response_count"]),
                excluded=bool(row["excluded"]),
                exclusion_reasons=tuple(str(reason) for reason in reasons),
            )
        )
    alpha = document["krippendorff_alpha"]
    return ReliabilityReport(
        annotators=tuple(annotators),
        excluded_annotation_ids=tuple(str(value) for value in excluded_ids),
        retained_count=int(document["retained_count"]),
        excluded_count=int(document["excluded_count"]),
        krippendorff_alpha=None if alpha is None else float(alpha),
    )


def retained_annotations(
    annotations: Sequence[RawAnnotation], report: ReliabilityReport
) -> tuple[RawAnnotation, ...]:
    excluded = set(report.excluded_annotation_ids)
    return tuple(annotation for annotation in annotations if annotation.annotation_id not in excluded)
