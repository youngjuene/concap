"""Raw annotation schema, aggregation, and reliability tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

from dpo.annotation.aggregate import aggregate_pair
from dpo.annotation.raw_annotations import AnnotationError, RawAnnotation, parse_annotation
from dpo.annotation.reliability import (
    build_reliability_report,
    krippendorff_alpha_nominal,
    parse_reliability_report,
    retained_annotations,
    two_sided_binomial_p,
)
from dpo.core.identity import sha256_bytes
from tests.conftest import PreferenceWorld


def _annotation(**overrides: object) -> RawAnnotation:
    base: dict[str, object] = {
        "annotation_id": "ann-1",
        "pair_id": "pair-1",
        "clip_id": "clip-1",
        "track": "visual",
        "candidate_a": "cand-a",
        "candidate_b": "cand-b",
        "display_order": ("cand-b", "cand-a"),
        "choice": "a_better",
        "tie_subtype": None,
        "preference_strength": 4,
        "confidence": 4,
        "reason_tags": ("coverage",),
        "annotator_id_hash": sha256_bytes(b"annotator"),
        "replay_count": 1,
        "response_time_ms": 5000,
        "collection_version": "1.0",
        "presentation": "muted_video",
    }
    base.update(overrides)
    return RawAnnotation(**base)  # type: ignore[arg-type]


def test_canonical_choice_resolves_through_display_order() -> None:
    # Displayed left is cand-b; choosing "a_better" (left) endorses cand-b.
    annotation = _annotation()
    assert annotation.canonical_choice() == "b_better"
    assert annotation.winner_id() == "cand-b"
    unflipped = _annotation(display_order=("cand-a", "cand-b"))
    assert unflipped.canonical_choice() == "a_better"
    assert unflipped.winner_id() == "cand-a"


def test_decisive_choices_require_strength_and_ties_forbid_it() -> None:
    with pytest.raises(AnnotationError, match="preference_strength"):
        _annotation(preference_strength=None)
    with pytest.raises(AnnotationError, match="no preference strength"):
        _annotation(choice="tie", tie_subtype="both_good")
    tie = _annotation(choice="tie", tie_subtype="both_good", preference_strength=None)
    assert tie.canonical_choice() == "tie"


def test_binary_forced_choice_is_insufficient() -> None:
    # The full response vocabulary must include ties and both-unacceptable.
    both_bad = _annotation(choice="both_unacceptable", tie_subtype="both_bad", preference_strength=None)
    assert both_bad.canonical_choice() == "both_unacceptable"


def test_roundtrip_through_document() -> None:
    annotation = _annotation()
    assert parse_annotation(annotation.document()) == annotation


def test_aggregate_counts_and_difficulty() -> None:
    rows = [
        _annotation(annotation_id="a-1", display_order=("cand-a", "cand-b"), choice="a_better"),
        _annotation(annotation_id="a-2", display_order=("cand-b", "cand-a"), choice="b_better"),
        _annotation(
            annotation_id="a-3",
            choice="tie",
            tie_subtype="both_acceptable",
            preference_strength=None,
        ),
    ]
    aggregate = aggregate_pair(rows)
    assert aggregate.a_better_count == 2
    assert aggregate.tie_count == 1
    assert aggregate.winner_id == "cand-a"
    assert 0.0 < aggregate.difficulty < 1.0


def test_repeat_inconsistency_and_attention_checks_drive_exclusions(
    world: PreferenceWorld,
) -> None:
    pair = world.pool.pairs[0]
    sloppy = sha256_bytes(b"sloppy-annotator")
    rows: list[RawAnnotation] = []
    for index in range(10):
        rows.append(
            _annotation(
                annotation_id=f"s-{index}",
                pair_id=f"attn-{index}",
                clip_id=pair.clip_id,
                candidate_a=pair.candidate_a,
                candidate_b=pair.candidate_b,
                display_order=(pair.candidate_a, pair.candidate_b),
                annotator_id_hash=sloppy,
                is_attention_check=True,
                choice="b_better",
            )
        )
    report = build_reliability_report(
        rows,
        attention_expected={f"attn-{index}": "a_better" for index in range(10)},
        min_response_ms=1000,
        max_position_bias=0.4,
        position_bias_alpha=0.01,
        min_attention_pass=0.8,
    )
    annotator_report = report.annotators[0]
    assert annotator_report.excluded
    assert "attention_check_failure" in annotator_report.exclusion_reasons
    assert retained_annotations(rows, report) == ()


def test_raw_store_is_append_only_and_never_overwritten(world: PreferenceWorld) -> None:
    from dpo.annotation.raw_annotations import load_annotations

    lines = "\n".join(__import__("json").dumps(annotation.document()) for annotation in world.annotations[:4])
    parsed = load_annotations(lines)
    assert len(parsed) == 4
    duplicated = lines + "\n" + lines.splitlines()[0]
    with pytest.raises(AnnotationError, match="append-only"):
        load_annotations(duplicated)


def test_annotations_must_match_the_frozen_pool(world: PreferenceWorld) -> None:
    from dpo.annotation.raw_annotations import validate_against_pool

    good = world.annotations[0]
    other_pair = next(pair for pair in world.pool.pairs if pair.candidate_a != good.candidate_a)
    tampered = replace(good, pair_id=other_pair.pair_id)
    with pytest.raises(AnnotationError, match="frozen pair"):
        validate_against_pool([tampered], world.pool)


class TestPositionBias:
    """A side lean costs an annotator every judgment they made, so the rule that
    convicts them has to survive both directions of error."""

    def _rows(self, left: int, right: int) -> list[RawAnnotation]:
        rows = []
        for index in range(left + right):
            rows.append(
                _annotation(
                    annotation_id=f"p-{index}",
                    pair_id=f"pair-{index}",
                    clip_id="clip-a",
                    candidate_a="cand-a",
                    candidate_b="cand-b",
                    display_order=("cand-a", "cand-b"),
                    annotator_id_hash=sha256_bytes(b"leaner"),
                    choice="a_better" if index < left else "b_better",
                )
            )
        return rows

    def _reasons(self, left: int, right: int, *, alpha: float = 0.01) -> tuple[str, ...]:
        report = build_reliability_report(
            self._rows(left, right),
            attention_expected={},
            min_response_ms=0,
            max_position_bias=0.2,
            position_bias_alpha=alpha,
            min_attention_pass=0.8,
        )
        return report.annotators[0].exclusion_reasons

    def test_a_lopsided_handful_is_not_evidence(self) -> None:
        # Five of five looks like total bias and is a coin landing heads five
        # times: p = 0.0625. The old `n >= 10` floor got this right by accident
        # and got n = 11 wrong; the test gets both right for a reason.
        assert "position_bias" not in self._reasons(5, 0)
        assert "position_bias" not in self._reasons(8, 3)

    def test_a_sustained_lean_is(self) -> None:
        assert "position_bias" in self._reasons(60, 20)

    def test_a_small_lean_survives_any_sample_size(self) -> None:
        # At 800 judgments a 0.55 lean is resolved well past any alpha we would
        # set, and it still distorts nothing. The effect-size floor is the half
        # of the rule that refuses to convict on it, and it has to keep doing so
        # however many judgments arrive.
        assert two_sided_binomial_p(440, 800) < 0.01
        assert "position_bias" not in self._reasons(440, 360)

    def test_the_p_value_is_published_with_the_decision(self) -> None:
        report = build_reliability_report(
            self._rows(60, 20),
            attention_expected={},
            min_response_ms=0,
            max_position_bias=0.2,
            position_bias_alpha=0.01,
            min_attention_pass=0.8,
        )
        assert report.annotators[0].position_bias_p_value < 0.01
        # And it survives the round trip, so a later stage reads the same decision.
        restored = parse_reliability_report(json.dumps(report.document()))
        assert restored.annotators[0].position_bias_p_value == report.annotators[0].position_bias_p_value


def test_two_sided_binomial_is_exact_and_symmetric() -> None:
    assert two_sided_binomial_p(0, 0) == 1.0
    assert two_sided_binomial_p(5, 10) == 1.0
    assert two_sided_binomial_p(8, 11) == two_sided_binomial_p(3, 11)
    # 10 heads in 10 flips, both tails: 2/1024.
    assert two_sided_binomial_p(10, 10) == pytest.approx(2 / 1024)


class TestKrippendorffAlpha:
    """The study's only real agreement number: at two judgments per pair the
    modal-share `agreement` can only be 0.5 or 1.0, and credits nothing to
    chance."""

    def _judgment(self, pair: str, annotator: str, order: tuple[str, str], choice: str) -> RawAnnotation:
        return _annotation(
            annotation_id=f"{pair}-{annotator}",
            pair_id=pair,
            candidate_a="cand-a",
            candidate_b="cand-b",
            display_order=order,
            annotator_id_hash=sha256_bytes(annotator.encode()),
            choice=choice,
        )

    def test_matches_the_reference_implementation(self) -> None:
        # Krippendorff's own worked example: three observers, fifteen units with
        # gaps, five categories. Checked once against the `krippendorff` PyPI
        # package in a throwaway venv, which agreed to one ULP -- that package is
        # GPL-3.0 and ships no py.typed, so it validates this code without ever
        # entering the lockfile.
        #
        # Five categories is more than `canonical_choice()` can express, so this
        # exercises the estimator directly. The domain tests below cover the part
        # that reads RawAnnotation.
        @dataclass(frozen=True)
        class Judgment:
            pair_id: str
            category: str
            is_attention_check: bool = False
            repeat_of: str | None = None

            def canonical_choice(self) -> str:
                return self.category

        columns = [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None, None, None, None],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, None, None, None, 3],
            [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None, 3, None, 4],
        ]
        rows = [
            Judgment(pair_id=f"u{unit}", category=str(value))
            for column in columns
            for unit, value in enumerate(column)
            if value is not None
        ]
        alpha = krippendorff_alpha_nominal(rows)  # type: ignore[arg-type]
        assert alpha is not None
        assert alpha == pytest.approx(0.6127596439169138, abs=1e-12)

    def test_display_order_is_resolved_before_agreement_is_counted(self) -> None:
        # Both annotators pressed the left button on every pair, and because the
        # order was flipped for one of them they endorsed opposite candidates
        # every time. A naive implementation reading `choice` scores this 1.0.
        rows: list[RawAnnotation] = []
        for index in range(8):
            rows.append(self._judgment(f"p{index}", "alice", ("cand-a", "cand-b"), "a_better"))
            rows.append(self._judgment(f"p{index}", "bob", ("cand-b", "cand-a"), "a_better"))
        alpha = krippendorff_alpha_nominal(rows)
        assert alpha is not None and alpha < 0.0

    def test_perfect_and_reversed(self) -> None:
        agreeing = [
            self._judgment(f"p{index}", who, ("cand-a", "cand-b"), "a_better" if index % 2 else "b_better")
            for index in range(6)
            for who in ("alice", "bob")
        ]
        assert krippendorff_alpha_nominal(agreeing) == 1.0

    def test_undefined_rather_than_zero_when_nothing_could_disagree(self) -> None:
        # One category everywhere: no disagreement was possible, so there is no
        # reliability to report. Zero would read as total disagreement.
        uniform = [
            self._judgment(f"p{index}", who, ("cand-a", "cand-b"), "a_better")
            for index in range(4)
            for who in ("alice", "bob")
        ]
        assert krippendorff_alpha_nominal(uniform) is None
        # And a pair only one annotator reached contributes nothing.
        alone = [self._judgment("p0", "alice", ("cand-a", "cand-b"), "a_better")]
        assert krippendorff_alpha_nominal(alone) is None

    def test_attention_checks_and_repeats_are_left_out(self) -> None:
        real = [
            self._judgment(f"p{index}", who, ("cand-a", "cand-b"), "a_better" if index % 2 else "b_better")
            for index in range(4)
            for who in ("alice", "bob")
        ]
        baseline = krippendorff_alpha_nominal(real)
        noise = [
            replace(real[0], annotation_id="attn", pair_id="attn-1", is_attention_check=True),
            replace(real[1], annotation_id="attn2", pair_id="attn-1", is_attention_check=True),
            # A repeat is the same annotator twice: within-rater consistency,
            # which repeat_consistency already reports.
            replace(real[0], annotation_id="rep", repeat_of=real[0].annotation_id),
        ]
        assert krippendorff_alpha_nominal([*real, *noise]) == baseline

    def test_the_report_carries_it(self, world: PreferenceWorld) -> None:
        report = build_reliability_report(
            world.annotations,
            attention_expected={},
            min_response_ms=0,
            max_position_bias=0.2,
            position_bias_alpha=0.01,
            min_attention_pass=0.8,
        )
        restored = parse_reliability_report(json.dumps(report.document()))
        assert restored.krippendorff_alpha == report.krippendorff_alpha
