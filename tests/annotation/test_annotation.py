"""Raw annotation schema, aggregation, and reliability tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dpo.annotation.aggregate import aggregate_pair
from dpo.annotation.raw_annotations import AnnotationError, RawAnnotation, parse_annotation
from dpo.annotation.reliability import build_reliability_report, retained_annotations
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
