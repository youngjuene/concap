"""Caption-control qualification checks for the v2 viewing study."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import Any

from dpo.regen.study_worker import InferenceProcess, excerpt
from dpo.regen.tests.qualify_captions import (
    axis_matrix,
    cue_from_args,
    default_cue,
    default_profile,
    qualify_axis_contract,
    qualify_four_corner_inference,
    qualify_repeated_inference,
    qualify_store_cache,
    reproducibility_record,
    reviewer_sheet_rows,
)


def window_reader_worker(pipe: Any, settings: Any) -> None:
    del settings
    while True:
        spec = pipe.recv()
        if spec is None:
            return
        excerpt(spec["audio"], spec["excerpt"], spec["start_ms"], spec["end_ms"])
        with wave.open(spec["excerpt"], "rb") as stream:
            values = sorted(set(stream.readframes(stream.getnframes())))
        pipe.send({"text": ",".join(str(value) for value in values), "fallback": False, "duration_ms": 0})


def test_four_corner_axis_contract_is_distinct_and_bounded() -> None:
    report = qualify_axis_contract(default_profile(), default_cue())

    assert report["passed"] is True
    assert report["checks"] == {
        "four_corners_present": True,
        "all_instruction_hashes_distinct": True,
        "texture_changes_descriptor_budget_at_low_context": True,
        "texture_changes_descriptor_budget_at_high_context": True,
        "context_policy_independent_of_texture_low_context": True,
        "context_policy_independent_of_texture_high_context": True,
        "context_policy_changes_at_low_texture": True,
        "context_policy_changes_at_high_texture": True,
        "uses_excerpt_relative_time": True,
        "includes_authored_evidence": True,
    }
    assert report["quality_scope"] == "mechanical contract only; no grounding or human independence score"


def test_axis_contract_keys_change_one_dimension_at_a_time() -> None:
    rows = {row["name"]: row for row in axis_matrix(default_profile(), default_cue())}

    assert rows["low_texture_low_context"]["descriptor_limit"] == 0
    assert rows["high_texture_low_context"]["descriptor_limit"] == 4
    assert rows["low_texture_high_context"]["descriptor_limit"] == 0
    assert rows["high_texture_high_context"]["descriptor_limit"] == 4
    assert (
        rows["low_texture_low_context"]["context_policy"]
        == rows["high_texture_low_context"]["context_policy"]
    )
    assert (
        rows["low_texture_high_context"]["context_policy"]
        == rows["high_texture_high_context"]["context_policy"]
    )
    assert (
        rows["low_texture_low_context"]["instruction_hash"]
        != rows["high_texture_low_context"]["instruction_hash"]
    )
    assert (
        rows["low_texture_low_context"]["instruction_hash"]
        != rows["low_texture_high_context"]["instruction_hash"]
    )


def test_repeat_inference_records_latency_and_no_backend_fallback(tmp_path: Path) -> None:
    audio = tmp_path / "calibration-audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 8000 * 5)
    spec = {
        "audio": str(audio),
        "excerpt": str(tmp_path / "excerpt.wav"),
        "start_ms": 0,
        "end_ms": 1000,
        "instruction": "Describe the audible sounds in one short English sentence.",
        "language": "en",
        "fallback": "A calibration fallback caption.",
    }
    engine = InferenceProcess({})
    try:
        report = qualify_repeated_inference(engine, spec, timeout=5, repeats=2)
    finally:
        engine.close()

    assert report["fallback_count"] == 2
    assert report["fallback_rate"] == 1
    assert report["cold_wall_ms"] >= 0
    assert report["warm_wall_ms_median"] >= 0
    assert [sample["reason"] for sample in report["samples"]] == ["model_not_configured"] * 2


def test_four_corner_inference_records_one_sample_per_axis_corner(tmp_path: Path) -> None:
    audio = tmp_path / "calibration-audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 8000 * 5)
    engine = InferenceProcess({})
    try:
        report = qualify_four_corner_inference(engine, audio, language="en", timeout=5)
    finally:
        engine.close()

    assert report["fallback_count"] == 4
    assert report["fallback_rate"] == 1
    assert report["identical_output_pairs"] == {
        "context_low_high_at_low_texture": True,
        "context_low_high_at_high_texture": True,
        "texture_low_high_at_low_context": True,
        "texture_low_high_at_high_context": True,
    }
    assert report["evidence_provenance"] == "synthetic_unverified_default"
    assert [sample["fallback"] for sample in report["samples"]] == [True] * 4
    assert len({sample["instruction_hash"] for sample in report["samples"]}) == 4


def test_cli_cue_overrides_record_operator_supplied_evidence() -> None:
    cue = cue_from_args(
        evidence="Authored cue: footsteps pass under a covered walkway.",
        fallback="Footsteps pass nearby.",
        language="en",
        start_ms=5000,
        end_ms=9000,
    )
    report = qualify_axis_contract(default_profile(), cue)

    assert report["evidence_provenance"] == "cli_operator_supplied_unscored"
    assert report["checks"]["includes_authored_evidence"] is True


def test_store_cache_probe_distinguishes_cold_worker_and_warm_cache(tmp_path: Path) -> None:
    report = qualify_store_cache(tmp_path)

    assert report["passed"] is True
    assert report["cold_state"] == "succeeded"
    assert report["warm_claimed_worker_job"] is False
    assert report["warm_state"] == "succeeded"


def test_inference_specs_preserve_nonzero_absolute_audio_window(tmp_path: Path) -> None:
    audio = tmp_path / "windowed.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes([1]) * 8000 * 5 + bytes([9]) * 8000 * 5)
    cue = cue_from_args(
        evidence="Authored cue for the second five seconds.",
        fallback="Second window fallback.",
        language="en",
        start_ms=5000,
        end_ms=10000,
    )
    engine = InferenceProcess({}, target=window_reader_worker)
    try:
        report = qualify_four_corner_inference(engine, audio, language="en", timeout=5, cue=cue)
    finally:
        engine.close()

    assert report["fallback_count"] == 0
    assert [sample["text"] for sample in report["samples"]] == ["9"] * 4


def test_reviewer_sheet_template_has_future_review_columns() -> None:
    rows = reviewer_sheet_rows(None, None)

    assert len(rows) == 3
    assert "grounding_notes" in rows[0]
    assert "axis_independence_notes" in rows[0]
    assert "context_low_high_at_low_texture_identical" in rows[0]


def test_reproducibility_record_hashes_source_files() -> None:
    args = argparse.Namespace(
        audio=None,
        manifest=None,
        media=None,
        backend_config=None,
        contract=None,
        checkpoint=None,
    )
    record = reproducibility_record(args, ["--language", "en"])

    assert record["argv"] == ["--language", "en"]
    assert record["source_hashes"]["harness"].startswith("sha256:")
    assert record["source_hashes"]["study_schema"].startswith("sha256:")
