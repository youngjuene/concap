"""Jobs are content-addressed; conditions map to the right model and call; guards refuse early."""

from __future__ import annotations

import pytest
from diff_fixtures import relabelled
from fake_trida_model import BOS, FakeTokenizer

from diff.analysis_decisions import AnalysisDecision
from diff.caption_sources import SourceSet, parse_sources
from diff.job_plan import (
    PlanError,
    build_phase1_plan,
    build_phase2_plan,
    freeze_plan,
    frozen_plan_from_record,
    plan_from_record,
)
from diff.model_profile import QWEN_7B, TRIDA_7B
from diff.model_runtime import chat_token_ids
from diff.study_protocol import (
    CANVAS_TOKENS,
    PROTOCOL_SHA256,
    lattice_cells,
    lattice_corners,
)

TOKENIZERS = {TRIDA_7B.profile_id: {"fake": True}, QWEN_7B.profile_id: {"fake": True}}


def test_phase1_counts_match_the_brief_shape(development_sources: SourceSet) -> None:
    taus = ("original", "trida_tau_070", "trida_tau_090", "trida_tau_100")
    trida = build_phase1_plan(development_sources.sources, taus, "development")
    with_qwen = build_phase1_plan(development_sources.sources, (*taus, "qwen"), "development")
    assert len(trida.jobs) == 8 * 3  # two clips x four cues x three τ levels; original has no job
    assert len(with_qwen.jobs) == 8 * 4
    assert len({job.job_id for job in with_qwen.jobs}) == len(with_qwen.jobs)


def test_trida_jobs_carry_the_whole_call_contract(development_sources: SourceSet) -> None:
    plan = build_phase1_plan(development_sources.sources, ("trida_tau_070",), "development")
    job = plan.jobs[0]
    assert job.profile_id == TRIDA_7B.profile_id and job.runner == "trida"
    assert job.generate == {
        "max_new_tokens": 96,
        "threshold": 0.7,
        "block_size": 32,
        "small_block_size": 8,
        "temperature": 0,
        "use_block_cache": False,
        "mask_id": 128012,
        "stop_token": 128001,
        "pad_token_id": 128004,
    }
    assert job.caps == {"canvas_tokens": 96, "timeout_s": 60.0}
    assert job.prompt.system_message == "You are a helpful assistant."


def test_identity_is_stable_and_round_trips(development_sources: SourceSet) -> None:
    first = build_phase1_plan(development_sources.sources, ("trida_tau_090",), "development")
    second = build_phase1_plan(development_sources.sources, ("trida_tau_090",), "development")
    assert first.plan_id == second.plan_id
    assert plan_from_record(first.record()).plan_id == first.plan_id


def test_every_canvas_is_whole_blocks() -> None:
    # Trida decodes max_new_tokens // 32 blocks; anything else would lose positions.
    assert all(canvas >= 32 and canvas % 32 == 0 for canvas in CANVAS_TOKENS.values())
    assert len(PROTOCOL_SHA256) == 64


def test_lattice_has_nine_cells_and_four_corners() -> None:
    cells = lattice_cells()
    assert len(cells) == 9 and sum(cell["is_centre"] for cell in cells) == 1
    assert [cell["cell_id"] for cell in lattice_corners()] == [
        "t0-broad",
        "t4-broad",
        "t0-relation",
        "t4-relation",
    ]


@pytest.mark.parametrize(
    ("purpose", "material"), [("participant", "development"), ("development", "synthetic")]
)
def test_material_must_match_the_purpose(purpose: str, material: str) -> None:
    sources = parse_sources(relabelled(material))
    with pytest.raises(PlanError, match="material cannot serve"):
        build_phase1_plan(sources.sources, ("trida_tau_090",), purpose)


def test_freeze_records_token_hashes(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer()
    plan = build_phase1_plan(development_sources.sources, ("trida_tau_090", "qwen"), "development")
    frozen = freeze_plan(plan, lambda job: chat_token_ids(tokenizer, job.prompt), TOKENIZERS)
    assert all(job.token_count > 0 and len(job.token_ids_sha256) == 64 for job in frozen.jobs)
    assert frozen_plan_from_record(frozen.record()).frozen_sha256 == frozen.frozen_sha256


def test_a_rendered_bos_is_refused_at_freeze(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer(prefix=(BOS,))
    plan = build_phase1_plan(development_sources.sources, ("trida_tau_090",), "development")
    with pytest.raises(PlanError, match="BOS"):
        freeze_plan(plan, lambda job: chat_token_ids(tokenizer, job.prompt), TOKENIZERS)


def test_the_context_limit_is_enforced_at_freeze(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer(prefix=tuple([1000] * 4000))
    plan = build_phase1_plan(development_sources.sources, ("trida_tau_090",), "development")
    with pytest.raises(PlanError, match="4096-token context"):
        freeze_plan(plan, lambda job: chat_token_ids(tokenizer, job.prompt), TOKENIZERS)


def g0_decision(outcome: str = "pass", profile_id: str = TRIDA_7B.profile_id) -> dict[str, object]:
    return AnalysisDecision(
        gate="G0",
        outcome=outcome,
        rule={},
        evidence={},
        run_manifest_sha256="a" * 64,
        model_profile_id=profile_id,
        decided_by="test",
        decided_at="2026-10-09T00:00:00+0900",
    ).record()


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (None, "no G0 decision"),
        (g0_decision("fail"), "'fail', not 'pass'"),
        (g0_decision(profile_id=QWEN_7B.profile_id), "not the current profile"),
        ({**g0_decision(), "outcome": "pass", "decision_id": "dec-forged"}, "invalid"),
    ],
)
def test_lattice_generation_needs_a_passing_g0_ac20(
    development_sources: SourceSet, decision: dict[str, object] | None, message: str
) -> None:
    with pytest.raises(PlanError, match=message):
        build_phase2_plan(development_sources.sources, "trida_tau_090", "3x3", "development", decision)


def test_lattice_generation_with_a_passing_g0(development_sources: SourceSet) -> None:
    plan = build_phase2_plan(
        development_sources.sources, "trida_tau_090", "3x3", "development", g0_decision()
    )
    assert len(plan.jobs) == 3 * 9
    assert {job.generate["max_new_tokens"] for job in plan.jobs} == {128}
    assert {job.generate["threshold"] for job in plan.jobs} == {0.9}
