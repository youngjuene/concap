"""G0 reads development material only, stops at a blind packet, and records its decision."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest
from diff_fixtures import relabelled
from fake_trida_model import FakeTokenizer

from diff import feasibility_gate as gate
from diff.analysis_decisions import G0_PASS_OR_WAIVER, decision_from_record, gate_problems
from diff.caption_sources import SourceSet, parse_sources
from diff.job_plan import FrozenPlan, PlanError, freeze_plan
from diff.model_profile import QWEN_7B, TRIDA_7B
from diff.model_runtime import chat_token_ids

TOKENIZERS = {TRIDA_7B.profile_id: {"fake": True}, QWEN_7B.profile_id: {"fake": True}}


def frozen_g0(sources: SourceSet, *, smoke: bool = False) -> FrozenPlan:
    tokenizer = FakeTokenizer()
    plan = gate.build_g0_plan(sources.sources, smoke=smoke)
    return freeze_plan(plan, lambda job: chat_token_ids(tokenizer, job.prompt), TOKENIZERS)


def test_g0_reads_development_cues_and_twelve_corner_cells(development_sources: SourceSet) -> None:
    items = gate.g0_items(development_sources.sources)
    phase1 = [item for item in items if item["template_phase"] == 1]
    cells = [item for item in items if item["template_phase"] == 2]
    assert len(phase1) == 8 and len(cells) == 12
    corners = [item["cell"]["cell_id"] for item in cells]
    assert {corner: corners.count(corner) for corner in set(corners)} == {
        "t0-broad": 3,
        "t4-broad": 3,
        "t0-relation": 3,
        "t4-relation": 3,
    }
    assert all(item["source"].phase == 1 for item in items)  # never a phase-2 (participant) video


@pytest.mark.parametrize("material", ["participant", "synthetic"])
def test_g0_refuses_anything_but_development_material(material: str) -> None:
    with pytest.raises(PlanError, match="ask the researcher"):
        gate.g0_items(parse_sources(relabelled(material)).sources)


def test_g0_needs_at_least_three_cues() -> None:
    document = relabelled("development")
    document["sources"] = [s for s in document["sources"] if s["unit"]["key"] == "smoke_clip_a"][:2]
    with pytest.raises(PlanError, match="at least 12"):
        gate.g0_items(parse_sources(document).sources)


def test_the_g0_plan_covers_four_conditions(development_sources: SourceSet) -> None:
    plan = gate.build_g0_plan(development_sources.sources)
    assert len(plan.jobs) == (8 + 12) * 4 and plan.purpose == "g0"
    assert {job.condition for job in plan.jobs} == set(gate.G0_CONDITIONS)
    cell_jobs = [job for job in plan.jobs if job.cell is not None]
    assert {job.phase for job in cell_jobs} == {2}
    assert {job.generate["max_new_tokens"] for job in cell_jobs} == {128}


def write_attempts(run_dir: Path, frozen: FrozenPlan, text: str, *, incomplete: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for number, entry in enumerate(frozen.jobs):
        lines.append(
            {
                "attempt_id": f"att-{number}",
                "run_id": "run-test",
                "job_id": entry.job.job_id,
                "status": "completed",
                "terminal": True,
                "technical_failure": False,
                "latency_ms": 100 + number,
                "output": {
                    "raw_text": text,
                    "text": text,
                    "completion": "incomplete" if incomplete else "terminated",
                    "terminator": None if incomplete else "endoftext",
                    "terminator_index": None if incomplete else 3,
                    "postprocess": {"tail_token_ids": [], "tail_text": ""},
                    "leftover_mask_count": 0,
                    "special_ids_in_text": [],
                    "new_token_count": entry.job.generate["max_new_tokens"],
                },
                "trace": {"summary": {"realized_per_step": [0, 1, 1, 2]}},
            }
        )
    (run_dir / "attempts.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )


def packet_for(
    tmp_path: Path, sources: SourceSet, *, smoke: bool = False, text: str = "버스가 지나간다."
) -> dict[str, Any]:
    frozen = frozen_g0(sources, smoke=smoke)
    write_attempts(tmp_path, frozen, text)
    latest = gate.latest_attempts(tmp_path)
    manifest = gate.run_manifest(frozen, tmp_path, sources.sources)
    report = gate.g0_report(frozen, latest, sources.by_id())
    rows, key = gate.blind_packet(frozen, latest, sources.by_id(), manifest["manifest_sha256"])
    return {"frozen": frozen, "manifest": manifest, "report": report, "rows": rows, "key": key}


def test_report_and_blind_packet(tmp_path: Path, development_sources: SourceSet) -> None:
    built = packet_for(tmp_path, development_sources)
    report, rows, key = built["report"], built["rows"], built["key"]
    assert report["conditions"]["trida_tau_090"]["jobs"] == 20
    assert report["conditions"]["trida_tau_090"]["realized_per_step"]["zero_commit_steps"] == 20
    assert len(rows) == 40 and {entry["condition"] for entry in key.values()} == {"trida_tau_090", "qwen"}
    packet = gate.packet_csv(rows)
    header = next(csv.reader(io.StringIO(packet)))
    assert header == list(gate.PACKET_COLUMNS)
    for word in ("trida", "qwen", "tau", TRIDA_7B.profile_id, "condition", "job-"):
        assert word not in packet.lower()
    assert built["manifest"]["materials"] == ["development"]


def ratings_for(key: dict[str, Any], score_for: dict[str, int]) -> str:
    return "blind_id,naturalness\n" + "".join(
        f"{blind_id},{score_for[entry['condition']]}\n" for blind_id, entry in key.items()
    )


def test_decide_passes_and_records_an_analysis_decision(
    tmp_path: Path, development_sources: SourceSet
) -> None:
    built = packet_for(tmp_path, development_sources)
    ratings = gate.read_ratings(ratings_for(built["key"], {"trida_tau_090": 4, "qwen": 5}), built["key"])
    decision = gate.decide(
        built["report"], built["key"], ratings, built["manifest"], decided_by="tester", decided_at="now"
    )
    assert decision.outcome == "pass" and decision.gate == "G0"
    assert decision.run_manifest_sha256 == built["manifest"]["manifest_sha256"]
    assert decision.evidence["qwen"]["naturalness_share"] == 1.0
    record = decision.record()
    assert decision_from_record(record).decision_id == record["decision_id"]
    assert gate_problems(record, gate="G0", model_profile_id=TRIDA_7B.profile_id) == []


def test_decide_fails_below_three_quarters(tmp_path: Path, development_sources: SourceSet) -> None:
    built = packet_for(tmp_path, development_sources)
    ratings = gate.read_ratings(ratings_for(built["key"], {"trida_tau_090": 2, "qwen": 5}), built["key"])
    decision = gate.decide(
        built["report"], built["key"], ratings, built["manifest"], decided_by="tester", decided_at="now"
    )
    assert decision.outcome == "fail" and decision.evidence["criteria"]["naturalness"] is False
    assert gate_problems(decision.record(), gate="G0", model_profile_id=TRIDA_7B.profile_id)


def test_script_failures_fail_g0(tmp_path: Path, development_sources: SourceSet) -> None:
    built = packet_for(tmp_path, development_sources, text="バスが通る。")
    ratings = gate.read_ratings(ratings_for(built["key"], {"trida_tau_090": 5, "qwen": 5}), built["key"])
    decision = gate.decide(
        built["report"], built["key"], ratings, built["manifest"], decided_by="tester", decided_at="now"
    )
    assert decision.evidence["criteria"]["script"] is False and decision.outcome == "fail"


def test_an_automatic_fail_is_recorded_without_ratings(
    tmp_path: Path, development_sources: SourceSet
) -> None:
    built = packet_for(tmp_path, development_sources, text="")
    decision = gate.decide(
        built["report"], built["key"], None, built["manifest"], decided_by="tester", decided_at="now"
    )
    assert decision.outcome == "fail"
    assert decision.evidence["criteria"] == {"terminal": True, "script": False, "naturalness": None}
    assert decision.evidence["trida_tau_090"]["rated"] == 0
    assert decision.evidence["trida_tau_090"]["naturalness_share"] is None  # unrated, not a share of 0
    record = decision.record()
    assert decision_from_record(record).decision_id == record["decision_id"]
    assert gate_problems(record, gate="G0", model_profile_id=TRIDA_7B.profile_id)


def test_a_pass_always_needs_the_rating(tmp_path: Path, development_sources: SourceSet) -> None:
    built = packet_for(tmp_path, development_sources)
    with pytest.raises(ValueError, match="needs the human rating"):
        gate.decide(
            built["report"], built["key"], None, built["manifest"], decided_by="tester", decided_at="now"
        )


def test_copies_of_the_source_count_against_the_script_share(
    tmp_path: Path, development_sources: SourceSet
) -> None:
    # Decided 2026-10-11: every output here repeats its cue's 원문, so none passes.
    frozen = frozen_g0(development_sources)
    write_attempts(tmp_path, frozen, "버스가 지나간다.")
    path = tmp_path / "attempts.jsonl"
    by_id = development_sources.by_id()
    source_of = {entry.job.job_id: by_id[entry.job.source_id] for entry in frozen.jobs}
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        text = source_of[line["job_id"]].text
        line["output"].update(raw_text=text, text=text)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    report = gate.g0_report(frozen, gate.latest_attempts(tmp_path), by_id)
    rated = report["conditions"]["trida_tau_090"]
    assert rated["terminal"] == rated["jobs"] == 20 and rated["script_pass"] == 0


def test_a_waiver_is_recorded_and_never_passes_as_a_pass(
    tmp_path: Path, development_sources: SourceSet
) -> None:
    # Decided 2026-10-11: participants rate live captions, so G0's human rating is waived.
    built = packet_for(tmp_path, development_sources)
    decision = gate.waive(
        built["report"], built["manifest"], decided_by="tester", decided_at="now", notes="participants rate"
    )
    assert decision.outcome == "waived" and decision.evidence["criteria"]["naturalness"] is None
    record = decision.record()
    assert decision_from_record(record).decision_id == record["decision_id"]
    profile_id = TRIDA_7B.profile_id
    refusals = gate_problems(record, gate="G0", model_profile_id=profile_id)
    assert any("'waived', not 'pass'" in refusal for refusal in refusals)
    assert gate_problems(record, gate="G0", model_profile_id=profile_id, accept=G0_PASS_OR_WAIVER) == []
    with pytest.raises(ValueError, match="reason"):
        gate.waive(built["report"], built["manifest"], decided_by="t", decided_at="now", notes=" ")


def test_a_timed_out_job_counts_against_the_script_share(
    tmp_path: Path, development_sources: SourceSet
) -> None:
    # Decided 2026-10-10: the share is over every job of the rated condition, so a missing output fails it.
    frozen = frozen_g0(development_sources)
    write_attempts(tmp_path, frozen, "버스가 지나간다.")
    path = tmp_path / "attempts.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rated_jobs = {entry.job.job_id for entry in frozen.jobs if entry.job.condition == "trida_tau_090"}
    timed_out = next(line for line in lines if line["job_id"] in rated_jobs)
    timed_out.update(status="timeout", output=None, trace=None, latency_ms=None)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    report = gate.g0_report(frozen, gate.latest_attempts(tmp_path), development_sources.by_id())
    rated_stats = report["conditions"]["trida_tau_090"]
    assert rated_stats["terminal"] == rated_stats["jobs"] == 20 and rated_stats["technical_failures"] == 0
    assert rated_stats["script_pass"] == 19 and rated_stats["script_pass_share"] == 19 / 20


def test_a_smoke_run_is_never_decided(tmp_path: Path, smoke_sources: SourceSet) -> None:
    built = packet_for(tmp_path, smoke_sources, smoke=True)
    ratings = gate.read_ratings(ratings_for(built["key"], {"trida_tau_090": 5, "qwen": 5}), built["key"])
    with pytest.raises(ValueError, match="smoke"):
        gate.decide(
            built["report"], built["key"], ratings, built["manifest"], decided_by="t", decided_at="now"
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda text: text + "b-unknown,3\n", "unknown blind_id"),
        (lambda text: text.replace(",4\n", ",9\n", 1), "from 1 to 5"),
        (lambda text: "\n".join(text.splitlines()[:-1]) + "\n", "not rated"),
    ],
)
def test_ratings_are_validated(
    tmp_path: Path, development_sources: SourceSet, mutate: Any, message: str
) -> None:
    built = packet_for(tmp_path, development_sources)
    text = ratings_for(built["key"], {"trida_tau_090": 4, "qwen": 4})
    with pytest.raises(ValueError, match=message):
        gate.read_ratings(mutate(text), built["key"])
