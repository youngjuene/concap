"""The supervisor kills overruns and retries once, only after a transient failure."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fake_dream_model import FakeTokenizer

from diff.caption_sources import SourceSet
from diff.generation_worker import (
    Supervisor,
    WorkerCommand,
    WorkerStartupError,
    job_decided,
    read_attempts,
    run_plan,
)
from diff.job_plan import FrozenPlan, build_phase1_plan, freeze_plan
from diff.model_profile import DREAM_7B
from diff.model_runtime import chat_token_ids

SCRIPT = Path(__file__).parent / "fake_generation_worker.py"


def command(tmp_path: Path, behaviour: str) -> WorkerCommand:
    env = {
        **os.environ,
        "FAKE_WORKER_BEHAVIOUR": behaviour,
        "FAKE_WORKER_MARKER": str(tmp_path / "marker"),
    }
    return WorkerCommand(
        argv=(sys.executable, str(SCRIPT)), env=env, cwd=tmp_path, log_path=tmp_path / "worker.log"
    )


def frozen(sources: SourceSet, count: int = 2) -> FrozenPlan:
    tokenizer = FakeTokenizer()
    plan = build_phase1_plan(sources.sources, ("dream_k1",), "development")
    trimmed = type(plan)(phase=plan.phase, purpose=plan.purpose, jobs=plan.jobs[:count])
    return freeze_plan(trimmed, lambda job: chat_token_ids(tokenizer, job.prompt), {DREAM_7B.profile_id: {}})


def run(tmp_path: Path, sources: SourceSet, behaviour: str, count: int = 1) -> list[dict[str, object]]:
    plan = frozen(sources, count)
    run_plan(
        plan,
        tmp_path / "run",
        {DREAM_7B.profile_id: command(tmp_path, behaviour)},
        {DREAM_7B.profile_id: {"files": {}, "weights": {}}},
        startup_timeout=30,
        log=lambda line: None,
    )
    return read_attempts(tmp_path / "run" / "attempts.jsonl")


def test_completed_attempts_are_recorded_with_their_trace(
    tmp_path: Path, development_sources: SourceSet
) -> None:
    attempts = run(tmp_path, development_sources, "ok", count=2)
    assert [a["status"] for a in attempts] == ["completed", "completed"]
    assert all(a["terminal"] and not a["technical_failure"] for a in attempts)
    trace = attempts[0]["trace"]
    assert (tmp_path / "run" / trace["path"]).is_file() and len(trace["sha256"]) == 64


def test_a_lost_worker_gets_one_identical_retry(tmp_path: Path, development_sources: SourceSet) -> None:
    attempts = run(tmp_path, development_sources, "die_once")
    assert [a["status"] for a in attempts] == ["worker_lost", "completed"]
    assert (
        attempts[1]["retry_of"] == attempts[0]["attempt_id"] and attempts[1]["retry_reason"] == "worker_lost"
    )


def test_a_second_transient_failure_stays_unresolved(tmp_path: Path, development_sources: SourceSet) -> None:
    attempts = run(tmp_path, development_sources, "device_error")
    assert [a["status"] for a in attempts] == ["device_error", "device_error"]
    assert attempts[1]["technical_failure"] and attempts[1]["terminal"]


def test_runner_errors_are_never_retried(tmp_path: Path, development_sources: SourceSet) -> None:
    attempts = run(tmp_path, development_sources, "runner_error")
    assert [a["status"] for a in attempts] == ["runner_error"]
    assert attempts[0]["technical_failure"]


def test_a_resumed_run_still_allows_only_one_retry() -> None:
    assert not job_decided([{"status": "worker_lost"}])
    assert job_decided([{"status": "worker_lost"}, {"status": "worker_lost"}])
    assert job_decided([{"status": "timeout"}])


def test_an_overrun_is_killed_and_recorded_as_timeout(tmp_path: Path, development_sources: SourceSet) -> None:
    job = frozen(development_sources, 1).jobs[0]
    supervisor = Supervisor(command(tmp_path, "sleep"), startup_timeout=30)
    try:
        result = supervisor.attempt(job.record(), timeout=0.5)
    finally:
        supervisor.close()
    assert result["status"] == "timeout" and result["wall_ms"] < 5000
    assert supervisor.worker is None


def test_a_worker_that_cannot_load_stops_the_run(tmp_path: Path, development_sources: SourceSet) -> None:
    with pytest.raises(WorkerStartupError, match="snapshot mismatch"):
        run(tmp_path, development_sources, "fatal")
