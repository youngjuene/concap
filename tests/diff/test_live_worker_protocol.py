"""Live (phase-2) jobs: frozen inside the worker, answered on the same protocol (2026-10-11)."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest
from fake_trida_model import MASK, FakeTokenizer, FakeTrida

from diff import trida_runner
from diff.caption_sources import SourceSet
from diff.generation_worker import Supervisor, WorkerCommand, serve
from diff.job_plan import Job, PlanError, build_phase1_plan
from diff.model_profile import TRIDA_7B
from diff.model_runtime import LoadedModel, chat_token_ids
from diff.study_protocol import sha256_tokens

SCRIPT = Path(__file__).parent / "fake_generation_worker.py"


def a_job(sources: SourceSet) -> Job:
    return build_phase1_plan(sources.sources, ("trida_tau_090",), "development").jobs[0]


def loaded(text: str = "버스가 지나간다.") -> LoadedModel:
    tokenizer = FakeTokenizer()
    return LoadedModel(
        model=FakeTrida(text, tokenizer=tokenizer), tokenizer=tokenizer, device="cpu", profile=TRIDA_7B
    )


def test_the_worker_freezes_a_live_job_with_its_own_tokenizer(development_sources: SourceSet) -> None:
    model, job = loaded(), a_job(development_sources)
    frozen = trida_runner.freeze_live(model, job)
    ids = chat_token_ids(model.tokenizer, job.prompt)
    assert frozen.token_ids_sha256 == sha256_tokens(ids) and frozen.token_count == len(ids)
    assert trida_runner.run_job(model, frozen)["status"] == "completed"


class MaskingTokenizer(FakeTokenizer):
    """Renders every message as the mask token, which the guards must refuse."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [MASK]


def test_a_live_prompt_still_meets_the_guards(development_sources: SourceSet) -> None:
    tokenizer = MaskingTokenizer()
    model = LoadedModel(
        model=FakeTrida("x", tokenizer=tokenizer), tokenizer=tokenizer, device="cpu", profile=TRIDA_7B
    )
    with pytest.raises(PlanError, match="mask token"):
        trida_runner.freeze_live(model, a_job(development_sources))


def serve_once(message: dict[str, object], **kwargs: object) -> dict[str, object]:
    out = io.StringIO()
    stdin = io.StringIO(json.dumps(message) + "\n" + json.dumps({"type": "stop"}) + "\n")
    serve(lambda: (loaded(), {}), trida_runner.run_job, out, stdin, **kwargs)  # type: ignore[arg-type]
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert lines[0]["type"] == "ready"
    return lines[1]


def test_serve_answers_a_live_message_with_the_token_hash(development_sources: SourceSet) -> None:
    job = a_job(development_sources)
    result = serve_once({"type": "live", "job": job.record()}, freeze=trida_runner.freeze_live)
    assert result["status"] == "completed" and result["job_id"] == job.job_id
    assert isinstance(result["token_ids_sha256"], str) and result["rng_unchanged"] is True


def test_a_worker_without_freeze_refuses_live_jobs(development_sources: SourceSet) -> None:
    result = serve_once({"type": "live", "job": a_job(development_sources).record()})
    assert result["status"] == "refused" and "live" in str(result["error"])


def test_the_supervisor_sends_live_messages(tmp_path: Path, development_sources: SourceSet) -> None:
    env = {**os.environ, "FAKE_WORKER_BEHAVIOUR": "ok"}
    command = WorkerCommand(
        argv=(sys.executable, str(SCRIPT)), env=env, cwd=tmp_path, log_path=tmp_path / "worker.log"
    )
    supervisor = Supervisor(command, startup_timeout=30)
    try:
        result = supervisor.attempt(a_job(development_sources).record(), 30, kind="live")
        assert result["status"] == "completed" and result["kind"] == "live"
        with pytest.raises(ValueError, match="kind"):
            supervisor.attempt({"job_id": "x"}, 30, kind="other")
    finally:
        supervisor.close()
