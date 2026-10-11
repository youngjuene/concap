"""Generation in a supervised subprocess: deadlines, kills and the one retry (§6.2).

Each model lives in a worker process (``python -m diff worker``) that speaks
JSON lines: a ``ready`` or ``fatal`` message after loading and warm-up, then
one ``result`` per ``job``. The parent enforces the per-cue deadline from the
outside — native generate cannot be interrupted from inside — by killing a
worker whose job overruns, recording ``timeout``, and starting a fresh worker
(model load and warm-up included) for the next job.

Retry policy: exactly one identical retry, with a fresh worker, and only after a
transient failure — the worker process died (``worker_lost``) or the device
failed (``device_error``). Nothing about output quality triggers a retry, and
neither do ``timeout``, ``refused``, ``runner_error`` or
``trace_inconsistent``. The decision is made from the attempt history on disk,
so a run resumed after a crash still allows only one retry per job; a retry
that fails transiently again leaves an unresolved technical failure, which G1
counts.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from diff.job_plan import FrozenJob, FrozenPlan, PlanError, job_from_record
from diff.model_profile import ProfileMismatch
from diff.study_protocol import sha256_of

ATTEMPT_SCHEMA: Final = "diff.attempt/v1"
DECODE_EVENT_SCHEMA: Final = "diff.decode-event/v2"
TRANSIENT: Final = frozenset({"worker_lost", "device_error"})
# count_mismatch: forward calls or new positions differ from steps/canvas (§6.2);
# rng_consumed: a greedy (Qwen) call changed an RNG state (AC-07); leftover_mask: a mask
# token survived in the canvas. None of these is transient, so none is retried.
TECHNICAL_FAILURES: Final = frozenset(
    {
        "worker_lost",
        "device_error",
        "runner_error",
        "trace_inconsistent",
        "count_mismatch",
        "rng_consumed",
        "leftover_mask",
    }
)
STATUSES: Final = frozenset({"completed", "timeout", "refused"}) | TECHNICAL_FAILURES


class WorkerStartupError(RuntimeError):
    """The worker could not load and warm up its model; the run stops."""


@dataclass(frozen=True)
class WorkerCommand:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path
    log_path: Path


class WorkerProcess:
    """One worker subprocess and a reader thread that queues its protocol lines."""

    def __init__(self, command: WorkerCommand) -> None:
        command.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = command.log_path.open("ab")
        self.process = subprocess.Popen(
            list(command.argv),
            cwd=command.cwd,
            env=dict(command.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, message: Mapping[str, Any]) -> bool:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    def receive(self, timeout: float) -> dict[str, Any] | None:
        """The next protocol message; ``None`` at end of output; ``TimeoutError`` on deadline."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError from exc
            if line is None:
                return None
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and "type" in message:
                return message

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self._log.close()

    def stop(self, grace: float = 10.0) -> None:
        self.send({"type": "stop"})
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=grace)
        self.kill()


class Supervisor:
    """Starts workers on demand and turns one job into one attempt result."""

    def __init__(self, command: WorkerCommand, *, startup_timeout: float) -> None:
        self.command, self.startup_timeout = command, startup_timeout
        self.worker: WorkerProcess | None = None
        self.ready: dict[str, Any] = {}

    def _start(self) -> WorkerProcess:
        worker = WorkerProcess(self.command)
        try:
            message = worker.receive(self.startup_timeout)
        except TimeoutError as exc:
            worker.kill()
            raise WorkerStartupError(f"no ready message within {self.startup_timeout} s") from exc
        if message is None or message.get("type") != "ready":
            worker.kill()
            detail = message.get("error") if message else "the worker exited"
            raise WorkerStartupError(f"worker failed to start: {detail}")
        self.worker, self.ready = worker, message
        return worker

    def restart(self) -> None:
        if self.worker is not None:
            self.worker.kill()
            self.worker = None

    def attempt(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        worker = self.worker or self._start()
        started_at, started = time.time(), time.monotonic()

        def finish(result: dict[str, Any]) -> dict[str, Any]:
            return {
                **result,
                "started_at": started_at,
                "finished_at": time.time(),
                "wall_ms": round((time.monotonic() - started) * 1000),
            }

        if not worker.send({"type": "job", "job": payload}):
            self.restart()
            return finish({"status": "worker_lost", "error": "the worker closed its input"})
        while True:
            try:
                message = worker.receive(timeout - (time.monotonic() - started))
            except TimeoutError:
                self.restart()
                return finish({"status": "timeout", "error": f"no result within {timeout} s; worker killed"})
            if message is None:
                self.worker = None
                worker.kill()
                return finish({"status": "worker_lost", "error": "the worker exited during the job"})
            if message.get("type") == "result" and message.get("job_id") == payload["job_id"]:
                result = {key: value for key, value in message.items() if key not in ("type", "job_id")}
                if result.get("status") not in STATUSES:
                    result = {
                        **result,
                        "status": "runner_error",
                        "error": f"unknown status {result.get('status')!r}",
                    }
                if result["status"] in TRANSIENT:
                    self.restart()
                return finish(result)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None


def read_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def job_decided(history: Iterable[Mapping[str, Any]]) -> bool:
    """A job is decided once any attempt was not transient, or after the one retry."""
    attempts = list(history)
    return any(attempt["status"] not in TRANSIENT for attempt in attempts) or len(attempts) >= 2


def _append_line(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_trace(run_dir: Path, attempt_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    path = run_dir / "traces" / f"{attempt_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps({"schema": DECODE_EVENT_SCHEMA, "attempt_id": attempt_id, **event}, ensure_ascii=False)
        + "\n"
        for event in events
    )
    path.write_text(body, encoding="utf-8")
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def attempt_record(
    *,
    run_id: str,
    frozen: FrozenJob,
    number: int,
    previous: Mapping[str, Any] | None,
    result: Mapping[str, Any],
    ready: Mapping[str, Any],
    verification: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    job = frozen.job
    attempt_id = "att-" + sha256_of([run_id, job.job_id, number])[:24]
    trace = dict(result.get("trace") or {})
    events = trace.pop("events", None)
    if events:
        trace.update(_write_trace(run_dir, attempt_id, events))
    status = str(result["status"])
    return {
        "schema": ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "job_id": job.job_id,
        "profile_id": job.profile_id,
        "attempt_number": number,
        "retry_of": previous["attempt_id"] if previous else None,
        "retry_reason": previous["status"] if previous else None,
        "environment": ready.get("environment"),
        "verified_files": verification.get("files"),
        "verified_weights": {
            name: entry.get("sha256") for name, entry in verification.get("weights", {}).items()
        },
        "loaded_code": ready.get("loaded_code"),
        "warmup": ready.get("warmup"),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "wall_ms": result.get("wall_ms"),
        "status": status,
        "error": result.get("error"),
        "prompt_tokens": result.get("prompt_tokens"),
        "output": result.get("output"),
        "output_token_ids_sha256": result.get("output_token_ids_sha256"),
        "resolved_config": result.get("resolved_config"),
        "caps": result.get("caps"),
        "rng_unchanged": result.get("rng_unchanged"),
        "seed": result.get("seed"),
        "latency_ms": result.get("latency_ms"),
        "trace": trace or None,
        "terminal": status not in TRANSIENT or number >= 2,
        "technical_failure": status in TECHNICAL_FAILURES and (status not in TRANSIENT or number >= 2),
    }


def run_plan(
    frozen: FrozenPlan,
    run_dir: Path,
    commands: Mapping[str, WorkerCommand],
    verifications: Mapping[str, Mapping[str, Any]],
    *,
    startup_timeout: float,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every undecided job, grouped by model so each model loads once per run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    history: dict[str, list[dict[str, Any]]] = {}
    for attempt in read_attempts(attempts_path):
        history.setdefault(attempt["job_id"], []).append(attempt)
    run_id = "run-" + uuid.uuid4().hex[:16]
    counts: dict[str, int] = {}
    for profile_id in sorted({job.job.profile_id for job in frozen.jobs}):
        pending = [
            job
            for job in frozen.jobs
            if job.job.profile_id == profile_id and not job_decided(history.get(job.job.job_id, []))
        ]
        if not pending:
            continue
        supervisor = Supervisor(commands[profile_id], startup_timeout=startup_timeout)
        try:
            for index, job in enumerate(pending, 1):
                prior = history.setdefault(job.job.job_id, [])
                while not job_decided(prior):
                    result = supervisor.attempt(job.record(), float(job.job.caps["timeout_s"]))
                    record = attempt_record(
                        run_id=run_id,
                        frozen=job,
                        number=len(prior) + 1,
                        previous=prior[-1] if prior else None,
                        result=result,
                        ready=supervisor.ready,
                        verification=verifications[profile_id],
                        run_dir=run_dir,
                    )
                    _append_line(attempts_path, record)
                    prior.append(record)
                    counts[record["status"]] = counts.get(record["status"], 0) + 1
                    log(
                        f"{profile_id} {index}/{len(pending)} {job.job.job_id} "
                        f"{record['status']} {record['wall_ms']} ms"
                    )
        finally:
            supervisor.close()
    return {"run_id": run_id, "attempts": counts, "attempts_path": str(attempts_path)}


def classify(exc: BaseException) -> str:
    """The attempt status a worker-side exception earns."""
    if isinstance(exc, (PlanError, ProfileMismatch)) or type(exc).__name__ == "RefusedCall":
        return "refused"
    text, name = str(exc), type(exc).__name__
    if name == "OutOfMemoryError" or any(marker in text for marker in ("CUDA", "CUBLAS", "cuDNN", "NCCL")):
        return "device_error"
    return "runner_error"


def serve(
    load: Callable[[], tuple[Any, dict[str, Any]]],
    run_job: Callable[[Any, FrozenJob], dict[str, Any]],
    out: IO[str],
    stdin: IO[str],
) -> int:
    """The worker side: load once, then answer each job line with one result line."""

    def emit(message: Mapping[str, Any]) -> None:
        out.write(json.dumps(message, ensure_ascii=False) + "\n")
        out.flush()

    try:
        loaded, warmup = load()
    except Exception as exc:  # noqa: BLE001 — reported to the parent, which stops the run
        emit({"type": "fatal", "status": classify(exc), "error": f"{type(exc).__name__}: {exc}"})
        return 1
    emit(
        {
            "type": "ready",
            "profile_id": loaded.profile.profile_id,
            "environment": loaded.environment,
            "loaded_code": loaded.loaded_code,
            "warmup": warmup,
        }
    )
    for line in stdin:
        message = json.loads(line)
        if message.get("type") == "stop":
            break
        record = message["job"]
        try:
            frozen = FrozenJob(
                job_from_record(record), str(record["token_ids_sha256"]), int(record["token_count"])
            )
            result = run_job(loaded, frozen)
        except Exception as exc:  # noqa: BLE001 — every failure becomes a recorded attempt status
            result = {"status": classify(exc), "error": f"{type(exc).__name__}: {exc}"}
        emit({"type": "result", "job_id": record.get("job_id"), **result})
        if result.get("status") == "device_error":
            break
    return 0
