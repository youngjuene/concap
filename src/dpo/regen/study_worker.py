"""A bounded, restartable inference process; no model work inside DB transactions."""

from __future__ import annotations

import io
import math
import multiprocessing as mp
import signal
import threading
import time
import wave
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from dpo.core.atomic import replace_atomically
from dpo.regen.study_store import StudyStore


class ModelPreparationError(RuntimeError):
    """The configured model could not become ready before participant entry."""


def excerpt(source: str, target: str, start_ms: int, end_ms: int) -> None:
    with wave.open(source, "rb") as audio:
        rate = audio.getframerate()
        start, end = round(start_ms * rate / 1000), round(end_ms * rate / 1000)
        if not 0 <= start < end <= audio.getnframes():
            raise ValueError("Audio excerpt is outside the staged audio")
        audio.setpos(start)
        output = io.BytesIO()
        with wave.open(output, "wb") as out:
            out.setparams(audio.getparams())
            out.writeframes(audio.readframes(end - start))
        replace_atomically(Path(target), output.getvalue())


def worker(pipe: Connection, settings: dict[str, Any]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # The parent owns shutdown and watchdog termination.
    adapter = None
    try:
        while True:
            spec = pipe.recv()
            if spec is None:
                return
            started = time.monotonic()
            attempt: dict[str, Any] = {}
            try:
                if settings.get("backend_config"):
                    if adapter is None:
                        from dpo.candidates.generation import verify_backend_pin
                        from dpo.contracts.study_contract import load_contract
                        from dpo.models.gemma4.adapter import GemmaCaptionAdapter
                        from dpo.models.gemma4.backend_config import load_config

                        verify_backend_pin(
                            load_contract(settings["contract"]),
                            track="audio",
                            backend_config_path=Path(settings["backend_config"]),
                        )
                        adapter = GemmaCaptionAdapter(
                            config=load_config(settings["backend_config"]),
                            contract=load_contract(settings["contract"]).tracks["audio"],
                            media_resolver=str,
                            adapter_dir=settings.get("checkpoint"),
                        )
                    if spec.get("kind") == "prepare":
                        model, _ = adapter._require_loaded()
                        pipe.send(
                            {
                                "ready": True,
                                "model_id": adapter.config.model.model_id,
                                "revision": adapter.config.model.revision,
                                "device": str(next(model.parameters()).device),
                                "duration_ms": round((time.monotonic() - started) * 1000),
                            }
                        )
                        continue
                    if spec.get("kind") == "stimulus":
                        raw = adapter.generate_stimulus(spec["messages"], **spec["decoding"])
                        pipe.send({"raw": raw})
                        continue
                    from dpo.models.gemma4.prompt import stimulus_messages

                    if not Path(spec["excerpt"]).is_file():
                        excerpt(spec["audio"], spec["excerpt"], spec["start_ms"], spec["end_ms"])
                    from dpo.regen.caption_controls import render_result, repair_instruction

                    attempts = []
                    for retry in range(2):
                        instruction = repair_instruction(spec) if retry else spec["instruction"]
                        messages = stimulus_messages(instruction, audio_reference=spec["excerpt"])
                        attempt["messages"] = messages
                        raw = adapter.generate_stimulus(
                            messages, temperature=0.0, top_p=1.0, max_new_tokens=96, seed=0
                        )
                        attempt["raw"] = raw
                        recorded: dict[str, Any] = {"raw": raw, "messages": messages}
                        attempts.append(recorded)
                        attempt["attempts"] = attempts
                        try:
                            rendered = render_result(raw, spec)
                            break
                        except ValueError as exc:
                            recorded["validation_error"] = str(exc)
                            if retry:
                                raise
                    result = {**rendered, **attempt, "fallback": False}
                else:
                    result = {"text": spec["fallback"], "fallback": True, "reason": "model_not_configured"}
            except Exception as exc:  # noqa: BLE001 — the job records failures and has an authored fallback
                result = (
                    {"ready": False, "reason": str(exc)}
                    if spec.get("kind") == "prepare"
                    else {"text": spec.get("fallback", ""), "fallback": True, "reason": str(exc), **attempt}
                )
            result["duration_ms"] = round((time.monotonic() - started) * 1000)
            pipe.send(result)
    except (EOFError, BrokenPipeError):
        return


class InferenceProcess:
    """Serialize both phases through one model, including queue and decode deadlines."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        target: Callable[..., None] | None = None,
        startup_timeout: float = 180,
    ) -> None:
        self.settings, self.target = settings, target or worker
        self.lock = threading.Lock()
        self.process: Any = None
        self.pipe: Any = None
        self.closed = False
        self.startup_timeout = self._startup_timeout(startup_timeout)
        self.readiness: dict[str, Any] | None = None
        self.readiness_events: list[dict[str, Any]] = []

    @staticmethod
    def _startup_timeout(value: float) -> float:
        if not math.isfinite(value) or not 0.1 <= value <= 600:
            raise ValueError("Model preparation timeout must be between 0.1 and 600 seconds")
        return value

    def _reset(self, *, graceful: bool = False) -> None:
        if self.process is not None:
            if graceful and self.process.is_alive():
                try:
                    self.pipe.send(None)
                    self.process.join(2)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(2)
        if self.pipe is not None:
            self.pipe.close()
        self.process = self.pipe = None
        self.readiness = None

    def _exchange(
        self, spec: dict[str, Any], deadline: float, stopped: threading.Event | None, phase: str
    ) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("Inference process is closed")
        if self.process is not None and not self.process.is_alive():
            self._reset()
        if self.process is None:
            context = mp.get_context("spawn")
            self.pipe, child = context.Pipe()
            self.process = context.Process(target=self.target, args=(child, self.settings), daemon=True)
            self.process.start()
            child.close()
        self.pipe.send(spec)
        while not self.pipe.poll(0.1):
            if (stopped is not None and stopped.is_set()) or time.monotonic() >= deadline:
                raise TimeoutError(f"{phase}_deadline")
        return dict(self.pipe.recv())

    def _prepare_locked(self, timeout: float, stopped: threading.Event | None) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("Inference process is closed")
        if not self.settings.get("backend_config"):
            return {"ready": True, "model_configured": False, "duration_ms": 0}
        if self.readiness and self.process is not None and self.process.is_alive():
            return dict(self.readiness)
        started = time.monotonic()
        record: dict[str, Any] = {
            "started_at": time.time(),
            "startup_timeout_seconds": timeout,
            "model_configured": True,
        }
        try:
            result = self._exchange({"kind": "prepare"}, started + timeout, stopped, "model_preparation")
            if result.get("ready") is not True:
                raise ModelPreparationError(str(result.get("reason", "Model readiness was not confirmed")))
            record.update(ready=True, worker_pid=self.process.pid, model=result)
            self.readiness = record
        except (RuntimeError, TimeoutError, EOFError, BrokenPipeError, OSError) as exc:
            record.update(ready=False, reason=str(exc), error_type=type(exc).__name__)
            self._reset()
            raise
        finally:
            record["duration_ms"] = round((time.monotonic() - started) * 1000)
            self.readiness_events.append(dict(record))
        return dict(record)

    def prepare(self, timeout: float | None = None, stopped: threading.Event | None = None) -> dict[str, Any]:
        budget = self._startup_timeout(self.startup_timeout if timeout is None else timeout)
        deadline = time.monotonic() + budget
        if not self.lock.acquire(timeout=budget):
            raise TimeoutError("model_preparation_queue_deadline")
        try:
            return self._prepare_locked(max(0, deadline - time.monotonic()), stopped)
        finally:
            self.lock.release()

    def infer(
        self, spec: dict[str, Any], timeout: float, stopped: threading.Event | None = None
    ) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + timeout
        if not self.lock.acquire(timeout=timeout):
            raise TimeoutError("inference_queue_deadline")
        try:
            remaining = max(0, deadline - time.monotonic())
            if self.settings.get("backend_config"):
                # Initial loading and recovery loading have their own bounded
                # readiness phase. Queue time still counts against caption work.
                self._prepare_locked(self.startup_timeout, stopped)
                deadline = time.monotonic() + remaining
            return self._exchange(spec, deadline, stopped, "inference")
        except (TimeoutError, EOFError, BrokenPipeError, OSError):
            self._reset()
            raise
        finally:
            self.lock.release()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self._reset(graceful=True)


class SharedAdapter:
    """The existing GemmaWriter API backed by the viewing worker's model."""

    def __init__(self, engine: InferenceProcess, adapter: Any, timeout: float = 30) -> None:
        self.engine, self.timeout = engine, timeout
        self.config, self.adapter_dir = adapter.config, adapter.adapter_dir

    def generate_stimulus(self, messages: list[dict[str, Any]], **decoding: Any) -> str:
        result = self.engine.infer(
            {"kind": "stimulus", "messages": messages, "decoding": decoding}, self.timeout
        )
        if result.get("fallback"):
            raise RuntimeError(result["reason"])
        return str(result["raw"])


class Supervisor:
    def __init__(
        self,
        store: StudyStore,
        settings: dict[str, Any],
        timeout: float = 30,
        engine: InferenceProcess | None = None,
        *,
        startup_timeout: float = 180,
    ) -> None:
        self.store, self.timeout, self.settings = store, timeout, settings
        self.startup_timeout = InferenceProcess._startup_timeout(startup_timeout)
        self.engine = engine or InferenceProcess(settings, startup_timeout=startup_timeout)
        if isinstance(self.engine, InferenceProcess):
            self.engine.startup_timeout = self.startup_timeout
        self.readiness: dict[str, Any] = {"ready": False}
        self.owns_engine = engine is None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        if self.settings.get("backend_config"):
            prepare = getattr(self.engine, "prepare", None)
            if not callable(prepare):
                raise ModelPreparationError("Configured model engine must confirm readiness with prepare()")
            self.readiness = prepare(timeout=self.startup_timeout, stopped=self.stopped)
            if (
                not isinstance(self.readiness, dict)
                or self.readiness.get("ready") is not True
                or self.readiness.get("model_configured") is False
            ):
                raise ModelPreparationError("Configured model engine did not confirm readiness")
        else:
            self.readiness = {"ready": True, "model_configured": False, "duration_ms": 0}
        self.store.recover()
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(self.timeout + 5)
        if self.owns_engine and not self.thread.is_alive():
            self.engine.close()

    def run(self) -> None:
        previous = None
        try:
            while not self.stopped.is_set():
                job = self.store.claim(previous)
                if job is None:
                    self.stopped.wait(0.1)
                    continue
                previous = job["session"]
                import json

                spec = json.loads(job["spec"])
                try:
                    result = self.engine.infer(spec, self.timeout, self.stopped)
                except (ModelPreparationError, TimeoutError, EOFError, BrokenPipeError, OSError) as exc:
                    result = {"text": spec["fallback"], "fallback": True, "reason": str(exc)}
                preparation = getattr(self.engine, "readiness_events", [])
                if preparation:
                    result["model_readiness"] = preparation[-1]
                self.store.finish(job, result)
        finally:
            if self.owns_engine:
                self.engine.close()
