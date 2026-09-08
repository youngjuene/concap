"""A bounded, restartable inference process; no model work inside DB transactions."""

from __future__ import annotations

import io
import multiprocessing as mp
import signal
import threading
import time
import wave
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from dpo.core.atomic import replace_atomically
from dpo.regen.captions import in_language
from dpo.regen.study_store import StudyStore


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
                    from dpo.models.gemma4.prompt import stimulus_messages

                    if not Path(spec["excerpt"]).is_file():
                        excerpt(spec["audio"], spec["excerpt"], spec["start_ms"], spec["end_ms"])
                    messages = stimulus_messages(spec["instruction"], audio_reference=spec["excerpt"])
                    attempt["messages"] = messages
                    raw = adapter.generate_stimulus(
                        messages, temperature=0.0, top_p=1.0, max_new_tokens=96, seed=0
                    )
                    attempt["raw"] = raw
                    text = " ".join(raw.split())
                    if not text or len(text) > 160 or not in_language(text, spec["language"]):
                        raise ValueError("Generated caption failed language/length validation")
                    result = {"text": text, "raw": raw, "messages": messages, "fallback": False}
                else:
                    result = {"text": spec["fallback"], "fallback": True, "reason": "model_not_configured"}
            except Exception as exc:  # noqa: BLE001 — the job records failures and has an authored fallback
                result = {"text": spec["fallback"], "fallback": True, "reason": str(exc), **attempt}
            result["duration_ms"] = round((time.monotonic() - started) * 1000)
            pipe.send(result)
    except (EOFError, BrokenPipeError):
        return


class Supervisor:
    def __init__(self, store: StudyStore, settings: dict[str, Any], timeout: float = 30) -> None:
        self.store, self.settings, self.timeout = store, settings, timeout
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.process: Any = None
        self.pipe: Any = None

    def start(self) -> None:
        self.store.recover()
        self.thread.start()

    def reset(self) -> None:
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(2)
        if self.pipe is not None:
            self.pipe.close()
        self.process = self.pipe = None

    def stop(self) -> None:
        self.stopped.set()
        self.thread.join(3)

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
                    if self.process is None:
                        context = mp.get_context("spawn")
                        self.pipe, child = context.Pipe()
                        self.process = context.Process(
                            target=worker, args=(child, self.settings), daemon=True
                        )
                        self.process.start()
                        child.close()
                    self.pipe.send(spec)
                    deadline = time.monotonic() + self.timeout
                    while not self.pipe.poll(0.1):
                        if self.stopped.is_set() or time.monotonic() >= deadline:
                            raise TimeoutError("inference_deadline")
                    result = self.pipe.recv()
                except (TimeoutError, EOFError, BrokenPipeError, OSError) as exc:
                    self.reset()
                    result = {"text": spec["fallback"], "fallback": True, "reason": str(exc)}
                self.store.finish(job, result)
        finally:
            self.reset()
