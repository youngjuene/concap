"""One killable process serves calibration and long-viewing requests."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from dpo.regen.study_worker import InferenceProcess


def fake_worker(pipe: Any, settings: Any) -> None:
    while True:
        spec = pipe.recv()
        if spec is None:
            return
        time.sleep(spec.get("delay", 0))
        pipe.send({"pid": os.getpid(), "request": spec["kind"]})


def test_shared_process_serializes_calibration_and_viewing() -> None:
    engine = InferenceProcess({}, target=fake_worker)
    try:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda kind: engine.infer({"kind": kind}, 5), ["calibration", "viewing"]))
        assert [r["request"] for r in results] == ["calibration", "viewing"]
        assert len({r["pid"] for r in results}) == 1
    finally:
        engine.close()
    assert engine.process is None


def test_watchdog_restarts_the_shared_process() -> None:
    engine = InferenceProcess({}, target=fake_worker)
    try:
        before = engine.infer({"kind": "calibration"}, 5)
        with pytest.raises(TimeoutError):
            engine.infer({"kind": "hung", "delay": 10}, 0.2)
        after = engine.infer({"kind": "viewing"}, 5)
        assert after["pid"] != before["pid"]
    finally:
        engine.close()


def test_queue_deadline_does_not_cancel_another_call() -> None:
    engine = InferenceProcess({}, target=fake_worker)
    try:
        engine.infer({"kind": "warm"}, 5)
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(engine.infer, {"kind": "calibration", "delay": 0.8}, 5)
            time.sleep(0.1)
            with pytest.raises(TimeoutError, match="queue"):
                engine.infer({"kind": "viewing"}, 0.1)
            assert pending.result()["request"] == "calibration"
    finally:
        engine.close()
