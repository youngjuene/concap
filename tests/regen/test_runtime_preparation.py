"""Runtime readiness cannot spend the caption budget or block the first video request."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from dpo.regen import study_media
from dpo.regen.study_api import build_study_app
from dpo.regen.study_schema import fingerprint
from dpo.regen.study_store import StudyStore
from dpo.regen.study_worker import InferenceProcess, Supervisor
from dpo.regen.tests.test_study import assets as _assets
from dpo.regen.tests.test_viewing_prepare import _video

assets = _assets


def readiness_worker(pipe: Any, settings: Any) -> None:
    preparations = 0
    while True:
        spec = pipe.recv()
        if spec is None:
            return
        if spec.get("kind") == "prepare":
            time.sleep(settings.get("startup_delay", 0.6))
            preparations += 1
            pipe.send(
                {
                    "ready": not settings.get("startup_fails"),
                    "reason": "fixture-startup-failure",
                    "pid": os.getpid(),
                    "preparations": preparations,
                }
            )
        else:
            time.sleep(spec.get("delay", 0))
            pipe.send(
                {"pid": os.getpid(), "preparations": preparations, "fallback": False, "text": "준비된 소리"}
            )


def test_cold_model_readiness_does_not_consume_caption_budget() -> None:
    engine = InferenceProcess({"backend_config": "fixture"}, target=readiness_worker)
    try:
        prepared = engine.prepare(timeout=5)
        assert prepared["ready"] is True
        result = engine.infer({"delay": 0.01}, timeout=0.2)
        assert result["pid"] == prepared["worker_pid"]
        assert result["preparations"] == 1
        assert engine.prepare(timeout=5)["worker_pid"] == prepared["worker_pid"]
        assert len(engine.readiness_events) == 1
        with pytest.raises(TimeoutError, match="inference_deadline"):
            engine.infer({"delay": 1}, timeout=0.1)
        assert engine.process is None
        # A replacement process also gets a separate startup budget; the
        # generation limit remains 0.2 seconds, below the 0.6 second model load.
        resumed = engine.infer({"delay": 0.01}, timeout=0.2)
        assert resumed["pid"] != result["pid"] and resumed["preparations"] == 1
        assert len(engine.readiness_events) == 2
    finally:
        engine.close()


def test_model_preparation_failure_is_explicit_and_does_not_leave_worker() -> None:
    engine = InferenceProcess(
        {"backend_config": "fixture", "startup_fails": True, "startup_delay": 0}, target=readiness_worker
    )
    try:
        with pytest.raises(RuntimeError, match="fixture-startup-failure"):
            engine.prepare(timeout=5)
        assert engine.process is None
        assert engine.readiness_events[-1]["ready"] is False
    finally:
        engine.close()


def test_model_preparation_has_its_own_bounded_timeout() -> None:
    engine = InferenceProcess({"backend_config": "fixture", "startup_delay": 10}, target=readiness_worker)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="preparation"):
            engine.prepare(timeout=0.1)
        assert time.monotonic() - started < 5
        assert engine.process is None
        assert engine.readiness_events[-1]["ready"] is False
    finally:
        engine.close()


def test_real_backend_cannot_silently_skip_engine_readiness(tmp_path: Path) -> None:
    class OldFakeEngine:
        def infer(self, *_: Any) -> dict[str, Any]:
            return {"fallback": False}

        def close(self) -> None:
            pass

    engine: Any = OldFakeEngine()
    supervisor = Supervisor(
        StudyStore(tmp_path / "study.sqlite3"), {"backend_config": "fixture"}, engine=engine
    )
    try:
        with pytest.raises(RuntimeError, match="readiness|prepare"):
            supervisor.start()
    finally:
        supervisor.stop()


def test_delivery_preparation_reuses_cache_and_preserves_original(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source.mp4"
    _video(source, 1)
    original_hash = fingerprint(source)
    manifest = {
        "calibration_clips": [],
        "viewing_videos": [
            {"id": "one", "status": "ready", "video": source.name, "media_hashes": {"video": original_hash}}
        ],
    }
    cache = tmp_path / "delivery"
    first = study_media.prepare_delivery_videos(manifest, tmp_path, cache)
    assert len(first) == 1 and first[0]["cache_hit"] is False
    delivered = Path(first[0]["path"])
    assert delivered != source and delivered.is_file()
    assert fingerprint(source) == original_hash
    assert first[0]["sha256"] == fingerprint(delivered)
    monkeypatch.setattr(
        "dpo.regen.study_media.subprocess.run", lambda *_a, **_kw: pytest.fail("cached media re-encoded")
    )
    second = study_media.prepare_delivery_videos(manifest, tmp_path, cache)
    assert second[0]["cache_hit"] is True
    assert study_media.prepared_video(source, cache, video_rate=1_600_000) == delivered


def test_request_path_refuses_unprepared_delivery_instead_of_encoding(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source fixture; route must not invoke encoder")
    monkeypatch.setattr(
        "dpo.regen.study_media.subprocess.run", lambda *_a, **_kw: pytest.fail("request encoded media")
    )
    with pytest.raises(ValueError, match="prepared"):
        study_media.prepared_video(source, tmp_path / "empty", video_rate=1_600_000)


def test_startup_prepares_delivery_and_real_model_before_accepting_participants(
    assets: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    root, manifest = assets
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    backend = tmp_path / "backend.toml"
    backend.write_text("# readiness is simulated at the process boundary\n")
    calls: list[tuple[str, float | None]] = []

    class ReadyEngine:
        def prepare(self, timeout: float, stopped: Any = None) -> dict[str, Any]:
            calls.append(("model", timeout))
            return {"ready": True, "worker_pid": 12345, "duration_ms": 600}

        def infer(self, *_: Any) -> dict[str, Any]:
            pytest.fail("No participant caption job exists during startup")

        def close(self) -> None:
            pass

    def prepare_media(*_: Any) -> list[dict[str, Any]]:
        calls.append(("delivery", None))
        return [{"id": "long-0", "cache_hit": True}]

    monkeypatch.setattr("dpo.regen.study_api.prepare_delivery_videos", prepare_media)
    engine: Any = ReadyEngine()
    app = build_study_app(path, root, tmp_path / "out", {"backend_config": str(backend)}, engine=engine)

    async def startup() -> None:
        async with app.router.lifespan_context(app):
            assert calls == [("delivery", None), ("model", 180)]
            assert app.state.preparation["ready"] is True
            assert app.state.preparation["media"][0]["cache_hit"] is True
            assert app.state.preparation["model"]["worker_pid"] == 12345
            evidence = json.loads((tmp_path / "out/preparation.json").read_text())
            assert evidence["ready"] is True and evidence["inference_timeout_seconds"] == 30

    asyncio.run(startup())


def test_unconfigured_engine_cannot_claim_configured_model_readiness(tmp_path: Path) -> None:
    engine = InferenceProcess({})
    supervisor = Supervisor(
        StudyStore(tmp_path / "study.sqlite3"), {"backend_config": "fixture"}, engine=engine
    )
    try:
        with pytest.raises(RuntimeError, match="readiness"):
            supervisor.start()
        assert engine.process is None
    finally:
        supervisor.stop()
        engine.close()


def test_startup_failure_is_recorded_before_lifespan_yields(assets: Any, tmp_path: Path) -> None:
    root, manifest = assets
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    backend = tmp_path / "backend.toml"
    backend.write_text("# readiness failure fixture\n")

    class FailedEngine:
        readiness_events = [{"ready": False, "reason": "model fixture failed", "duration_ms": 20}]

        def prepare(self, timeout: float, stopped: Any = None) -> dict[str, Any]:
            raise RuntimeError("model fixture failed")

        def close(self) -> None:
            pass

    engine: Any = FailedEngine()
    app = build_study_app(path, root, tmp_path / "out", {"backend_config": str(backend)}, engine=engine)

    async def startup() -> None:
        with pytest.raises(RuntimeError, match="model fixture failed"):
            async with app.router.lifespan_context(app):
                pytest.fail("Participant entry must wait for model readiness")

    asyncio.run(startup())
    evidence = json.loads((tmp_path / "out/preparation.json").read_text())
    assert evidence["ready"] is False and evidence["phase"] == "failed"
    assert evidence["model"]["ready"] is False and evidence["model"]["duration_ms"] == 20
    assert evidence["inference_timeout_seconds"] == 30
