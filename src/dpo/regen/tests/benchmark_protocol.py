"""Bounded no-model protocol performance evidence for long-viewing settings."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from dpo.regen.study_api import build_study_app
from dpo.regen.tests.test_study import Session
from dpo.regen.tests.test_study import assets as _assets

assets = _assets

OUTPUT = Path("/tmp/regen-protocol-performance.json")
REPETITIONS = 20


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _api_post(session: Session, action: str, data: dict[str, Any], key: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = session.client.post(
        f"/api/study/{action}",
        json={"revision": session.state["revision"], "key": key, "data": data},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert response.status_code == 200, response.text
    session.state = response.json()
    return session.state, elapsed_ms


def _job_counts(store: Any, token: str) -> dict[str, int]:
    exported = store.export(token)
    counts: dict[str, int] = {"total": len(exported["jobs"])}
    for job in exported["jobs"]:
        state = str(job["state"])
        counts[state] = counts.get(state, 0) + 1
    return counts


def _run_case(tmp_path: Path, assets: Any, *, protocol3: bool) -> dict[str, Any]:
    root, manifest = assets
    path = tmp_path / ("protocol3.json" if protocol3 else "legacy.json")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    app = build_study_app(path, root, tmp_path / ("out-protocol3" if protocol3 else "out-legacy"))
    session = Session(TestClient(app))
    session.calibrate()
    session.submit("start-viewing")
    if not protocol3:
        app.state.store.mutate(
            session.token,
            "remove-protocol-version",
            session.state["revision"],
            "legacy-fixture",
            lambda state: state.pop("protocol_version", None),
        )
        session.state = session.client.get("/api/study/state").json()

    settings_latencies = []
    playback_latencies = []
    settings_payload = {
        "video_id": "long-0",
        "texture": 0.75,
        "context": 0.25,
        "position_hint_ms": 45000,
        "origin": "slider:texture",
    }
    for index in range(REPETITIONS):
        _, elapsed = _api_post(session, "settings", settings_payload, f"settings-{index}")
        settings_latencies.append(elapsed)

    playback_clock = [2000.0]
    with patch("dpo.regen.playback.time.time", lambda: playback_clock[0]):
        for sequence, position in enumerate((0, 1000, 2000, 3000, 4000)):
            playback_clock[0] += 1
            _, elapsed = _api_post(
                session,
                "playback",
                {
                    "video_id": "long-0",
                    "position_ms": position,
                    "sequence": sequence,
                    "playing": False,
                },
                f"playback-{sequence}",
            )
            playback_latencies.append(elapsed)

    state = app.state.store.state(session.token)
    jobs = _job_counts(app.state.store, session.token)
    return {
        "label": "protocol3" if protocol3 else "historical_missing_protocol_version",
        "requests": {"settings": REPETITIONS, "playback": len(playback_latencies)},
        "api_ack_ms": {
            "settings_p50": round(statistics.median(settings_latencies), 3),
            "settings_p95": round(_percentile(settings_latencies, 0.95), 3),
            "playback_p50": round(statistics.median(playback_latencies), 3),
            "playback_p95": round(_percentile(playback_latencies, 0.95), 3),
        },
        "state": {
            "revision": state["revision"],
            "settings_revision": state["settings_revision"],
            "epoch": state["epoch"],
            "position_ms": state["position_ms"],
        },
        "jobs": jobs,
        "work_count": jobs["total"],
    }


def test_no_model_protocol_settings_benchmark(assets: Any, tmp_path: Path) -> None:
    protocol3 = _run_case(tmp_path, assets, protocol3=True)
    legacy = _run_case(tmp_path, assets, protocol3=False)
    document = {
        "benchmark": "no-model long-viewing settings orchestration",
        "cannot_infer": [
            "GPU caption latency",
            "caption quality",
            "worker throughput with a real model",
        ],
        "driver_status": (
            "nvidia-smi cannot communicate in this environment; actual GPU qualification not run"
        ),
        "baseline_fairness": (
            "Both cases use identical synthetic media, no model worker/lifespan, the same TestClient path, "
            "20 identical settings requests, and the same playback acknowledgements. The legacy baseline is "
            "simulated by removing protocol_version from the saved session after start-viewing."
        ),
        "cases": [protocol3, legacy],
        "comparison": {
            "settings_revision_saved": legacy["state"]["settings_revision"]
            - protocol3["state"]["settings_revision"],
            "work_count_saved": legacy["work_count"] - protocol3["work_count"],
        },
    }
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    assert protocol3["state"]["settings_revision"] == 0
    assert legacy["state"]["settings_revision"] == REPETITIONS
    assert protocol3["work_count"] <= legacy["work_count"]
