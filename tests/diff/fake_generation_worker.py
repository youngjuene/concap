"""A stand-in worker speaking diff's JSON-lines protocol, for the supervisor tests.

Run as a script by the tests. FAKE_WORKER_BEHAVIOUR selects what it does with
each job; FAKE_WORKER_MARKER names a file used to fail only the first time.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

behaviour = os.environ.get("FAKE_WORKER_BEHAVIOUR", "ok")
marker = Path(os.environ.get("FAKE_WORKER_MARKER", "/nonexistent"))


def emit(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


if behaviour == "fatal":
    emit({"type": "fatal", "status": "refused", "error": "snapshot mismatch"})
    sys.exit(1)
print("library chatter that is not protocol")
emit({"type": "ready", "environment": {"fake": True}, "loaded_code": {}, "warmup": {}})
for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "stop":
        break
    job = message["job"]
    if behaviour == "sleep":
        time.sleep(30)
    if behaviour == "die" or (behaviour == "die_once" and not marker.exists()):
        marker.write_text("failed once", encoding="utf-8")
        os._exit(3)
    if behaviour in ("device_error", "runner_error"):
        emit({"type": "result", "job_id": job["job_id"], "status": behaviour, "error": f"fake {behaviour}"})
        continue
    emit(
        {
            "type": "result",
            "job_id": job["job_id"],
            "status": "completed",
            "error": None,
            "prompt_tokens": 10,
            "output": {"text": "버스가 지나간다."},
            "latency_ms": 5,
            "kind": message["type"],
            "trace": {
                "events": [{"kind": "step", "seq": 0}],
                "level": "history-counts",
                "summary": {},
                "problems": [],
            },
        }
    )
