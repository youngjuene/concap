"""Real-media integrated QA server for the final regen viewing survey.

The ready viewing manifest and extracted WAVs are prepared outside this file.
This wrapper serves that exact artifact against the normal legacy-calibration
continuation flow, while keeping all QA response records under /tmp.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse

from dpo.regen.app import build_app
from dpo.regen.continuation import Continuation, ViewingConfig
from dpo.regen.regeneration import RegenTemplateWriter
from dpo.regen.study_schema import digest, fingerprint, load_manifest
from dpo.regen.study_worker import InferenceProcess

LIVE_ROOT = Path(os.environ.get("REGEN_REAL_LIVE_ROOT", "/mnt/hdd/research/2026/concap/data/live"))
MEDIA_ROOT = Path(os.environ.get("REGEN_REAL_MEDIA_ROOT", LIVE_ROOT / "regen-media"))
LEGACY_DOCUMENT = Path(os.environ.get("REGEN_REAL_LEGACY_DOCUMENT", LIVE_ROOT / "regen.json"))
READY_MANIFEST = Path(
    os.environ.get("REGEN_REAL_READY_MANIFEST", "/tmp/regen-real-video-20260909/viewing-study.ready.json")
)
PORT = int(os.environ.get("REGEN_REAL_PORT", "0"))
KEEP = os.environ.get("REGEN_REAL_KEEP") == "1"
REQUIRE_MODEL = os.environ.get("REGEN_REAL_REQUIRE_MODEL", "1") != "0"
QA_ROOT = os.environ.get("REGEN_REAL_QA_ROOT")


def _model_settings() -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "backend_config": os.environ.get("REGEN_REAL_BACKEND_CONFIG"),
            "contract": os.environ.get("REGEN_REAL_CONTRACT"),
            "checkpoint": os.environ.get("REGEN_REAL_CHECKPOINT"),
        }.items()
        if value
    }


def _legacy_writer(document: dict[str, Any], engine: InferenceProcess, model: dict[str, Any]) -> Any:
    from dpo.cli.regen import _gemma_writer

    arguments = Namespace(
        backend_config=model["backend_config"],
        contract=model["contract"],
        checkpoint=model.get("checkpoint"),
    )
    writer = _gemma_writer(arguments, document, engine)
    if writer is None:
        raise RuntimeError("Gemma writer could not be initialized")
    return writer


def _port() -> int:
    if PORT:
        return PORT
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _check_model(model: dict[str, Any]) -> None:
    missing = [key for key in ("backend_config", "contract") if key not in model]
    if missing and REQUIRE_MODEL:
        names = ", ".join(f"REGEN_REAL_{key.upper()}" for key in missing)
        raise RuntimeError(f"Real final QA requires configured Gemma; set {names}")


def main() -> None:
    temporary_root = Path(tempfile.gettempdir()).resolve()
    qa_root = Path(QA_ROOT).resolve() if QA_ROOT else Path(tempfile.mkdtemp(prefix="regen-real-study-"))
    if qa_root == temporary_root or not qa_root.is_relative_to(temporary_root):
        raise ValueError("QA output must be a dedicated subdirectory of the temporary directory")
    qa_root.mkdir(parents=True, exist_ok=True)
    out = qa_root / "out"
    viewing_out = qa_root / "viewing-out"
    document = json.loads(LEGACY_DOCUMENT.read_text(encoding="utf-8"))
    manifest = load_manifest(READY_MANIFEST, MEDIA_ROOT)
    pending = [video["id"] for video in manifest["viewing_videos"] if video["status"] != "ready"]
    if pending:
        raise RuntimeError(f"Ready manifest still has pending viewing videos: {pending}")
    model = _model_settings()
    _check_model(model)
    engine = InferenceProcess(model) if model else None
    writer = _legacy_writer(document, engine, model) if engine is not None else RegenTemplateWriter()
    app = build_app(
        document,
        MEDIA_ROOT,
        out,
        writer,
        derive=False,
        viewing=ViewingConfig(
            READY_MANIFEST,
            MEDIA_ROOT,
            viewing_out,
            model=model,
            inference_timeout=float(os.environ.get("REGEN_REAL_INFERENCE_TIMEOUT", "30")),
            engine=engine,
        ),
    )
    continuation: Continuation = app.state.continuation

    @continuation.app.get("/api/study/export")
    def export(request: Request) -> JSONResponse:
        token = request.cookies.get(continuation.app.state.cookie_name, "")
        return JSONResponse(continuation.store.export(token))

    @app.get("/qa/source")
    def source() -> JSONResponse:
        videos = []
        for video in manifest["viewing_videos"]:
            video_path = (MEDIA_ROOT / video["video"]).resolve()
            audio_path = (MEDIA_ROOT / video["audio"]).resolve()
            videos.append(
                {
                    "id": video["id"],
                    "video": str(video_path),
                    "audio": str(audio_path),
                    "duration_ms": video["duration_ms"],
                    "video_sha256": fingerprint(video_path),
                    "audio_sha256": fingerprint(audio_path),
                    "cue_count": len(video["cues"]),
                }
            )
        return JSONResponse(
            {
                "mode": "real-media-final-survey-qa",
                "qa_root": str(qa_root),
                "legacy_document": str(LEGACY_DOCUMENT.resolve()),
                "media_root": str(MEDIA_ROOT.resolve()),
                "ready_manifest": str(READY_MANIFEST.resolve()),
                "manifest_hash": digest(manifest),
                "model_configured": bool(model),
                "derive": False,
                "cookie_name": continuation.app.state.cookie_name,
                "videos": videos,
            }
        )

    port = _port()
    print(json.dumps({"root": str(qa_root), "url": f"http://127.0.0.1:{port}"}, sort_keys=True), flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        if engine is not None:
            engine.close()
        if not KEEP and not QA_ROOT:
            shutil.rmtree(qa_root, ignore_errors=True)


if __name__ == "__main__":
    main()
