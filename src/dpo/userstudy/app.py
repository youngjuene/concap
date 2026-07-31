"""The human study's local web app: video, a congruency slider, two questions.

Serves one published ``dpo.study-export/v1`` document. Deliberately separate
from ``dpo.annotation``: that instrument collects pairwise preferences from
expert annotators and its responses are ``RawAnnotation`` rows that feed
training; this one measures how a participant places a caption on the
audiovisual congruency axis, and its responses feed nothing but the study's own
analysis. Sharing a schema between them would let one study's data silently
satisfy the other's validator.

The study export is a public-derived artifact, so this process reads captions
without holding a study capability while the protected clip rows stay sealed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from dpo.annotation.webapp import VIDEO_SUFFIXES
from dpo.core.atomic import atomic_write_bytes

RESPONSES_SCHEMA = "dpo.userstudy-responses/v1"
STUDY_EXPORT_SCHEMA = "dpo.study-export/v1"
CLIP_ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")


def _study_document(export: Mapping[str, Any]) -> dict[str, Any]:
    """The participant-facing subset: clips, their ladders, and the axis.

    The export also carries the winning experiment, its validation accuracy,
    the reuse rate, and every rung's measured congruency. None of that reaches
    the browser — a participant who can read which arm produced a caption, or
    which stop the measure ranked highest, is no longer blind to either.
    """
    if export.get("schema") != STUDY_EXPORT_SCHEMA:
        raise ValueError(f"study document schema must be {STUDY_EXPORT_SCHEMA!r}")
    return {
        "clips": [
            {
                "clip_id": str(clip["clip_id"]),
                # Position and text only: the stop's place on the axis is what
                # the control needs, and everything else about how it got there
                # is exactly what a blind participant must not see.
                "levels": [
                    {"position": float(entry["position"]), "text": str(entry["text"])}
                    for entry in clip["levels"]
                ],
            }
            for clip in export["clips"]
        ],
    }


def build_app(export: Mapping[str, Any], media_dir: Path, out_path: Path) -> FastAPI:
    app = FastAPI(title="dpo user study", docs_url=None, redoc_url=None, openapi_url=None)
    page = files("dpo.userstudy").joinpath("page.html").read_text(encoding="utf-8")
    study = _study_document(export)
    out_path.mkdir(parents=True, exist_ok=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(page)

    @app.get("/api/study")
    def study_document() -> dict[str, Any]:
        return study

    @app.get("/media/{clip_id}")
    def media(clip_id: str) -> Any:
        # Always the clip WITH its own soundtrack: the participant is judging a
        # caption against what they hear, so a muted render would make the task
        # unanswerable rather than merely harder.
        if not CLIP_ID_RE.fullmatch(clip_id):
            return JSONResponse({"error": f"invalid clip id {clip_id!r}"}, status_code=400)
        for base in (media_dir / "unmuted_video", media_dir):
            for suffix in VIDEO_SUFFIXES:
                candidate = base / f"{clip_id}{suffix}"
                if candidate.is_file():
                    return FileResponse(candidate)
        return JSONResponse(
            {"error": f"no video with sound for clip {clip_id!r} under {media_dir}"},
            status_code=404,
        )

    @app.post("/api/responses")
    def save_responses(document: dict[str, Any]) -> dict[str, object]:
        if document.get("schema") != RESPONSES_SCHEMA:
            raise HTTPException(status_code=400, detail=f"schema must be {RESPONSES_SCHEMA!r}")
        participant = str(document.get("participant") or "").strip()
        if not participant:
            raise HTTPException(status_code=400, detail="a non-empty participant name is required")
        responses = document.get("responses")
        if not isinstance(responses, list) or not responses:
            raise HTTPException(status_code=400, detail="responses must be a non-empty list")
        known = {clip["clip_id"] for clip in study["clips"]}
        for response in responses:
            clip_id = str(response.get("clip_id", ""))
            if clip_id not in known:
                raise HTTPException(status_code=400, detail=f"unknown clip {clip_id!r}")
        safe = "".join(char for char in participant if char.isalnum() or char in "-_")[:64]
        if not safe:
            raise HTTPException(status_code=400, detail="participant name has no usable characters")
        destination = out_path / f"responses-{safe}.json"
        payload = json.dumps(
            {"schema": RESPONSES_SCHEMA, "participant": participant, "responses": responses},
            ensure_ascii=False,
            indent=2,
        )
        atomic_write_bytes(destination, payload.encode("utf-8") + b"\n")
        return {"saved": str(destination), "responses": len(responses)}

    return app


def run_study_app(
    *,
    export_path: Path,
    media_dir: Path,
    out_path: Path,
    host: str = "127.0.0.1",
    port: int = 8776,
) -> None:
    export = json.loads(Path(export_path).read_text(encoding="utf-8"))
    uvicorn.run(build_app(export, media_dir, out_path), host=host, port=port, log_level="warning")
