"""The local preference-collection web app over one exported tasks document.

``annotation serve`` runs this FastAPI app for one annotator session: it
serves the self-contained collection page (``collect_page.html``), the tasks
document exported by ``annotation export-tasks``, the contract response
vocabularies, and per-clip media files, and it saves the annotator's full
responses document — schema ``dpo.collection-responses/v1``, ready for
``annotation ingest`` — into the output directory.

The app is deliberately dumb about response content: it checks the document
schema and that every referenced task exists, and leaves the full
``RawAnnotation`` validation to the ingest boundary, so a save can never
destroy an annotator's work over a rule the UI already enforces.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from dpo.annotation.collection_tasks import RESPONSES_SCHEMA, TASKS_SCHEMA
from dpo.annotation.raw_annotations import AnnotationError
from dpo.contracts.study_contract import CHOICES, REASON_TAGS, TIE_SUBTYPES

CLIP_ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov")
# Presentations whose media a flat directory cannot disambiguate: they are all
# videos of the same clip differing only in soundtrack, so ``{clip_id}.mp4``
# could serve any of them and the annotator would never know which they got.
# These must come from a presentation-scoped subdirectory or 404.
SCOPED_ONLY_PRESENTATIONS = ("unmuted_video", "substituted_audio_video")


def resolve_presentation_media(media_dir: Path, clip_id: str, presentation: str) -> Path | None:
    """The file for one clip under one presentation, or None.

    ``media_dir/<presentation>/<clip_id>.<ext>`` wins. A flat
    ``media_dir/<clip_id>.<ext>`` is accepted only for the presentations whose
    content it cannot get wrong — a muted video and an audio-only track are
    exactly what a track-staged corpus already holds.
    """
    suffixes = AUDIO_SUFFIXES if presentation == "audio_only" else VIDEO_SUFFIXES
    bases = [media_dir / presentation]
    if presentation not in SCOPED_ONLY_PRESENTATIONS:
        bases.append(media_dir)
    for base in bases:
        for suffix in suffixes:
            candidate = base / f"{clip_id}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def build_app(tasks_document: Mapping[str, Any], media_dir: Path, out_path: Path) -> FastAPI:
    """The collection app over one tasks document.

    Split from :func:`run_collection_app` so tests can drive the endpoints
    through ``fastapi.testclient`` without a uvicorn server.
    """
    tasks_value = tasks_document.get("tasks")
    if not isinstance(tasks_value, list):
        raise AnnotationError("tasks document requires a tasks array")
    task_ids = {str(task["task_id"]) for task in tasks_value}
    page = files("dpo.annotation").joinpath("collect_page.html").read_text(encoding="utf-8")
    app = FastAPI(title="dpo collection", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(page)

    @app.get("/api/meta")
    def meta() -> dict[str, object]:
        return {
            "choices": list(CHOICES),
            "tie_subtypes": list(TIE_SUBTYPES),
            "reason_tags": list(REASON_TAGS),
        }

    @app.get("/api/tasks")
    def tasks() -> dict[str, Any]:
        return dict(tasks_document)

    @app.get("/media/{clip_id}")
    def media(clip_id: str, presentation: str = "muted_video") -> Response:
        if not CLIP_ID_RE.fullmatch(clip_id):
            return JSONResponse({"error": f"invalid clip id {clip_id!r}"}, status_code=400)
        resolved = resolve_presentation_media(media_dir, clip_id, presentation)
        if resolved is not None:
            return FileResponse(resolved)
        suffixes = AUDIO_SUFFIXES if presentation == "audio_only" else VIDEO_SUFFIXES
        where = f"{media_dir / presentation}"
        if presentation not in SCOPED_ONLY_PRESENTATIONS:
            where += f" or {media_dir}"
        return JSONResponse(
            {
                "error": f"no media for clip {clip_id!r} with presentation {presentation!r};"
                f" expected {clip_id}<{'|'.join(suffixes)}> under {where}"
            },
            status_code=404,
        )

    @app.post("/api/responses")
    def save_responses(document: dict[str, Any]) -> dict[str, object]:
        if document.get("schema") != RESPONSES_SCHEMA:
            raise HTTPException(
                status_code=400, detail=f"responses document schema must be {RESPONSES_SCHEMA!r}"
            )
        annotator = str(document.get("annotator") or "").strip()
        if not annotator:
            raise HTTPException(
                status_code=400, detail="responses document requires a non-empty annotator name"
            )
        responses = document.get("responses")
        if not isinstance(responses, list) or not responses:
            raise HTTPException(
                status_code=400, detail="responses document requires a non-empty responses array"
            )
        for position, response in enumerate(responses):
            if not isinstance(response, dict) or "task_id" not in response:
                raise HTTPException(status_code=400, detail=f"responses[{position}] requires a task_id")
            task_id = str(response["task_id"])
            if task_id not in task_ids:
                raise HTTPException(status_code=400, detail=f"response references unknown task {task_id!r}")
        slug = re.sub(r"[^a-z0-9]+", "-", annotator.casefold()).strip("-") or "annotator"
        out_path.mkdir(parents=True, exist_ok=True)
        destination = out_path / f"responses-{slug}.json"
        destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"saved": str(destination), "responses": len(responses)}

    return app


def run_collection_app(*, tasks_path: Path, media_dir: Path, out_path: Path, host: str, port: int) -> None:
    """Serve the collection UI for one exported tasks document until interrupted."""
    document = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != TASKS_SCHEMA:
        raise AnnotationError(f"tasks document schema must be {TASKS_SCHEMA!r}")
    app = build_app(document, media_dir, out_path)
    uvicorn.run(app, host=host, port=port, log_level="info")
