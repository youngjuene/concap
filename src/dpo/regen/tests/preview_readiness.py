"""Synthetic integrated readiness server; never use for participant studies."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from PIL import Image

from dpo.caption.writer import CaptionRequest, Written
from dpo.regen.app import build_app, warm_derivatives
from dpo.regen.config import DEFAULT_SCALE_ANCHORS_BY_LANGUAGE, Calibration, Configuration, Scale
from dpo.regen.continuation import Continuation, ViewingConfig
from dpo.regen.document import REGEN_SCHEMA
from dpo.regen.tests.test_study import make_media

ROOT = Path(tempfile.mkdtemp(prefix="regen-readiness-"))
DURATION_MS = 10000
FRAMES = (1000, 3000, 5000, 7000, 9000)


class SyntheticCaptionWriter:
    identity = "synthetic-readiness"

    def write_attributed(self, request: CaptionRequest) -> Written:
        caption = self.write(request)
        return Written(caption, self.identity)

    def write(self, request: CaptionRequest) -> str:
        slot = int(request.shot_id.removeprefix("cue")) + 1 if request.shot_id.startswith("cue") else 1
        if getattr(request, "language", "en") == "ko":
            return f"합성 QA 자막 {slot}: 버스와 발걸음 소리."
        return f"Synthetic QA caption {slot}: bus and footsteps shape the scene."


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _video(path: Path, colour: str, seconds: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colour}:s=320x180:r=10",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=8000",
            "-t",
            str(seconds),
            "-c:v",
            "libvpx",
            "-b:v",
            "70k",
            "-c:a",
            "libvorbis",
            "-shortest",
            "-y",
            str(path),
        ]
    )


def _wav(path: Path, seconds: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 8000 * seconds)


def _mask(path: Path, *, left: int, top: int, right: int, bottom: int) -> None:
    image = Image.new("L", (320, 180), 0)
    for x in range(left, right):
        for y in range(top, bottom):
            image.putpixel((x, y), 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _still(path: Path, colour: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), colour).save(path)


def _track(name: str, prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "start_ms": index * 2500,
            "end_ms": (index + 1) * 2500,
            "text": {
                "en": f"{prefix} {name} cue {index + 1}.",
                "ko": f"{prefix} {name} 큐 {index + 1} 소리",
            },
        }
        for index in range(4)
    ]


def _waveform() -> list[float]:
    return [round(0.45 + math.sin(index / 5) * 0.25, 4) for index in range(64)]


def _segment(root: Path, name: str, colour: str, stems: list[tuple[str, str, str]]) -> dict[str, Any]:
    _video(root / name / "clip.webm", colour)
    _wav(root / name / "audio.wav")
    for stem, _, _ in stems:
        _wav(root / name / "stems" / f"{stem}.wav")
    for index, _at_ms in enumerate(FRAMES):
        _still(root / name / "frames" / f"{index}.png", colour.replace("0x", "#"))
        _mask(root / name / "masks" / str(index) / "building.png", left=0, top=0, right=160, bottom=180)
        _mask(root / name / "masks" / str(index) / "person.png", left=32, top=24, right=96, bottom=120)
    return {
        "segment": name,
        "clip_id": f"synthetic_{name.lower()}",
        "video": f"{name}/clip.webm",
        "audio": f"{name}/audio.wav",
        "duration_ms": DURATION_MS,
        "frames": [
            {
                "at_ms": at_ms,
                "still": f"{name}/frames/{index}.png",
                "objects": [
                    {"id": "building", "label": "Building", "mask": f"{name}/masks/{index}/building.png"},
                    {"id": "person", "label": "Person", "mask": f"{name}/masks/{index}/person.png"},
                ],
            }
            for index, at_ms in enumerate(FRAMES)
        ],
        "stems": [
            {
                "id": stem,
                "label": label,
                "parent": parent,
                "audio": f"{name}/stems/{stem}.wav",
                "colour": "#3F83D1",
                "gain": 1.0,
                "waveform": _waveform(),
            }
            for stem, label, parent in stems
        ],
        "prepared_track": _track(name, "Prepared synthetic QA"),
        "fallback_track": _track(name, "Fallback synthetic QA"),
    }


def _legacy_document(media: Path) -> dict[str, Any]:
    configuration = Configuration(
        study_id="readiness",
        corpus_id="synthetic",
        calibration=Calibration(
            cue_slots=4,
            minimum_points=2,
            languages=("en", "ko"),
            scale=Scale(anchors_by_language=DEFAULT_SCALE_ANCHORS_BY_LANGUAGE),
        ),
    )
    return {
        "schema": REGEN_SCHEMA,
        "session_id": "regen-readiness",
        "config": configuration.artifact(),
        "segments": {
            "A": _segment(
                media,
                "A",
                "0x425f78",
                [("traffic", "Traffic noise", "Sounds of things"), ("bird", "Birds", "Animal")],
            ),
            "B": _segment(
                media,
                "B",
                "0x506b42",
                [("siren", "Siren", "Sounds of things"), ("footsteps", "Footsteps", "Human sounds")],
            ),
        },
    }


def _viewing_manifest(root: Path) -> dict[str, Any]:
    manifest = make_media(root)
    manifest["language"] = "en"
    manifest["languages"] = ["en", "ko"]
    for video in manifest["viewing_videos"]:
        for cue_index, cue in enumerate(video["cues"], start=1):
            cue["fallback"]["en"] = f"Synthetic QA fallback caption {cue_index}."
            cue["fallback"]["ko"] = f"합성 QA 대체 자막 {cue_index} 소리"
    return manifest


def main() -> None:
    media = ROOT / "legacy-media"
    viewing_media = ROOT / "viewing-media"
    out = ROOT / "out"
    viewing_out = ROOT / "viewing-out"
    document = _legacy_document(media)
    manifest = _viewing_manifest(viewing_media)
    manifest_path = ROOT / "viewing-study.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    warm_derivatives(document, media)
    app = build_app(
        document,
        media,
        out,
        SyntheticCaptionWriter(),
        derive=True,
        viewing=ViewingConfig(manifest_path, viewing_media, viewing_out, inference_timeout=5),
    )
    continuation: Continuation = app.state.continuation

    @continuation.app.get("/api/study/export")
    def export(request: Request) -> JSONResponse:
        token = request.cookies.get(continuation.app.state.cookie_name, "")
        return JSONResponse(continuation.store.export(token))

    print(json.dumps({"root": str(ROOT), "url": "http://127.0.0.1:18781"}, sort_keys=True), flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=18781, log_level="warning")
    finally:
        if not bool(int(__import__("os").environ.get("REGEN_READINESS_KEEP", "0"))):
            shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
