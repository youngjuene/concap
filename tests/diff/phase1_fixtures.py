"""A synthetic phase-1 workspace: three clips with frames and masks, an ontology and a corpus."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image

import dpo.regen.sheet_instrument as sheet_instrument
from diff.phase1_setup import approve, draft_setup

CLIPS = ("kyoto_101", "lima_202", "oslo_303")
FRAMES = ("00060", "00180", "00300")
DURATION_MS = 10000
WIDTH, HEIGHT = 64, 36
ONTOLOGY = [
    {"id": "/t", "name": "Sounds of things", "child_ids": ["/t/vehicle", "/t/wind_mic"]},
    {"id": "/t/vehicle", "name": "Vehicle", "child_ids": ["/t/bus", "/t/horn"]},
    {"id": "/t/bus", "name": "Bus", "child_ids": []},
    {"id": "/t/horn", "name": "Vehicle horn, car horn, honking", "child_ids": []},
    {"id": "/h", "name": "Human sounds", "child_ids": ["/h/speech"]},
    {"id": "/h/speech", "name": "Speech", "child_ids": ["/h/male"]},
    {"id": "/h/male", "name": "Male speech, man speaking", "child_ids": []},
    {"id": "/n", "name": "Natural sounds", "child_ids": ["/t/wind_mic"]},
    {"id": "/t/wind_mic", "name": "Wind noise (microphone)", "child_ids": []},
    {"id": "/a", "name": "Animal", "child_ids": ["/a/bird"]},
    {"id": "/a/bird", "name": "Bird", "child_ids": []},
]
CORPUS = {
    "kyoto_101": (
        "c2_video_audio_caption",
        "a bus idles while a man speaks.",
        "Vehicle|Bus|Speech|Male speech, man speaking",
    ),
    "lima_202": (
        "c2_video_audio_caption",
        "traffic with honking horns.",
        "Vehicle|Vehicle horn, car horn, honking",
    ),
    "oslo_303": ("c2_video_audio_caption", "a bird sings in the wind.", "Bird|Wind noise (microphone)|Bird"),
    "practice": ("c1_video_audio", "people talk near a bus.", "Speech|Bus"),
}
# Each clip's sounds as the assistant drafts them from orig_text: (label, family, orig_phrase).
SOUNDS = {
    "kyoto_101": [("버스 소리", "things", "a bus idles"), ("남자의 말소리", "human", "a man speaks")],
    "lima_202": [("차량이 오가는 소리", "things", "traffic"), ("경적 소리", "things", "honking horns")],
    "oslo_303": [("새소리", "animal", "a bird sings"), ("바람 소리", "natural", "the wind")],
    "practice": [
        ("사람들이 이야기하는 소리", "human", "people talk"),
        ("근처의 버스 소리", "things", "a bus"),
    ],
}


def sound_options(key: str) -> list[dict[str, Any]]:
    return [
        {"value": f"s{number}", "label": label, "family": family, "orig_phrase": phrase}
        for number, (label, family, phrase) in enumerate(SOUNDS[key], 1)
    ]


def _png(path: Path, box: tuple[int, int, int, int] | None = None, colour: str = "grey") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L" if box else "RGB", (WIDTH, HEIGHT), 0 if box else colour)
    if box:
        image.paste(255, box)
    image.save(path)


def fake_probe(path: Path) -> tuple[int, float]:
    return DURATION_MS, 60.0


def fake_extract(video: Path, at_ms: int, out: Path) -> None:
    _png(out, colour="white")


def make_workspace(root: Path) -> dict[str, Path]:
    """Media, frames and masks (``road`` = bottom half, ``sky`` = top half), ontology, corpus."""
    media = root / "data" / "media"
    for clip in CLIPS:
        (media / clip).mkdir(parents=True, exist_ok=True)
        (media / clip / "clip.mp4").write_bytes(b"\x00fake-mp4")
        for frame in FRAMES:
            _png(media / clip / "frames" / f"{frame}.png")
            _png(media / clip / "masks" / frame / "road.png", (0, HEIGHT // 2, WIDTH, HEIGHT))
            _png(media / clip / "masks" / frame / "sky.png", (0, 0, WIDTH, HEIGHT // 2))
    videos = root / "data" / "video"
    (videos / "c1_video_audio").mkdir(parents=True, exist_ok=True)
    (videos / "c1_video_audio" / "practice.mp4").write_bytes(b"\x00fake-mp4")
    ontology = root / "ontology.json"
    ontology.write_text(json.dumps(ONTOLOGY), encoding="utf-8")
    corpus = root / "data" / "tidy.csv"
    with corpus.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_index", "file_path", "orig_text", "final_labels"])
        for clip, (condition, text, labels) in CORPUS.items():
            name = "practice.mp4" if clip == "practice" else f"{clip}.mp4"
            for _ in range(2):
                writer.writerow([clip, f"{condition}/{name}", text, labels])
    return {"media": media, "videos": videos, "ontology": ontology, "corpus": corpus}


def drafted(root: Path) -> dict[str, Any]:
    paths = make_workspace(root)
    return draft_setup(
        root=root,
        media_root=paths["media"],
        corpus=paths["corpus"],
        ontology=paths["ontology"],
        video_root=paths["videos"],
        practice_clip="practice",
        practice_still=root / "artifacts" / "practice.png",
        instrument=sheet_instrument,
        probe=fake_probe,
        extract=fake_extract,
    )


def filled(setup: dict[str, Any]) -> dict[str, Any]:
    """The captions and options written from orig_text, as the assistant's drafts would be."""
    for number, clip in enumerate(setup["clips"], 1):
        clip["caption"]["text"] = f"{number}번 영상의 소리가 들린다."
        clip["options"]["A1s"] = sound_options(clip["key"])
        label, _, phrase = SOUNDS[clip["key"]][0]
        clip["options"]["R2"] = [
            {"value": "m1", "label": label, "mentioned": True, "orig_phrase": phrase},
            {"value": "m2", "label": "음악", "mentioned": False},
        ]
    setup["practice"]["options"]["A1s"] = sound_options("practice")
    return setup


def approved_setup(root: Path) -> dict[str, Any]:
    setup = filled(drafted(root))
    return approve(
        setup,
        by="researcher",
        at="2026-10-18T12:00:00+0900",
        clips=[clip["key"] for clip in setup["clips"]],
        practice=True,
        wording=True,
    )
