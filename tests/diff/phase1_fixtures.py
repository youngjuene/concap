"""A synthetic phase-1 workspace: three clips, frames, masks, captions, ontology, corpus and Sa2VA run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image

import diff.workbook_instrument as workbook_instrument
from diff.phase1_scene import add_scenes
from diff.phase1_setup import approve, draft_setup

CLIPS = ("kyoto_101", "lima_202", "oslo_303")
FRAMES = ("00060", "00180", "00300")
DURATION_MS = 10000
WIDTH, HEIGHT = 64, 36
ONTOLOGY = [
    {"id": "/t", "name": "Sounds of things", "child_ids": ["/t/vehicle", "/t/misc"]},
    {"id": "/t/vehicle", "name": "Vehicle", "child_ids": ["/t/bus", "/t/horn"]},
    {"id": "/t/bus", "name": "Bus", "child_ids": []},
    {"id": "/t/horn", "name": "Vehicle horn, car horn, honking", "child_ids": []},
    {"id": "/t/misc", "name": "Miscellaneous sources", "child_ids": ["/t/wind_mic"]},
    {"id": "/h", "name": "Human sounds", "child_ids": ["/h/voice"]},
    {"id": "/h/voice", "name": "Human voice", "child_ids": ["/h/speech"]},
    {"id": "/h/speech", "name": "Speech", "child_ids": ["/h/male"]},
    {"id": "/h/male", "name": "Male speech, man speaking", "child_ids": []},
    {"id": "/n", "name": "Natural sounds", "child_ids": ["/n/wind"]},
    {"id": "/n/wind", "name": "Wind", "child_ids": ["/t/wind_mic"]},
    {"id": "/t/wind_mic", "name": "Wind noise (microphone)", "child_ids": []},
    {"id": "/a", "name": "Animal", "child_ids": ["/a/wild"]},
    {"id": "/a/wild", "name": "Wild animals", "child_ids": ["/a/bird"]},
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
# The prepared Korean captions, shown raw: one runs to three lines, as singapore_067's does.
PREPARED = {
    "kyoto_101": "버스가 공회전하는 소리",
    "lima_202": "도시의 거리는 차들로 가득하다.\n경적 소리가 끊임없이 들린다.\n멀리서 사람들이 말한다.",
    "oslo_303": "바람 속에서 새가 지저귄다.",
}
# Each clip's sounds as the assistant drafts them from orig_text:
# (label, family, orig_phrase, AudioSet label).
SOUNDS = {
    "kyoto_101": [
        ("버스 소리", "things", "a bus idles", "Bus"),
        ("남자의 말소리", "human", "a man speaks", "Male speech, man speaking"),
    ],
    "lima_202": [
        ("차량이 오가는 소리", "things", "traffic", "Vehicle"),
        ("경적 소리", "things", "honking horns", "Vehicle horn, car horn, honking"),
    ],
    "oslo_303": [
        ("새소리", "animal", "a bird sings", "Bird"),
        ("바람 소리", "natural", "the wind", "Wind noise (microphone)"),
    ],
    "practice": [
        ("사람들이 이야기하는 소리", "human", "people talk", None),
        ("근처의 버스 소리", "things", "a bus", None),
    ],
}
# The Sa2VA run's frames: the three stills and three more between them.
SA2VA_FRAMES = ("00000", "00060", "00120", "00180", "00240", "00300")
# Sa2VA's visual categories (road = bottom half, sky = top half, a person bottom left,
# vegetation top right, larger at oslo) and where each sound's audio mask lies: the bus,
# the traffic and the bird on their categories; the man on the person in half the frames;
# the horn in the sky and the microphone wind on the road, where neither is seen.
BOTTOM_RIGHT = (16, HEIGHT // 2, WIDTH, HEIGHT)
PERSON = (0, HEIGHT // 2, 16, HEIGHT)
AUDIO_BOXES: dict[str, dict[str, tuple[tuple[int, int, int, int] | None, ...]]] = {
    "kyoto_101": {
        "Bus": (BOTTOM_RIGHT,) * 6,
        "Male speech, man speaking": (PERSON, PERSON, PERSON, None, None, None),
    },
    "lima_202": {
        "Vehicle": (BOTTOM_RIGHT,) * 6,
        "Vehicle horn, car horn, honking": ((0, 0, 32, HEIGHT // 2),) * 6,
    },
    "oslo_303": {"Bird": ((48, 0, WIDTH, HEIGHT // 2),) * 6, "Wind noise (microphone)": (BOTTOM_RIGHT,) * 6},
}


def sound_options(key: str) -> list[dict[str, Any]]:
    options = []
    for number, (label, family, phrase, audioset) in enumerate(SOUNDS[key], 1):
        option = {"value": f"s{number}", "label": label, "family": family, "orig_phrase": phrase}
        if audioset:
            option["audioset_label"] = audioset
        options.append(option)
    return options


def _png(path: Path, box: tuple[int, int, int, int] | None = None, colour: str = "grey") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L" if box else "RGB", (WIDTH, HEIGHT), 0 if box else colour)
    if box:
        image.paste(255, box)
    image.save(path)


def _mask_png(path: Path, box: tuple[int, int, int, int] | None) -> None:
    """A binary mask, empty when there is no box (Sa2VA writes an all-zero PNG for no result)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    if box:
        image.paste(255, box)
    image.save(path)


def fake_probe(path: Path) -> tuple[int, float]:
    return DURATION_MS, 60.0


def fake_extract(video: Path, at_ms: int, out: Path) -> None:
    _png(out, colour="white")


def make_sa2va(run: Path) -> None:
    """A Sa2VA run's masks for the three clips: ``visual/<clip>/<Category>``, ``audio/<clip>/<label>``."""
    for clip in CLIPS:
        vegetation = (32 if clip == "oslo_303" else 48, 0, WIDTH, HEIGHT // 2)
        visual = {
            "Road": (0, HEIGHT // 2, WIDTH, HEIGHT),
            "Sky": (0, 0, WIDTH, HEIGHT // 2),
            "Person": PERSON,
            "Vegetation": vegetation,
        }
        for frame in SA2VA_FRAMES:
            for category, box in visual.items():
                _png(run / "visual" / clip / category / f"{frame}.png", box)
        for label, boxes in AUDIO_BOXES[clip].items():
            for frame, audio_box in zip(SA2VA_FRAMES, boxes, strict=True):
                _mask_png(run / "audio" / clip / label / f"{frame}.png", audio_box)


def make_workspace(root: Path) -> dict[str, Path]:
    """Media, frames and masks (``road`` = bottom half, ``sky`` = top half), captions, ontology, corpus."""
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
    captions = root / "scripts" / "captions.json"
    captions.parent.mkdir(parents=True, exist_ok=True)
    document = {"provenance": "transcribed-from-video", "captions": PREPARED}
    captions.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    corpus = root / "data" / "tidy.csv"
    with corpus.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_index", "file_path", "orig_text", "final_labels"])
        for clip, (condition, text, labels) in CORPUS.items():
            name = "practice.mp4" if clip == "practice" else f"{clip}.mp4"
            for _ in range(2):
                writer.writerow([clip, f"{condition}/{name}", text, labels])
    make_sa2va(root / "sa2va")
    return {
        "media": media,
        "videos": videos,
        "ontology": ontology,
        "corpus": corpus,
        "captions": captions,
        "sa2va": root / "sa2va",
    }


def drafted(root: Path) -> dict[str, Any]:
    paths = make_workspace(root)
    return draft_setup(
        root=root,
        media_root=paths["media"],
        corpus=paths["corpus"],
        captions=paths["captions"],
        ontology=paths["ontology"],
        video_root=paths["videos"],
        practice_clip="practice",
        practice_still=root / "artifacts" / "practice.png",
        instrument=workbook_instrument,
        probe=fake_probe,
        extract=fake_extract,
    )


def filled(setup: dict[str, Any]) -> dict[str, Any]:
    """The options written from orig_text, as the assistant's drafts would be."""
    for clip in setup["clips"]:
        clip["options"]["A1s"] = sound_options(clip["key"])
        label, _, phrase, _ = SOUNDS[clip["key"]][0]
        clip["options"]["R2"] = [
            {"value": "m1", "label": label, "mentioned": True, "orig_phrase": phrase},
            {"value": "m2", "label": "음악", "mentioned": False},
        ]
    setup["practice"]["options"]["A1s"] = sound_options("practice")
    return setup


def with_scenes(setup: dict[str, Any], root: Path) -> dict[str, Any]:
    """The filled setup with each clip's scene part from the synthetic Sa2VA run (every frame read)."""
    return add_scenes(setup, run=root / "sa2va", ontology=root / "ontology.json", root=root, stride=1)


def approved_setup(root: Path) -> dict[str, Any]:
    setup = with_scenes(filled(drafted(root)), root)
    return approve(
        setup,
        by="researcher",
        at="2026-10-18T12:00:00+0900",
        clips=[clip["key"] for clip in setup["clips"]],
        practice=True,
    )
