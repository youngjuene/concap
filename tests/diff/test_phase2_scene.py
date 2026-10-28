"""Phase 2's scene parts and mode: keys, per-window visibility, the carried profile, the scene prompts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from phase1_fixtures import ONTOLOGY
from PIL import Image

from diff.caption_sources import parse_sources
from diff.job_plan import live_job
from diff.phase1_binding import p_bound, tendency
from diff.phase1_scene import SceneError
from diff.phase2_scene import (
    build_scenes,
    cue_steer,
    read_keys,
    scene_problems,
    scene_profile,
    scene_slots,
)
from diff.rewrite_prompts import (
    LIVE_CELL_PROMPTS,
    LIVE_SCENE_CELL_PROMPTS,
    render_live,
)
from diff.study_protocol import CONTEXT_STEPS

VIDEO = "viewing-9"
WIDTH, HEIGHT = 8, 8
TUNER = "튜너"
KEY_FIELDS = ("video", "window", "sound_index", "sound", "audioset_label", "mask_match", "mask_dir")


def _png(path: Path, box: tuple[int, int, int, int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    if box is not None:
        image.paste(255, box)
    image.save(path)


def sources(texts: list[str]) -> list[Any]:
    document = {
        "schema": "diff.caption-sources/v1",
        "sources": [
            {
                "schema": "diff.caption-source/v1",
                "phase": 2,
                "unit": {"kind": "video", "key": VIDEO},
                "cue_index": index,
                "start_ms": index * 1000,
                "end_ms": (index + 1) * 1000,
                "text": text,
                "evidence": {"events": []},
                "provenance": {
                    "text": "synthetic",
                    "evidence": "synthetic",
                    "reviewed_by": [],
                    "reviewed_at": None,
                    "origin": "test",
                },
                "status": "draft",
                "material": "synthetic",
            }
            for index, text in enumerate(texts)
        ],
    }
    return list(parse_sources(document).sources)


@pytest.fixture
def run(tmp_path: Path) -> dict[str, Any]:
    """Two 1 s windows at 60 fps: road on the bottom half and sky on the top in window 0, vegetation
    under a quarter of sky in window 1. A bus mask on the road in window 0; a voice mask in the sky
    in window 1."""
    masks = tmp_path / "masks"
    for frame in range(120):
        name = f"{frame:05d}.png"
        first = frame < 60
        _png(masks / "visual" / VIDEO / "Road" / name, (0, HEIGHT // 2, WIDTH, HEIGHT) if first else None)
        _png(masks / "visual" / VIDEO / "Sky" / name, (0, 0, WIDTH, HEIGHT // (2 if first else 4)))
        _png(
            masks / "visual" / VIDEO / "Vegetation" / name, None if first else (0, HEIGHT // 4, WIDTH, HEIGHT)
        )
        if first:
            _png(masks / "audio" / VIDEO / "a bus idling" / name, (0, HEIGHT - 2, WIDTH, HEIGHT))
        else:
            _png(masks / "audio" / VIDEO / "a man speaking" / name, (0, 0, WIDTH, 2))
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps(ONTOLOGY), encoding="utf-8")
    keys = tmp_path / "keys.csv"
    rows = [
        (VIDEO, 0, 0, "버스 공회전 소리", "Bus", "label", "a bus idling"),
        (VIDEO, 0, 1, "마이크 바람 소리", "Wind noise (microphone)", "none", ""),
        (VIDEO, 1, 0, "남자 목소리", "Male speech, man speaking", "label", "a man speaking"),
        (VIDEO, 1, 1, "새소리", "Bird", "none", ""),
    ]
    with keys.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(KEY_FIELDS)
        writer.writerows(rows)
    cues = sources(["버스 공회전 소리, 마이크 바람 소리", "남자 목소리, 새소리"])
    scenes = build_scenes(cues, keys=keys, run=masks, ontology=ontology, stride=1)
    return {"cues": cues, "scenes": scenes, "keys": keys, "masks": masks, "ontology": ontology}


def by_cue(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [run["scenes"]["cues"][cue.source_id] for cue in run["cues"]]


def test_each_window_reads_only_its_own_frames(run: dict[str, Any]) -> None:
    first, second = by_cue(run)
    assert first["frames"] == [0, 60] and second["frames"] == [60, 120]
    assert first["composition"] == {"road": 0.5, "sky": 0.5, "vegetation": 0.0, "none": 0.0}
    assert second["elements"]["nature"] == 0.75
    assert (first["distinctive"], second["distinctive"]) == ("traffic", "nature")


def test_a_sound_is_on_screen_only_through_a_mask_on_its_categories(run: dict[str, Any]) -> None:
    first, second = by_cue(run)
    bus, wind = first["sounds"]
    assert bus["visibility"]["share"] == 1.0 and bus["visibility"]["on_screen"]
    assert wind["mask_dir"] is None and not wind["visibility"]["on_screen"]
    # A voice mask in the sky is not a person seen; a sound with no mask has none (D2).
    voice, bird = second["sounds"]
    assert voice["visibility"]["share"] == 0.0 and not voice["visibility"]["on_screen"]
    assert bird["mask_match"] == "none" and bird["visibility"]["mean"] == pytest.approx(1 / 3, abs=1e-4)


def test_the_keys_must_be_the_captions_sounds(run: dict[str, Any], tmp_path: Path) -> None:
    other = sources(["버스 소리, 마이크 바람 소리", "남자 목소리, 새소리"])
    with pytest.raises(SceneError, match="not the caption's"):
        build_scenes(other, keys=run["keys"], run=run["masks"], ontology=run["ontology"], stride=1)
    broken = tmp_path / "broken.csv"
    broken.write_text(",".join(KEY_FIELDS) + f"\n{VIDEO},0,0,소리,Bus,none,a bus idling\n", encoding="utf-8")
    with pytest.raises(SceneError, match="mask_dir disagrees"):
        read_keys(broken)


def test_scene_problems(run: dict[str, Any]) -> None:
    assert scene_problems(run["scenes"], run["cues"]) == []
    assert scene_problems(None, run["cues"])[0].startswith("phase 2 has no scene parts")
    edited = sources(["버스 공회전 소리", "남자 목소리, 새소리"])
    assert scene_problems(run["scenes"], edited) == [
        f"{VIDEO} cue 0: no scene part",
    ]


def test_a_scene_file_the_cues_cannot_use_is_a_problem(run: dict[str, Any]) -> None:
    key = next(iter(run["scenes"]["cues"]))
    lost = {**run["scenes"]["cues"][key], "distinctive": None}
    assert scene_problems({**run["scenes"], "cues": {**run["scenes"]["cues"], key: lost}}, run["cues"]) == [
        f"{VIDEO} cue 0: no distinctive element"
    ]
    sounds = run["scenes"]["cues"][key]["sounds"]
    windy = {
        **run["scenes"]["cues"][key],
        "sounds": [{**sounds[0], "audioset_label": "Wind noise (microphone)"}, *sounds[1:]],
    }
    problems = scene_problems({**run["scenes"], "cues": {**run["scenes"]["cues"], key: windy}}, run["cues"])
    assert problems == [f"{VIDEO} cue 0: the first sound is a sound of the recording itself"]


def test_serve_refuses_a_scene_file_that_does_not_fit_its_cues_for_any_purpose(run: dict[str, Any]) -> None:
    from types import SimpleNamespace

    from diff.__main__ import _scene_file_or_exit

    videos = [SimpleNamespace(stream=SimpleNamespace(cues=run["cues"]))]
    _scene_file_or_exit(run["scenes"], videos)  # fits: no exit
    edited = [
        SimpleNamespace(stream=SimpleNamespace(cues=sources(["버스 공회전 소리", "남자 목소리, 새소리"])))
    ]
    with pytest.raises(SystemExit, match=r"cannot steer these videos' cues: viewing-9 cue 0: no scene part"):
        _scene_file_or_exit(run["scenes"], edited)


def test_the_slots_leave_out_the_recording_itself(run: dict[str, Any]) -> None:
    first, second = by_cue(run)
    assert scene_slots(first) == {
        "seen": "도로와 차량",
        "visible_sounds": "버스 공회전 소리",
        "unseen_sounds": "없음",
    }
    assert scene_slots(second) == {
        "seen": "나무와 풀",
        "visible_sounds": "없음",
        "unseen_sounds": "남자 목소리, 새소리",
    }


def test_the_profile_is_the_tendency_from_every_stage_a_clip() -> None:
    records = {
        "b": {"clip_index": 1, "model": {"visibility": 0.8, "attention": {"log_ratio": 1.2}}},
        "a": {"clip_index": 0, "model": {"visibility": 0.3, "attention": {"log_ratio": 0.0}}},
    }
    profile = scene_profile(records)
    assert profile["history"] == [[0.3, 0.0], [0.8, 1.2]] and profile["clips"] == 2
    assert scene_profile({})["tendency"]["mean"] == pytest.approx(0.5)


def test_each_cue_follows_the_phase1_models_recommendation(run: dict[str, Any]) -> None:
    first, second = by_cue(run)
    flat = cue_steer(first, [])
    q = first["sounds"][0]["visibility"]["mean"]
    assert flat["p_bound"] == round(p_bound(tendency([]), q, 0.0), 4)
    assert flat["mode"] == "scene" and flat["slots"]["seen"] == "도로와 차량"
    assert cue_steer(second, [])["mode"] == "acousmatic" and cue_steer(second, [])["slots"] == {}
    # A participant who bound little in phase 1 hears even a visible bus detached.
    detached = [[0.9, -0.25]] * 12
    assert cue_steer(first, detached)["p_bound"] < flat["p_bound"]
    assert cue_steer(None, [])["mode"] == "acousmatic"


def test_the_scene_cells_keep_the_acousmatic_nine_and_need_the_slots(run: dict[str, Any]) -> None:
    cue = run["cues"][0]
    for texture in range(3):
        for context in CONTEXT_STEPS:
            plain = render_live(cue, texture, context, TUNER)
            assert plain.template_sha256 == LIVE_CELL_PROMPTS[(texture, context)].sha256
            scene = LIVE_SCENE_CELL_PROMPTS[(texture, context)]
            assert "보이지 않는 곳에서" in scene.text and "위치는 쓰지 마세요" not in scene.text
    with pytest.raises(ValueError, match="takes the slots"):
        render_live(cue, 1, "relation", TUNER, "scene")
    steer = cue_steer(by_cue(run)[0], [])
    job = live_job(cue, 1, "relation", TUNER, "smoke", steer)
    assert job.cell["cell_id"] == "t1-relation-scene" and job.cell["steer"]["mode"] == "scene"
    assert job.prompt.template_id == "live-p2-scene"
    assert "장면의 배경: 도로와 차량\n화면에 음원이 보이는 소리: 버스 공회전 소리\n" in job.prompt.text
    unsteered = live_job(cue, 1, "relation", TUNER, "smoke")
    assert unsteered.cell["cell_id"] == "t1-relation" and "mode" not in unsteered.cell
