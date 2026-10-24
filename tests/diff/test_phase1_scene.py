"""Scene parts from a Sa2VA run: groups, composition, visibility, the caption-type inputs and click masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from phase1_fixtures import drafted, filled, with_scenes

import diff.workbook_instrument as workbook_instrument
from diff.phase1_scene import (
    SceneError,
    add_scenes,
    dilated,
    on_labels,
    ontology_groups,
    openness,
    scene_problems,
)
from diff.phase1_setup import setup_problems


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Any]:
    return with_scenes(filled(drafted(tmp_path)), tmp_path)


def by_key(setup: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {clip["key"]: clip["scene"] for clip in setup["clips"]}


def test_groups_are_the_ontology_level_under_its_top_categories(tmp_path: Path) -> None:
    drafted(tmp_path)
    groups = ontology_groups(tmp_path / "ontology.json")
    assert groups("Bus") == ("Vehicle",) and groups("Male speech, man speaking") == ("Human voice",)
    assert groups("Wind noise (microphone)") == ("Miscellaneous sources", "Wind")
    with pytest.raises(SceneError, match="not in the ontology"):
        groups("Thunder")


def test_overlapping_masks_count_each_pixel_once(setup: dict[str, Any]) -> None:
    kyoto, oslo = by_key(setup)["kyoto_101"], by_key(setup)["oslo_303"]
    assert kyoto["composition"] == {
        "person": 0.125,
        "road": 0.375,
        "sky": 0.375,
        "vegetation": 0.125,
        "none": 0.0,
    }
    assert sum(oslo["composition"].values()) == pytest.approx(1)
    assert kyoto["elements"]["traffic"] == 0.375 and oslo["elements"]["nature"] == 0.25


def test_the_caption_type_inputs(setup: dict[str, Any]) -> None:
    scenes = by_key(setup)
    # Oslo's vegetation is 1.5 times the setup's mean; kyoto has no such lift, so its largest
    # element stands (sky and road tie, and the elements' order breaks the tie).
    assert (scenes["oslo_303"]["distinctive"], scenes["oslo_303"]["distinctive_by"]) == ("nature", "lift")
    assert (scenes["kyoto_101"]["distinctive"], scenes["kyoto_101"]["distinctive_by"]) == ("open", "largest")
    assert {scene["openness"] for scene in scenes.values()} == {"open"}
    assert [openness(sky) for sky in (0.229, 0.128, 0.084, 0.063, 0.041)] == [
        "open",
        "neutral",
        "neutral",
        "enclosed",
        "enclosed",
    ]


def test_a_source_is_seen_only_inside_the_categories_its_group_can_be_seen_in(setup: dict[str, Any]) -> None:
    scenes = by_key(setup)
    bus, man = scenes["kyoto_101"]["sounds"]["s1"], scenes["kyoto_101"]["sounds"]["s2"]
    assert bus["plausible"] == ["road", "road_transport"] and bus["visibility"]["share"] == 1.0
    assert bus["visibility"]["mean"] == pytest.approx(11 / 12, abs=1e-4) and bus["visibility"]["on_screen"]
    # The man is on his Person mask in half the frames and has no mask in the others: on screen, just.
    assert man["visibility"]["share"] == 0.5 and man["visibility"]["mean"] == 0.5
    assert man["visibility"]["on_screen"] and man["click_frames"]["0"]["shown"]
    horn = scenes["lima_202"]["sounds"]["s2"]  # a mask in the sky is not a horn seen
    assert horn["visibility"]["share"] == 0.0 and not horn["visibility"]["on_screen"]
    assert horn["click_frames"]["0"]["shown"] is False
    wind = scenes["oslo_303"]["sounds"]["s2"]  # microphone wind has no visible source at all
    assert wind["plausible"] == [] and wind["visibility"]["mean"] == pytest.approx(1 / 12, abs=1e-4)


def test_only_pixels_on_the_source_s_own_categories_count() -> None:
    names = ["person", "road", "sky"]
    winner = np.full((4, 4), len(names), dtype=np.int8)  # nothing labelled
    winner[:, :2] = 2  # sky on the left half
    winner[0, 2:] = 0  # a person top right
    talkers = np.zeros((4, 4), dtype=bool)
    talkers[:, 2:] = True  # talkers: one row on the person, three Sa2VA's Person mask missed
    assert on_labels(talkers, winner, names, ["person"]) == 0.25
    assert on_labels(~talkers, winner, names, ["person"]) == 0.0  # a voice in the sky
    assert on_labels(talkers, winner, names, []) == 0.0  # microphone wind is seen nowhere
    assert on_labels(np.zeros((4, 4), dtype=bool), winner, names, ["person"]) == 0.0


def test_a_widened_mask_is_the_disc_around_each_pixel() -> None:
    point = np.zeros((7, 7), dtype=bool)
    point[3, 3] = True
    wide = dilated(point, 2)
    assert wide.sum() == 13 and wide[1, 3] and wide[3, 5] and not wide[1, 1]
    assert (dilated(point, 0) == point).all()


def test_click_masks_sit_beside_the_media_and_their_absence_is_a_problem(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    clip = setup["clips"][0]
    frame = clip["scene"]["sounds"]["s1"]["click_frames"]["0"]
    assert frame["mask"] == "data/media/kyoto_101/audio-masks/00060/s1.png" and frame["shown"] is True
    # The bus's 48 × 18 box, widened by a one-pixel disc inside the 64 × 36 frame: one more
    # column and row, less the corner a disc does not reach.
    assert frame["area"] == round((49 * 19 - 1) / (64 * 36), 5)
    assert scene_problems(clip, tmp_path) == []
    (tmp_path / frame["mask"]).unlink()
    assert any("missing data/media/kyoto_101" in problem for problem in scene_problems(clip, tmp_path))
    assert scene_problems({**clip, "scene": None}, tmp_path) == [
        "kyoto_101.scene: missing; run `python -m diff phase1 scene`"
    ]


def test_a_setup_needs_each_option_s_label_and_a_mapped_group(tmp_path: Path) -> None:
    setup = filled(drafted(tmp_path))
    del setup["clips"][0]["options"]["A1s"][0]["audioset_label"]
    problems = setup_problems(setup, "smoke", root=tmp_path, instrument=workbook_instrument)
    assert any("audioset_label must be one of" in problem for problem in problems)
    assert any(problem.startswith("kyoto_101.scene: missing") for problem in problems)
    with pytest.raises(SceneError, match="is not one of the clip's labels"):
        add_scenes(setup, run=tmp_path / "sa2va", ontology=tmp_path / "ontology.json", root=tmp_path)
    setup = filled(drafted(tmp_path))
    setup["clips"][0]["options"]["A1s"][0]["audioset_label"] = "Vehicle"  # kyoto's labels include it
    (tmp_path / "sa2va" / "audio" / "kyoto_101" / "Bus").rename(
        tmp_path / "sa2va" / "audio" / "kyoto_101" / "Vehicle"
    )
    assert add_scenes(setup, run=tmp_path / "sa2va", ontology=tmp_path / "ontology.json", root=tmp_path)


def test_a_judgement_left_from_before_is_refused_and_dropped_on_rebuild(tmp_path: Path) -> None:
    setup = with_scenes(filled(drafted(tmp_path)), tmp_path)
    assert scene_problems(setup["clips"][1], tmp_path) == []
    setup["clips"][1]["options"]["A1s"][1]["source_visible"] = "partly"  # a researcher's, from v2
    assert any(
        "measured from the masks" in problem for problem in scene_problems(setup["clips"][1], tmp_path)
    )
    rebuilt = add_scenes(setup, run=tmp_path / "sa2va", ontology=tmp_path / "ontology.json", root=tmp_path)
    assert "source_visible" not in rebuilt["clips"][1]["options"]["A1s"][1]
    assert scene_problems(rebuilt["clips"][1], tmp_path) == []
