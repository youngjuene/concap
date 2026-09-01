"""Sa2VA masks become auditable caption-parameter evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from dpo.cli.session import _session_link_masks
from dpo.session.mask_parameters import (
    AUDIO_VISUAL_CATEGORIES,
    MASK_LINK_SCHEMA,
    derive_mask_links,
)


def _mask(path: Path, pixels: set[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (4, 4), 0)
    for x, y in pixels:
        image.putpixel((x, y), 255)
    image.save(path)


def _tidy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_index", "final_labels", "top_level_parent_name"],
        )
        writer.writeheader()
        # Duplicate rows are expected in the real file (one per participant),
        # and must not multiply tag counts.
        for _ in range(2):
            writer.writerow(
                {
                    "file_index": "clip_001",
                    "final_labels": "Siren|Siren|Speech",
                    "top_level_parent_name": "Sounds of things:2|Human sounds:1",
                }
            )


def _ontology(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"id": "things", "name": "Sounds of things", "child_ids": ["siren"]},
                {"id": "siren", "name": "Siren", "child_ids": []},
                {"id": "human", "name": "Human sounds", "child_ids": ["speech"]},
                {"id": "speech", "name": "Speech", "child_ids": []},
            ]
        ),
        encoding="utf-8",
    )


def test_audio_links_tags_to_eye_ear_weights_and_mask_coordinates(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    _mask(root / "audio" / "clip_001" / "Siren" / "00000.png", {(2, 2), (3, 2)})
    _mask(root / "audio" / "clip_001" / "Siren" / "00001.png", {(3, 3)})
    _mask(root / "audio" / "clip_001" / "Speech" / "00000.png", {(0, 0)})
    _mask(root / "audio" / "clip_001" / "Speech" / "00001.png", set())
    tidy = tmp_path / "tidy.csv"
    ontology = tmp_path / "ontology.json"
    _tidy(tidy)
    _ontology(ontology)

    linked = derive_mask_links(root, tidy, ontology, fps=1.0, clips=["clip_001"])

    assert linked["schema"] == MASK_LINK_SCHEMA
    audio = {entry["label"]: entry for entry in linked["clips"]["clip_001"]["audio"]}
    assert set(audio) == {"Siren", "Speech"}
    assert audio["Siren"]["top_level_parent_name"] == "Sounds of things"
    assert audio["Siren"]["tag_count"] == 2
    assert audio["Siren"]["parameters"]["balance"]["weights"][1] == 1.0
    assert audio["Speech"]["parameters"]["balance"]["weights"][1] == 0.5
    assert audio["Siren"]["mask"]["frames_nonempty"] == 2
    assert audio["Siren"]["mask"]["union_bbox"] == [0.5, 0.5, 1.0, 1.0]
    assert audio["Siren"]["mask"]["mean_bbox_center"] == [0.8125, 0.75]
    assert audio["Siren"]["mask"]["last_timestamp_ms"] == 1000
    assert audio["Siren"]["parameters"]["admission"]["candidate"] is True
    assert audio["Siren"]["parameters"]["balance"]["eye"] > audio["Speech"]["parameters"]["balance"]["eye"]
    assert audio["Siren"]["session_source"]["weights"] == audio["Siren"]["parameters"]["balance"]["weights"]
    assert "weights[1]" in audio["Siren"]["session_source"]["review_required"]


def test_ungrounded_audio_tag_remains_an_admission_candidate(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    (root / "audio" / "clip_001").mkdir(parents=True)
    tidy = tmp_path / "tidy.csv"
    ontology = tmp_path / "ontology.json"
    _tidy(tidy)
    _ontology(ontology)

    linked = derive_mask_links(root, tidy, ontology, fps=1.0, clips=["clip_001"])
    audio = linked["clips"]["clip_001"]["audio"]

    assert len(audio) == 2
    assert all(entry["parameters"]["admission"]["candidate"] for entry in audio)
    assert all(entry["mask"]["frames_nonempty"] == 0 for entry in audio)
    assert all(entry["parameters"]["balance"]["eye"] == 0.0 for entry in audio)


def test_visual_branch_uses_fixed_categories_and_near_far_geometry(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    # Road is low and large (near); Sky is high and large (far).
    _mask(root / "visual" / "clip_001" / "Road" / "00000.png", {(0, 2), (1, 2), (0, 3), (1, 3)})
    _mask(root / "visual" / "clip_001" / "Sky" / "00000.png", {(0, 0), (1, 0), (0, 1), (1, 1)})
    tidy = tmp_path / "tidy.csv"
    ontology = tmp_path / "ontology.json"
    _tidy(tidy)
    _ontology(ontology)

    linked = derive_mask_links(root, tidy, ontology, fps=1.0, clips=["clip_001"])
    visual = {entry["label"]: entry for entry in linked["clips"]["clip_001"]["visual"]}

    assert tuple(visual) == AUDIO_VISUAL_CATEGORIES
    assert visual["Road"]["parameters"]["balance"]["near"] > visual["Sky"]["parameters"]["balance"]["near"]
    assert visual["Sky"]["parameters"]["balance"]["far"] > visual["Road"]["parameters"]["balance"]["far"]
    assert visual["Road"]["parameters"]["detail"]["role"] == "fixed"
    assert visual["Sky"]["parameters"]["detail"]["role"] == "backdrop"
    assert visual["Person"]["parameters"]["admission"]["candidate"] is False


def test_one_frame_per_second_release_layout_is_supported(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    _mask(root / "audio" / "clip_001" / "Siren_0.png", {(1, 1)})
    _mask(root / "audio" / "clip_001" / "Siren_1.png", {(2, 2)})
    tidy = tmp_path / "tidy.csv"
    ontology = tmp_path / "ontology.json"
    _tidy(tidy)
    _ontology(ontology)

    linked = derive_mask_links(root, tidy, ontology, fps=60.0, clips=["clip_001"])
    siren = next(entry for entry in linked["clips"]["clip_001"]["audio"] if entry["label"] == "Siren")

    assert siren["mask"]["layout"] == "perframe-seconds"
    assert siren["mask"]["frames_observed"] == 2


def test_csv_family_counts_resolve_an_ontology_dag_tie(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    (root / "audio" / "clip_001").mkdir(parents=True)
    tidy = tmp_path / "tidy.csv"
    with tidy.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_index", "final_labels", "top_level_parent_name"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_index": "clip_001",
                "final_labels": "Bell|Vehicle",
                "top_level_parent_name": "Music:1|Sounds of things:1",
            }
        )
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        json.dumps(
            [
                {"id": "things", "name": "Sounds of things", "child_ids": ["bell", "vehicle"]},
                {"id": "music", "name": "Music", "child_ids": ["bell"]},
                {"id": "bell", "name": "Bell", "child_ids": []},
                {"id": "vehicle", "name": "Vehicle", "child_ids": []},
            ]
        ),
        encoding="utf-8",
    )

    linked = derive_mask_links(root, tidy, ontology, clips=["clip_001"])
    audio = {entry["label"]: entry for entry in linked["clips"]["clip_001"]["audio"]}

    assert audio["Bell"]["top_level_parent_name"] == "Music"
    assert audio["Vehicle"]["top_level_parent_name"] == "Sounds of things"


def test_link_masks_cli_writes_a_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "masks"
    (root / "audio" / "clip_001").mkdir(parents=True)
    tidy = tmp_path / "tidy.csv"
    ontology = tmp_path / "ontology.json"
    out = tmp_path / "links.json"
    _tidy(tidy)
    _ontology(ontology)

    result = _session_link_masks(
        argparse.Namespace(
            mask_root=str(root),
            tidy_data=str(tidy),
            ontology=str(ontology),
            out=str(out),
            fps=60.0,
            clips=["clip_001"],
        )
    )

    assert result == 0
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == MASK_LINK_SCHEMA
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == "linked"


# ---- the role a mask cannot measure ----------------------------------------


def test_audio_role_is_unset_with_its_evidence_reported(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    # A siren grounded in every frame: the case that used to come back
    # "underneath" because something siren-shaped was visible throughout.
    _mask(root / "audio" / "clip_001" / "Siren" / "00000.png", {(2, 2), (3, 2)})
    _mask(root / "audio" / "clip_001" / "Siren" / "00001.png", {(2, 2), (3, 3)})
    _mask(root / "audio" / "clip_001" / "Speech" / "00000.png", {(0, 0)})
    _mask(root / "audio" / "clip_001" / "Speech" / "00001.png", set())
    tidy, ontology = tmp_path / "tidy.csv", tmp_path / "ontology.json"
    _tidy(tidy)
    _ontology(ontology)

    linked = derive_mask_links(root, tidy, ontology, fps=1.0, clips=["clip_001"])

    audio = {entry["label"]: entry for entry in linked["clips"]["clip_001"]["audio"]}
    for entry in audio.values():
        assert entry["session_source"]["role"] is None
        assert "role" in entry["session_source"]["review_required"]
        detail = entry["parameters"]["detail"]
        assert detail["role"] is None
        # The evidence a researcher annotates from is still on the entry.
        assert detail["grounding_presence"] == entry["mask"]["presence_ratio"]
        assert detail["tag_count"] == entry["tag_count"]
    assert audio["Siren"]["parameters"]["detail"]["grounding_presence"] == 1.0


def test_visual_role_stays_set_because_the_category_names_it(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    _mask(root / "visual" / "clip_001" / "Road" / "00000.png", {(1, 3)})
    tidy, ontology = tmp_path / "tidy.csv", tmp_path / "ontology.json"
    _tidy(tidy)
    _ontology(ontology)

    linked = derive_mask_links(root, tidy, ontology, fps=1.0, clips=["clip_001"])

    visual = {entry["label"]: entry for entry in linked["clips"]["clip_001"]["visual"]}
    assert visual["Road"]["session_source"]["role"] == "fixed"
    assert "role" not in visual["Road"]["session_source"]["review_required"]
