"""Reading the mask tree: shots cut on composition, labels grouped on agreement."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from dpo.caption.ontology import OntologyError, TagRecord, load_tags
from dpo.console.config import Calibration
from dpo.console.masks import (
    MANIFEST_SCHEMA,
    VISUAL_CATEGORIES,
    ClipMasks,
    MaskReadError,
    clip_segments,
    derive_clip,
    derive_manifest,
    group_audio_labels,
    group_quantities,
)

SIZE = 8


def _mask(path: Path, box: tuple[int, int, int, int] | None) -> None:
    """One frame. ``box`` is (left, top, right, bottom) in an 8x8 grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (SIZE, SIZE), 0)
    if box is not None:
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                image.putpixel((x, y), 255)
    image.save(path)


def _visual(root: Path, clip: str, frames: int, *, split_at: int | None = None) -> None:
    """Frames whose composition flips from all-road to all-sky at ``split_at``."""
    for index in range(frames):
        second_half = split_at is not None and index >= split_at
        _mask(root / "visual" / clip / "Road" / f"{index:05d}.png", None if second_half else (0, 0, 8, 8))
        _mask(root / "visual" / clip / "Sky" / f"{index:05d}.png", (0, 0, 8, 8) if second_half else None)
        for label in VISUAL_CATEGORIES:
            if label not in ("Road", "Sky"):
                _mask(root / "visual" / clip / label / f"{index:05d}.png", None)


@pytest.fixture
def masks(tmp_path: Path) -> Path:
    return tmp_path / "masks"


class TestSegmentation:
    def test_a_composition_flip_becomes_two_shots(self, masks: Path) -> None:
        _visual(masks, "clip_001", 10, split_at=5)
        reader = ClipMasks(masks / "visual" / "clip_001", masks / "audio" / "clip_001", 1)
        calibration = Calibration(cut_threshold=0.5, minimum_shot_ms=1000, stride_ms=1000)
        assert clip_segments(reader, calibration, fps=1.0) == [(0, 5), (5, 10)]

    def test_a_clip_that_never_changes_is_one_shot(self, masks: Path) -> None:
        _visual(masks, "clip_001", 10)
        reader = ClipMasks(masks / "visual" / "clip_001", masks / "audio" / "clip_001", 1)
        assert clip_segments(reader, Calibration(), fps=1.0) == [(0, 10)]

    def test_a_missing_visual_branch_is_named_rather_than_silently_empty(self, masks: Path) -> None:
        reader = ClipMasks(masks / "visual" / "nothing", masks / "audio" / "nothing", 1)
        with pytest.raises(MaskReadError, match="no visual masks"):
            clip_segments(reader, Calibration(), fps=1.0)


class TestGrouping:
    def test_labels_over_the_same_pixels_become_one_source(self, masks: Path) -> None:
        for index in range(3):
            _mask(masks / "audio" / "c" / "Speech" / f"{index:05d}.png", (0, 0, 4, 4))
            _mask(masks / "audio" / "c" / "Footsteps" / f"{index:05d}.png", (0, 0, 4, 4))
            _mask(masks / "audio" / "c" / "Siren" / f"{index:05d}.png", (6, 6, 8, 8))
        reader = ClipMasks(masks / "visual" / "c", masks / "audio" / "c", 1)
        groups = group_audio_labels(reader, ["Speech", "Footsteps", "Siren"], 0.5)
        assert sorted(groups, key=len) == [("Siren",), ("Speech", "Footsteps")]

    def test_labels_over_different_pixels_stay_apart(self, masks: Path) -> None:
        for index in range(3):
            _mask(masks / "audio" / "c" / "Speech" / f"{index:05d}.png", (0, 0, 3, 3))
            _mask(masks / "audio" / "c" / "Siren" / f"{index:05d}.png", (5, 5, 8, 8))
        reader = ClipMasks(masks / "visual" / "c", masks / "audio" / "c", 1)
        assert group_audio_labels(reader, ["Speech", "Siren"], 0.5) == [("Speech",), ("Siren",)]

    def test_agreement_is_over_the_window_not_a_mean_of_per_frame_scores(self, masks: Path) -> None:
        """Two frames of perfect overlap must not carry four frames of none."""
        for index in range(2):
            _mask(masks / "audio" / "c" / "A" / f"{index:05d}.png", (0, 0, 4, 4))
            _mask(masks / "audio" / "c" / "B" / f"{index:05d}.png", (0, 0, 4, 4))
        for index in range(2, 6):
            _mask(masks / "audio" / "c" / "A" / f"{index:05d}.png", (0, 0, 4, 4))
            _mask(masks / "audio" / "c" / "B" / f"{index:05d}.png", (4, 4, 8, 8))
        reader = ClipMasks(masks / "visual" / "c", masks / "audio" / "c", 1)
        assert group_audio_labels(reader, ["A", "B"], 0.5) == [("A",), ("B",)]

    def test_a_group_counts_its_area_once_from_the_union(self, masks: Path) -> None:
        # Two labels each covering a quarter, overlapping on half of that.
        _mask(masks / "audio" / "c" / "A" / "00000.png", (0, 0, 4, 4))
        _mask(masks / "audio" / "c" / "B" / "00000.png", (2, 0, 6, 4))
        reader = ClipMasks(masks / "visual" / "c", masks / "audio" / "c", 1)
        share, presence = group_quantities(reader, ["A", "B"], [0])
        # Union is 6x4 of 8x8, not the 8x4 that summing the two would give.
        assert share == pytest.approx(24 / 64)
        assert presence == 1.0

    def test_presence_counts_the_frames_the_group_occupies_anything(self, masks: Path) -> None:
        _mask(masks / "audio" / "c" / "A" / "00000.png", (0, 0, 4, 4))
        _mask(masks / "audio" / "c" / "A" / "00001.png", None)
        reader = ClipMasks(masks / "visual" / "c", masks / "audio" / "c", 1)
        share, presence = group_quantities(reader, ["A"], [0, 1])
        assert presence == 0.5
        assert share == pytest.approx(16 / 64 / 2)


def _visual_per_second(root: Path, clip: str, seconds: int) -> None:
    """The release layout: one mask per label per second, named ``{label}_{second}``."""
    for second in range(seconds):
        for label in VISUAL_CATEGORIES:
            _mask(root / "visual" / clip / f"{label}_{second}.png", (0, 0, 8, 8) if label == "Road" else None)


class TestLayouts:
    def test_the_one_frame_per_second_release_layout_is_read(self, masks: Path) -> None:
        _mask(masks / "audio" / "c" / "Siren_0.png", (0, 0, 4, 4))
        _mask(masks / "audio" / "c" / "Siren_1.png", (0, 0, 4, 4))
        reader = ClipMasks(masks / "visual" / "c", masks / "audio" / "c", 1)
        assert reader.audio_labels() == ["Siren"]
        assert group_quantities(reader, ["Siren"], [0, 1])[1] == 1.0

    def test_a_per_second_index_is_a_second_whatever_the_video_frame_rate(self, masks: Path) -> None:
        _visual_per_second(masks, "c", 10)
        _mask(masks / "audio" / "c" / "Siren_0.png", (0, 0, 4, 4))
        clip = derive_clip(masks, "c", fps=60.0, downsample=1)
        shot = clip["shots"][0]
        # Ten masks a second apart are ten seconds of clip, not ten frames of sixty.
        assert (shot["start_ms"], shot["end_ms"]) == (0, 10000)
        assert shot["frames"] == [0, 10]

    def test_a_per_frame_index_is_a_frame_at_the_rate_even_when_frames_are_missing(self, masks: Path) -> None:
        for index in (0, 2, 4, 6):
            for label in VISUAL_CATEGORIES:
                _mask(
                    masks / "visual" / "c" / label / f"{index:05d}.png",
                    (0, 0, 8, 8) if label == "Road" else None,
                )
        _mask(masks / "audio" / "c" / "Siren" / "00000.png", (0, 0, 4, 4))
        clip = derive_clip(masks, "c", fps=2.0, downsample=1)
        shot = clip["shots"][0]
        # Frames 0..6 at two a second: the shot runs to the end of frame 6.
        assert (shot["start_ms"], shot["end_ms"]) == (0, 3500)
        assert shot["frames"] == [0, 7]


class TestManifest:
    def test_a_clip_becomes_shots_of_grouped_sources(self, masks: Path) -> None:
        _visual(masks, "clip_001", 10, split_at=5)
        for index in range(10):
            _mask(masks / "audio" / "clip_001" / "Speech" / f"{index:05d}.png", (0, 0, 4, 4))
            _mask(masks / "audio" / "clip_001" / "Footsteps" / f"{index:05d}.png", (0, 0, 4, 4))
        clip = derive_clip(
            masks,
            "clip_001",
            calibration=Calibration(cut_threshold=0.5, minimum_shot_ms=1000, stride_ms=1000),
            fps=1.0,
            downsample=1,
        )
        assert [shot["shot_id"] for shot in clip["shots"]] == ["s1", "s2"]
        # Labels come back in the tree's sorted order, so the group is named
        # the same way on every run whatever the filesystem hands back.
        assert clip["grouping"] == [["Footsteps", "Speech"]]
        source = clip["shots"][0]["sources"][0]
        assert source["labels"] == ["Footsteps", "Speech"]
        assert source["prose"] == "footsteps and speech"

    def test_confidence_and_energy_are_null_because_no_mask_measures_them(self, masks: Path) -> None:
        _visual(masks, "clip_001", 4)
        _mask(masks / "audio" / "clip_001" / "Siren" / "00000.png", (0, 0, 4, 4))
        clip = derive_clip(masks, "clip_001", fps=1.0, downsample=1)
        source = clip["shots"][0]["sources"][0]
        assert source["confidence"] is None and source["energy"] is None
        assert "confidence" in source["review_required"] and "energy" in source["review_required"]

    def test_a_provisional_salience_announces_itself_in_the_manifest(self, masks: Path) -> None:
        _visual(masks, "clip_001", 4)
        _mask(masks / "audio" / "clip_001" / "Siren" / "00000.png", (0, 0, 4, 4))
        manifest = derive_manifest(
            masks,
            ["clip_001"],
            fps=1.0,
            downsample=1,
            tags={
                "clip_001": {
                    "counts": Counter({"Siren": 2}),
                    "parents": {"Siren": "Sounds of things"},
                    "family_counts": Counter({"Sounds of things": 2}),
                }
            },
            provisional_salience=True,
        )
        assert manifest["schema"] == MANIFEST_SCHEMA
        assert manifest["provisional_salience"] is True
        assert any("provisional" in line for line in manifest["limitations"])
        source = manifest["clips"]["clip_001"]["shots"][0]["sources"][0]
        assert source["confidence"] is not None and source["energy"] is not None

    def test_the_provenance_records_what_r_g_depends_on(self, masks: Path) -> None:
        _visual(masks, "clip_001", 4)
        _mask(masks / "audio" / "clip_001" / "Siren" / "00000.png", (0, 0, 4, 4))
        manifest = derive_manifest(masks, ["clip_001"], fps=1.0, downsample=2)
        assert manifest["source"]["downsample"] == 2
        assert manifest["source"]["fps"] == 1.0
        assert manifest["source"]["iou_threshold"] == Calibration().iou_threshold


class TestTags:
    """The tidy CSV and the ontology, read through the shared loader."""

    def _write(self, tmp_path: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
        tidy = tmp_path / "tidy.csv"
        with tidy.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["file_index", "final_labels", "top_level_parent_name"]
            )
            writer.writeheader()
            writer.writerows(rows)
        ontology = tmp_path / "ontology.json"
        ontology.write_text(
            json.dumps(
                [
                    {"id": "things", "name": "Sounds of things", "child_ids": ["siren", "traffic"]},
                    {"id": "animal", "name": "Animal", "child_ids": ["bird"]},
                    {"id": "siren", "name": "Siren", "child_ids": []},
                    {"id": "traffic", "name": "Traffic noise", "child_ids": []},
                    {"id": "bird", "name": "Bird", "child_ids": []},
                ]
            ),
            encoding="utf-8",
        )
        return tidy, ontology

    def test_repeated_rows_do_not_multiply_a_count(self, tmp_path: Path) -> None:
        tidy, ontology = self._write(
            tmp_path,
            [
                {
                    "file_index": "c",
                    "final_labels": "Siren|Siren|Bird",
                    "top_level_parent_name": "Sounds of things:2|Animal:1",
                }
            ]
            * 3,
        )
        tags = load_tags(tidy, ontology)
        assert dict(tags["c"]["counts"]) == {"Siren": 2, "Bird": 1}
        assert tags["c"]["parents"] == {"Siren": "Sounds of things", "Bird": "Animal"}

    def test_rows_that_disagree_are_a_data_problem_not_an_average(self, tmp_path: Path) -> None:
        tidy, ontology = self._write(
            tmp_path,
            [
                {"file_index": "c", "final_labels": "Siren", "top_level_parent_name": "Sounds of things:1"},
                {"file_index": "c", "final_labels": "Bird", "top_level_parent_name": "Animal:1"},
            ],
        )
        with pytest.raises(OntologyError, match="disagree"):
            load_tags(tidy, ontology)


class TestFamilyConstraint:
    """§4 merges labels that share a physical source; a bird and a car do not."""

    def _same_blob(self, masks: Path) -> ClipMasks:
        # Three prompts, one blob: what the segmenter did on a real clip.
        for index in range(3):
            for label in ("Bird", "Traffic noise", "Vehicle horn"):
                _mask(masks / "audio" / "c" / label / f"{index:05d}.png", (0, 0, 4, 4))
        return ClipMasks(masks / "visual" / "c", masks / "audio" / "c", 1)

    def test_without_families_the_blob_becomes_one_source(self, masks: Path) -> None:
        reader = self._same_blob(masks)
        assert group_audio_labels(reader, ["Bird", "Traffic noise", "Vehicle horn"], 0.5) == [
            ("Bird", "Traffic noise", "Vehicle horn")
        ]

    def test_with_families_the_animal_stays_apart_from_the_vehicles(self, masks: Path) -> None:
        reader = self._same_blob(masks)
        families = {"Bird": "Animal", "Traffic noise": "Sounds of things", "Vehicle horn": "Sounds of things"}
        groups = group_audio_labels(reader, ["Bird", "Traffic noise", "Vehicle horn"], 0.5, families)
        assert sorted(groups, key=len) == [("Bird",), ("Traffic noise", "Vehicle horn")]

    def test_the_manifest_records_which_grouping_it_used(self, masks: Path) -> None:
        _visual(masks, "clip_001", 4)
        _mask(masks / "audio" / "clip_001" / "Bird" / "00000.png", (0, 0, 4, 4))
        _mask(masks / "audio" / "clip_001" / "Traffic noise" / "00000.png", (0, 0, 4, 4))
        loose = derive_clip(masks, "clip_001", fps=1.0, downsample=1)
        assert loose["grouping_constraint"] == "none"
        assert loose["grouping"] == [["Bird", "Traffic noise"]]
        tags: TagRecord = {
            "counts": Counter({"Bird": 1, "Traffic noise": 1}),
            "parents": {"Bird": "Animal", "Traffic noise": "Sounds of things"},
            "family_counts": Counter({"Animal": 1, "Sounds of things": 1}),
        }
        strict = derive_clip(masks, "clip_001", fps=1.0, downsample=1, tags=tags)
        assert strict["grouping_constraint"] == "family"
        assert strict["grouping"] == [["Bird"], ["Traffic noise"]]
        assert strict["families"] == tags["parents"]


class TestScaffoldedDocument:
    def test_a_manifest_scaffolds_into_a_document_that_does_not_yet_validate(
        self, masks: Path, tmp_path: Path
    ) -> None:
        import argparse

        from dpo.cli.console import _console_scaffold

        _visual(masks, "clip_001", 4)
        _mask(masks / "audio" / "clip_001" / "Siren" / "00000.png", (0, 0, 4, 4))
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(derive_manifest(masks, ["clip_001"], fps=1.0, downsample=1)), encoding="utf-8"
        )
        out = tmp_path / "console.json"
        arguments = argparse.Namespace(
            manifest=str(manifest_path),
            out=str(out),
            study_id="s",
            corpus_id="c",
            clips=None,
            half_saturation=None,
            iou_threshold=None,
            cut_threshold=None,
            minimum_shot_ms=None,
        )
        assert _console_scaffold(arguments) == 0

        from dpo.console.document import ConsoleDocumentError, load_console_document

        with pytest.raises(ConsoleDocumentError):
            load_console_document(out)


class TestSourceNaming:
    """A source is named by its label's canonical clause, not the synonym list."""

    def test_the_first_clause_names_the_source(self, masks: Path) -> None:
        _visual(masks, "clip_001", 2)
        _mask(masks / "audio" / "clip_001" / "Vehicle horn, car horn, honking" / "00000.png", (0, 0, 4, 4))
        clip = derive_clip(masks, "clip_001", fps=1.0, downsample=1)
        assert clip["shots"][0]["sources"][0]["prose"] == "vehicle horn"
        # The label itself is kept whole for the audit trail.
        assert clip["shots"][0]["sources"][0]["labels"] == ["Vehicle horn, car horn, honking"]

    def test_a_merged_source_joins_canonical_names(self, masks: Path) -> None:
        _visual(masks, "clip_001", 2)
        for label in ("Traffic noise, roadway noise", "Vehicle horn, car horn, honking"):
            _mask(masks / "audio" / "clip_001" / label / "00000.png", (0, 0, 4, 4))
        clip = derive_clip(masks, "clip_001", fps=1.0, downsample=1)
        assert clip["shots"][0]["sources"][0]["prose"] == "traffic noise and vehicle horn"


class TestMaskCache:
    """A second run over the same clip reads no PNG and gets the same numbers."""

    def _clip(self, masks: Path) -> None:
        _visual(masks, "clip_001", 6, split_at=3)
        for index in range(6):
            _mask(
                masks / "audio" / "clip_001" / "Siren" / f"{index:05d}.png",
                (0, 0, 4, 4) if index < 4 else None,
            )
            _mask(masks / "audio" / "clip_001" / "Speech" / f"{index:05d}.png", (2, 2, 6, 6))

    def test_the_cached_run_reproduces_the_uncached_manifest(self, masks: Path, tmp_path: Path) -> None:
        self._clip(masks)
        calibration = Calibration(cut_threshold=0.5, minimum_shot_ms=1000, stride_ms=1000)
        plain = derive_clip(masks, "clip_001", calibration=calibration, fps=1.0, downsample=1)
        cache_dir = tmp_path / "cache"
        first = derive_clip(
            masks, "clip_001", calibration=calibration, fps=1.0, downsample=1, cache_dir=cache_dir
        )
        second = derive_clip(
            masks, "clip_001", calibration=calibration, fps=1.0, downsample=1, cache_dir=cache_dir
        )
        assert first == plain == second
        assert (cache_dir / "clip_001-x1.npz").is_file()

    def test_the_second_run_decodes_nothing(
        self, masks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dpo.console.masks as module

        self._clip(masks)
        cache_dir = tmp_path / "cache"
        derive_clip(masks, "clip_001", fps=1.0, downsample=1, cache_dir=cache_dir)
        decoded: list[Path] = []
        original = module._decode

        def counting(path: Path, downsample: int) -> Any:
            decoded.append(path)
            return original(path, downsample)

        monkeypatch.setattr(module, "_decode", counting)
        derive_clip(masks, "clip_001", fps=1.0, downsample=1, cache_dir=cache_dir)
        assert decoded == []

    def test_the_cache_key_does_not_depend_on_how_the_path_is_spelled(self, masks: Path) -> None:
        from dpo.console.masks import MaskCache

        absolute = masks / "visual" / "c" / "Road" / "00000.png"
        relative = Path(os.path.relpath(absolute))
        assert MaskCache.key_for(absolute) == MaskCache.key_for(relative)

    def test_a_different_downsample_is_a_different_cache(self, masks: Path, tmp_path: Path) -> None:
        self._clip(masks)
        cache_dir = tmp_path / "cache"
        derive_clip(masks, "clip_001", fps=1.0, downsample=1, cache_dir=cache_dir)
        derive_clip(masks, "clip_001", fps=1.0, downsample=2, cache_dir=cache_dir)
        assert {p.name for p in cache_dir.iterdir()} == {"clip_001-x1.npz", "clip_001-x2.npz"}

    def test_without_a_cache_dir_nothing_is_written(self, masks: Path, tmp_path: Path) -> None:
        self._clip(masks)
        derive_clip(masks, "clip_001", fps=1.0, downsample=1)
        assert not list(tmp_path.rglob("*.npz"))
