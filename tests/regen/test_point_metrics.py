"""Whole-mask occupancy uses the selected frame and the exact submitted file bytes."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from dpo.regen import points as points_api
from dpo.regen.points import MaskObject, Point, PointError, match_points
from tests.regen.test_sheet_backend_regressions import questionnaire, send


def frame(path: Path, size: tuple[int, int] = (10, 100), colour: str = "white") -> Path:
    Image.new("RGB", size, colour).save(path)
    return path


def mask(path: Path, size: tuple[int, int], box: tuple[int, int, int, int]) -> MaskObject:
    image = Image.new("L", size, 0)
    image.paste(255, box)
    image.save(path)
    return MaskObject(path.stem, path.stem.title(), path)


def measured(point: Point, objects: list[MaskObject], image: Path) -> Any:
    return points_api.measure_points([point], {point.frame: objects}, {point.frame: image})[0]


def test_100_pixel_mask_over_1000_pixel_frame_is_point_one(tmp_path: Path) -> None:
    image = frame(tmp_path / "frame.png")
    selected = mask(tmp_path / "selected.png", (10, 100), (0, 0, 10, 10))
    point = Point(0, 0, 0.2, 0.05)
    result = measured(point, [selected], image)
    metric = result.metrics
    assert result.match == match_points([point], {0: [selected]})[0]
    assert metric["selected_mask_area_px"] == 100 and metric["frame_area_px"] == 1000
    assert metric["selected_mask_area_ratio"] == 0.1
    assert metric["frame_width"] == 10 and metric["frame_height"] == 100
    assert metric["mask_width"] == 10 and metric["mask_height"] == 100
    assert metric["selected_mask_id"] == "selected" and metric["label"] == "Selected"
    assert metric["status"] == "matched" and metric["frame_id"] == "0"
    assert metric["matching_rule"] == "smallest-containing-mask-v1" and metric["threshold"] == 127
    assert metric["frame_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert metric["mask_sha256"] == hashlib.sha256(selected.path.read_bytes()).hexdigest()


def test_nested_masks_keep_whole_area_and_all_hit_candidates(tmp_path: Path) -> None:
    image = frame(tmp_path / "frame.png", (20, 20))
    large = mask(tmp_path / "large.png", (20, 20), (0, 0, 20, 20))
    small = mask(tmp_path / "small.png", (20, 20), (0, 0, 10, 10))
    tiny = mask(tmp_path / "tiny.png", (20, 20), (0, 0, 4, 4))
    result = measured(Point(0, 0, 0.4, 0.4), [large, small, tiny], image)
    assert result.match.object_id == "small"
    assert result.metrics["selected_mask_area_px"] == 100  # Not 100 - 16 winner pixels.
    candidates = result.metrics["candidate_masks"]
    assert [candidate["object_id"] for candidate in candidates] == ["large", "small"]
    assert [candidate["mask_area_ratio"] for candidate in candidates] == [1.0, 0.25]
    assert [candidate["mask_area_px"] for candidate in candidates] == [400, 100]
    assert sum(candidate["mask_area_ratio"] for candidate in candidates) > 1
    tied = MaskObject("same-area", "Second in document", small.path)
    assert measured(Point(0, 0, 0.4, 0.4), [tied, small], image).match.object_id == "same-area"


def test_no_hit_is_na_not_zero(tmp_path: Path) -> None:
    image = frame(tmp_path / "frame.png")
    selected = mask(tmp_path / "selected.png", (10, 100), (0, 0, 10, 10))
    metric = measured(Point(0, 0, 0.5, 0.9), [selected], image).metrics
    assert metric["status"] == "unclassified" and metric["label"] == "unclassified"
    assert metric["candidate_masks"] == []
    for key in (
        "selected_mask_id",
        "selected_mask_area_px",
        "selected_mask_area_ratio",
        "mask_sha256",
        "mask_width",
        "mask_height",
    ):
        assert metric[key] is None
    assert metric["frame_area_px"] == 1000


def test_resolution_mismatch_preserves_legacy_choice_but_not_occupancy(tmp_path: Path) -> None:
    image = frame(tmp_path / "frame.png", (10, 100))
    different = mask(tmp_path / "different.png", (20, 50), (0, 0, 20, 50))
    point = Point(0, 0, 0.5, 0.5)
    result = measured(point, [different], image)
    assert result.match == match_points([point], {0: [different]})[0]
    assert result.match.label == "Different"
    assert result.metrics["status"] == "resolution_mismatch"
    assert result.metrics["selected_mask_area_ratio"] is None
    assert result.metrics["selected_mask_area_px"] == 1000
    assert result.metrics["candidate_masks"][0]["mask_area_ratio"] is None
    assert result.metrics["mask_width"] == 20 and result.metrics["mask_height"] == 50


def test_same_path_size_and_mtime_rewrite_does_not_reuse_old_decoded_mask(tmp_path: Path) -> None:
    image = frame(tmp_path / "frame.bmp")
    selected = mask(tmp_path / "selected.bmp", (10, 100), (0, 0, 10, 10))
    before = selected.path.stat()
    point = Point(0, 0, 0.2, 0.15)
    first = measured(point, [selected], image)
    assert first.match.object_id is None
    mask(selected.path, (10, 100), (0, 0, 10, 20))
    os.utime(selected.path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert selected.path.stat().st_size == before.st_size
    assert selected.path.stat().st_mtime_ns == before.st_mtime_ns
    second = measured(point, [selected], image)
    assert second.match.object_id == "selected"
    assert second.metrics["selected_mask_area_px"] == 200
    assert second.metrics["mask_sha256"] == hashlib.sha256(selected.path.read_bytes()).hexdigest()
    assert match_points([point], {0: [selected]})[0].object_id == "selected"


def test_bytes_used_for_hash_and_geometry_are_read_once_per_submission(
    tmp_path: Path, monkeypatch: Any
) -> None:
    image = frame(tmp_path / "frame.bmp")
    selected = mask(tmp_path / "selected.bmp", (10, 100), (0, 0, 10, 10))
    old_frame, old_mask = image.read_bytes(), selected.path.read_bytes()
    replacement_frame = frame(tmp_path / "replacement-frame.bmp", (20, 50), "black").read_bytes()
    replacement_mask = mask(tmp_path / "replacement-mask.bmp", (10, 100), (0, 0, 10, 20)).path.read_bytes()
    original = Path.read_bytes
    reads: Counter[Path] = Counter()

    def read_then_replace(path: Path) -> bytes:
        contents = original(path)
        reads[path] += 1
        if path == image:
            path.write_bytes(replacement_frame)
        elif path == selected.path:
            path.write_bytes(replacement_mask)
        return contents

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    results = points_api.measure_points(
        [Point(0, 0, 0.1, 0.05), Point(1, 0, 0.2, 0.05)], {0: [selected]}, {0: image}
    )
    assert reads[image] == reads[selected.path] == 1
    for result in results:
        assert result.metrics["selected_mask_area_px"] == 100
        assert result.metrics["frame_width"] == 10 and result.metrics["frame_height"] == 100
        assert result.metrics["frame_sha256"] == hashlib.sha256(old_frame).hexdigest()
        assert result.metrics["mask_sha256"] == hashlib.sha256(old_mask).hexdigest()


def test_each_point_uses_its_own_frame_and_threshold_is_strict(tmp_path: Path) -> None:
    first = frame(tmp_path / "first.png", (4, 1))
    second = frame(tmp_path / "second.png", (10, 10))
    threshold = Image.new("L", (4, 1))
    threshold.putdata([127, 128, 0, 255])
    threshold.save(tmp_path / "threshold.png")
    a = MaskObject("threshold", "Threshold", tmp_path / "threshold.png")
    b = mask(tmp_path / "other.png", (10, 10), (0, 0, 5, 5))
    results = points_api.measure_points(
        [Point(0, 0, 1 / 3, 0), Point(1, 0, 0, 0), Point(2, 1, 0.1, 0.1)],
        {0: [a], 1: [b]},
        {0: first, 1: second},
    )
    assert results[0].metrics["selected_mask_area_px"] == 2
    assert results[1].metrics["status"] == "unclassified"
    assert results[2].metrics["frame_id"] == "1"
    assert results[2].metrics["selected_mask_area_ratio"] == 0.25
    assert results[2].metrics["frame_sha256"] == hashlib.sha256(second.read_bytes()).hexdigest()


@pytest.mark.parametrize("page,code", [("P2", "V1"), ("P7", "V2")])
def test_sheet_visual_submission_stores_metrics_atomically(
    document: Any, media_dir: Path, tmp_path: Path, page: str, code: str
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    token = protocol.create()
    protocol.store.mutate(
        token,
        "visual-fixture",
        protocol.checked(token)["revision"],
        "fixture",
        lambda state: state.update(sheet_page=page, sheet_consent={"accepted": True}),
    )
    send(protocol, token, "submit", {"answers": {code: [{"frame_id": "0", "x": 0.2, "y": 0.2}]}})
    record = protocol.store.state(token)["sheet_responses"][-1]
    assert record["matches"] == [
        {"order": 0, "frame": 0, "x": 0.2, "y": 0.2, "object_id": "person", "label": "Person"}
    ]
    assert record["labels"] == ["Person"]
    assert len(record["visual_metrics"]) == 1
    assert record["visual_metrics"][0]["selected_mask_area_ratio"] == 0.04


def test_unreadable_frame_rolls_back_visual_submission(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    token = protocol.create()
    protocol.store.mutate(
        token,
        "visual-fixture",
        protocol.checked(token)["revision"],
        "fixture",
        lambda state: state.update(sheet_page="P2", sheet_consent={"accepted": True}),
    )
    (media_dir / document["segments"]["A"]["frames"][0]["still"]).write_bytes(b"invalid image")
    before = protocol.store.export(token)
    with pytest.raises(PointError):
        send(protocol, token, "submit", {"answers": {"V1": [{"frame_id": "0", "x": 0.2, "y": 0.2}]}})
    assert protocol.store.export(token) == before


@pytest.mark.parametrize("mapping", [None, [], "invalid", {"traffic": []}, {}])
def test_missing_or_malformed_family_mapping_preserves_raw_selection(
    document: Any, media_dir: Path, tmp_path: Path, mapping: Any
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    token = protocol.create()
    state = protocol.checked(token)
    state["sheet_responses"] = [
        {"page": page, "clip_index": index, "answers": {"A1s": "traffic"}}
        for index in range(2)
        for page in ("P2", "P3")
    ]
    for entry in state["sheet_config"]["clips"].values():
        entry["source_families"] = mapping
    rows = protocol._observations(state)
    assert all(row["selected_sound"] == "traffic" and row["heard"] == {} for row in rows)


def test_handoff_uses_declared_type_alias_and_family_without_correctness_checks(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    state = protocol.checked(protocol.create())
    state["sheet_responses"] = [
        {"page": page, "clip_index": index, "answers": {"A1s": "traffic"}}
        for index in range(2)
        for page in ("P2", "P3")
    ]
    for entry in state["sheet_config"]["clips"].values():
        entry["source_families"] = []
        entry["analysis_codebook"] = {
            "version": "custom-v1",
            "sound_types": {"traffic": "engine"},
            "sound_families": {"engine": "researcher-declared-family"},
        }
    rows = protocol._observations(state)
    assert all(row["selected_sound"] == "traffic" for row in rows)
    assert all(row["heard"] == {"researcher-declared-family": True} for row in rows)
