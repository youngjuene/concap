"""`dpo regen`: the three commands, and what each refuses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from dpo.cli import build_parser
from dpo.cli.regen import _regen_scaffold, _regen_validate
from dpo.regen.config import load_configuration
from dpo.regen.document import REGEN_SCHEMA


def _scaffold_arguments(tmp_path: Path, media: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "media_dir": str(media),
        "out": str(tmp_path / "session.json"),
        "session_id": "street-regen",
        "study_id": "street2026",
        "corpus_id": "amsterdam",
        "clips": ["amsterdam_006", "amsterdam_012"],
        "duration_ms": 10000,
        "cue_slots": None,
        "minimum_points": None,
        "latency_ceiling_ms": None,
        "language": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestParser:
    def test_the_command_is_attached_beside_the_other_instruments(self) -> None:
        parsed = build_parser().parse_args(
            ["regen", "serve", "--session", "s.json", "--media-dir", "m", "--out", "o"]
        )
        assert parsed.command == "regen"
        assert parsed.action == "serve"

    def test_it_serves_one_port_past_the_console(self) -> None:
        parsed = build_parser().parse_args(
            ["regen", "serve", "--session", "s.json", "--media-dir", "m", "--out", "o"]
        )
        assert parsed.port == 8779


class TestScaffold:
    def test_it_writes_a_document_from_a_staged_media_directory(
        self, media_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _regen_scaffold(_scaffold_arguments(tmp_path, media_dir)) == 0
        document = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
        assert document["schema"] == REGEN_SCHEMA
        assert set(document["segments"]) == {"A", "B"}
        strip = document["segments"]["A"]["frames"]
        assert len(strip) == 5, "one entry per frame the media directory staged"
        assert [entry["label"] for entry in strip[0]["objects"]] == ["building", "person"]
        # Every frame carries the masks cut from that frame, not the segment's.
        assert len({frame["objects"][0]["mask"] for frame in strip}) == len(strip)

    def test_it_says_what_a_researcher_still_has_to_author(
        self, media_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _regen_scaffold(_scaffold_arguments(tmp_path, media_dir))
        emitted = json.loads(capsys.readouterr().out)
        assert emitted["status"] == "scaffolded"
        assert "segments.*.prepared_track[*].text" in emitted["authoring_required"]
        assert "segments.*.frames[*].at_ms" in emitted["authoring_required"]

    def test_the_slot_count_reaches_both_tracks_of_both_segments(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        _regen_scaffold(_scaffold_arguments(tmp_path, media_dir, cue_slots=6))
        document = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
        for name in ("A", "B"):
            for track in ("prepared_track", "fallback_track"):
                assert len(document["segments"][name][track]) == 6
        assert load_configuration(document["config"]).cue_slots == 6

    def test_a_scaffold_never_validates(
        self, media_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The prepared track is the study's stimulus and no tool can write it;
        # a scaffold that passed would let empty captions reach a participant.
        _regen_scaffold(_scaffold_arguments(tmp_path, media_dir))
        capsys.readouterr()
        arguments = argparse.Namespace(session=str(tmp_path / "session.json"), items=None)
        assert _regen_validate(arguments) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "invalid"

    def test_a_media_directory_without_frames_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bare = tmp_path / "bare"
        (bare / "A" / "frames").mkdir(parents=True)
        (bare / "A" / "stems").mkdir(parents=True)
        assert _regen_scaffold(_scaffold_arguments(tmp_path, bare)) == 2
        assert "no frames" in json.loads(capsys.readouterr().out)["error"]

    def test_one_clip_id_is_refused(
        self, media_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _regen_scaffold(_scaffold_arguments(tmp_path, media_dir, clips=["only_one"])) == 2
        assert "exactly two clip ids" in json.loads(capsys.readouterr().out)["error"]

    def test_a_calibration_flag_the_configuration_refuses_reads_as_a_refusal(
        self, media_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A researcher's typo, not a crash: the same shape as the other errors."""
        assert _regen_scaffold(_scaffold_arguments(tmp_path, media_dir, cue_slots=0)) == 2
        assert "at least one cue slot" in json.loads(capsys.readouterr().out)["error"]


class TestValidate:
    def test_an_authored_document_passes_and_reports_its_stamp(
        self, document: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "session.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert _regen_validate(argparse.Namespace(session=str(path), items=None)) == 0
        emitted = json.loads(capsys.readouterr().out)
        assert emitted["status"] == "valid"
        assert emitted["cue_slots"] == 4
        assert emitted["items_provenance"] == "placeholder"

    def test_it_names_the_path_that_fails(
        self, document: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document["segments"]["A"]["stems"][0]["colour"] = "blue"
        path = tmp_path / "session.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert _regen_validate(argparse.Namespace(session=str(path), items=None)) == 2
        assert "stems[0].colour" in json.loads(capsys.readouterr().out)["error"]
