"""`dpo console`: the four commands, and what each refuses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from dpo.cli import build_parser
from dpo.cli.console import ConsoleUsageError, _console_scaffold, _console_serve, _console_validate
from dpo.console.config import load_configuration
from dpo.console.masks import MANIFEST_SCHEMA

FIXTURE = Path(__file__).parent / "fixtures" / "console.json"


def _scaffold_arguments(tmp_path: Path, manifest: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "manifest": str(manifest),
        "out": str(tmp_path / "console.json"),
        "study_id": "street2026",
        "corpus_id": "amsterdam",
        "clips": None,
        "half_saturation": None,
        "iou_threshold": None,
        "cut_threshold": None,
        "minimum_shot_ms": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _manifest(
    path: Path, clip_id: str = "clip_001", *, provisional: bool = False, iou_threshold: float = 0.5
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "source": {
                    "fps": 60.0,
                    "downsample": 4,
                    "iou_threshold": iou_threshold,
                    "cut_threshold": 0.25,
                    "minimum_shot_ms": 2000,
                    "stride_ms": 250,
                },
                "limitations": [],
                "provisional_salience": provisional,
                "clips": {
                    clip_id: {
                        "clip_id": clip_id,
                        "opening": {"grain": "itemized", "alpha": 0.5, "admitted": None},
                        "grouping": [["Siren"]],
                        "shots": [
                            {
                                "shot_id": "s1",
                                "start_ms": 0,
                                "end_ms": 10000,
                                "raw_caption": "",
                                "default_caption": "",
                                "frames": [0, 600],
                                "sources": [
                                    {
                                        "id": "siren",
                                        "labels": ["Siren"],
                                        "prose": "siren",
                                        "share": 0.05,
                                        "presence": 0.8,
                                        "confidence": None,
                                        "energy": None,
                                        "review_required": ["prose", "confidence", "energy"],
                                    }
                                ],
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class TestParser:
    def test_the_console_command_is_registered_beside_the_session_one(self) -> None:
        parser = build_parser()
        arguments = parser.parse_args(["console", "validate", "--session", str(FIXTURE)])
        assert arguments.handler is _console_validate

    def test_the_two_instruments_default_to_different_ports(self) -> None:
        parser = build_parser()
        console = parser.parse_args(["console", "serve", "--session", "s", "--media-dir", "m", "--out", "o"])
        session = parser.parse_args(["session", "serve", "--session", "s", "--media-dir", "m", "--out", "o"])
        assert console.port == 8778
        assert session.port == 8777


class TestValidate:
    def test_the_demo_fixture_validates_and_reports_its_stamp(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _console_validate(argparse.Namespace(session=str(FIXTURE))) == 0
        emitted = json.loads(capsys.readouterr().out)
        assert emitted["status"] == "valid"
        assert len(emitted["config_hash"]) == 12
        assert emitted["shots"] == 2
        # The operator sees on the same line whether this is a dry run.
        assert emitted["provisional_salience"] is False

    def test_an_invalid_document_names_the_path_and_never_the_command_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = json.loads(FIXTURE.read_text(encoding="utf-8"))
        broken["clips"][0]["shots"][0]["raw_caption"] = ""
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(broken), encoding="utf-8")

        assert _console_validate(argparse.Namespace(session=str(path))) == 2
        emitted = json.loads(capsys.readouterr().out)
        assert emitted["status"] == "invalid"
        assert "raw_caption" in emitted["error"]


class TestScaffold:
    def test_a_manifest_becomes_a_document_carrying_the_named_calibration(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = _manifest(tmp_path / "manifest.json")
        arguments = _scaffold_arguments(tmp_path, manifest, half_saturation=0.05)
        assert _console_scaffold(arguments) == 0
        emitted = json.loads(capsys.readouterr().out)

        document = json.loads((tmp_path / "console.json").read_text(encoding="utf-8"))
        assert document["config"]["calibration"]["half_saturation"] == 0.05
        assert document["config"]["study_id"] == "street2026"
        # The stamp the command reported is the stamp the document carries.
        assert emitted["config_hash"] == load_configuration(document["config"]).hash
        assert document["config"]["provisional_salience"] is False
        assert emitted["provisional_salience"] is False
        assert "never recruit" not in emitted["next"]

    def test_a_provisional_manifest_scaffolds_a_document_that_says_so_in_its_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        provisional = _manifest(tmp_path / "provisional.json", provisional=True)
        assert _console_scaffold(_scaffold_arguments(tmp_path, provisional)) == 0
        emitted = json.loads(capsys.readouterr().out)

        document = json.loads((tmp_path / "console.json").read_text(encoding="utf-8"))
        # The flag travels from the manifest into the document (and so into
        # the stamp: test_config pins that a dry run never shares one), and
        # the operator is told on the same line.
        assert document["config"]["provisional_salience"] is True
        assert emitted["provisional_salience"] is True
        assert "never recruit" in emitted["next"]

    def test_the_stamp_carries_the_thresholds_the_manifest_was_computed_under(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        loose = _manifest(tmp_path / "loose.json", iou_threshold=0.3)
        tight = _manifest(tmp_path / "tight.json", iou_threshold=0.7)
        assert (
            _console_scaffold(_scaffold_arguments(tmp_path, loose, out=str(tmp_path / "loose-doc.json"))) == 0
        )
        first = json.loads(capsys.readouterr().out)
        assert (
            _console_scaffold(_scaffold_arguments(tmp_path, tight, out=str(tmp_path / "tight-doc.json"))) == 0
        )
        second = json.loads(capsys.readouterr().out)
        document = json.loads((tmp_path / "loose-doc.json").read_text(encoding="utf-8"))
        # Two preprocessings under different thresholds are two studies; the
        # stamp says which, from the manifest and not from a scaffold flag.
        assert document["config"]["calibration"]["iou_threshold"] == 0.3
        assert first["config_hash"] != second["config_hash"]

    def test_scaffold_takes_no_preprocessing_threshold_of_its_own(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "console",
                    "scaffold",
                    "--manifest",
                    "m",
                    "--out",
                    "o",
                    "--study-id",
                    "s",
                    "--corpus-id",
                    "c",
                    "--iou-threshold",
                    "0.3",
                ]
            )

    def test_the_calibration_flags_change_the_stamp(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path / "manifest.json")
        stamps = []
        for r0 in (0.02, 0.05):
            out = tmp_path / f"console-{r0}.json"
            _console_scaffold(_scaffold_arguments(tmp_path, manifest, out=str(out), half_saturation=r0))
            stamps.append(json.loads(out.read_text(encoding="utf-8"))["config"]["calibration"])
        assert stamps[0] != stamps[1]

    def test_the_manifest_provenance_does_not_leak_into_the_document(self, tmp_path: Path) -> None:
        """``review_required`` and ``frames`` are for the researcher, not the instrument."""
        manifest = _manifest(tmp_path / "manifest.json")
        _console_scaffold(_scaffold_arguments(tmp_path, manifest))
        document = json.loads((tmp_path / "console.json").read_text(encoding="utf-8"))
        shot = document["clips"][0]["shots"][0]
        assert "frames" not in shot
        assert "review_required" not in shot["sources"][0]

    def test_a_clip_the_manifest_never_preprocessed_is_an_error(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path / "manifest.json")
        assert _console_scaffold(_scaffold_arguments(tmp_path, manifest, clips=["clip_404"])) == 2

    def test_a_file_of_the_wrong_schema_is_refused(self, tmp_path: Path) -> None:
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"schema": "something/v1", "clips": {}}), encoding="utf-8")
        assert _console_scaffold(_scaffold_arguments(tmp_path, other)) == 2


class TestServe:
    def _arguments(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, Any] = {
            "session": str(FIXTURE),
            "media_dir": "media",
            "out": "out",
            "writer": "gemma",
            "backend_config": None,
            "contract": "configs/study/street-audio.toml",
            "checkpoint": None,
            "host": "127.0.0.1",
            "port": 8778,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_gemma_without_a_backend_config_blames_the_command_line_not_the_document(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from dpo.cli.console import _gemma_writer

        with pytest.raises(ConsoleUsageError, match="--backend-config"):
            _gemma_writer(self._arguments(), {})
        assert _console_serve(self._arguments()) == 2
        emitted = json.loads(capsys.readouterr().out)
        assert emitted["status"] == "error" and emitted["command"] == "console serve"
        assert "session" not in emitted
