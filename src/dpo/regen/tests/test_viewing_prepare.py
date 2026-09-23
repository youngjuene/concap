"""Viewing-media preparation tests for real final-study staging."""

from __future__ import annotations

import json
import subprocess
import wave
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from dpo.regen.study_prepare import cues_from_file, main, stage_viewing_videos
from dpo.regen.study_schema import load_manifest


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _video(path: Path, seconds: float, colour: str = "0x425f78") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colour}:s=96x54:r=2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=8000",
            "-t",
            f"{seconds:.3f}",
            "-c:v",
            "mpeg4",
            "-q:v",
            "31",
            "-c:a",
            "aac",
            "-b:a",
            "16k",
            "-shortest",
            "-y",
            str(path),
        ]
    )


def _tone_video(path: Path, seconds: float, frequency: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x425f78:s=96x54:r=2",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate=8000",
            "-t",
            f"{seconds:.3f}",
            "-c:v",
            "mpeg4",
            "-q:v",
            "31",
            "-c:a",
            "aac",
            "-b:a",
            "16k",
            "-shortest",
            "-y",
            str(path),
        ]
    )


def _duration_ms(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(round(float(result.stdout.strip()) * 1000))


def _rewrite_same_pcm(source: Path, target: Path) -> None:
    with wave.open(str(source), "rb") as original:
        params = original.getparams()
        frames = original.readframes(original.getnframes())
    with wave.open(str(target), "wb") as rewritten:
        rewritten.setparams(params)
        rewritten.writeframes(frames)


def _base_manifest(root: Path) -> dict[str, Any]:
    _video(root / "calibration.mp4", 2)
    Image.new("RGB", (96, 54), "#425f78").save(root / "frame.png")
    Image.new("L", (96, 54), 255).save(root / "mask.png")
    return {
        "schema": "dpo.caption-study/v2",
        "language": "en",
        "calibration_clips": [
            {
                "id": f"clip-{index}",
                "title": f"Calibration clip {index + 1}",
                "video": "calibration.mp4",
                "duration_ms": 2000,
                "present_families": ["things"],
                "frames": [
                    {
                        "at_ms": 1000,
                        "still": "frame.png",
                        "objects": [{"id": "mask", "label": "Visible object", "mask": "mask.png"}],
                    }
                ],
            }
            for index in range(2)
        ],
        "viewing_videos": [
            {
                "id": f"viewing-{index + 1}",
                "title": f"Video {index + 1}",
                "status": "pending",
                "planned_duration_ms": 300000,
            }
            for index in range(3)
        ],
    }


@pytest.fixture(scope="module")
def viewing_sources(tmp_path_factory: Any) -> tuple[Path, list[Path], dict[str, Any], list[int]]:
    root = tmp_path_factory.mktemp("viewing-prepare-media")
    manifest = _base_manifest(root)
    sources = []
    for index, seconds in enumerate((299.679, 300.259, 300.142), start=1):
        path = root / "incoming" / f"viewing-{index}.mp4"
        _video(path, seconds, ("0x425f78", "0x506b42", "0x805344")[index - 1])
        sources.append(path)
    return root, sources, manifest, [_duration_ms(source) for source in sources]


def _cues(durations: list[int]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for index, duration in enumerate(durations, start=1):
        starts = list(range(0, duration, 5000))
        result[f"viewing-{index}"] = [
            {
                "start_ms": start,
                "end_ms": min(start + 5000, duration),
                "evidence": f"Operator-authored cue evidence for viewing {index} at {start} ms.",
                "fallback": {"en": f"Sound caption {index} at {start} ms."},
            }
            for start in starts
        ]
    return result


def test_stage_viewing_videos_writes_schema_ready_entries_and_aligned_wavs(viewing_sources: Any) -> None:
    root, sources, manifest, durations = viewing_sources
    originals = [source.read_bytes() for source in sources]

    prepared = stage_viewing_videos(
        manifest,
        root,
        sources,
        _cues(durations),
        provenance="operator-authored-final-survey-test",
    )

    path = root / "prepared.json"
    path.write_text(json.dumps(prepared, ensure_ascii=False), encoding="utf-8")
    loaded = load_manifest(path, root)
    assert [video["status"] for video in loaded["viewing_videos"]] == ["ready", "ready", "ready"]
    assert [video["duration_ms"] for video in loaded["viewing_videos"]] == durations
    assert loaded["viewing_videos"][0]["cues"][-1]["end_ms"] == durations[0]
    assert loaded["viewing_videos"][0]["cues"][-1]["start_ms"] == 295000
    assert [source.read_bytes() for source in sources] == originals

    for video in loaded["viewing_videos"]:
        with wave.open(str(root / video["audio"]), "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 16000
            assert audio.getnframes() == video["duration_ms"] * 16


def test_stage_viewing_videos_rejects_missing_or_invalid_cues_without_ready_manifest(
    viewing_sources: Any, tmp_path: Path
) -> None:
    root, sources, manifest, durations = viewing_sources
    output = tmp_path / "should-not-exist.json"
    bad = _cues(durations)
    bad.pop("viewing-3")

    with pytest.raises(ValueError, match="cover exactly"):
        stage_viewing_videos(
            manifest,
            root,
            sources,
            bad,
            provenance="operator-authored-final-survey-test",
        )
    assert not output.exists()

    bad = _cues(durations)
    bad["viewing-1"][1]["start_ms"] += 1
    with pytest.raises(ValueError, match="without gaps"):
        stage_viewing_videos(
            manifest,
            root,
            sources,
            bad,
            provenance="operator-authored-final-survey-test",
        )
    assert not output.exists()


def test_stage_viewing_videos_rejects_missing_sources_and_provenance(viewing_sources: Any) -> None:
    root, sources, manifest, durations = viewing_sources
    cues = _cues(durations)

    with pytest.raises(ValueError, match="provenance"):
        stage_viewing_videos(manifest, root, sources, cues, provenance="")
    with pytest.raises(ValueError, match="Missing viewing source"):
        stage_viewing_videos(
            manifest,
            root,
            [sources[0], root / "missing.mp4", sources[2]],
            cues,
            provenance="operator-authored-final-survey-test",
        )


def test_stage_viewing_videos_refuses_to_overwrite_unrelated_media(
    viewing_sources: Any, tmp_path: Path
) -> None:
    root, sources, manifest, durations = viewing_sources
    target_root = tmp_path / "media"
    target_manifest = deepcopy(manifest)
    for source in ("calibration.mp4", "frame.png", "mask.png"):
        (target_root / source).parent.mkdir(parents=True, exist_ok=True)
        (target_root / source).write_bytes((root / source).read_bytes())
    conflict = target_root / "viewing" / sources[0].name
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_bytes(b"unrelated")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage_viewing_videos(
            target_manifest,
            target_root,
            sources,
            _cues(durations),
            provenance="operator-authored-final-survey-test",
        )


def test_stage_viewing_videos_rejects_same_duration_stale_existing_wav(tmp_path: Path) -> None:
    root = tmp_path / "media"
    manifest = _base_manifest(root)
    sources = []
    for index, frequency in enumerate((440, 660, 880), start=1):
        source = root / "viewing" / f"viewing-{index}.mp4"
        _tone_video(source, 299.679, frequency)
        sources.append(source)
    durations = [_duration_ms(source) for source in sources]
    stage_viewing_videos(
        manifest,
        root,
        sources,
        _cues(durations),
        provenance="operator-authored-final-survey-test",
    )
    stale_audio = root / "viewing" / "viewing-1.wav"
    original_bytes = stale_audio.read_bytes()
    stale_audio.write_bytes((root / "viewing" / "viewing-2.wav").read_bytes())
    stale_bytes = stale_audio.read_bytes()

    with pytest.raises(ValueError, match="not bound to source video"):
        stage_viewing_videos(
            manifest,
            root,
            sources,
            _cues(durations),
            provenance="operator-authored-final-survey-test",
        )
    assert stale_audio.read_bytes() == stale_bytes
    assert stale_audio.read_bytes() != original_bytes


def test_stage_viewing_videos_accepts_equivalent_pcm_existing_wav(tmp_path: Path) -> None:
    root = tmp_path / "media"
    manifest = _base_manifest(root)
    sources = []
    for index, frequency in enumerate((440, 660, 880), start=1):
        source = root / "viewing" / f"viewing-{index}.mp4"
        _tone_video(source, 299.679, frequency)
        sources.append(source)
    durations = [_duration_ms(source) for source in sources]
    stage_viewing_videos(
        manifest,
        root,
        sources,
        _cues(durations),
        provenance="operator-authored-final-survey-test",
    )
    audio = root / "viewing" / "viewing-1.wav"
    rewritten = root / "viewing" / "rewritten.wav"
    _rewrite_same_pcm(audio, rewritten)
    original_file_bytes = audio.read_bytes()
    audio.write_bytes(rewritten.read_bytes())

    prepared = stage_viewing_videos(
        manifest,
        root,
        sources,
        _cues(durations),
        provenance="operator-authored-final-survey-test",
    )

    assert prepared["viewing_videos"][0]["status"] == "ready"
    assert audio.read_bytes() != original_file_bytes


def test_cues_from_file_accepts_keyed_or_video_list_shape(tmp_path: Path) -> None:
    keyed = tmp_path / "keyed.json"
    keyed.write_text(json.dumps({"viewing-1": []}), encoding="utf-8")
    listed = tmp_path / "listed.json"
    listed.write_text(json.dumps({"videos": [{"id": "viewing-1", "cues": []}]}), encoding="utf-8")

    assert cues_from_file(keyed) == {"viewing-1": []}
    assert cues_from_file(listed) == {"viewing-1": []}


def test_cli_stage_viewing_writes_validated_manifest(
    viewing_sources: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sources, manifest, durations = viewing_sources
    manifest_path = tmp_path / "pending.json"
    cues_path = tmp_path / "cues.json"
    output = tmp_path / "ready.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    cues_path.write_text(json.dumps(_cues(durations), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "study_prepare",
            "--stage-viewing",
            str(manifest_path),
            "--media",
            str(root),
            "--sources",
            *(str(source) for source in sources),
            "--cues",
            str(cues_path),
            "--provenance",
            "operator-authored-final-survey-test",
            "--output",
            str(output),
        ],
    )

    main()

    loaded = load_manifest(output, root)
    assert [video["status"] for video in loaded["viewing_videos"]] == ["ready", "ready", "ready"]
