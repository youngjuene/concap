"""Prepare a v2 manifest from staged calibration clips and explicit viewing media."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dpo.regen.config import FAMILY_OF_PARENT
from dpo.regen.document import load_regen_document
from dpo.regen.study_schema import SCHEMA, load_manifest


def from_legacy(paths: list[Path], language: str = "en") -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    seen = set()
    for path in paths:
        document = load_regen_document(path)
        for segment in document["segments"].values():
            if segment["clip_id"] in seen:
                continue
            seen.add(segment["clip_id"])
            clips.append(
                {
                    "id": segment["clip_id"],
                    "title": f"Calibration clip {len(clips) + 1}",
                    "video": segment["video"],
                    "duration_ms": segment["duration_ms"],
                    "frames": segment["frames"],
                    "present_families": sorted(
                        {
                            FAMILY_OF_PARENT[stem["parent"]]
                            for stem in segment["stems"]
                            if stem.get("parent") in FAMILY_OF_PARENT
                        }
                    ),
                }
            )
    return {
        "schema": SCHEMA,
        "language": language,
        "calibration_clips": clips,
        "viewing_videos": [
            {
                "id": f"viewing-{i + 1}",
                "title": f"Video {i + 1}",
                "status": "pending",
                "planned_duration_ms": 300000,
            }
            for i in range(3)
        ],
    }


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _probe(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Cannot inspect viewing source {path}") from exc
    document = json.loads(result.stdout)
    streams = [stream.get("codec_type") for stream in document.get("streams", [])]
    if "video" not in streams:
        raise ValueError(f"Viewing source has no video stream: {path}")
    if "audio" not in streams:
        raise ValueError(f"Viewing source has no audio stream: {path}")
    duration_ms = int(round(float(document["format"]["duration"]) * 1000))
    if not 299000 <= duration_ms <= 301000:
        raise ValueError(f"Viewing source is not five minutes within tolerance: {path}")
    return {"duration_ms": duration_ms}


def _relative_media(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Staged media must stay under {root}") from exc


def _copy_if_needed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.resolve() == target.resolve() or filecmp.cmp(source, target, shallow=False):
            return
        raise FileExistsError(f"Refusing to overwrite existing media: {target}")
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_wav_extract(source: Path, target: Path, duration_ms: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            "-af",
            f"apad,atrim=duration={duration_ms / 1000:.3f}",
            "-y",
            str(target),
        ],
        check=True,
        timeout=120,
    )


def _extract_wav(source: Path, target: Path, duration_ms: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing audio: {target}")
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp.wav", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        _write_wav_extract(source, temporary, duration_ms)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_existing_wav(path: Path, duration_ms: int) -> None:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 16000:
            raise ValueError(f"Existing viewing audio is not mono 16 kHz PCM WAV: {path}")
        if audio.getnframes() != duration_ms * 16:
            raise ValueError(f"Existing viewing audio is not aligned to the video duration: {path}")


def _pcm_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as audio:
        return audio.readframes(audio.getnframes())


def _validate_bound_wav(source: Path, path: Path, duration_ms: int) -> None:
    _validate_existing_wav(path, duration_ms)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.check.", suffix=".wav", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        _write_wav_extract(source, temporary, duration_ms)
        _validate_existing_wav(temporary, duration_ms)
        if _pcm_frames(temporary) != _pcm_frames(path):
            raise ValueError(f"Existing viewing audio is not bound to source video: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def cues_from_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("videos"), list):
        return {str(video["id"]): list(video["cues"]) for video in raw["videos"]}
    if isinstance(raw, dict):
        return {str(key): list(value) for key, value in raw.items()}
    raise ValueError("Cue JSON must be an object keyed by viewing video id")


def _validate_cues(cues: list[dict[str, Any]], duration_ms: int) -> None:
    end = 0
    for cue in cues:
        start = cue.get("start_ms")
        finish = cue.get("end_ms")
        if type(start) is not int or type(finish) is not int:
            raise ValueError("Cue boundaries must be integer milliseconds")
        if start != end or finish <= start or finish > duration_ms:
            raise ValueError("Viewing cues must cover the video without gaps or overlap")
        if finish - start > 30000:
            raise ValueError("Caption windows must be at most 30 seconds")
        if not isinstance(cue.get("evidence"), str) or len(cue["evidence"]) > 4000:
            raise ValueError("Each cue needs bounded, authored audiovisual evidence")
        fallback = cue.get("fallback")
        if not isinstance(fallback, dict) or not fallback:
            raise ValueError("Each cue needs explicit fallback captions")
        for text in fallback.values():
            if not isinstance(text, str) or not text.strip() or len(text) > 160:
                raise ValueError("Each cue needs bounded fallback captions")
        end = finish
    if end != duration_ms:
        raise ValueError("Cues must cover the entire viewing video")


def stage_viewing_videos(
    manifest: Mapping[str, Any],
    media_root: Path,
    sources: list[Path],
    cues_by_video: Mapping[str, list[dict[str, Any]]],
    *,
    staging_dir: str = "viewing",
    provenance: str,
) -> dict[str, Any]:
    if not provenance.strip():
        raise ValueError("Viewing staging requires explicit provenance")
    document: dict[str, Any] = json.loads(json.dumps(manifest, ensure_ascii=False))
    videos = document.get("viewing_videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("At least one viewing entry is required")
    if len(sources) != len(videos):
        raise ValueError("Viewing source count must match the manifest viewing entries")
    ids = [video["id"] for video in videos]
    if set(cues_by_video) != set(ids):
        raise ValueError("Cue JSON must cover exactly the manifest viewing video ids")
    if len({source.resolve() for source in sources}) != len(sources):
        raise ValueError("Viewing sources must be distinct")

    probed = []
    for video, source in zip(videos, sources, strict=True):
        source = source.resolve()
        if not source.is_file():
            raise ValueError(f"Missing viewing source: {source}")
        facts = _probe(source)
        cues = list(cues_by_video[video["id"]])
        _validate_cues(cues, facts["duration_ms"])
        probed.append((source, facts, cues))

    staged_entries: list[dict[str, Any]] = []
    for video, (source, facts, cues) in zip(videos, probed, strict=True):
        try:
            video_target = source.relative_to(media_root.resolve())
            staged_video = media_root.resolve() / video_target
        except ValueError:
            staged_video = media_root.resolve() / staging_dir / source.name
            _copy_if_needed(source, staged_video)
        audio_target = staged_video.with_suffix(".wav")
        if not audio_target.exists():
            _extract_wav(source, audio_target, facts["duration_ms"])
        else:
            _validate_bound_wav(source, audio_target, facts["duration_ms"])
        staged_entries.append(
            {
                "id": video["id"],
                "title": video.get("title", f"Video {len(staged_entries) + 1}"),
                "status": "ready",
                "planned_duration_ms": 300000,
                "duration_ms": facts["duration_ms"],
                "video": _relative_media(media_root, staged_video),
                "audio": _relative_media(media_root, audio_target),
                "media_provenance": provenance,
                "cues": [
                    {
                        **cue,
                        "evidence_provenance": cue.get("evidence_provenance", provenance),
                    }
                    for cue in cues
                ],
            }
        )
    document["viewing_videos"] = staged_entries
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, nargs="+", help="One or more existing regen documents")
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--output", type=Path, help="New manifest; existing files are never overwritten")
    parser.add_argument("--validate", type=Path, help="Validate a prepared or pending v2 manifest")
    parser.add_argument("--media", type=Path, help="Shared root of staged media for validation")
    parser.add_argument(
        "--stage-viewing",
        type=Path,
        help="Existing v2 manifest to copy and mark viewing videos ready",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        help="Five-minute source videos in manifest order",
    )
    parser.add_argument("--cues", type=Path, help="JSON cues keyed by viewing video id")
    parser.add_argument(
        "--staging-dir",
        default="viewing",
        help="Media-root subdirectory for copied source videos",
    )
    parser.add_argument("--provenance", help="Operator-supplied provenance for media and cue evidence")
    args = parser.parse_args()
    if args.validate:
        if args.media is None:
            parser.error("--validate requires --media")
        manifest = load_manifest(args.validate, args.media)
        print(
            json.dumps(
                {
                    "calibration_clips": len(manifest["calibration_clips"]),
                    "viewing_ready": sum(v["status"] == "ready" for v in manifest["viewing_videos"]),
                }
            )
        )
    elif args.legacy and args.output:
        document = from_legacy(args.legacy, args.language)
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(document, target, ensure_ascii=False, indent=2)
            target.write("\n")
        print(f"Created {args.output}; three viewing videos remain pending.")
    elif args.stage_viewing and args.output:
        if args.media is None or args.sources is None or args.cues is None or args.provenance is None:
            parser.error("--stage-viewing requires --media, --sources, --cues, --provenance, and --output")
        if args.output.exists():
            parser.error("--output already exists")
        source_manifest = json.loads(args.stage_viewing.read_text(encoding="utf-8"))
        document = stage_viewing_videos(
            source_manifest,
            args.media,
            args.sources,
            cues_from_file(args.cues),
            staging_dir=args.staging_dir,
            provenance=args.provenance,
        )
        temporary_manifest = args.output.with_name(f".{args.output.name}.validate.tmp")
        _atomic_json(temporary_manifest, document)
        try:
            prepared = load_manifest(temporary_manifest, args.media)
            _atomic_json(args.output, prepared)
        finally:
            temporary_manifest.unlink(missing_ok=True)
        print(f"Created {args.output}; {len(document['viewing_videos'])} viewing videos are ready.")
    else:
        parser.error("Use --legacy with --output, --stage-viewing with --output, or --validate with --media")


if __name__ == "__main__":
    main()
