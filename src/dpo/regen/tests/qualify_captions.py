"""Reusable caption-control qualification helpers for the Regen viewing study.

This file is intentionally mechanical. It verifies that the software sends
different, cache-distinct instructions for the two control axes and records
timing/fallback evidence for a worker path. It does not score caption quality,
grounding, or human-perceived independence of the controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dpo.regen import study_schema
from dpo.regen.study_schema import caption_instruction, digest, load_manifest
from dpo.regen.study_store import StudyStore
from dpo.regen.study_worker import InferenceProcess

AXIS_CORNERS: tuple[tuple[str, dict[str, float]], ...] = (
    ("low_texture_low_context", {"texture": 0.0, "context": 0.0}),
    ("high_texture_low_context", {"texture": 1.0, "context": 0.0}),
    ("low_texture_high_context", {"texture": 0.0, "context": 1.0}),
    ("high_texture_high_context", {"texture": 1.0, "context": 1.0}),
)


def default_profile(language: str = "en") -> dict[str, Any]:
    return {
        "version": 1,
        "language": language,
        "sample_count": 3,
        "defaults": {"texture": 0.5, "context": 0.5},
        "template": (
            f"Write sound captions in {'Korean' if language == 'ko' else 'English'}. "
            "Describe only supported audible events in the current excerpt. "
            "Current evidence and requested detail levels take priority."
        ),
        "hash": "qualification-profile",
    }


def default_cue() -> dict[str, Any]:
    return {
        "start_ms": 0,
        "end_ms": 5000,
        "evidence": "A bus idles beside a crosswalk while tires pass on wet pavement.",
        "fallback": {"en": "A bus engine rumbles.", "ko": "버스 엔진 소리가 울린다."},
        "evidence_provenance": "synthetic_unverified_default",
    }


def cue_from_args(
    *,
    evidence: str | None,
    fallback: str | None,
    language: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    cue = default_cue()
    if evidence is not None:
        cue["evidence"] = evidence
        cue["evidence_provenance"] = "cli_operator_supplied_unscored"
    if fallback is not None:
        cue["fallback"] = {language: fallback}
    cue["start_ms"] = start_ms
    cue["end_ms"] = end_ms
    return cue


def axis_matrix(
    profile: Mapping[str, Any] | None = None,
    cue: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    profile = profile or default_profile()
    cue = cue or default_cue()
    rows = []
    for name, axes in AXIS_CORNERS:
        instruction = caption_instruction(profile, axes, cue)
        rows.append(
            {
                "name": name,
                "axes": axes,
                "instruction_hash": digest({"instruction": instruction})[:16],
                "instruction": instruction,
                "descriptor_limit": _descriptor_limit(instruction),
                "context_policy": _context_policy(instruction),
            }
        )
    return rows


def qualify_axis_contract(
    profile: Mapping[str, Any] | None = None,
    cue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cue = cue or default_cue()
    rows = axis_matrix(profile, cue)
    by_name = {row["name"]: row for row in rows}
    checks = {
        "four_corners_present": len(rows) == 4,
        "all_instruction_hashes_distinct": len({row["instruction_hash"] for row in rows}) == 4,
        "texture_changes_descriptor_budget_at_low_context": (
            by_name["low_texture_low_context"]["descriptor_limit"]
            < by_name["high_texture_low_context"]["descriptor_limit"]
        ),
        "texture_changes_descriptor_budget_at_high_context": (
            by_name["low_texture_high_context"]["descriptor_limit"]
            < by_name["high_texture_high_context"]["descriptor_limit"]
        ),
        "context_policy_independent_of_texture_low_context": (
            by_name["low_texture_low_context"]["context_policy"]
            == by_name["high_texture_low_context"]["context_policy"]
        ),
        "context_policy_independent_of_texture_high_context": (
            by_name["low_texture_high_context"]["context_policy"]
            == by_name["high_texture_high_context"]["context_policy"]
        ),
        "context_policy_changes_at_low_texture": (
            by_name["low_texture_low_context"]["context_policy"]
            != by_name["low_texture_high_context"]["context_policy"]
        ),
        "context_policy_changes_at_high_texture": (
            by_name["high_texture_low_context"]["context_policy"]
            != by_name["high_texture_high_context"]["context_policy"]
        ),
        "uses_excerpt_relative_time": all(
            "starts at 0" in row["instruction"]
            and (cue["start_ms"] == 0 or str(cue["start_ms"]) not in row["instruction"])
            for row in rows
        ),
        "includes_authored_evidence": all(str(cue["evidence"]) in row["instruction"] for row in rows),
    }
    return {
        "kind": "axis_contract",
        "passed": all(checks.values()),
        "checks": checks,
        "matrix": [{key: row[key] for key in ("name", "axes", "instruction_hash")} for row in rows],
        "limits": {
            row["name"]: {
                "descriptor_limit": row["descriptor_limit"],
                "context_policy": row["context_policy"],
            }
            for row in rows
        },
        "quality_scope": "mechanical contract only; no grounding or human independence score",
        "evidence_provenance": cue.get("evidence_provenance", "unknown"),
    }


def qualify_repeated_inference(
    engine: InferenceProcess,
    spec: Mapping[str, Any],
    *,
    timeout: float,
    repeats: int = 2,
) -> dict[str, Any]:
    samples = []
    for index in range(repeats):
        started = time.monotonic()
        result = engine.infer(dict(spec), timeout=timeout)
        elapsed = round((time.monotonic() - started) * 1000)
        samples.append(
            {
                "index": index,
                "wall_ms": elapsed,
                "reported_ms": result.get("duration_ms"),
                "fallback": bool(result.get("fallback")),
                "reason": result.get("reason", ""),
                "text_length": len(str(result.get("text", ""))),
            }
        )
    return {
        "kind": "repeated_inference",
        "repeats": repeats,
        "fallback_count": sum(1 for sample in samples if sample["fallback"]),
        "fallback_rate": sum(1 for sample in samples if sample["fallback"]) / repeats,
        "cold_wall_ms": samples[0]["wall_ms"],
        "warm_wall_ms_median": statistics.median(sample["wall_ms"] for sample in samples[1:])
        if repeats > 1
        else samples[0]["wall_ms"],
        "samples": samples,
        "quality_scope": "latency and fallback evidence only; no caption-quality inference",
    }


def qualify_four_corner_inference(
    engine: InferenceProcess,
    audio: Path,
    *,
    language: str,
    timeout: float,
    cue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cue = cue or default_cue()
    profile = default_profile(language)
    samples = []
    with tempfile.TemporaryDirectory() as root:
        excerpt_dir = Path(root)
        for name, axes in AXIS_CORNERS:
            instruction = caption_instruction(profile, axes, cue)
            spec = {
                "audio": str(audio),
                "excerpt": str(excerpt_dir / f"{name}.wav"),
                "start_ms": cue["start_ms"],
                "end_ms": cue["end_ms"],
                "instruction": instruction,
                "language": language,
                "fallback": cue["fallback"][language],
            }
            started = time.monotonic()
            result = engine.infer(spec, timeout=timeout)
            samples.append(
                {
                    "name": name,
                    "axes": axes,
                    "instruction_hash": digest({"instruction": instruction})[:16],
                    "wall_ms": round((time.monotonic() - started) * 1000),
                    "reported_ms": result.get("duration_ms"),
                    "fallback": bool(result.get("fallback")),
                    "reason": result.get("reason", ""),
                    "text": str(result.get("text", "")),
                }
            )
    by_name = {sample["name"]: sample for sample in samples}
    identical_output_pairs = {
        "context_low_high_at_low_texture": (
            by_name["low_texture_low_context"]["text"] == by_name["low_texture_high_context"]["text"]
        ),
        "context_low_high_at_high_texture": (
            by_name["high_texture_low_context"]["text"] == by_name["high_texture_high_context"]["text"]
        ),
        "texture_low_high_at_low_context": (
            by_name["low_texture_low_context"]["text"] == by_name["high_texture_low_context"]["text"]
        ),
        "texture_low_high_at_high_context": (
            by_name["low_texture_high_context"]["text"] == by_name["high_texture_high_context"]["text"]
        ),
    }
    return {
        "kind": "four_corner_inference",
        "fallback_count": sum(1 for sample in samples if sample["fallback"]),
        "fallback_rate": sum(1 for sample in samples if sample["fallback"]) / len(samples),
        "identical_output_pairs": identical_output_pairs,
        "samples": samples,
        "evidence_provenance": cue.get("evidence_provenance", "unknown"),
        "quality_scope": (
            "real or fallback model outputs for inspection; string equality flags possible control "
            "collapse, but string difference is not a semantic independence score"
        ),
    }


def qualify_store_cache(tmp_path: Path | None = None) -> dict[str, Any]:
    owned = tempfile.TemporaryDirectory() if tmp_path is None else None
    root = Path(owned.name) if owned is not None else tmp_path
    assert root is not None
    try:
        store = StudyStore(root / "study.sqlite3")
        state_hash = "qualification"
        first = store.create(state_hash, "en")
        second = store.create(state_hash, "en")
        first_state = _watch_state(store.state(first))
        second_state = _watch_state(store.state(second))
        content_key = digest({"instruction": "same", "axes": {"texture": 0.5, "context": 0.5}})
        spec = {"current_index": 0, "queue_timeout": 15, "fallback": "Authored fallback."}
        store.mutate(first, "watch-1", 0, "watch", lambda state: state.update(first_state))
        store.mutate(second, "watch-2", 0, "watch", lambda state: state.update(second_state))
        store.enqueue(first, store.state(first), 0, content_key, spec, limit=32)
        claimed = store.claim()
        assert claimed is not None
        store.finish(claimed, {"text": "Generated cached caption.", "fallback": False, "duration_ms": 7})
        cold_jobs = store.captions(first, store.state(first))
        store.enqueue(second, store.state(second), 0, content_key, spec, limit=32)
        warm_claim = store.claim()
        warm_jobs = store.captions(second, store.state(second))
        return {
            "kind": "store_cache",
            "passed": (
                cold_jobs[0]["state"] == "succeeded"
                and cold_jobs[0]["result"]["fallback"] is False
                and warm_claim is None
                and warm_jobs[0]["state"] == "succeeded"
                and warm_jobs[0]["result"]["text"] == "Generated cached caption."
            ),
            "cold_state": cold_jobs[0]["state"],
            "warm_claimed_worker_job": warm_claim is not None,
            "warm_state": warm_jobs[0]["state"],
            "quality_scope": "cache delivery contract only; model quality not measured",
        }
    finally:
        if owned is not None:
            owned.cleanup()


def manifest_axis_report(manifest_path: Path, media_root: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, media_root)
    pending = [video["id"] for video in manifest["viewing_videos"] if video["status"] == "pending"]
    ready = [video for video in manifest["viewing_videos"] if video["status"] == "ready"]
    profile = default_profile(manifest["language"])
    cue_reports = []
    for video in ready:
        for index, cue in enumerate(video["cues"]):
            cue_reports.append(
                {
                    "video_id": video["id"],
                    "cue": index,
                    "axis_contract": qualify_axis_contract(profile, cue)["passed"],
                    "duration_ms": cue["end_ms"] - cue["start_ms"],
                }
            )
    return {
        "kind": "manifest_axis_report",
        "language": manifest["language"],
        "ready_videos": len(ready),
        "pending_video_ids": pending,
        "cue_count": len(cue_reports),
        "all_ready_cues_pass_axis_contract": all(row["axis_contract"] for row in cue_reports),
        "cue_reports": cue_reports,
        "quality_scope": (
            "manifest/schema and prompt contract only; final-media quality requires human review"
        ),
    }


def reviewer_sheet_rows(manifest_path: Path | None, media_root: Path | None) -> list[dict[str, Any]]:
    columns = {
        "video_id": "",
        "cue": "",
        "status": "pending",
        "start_ms": "",
        "end_ms": "",
        "evidence_provenance": "",
        "evidence_text": "",
        "fallback_caption": "",
        "low_texture_low_context": "",
        "high_texture_low_context": "",
        "low_texture_high_context": "",
        "high_texture_high_context": "",
        "context_low_high_at_low_texture_identical": "",
        "context_low_high_at_high_texture_identical": "",
        "texture_low_high_at_low_context_identical": "",
        "texture_low_high_at_high_context_identical": "",
        "grounding_notes": "",
        "axis_independence_notes": "",
        "reviewer": "",
    }
    if manifest_path is None or media_root is None:
        return [columns.copy() for _ in range(3)]
    manifest = load_manifest(manifest_path, media_root)
    rows = []
    for video in manifest["viewing_videos"]:
        if video["status"] == "pending":
            row = columns.copy()
            row["video_id"] = video["id"]
            rows.append(row)
            continue
        for index, cue in enumerate(video["cues"]):
            row = columns.copy()
            row.update(
                {
                    "video_id": video["id"],
                    "cue": str(index),
                    "status": "ready",
                    "start_ms": str(cue["start_ms"]),
                    "end_ms": str(cue["end_ms"]),
                    "evidence_provenance": "manifest_authored_unscored",
                    "evidence_text": str(cue["evidence"]),
                    "fallback_caption": str(cue["fallback"][manifest["language"]]),
                }
            )
            rows.append(row)
    return rows


def write_reviewer_sheet(
    path: Path,
    manifest_path: Path | None = None,
    media_root: Path | None = None,
) -> None:
    rows = reviewer_sheet_rows(manifest_path, media_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproducibility_record(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    sources = {
        "harness": _file_hash(Path(__file__)),
        "study_schema": _file_hash(Path(study_schema.__file__)),
    }
    for name in ("audio", "manifest", "media", "backend_config", "contract", "checkpoint"):
        value = getattr(args, name, None)
        if not value:
            continue
        path = Path(value)
        if path.is_file():
            sources[name] = _file_hash(path)
        elif path.is_dir():
            sources[name] = "directory"
        else:
            sources[name] = "missing"
    return {
        "argv": list(argv),
        "cwd": str(Path.cwd()),
        "source_hashes": sources,
    }


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _watch_state(state: dict[str, Any]) -> dict[str, Any]:
    state.update(
        stage="watch",
        viewing=[
            {
                "id": "cache-video",
                "duration_ms": 1000,
                "media_hashes": {"audio": "unused", "video": "unused"},
                "cues": [{"start_ms": 0, "end_ms": 1000, "evidence": "e", "fallback": {"en": "Fallback."}}],
            }
        ],
        profile=default_profile(),
    )
    return state


def _descriptor_limit(instruction: str) -> int:
    marker = "Use up to "
    start = instruction.index(marker) + len(marker)
    end = instruction.index(" supported acoustic descriptors", start)
    return int(instruction[start:end])


def _context_policy(instruction: str) -> str:
    for phrase in (
        "Use broad source categories.",
        "Name supported specific sources.",
        "Name supported specific sources and their visible location or scene relationship.",
    ):
        if phrase in instruction:
            return phrase
    raise ValueError("No context policy found in instruction")


def _spec(audio: Path, excerpt_dir: Path, language: str, cue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audio": str(audio),
        "excerpt": str(excerpt_dir / "qualification-excerpt.wav"),
        "start_ms": cue["start_ms"],
        "end_ms": cue["end_ms"],
        "instruction": caption_instruction(default_profile(language), {"texture": 0.5, "context": 0.5}, cue),
        "language": language,
        "fallback": cue["fallback"][language],
    }


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--media", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--language", default="en", choices=("en", "ko"))
    parser.add_argument("--evidence")
    parser.add_argument("--fallback")
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--end-ms", type=int, default=5000)
    parser.add_argument("--backend-config")
    parser.add_argument("--contract")
    parser.add_argument("--checkpoint")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--four-corners", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reviewer-sheet", type=Path)
    args = parser.parse_args(effective_argv)

    report: dict[str, Any] = {"schema": "dpo.regen.caption-qualification/v1"}
    report["reproducibility"] = reproducibility_record(args, effective_argv)
    cue = cue_from_args(
        evidence=args.evidence,
        fallback=args.fallback,
        language=args.language,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
    )
    report["axis_contract"] = qualify_axis_contract(default_profile(args.language), cue)
    report["store_cache"] = qualify_store_cache()
    if args.manifest and args.media:
        report["manifest"] = manifest_axis_report(args.manifest, args.media)
    if args.audio:
        settings = {
            key: value
            for key, value in {
                "backend_config": args.backend_config,
                "contract": args.contract,
                "checkpoint": args.checkpoint,
            }.items()
            if value
        }
        engine = InferenceProcess(settings)
        with tempfile.TemporaryDirectory() as root:
            try:
                report["repeated_inference"] = qualify_repeated_inference(
                    engine,
                    _spec(args.audio, Path(root), args.language, cue),
                    timeout=args.timeout,
                    repeats=args.repeats,
                )
                if args.four_corners:
                    report["four_corner_inference"] = qualify_four_corner_inference(
                        engine,
                        args.audio,
                        language=args.language,
                        timeout=args.timeout,
                        cue=cue,
                    )
            finally:
                engine.close()
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.reviewer_sheet:
        write_reviewer_sheet(args.reviewer_sheet, args.manifest, args.media)
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
