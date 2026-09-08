"""Prepare a v2 manifest from staged calibration clips without inventing long videos."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, nargs="+", help="One or more existing regen documents")
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--output", type=Path, help="New manifest; existing files are never overwritten")
    parser.add_argument("--validate", type=Path, help="Validate a prepared or pending v2 manifest")
    parser.add_argument("--media", type=Path, help="Shared root of staged media for validation")
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
    else:
        parser.error("Use --legacy with --output, or --validate with --media")


if __name__ == "__main__":
    main()
