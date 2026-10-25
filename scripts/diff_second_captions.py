"""Draft every second caption phase 1 can show, for the researcher to review before the study.

    uv run python scripts/diff_second_captions.py variants --phase1-setup SETUP --out VARIANTS.json
    CUDA_VISIBLE_DEVICES=1 envs/diff-generator/.venv/bin/python scripts/diff_second_captions.py draft \\
        --variants VARIANTS.json --trida-dir artifacts/diff/models/trida-7b-c37755e8 \\
        [--candidate CANDIDATE.json …] --out DRAFTS.json --sheet REVIEW.csv
    uv run python scripts/diff_second_captions.py rescore --variants VARIANTS.json \\
        --drafts DRAFTS.json --sheet REVIEW.csv

A second caption's prompt depends on the participant only through the clip's mode (the binding
model's and the plan's) and the sound they attended to (A1s, a required choice); the scene-aware
setting is the clip's own (``diff.phase1-tweak/v5``). The set of prompts is therefore small and
finite, and the researcher reviews every caption before the study (2026-10-24) rather than the app
showing whatever Trida writes and the check lets through: on the bench, the best wording delivered
about half.

``variants`` lists that set from the setup with ``phase1_tweak.reachable_slots``, the function the
app's gate uses. Each slot set is one variant, keyed by ``phase1_tweak.variant_id`` (a hash of its
slots), by which an approved caption is looked up.

``draft`` writes every variant with Trida once per wording (the checkout's, or each candidate's in
the prompt bench's format), at the study's settings, and scores each draft with the server's check
and the bench's measures. It writes the drafts and a review sheet (UTF-8 with a byte-order mark, for
spreadsheet programs): one row per variant, with its slots, the drafts side by side with their
issues, and empty ``approved_text``, ``approved_from`` and ``note`` columns for the researcher.

``rescore`` scores the drafts again from their stored text after a change to the measures (e.g.
``phase1_tweak.named_in``, 2026-10-25), rewriting the drafts and the sheet in place without Trida.
It keeps whatever the researcher has filled in on the sheet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diff_prompt_bench import apply_candidate, manipulation, measure  # noqa: E402

from diff import rewrite_prompts  # noqa: E402
from diff.job_plan import PlanError  # noqa: E402
from diff.phase1_tweak import (  # noqa: E402
    TWEAK_POLICY,
    check_record,
    reachable_slots,
    tweak_job_for,
    variant_id,
)

VARIANTS_SCHEMA = "diff.second-caption-variants/v1"
DRAFTS_SCHEMA = "diff.second-caption-drafts/v1"
PURPOSE = "smoke"  # drafts for review, not participant data
MODES = ("scene", "acousmatic")
WORDING_NAMES = ("LIVE_CELL_PROMPTS", "TWEAK_PROMPTS", "TUNER_SYSTEM")
# Hints for the reviewer, not gates (the bench's wording round, 2026-10-24): a verb cut at a comma
# ("퍼,", "흐,"), a stray syllable, a doubled word, and a sound said to come from the sky, the plants,
# the ground or the pavement.
DAMAGE = re.compile(
    r"(?:퍼|흐|채)[,.](?!\.)|(?<!버)스[,.](?!\.)|(?:^|\s)(?:들|울|공|적|체)[,.\s]|(?<!경)적음"
    r"|에서에서|의에서|([가-힣]{2,}(?: 소리가)?)\s\1(?![가-힣])"
)
ODD_SOURCE = re.compile(r"(하늘|나무|풀|땅|인도)(에서|에서부터)\s*(들려|울려|울린|들린|흐른)")
FILLED = ("approved_text", "approved_from", "note")  # the sheet's columns for the researcher


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variants(args: argparse.Namespace) -> int:
    setup = json.loads(args.phase1_setup.read_text(encoding="utf-8"))
    listed = [
        {
            "variant_id": variant_id(slots),
            "clip": clip["key"],
            "mode": slots["mode"],
            "attended": next(
                o["value"] for o in clip["options"]["A1s"] if o["label"] == slots["attended_sound"]
            ),
            "slots": slots,
        }
        for clip in setup["clips"]
        for slots in reachable_slots(clip)
    ]
    document = {
        "schema": VARIANTS_SCHEMA,
        "phase1_setup": str(args.phase1_setup),
        "phase1_setup_sha256": sha256_file(args.phase1_setup),
        "tweak_policy": TWEAK_POLICY,
        "clips": {
            clip["key"]: {name: clip[name] for name in ("key", "caption", "options", "scene")}
            for clip in setup["clips"]
        },
        "variants": sorted(listed, key=lambda v: (v["clip"], v["mode"], v["attended"])),
    }
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {mode: sum(v["mode"] == mode for v in listed) for mode in MODES}
    print(f"{args.out}: {len(listed)} variants over {len(setup['clips'])} clips ({counts})")
    return 0


def issues(draft: Mapping[str, Any], mode: str) -> list[str]:
    """What a reviewer should look at in one draft; rough where the bench's measures are."""
    shown = draft["shown"]
    found = [f"check failed: {code}" for code in draft["failures"]]
    if draft["list_form"] or not shown:
        found.append("no sentence")
    if draft["near_copy"]:
        found.append("near-copy of the first caption")
    found += [f"wrong sound from the first caption: {word}" for word in draft["wrong_sounds_kept"]]
    found += [f"names an R2 distractor: {name}" for name in draft["distractors_named"]]
    if draft["sounds_kept"] < draft["sounds"]:
        found.append(f"keeps {draft['sounds_kept']} of {draft['sounds']} sounds (rough)")
    if draft["broken"] or DAMAGE.search(shown):
        found.append("cut or doubled words? (rough)")
    if mode == "scene":
        found += [f"place the prompt did not give: {word}" for word in draft["invented_places"]]
        found += [f"sound from {m.group(1)}? (rough)" for m in ODD_SOURCE.finditer(shown)]
    else:
        found += [f"names a place or object: {word}" for word in draft["place_words"]]
    return found


def draft(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from diff import trida_runner
    from diff.model_profile import TRIDA_7B, verify_snapshot

    document = json.loads(args.variants.read_text(encoding="utf-8"))
    if document.get("schema") != VARIANTS_SCHEMA:
        raise SystemExit(f"{args.variants} is not a {VARIANTS_SCHEMA} file")
    clips = document["clips"]
    checkout = {name: getattr(rewrite_prompts, name) for name in WORDING_NAMES}
    wordings: list[tuple[str, dict[str, Any]]] = [
        (path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in args.candidate
    ] or [("checkout", {})]

    print("verifying the Trida snapshot (hashing every weight shard) …", flush=True)
    verification = verify_snapshot(args.trida_dir, TRIDA_7B, weights=True)
    drafts: list[dict[str, Any]] = []
    templates: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="hf-modules-") as modules:
        os.environ["HF_MODULES_CACHE"] = modules
        loaded = trida_runner.load(args.trida_dir, verification, device=args.device)
        trida_runner.warm_up(loaded)
        for name, candidate in wordings:
            for attribute, value in checkout.items():
                setattr(rewrite_prompts, attribute, value)
            apply_candidate(candidate)
            templates[name] = {
                "candidate": candidate.get("name"),
                "sha256": {mode: rewrite_prompts.TWEAK_PROMPTS[mode].sha256 for mode in MODES},
            }
            for variant in document["variants"]:
                clip = clips[variant["clip"]]
                source, job, _ = tweak_job_for(clip, variant["slots"], PURPOSE)
                try:
                    result = trida_runner.run_job(loaded, trida_runner.freeze_live(loaded, job))
                except PlanError as exc:
                    result = {"status": "refused", "error": str(exc)}
                text = (result.get("output") or {}).get("text")
                checks = check_record(source, job, result) if result.get("output") else None
                measured = measure(text, source.text, checks)
                scored = {**measured, **manipulation(measured["shown"], clip, variant["slots"])}
                drafts.append(
                    {
                        "variant_id": variant["variant_id"],
                        "wording": name,
                        "job_id": job.job_id,
                        "status": result["status"],
                        "error": result.get("error"),
                        "text": text,
                        **scored,
                        "issues": issues(scored, variant["mode"]),
                    }
                )
            print(f"{name}: {len(document['variants'])} drafts", flush=True)
    report = {
        "schema": DRAFTS_SCHEMA,
        "variants": {"path": str(args.variants), "sha256": sha256_file(args.variants)},
        "phase1_setup_sha256": document["phase1_setup_sha256"],
        "model": {"profile": TRIDA_7B.profile_id, "device": args.device, "purpose": PURPOSE},
        "wordings": templates,
        "drafts": drafts,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_sheet(args.sheet, document, drafts, [name for name, _ in wordings])
    clean = sum(not d["issues"] for d in drafts)
    print(f"{args.out}: {len(drafts)} drafts, {clean} with no issue; sheet {args.sheet}")
    return 0


def rescore(args: argparse.Namespace) -> int:
    document = json.loads(args.variants.read_text(encoding="utf-8"))
    report = json.loads(args.drafts.read_text(encoding="utf-8"))
    if document.get("schema") != VARIANTS_SCHEMA or report.get("schema") != DRAFTS_SCHEMA:
        raise SystemExit(f"{args.variants} and {args.drafts} must be {VARIANTS_SCHEMA} and {DRAFTS_SCHEMA}")
    if report["variants"]["sha256"] != sha256_file(args.variants):
        raise SystemExit(f"{args.drafts} was drafted from another {args.variants.name}")
    variants = {v["variant_id"]: v for v in document["variants"]}
    before = sum(not d["issues"] for d in report["drafts"])
    for index, written in enumerate(report["drafts"]):
        variant = variants[written["variant_id"]]
        clip = document["clips"][variant["clip"]]
        scored = {**written, **manipulation(written["shown"], clip, variant["slots"])}
        report["drafts"][index] = {**scored, "issues": issues(scored, variant["mode"])}
    with args.sheet.open(encoding="utf-8-sig", newline="") as handle:
        kept = {row["variant_id"]: {c: row[c] for c in FILLED} for row in csv.DictReader(handle)}
    args.drafts.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_sheet(args.sheet, document, report["drafts"], list(report["wordings"]), kept)
    after = sum(not d["issues"] for d in report["drafts"])
    filled = sum(any(row.values()) for row in kept.values())
    print(
        f"{args.drafts}: {len(report['drafts'])} drafts, {after} with no issue (was {before}); "
        f"sheet {args.sheet}, {filled} filled rows kept"
    )
    return 0


def write_sheet(
    path: Path,
    document: Mapping[str, Any],
    drafts: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    kept: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    by_key = {(d["variant_id"], d["wording"]): d for d in drafts}
    columns = [
        "variant_id",
        "clip",
        "mode",
        "attended_sound",
        "seen",
        "scene_elements",
        "visible_sounds",
        "offscreen",
        "space",
        "sounds",
        "first_caption",
        *(column for name in names for column in (f"draft:{name}", f"issues:{name}")),
        "approved_text",
        "approved_from",
        "note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for variant in document["variants"]:
            slots = variant["slots"]
            row = {
                "variant_id": variant["variant_id"],
                "clip": variant["clip"],
                "mode": variant["mode"],
                "attended_sound": slots["attended_sound"],
                "seen": slots.get("seen", ""),
                "scene_elements": slots.get("scene_elements", ""),
                "visible_sounds": slots.get("visible_sounds", ""),
                "offscreen": slots.get("offscreen", "").strip(),
                "space": slots.get("space", "").strip(),
                "sounds": slots["sounds"],
                "first_caption": slots["caption"],
                **{column: (kept or {}).get(variant["variant_id"], {}).get(column, "") for column in FILLED},
            }
            for name in names:
                written = by_key[(variant["variant_id"], name)]
                row[f"draft:{name}"] = written["shown"]
                row[f"issues:{name}"] = "; ".join(written["issues"])
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listed = commands.add_parser("variants", help="[repository env] every prompt a participant can reach")
    listed.add_argument("--phase1-setup", type=Path, required=True)
    listed.add_argument("--out", type=Path, required=True)
    listed.set_defaults(handler=variants)
    drafted = commands.add_parser("draft", help="[generator env, GPU] write every variant with Trida")
    drafted.add_argument("--variants", type=Path, required=True)
    drafted.add_argument("--trida-dir", type=Path, required=True)
    drafted.add_argument(
        "--candidate",
        type=Path,
        action="append",
        default=[],
        help="a wording in the prompt bench's format, repeatable (default: the checkout's)",
    )
    drafted.add_argument("--device", default="cuda:0", help="within CUDA_VISIBLE_DEVICES")
    drafted.add_argument("--out", type=Path, required=True)
    drafted.add_argument("--sheet", type=Path, required=True)
    drafted.set_defaults(handler=draft)
    rescored = commands.add_parser("rescore", help="[repository env] score the drafts again, keep the sheet")
    rescored.add_argument("--variants", type=Path, required=True)
    rescored.add_argument("--drafts", type=Path, required=True)
    rescored.add_argument("--sheet", type=Path, required=True)
    rescored.set_defaults(handler=rescore)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
