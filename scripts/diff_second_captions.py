"""Draft every second caption phase 1 can show, down the ladder as the live server does.

    uv run python scripts/diff_second_captions.py variants --phase1-setup SETUP --out VARIANTS.json
    ANTHROPIC_API_KEY=… uv run python scripts/diff_second_captions.py draft \\
        --variants VARIANTS.json [--candidate CANDIDATE.json …] --out DRAFTS.json --sheet SHEET.csv
    uv run python scripts/diff_second_captions.py rescore --variants VARIANTS.json \\
        --drafts DRAFTS.json --sheet SHEET.csv

A second caption's prompt depends on the participant only through the clip's mode (the binding
model's and the plan's), the sound they attended to (A1s, a required choice) and the restorative
emphasis their P4 ratings reflect (one of three, since ``diff.phase1-tweak/v7``). The set of prompts
is therefore small and finite. Since 2026-10-26 captions are written live and shown once they pass the
gate (``diff.phase1_usable``); from 2026-10-24 to 26 the researcher reviewed every one before the
study instead, on the sheet this script wrote.

``variants`` lists that set from the setup with ``phase1_tweak.reachable_slots``. Each slot set is one
variant, keyed by ``phase1_tweak.variant_id`` (a hash of its slots).

``draft`` writes every variant with the second-caption writer (Claude through the API since
2026-10-27, ``diff.claude_writer``; Qwen3.5-9B on 2026-10-26) once per wording (the checkout's ladder,
or each candidate in the prompt bench's format in place of its first wording), at the study's
settings, down ``rewrite_prompts.TWEAK_LADDER`` until the gate passes one, as the server would: the
delivery rate, rung by rung, what the gate refused, and each call's latency and tokens. ``--jobs``
variants run at once (each one's ladder in order), since the API, unlike a GPU, serves several. Each
delivered (or last) caption is also scored with the bench's rougher measures. It writes the drafts and a
sheet for reading (UTF-8 with a byte-order mark, for spreadsheet programs): one row per variant, with
its slots and the drafts side by side with the rung that delivered them and their issues.

``rescore`` runs the gate and the measures again from the stored outputs after a change to either,
rewriting the drafts and the sheet in place without the model; a variant whose ladder stopped early is
marked when the new gate would have gone further. It keeps the sheet's notes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diff_prompt_bench import apply_candidate, manipulation, measure  # noqa: E402

from diff import rewrite_prompts  # noqa: E402
from diff.output_checks import readable, shown_text  # noqa: E402
from diff.phase1_tweak import (  # noqa: E402
    EMPHASIS_KO,
    TWEAK_POLICY,
    reachable_slots,
    tweak_job_for,
    variant_id,
)
from diff.phase1_usable import DAMAGE, check_record  # noqa: E402
from diff.rewrite_prompts import TWEAK_LADDER  # noqa: E402

VARIANTS_SCHEMA = "diff.second-caption-variants/v2"  # v2: one variant per emphasis too
DRAFTS_SCHEMA = "diff.second-caption-drafts/v2"  # v2: each variant down the ladder, the gate's verdicts
PURPOSE = "smoke"  # drafts for review, not participant data
MODES = ("scene", "acousmatic")
WORDING_NAMES = ("LIVE_CELL_PROMPTS", "TWEAK_PROMPTS", "TWEAK_WORDINGS", "TUNER_SYSTEM")
ODD_SOURCE = re.compile(r"(하늘|나무|풀|땅|인도)(에서|에서부터)\s*(들려|울려|울린|들린|흐른)")
FILLED = ("note",)  # the sheet's column for the reader
EMPHASIS_OF = {sentence: name for name, sentence in EMPHASIS_KO.items()}


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
            "emphasis": EMPHASIS_OF[slots["emphasis"]],
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
        "variants": sorted(listed, key=lambda v: (v["clip"], v["mode"], v["attended"], v["emphasis"])),
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
    from concurrent.futures import ThreadPoolExecutor

    from diff.claude_writer import ClaudeWriter
    from diff.live_generation import LIVE_DEADLINE_S
    from diff.model_profile import CLAUDE_WRITER

    document = json.loads(args.variants.read_text(encoding="utf-8"))
    if document.get("schema") != VARIANTS_SCHEMA:
        raise SystemExit(f"{args.variants} is not a {VARIANTS_SCHEMA} file")
    clips = document["clips"]
    checkout = {name: getattr(rewrite_prompts, name) for name in WORDING_NAMES}
    wordings: list[tuple[str, dict[str, Any]]] = [
        (path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in args.candidate
    ] or [("checkout", {})]

    writer = ClaudeWriter()
    print(f"{CLAUDE_WRITER.model}: {writer.warm_up()}", flush=True)

    def ladder(variant: Mapping[str, Any], name: str) -> dict[str, Any]:
        clip = clips[variant["clip"]]
        attempts = []
        for rung in range(len(TWEAK_LADDER)):
            source, job, steer = tweak_job_for(clip, variant["slots"], PURPOSE, rung)
            result = writer.attempt(job.record(), LIVE_DEADLINE_S)
            if result["status"] == "api_unavailable":  # the server's one identical retry
                result = writer.attempt(job.record(), LIVE_DEADLINE_S)
            attempts.append(
                {
                    "rung": rung,
                    "wording": TWEAK_LADDER[rung],
                    "job_id": job.job_id,
                    "status": result["status"],
                    "error": result.get("error"),
                    "output": result.get("output"),
                    "latency_ms": result.get("latency_ms"),
                    "api": result.get("api"),
                }
            )
            if judged(source, steer, attempts[-1])["delivered"]:
                break
        return scored_draft(variant, clip, name, attempts)

    drafts: list[dict[str, Any]] = []
    templates: dict[str, Any] = {}
    for name, candidate in wordings:
        for attribute, value in checkout.items():
            setattr(rewrite_prompts, attribute, value)
        apply_candidate(candidate)
        templates[name] = {
            "candidate": candidate.get("name"),
            "sha256": {mode: rewrite_prompts.TWEAK_PROMPTS[mode].sha256 for mode in MODES},
        }
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            drafts += pool.map(ladder, document["variants"], [name] * len(document["variants"]))
        print(f"{name}: {len(document['variants'])} drafts", flush=True)
    report = {
        "schema": DRAFTS_SCHEMA,
        "variants": {"path": str(args.variants), "sha256": sha256_file(args.variants)},
        "phase1_setup_sha256": document["phase1_setup_sha256"],
        "model": {"profile": CLAUDE_WRITER.profile_id, "purpose": PURPOSE},
        "wordings": templates,
        "drafts": drafts,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_sheet(args.sheet, document, drafts, [name for name, _ in wordings])
    print(f"{args.out}: {delivery(drafts)}; {calls(drafts)}; sheet {args.sheet}")
    return 0


def calls(drafts: Sequence[Mapping[str, Any]]) -> str:
    """The API calls behind the drafts: how many, their median and slowest latency, tokens, fallbacks."""
    made = [a for d in drafts for a in d["attempts"] if a.get("api")]
    latencies = sorted(a["latency_ms"] for a in made if a.get("latency_ms") is not None)
    if not latencies:
        return "no API calls"
    tokens = {side: sum(int(a["api"].get(f"{side}_tokens", 0)) for a in made) for side in ("input", "output")}
    fallbacks = sum(bool(a["api"].get("fallback")) for a in made)
    return (
        f"{len(made)} calls, median {latencies[len(latencies) // 2]} ms, slowest {latencies[-1]} ms, "
        f"{tokens['input']} input and {tokens['output']} output tokens, {fallbacks} fallbacks"
    )


def judged(source: Any, steer: Mapping[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    """One attempt with the gate's verdict, as the server reaches it (``live_generation.tweak_outcome``)."""
    output = attempt.get("output")
    checks = check_record(source, steer, {"output": output}) if output else None
    shown = shown_text(output["text"]) if output else ""
    delivered = attempt["status"] == "completed" and checks is not None and not checks["excluded"]
    attempt.update(
        shown=shown,
        delivered=delivered and readable(shown),
        failures=[f["code"] for f in (checks or {}).get("failures", [])],
        flags=[f["code"] for f in (checks or {}).get("flags", [])],
        checks=checks,
    )
    return attempt


def scored_draft(
    variant: Mapping[str, Any], clip: Mapping[str, Any], wording: str, attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    """A variant's ladder judged by the current gate, and its delivered (or last) caption scored."""
    for attempt in attempts:
        source, _, steer = tweak_job_for(clip, variant["slots"], PURPOSE, attempt["rung"])
        judged(source, steer, attempt)
    delivered = next((a for a in attempts if a["delivered"]), None)
    last = delivered or attempts[-1]
    stopped = delivered is not None and delivered is not attempts[-1]
    text = (last.get("output") or {}).get("text")
    measured = measure(text, clip["caption"]["text"], last["checks"])
    scored = {**measured, **manipulation(measured["shown"], clip, variant["slots"])}
    return {
        "variant_id": variant["variant_id"],
        "wording": wording,
        "attempts": attempts,
        "delivered_rung": delivered["rung"] if delivered else None,
        "ladder_cut_short": stopped,
        "job_id": last["job_id"],
        "status": last["status"],
        "error": last.get("error"),
        "text": text,
        **scored,
        "issues": issues(scored, variant["mode"]),
    }


def delivery(drafts: Sequence[Mapping[str, Any]]) -> str:
    """How many drafts the ladder delivered, at which rung, and how many of those have no issue."""
    rungs = [sum(d["delivered_rung"] == rung for d in drafts) for rung in range(len(TWEAK_LADDER))]
    delivered = sum(rungs)
    clean = sum(d["delivered_rung"] is not None and not d["issues"] for d in drafts)
    by_rung = ", ".join(
        f"{rung}:{wording} {count}"
        for rung, (wording, count) in enumerate(zip(TWEAK_LADDER, rungs, strict=True))
    )
    return f"{len(drafts)} drafts, {delivered} delivered ({by_rung}), {clean} of them with no issue"


def rescore(args: argparse.Namespace) -> int:
    document = json.loads(args.variants.read_text(encoding="utf-8"))
    report = json.loads(args.drafts.read_text(encoding="utf-8"))
    if document.get("schema") != VARIANTS_SCHEMA or report.get("schema") != DRAFTS_SCHEMA:
        raise SystemExit(f"{args.variants} and {args.drafts} must be {VARIANTS_SCHEMA} and {DRAFTS_SCHEMA}")
    if report["variants"]["sha256"] != sha256_file(args.variants):
        raise SystemExit(f"{args.drafts} was drafted from another {args.variants.name}")
    variants = {v["variant_id"]: v for v in document["variants"]}
    before = delivery(report["drafts"])
    for index, written in enumerate(report["drafts"]):
        variant = variants[written["variant_id"]]
        clip = document["clips"][variant["clip"]]
        report["drafts"][index] = scored_draft(variant, clip, written["wording"], written["attempts"])
    with args.sheet.open(encoding="utf-8-sig", newline="") as handle:
        kept = {row["variant_id"]: {c: row.get(c, "") for c in FILLED} for row in csv.DictReader(handle)}
    args.drafts.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_sheet(args.sheet, document, report["drafts"], list(report["wordings"]), kept)
    short = sum(d["ladder_cut_short"] for d in report["drafts"])
    print(f"{args.drafts}: {delivery(report['drafts'])} (was: {before})")
    if short:
        print(f"  {short} ladders stopped before a rung the current gate now passes first: draft again")
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
        "emphasis",
        "seen",
        "scene_elements",
        "visible_sounds",
        "offscreen",
        "sounds",
        "first_caption",
        *(
            column
            for name in names
            for column in (f"rung:{name}", f"draft:{name}", f"failures:{name}", f"issues:{name}")
        ),
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
                "emphasis": variant["emphasis"],
                "seen": slots.get("seen", ""),
                "scene_elements": slots.get("scene_elements", ""),
                "visible_sounds": slots.get("visible_sounds", ""),
                "offscreen": slots.get("offscreen", "").strip(),
                "sounds": slots["sounds"],
                "first_caption": slots["caption"],
                **{column: (kept or {}).get(variant["variant_id"], {}).get(column, "") for column in FILLED},
            }
            for name in names:
                written = by_key[(variant["variant_id"], name)]
                rung = written["delivered_rung"]
                row[f"rung:{name}"] = "none" if rung is None else f"{rung}:{TWEAK_LADDER[rung]}"
                row[f"draft:{name}"] = written["shown"]
                row[f"failures:{name}"] = " | ".join(
                    f"{a['rung']}:{a.get('wording', a.get('condition'))}: "
                    f"{', '.join(a['failures']) or a['status']}"
                    for a in written["attempts"]
                    if not a["delivered"]
                )
                row[f"issues:{name}"] = "; ".join(written["issues"])
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listed = commands.add_parser("variants", help="[repository env] every prompt a participant can reach")
    listed.add_argument("--phase1-setup", type=Path, required=True)
    listed.add_argument("--out", type=Path, required=True)
    listed.set_defaults(handler=variants)
    drafted = commands.add_parser("draft", help="[repository env, API key] every variant down the ladder")
    drafted.add_argument("--variants", type=Path, required=True)
    drafted.add_argument(
        "--candidate",
        type=Path,
        action="append",
        default=[],
        help="a wording in the prompt bench's format, repeatable (default: the checkout's)",
    )
    drafted.add_argument("--jobs", type=int, default=4, help="variants written at once")
    drafted.add_argument("--out", type=Path, required=True)
    drafted.add_argument("--sheet", type=Path, required=True)
    drafted.set_defaults(handler=draft)
    rescored = commands.add_parser(
        "rescore", help="[repository env] run the gate and measures again, keep the sheet's notes"
    )
    rescored.add_argument("--variants", type=Path, required=True)
    rescored.add_argument("--drafts", type=Path, required=True)
    rescored.add_argument("--sheet", type=Path, required=True)
    rescored.set_defaults(handler=rescore)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
