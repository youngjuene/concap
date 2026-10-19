"""Score wordings of the live prompt, the second-caption prompt and the tuner with Trida, on fixed answers.

    uv run python scripts/diff_prompt_bench.py inputs --store STORE --log-dir DIR [--log-dir DIR …] \\
        --phase1-setup SETUP --sources SOURCES --note TEXT --out INPUTS.json
    CUDA_VISIBLE_DEVICES=1 envs/diff-generator/.venv/bin/python scripts/diff_prompt_bench.py run \\
        --inputs INPUTS.json --trida-dir artifacts/diff/models/trida-7b-c37755e8 \\
        [--candidate CANDIDATE.json] --out REPORT.json

rewrite_prompts records how its wordings were chosen: draft candidates, run them on the same
material at the study's settings, and pick by a rule written down before the run. This runs that
comparison for live-p2, tweak-p1 and the tuner.

``inputs`` keeps what those prompts are built from, out of a run's store and logs:
- per session, each clip's first-step answers (P2–P5) and P10 ratings;
- each clip's caption and options, and the phase-2 cues;
- every cue and slider step a live call was written for.
``run`` rebuilds every prompt from those answers with the checkout's own code. A candidate that
changes the wording, or the checkout's slots, therefore runs on the same participants.

``run`` loads Trida on one GPU and writes, for each session:
- every live (cue, texture, context) under the session's tuner, and the same prompts under the
  plain system message, which separates the prompt's effect from the tuner's;
- each clip's second caption, and the same with all four degrees at 0% and at 100%.
Calls take the worker's own path (freeze_live, run_job) at the study's settings. Trida is
deterministic at temperature 0, so a rerun gives the same outputs.

It reports:
- how often an output copies its starting caption;
- whether a live output has as many texture words as its step asks (0, 1 or 2);
- check failures and flags, from the server's own checks, and 「」 left in the text;
- how many second captions change between 0% and 100% degrees.

A candidate is a JSON file that replaces any of the checkout's wordings for this run only:
    {"name": "…",
     "live": {"text": "…{cue_text}…{texture_rule} {context_rule}…",
              "texture_rules": {"0": "…", "1": "…", "2": "…"},
              "context_rules": {"broad": "…", "specific": "…", "relation": "…"}},
     "tweak": {"text": "…{caption}…{sounds}…"},
     "tuner": {"text": "…{heard_families}…{s1}…"}}
Without one, the checkout's prompts run: the baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diff import rewrite_prompts  # noqa: E402
from diff.caption_sources import CaptionSource, parse_sources  # noqa: E402
from diff.job_plan import Job, PlanError, live_job  # noqa: E402
from diff.live_calibration import calibrate  # noqa: E402
from diff.output_checks import check_output, output_from_record, quality_keys_in  # noqa: E402
from diff.phase1_tweak import DIRECTIONS, check_record, tweak_job_for, tweak_slots, tweak_source  # noqa: E402
from diff.study_protocol import LIVE_MAX_CHARS, SYSTEM_MESSAGE  # noqa: E402

INPUTS_SCHEMA = "diff.prompt-bench-inputs/v1"
REPORT_SCHEMA = "diff.prompt-bench-report/v1"
PURPOSE = "smoke"
KEPT_PAGES = ("P2", "P3", "P4", "P5", "P10")
DEGREE_ENDS = ("0", "100")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inputs(args: argparse.Namespace) -> int:
    setup = json.loads(args.phase1_setup.read_text(encoding="utf-8"))
    clips = {clip["key"]: clip for clip in setup["clips"]}
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    db = sqlite3.connect(args.store.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT session_id, participant, state FROM sessions ORDER BY created_at"
        ).fetchall()
    finally:
        db.close()
    sessions = []
    used: set[str] = set()
    for session_id, participant, raw in rows:
        state = json.loads(raw)
        records: dict[tuple[int, str], dict[str, Any]] = {}
        for record in state["responses"]:
            if not record.get("analysis_excluded") and record["page"] in KEPT_PAGES:
                records[(int(record["clip_index"]), record["page"])] = {
                    name: record.get(name) for name in ("answers", "matches", "labels")
                }
        kept = []
        for index, key in enumerate(state["clip_order"]):
            pages = {page: records[(index, page)] for page in KEPT_PAGES if (index, page) in records}
            if pages:
                kept.append({"key": key, "records": pages})
                used.add(key)
        live: dict[tuple[str, int, int, str], None] = {}
        for log_dir in args.log_dir:
            for path in sorted(log_dir.rglob(f"live-{session_id}.jsonl")):
                for call in read_jsonl(path):
                    if call.get("kind") != "tweak":
                        spec = call["spec"]
                        live[(call["video_id"], call["cue"], spec["texture"], spec["context"])] = None
        sessions.append(
            {
                "participant": participant,
                "clips": kept,
                "live": [
                    {"video_id": video, "cue": cue, "texture": texture, "context": context}
                    for video, cue, texture, context in live
                ],
            }
        )
    document = {
        "schema": INPUTS_SCHEMA,
        "note": args.note,
        "made_from": {
            "store": str(args.store),
            "phase1_setup_sha256": sha256_file(args.phase1_setup),
            "sources_sha256": sha256_file(args.sources),
        },
        "clips": {
            key: {name: clips[key][name] for name in ("key", "caption", "options")} for key in sorted(used)
        },
        "sources": {**sources, "sources": [s for s in sources["sources"] if s["phase"] == 2]},
        "sessions": sessions,
    }
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    calls = sum(len(s["live"]) for s in sessions)
    print(f"{args.out}: {len(sessions)} sessions, {len(used)} clips, {calls} live (cue, step) pairs")
    return 0


def apply_candidate(candidate: Mapping[str, Any]) -> None:
    """Replace the checkout's wordings for this run; the render functions read them at call time."""
    replacements: dict[str, Any] = {}
    live = candidate.get("live", {})
    if "text" in live:
        replacements["LIVE_REWRITE"] = replace(rewrite_prompts.LIVE_REWRITE, text=live["text"])
    if "texture_rules" in live:
        replacements["LIVE_TEXTURE_RULES"] = {int(step): rule for step, rule in live["texture_rules"].items()}
    if "context_rules" in live:
        replacements["LIVE_CONTEXT_RULES"] = dict(live["context_rules"])
    for key, name in (("tweak", "TWEAK_P1"), ("tuner", "TUNER_SYSTEM")):
        if "text" in candidate.get(key, {}):
            replacements[name] = replace(getattr(rewrite_prompts, name), text=candidate[key]["text"])
    for name, value in replacements.items():
        setattr(rewrite_prompts, name, value)


def profile_inputs(
    clips: Mapping[str, Any], session: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The calibration's inputs, as diff.phase1_flow builds them: P3's family and P2's labels, P10."""
    observations: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []
    for entry in session["clips"]:
        records = entry["records"]
        if "P2" in records and "P3" in records:
            families = {o["value"]: o["family"] for o in clips[entry["key"]]["options"]["A1s"]}
            heard = families.get(records["P3"]["answers"]["A1s"])
            observations.append({"heard": {heard: True} if heard else {}, "labels": records["P2"]["labels"]})
        if "P10" in records:
            ratings.append({item: records["P10"]["answers"][item] for item in ("S1", "S2", "S3")})
    return observations, ratings


def measure(text: str | None, source: str, checks: Mapping[str, Any] | None) -> dict[str, Any]:
    text = text or ""
    return {
        "copy": text == source,
        "texture_words": quality_keys_in(text),
        "quotes": "「" in text or "」" in text,
        "failures": [f["code"] for f in (checks or {}).get("failures", [])],
        "flags": [f["code"] for f in (checks or {}).get("flags", [])],
    }


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for system in ("tuner", "plain"):
        live = [r for r in rows if r["kind"] == "live" and r["system"] == system]
        by_texture: dict[int, Counter[str]] = defaultdict(Counter)
        by_context: dict[str, Counter[str]] = defaultdict(Counter)
        for row in live:
            words = len(row["texture_words"])
            texture, context = row["spec"]["texture"], row["spec"]["context"]
            by_texture[texture].update(
                calls=1, copies=row["copy"], as_asked=words == texture, any_texture_words=words > 0
            )
            by_context[context].update(calls=1, copies=row["copy"])
        summary[f"live_{system}"] = {
            "calls": len(live),
            "copies": sum(r["copy"] for r in live),
            "not_completed": sum(r["status"] != "completed" for r in live),
            "quotes": sum(r["quotes"] for r in live),
            "by_texture": {str(k): dict(v) for k, v in sorted(by_texture.items())},
            "by_context": {k: dict(v) for k, v in sorted(by_context.items())},
            "failures": dict(Counter(code for r in live for code in r["failures"])),
            "flags": dict(Counter(code for r in live for code in r["flags"])),
        }
    tweaks = [r for r in rows if r["kind"] == "tweak"]
    summary["tweak"] = {
        "calls": len(tweaks),
        "copies": sum(r["copy"] for r in tweaks),
        "failed": sum(r["status"] != "completed" or bool(r["failures"]) for r in tweaks),
        "quotes": sum(r["quotes"] for r in tweaks),
        "failures": dict(Counter(code for r in tweaks for code in r["failures"])),
        "flags": dict(Counter(code for r in tweaks for code in r["flags"])),
    }
    ends = [r for r in rows if r["kind"] == "tweak-degrees"]
    summary["tweak_degrees"] = {"pairs": len(ends), "differ": sum(r["differ"] for r in ends)}
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    for system in ("tuner", "plain"):
        live = summary[f"live_{system}"]
        print(
            f"live under {system}: {live['copies']}/{live['calls']} copies, "
            f"{live['not_completed']} not completed, {live['quotes']} with 「」"
        )
        for step, counts in live["by_texture"].items():
            print(
                f"  texture {step}: {counts.get('copies', 0)}/{counts['calls']} copies, "
                f"{counts.get('as_asked', 0)}/{counts['calls']} with {step} texture word(s)"
            )
        for step, counts in live["by_context"].items():
            print(f"  context {step}: {counts.get('copies', 0)}/{counts['calls']} copies")
        print(f"  failures {live['failures']}  flags {live['flags']}")
    tweak = summary["tweak"]
    print(
        f"second caption: {tweak['copies']}/{tweak['calls']} copies, {tweak['failed']} failed, "
        f"{tweak['quotes']} with 「」\n  failures {tweak['failures']}  flags {tweak['flags']}"
    )
    ends = summary["tweak_degrees"]
    print(f"  degrees 0% vs 100%: {ends['differ']}/{ends['pairs']} outputs differ")


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from diff import trida_runner
    from diff.model_profile import TRIDA_7B, verify_snapshot

    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    if document.get("schema") != INPUTS_SCHEMA:
        raise SystemExit(f"{args.inputs} is not a {INPUTS_SCHEMA} file")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8")) if args.candidate else {}
    apply_candidate(candidate)
    clips = document["clips"]
    cues = {(s.unit_key, s.cue_index): s for s in parse_sources(document["sources"]).sources}

    print("verifying the Trida snapshot (hashing every weight shard) …", flush=True)
    verification = verify_snapshot(args.trida_dir, TRIDA_7B, weights=True)
    with tempfile.TemporaryDirectory(prefix="hf-modules-") as modules:
        os.environ["HF_MODULES_CACHE"] = modules
        loaded = trida_runner.load(args.trida_dir, verification, device=args.device)
        trida_runner.warm_up(loaded)
        results: dict[str, dict[str, Any]] = {}

        def call(job: Job) -> dict[str, Any]:
            if job.job_id not in results:
                try:
                    results[job.job_id] = trida_runner.run_job(loaded, trida_runner.freeze_live(loaded, job))
                except PlanError as exc:  # a prompt the guards refuse is a result of the wording
                    results[job.job_id] = {"status": "refused", "error": str(exc)}
            return results[job.job_id]

        rows = bench(document["sessions"], clips, cues, call)
    summary = summarise(rows)
    print_summary(summary)
    report = {
        "schema": REPORT_SCHEMA,
        "candidate": candidate or None,
        "templates": {
            name: {"id": template.template_id, "version": template.version, "sha256": template.sha256}
            for name, template in (
                ("live", rewrite_prompts.LIVE_REWRITE),
                ("tweak", rewrite_prompts.TWEAK_P1),
                ("tuner", rewrite_prompts.TUNER_SYSTEM),
            )
        },
        "live_rules": {
            "texture": rewrite_prompts.LIVE_TEXTURE_RULES,
            "context": rewrite_prompts.LIVE_CONTEXT_RULES,
        },
        "inputs": {
            "path": str(args.inputs),
            "sha256": sha256_file(args.inputs),
            "note": document.get("note"),
        },
        "model": {"profile": TRIDA_7B.profile_id, "device": args.device},
        "distinct_calls": len(results),
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"report: {args.out}")
    return 0


def bench(
    sessions: Sequence[Mapping[str, Any]],
    clips: Mapping[str, Any],
    cues: Mapping[tuple[str, int], CaptionSource],
    call: Callable[[Job], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        who = session["participant"]
        tuner = calibrate(*profile_inputs(clips, session)).system_message
        for live in session["live"]:
            source = cues[(live["video_id"], live["cue"])]
            spec = {"texture": live["texture"], "context": live["context"]}
            for system, message in (("tuner", tuner), ("plain", SYSTEM_MESSAGE)):
                job = live_job(source, live["texture"], live["context"], message, PURPOSE)
                result = call(job)
                text = (result.get("output") or {}).get("text")
                checks = (
                    check_output(
                        source,
                        output_from_record(result["output"]),
                        template_phase=2,
                        cell=job.cell,
                        max_chars=LIVE_MAX_CHARS,
                    ).record()
                    if result.get("output")
                    else None
                )
                rows.append(
                    {
                        "kind": "live",
                        "participant": who,
                        "video_id": live["video_id"],
                        "cue": live["cue"],
                        "spec": spec,
                        "system": system,
                        "status": result["status"],
                        "error": result.get("error"),
                        "source": source.text,
                        "text": text,
                        **measure(text, source.text, checks),
                    }
                )
        for entry in session["clips"]:
            clip, records = clips[entry["key"]], entry["records"]
            if not all(page in records for page in ("P2", "P3", "P4", "P5")):
                continue
            slots = tweak_slots(clip, records)
            source = tweak_source(clip)
            texts: dict[str, str | None] = {}
            for kind, steer in (
                ("tweak", slots),
                *((f"degrees-{end}", {**slots, **dict.fromkeys(DIRECTIONS, end)}) for end in DEGREE_ENDS),
            ):
                _, job, _ = tweak_job_for(clip, steer, PURPOSE)
                result = call(job)
                texts[kind] = (result.get("output") or {}).get("text")
                if kind == "tweak":
                    checks = check_record(source, job, result) if result.get("output") else None
                    rows.append(
                        {
                            "kind": "tweak",
                            "participant": who,
                            "clip": entry["key"],
                            "slots": slots,
                            "status": result["status"],
                            "error": result.get("error"),
                            "source": source.text,
                            "text": texts[kind],
                            **measure(texts[kind], source.text, checks),
                        }
                    )
            rows.append(
                {
                    "kind": "tweak-degrees",
                    "participant": who,
                    "clip": entry["key"],
                    **{f"at_{end}": texts[f"degrees-{end}"] for end in DEGREE_ENDS},
                    "differ": texts["degrees-0"] != texts["degrees-100"],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    kept = commands.add_parser("inputs", help="[repository env] keep a run's answers as bench inputs")
    kept.add_argument("--store", type=Path, required=True)
    kept.add_argument("--log-dir", type=Path, action="append", required=True, help="repeatable")
    kept.add_argument("--phase1-setup", type=Path, required=True)
    kept.add_argument("--sources", type=Path, required=True, help="the phase-2 cues the run used")
    kept.add_argument("--note", required=True, help="what the run was, for whoever reads the report")
    kept.add_argument("--out", type=Path, required=True)
    kept.set_defaults(handler=inputs)
    ran = commands.add_parser("run", help="[generator env, GPU] write every prompt with Trida and score it")
    ran.add_argument("--inputs", type=Path, required=True)
    ran.add_argument("--trida-dir", type=Path, required=True)
    ran.add_argument(
        "--candidate", type=Path, help="a JSON file of replacement wordings (default: the checkout's)"
    )
    ran.add_argument("--device", default="cuda:0", help="within CUDA_VISIBLE_DEVICES")
    ran.add_argument("--out", type=Path, required=True)
    ran.set_defaults(handler=run)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
