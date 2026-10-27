"""Score wordings of the live, second-caption and tuner prompts with the writer, on fixed answers.

    uv run python scripts/diff_prompt_bench.py inputs --store STORE --log-dir DIR [--log-dir DIR …] \\
        --phase1-setup SETUP --sources SOURCES --note TEXT --out INPUTS.json
    ANTHROPIC_API_KEY=… uv run python scripts/diff_prompt_bench.py run \\
        --inputs INPUTS.json [--candidate CANDIDATE.json] --out REPORT.json

rewrite_prompts records how its wordings were chosen: draft candidates, run them on the same
material at the study's settings, and pick by a rule written down before the run. This runs that
comparison for live-p2, tweak-p1 and the tuner.

``inputs`` keeps what those prompts are built from, out of a run's store and logs:
- per session, each clip's first-step answers (P2–P5) and P10 ratings, and its follow/counter plan;
- each clip's caption, options and scene part, and the phase-2 cues;
- every cue and slider step a live call was written for;
- each clip's second-caption slots in both modes, from the checkout's binding model. It reads the
  click masks, which the generator env cannot (no Pillow), so this is decided here.
``run`` rebuilds every prompt from those answers and slots with the checkout's own code. A
candidate that changes the wording therefore runs on the same participants.

``run`` writes with the study's writer (Claude through the API since 2026-10-27, ``diff.claude_writer``;
Trida on a GPU before), for each session:
- every live (cue, texture, context) under the session's tuner, and the same prompts under the
  plain system message, which separates the prompt's effect from the tuner's;
- each clip's second caption in the mode the binding model and the plan chose, and the same
  answers in the other mode.
Calls take the server's own path (``ClaudeWriter.attempt``) at the study's settings, with the
server's one retry after a busy or unreachable API. Each distinct prompt is called once. Claude's
output varies, so a rerun gives other outputs (Trida, at temperature 0, gave the same).

It reports:
- how often an output copies its starting caption;
- whether a live output has as many texture words as its step asks (0, 1 or 2);
- check failures and flags, from the server's own checks on the shown text (the display rule);
- how often the writer's own text has 「」 or a leading prompt label, which the display rule removes;
- visibly broken text in what is shown: "소" cut from "소리", doubled commas or full stops. The
  count is rough (it misses typos such as "웅거림" and would count the animal 소), so a fluent
  reader still rates the chosen wording;
- unusable live captions (nothing readable left, a check failure or a repetition), which show
  their starting caption instead (counted as a copy), and blank second captions, which fail;
- how many second captions differ between the scene-aware and the acousmatic mode, and per mode
  (the manipulation, before any participant): how long they are, how many name a place or an
  object on screen outside the sounds' own names (:data:`PLACE_WORDS`; an acousmatic caption
  should name none), how many scene-aware ones name one the prompt did not give, and how many
  of the prompt's sounds they keep (``phase1_tweak.named_in``, rough); and both modes' captions
  for the clips whose first caption contradicts their audio, for a reader;
- the delivered-text gate the hypothesis team asked for (2026-10-24), per mode: near-copies of
  the first caption (similarity at least :data:`NEAR_COPY`), a wrong sound of the first caption
  that survives (:data:`FIRST_CAPTION_ONLY`), an R2 distractor named (which would make R2's
  intended key wrong), a questionnaire item's wording echoed (:data:`ITEM_STEMS`), and a list of
  sounds with no sentence (no sentence ends in "다").

A candidate is a JSON file that replaces any of the checkout's wordings for this run only:
    {"name": "…",
     "cells": {"0-broad": "…{cue_text}…", "0-specific": "…", …, "2-relation": "…"},
     "tweak": {"scene": "…{caption}…{scene_elements}…", "acousmatic": "…{caption}…{space}…"},
     "tuner": {"text": "…{heard_families}…{s1}…"}}
Without one, the checkout's prompts run: the baseline.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diff import rewrite_prompts  # noqa: E402
from diff.caption_sources import CaptionSource, parse_sources  # noqa: E402
from diff.job_plan import Job, live_job  # noqa: E402
from diff.live_calibration import calibrate  # noqa: E402
from diff.live_generation import usable  # noqa: E402
from diff.output_checks import (  # noqa: E402
    LABEL_PREFIXES,
    check_output,
    displayed,
    output_from_record,
    quality_keys_in,
    readable,
    shown_text,
)
from diff.phase1_binding import COUNTER, FOLLOW, assignment_plan  # noqa: E402
from diff.phase1_tweak import named_in, tweak_job_for, tweak_source, tweak_steer_all  # noqa: E402
from diff.phase1_usable import BROKEN, ITEM_STEMS, NEAR_COPY, PLACE_WORDS, check_record  # noqa: E402
from diff.study_protocol import LIVE_MAX_CHARS, SYSTEM_MESSAGE  # noqa: E402

INPUTS_SCHEMA = "diff.prompt-bench-inputs/v2"  # v2: second-caption slots decided in `inputs`
REPORT_SCHEMA = "diff.prompt-bench-report/v1"
PURPOSE = "smoke"
KEPT_PAGES = ("P2", "P3", "P4", "P5", "P10")
STAGE_A = ("P2", "P3", "P4", "P5")
# The clips whose prepared first caption contradicts their audio (plan, 2026-10-18), and
# singapore_067, which the hypothesis team flagged (2026-10-24).
CONTRADICTING = ("amsterdam_012", "singapore_040", "amsterdam_057", "bangkok_034", "singapore_067")
# Words of a first caption for a sound its orig_text does not name, read 2026-10-24 (bangkok_034's
# caption names no wrong sound; it leaves the horn out).
FIRST_CAPTION_ONLY = {
    "amsterdam_012": ("기차",),
    "amsterdam_057": ("경적", "대화"),
    "singapore_040": ("오토바이",),
    "singapore_067": ("경적", "체증"),
}


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
                kept.append({"key": key, "index": index, "records": pages})
                used.add(key)
        live: dict[tuple[str, int, int, str], None] = {}
        for log_dir in args.log_dir:
            for path in sorted(log_dir.rglob(f"live-{session_id}.jsonl")):
                for call in read_jsonl(path):
                    if call.get("kind") != "tweak":
                        spec = call["spec"]
                        live[(call["video_id"], call["cue"], spec["texture"], spec["context"])] = None
        plan = state.get("assignment", {}).get("plan") or assignment_plan(
            int(state["order_seed"]), len(state["clip_order"])
        )
        complete = [entry for entry in kept if all(page in entry["records"] for page in STAGE_A)]
        tweaks = []
        if complete:
            chosen = [clips[entry["key"]] for entry in complete]
            answered = [entry["records"] for entry in complete]
            planned = [plan[entry["index"]] for entry in complete]
            flipped = [COUNTER if assignment == FOLLOW else FOLLOW for assignment in planned]
            ways = (
                tweak_steer_all(chosen, answered, planned, root=REPO_ROOT),
                tweak_steer_all(chosen, answered, flipped, root=REPO_ROOT),
            )
            for entry, (slots, model), (other, _) in zip(complete, *ways, strict=True):
                tweaks.append(
                    {"key": entry["key"], "planned": slots, "other": other, "p_bound": model["p_bound"]}
                )
        sessions.append(
            {
                "participant": participant,
                "plan": plan,
                "clips": kept,
                "tweaks": tweaks,
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
            key: {name: clips[key][name] for name in ("key", "caption", "options", "scene")}
            for key in sorted(used)
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
    if "cells" in candidate:  # a slider cell the candidate leaves out keeps the checkout's prompt
        cells = dict(rewrite_prompts.LIVE_CELL_PROMPTS)
        for key, text in candidate["cells"].items():
            texture, context = key.split("-")
            cells[(int(texture), context)] = replace(cells[(int(texture), context)], text=text)
        replacements["LIVE_CELL_PROMPTS"] = cells
    if "tweak" in candidate:  # it replaces the ladder's first wording; a mode left out keeps the checkout's
        first = rewrite_prompts.TWEAK_LADDER[0]
        modes = dict(rewrite_prompts.TWEAK_WORDINGS[first])
        for mode, text in candidate["tweak"].items():
            modes[mode] = replace(modes[mode], text=text)
        replacements["TWEAK_WORDINGS"] = {**rewrite_prompts.TWEAK_WORDINGS, first: modes}
        replacements["TWEAK_PROMPTS"] = modes
    if "text" in candidate.get("tuner", {}):
        replacements["TUNER_SYSTEM"] = replace(rewrite_prompts.TUNER_SYSTEM, text=candidate["tuner"]["text"])
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


def measure(
    text: str | None, source: str, checks: Mapping[str, Any] | None, *, falls_back: bool = False
) -> dict[str, Any]:
    """Copies and texture words in what is shown; quotes and labels in what the writer wrote.

    ``falls_back``: a live caption, which shows its starting caption when it is not usable (nothing
    readable left, a check failure or a repetition; ``live_generation.usable``). The bench has no
    earlier settings' caption to show instead.
    """
    written = text or ""
    shown = shown_text(written)
    blank = not readable(shown)
    unusable = falls_back and not usable({"output": {"text": written}, "checks": checks or {}})
    if unusable:
        shown = source
    return {
        "shown": shown,
        "blank": blank,
        "unusable": unusable,
        "copy": shown == source,
        "texture_words": quality_keys_in(shown),
        "quotes": "「" in written or "」" in written,
        "label": written.startswith(LABEL_PREFIXES),
        "broken": [match.group(0) for match in BROKEN.finditer(shown)],
        "failures": [f["code"] for f in (checks or {}).get("failures", [])],
        "flags": [f["code"] for f in (checks or {}).get("flags", [])],
    }


def manipulation(shown: str, clip: Mapping[str, Any], slots: Mapping[str, str]) -> dict[str, Any]:
    """What a second caption does with the screen: place words outside the sounds' names, those the
    prompt did not give, and the prompt's sounds it keeps (roughly)."""
    names = [o["label"] for o in clip["options"]["A1s"]] + [o["label"] for o in clip["options"]["R2"]]
    places = [w for w in PLACE_WORDS if w in shown and not any(w in name for name in names)]
    given = " ".join((slots.get("scene_elements", ""), slots.get("seen", ""), clip["caption"]["text"]))
    sounds = [o["label"] for o in clip["options"]["R2"] if o.get("mentioned")]
    distractors = [o["label"] for o in clip["options"]["R2"] if not o.get("mentioned")]
    first = re.sub(r"\s+", "", clip["caption"]["text"])
    return {
        "chars": len(shown),
        "place_words": places,
        "invented_places": [w for w in places if w not in given],
        "sounds_kept": sum(named_in(shown, name) for name in sounds),
        "sounds": len(sounds),
        "near_copy": difflib.SequenceMatcher(None, re.sub(r"\s+", "", shown), first).ratio() >= NEAR_COPY,
        "wrong_sounds_kept": [w for w in FIRST_CAPTION_ONLY.get(clip["key"], ()) if w in shown],
        "distractors_named": [name for name in distractors if named_in(shown, name)],
        "item_stems": [stem for stem in ITEM_STEMS if stem in shown],
        "list_form": bool(shown) and not re.search(r"다[.!?]?(\s|$)", shown),
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
            "labels": sum(r["label"] for r in live),
            "broken": sum(bool(r["broken"]) for r in live),
            "blank": sum(r["blank"] for r in live),
            "unusable": sum(r["unusable"] for r in live),
            "by_texture": {str(k): dict(v) for k, v in sorted(by_texture.items())},
            "by_context": {k: dict(v) for k, v in sorted(by_context.items())},
            "failures": dict(Counter(code for r in live for code in r["failures"])),
            "flags": dict(Counter(code for r in live for code in r["flags"])),
        }
    every = [r for r in rows if r["kind"] == "tweak"]
    tweaks = [r for r in every if r["planned"]]
    summary["tweak"] = {
        "calls": len(tweaks),
        "copies": sum(r["copy"] for r in tweaks),
        "failed": sum(r["status"] != "completed" or bool(r["failures"]) for r in tweaks),
        "quotes": sum(r["quotes"] for r in tweaks),
        "labels": sum(r["label"] for r in tweaks),
        "broken": sum(bool(r["broken"]) for r in tweaks),
        "blank": sum(r["blank"] for r in tweaks),
        "failures": dict(Counter(code for r in tweaks for code in r["failures"])),
        "flags": dict(Counter(code for r in tweaks for code in r["flags"])),
    }
    pairs = [r for r in rows if r["kind"] == "tweak-modes"]
    by_mode = {}
    for mode in ("scene", "acousmatic"):
        written = [r for r in every if r["mode"] == mode and not r["blank"]]
        by_mode[mode] = {
            "captions": len(written),
            "mean_chars": round(sum(r["chars"] for r in written) / len(written), 1) if written else None,
            "with_place_words": sum(bool(r["place_words"]) for r in written),
            "with_invented_places": sum(bool(r["invented_places"]) for r in written),
            "sounds_kept": sum(r["sounds_kept"] for r in written),
            "sounds": sum(r["sounds"] for r in written),
            "near_copies": sum(r["near_copy"] for r in written),
            "wrong_sounds_kept": sum(bool(r["wrong_sounds_kept"]) for r in written),
            "distractors_named": sum(bool(r["distractors_named"]) for r in written),
            "item_stems": sum(bool(r["item_stems"]) for r in written),
            "list_form": sum(r["list_form"] for r in written),
            "place_words": dict(Counter(w for r in written for w in r["place_words"])),
        }
    summary["tweak_modes"] = {
        "pairs": len(pairs),
        "differ": sum(r["differ"] for r in pairs),
        "by_mode": by_mode,
        "contradicting": [
            {
                "participant": r["participant"],
                "clip": r["clip"],
                "scene": r["scene"],
                "acousmatic": r["acousmatic"],
            }
            for r in pairs
            if r["clip"] in CONTRADICTING
        ],
    }
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    for system in ("tuner", "plain"):
        live = summary[f"live_{system}"]
        print(
            f"live under {system}: {live['copies']}/{live['calls']} copies, "
            f"{live['not_completed']} not completed; the writer wrote 「」 in {live['quotes']}, "
            f"a label in {live['labels']}; {live['broken']} visibly broken; "
            f"{live['unusable']} unusable ({live['blank']} blank), shown as the starting caption"
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
        f"the writer wrote 「」 in {tweak['quotes']}, a label in {tweak['labels']}; "
        f"{tweak['broken']} visibly broken; {tweak['blank']} blank, failed"
        f"\n  failures {tweak['failures']}  flags {tweak['flags']}"
    )
    pairs = summary["tweak_modes"]
    print(f"  scene-aware vs acousmatic: {pairs['differ']}/{pairs['pairs']} outputs differ")
    for mode, counts in pairs["by_mode"].items():
        print(
            f"  {mode}: {counts['captions']} captions, {counts['mean_chars']} chars on average; "
            f"{counts['with_place_words']} name a place or object on screen "
            f"({counts['with_invented_places']} one the prompt did not give); "
            f"{counts['sounds_kept']}/{counts['sounds']} prompt sounds kept (rough)"
        )
        print(f"    place words: {counts['place_words']}")
        print(
            f"    gate: {counts['near_copies']} near-copies of the first caption, "
            f"{counts['wrong_sounds_kept']} keep a wrong first-caption sound, "
            f"{counts['distractors_named']} name an R2 distractor, {counts['item_stems']} echo an item, "
            f"{counts['list_form']} are lists"
        )
    shown = set()
    for pair in pairs["contradicting"]:
        if pair["clip"] in shown:  # one participant's pair per clip is enough to read
            continue
        shown.add(pair["clip"])
        print(f"  {pair['clip']} ({pair['participant']}):")
        print(f"    scene: {pair['scene']}\n    acousmatic: {pair['acousmatic']}")


def run(args: argparse.Namespace) -> int:
    from diff.claude_writer import UNAVAILABLE, ClaudeWriter
    from diff.live_generation import LIVE_DEADLINE_S
    from diff.model_profile import CLAUDE_WRITER

    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    if document.get("schema") != INPUTS_SCHEMA:
        raise SystemExit(f"{args.inputs} is not a {INPUTS_SCHEMA} file")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8")) if args.candidate else {}
    apply_candidate(candidate)
    clips = document["clips"]
    cues = {(s.unit_key, s.cue_index): s for s in parse_sources(document["sources"]).sources}

    writer = ClaudeWriter()
    print(f"{CLAUDE_WRITER.model}: {writer.warm_up()}", flush=True)
    results: dict[str, dict[str, Any]] = {}

    def call(job: Job) -> dict[str, Any]:
        if job.job_id not in results:
            result = writer.attempt(job.record(), LIVE_DEADLINE_S)
            if result["status"] == UNAVAILABLE:  # the server's one identical retry
                result = writer.attempt(job.record(), LIVE_DEADLINE_S)
            results[job.job_id] = result
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
                *((f"tweak-{mode}", t) for mode, t in rewrite_prompts.TWEAK_PROMPTS.items()),
                ("tuner", rewrite_prompts.TUNER_SYSTEM),
            )
        },
        "live_cells": {
            f"{texture}-{context}": {"id": t.template_id, "version": t.version, "sha256": t.sha256}
            for (texture, context), t in rewrite_prompts.LIVE_CELL_PROMPTS.items()
        },
        "inputs": {
            "path": str(args.inputs),
            "sha256": sha256_file(args.inputs),
            "note": document.get("note"),
        },
        "model": {"profile": CLAUDE_WRITER.profile_id},
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
                        displayed(output_from_record(result["output"])),
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
                        **measure(text, source.text, checks, falls_back=True),
                    }
                )
        for entry in session["tweaks"]:
            clip = clips[entry["key"]]
            source = tweak_source(clip)
            texts: dict[str, str] = {}
            for is_planned, steer in ((True, entry["planned"]), (False, entry["other"])):
                _, job, record = tweak_job_for(clip, steer, PURPOSE)
                result = call(job)
                text = (result.get("output") or {}).get("text")
                checks = check_record(source, record, result) if result.get("output") else None
                measured = measure(text, source.text, checks)
                texts[steer["mode"]] = measured["shown"]
                rows.append(
                    {
                        "kind": "tweak",
                        "participant": who,
                        "clip": entry["key"],
                        "mode": steer["mode"],
                        "planned": is_planned,
                        "slots": steer,
                        "p_bound": entry["p_bound"],
                        "status": result["status"],
                        "error": result.get("error"),
                        "source": source.text,
                        "text": text,
                        **measured,
                        **manipulation(measured["shown"], clip, steer),
                    }
                )
            rows.append(
                {
                    "kind": "tweak-modes",
                    "participant": who,
                    "clip": entry["key"],
                    **texts,
                    "differ": texts["scene"] != texts["acousmatic"],
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
    ran = commands.add_parser("run", help="[repository env, API key] write every prompt and score it")
    ran.add_argument("--inputs", type=Path, required=True)
    ran.add_argument(
        "--candidate", type=Path, help="a JSON file of replacement wordings (default: the checkout's)"
    )
    ran.add_argument("--out", type=Path, required=True)
    ran.set_defaults(handler=run)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
