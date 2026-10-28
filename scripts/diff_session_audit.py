"""Audit diff sessions: recompute from each participant's answers what the writer should have received.

    uv run python scripts/diff_session_audit.py check --store STORE --log-dir DIR [--log-dir DIR …] \\
        --phase1-setup SETUP --sources SOURCES [--export EXPORT_DIR] [--out REPORT.json]
    envs/diff-generator/.venv/bin/python scripts/diff_session_audit.py tokens \\
        --trida-dir artifacts/diff/models/trida-7b-c37755e8 --log-dir DIR [--log-dir DIR …]

``check`` runs on every pilot and participant session before analysis. The answer side is
recomputed here rather than by diff: the object under each P2 point (re-reading the masks), the
binding model over the whole stage A (the point against the noticed sound's click mask, E1/E3
centred on the participant's mean, the tendency from the other clips' points, P(bound)), the
balanced plan from the order seed, each clip's mode, restorative emphasis (the subscale the clip
stands highest on, centred on the participant's own means) and slots, the V2 point against the
noticed source, the attended sound, the tuner's heard families and rating means. diff's own
prompt functions then render what those values should have produced, and every logged call is
compared with that, so new prompt wording needs no change here. A session that ran an older
template than the checkout is reported once, not call by call. It also checks that:
- each clip's second caption went down the ladder as the server must take it (2026-10-26): every
  logged attempt is the recomputed job at its rung, the request an API writer logged (2026-10-27) is
  the one that job sends, re-running the check and gate gives its logged
  verdict, each rung was left only after a refusal (or a second transient failure), and the
  caption shown at P6 is the one the gate passed, or a clip that never passed was skipped
  (sessions of 2026-10-24 to 26 showed the researcher's approved text: audit them at that commit);
- each attempt's shown text is the writer's text under the display rule (logs from 2026-10-20 on);
- every slider change got a live call, unless no cue boundary was left to write, and each live call
  an API writer logged (2026-10-27) sent the request its recomputed job sends;
- every caption shown is one the writer wrote for that cue or an earlier one (the kept caption);
- with ``--export``, that the export matches the store and the logs.

``tokens`` re-applies Trida's chat template to every logged Trida call (its system message and
prompt) and compares the token hash with the one the worker reported, which shows the logged prompt
is what reached the model. Run it in the generator environment, for sessions from before
2026-10-27. Since then Claude writes both phases' captions through the API and has no tokenizer
here; ``check`` compares each request it logged with the recomputed job's instead.

Both only read; the store is opened read-only. Each exits 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diff.caption_sources import CATEGORIES_KO, CaptionSource, parse_sources  # noqa: E402
from diff.claude_writer import request_body  # noqa: E402
from diff.export_bundle import caption_id  # noqa: E402
from diff.generation_worker import TRANSIENT  # noqa: E402
from diff.job_plan import live_job  # noqa: E402
from diff.live_generation import STARTING_CAPTION, usable  # noqa: E402
from diff.output_checks import DISPLAY_RULE, readable, shown_text  # noqa: E402
from diff.phase1_binding import (  # noqa: E402
    AREA_FLOOR,
    ASSIGNMENT_POLICY,
    BALANCED_POLICY,
    CONGRUENCE_ITEMS,
    CONGRUENCE_K,
    CONGRUENCE_SPAN,
    DRAWN,
    GRID_SIZE,
    THRESHOLD,
)
from diff.phase1_setup import read_setup  # noqa: E402
from diff.phase1_tweak import (  # noqa: E402
    ELEMENTS_KO,
    EMPHASIS_KO,
    OFFSCREEN_KO,
    RECORDING_ARTEFACTS,
    tweak_job_for,
    tweak_source,
)
from diff.phase1_usable import check_record  # noqa: E402
from diff.rewrite_prompts import (  # noqa: E402
    LIVE_CELL_PROMPTS,
    LIVE_MODE_PROMPTS,
    LIVE_RULES_VERSION,
    NOT_SEEN,
    TUNER_SYSTEM,
    TWEAK_LADDER,
    TWEAK_PROMPTS,
    render_tuner_system,
)
from diff.study_protocol import sha256_of  # noqa: E402

STEP1_PAGES = ("P2", "P3", "P4", "P5")
OTHER_MODE = {"scene": "acousmatic", "acousmatic": "scene"}
SECOND_SLOTS = ("attended_sound", "seen")
SECOND_MODEL = ("recommended", "assignment", "mode", "p_bound", "moderator_scene", "moderator_participant")
SECOND_FIELDS = (
    "status",
    "second_caption",
    "caption_source",
    "variant_id",
    "rung",
    "writer_model",
    "emphasis",
    *SECOND_SLOTS,
    *SECOND_MODEL,
)
# The emphasis's subscales (P4, Payne 2013), in the order that breaks a tie left after the raw means.
SUBSCALES = {
    "fascination": ("F1", "F2", "F3"),
    "being_away_from": ("BF1", "BF2", "BF3"),
    "extent_coherence": ("E1", "E2", "E3"),
}
RATINGS = ("S1", "S2", "S3")
MASK_THRESHOLD = 127
SUPERSEDED_MS = 1500  # a setting replaced this soon may never get a call: Trida takes about 1 s


class Audit:
    """Counts per check group and every mismatch, in the order found."""

    def __init__(self) -> None:
        self.counts: dict[str, Counter[str]] = {}
        self.problems: list[str] = []

    def ok(self, group: str, condition: bool, message: str) -> bool:
        self.counts.setdefault(group, Counter())["ok" if condition else "bad"] += 1
        if not condition:
            self.problems.append(f"[{group}] {message}")
        return condition

    def note(self, group: str, what: str) -> None:
        self.counts.setdefault(group, Counter())[what] += 1

    def print(self) -> None:
        for group, counts in self.counts.items():
            extra = "  ".join(f"{name} {n}" for name, n in counts.items() if name not in ("ok", "bad"))
            print(
                f"{group:16s} ok {counts['ok']:5d}  bad {counts['bad']}" + (f"  ({extra})" if extra else "")
            )
        print("\n".join(self.problems) or "no mismatches")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_store(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Every session and its store events, read-only so a study store in use is never touched."""
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT session_id, participant, state, revision, created_at, finished_at"
            " FROM sessions ORDER BY created_at"
        ).fetchall()
        sessions = [
            {
                "session_id": sid,
                "participant": who,
                "state": json.loads(state),
                "revision": revision,
                "created_at": created,
                "finished_at": finished,
            }
            for sid, who, state, revision, created, finished in rows
        ]
        events = {
            session["session_id"]: [
                {"at": at, **json.loads(body)}
                for at, body in db.execute(
                    "SELECT at, body FROM events WHERE session_id = ? ORDER BY id", (session["session_id"],)
                )
            ]
            for session in sessions
        }
    finally:
        db.close()
    return sessions, events


def session_logs(log_dirs: Sequence[Path], session_id: str) -> dict[str, list[dict[str, Any]]]:
    """The session's live events and attempts, read the way the export reads them."""
    events: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for log_dir in log_dirs:
        for path in sorted(log_dir.rglob(f"session-{session_id}.jsonl")):
            events += read_jsonl(path)
        for path in sorted(log_dir.rglob(f"live-{session_id}.jsonl")):
            attempts += read_jsonl(path)
    return {"events": events, "attempts": attempts}


def formal_records(state: Mapping[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    """The last formal (non-practice) response per clip and page."""
    records: dict[tuple[int, str], dict[str, Any]] = {}
    for record in state["responses"]:
        if not record.get("analysis_excluded"):
            records[(int(record["clip_index"]), str(record["page"]))] = record
    return records


def mask_at(path: Path, x: float, y: float) -> tuple[bool, int]:
    """Whether the mask holds the pixel nearest the normalised point, and the mask's area."""
    import numpy as np
    from PIL import Image

    inside = np.asarray(Image.open(path).convert("L")) > MASK_THRESHOLD
    height, width = inside.shape
    hit = bool(inside[min(height - 1, round(y * (height - 1))), min(width - 1, round(x * (width - 1)))])
    return hit, int(inside.sum())


def object_at(frame: Mapping[str, Any], x: float, y: float) -> str | None:
    """The annotated object under a point: the smallest mask containing it, as regen matches."""
    best: tuple[str, int] | None = None
    for obj in frame["objects"]:
        hit, area = mask_at(REPO_ROOT / obj["mask"], x, y)
        if hit and (best is None or area < best[1]):
            best = (str(obj["id"]), area)
    return best[0] if best else None


def expected_balanced(seed: int, count: int) -> list[str]:
    """``balanced-v1`` (sessions begun before 2026-10-28): half follow, shuffled at entry."""
    plan = ["follow"] * (count // 2) + ["counter"] * (count - count // 2)
    random.Random(f"{BALANCED_POLICY}:{seed}").shuffle(plan)
    return plan


def expected_stratified(seed: int, recommended: list[str]) -> list[str]:
    """``stratified-v2``: half follow within each recommendation; a seeded coin for odd strata."""
    rng = random.Random(f"{ASSIGNMENT_POLICY}:{seed}")
    strata = [[i for i, mode in enumerate(recommended) if mode == m] for m in ("scene", "acousmatic")]
    follows = [len(stratum) // 2 for stratum in strata]
    odd = [k for k, stratum in enumerate(strata) if len(stratum) % 2]
    for k in rng.sample(odd, len(recommended) // 2 - sum(follows)):
        follows[k] += 1
    plan = ["counter"] * len(recommended)
    for stratum, count in zip(strata, follows, strict=True):
        for i in rng.sample(stratum, count):
            plan[i] = "follow"
    return plan


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def bound_given(pi: float, visibility: float) -> float:
    return 1 / (1 + math.exp(-(logit(pi) + logit(visibility))))


def point_on_source(clip: Mapping[str, Any], noticed: str, point: Mapping[str, Any]) -> tuple[str, float]:
    """A point against the noticed sound's click mask on its frame: the status and its log ratio."""
    sound = clip["scene"]["sounds"][noticed]
    click = sound["click_frames"].get(str(int(point["frame_id"])), {})
    if not sound["visibility"]["on_screen"] or not click.get("mask") or not click.get("shown"):
        return "source not seen", 0.0
    area = max(float(click["area"]), AREA_FLOOR)
    hit, _ = mask_at(REPO_ROOT / click["mask"], float(point["x"]), float(point["y"]))
    return ("hit", math.log((DRAWN + (1 - DRAWN) * area) / area)) if hit else ("miss", math.log(1 - DRAWN))


def expected_bindings(
    clips: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The binding model over a whole stage A, recomputed from the answers and the masks on disk."""
    grid = [(index + 0.5) / GRID_SIZE for index in range(GRID_SIZE)]
    noticed = [str(given["A1s"]) for given in answers]
    visibility = [
        float(clip["scene"]["sounds"][value]["visibility"]["mean"])
        for clip, value in zip(clips, noticed, strict=True)
    ]
    seen = [
        point_on_source(clip, value, point) for clip, value, point in zip(clips, noticed, points, strict=True)
    ]
    fits = [statistics.mean(float(given[item]) for item in CONGRUENCE_ITEMS) for given in answers]
    centre = statistics.mean(fits)
    results = []
    for index in range(len(clips)):
        weights = [1.0] * GRID_SIZE
        for other in range(len(clips)):
            if other == index:
                continue
            ratio = math.exp(seen[other][1])
            weights = [
                w * (1 - bound_given(pi, visibility[other]) + bound_given(pi, visibility[other]) * ratio)
                for w, pi in zip(weights, grid, strict=True)
            ]
            weights = [w / sum(weights) for w in weights]
        log_ratio = seen[index][1] + CONGRUENCE_K * (fits[index] - centre) / CONGRUENCE_SPAN
        q = visibility[index]
        bound = sum(w * bound_given(pi, q) * math.exp(log_ratio) for w, pi in zip(weights, grid, strict=True))
        detached = sum(w * (1 - bound_given(pi, q)) for w, pi in zip(weights, grid, strict=True))
        p_bound = bound / (bound + detached)
        no_answers = statistics.mean(bound_given(pi, q) for pi in grid)  # flat tendency, no evidence
        results.append(
            {
                "visibility": q,
                "attention": seen[index][0],
                "p_bound": p_bound,
                "moderator_scene": logit(q),
                "moderator_participant": logit(p_bound) - logit(no_answers),
                "recommended": "scene" if p_bound >= THRESHOLD else "acousmatic",
            }
        )
    return results


def expected_emphases(answers: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each clip's reflected subscale: its mean minus the participant's own mean of it, highest wins."""
    means = [
        {name: statistics.mean(float(given[i]) for i in items) for name, items in SUBSCALES.items()}
        for given in answers
    ]
    centre = {name: statistics.mean(m[name] for m in means) for name in SUBSCALES}
    order = list(SUBSCALES)
    return [min(order, key=lambda n: (-round(m[n] - centre[n], 6), -m[n], order.index(n))) for m in means]


def expected_slots(
    clip: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    mode: str,
    emphasis: str,
) -> dict[str, str]:
    """The tweak slots for the mode, recomputed from the clip's first-step answers, its reflected
    subscale and its scene part."""
    answers: dict[str, Any] = {}
    for page in STEP1_PAGES:
        answers.update(records[page]["answers"])
    sounds = {option["value"]: option["label"] for option in clip["options"]["A1s"]}
    slots = {
        "mode": mode,
        "caption": clip["caption"]["text"],
        "sounds": ", ".join(option["label"] for option in clip["options"]["R2"] if option.get("mentioned")),
        "attended_sound": sounds.get(answers.get("A1s"), NOT_SEEN),
        "emphasis": EMPHASIS_KO[emphasis],
    }
    if mode == "acousmatic":
        return slots
    scene = clip["scene"]
    distinctive = scene["distinctive"]
    shares = {name: share for name, share in scene["elements"].items() if name != distinctive}
    second = max(shares, key=lambda name: shares[name])  # the first of equal shares, as diff sorts
    shown = {value: clip["scene"]["sounds"][value]["visibility"]["on_screen"] for value in sounds}
    visible = [sounds[value] for value in sounds if shown[value]]
    artefacts = {o["value"] for o in clip["options"]["A1s"] if o["audioset_label"] in RECORDING_ARTEFACTS}
    unseen = [sounds[value] for value in sounds if not shown[value] and value not in artefacts]
    return {
        **slots,
        "scene_elements": f"{ELEMENTS_KO[distinctive]}, {ELEMENTS_KO[second]}",
        "seen": ELEMENTS_KO[distinctive],
        "visible_sounds": ", ".join(dict.fromkeys(visible)) or NOT_SEEN,
        "offscreen": OFFSCREEN_KO.format(sounds=", ".join(dict.fromkeys(unseen))) if unseen else "",
    }


def plain(value: Any) -> Any:
    """A value as it reads back from a JSON log, so tuples compare equal to lists."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def differing(expected: Mapping[str, Any], logged: Mapping[str, Any]) -> list[str]:
    return sorted(key for key in set(expected) | set(logged) if expected.get(key) != logged.get(key))


def shown_on_screen(
    audit: Audit, where: str, attempt: Mapping[str, Any], starting: str | None = None
) -> str | None:
    """What the attempt put on screen: the display rule's text, or Trida's own before the rule existed.

    A live caption with nothing readable left shows the cue's starting caption (``shown_from``).
    """
    written = (attempt.get("output") or {}).get("text")
    if written is None or "display_rule" not in attempt:
        return written
    if not audit.ok(
        "display", attempt["display_rule"] == DISPLAY_RULE, f"{where}: {attempt['display_rule']}"
    ):
        return attempt.get("shown_text")
    expected = shown_text(written)
    if "usable" in attempt:  # live attempts from diff.live-attempt/v4 on
        audit.ok(
            "display",
            attempt["usable"] == usable(attempt),
            f"{where}: logged usable {attempt['usable']}, its text and check say {usable(attempt)}",
        )
    if "shown_from" in attempt:
        fell_back = not readable(expected)
        audit.ok(
            "display",
            attempt["shown_from"] == ("starting caption" if fell_back else "trida"),
            f"{where}: shown from {attempt['shown_from']}, Trida's text blank: {fell_back}",
        )
        if fell_back and starting is not None:
            expected = starting
    audit.ok(
        "display",
        attempt.get("shown_text") == expected,
        f"{where}: the logged shown text is not the display rule's",
    )
    return expected


def audit_phase1(
    audit: Audit,
    who: str,
    state: Mapping[str, Any],
    clips: Mapping[str, Any],
    tweaks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Slots, then the tweak calls down the ladder, their checks and the outcome, per clip; returns
    one report row per clip."""
    records = formal_records(state)
    rows: list[dict[str, Any]] = []
    ran: set[str] = set()
    seed = int(state["order_seed"])
    policy, stored = state["assignment"]["policy"], state["assignment"]["plan"]
    known = policy in (BALANCED_POLICY, ASSIGNMENT_POLICY)
    if not audit.ok("tweak-model", known, f"{who}: unknown plan {policy}"):
        return rows
    if policy == BALANCED_POLICY:
        plan = expected_balanced(seed, len(state["clip_order"]))
        audit.ok("tweak-model", stored == plan, f"{who}: the stored plan is not the seed's")
    order = list(state["clip_order"])
    steps = [
        {page: records[(index, page)] for page in STEP1_PAGES if (index, page) in records}
        for index in range(len(order))
    ]
    if any(len(step1) < len(STEP1_PAGES) for step1 in steps):  # the modes are decided as stage A closes
        audit.ok("tweak-slots", not state["second_captions"], f"{who}: second captions before stage A closed")
        if policy == ASSIGNMENT_POLICY:
            audit.ok("tweak-model", stored is None, f"{who}: a plan drawn before stage A closed")
        return rows
    answers_of = [
        {k: v for page in STEP1_PAGES for k, v in step1[page]["answers"].items()} for step1 in steps
    ]
    points = [step1["P2"]["answers"]["V1"][0] for step1 in steps]
    bindings = expected_bindings([clips[key] for key in order], answers_of, points)
    if policy == ASSIGNMENT_POLICY:
        plan = expected_stratified(seed, [binding["recommended"] for binding in bindings])
        audit.ok(
            "tweak-model",
            stored == plan,
            f"{who}: the stored plan is not the seed's for these recommendations",
        )
    emphases = expected_emphases(answers_of)
    for index, key in enumerate(order):
        clip, step1 = clips[key], steps[index]
        entry = state["second_captions"].get(key)
        if not audit.ok("tweak-slots", entry is not None, f"{who} {key}: no second-caption entry"):
            continue
        assert entry is not None
        point = points[index]
        frame = next(
            (f for i, f in enumerate(clip["frames"]) if int(f.get("index", i)) == int(point["frame_id"])),
            None,
        )
        matched = [m["object_id"] for m in step1["P2"].get("matches", []) if m.get("object_id")]
        under = None
        if audit.ok("tweak-slots", frame is not None, f"{who} {key}: P2 point on unknown frame {point}"):
            assert frame is not None
            under = object_at(frame, float(point["x"]), float(point["y"]))
            audit.ok(
                "tweak-slots",
                under == (matched[0] if matched else None),
                f"{who} {key}: the masks put {point} on {under}, the record matched {matched}",
            )
        binding = bindings[index]
        p7 = records.get((index, "P7"))
        if p7 is not None:
            status, _ = point_on_source(clip, str(answers_of[index]["A1s"]), p7["answers"]["V2"][0])
            logged = (p7.get("source_attention") or {}).get("status")
            audit.ok(
                "source-attention",
                logged == status,
                f"{who} {key}: V2 on the source {logged}, recomputed {status}",
            )
        mode = binding["recommended"] if plan[index] == "follow" else OTHER_MODE[binding["recommended"]]
        model = entry.get("model", {})
        audit.ok(
            "tweak-model",
            abs(float(model.get("p_bound", -1)) - binding["p_bound"]) <= 1e-4
            and all(
                abs(float(model.get(name, math.inf)) - binding[name]) <= 1e-4
                for name in ("moderator_scene", "moderator_participant")
            )
            and (
                model.get("attention", {}).get("status"),
                model.get("recommended"),
                model.get("assignment"),
                model.get("mode"),
            )
            == (binding["attention"], binding["recommended"], plan[index], mode),
            f"{who} {key}: stored model {model.get('p_bound')} {model.get('mode')} "
            f"({model.get('moderator_scene')}, {model.get('moderator_participant')}), recomputed "
            f"{binding['p_bound']:.4f} {mode} ({binding['attention']}; "
            f"{binding['moderator_scene']:.6f}, {binding['moderator_participant']:.6f})",
        )
        audit.ok(
            "tweak-model",
            model.get("emphasis", {}).get("chosen") == emphases[index],
            f"{who} {key}: stored emphasis {model.get('emphasis', {}).get('chosen')}, "
            f"recomputed {emphases[index]}",
        )
        expect = expected_slots(clip, step1, mode, emphases[index])
        audit.ok(
            "tweak-slots",
            entry["slots"] == expect,
            f"{who} {key}: stored slots differ in {differing(expect, entry['slots'])}",
        )
        attempts = [t for t in tweaks if t["clip"] == key]
        p6 = [v for v in state["viewings"] if v["clip_index"] == index and v["page"] == "P6"]
        if not audit.ok("tweak-call", bool(attempts), f"{who} {key}: no tweak call logged"):
            continue
        last = attempts[-1]
        for attempt in attempts:
            audit.ok(
                "tweak-call",
                attempt["steer"]["slots"] == entry["slots"],
                f"{who} {key}: logged slots != stored",
            )
        if entry["template"]["sha256"] != TWEAK_PROMPTS[expect["mode"]].sha256:
            ran.add(str(entry["template"]["sha256"]))
        else:
            text = None
            for at, attempt in enumerate(attempts):
                rung = int(attempt["steer"].get("rung", 0))
                where = f"{who} {key} rung {rung} attempt {attempt['attempt']}"
                _, job, steer = tweak_job_for(clip, expect, str(attempt["job"]["purpose"]), rung)
                expected_job = plain(job.record())
                audit.ok(
                    "tweak-call",
                    expected_job == attempt["job"],
                    f"{where}: the logged job differs in {differing(expected_job, attempt['job'])}",
                )
                audit.ok(
                    "tweak-call",
                    attempt["system_message"] == job.prompt.system_message,
                    f"{where}: tweak system message {attempt['system_message']!r}",
                )
                if job.runner == "claude":
                    logged = (attempt.get("api") or {}).get("request_sha256")
                    audit.ok(
                        "tweak-call",
                        logged == sha256_of(request_body(job)),
                        f"{where}: the logged request {str(logged)[:12]} is not the job's",
                    )
                text = shown_on_screen(audit, where, attempt)
                passed = False
                if attempt["status"] == "completed" and attempt.get("output"):
                    rerun = plain(check_record(tweak_source(clip), steer, attempt))
                    audit.ok(
                        "tweak-check",
                        rerun == attempt["checks"],
                        f"{where}: re-running the check and gate gives a different result than logged",
                    )
                    passed = not rerun["excluded"] and readable(text or "")
                expected_next = (
                    "ready"
                    if passed
                    else "retry"
                    if attempt["status"] in TRANSIENT and attempt["attempt"] == 1
                    else "next"
                    if attempt["status"] == "completed" and rung + 1 < len(TWEAK_LADDER)
                    else "failed"
                )
                audit.ok(
                    "tweak-ladder",
                    attempt["outcome"] == expected_next,
                    f"{where}: outcome {attempt['outcome']}, the ladder says {expected_next}",
                )
                if at + 1 < len(attempts):
                    following = int(attempts[at + 1]["steer"].get("rung", 0))
                    audit.ok(
                        "tweak-ladder",
                        expected_next in ("retry", "next")
                        and following == (rung if expected_next == "retry" else rung + 1),
                        f"{where}: then rung {following} after {expected_next}",
                    )
            audit.ok(
                "tweak-ladder",
                int(last["steer"].get("rung", 0)) == int(entry.get("rung", 0)),
                f"{who} {key}: the stored rung {entry.get('rung')} is not the last attempt's",
            )
            if entry["status"] == "ready":
                audit.ok(
                    "tweak-outcome",
                    last["outcome"] == "ready" and entry["text"] == text,
                    f"{who} {key}: stored caption != the one the gate passed",
                )
                served = (last.get("api") or {}).get("model")
                audit.ok(
                    "tweak-outcome",
                    (entry.get("attempts") or [{}])[-1].get("served_by") == served,
                    f"{who} {key}: the stored writer model is not {served}, the logged one",
                )
                audit.ok(
                    "tweak-outcome",
                    bool(p6) and p6[-1]["captions"][0]["text"] == text,
                    f"{who} {key}: the P6 viewing did not show Trida's caption",
                )
        text = shown_on_screen(audit, f"{who} {key}", last)
        if entry["status"] != "ready":
            audit.ok(
                "tweak-outcome",
                any(skip["clip"] == key for skip in state["skipped"]),
                f"{who} {key}: {entry['status']} but its second step was not skipped",
            )
        rows.append(
            {
                "participant": who,
                "clip": key,
                "status": entry["status"],
                "reason": entry.get("reason"),
                "slots": entry["slots"],
                "second": text,
                "attempts": len(attempts),
                "latency_ms": last.get("latency_ms"),
                "flags": [f["code"] for f in (last.get("checks") or {}).get("flags", [])],
                "failures": [f["code"] for f in (last.get("checks") or {}).get("failures", [])],
            }
        )
    if ran:
        audit.ok(
            "versions",
            False,
            f"{who}: the second captions ran tweak-p1 {sorted(sha[:12] for sha in ran)}, "
            f"this checkout renders {sorted(t.sha256[:12] for t in TWEAK_PROMPTS.values())}; "
            "audit at the commit the session ran",
        )
    return rows


def audit_tuner(audit: Audit, who: str, state: Mapping[str, Any], clips: Mapping[str, Any]) -> str | None:
    """The tuner's slots from P3 families and P10 means; returns the session's system message."""
    calibration = state.get("calibration")
    if not audit.ok("tuner", calibration is not None, f"{who}: no calibration (did not reach the handoff)"):
        return None
    assert calibration is not None
    records = formal_records(state)
    heard: Counter[str] = Counter()
    ratings: dict[str, list[float]] = {item: [] for item in RATINGS}
    for index, key in enumerate(state["clip_order"]):
        if (index, "P2") in records and (index, "P3") in records:
            families = {o["value"]: o["family"] for o in clips[key]["options"]["A1s"]}
            family = families.get(records[(index, "P3")]["answers"]["A1s"])
            if family:
                heard[family] += 1
        if (index, "P10") in records:
            for item in RATINGS:
                ratings[item].append(float(records[(index, "P10")]["answers"][item]))

    def shown(values: list[float]) -> str:
        return f"{round(statistics.mean(values), 2):g}" if values else NOT_SEEN

    named = [(CATEGORIES_KO[f], n) for f, n in sorted(heard.items()) if f in CATEGORIES_KO]
    slots = {
        "heard_families": ", ".join(f"{name} {n}번" for name, n in named) or NOT_SEEN,
        "s1": shown(ratings["S1"]),
        "s2": shown(ratings["S2"]),
        "s3": shown(ratings["S3"]),
    }
    record = calibration["record"]
    audit.ok(
        "tuner",
        record["slots"] == slots,
        f"{who}: tuner slots differ in {differing(slots, record['slots'])}",
    )
    if record["tuner"]["sha256"] != TUNER_SYSTEM.sha256:
        audit.ok(
            "versions",
            False,
            f"{who}: the tuner ran {record['tuner']['sha256'][:12]}, this checkout renders "
            f"{TUNER_SYSTEM.sha256[:12]}; audit at the commit the session ran",
        )
    else:
        audit.ok(
            "tuner",
            render_tuner_system(slots) == calibration["system_message"],
            f"{who}: the system message is not the tuner rendered from the recomputed slots",
        )
    return str(calibration["system_message"])


def expected_history(state: Mapping[str, Any]) -> list[list[float]]:
    """phase2_scene.scene_profile's history, re-derived: each stage-A clip's (visibility, attention
    log ratio) as its binding record holds them, in stage-A order."""
    records = sorted(state.get("second_captions", {}).values(), key=lambda r: int(r["clip_index"]))
    return [[float(r["model"]["visibility"]), float(r["model"]["attention"]["log_ratio"])] for r in records]


def expected_steer(scene: Mapping[str, Any], history: Sequence[Sequence[float]]) -> dict[str, Any]:
    """phase2_scene.cue_steer's mode and P(b), re-derived on its own grid: the tendency learnt from the
    history, then the first sound's visibility as the prior, with no evidence of the cue's own."""
    grid = [(i + 0.5) / GRID_SIZE for i in range(GRID_SIZE)]

    def bound(pi: float, q: float) -> float:
        odds = (pi / (1 - pi)) * (q / (1 - q))
        return odds / (1 + odds)

    weights = [1.0] * GRID_SIZE
    for visibility, log_ratio in history:
        weights = [
            w * ((1 - bound(pi, visibility)) + bound(pi, visibility) * math.exp(log_ratio))
            for w, pi in zip(weights, grid, strict=True)
        ]
        total = sum(weights)
        weights = [w / total for w in weights]
    total = sum(weights)
    q = float(scene["sounds"][0]["visibility"]["mean"])
    probability = sum(w / total * bound(pi, q) for w, pi in zip(weights, grid, strict=True))
    placed = [s for s in scene["sounds"] if s["audioset_label"] not in RECORDING_ARTEFACTS]
    visible = [s["sound"] for s in placed if s["visibility"]["on_screen"]]
    unseen = [s["sound"] for s in placed if not s["visibility"]["on_screen"]]
    mode = "scene" if probability >= THRESHOLD else "acousmatic"
    slots = {
        "seen": ELEMENTS_KO[scene["distinctive"]],
        "visible_sounds": ", ".join(dict.fromkeys(visible)) or NOT_SEEN,
        "unseen_sounds": ", ".join(dict.fromkeys(unseen)) or NOT_SEEN,
    }
    return {"mode": mode, "p_bound": round(probability, 4), "slots": slots if mode == "scene" else {}}


def audit_live(
    audit: Audit,
    who: str,
    tuner: str,
    lives: list[dict[str, Any]],
    events: list[dict[str, Any]],
    cues: Mapping[tuple[str, int], CaptionSource],
    scenes: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every live call against the slider steps, the tuner and (since 2026-10-28) the cue's mode for this
    participant; settings and exposures against the calls."""
    stale = False
    profile = ((state or {}).get("calibration", {}).get("record", {}) or {}).get("scene_profile")
    history = expected_history(state or {}) if profile is not None else []
    if profile is not None:
        audit.ok(
            "live-mode",
            profile["history"] == history,
            f"{who}: the scene profile's history is not the stage-A binding records'",
        )
    for call in lives:
        where = f"{who} {call['video_id']}#{call['cue']}"
        source = cues.get((call["video_id"], call["cue"]))
        if not audit.ok("live-call", source is not None, f"{where}: no such cue in --sources"):
            continue
        assert source is not None
        audit.ok(
            "live-call", call["system_message"] == tuner, f"{where}: the system message is not the tuner"
        )
        spec = call["spec"]
        logged_steer = (call["job"].get("cell") or {}).get("steer")
        mode = "acousmatic"
        if logged_steer is not None:
            scene = (scenes or {}).get("cues", {}).get(source.source_id)
            if not audit.ok(
                "live-mode", scene is not None, f"{where}: a steered call, no scene part (--phase2-scene)"
            ):
                continue
            assert scene is not None
            expected = expected_steer(scene, history)
            got = {key: logged_steer.get(key) for key in expected}
            audit.ok(
                "live-mode", got == expected, f"{where}: the cue's mode differs in {differing(expected, got)}"
            )
            mode = str(logged_steer["mode"])
        recomputed = live_job(
            source,
            int(spec["texture"]),
            str(spec["context"]),
            tuner,
            str(call["job"]["purpose"]),
            logged_steer,
        )
        job = plain(recomputed.record())
        prompt = call["job"]["prompt"]
        template = LIVE_MODE_PROMPTS[mode].get((int(spec["texture"]), str(spec["context"])))
        if template is None or (prompt["template_sha256"], prompt["context_rules_version"]) != (
            template.sha256,
            LIVE_RULES_VERSION,
        ):
            stale = True
        else:
            audit.ok(
                "live-call",
                job == call["job"],
                f"{where}: the logged job differs in {differing(job, call['job'])}",
            )
            if recomputed.runner == "claude":  # the API writer, since 2026-10-27
                logged = (call.get("api") or {}).get("request_sha256")
                audit.ok(
                    "live-call",
                    logged == sha256_of(request_body(recomputed)),
                    f"{where}: the logged request {str(logged)[:12]} is not the job's",
                )
        # A busy or unreachable API (or a lost worker) is retried once by the scheduler, not a mismatch.
        audit.ok(
            "live-call",
            call["status"] == "completed" or call["status"] in TRANSIENT,
            f"{where}: {call['status']} {call.get('error')}",
        )
        if call["status"] == "completed":
            shown_on_screen(audit, where, call, source.text)
            audit.ok(
                "live-call",
                call["target_ms"] <= source.start_ms
                or call["target_ms"] <= call["position_ms"] + call["latency_estimate_ms"],
                f"{where}: targeted {call['target_ms']} ms, after the cue's start",
            )
    if stale:
        audit.ok(
            "versions",
            False,
            f"{who}: live calls ran another live-p2 template or rule version than this checkout "
            f"(v{LIVE_CELL_PROMPTS[(0, 'broad')].version}, rules v{LIVE_RULES_VERSION}); "
            "audit at the commit the session ran",
        )

    # Events name the video they belong to. Logs from before 2026-10-20 give only its index in the
    # watch order; videos are watched one after the other and each opens with its first cues written,
    # so the attempts name them in that same order.
    order = list(dict.fromkeys(call["video_id"] for call in lives))
    watched = {e["video_index"]: e.get("video_id") for e in events if e["event"] == "watch_started"}
    audit.ok(
        "live-videos", len(order) == len(watched), f"{who}: watched {sorted(watched)}, calls for {order}"
    )
    for index, named in watched.items():
        if named is not None:
            audit.ok(
                "live-videos",
                index < len(order) and order[index] == named,
                f"{who}: video {index} is {named} in the events, {order[index : index + 1]} in the calls",
            )

    def video(index: int) -> str | None:
        return watched.get(index) or (order[index] if index < len(order) else None)

    settings = [e for e in events if e["event"] == "settings"]
    for index, change in enumerate(settings):
        video_id = video(change["video_index"])
        following = next(
            (s for s in settings[index + 1 :] if s["video_index"] == change["video_index"]), None
        )
        if following is not None and following["position_ms"] - change["position_ms"] < SUPERSEDED_MS:
            audit.note("live-settings", "replaced within 1.5 s, before a call")  # a burst of moves
            continue
        after = [c for c in lives if c["spec"] == change["steps"] and c["video_id"] == video_id]
        estimates = [
            c["latency_estimate_ms"]
            for c in lives
            if c["video_id"] == video_id and c["position_ms"] <= change["position_ms"]
        ]
        starts = [cue.start_ms for (unit, _), cue in cues.items() if unit == video_id]
        # The caption changes only at a boundary: none left at or after p + L means nothing to write.
        late = bool(estimates and starts) and change["position_ms"] + estimates[-1] > max(starts)
        audit.ok(
            "live-settings",
            bool(after) or late,
            f"{who}: no call under {change['steps']} after the change at {change['position_ms']} ms",
        )
        if late and not after:
            audit.note("live-settings", "after the last boundary")
    by_job = {c["job"]["job_id"]: c for c in lives}
    exposures = [
        {**entry, "video_index": e["video_index"]}
        for e in events
        if e["event"] == "exposures"
        for entry in e["entries"]
    ]
    # From diff.live-attempt/v4 on, every window shows a caption about its own audio: a usable one
    # Trida wrote for that cue (under any settings) or that cue's starting caption.
    own_window = any(c["schema"].endswith("/v4") for c in lives)
    for shown in exposures:
        video_id, job_id = video(shown["video_index"]), str(shown.get("job_id") or "")
        if own_window and job_id.startswith(f"{STARTING_CAPTION}:"):
            audit.ok(
                "exposures",
                job_id == f"{STARTING_CAPTION}:{video_id}:{shown['cue']}",
                f"{who}: cue {shown['cue']} of {video_id} showed {job_id}",
            )
            audit.note("exposures", "starting caption shown")
            continue
        written = by_job.get(job_id)
        if audit.ok("exposures", written is not None, f"{who}: exposure {shown} has no live call"):
            assert written is not None
            same_video = written["video_id"] == video_id
            if own_window:
                fits = same_video and written["cue"] == shown["cue"] and written.get("usable") is True
            else:  # before v4 a cue kept the previous cue's caption until a new one was ready
                fits = same_video and written["cue"] <= shown["cue"]
            audit.ok(
                "exposures",
                fits,
                f"{who}: cue {shown['cue']} of video {shown['video_index']} showed a caption written "
                f"for {written['video_id']}#{written['cue']} (usable: {written.get('usable')})",
            )
    latencies = [c["latency_ms"] for c in lives if c.get("latency_ms") is not None]
    return {
        "live_calls": len(lives),
        "settings_changes": len(settings),
        "exposures": len(exposures),
        "kept_previous": sum(
            1 for s in exposures if by_job.get(s.get("job_id", ""), {}).get("cue") != s["cue"]
        ),
        "live_latency_ms_median": statistics.median(latencies) if latencies else None,
        "live_latency_ms_max": max(latencies) if latencies else None,
        "live_check_flags": dict(
            Counter(f["code"] for c in lives for f in (c.get("checks") or {}).get("flags", []))
        ),
        "live_check_failures": dict(
            Counter(f["code"] for c in lives for f in (c.get("checks") or {}).get("failures", []))
        ),
    }


def cell(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_export(
    audit: Audit,
    export_dir: Path,
    setup: Mapping[str, Any],
    sessions: list[dict[str, Any]],
    store_events: Mapping[str, list[dict[str, Any]]],
    logs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> None:
    """The export against the store and the logs, file by file."""
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        actual = hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
        audit.ok("export", actual == digest, f"{name}: hash differs from the manifest")
    audit.ok("export", manifest["setup_sha256"] == sha256_of(setup), "the export was made with another setup")
    lines = {line["session_id"]: line for line in read_jsonl(export_dir / "participants.jsonl")}
    audit.ok(
        "export",
        set(lines) == {s["session_id"] for s in sessions},
        f"sessions differ: export {sorted(lines)}, store {[s['session_id'] for s in sessions]}",
    )
    phase1: dict[tuple[str, ...], str] = {}
    seconds: dict[tuple[str, str], dict[str, str]] = {}
    post_watch: dict[tuple[str, str], str] = {}
    for session in sessions:
        sid, who, state = session["session_id"], session["participant"], session["state"]
        line = lines.get(sid)
        if line is None:
            continue
        for field in ("participant", "created_at", "finished_at", "revision", "state"):
            audit.ok("export", line[field] == session[field], f"{who}: participants.jsonl {field} != store")
        audit.ok("export", line["store_events"] == store_events[sid], f"{who}: store events differ")
        audit.ok("export", line["live"] == logs[sid], f"{who}: live logs differ")
        for record in state["responses"]:
            if record.get("analysis_excluded"):
                continue
            viewing = str(record["view_id"]).rsplit("-", 1)[-1]
            for item in record["presented_order"]:
                key = (sid, str(record["clip_index"]), viewing, str(record["page"]), str(item))
                phase1[key] = cell(record["answers"].get(item))
        served = {  # the model that wrote each clip's delivered caption, from the logs
            record["clip"]: (record.get("api") or {}).get("model") or ""
            for record in logs[sid]["attempts"]
            if record.get("kind") == "tweak" and record.get("outcome") == "ready"
        }
        for clip, entry in state.get("second_captions", {}).items():
            slots, model = entry.get("slots", {}), entry.get("model", {})
            seconds[(sid, clip)] = {
                "status": entry["status"],
                "second_caption": entry.get("text") or "",
                "caption_source": entry.get("source", "trida"),
                "variant_id": entry.get("variant_id", ""),
                "rung": str(entry.get("rung", "")) if entry.get("source", "trida") == "trida" else "",
                "writer_model": served.get(clip, "") if entry["status"] == "ready" else "",
                "emphasis": model.get("emphasis", {}).get("chosen", ""),
                **{name: slots.get(name, "") for name in SECOND_SLOTS},
                **{name: str(model.get(name, "")) for name in SECOND_MODEL},
            }
        for item, answer in state.get("live", {}).get("answers", {}).items():
            post_watch[(sid, item)] = str(answer)
    exported_phase1 = {
        (r["session_id"], r["clip_index"], r["viewing"], r["page"], r["item"]): r["answer"]
        for r in read_csv(export_dir / "phase1.csv")
    }
    changed = set(exported_phase1.items()) ^ set(phase1.items())
    audit.ok("export", not changed, f"phase1.csv differs from the store in {len(changed)} cells")
    exported_seconds = {
        (r["session_id"], r["clip"]): {name: r[name] for name in SECOND_FIELDS}
        for r in read_csv(export_dir / "second_captions.csv")
    }
    audit.ok("export", exported_seconds == seconds, "second_captions.csv differs from the store")
    shown = {
        (caption_id(clip, entry["text"]), option["value"])
        for session in sessions
        for clip, entry in session["state"].get("second_captions", {}).items()
        if entry.get("text")
        for option in setup_clips(setup)[clip]["options"]["R2"]
    }
    coding = read_csv(export_dir / "r2_coding.csv")
    audit.ok(
        "export",
        {(r["caption_id"], r["option"]) for r in coding} == shown and len(coding) == len(shown),
        "r2_coding.csv is not one row per shown caption and R2 option",
    )
    audit.ok(
        "export",
        all(
            r["named"] == "" and set(r) == {"caption_id", "clip", "caption", "option", "label", "named"}
            for r in coding
        ),
        "r2_coding.csv carries more than the blind coder may see, or a pre-filled answer",
    )
    exported_post = {
        (r["session_id"], r["item"]): r["answer"] for r in read_csv(export_dir / "post_watch.csv")
    }
    audit.ok("export", exported_post == post_watch, "post_watch.csv differs from the store")


def setup_clips(setup: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {clip["key"]: clip for clip in setup["clips"]}


def check(args: argparse.Namespace) -> int:
    audit = Audit()
    setup = read_setup(args.phase1_setup)
    clips = {clip["key"]: clip for clip in setup["clips"]}
    with args.sources.open(encoding="utf-8") as handle:
        sources = parse_sources(json.load(handle)).sources
    cues = {(s.unit_key, s.cue_index): s for s in sources if s.phase == 2}
    scenes = json.loads(args.phase2_scene.read_text(encoding="utf-8")) if args.phase2_scene else None
    sessions, store_events = read_store(args.store)
    logs = {s["session_id"]: session_logs(args.log_dir, s["session_id"]) for s in sessions}
    report: dict[str, Any] = {"sessions": [], "second_captions": []}
    for session in sessions:
        who, state = session["participant"], session["state"]
        attempts = logs[session["session_id"]]["attempts"]
        tweaks = [a for a in attempts if a.get("kind") == "tweak"]
        lives = [a for a in attempts if a.get("kind") != "tweak"]
        report["second_captions"] += audit_phase1(audit, who, state, clips, tweaks)
        tuner = audit_tuner(audit, who, state, clips)
        row: dict[str, Any] = {"participant": who, "finished": session["finished_at"] is not None}
        if tuner is not None:
            row |= {"tuner": tuner, "post_watch": state.get("live", {}).get("answers")}
            events = logs[session["session_id"]]["events"]
            row |= audit_live(audit, who, tuner, lives, events, cues, scenes, state)
        report["sessions"].append(row)
    if args.export is not None:
        audit_export(audit, args.export, setup, sessions, store_events, logs)
    audit.print()
    if args.out is not None:
        report["counts"] = {group: dict(counts) for group, counts in audit.counts.items()}
        report["problems"] = audit.problems
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"report: {args.out}")
    return 1 if audit.problems else 0


def tokens(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    from diff.model_profile import TRIDA_7B

    model = TRIDA_7B
    tokenizer: Any = AutoTokenizer.from_pretrained(str(args.trida_dir), local_files_only=True)
    counts: Counter[str] = Counter()
    mismatches: list[str] = []
    for log_dir in args.log_dir:
        for path in sorted(log_dir.rglob("live-*.jsonl")):
            for record in read_jsonl(path):
                kind = record.get("kind", "live")
                if record["job"]["model"]["profile_id"] != model.profile_id:
                    counts[f"{kind} by another model, not checked here"] += 1
                    continue
                if not record.get("token_ids_sha256"):
                    counts[f"{kind} not tokenized (failed first)"] += 1
                    continue
                messages = [
                    {"role": "system", "content": record["system_message"]},
                    {"role": "user", "content": record["job"]["prompt"]["text"]},
                ]
                encoded = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **dict(model.chat_options)
                )
                ids = [int(t) for t in (encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)]
                digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
                same = digest == record["token_ids_sha256"]
                counts[f"{kind} {'match' if same else 'MISMATCH'}"] += 1
                if not same:
                    where = record.get("clip") or f"{record.get('video_id')}#{record.get('cue')}"
                    mismatches.append(f"{path.name} {kind} {where}")
    print(dict(counts))
    print("\n".join(mismatches) or "every logged prompt is what the worker tokenized")
    return 1 if mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checked = commands.add_parser("check", help="[repository env] recompute every session from its answers")
    checked.add_argument("--store", type=Path, required=True)
    checked.add_argument("--log-dir", type=Path, action="append", required=True, help="repeatable")
    checked.add_argument("--phase1-setup", type=Path, required=True, help="the setup the sessions ran with")
    checked.add_argument("--sources", type=Path, required=True, help="the phase-2 cues the sessions ran with")
    checked.add_argument("--phase2-scene", type=Path, help="the cues' scene parts the sessions ran with")
    checked.add_argument("--export", type=Path, help="a python -m diff export directory to compare")
    checked.add_argument("--out", type=Path, help="write the report as JSON")
    checked.set_defaults(handler=check)
    tokenized = commands.add_parser("tokens", help="[generator env] re-tokenize every logged Trida call")
    tokenized.add_argument("--trida-dir", type=Path, required=True, help="Trida's pinned snapshot")
    tokenized.add_argument("--log-dir", type=Path, action="append", required=True, help="repeatable")
    tokenized.set_defaults(handler=tokens)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
