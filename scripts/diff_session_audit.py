"""Audit diff sessions: recompute from each participant's answers what Trida should have received.

    uv run python scripts/diff_session_audit.py check --store STORE --log-dir DIR [--log-dir DIR …] \\
        --phase1-setup SETUP --sources SOURCES [--export EXPORT_DIR] [--out REPORT.json]
    envs/diff-generator/.venv/bin/python scripts/diff_session_audit.py tokens \\
        --trida-dir artifacts/diff/models/trida-7b-c37755e8 --log-dir DIR [--log-dir DIR …]

``check`` runs on every pilot and participant session before analysis. The answer side is
recomputed here rather than by diff: the object under each P2 point (re-reading the masks), each
direction's degree, the attended sound, the tuner's heard families and rating means. diff's own
prompt functions then render what those values should have produced, and every logged call is
compared with that, so new prompt wording needs no change here. A session that ran an older
template than the checkout is reported once, not call by call. It also checks that:
- each clip's second caption is the one Trida wrote and P6 showed, or that a failed one was skipped;
- each attempt's shown text is Trida's text under the display rule (logs from 2026-10-20 on);
- every slider change got a live call, unless no cue boundary was left to write;
- every caption shown is one Trida wrote for that cue or an earlier one (the kept caption);
- with ``--export``, that the export matches the store and the logs.

``tokens`` runs in the generator environment. It re-applies Trida's chat template to every
logged call's system message and prompt and compares the token hash with the one the worker
reported, which shows the logged prompt is what reached the model. Run it when the model, the
tokenizer or the generator environment changes.

Both only read; the store is opened read-only. Each exits 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from diff.job_plan import live_job  # noqa: E402
from diff.output_checks import DISPLAY_RULE, readable, shown_text  # noqa: E402
from diff.phase1_setup import read_setup  # noqa: E402
from diff.phase1_tweak import (  # noqa: E402
    DEGREE_RULE,
    DIRECTIONS,
    NOT_SEEN,
    OBJECTS_KO,
    SCALE_MAX,
    check_record,
    tweak_job_for,
    tweak_source,
)
from diff.rewrite_prompts import (  # noqa: E402
    LIVE_REWRITE,
    LIVE_RULES_VERSION,
    TUNER_SYSTEM,
    TWEAK_P1,
    render_tuner_system,
)
from diff.study_protocol import sha256_of  # noqa: E402

STEP1_PAGES = ("P2", "P3", "P4", "P5")
SECOND_FIELDS = ("status", "second_caption", "attended_sound", "attended_object", *DIRECTIONS)
RATINGS = ("S1", "S2", "S3")
MASK_THRESHOLD = 127


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


def object_at(frame: Mapping[str, Any], x: float, y: float) -> str | None:
    """The annotated object under a point: the smallest mask containing it, as regen matches."""
    import numpy as np
    from PIL import Image

    best: tuple[str, int] | None = None
    for obj in frame["objects"]:
        inside = np.asarray(Image.open(REPO_ROOT / obj["mask"]).convert("L")) > MASK_THRESHOLD
        height, width = inside.shape
        if inside[min(height - 1, round(y * (height - 1))), min(width - 1, round(x * (width - 1)))]:
            area = int(inside.sum())
            if best is None or area < best[1]:
                best = (str(obj["id"]), area)
    return best[0] if best else None


def expected_slots(clip: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """The tweak slots, recomputed from the clip's first-step answers."""
    answers: dict[str, Any] = {}
    for page in STEP1_PAGES:
        answers.update(records[page]["answers"])
    sounds = {option["value"]: option["label"] for option in clip["options"]["A1s"]}
    matched = [match["object_id"] for match in records["P2"].get("matches", []) if match.get("object_id")]
    slots = {
        "caption": clip["caption"]["text"],
        "sounds": ", ".join(option["label"] for option in clip["options"]["R2"] if option.get("mentioned")),
        "attended_sound": sounds.get(answers.get("A1s"), NOT_SEEN),
        "attended_object": OBJECTS_KO.get(matched[0], NOT_SEEN) if matched else NOT_SEEN,
    }
    for direction, items in DIRECTIONS.items():
        shown = statistics.mean(float(answers[item]) for item in items) / SCALE_MAX
        slots[direction] = str(round((shown if DEGREE_RULE == "reflect" else 1 - shown) * 100))
    return slots


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
    audit: Audit, who: str, state: Mapping[str, Any], clips: Mapping[str, Any], tweaks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Slots, the tweak call, its check and outcome, per clip; returns one report row per clip."""
    records = formal_records(state)
    rows: list[dict[str, Any]] = []
    ran: set[str] = set()
    for index, key in enumerate(state["clip_order"]):
        clip = clips[key]
        step1 = {page: records[(index, page)] for page in STEP1_PAGES if (index, page) in records}
        entry = state["second_captions"].get(key)
        if len(step1) < len(STEP1_PAGES):
            if entry is not None:
                audit.ok("tweak-slots", False, f"{who} {key}: a second caption without a complete first step")
            continue
        if not audit.ok("tweak-slots", entry is not None, f"{who} {key}: no second-caption entry"):
            continue
        assert entry is not None
        expect = expected_slots(clip, step1)
        audit.ok(
            "tweak-slots",
            entry["slots"] == expect,
            f"{who} {key}: stored slots differ in {differing(expect, entry['slots'])}",
        )
        point = step1["P2"]["answers"]["V1"][0]
        frame = next(
            (f for i, f in enumerate(clip["frames"]) if int(f.get("index", i)) == int(point["frame_id"])),
            None,
        )
        matched = [m["object_id"] for m in step1["P2"].get("matches", []) if m.get("object_id")]
        if audit.ok("tweak-slots", frame is not None, f"{who} {key}: P2 point on unknown frame {point}"):
            assert frame is not None
            under = object_at(frame, float(point["x"]), float(point["y"]))
            audit.ok(
                "tweak-slots",
                under == (matched[0] if matched else None),
                f"{who} {key}: the masks put {point} on {under}, the record matched {matched}",
            )
        attempts = [t for t in tweaks if t["clip"] == key]
        if not audit.ok("tweak-call", bool(attempts), f"{who} {key}: no tweak call logged"):
            continue
        last = attempts[-1]
        audit.ok(
            "tweak-call", last["steer"]["slots"] == entry["slots"], f"{who} {key}: logged slots != stored"
        )
        _, job, _ = tweak_job_for(clip, expect, str(last["job"]["purpose"]))
        expected_job = plain(job.record())
        if entry["template"]["sha256"] != TWEAK_P1.sha256:
            ran.add(str(entry["template"]["sha256"]))
        else:
            audit.ok(
                "tweak-call",
                expected_job == last["job"],
                f"{who} {key}: the logged job differs in {differing(expected_job, last['job'])}",
            )
        audit.ok(
            "tweak-call",
            last["system_message"] == job.prompt.system_message,
            f"{who} {key}: tweak system message {last['system_message']!r}",
        )
        text = shown_on_screen(audit, f"{who} {key}", last)
        if last["status"] == "completed" and last.get("output"):
            if last.get("display_rule") == DISPLAY_RULE:
                audit.ok(
                    "tweak-check",
                    plain(check_record(tweak_source(clip), job, last)) == last["checks"],
                    f"{who} {key}: re-running the tweak check gives a different result than logged",
                )
            else:
                audit.note("tweak-check", "before the display rule, not re-run")
        failed = last["status"] != "completed" or bool((last.get("checks") or {}).get("failures"))
        if entry["status"] == "ready":
            audit.ok("tweak-outcome", not failed, f"{who} {key}: ready although the last attempt failed")
            audit.ok("tweak-outcome", entry["text"] == text, f"{who} {key}: stored caption != Trida's output")
            p6 = [v for v in state["viewings"] if v["clip_index"] == index and v["page"] == "P6"]
            audit.ok(
                "tweak-outcome",
                bool(p6) and p6[-1]["captions"][0]["text"] == text,
                f"{who} {key}: the P6 viewing did not show Trida's caption",
            )
        else:
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
            f"{who}: the second captions ran tweak-p1 {sorted(sha[:12] for sha in ran)}, this checkout "
            f"renders {TWEAK_P1.sha256[:12]}; audit at the commit the session ran",
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


def audit_live(
    audit: Audit,
    who: str,
    tuner: str,
    lives: list[dict[str, Any]],
    events: list[dict[str, Any]],
    cues: Mapping[tuple[str, int], CaptionSource],
) -> dict[str, Any]:
    """Every live call against the slider steps and the tuner; settings and exposures against the calls."""
    stale = False
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
        job = plain(
            live_job(
                source, int(spec["texture"]), str(spec["context"]), tuner, str(call["job"]["purpose"])
            ).record()
        )
        prompt = call["job"]["prompt"]
        if (prompt["template_sha256"], prompt["context_rules_version"]) != (
            LIVE_REWRITE.sha256,
            LIVE_RULES_VERSION,
        ):
            stale = True
        else:
            audit.ok(
                "live-call",
                job == call["job"],
                f"{where}: the logged job differs in {differing(job, call['job'])}",
            )
        audit.ok("live-call", call["status"] == "completed", f"{where}: {call['status']} {call.get('error')}")
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
            f"(v{LIVE_REWRITE.version}, rules v{LIVE_RULES_VERSION}); audit at the commit the session ran",
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
    for change in settings:
        video_id = video(change["video_index"])
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
    for shown in exposures:
        written = by_job.get(shown.get("job_id", ""))
        if audit.ok("exposures", written is not None, f"{who}: exposure {shown} has no live call"):
            assert written is not None
            audit.ok(
                "exposures",
                written["cue"] <= shown["cue"] and written["video_id"] == video(shown["video_index"]),
                f"{who}: cue {shown['cue']} of video {shown['video_index']} showed a caption written "
                f"for {written['video_id']}#{written['cue']}",
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
        for clip, entry in state.get("second_captions", {}).items():
            slots = entry.get("slots", {})
            seconds[(sid, clip)] = {
                "status": entry["status"],
                "second_caption": entry.get("text") or "",
                **{name: slots.get(name, "") for name in SECOND_FIELDS[2:]},
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
    exported_post = {
        (r["session_id"], r["item"]): r["answer"] for r in read_csv(export_dir / "post_watch.csv")
    }
    audit.ok("export", exported_post == post_watch, "post_watch.csv differs from the store")


def check(args: argparse.Namespace) -> int:
    audit = Audit()
    setup = read_setup(args.phase1_setup)
    clips = {clip["key"]: clip for clip in setup["clips"]}
    with args.sources.open(encoding="utf-8") as handle:
        sources = parse_sources(json.load(handle)).sources
    cues = {(s.unit_key, s.cue_index): s for s in sources if s.phase == 2}
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
            row |= audit_live(audit, who, tuner, lives, logs[session["session_id"]]["events"], cues)
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

    tokenizer: Any = AutoTokenizer.from_pretrained(str(args.trida_dir), local_files_only=True)
    counts: Counter[str] = Counter()
    mismatches: list[str] = []
    for log_dir in args.log_dir:
        for path in sorted(log_dir.rglob("live-*.jsonl")):
            for record in read_jsonl(path):
                kind = record.get("kind", "live")
                if not record.get("token_ids_sha256"):
                    counts[f"{kind} not tokenized (failed first)"] += 1
                    continue
                messages = [
                    {"role": "system", "content": record["system_message"]},
                    {"role": "user", "content": record["job"]["prompt"]["text"]},
                ]
                ids = [
                    int(t)
                    for t in tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True
                    )
                ]
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
    checked.add_argument("--export", type=Path, help="a python -m diff export directory to compare")
    checked.add_argument("--out", type=Path, help="write the report as JSON")
    checked.set_defaults(handler=check)
    tokenized = commands.add_parser("tokens", help="[generator env] re-tokenize every logged call")
    tokenized.add_argument("--trida-dir", type=Path, required=True)
    tokenized.add_argument("--log-dir", type=Path, action="append", required=True, help="repeatable")
    tokenized.set_defaults(handler=tokens)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
