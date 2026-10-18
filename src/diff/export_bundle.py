"""The research export: every participant's session, from the store and the live logs.

Researcher's decision of 2026-10-18: export is a command on this machine
(``python -m diff export``); nothing a browser can reach exports data. This is
the only place where the real clip keys, each A1s answer's sound family and
the R2 answer key (``mentioned``) appear beside the answers.

Files written to ``--out``:
- ``participants.jsonl``: one line per session, holding everything:
  - the state (seed, clip order, consent, phase-1 responses and viewings,
    second captions, skipped steps, practice flagged ``analysis_excluded``,
    calibration, live progress);
  - the store's events;
  - the live event and attempt logs, with every second-caption attempt.
- ``phase1.csv``: one row per participant × clip × item, practice excluded.
  ``viewing`` says which viewing the answer belongs to (``P1`` or ``P6``). It
  adds, for A1s and A2s, the chosen sound's label, family and the
  ``orig_text`` words it renders; for V1 and V2, the matched objects; and for
  R2, the intended key, the sounds the second-caption prompt named.
- ``second_captions.csv``: one row per participant × clip: the status (and why
  a step was skipped), the first and second captions, what steered the second
  (the attended sound and object, the four degrees) and its check flags.
- ``post_watch.csv``: one row per participant × post-watch item.
- ``manifest.json``: counts, the setup's hash and each file's SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from diff.session_store import SessionStore
from diff.study_protocol import sha256_of

EXPORT_SCHEMA: Final = "diff.export/v2"
PHASE1_COLUMNS: Final = (
    "participant",
    "session_id",
    "finished",
    "clip_index",
    "clip",
    "viewing",
    "page",
    "item",
    "position",
    "answer",
    "sound_label",
    "sound_family",
    "sound_orig_phrase",
    "point_object",
    "r2_intended_key",
    "submitted_at",
)
SECOND_CAPTION_COLUMNS: Final = (
    "participant",
    "session_id",
    "finished",
    "clip_index",
    "clip",
    "status",
    "reason",
    "attempts",
    "first_caption",
    "second_caption",
    "attended_sound",
    "attended_object",
    "fascination",
    "being_away",
    "coherence",
    "familiarity",
    "intended_r2_key",
    "template_sha256",
    "template_provisional",
    "job_id",
    "check_flags",
)
SOUND_ITEMS: Final = frozenset({"A1s", "A2s"})
POINT_ITEMS: Final = frozenset({"V1", "V2"})
POST_WATCH_COLUMNS: Final = ("participant", "session_id", "finished", "item", "answer")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def live_logs(log_dirs: Sequence[Path], session_id: str) -> dict[str, list[dict[str, Any]]]:
    """The session's live events (``session-<id>.jsonl``) and attempts (``live-<id>.jsonl``)."""
    events: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for log_dir in log_dirs:
        for path in sorted(log_dir.rglob(f"session-{session_id}.jsonl")):
            events += _jsonl(path)
        for path in sorted(log_dir.rglob(f"live-{session_id}.jsonl")):
            attempts += _jsonl(path)
    return {"events": events, "attempts": attempts}


def _cell(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def phase1_rows(
    participant: str, session_id: str, finished: bool, state: Mapping[str, Any], clips: Mapping[str, Any]
) -> Iterable[dict[str, Any]]:
    """Flat rows for one session's formal phase-1 answers (practice is never here)."""
    for record in state.get("responses", []):
        if record.get("analysis_excluded"):
            continue
        clip = clips.get(record["clip"], {})
        sounds = {option["value"]: option for option in clip.get("options", {}).get("A1s", [])}
        mentioned = [
            option["value"] for option in clip.get("options", {}).get("R2", []) if option.get("mentioned")
        ]
        viewing = str(record["view_id"]).rsplit("-", 1)[-1]
        for position, item in enumerate(record["presented_order"]):
            answer = record["answers"].get(item)
            row = {
                "participant": participant,
                "session_id": session_id,
                "finished": finished,
                "clip_index": record["clip_index"],
                "clip": record["clip"],
                "viewing": viewing,
                "page": record["page"],
                "item": item,
                "position": position,
                "answer": _cell(answer),
                "sound_label": "",
                "sound_family": "",
                "sound_orig_phrase": "",
                "point_object": "",
                "r2_intended_key": "",
                "submitted_at": record["submitted_at"],
            }
            if item in SOUND_ITEMS and answer in sounds:
                row.update(
                    sound_label=sounds[answer]["label"],
                    sound_family=sounds[answer]["family"],
                    sound_orig_phrase=sounds[answer].get("orig_phrase", ""),
                )
            if item in POINT_ITEMS:
                row["point_object"] = _cell(
                    [match.get("object_id") or "unclassified" for match in record.get("matches", [])]
                )
            if item == "R2":
                row["r2_intended_key"] = _cell(mentioned)
            yield row


def second_caption_rows(
    participant: str, session_id: str, finished: bool, state: Mapping[str, Any], clips: Mapping[str, Any]
) -> Iterable[dict[str, Any]]:
    """One row per clip whose first step closed: how its second caption was made, or why it was not."""
    entries = sorted(state.get("second_captions", {}).items(), key=lambda item: item[1]["clip_index"])
    for key, entry in entries:
        clip = clips.get(key, {})
        slots = entry.get("slots", {})
        flags = [finding["code"] for finding in (entry.get("checks") or {}).get("flags", [])]
        yield {
            "participant": participant,
            "session_id": session_id,
            "finished": finished,
            "clip_index": entry["clip_index"],
            "clip": key,
            "status": entry["status"],
            "reason": entry.get("reason") or "",
            "attempts": len(entry.get("attempts", [])),
            "first_caption": clip.get("caption", {}).get("text", ""),
            "second_caption": entry.get("text", ""),
            "attended_sound": slots.get("attended_sound", ""),
            "attended_object": slots.get("attended_object", ""),
            "fascination": slots.get("fascination", ""),
            "being_away": slots.get("being_away", ""),
            "coherence": slots.get("coherence", ""),
            "familiarity": slots.get("familiarity", ""),
            "intended_r2_key": _cell(
                [o["value"] for o in clip.get("options", {}).get("R2", []) if o.get("mentioned")]
            ),
            "template_sha256": entry.get("template", {}).get("sha256", ""),
            "template_provisional": entry.get("template", {}).get("provisional", ""),
            "job_id": entry.get("job_id", ""),
            "check_flags": _cell(flags),
        }


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def export(
    store: SessionStore, log_dirs: Sequence[Path], setup: Mapping[str, Any], out: Path, *, generated_at: str
) -> dict[str, Any]:
    """Write the export files into ``out`` (which must be empty or new) and return the manifest."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; export into a new directory")
    out.mkdir(parents=True, exist_ok=True)
    clips = {clip["key"]: clip for clip in setup.get("clips", [])}
    sessions = store.sessions()
    phase1: list[dict[str, Any]] = []
    seconds: list[dict[str, Any]] = []
    post_watch: list[dict[str, Any]] = []
    with (out / "participants.jsonl").open("w", encoding="utf-8") as handle:
        for session in sessions:
            finished = session.finished_at is not None
            record = {
                "schema": EXPORT_SCHEMA,
                "participant": session.participant,
                "session_id": session.session_id,
                "created_at": session.created_at,
                "finished_at": session.finished_at,
                "revision": session.revision,
                "state": session.state,
                "store_events": store.events(session.session_id),
                "live": live_logs(log_dirs, session.session_id),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            phase1 += phase1_rows(session.participant, session.session_id, finished, session.state, clips)
            seconds += second_caption_rows(
                session.participant, session.session_id, finished, session.state, clips
            )
            for item, answer in sorted(session.state.get("live", {}).get("answers", {}).items()):
                post_watch.append(
                    {
                        "participant": session.participant,
                        "session_id": session.session_id,
                        "finished": finished,
                        "item": item,
                        "answer": answer,
                    }
                )
    counts = {
        "sessions": len(sessions),
        "finished": sum(session.finished_at is not None for session in sessions),
        "phase1_rows": _write_csv(out / "phase1.csv", PHASE1_COLUMNS, phase1),
        "second_caption_rows": _write_csv(out / "second_captions.csv", SECOND_CAPTION_COLUMNS, seconds),
        "post_watch_rows": _write_csv(out / "post_watch.csv", POST_WATCH_COLUMNS, post_watch),
    }
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.iterdir())
        if path.name != "manifest.json"
    }
    manifest = {
        "schema": EXPORT_SCHEMA,
        "generated_at": generated_at,
        "store": str(store.path),
        "log_dirs": [str(path) for path in log_dirs],
        "setup_sha256": sha256_of(setup),
        "counts": counts,
        "files": files,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
