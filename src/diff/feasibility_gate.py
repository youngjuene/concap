"""G0: Korean feasibility of Dream, checked before any other Dream work (follow-up §7).

Dream's continued pretraining and instruction data are mostly English and code
and its reported evaluations are English, so its Korean captions are unknown.
G0 asks whether it can write them at all, on development material only — never
a participant clip or video:

* every development-clip cue with the phase-1 rewrite prompt, and
* at least 12 phase-2 cell prompts over development-clip evidence, covering the
  lattice's four corners,

each generated as ``dream_k1``, ``dream_k2``, ``dream_k4`` and ``qwen``.

From the run come an automatic report per model and condition (technical
status, terminator inside the canvas, leftover masks, the script check, exact
copies, repetition, character limits, realized per-step counts, latency) and a
blind packet: ``dream_k1`` and ``qwen`` outputs mixed and shuffled under opaque
IDs with cue text, evidence and output, the key written to a separate
directory. One fluent Korean reader rates naturalness 1–5 and returns
``blind_id,naturalness``. :func:`decide` applies the provisional rule over
``dream_k1`` and writes an :class:`~diff.analysis_decisions.AnalysisDecision`
naming the run manifest's hash; Qwen's numbers are reported beside it.

The rating needs a person, so the pipeline stops once the packet exists. A
plan built from synthetic material is a smoke run: it can exercise every step
up to the packet, and :func:`decide` refuses it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from diff.analysis_decisions import AnalysisDecision
from diff.caption_sources import CaptionSource
from diff.generation_worker import read_attempts
from diff.job_plan import FrozenPlan, Plan, PlanError, condition_settings, make_job
from diff.model_profile import DREAM_7B
from diff.output_checks import check_output, output_from_record, script_passes
from diff.rewrite_prompts import render_cell, render_evidence, render_phase1
from diff.study_protocol import GATES, lattice_corners, sha256_of

G0_CONDITIONS: Final = ("dream_k1", "dream_k2", "dream_k4", "qwen")
RATED_CONDITIONS: Final = ("dream_k1", "qwen")
RUN_SCHEMA: Final = "diff.g0-run/v1"
PACKET_COLUMNS: Final = ("blind_id", "cue_text", "evidence", "output")


def g0_items(sources: Iterable[CaptionSource], *, material: str = "development") -> list[dict[str, Any]]:
    """The G0 prompts: every phase-1 cue of the material, then corner cells over the same cues."""
    chosen = sorted(
        (s for s in sources if s.phase == 1 and s.material == material and s.status != "rejected"),
        key=lambda s: (s.unit_key, s.cue_index),
    )
    if not chosen:
        raise PlanError(
            [
                f"G0 needs {material}-clip cues with evidence and none were supplied; "
                "ask the researcher for them"
            ]
        )
    bare = [f"{s.unit_key} cue {s.cue_index}" for s in chosen if not s.events]
    if bare:
        raise PlanError([f"these {material} cues have no evidence events: {', '.join(bare)}"])
    needed = int(GATES["G0"]["minimum_phase2_cell_prompts"])
    pairs = [(source, corner) for source in chosen for corner in lattice_corners()]
    if len(pairs) < needed:
        raise PlanError(
            [
                f"{len(chosen)} cues give {len(pairs)} corner prompts; "
                f"G0 needs at least {needed} (3 cues or more)"
            ]
        )
    items: list[dict[str, Any]] = [{"source": source, "template_phase": 1, "cell": None} for source in chosen]
    items += [{"source": source, "template_phase": 2, "cell": corner} for source, corner in pairs[:needed]]
    return items


def build_g0_plan(sources: Iterable[CaptionSource], *, smoke: bool = False) -> Plan:
    """Every G0 prompt under every G0 condition; ``smoke`` reads synthetic material instead."""
    purpose, material = ("smoke", "synthetic") if smoke else ("g0", "development")
    jobs = []
    for item in g0_items(sources, material=material):
        source, phase, cell = item["source"], int(item["template_phase"]), item["cell"]
        prompt = (
            render_phase1(source)
            if phase == 1
            else render_cell(source, int(cell["quality_limit"]), str(cell["context"]))
        )
        for condition in G0_CONDITIONS:
            model, generate = condition_settings(phase, condition)
            jobs.append(make_job(source, purpose, condition, cell, model, prompt, generate, phase=phase))
    return Plan(phase=0, purpose=purpose, jobs=tuple(jobs))


def latest_attempts(run_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for attempt in read_attempts(run_dir / "attempts.jsonl"):
        latest[attempt["job_id"]] = attempt
    return latest


def run_manifest(frozen: FrozenPlan, run_dir: Path, sources: Sequence[CaptionSource]) -> dict[str, Any]:
    """What the G0 decision names: plan, models, sources and the exact attempt log."""
    attempts_path = run_dir / "attempts.jsonl"
    attempts_bytes = attempts_path.read_bytes() if attempts_path.is_file() else b""
    used = {job.job.source_id for job in frozen.jobs}
    chosen = sorted((s for s in sources if s.source_id in used), key=lambda s: s.source_id)
    body = {
        "schema": RUN_SCHEMA,
        "plan_id": frozen.plan.plan_id,
        "purpose": frozen.plan.purpose,
        "frozen_sha256": frozen.frozen_sha256,
        "model_profiles": sorted({job.job.profile_id for job in frozen.jobs}),
        "sources": [source.source_id for source in chosen],
        "materials": sorted({source.material for source in chosen}),
        "sources_sha256": sha256_of([source.source_sha256 for source in chosen]),
        "attempts_sha256": hashlib.sha256(attempts_bytes).hexdigest(),
        "runs": sorted({attempt["run_id"] for attempt in read_attempts(attempts_path)}),
    }
    return {**body, "manifest_sha256": sha256_of(body)}


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def g0_report(
    frozen: FrozenPlan, latest: Mapping[str, Mapping[str, Any]], sources: Mapping[str, CaptionSource]
) -> dict[str, Any]:
    """The automatic report, per model and condition, with every job's check result."""
    lexicon = sorted({term for source in sources.values() for term in source.evidence_terms()["sources"]})
    jobs_out, conditions = [], {}
    for condition in G0_CONDITIONS:
        entries = [job for job in frozen.jobs if job.job.condition == condition]
        statuses: dict[str, int] = {}
        stats = {
            "jobs": len(entries),
            "terminal": 0,
            "technical_failures": 0,
            "completed": 0,
            "terminated": 0,
            "incomplete": 0,
            "leftover_mask": 0,
            "script_pass": 0,
            "exact_copies": 0,
            "repetition": 0,
            "char_limit_violations": 0,
        }
        latencies: list[float] = []
        per_step: list[int] = []
        zero_steps = 0
        for entry in entries:
            job, attempt = entry.job, latest.get(entry.job.job_id)
            status = attempt["status"] if attempt else "not_run"
            statuses[status] = statuses.get(status, 0) + 1
            stats["terminal"] += int(bool(attempt and attempt["terminal"]))
            stats["technical_failures"] += int(bool(attempt and attempt["technical_failure"]))
            row: dict[str, Any] = {"job_id": job.job_id, "condition": condition, "status": status}
            if attempt and attempt.get("latency_ms") is not None:
                latencies.append(float(attempt["latency_ms"]))
            summary = (attempt or {}).get("trace") or {}
            counts = (
                summary.get("summary", {}).get("realized_per_step") if isinstance(summary, dict) else None
            )
            if counts:
                per_step.extend(int(count) for count in counts)
                zero_steps += sum(1 for count in counts if count == 0)
            if attempt and status == "completed" and attempt.get("output"):
                output = output_from_record(attempt["output"])
                result = check_output(
                    sources[job.source_id], output, template_phase=job.phase, cell=job.cell, lexicon=lexicon
                )
                codes = {finding.code for finding in result.failures}
                flags = {finding.code for finding in result.flags}
                stats["completed"] += 1
                stats["terminated"] += int(output.terminator is not None)
                stats["incomplete"] += int(output.terminator is None)
                stats["leftover_mask"] += int(output.leftover_mask_count > 0)
                stats["script_pass"] += int(script_passes(result))
                stats["exact_copies"] += int("identical_to_source" in flags)
                stats["repetition"] += int("repetition" in flags)
                stats["char_limit_violations"] += int("length_over_limit" in codes)
                row.update(text=output.text, checks=result.record())
            jobs_out.append(row)
        conditions[condition] = {
            **stats,
            "statuses": statuses,
            "script_pass_share": stats["script_pass"] / stats["jobs"] if stats["jobs"] else 0.0,
            "latency_ms": {
                "median": statistics.median(latencies) if latencies else None,
                "p95": _quantile(latencies, 0.95),
            },
            "realized_per_step": {
                "steps_observed": len(per_step),
                "zero_commit_steps": zero_steps,
                "max_commit": max(per_step, default=0),
            },
        }
    return {"conditions": conditions, "jobs": jobs_out}


def blind_packet(
    frozen: FrozenPlan,
    latest: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, CaptionSource],
    manifest_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """``dream_k1`` and ``qwen`` outputs under opaque IDs, shuffled; the key separately."""
    rows, key = [], {}
    for entry in frozen.jobs:
        job, attempt = entry.job, latest.get(entry.job.job_id)
        if job.condition not in RATED_CONDITIONS or not attempt or attempt["status"] != "completed":
            continue
        blind_id = "b" + sha256_of([manifest_sha256, job.job_id])[:10]
        source = sources[job.source_id]
        rows.append(
            {
                "blind_id": blind_id,
                "cue_text": source.text,
                "evidence": render_evidence(source.events),
                "output": str(attempt["output"]["text"]),
            }
        )
        key[blind_id] = {
            "job_id": job.job_id,
            "condition": job.condition,
            "profile_id": job.profile_id,
            "template_phase": job.phase,
            "cell": dict(job.cell) if job.cell else None,
        }
    rows.sort(key=lambda row: sha256_of([manifest_sha256, "order", row["blind_id"]]))
    return rows, key


def packet_csv(rows: Sequence[Mapping[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(PACKET_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def ratings_template(rows: Sequence[Mapping[str, str]]) -> str:
    return "blind_id,naturalness\n" + "".join(f"{row['blind_id']},\n" for row in rows)


def read_ratings(text: str, key: Mapping[str, Any]) -> dict[str, int]:
    """``blind_id,naturalness`` with every key ID rated exactly once on 1–5; anything else is refused."""
    problems: list[str] = []
    ratings: dict[str, int] = {}
    seen: set[str] = set()
    for line, row in enumerate(csv.DictReader(io.StringIO(text)), 2):
        blind_id, raw = (row.get("blind_id") or "").strip(), (row.get("naturalness") or "").strip()
        if blind_id not in key:
            problems.append(f"line {line}: unknown blind_id {blind_id!r}")
            continue
        if blind_id in seen:
            problems.append(f"line {line}: {blind_id} rated twice")
            continue
        seen.add(blind_id)
        if raw not in {"1", "2", "3", "4", "5"}:
            problems.append(f"line {line}: naturalness must be an integer from 1 to 5, got {raw!r}")
        else:
            ratings[blind_id] = int(raw)
    missing = sorted(set(key) - seen)
    if missing:
        problems.append(f"{len(missing)} blind IDs were not rated, e.g. {', '.join(missing[:5])}")
    if problems:
        raise ValueError("; ".join(problems))
    return ratings


def decide(
    report: Mapping[str, Any],
    key: Mapping[str, Mapping[str, Any]],
    ratings: Mapping[str, int],
    manifest: Mapping[str, Any],
    *,
    decided_by: str,
    decided_at: str,
    notes: str = "",
) -> AnalysisDecision:
    """The provisional G0 rule over ``dream_k1``, with Qwen's numbers beside it."""
    if manifest.get("purpose") != "g0" or manifest.get("materials") != ["development"]:
        raise ValueError("only a G0 run over development material can be decided; smoke runs never can")
    rule = GATES["G0"]
    evidence: dict[str, Any] = {}
    for condition in RATED_CONDITIONS:
        stats = report["conditions"][condition]
        rated = [score for blind_id, score in ratings.items() if key[blind_id]["condition"] == condition]
        evidence[condition] = {
            "jobs": stats["jobs"],
            "terminal": stats["terminal"],
            "technical_failures": stats["technical_failures"],
            "script_pass_share": stats["script_pass_share"],
            "rated": len(rated),
            "naturalness_share": (
                sum(1 for score in rated if score >= rule["min_naturalness"]) / len(rated) if rated else 0.0
            ),
            "naturalness_median": statistics.median(rated) if rated else None,
        }
    k1 = evidence["dream_k1"]
    checks = {
        "terminal": k1["terminal"] == k1["jobs"] and k1["technical_failures"] == 0,
        "script": k1["script_pass_share"] >= rule["min_script_pass_share"],
        "naturalness": k1["naturalness_share"] >= rule["min_naturalness_share"],
    }
    evidence["criteria"] = checks
    return AnalysisDecision(
        gate="G0",
        outcome="pass" if all(checks.values()) else "fail",
        rule=rule,
        evidence=evidence,
        run_manifest_sha256=str(manifest["manifest_sha256"]),
        model_profile_id=DREAM_7B.profile_id,
        decided_by=decided_by,
        decided_at=decided_at,
        notes=notes,
    )
