"""``python -m diff`` — the generator's command line (plan §10, Stage 1).

Run from the repository root. Commands marked [generator] need the locked
generator environment and run as::

    PYTHONPATH=src uv run --project envs/diff-generator --frozen python -m diff <command> …

Every other command runs in the repository's own environment
(``uv run python -m diff …``). Each handler imports what it needs, so
``--help`` and the server-side commands never import torch or transformers.
There is deliberately no console script: a ``diff`` executable in the venv
would shadow the system ``diff``.

G0 comes first (follow-up §7): ``g0 plan`` → ``plan freeze`` → ``generate`` →
``g0 report`` (stops at the blind packet) → a person rates → ``g0 decide``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

GENERATOR_LOCK = Path("envs/diff-generator/uv.lock")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sources(path: Path) -> Any:
    from diff.caption_sources import SourceError, parse_sources

    try:
        return parse_sources(_read_json(path))
    except SourceError as exc:
        raise SystemExit("invalid sources:\n  " + "\n  ".join(exc.problems)) from exc


def _model_dirs(args: argparse.Namespace) -> dict[str, Path]:
    from diff.model_profile import DREAM_7B, QWEN_7B

    dirs = {DREAM_7B.profile_id: args.dream_dir, QWEN_7B.profile_id: args.qwen_dir}
    return {key: value for key, value in dirs.items() if value is not None}


def sources_validate(args: argparse.Namespace) -> int:
    from diff.caption_sources import bundle_problems

    source_set = _sources(args.sources)
    units: dict[tuple[int, str, str], int] = {}
    for source in source_set.sources:
        unit = (source.phase, source.unit_key, source.material)
        units[unit] = units.get(unit, 0) + 1
    for (phase, key, material), count in sorted(units.items()):
        print(f"phase {phase} {key} ({material}): {count} cues")
    ineligible = [problem for source in source_set.sources for problem in bundle_problems(source)]
    print(f"{len(source_set.sources)} sources; {len(ineligible)} bundle-eligibility problems")
    for problem in ineligible[: args.show]:
        print(f"  {problem}")
    return 0


def sources_import(args: argparse.Namespace) -> int:
    from diff.caption_sources import import_regen_document, import_viewing_manifest

    document = _read_json(args.input)
    importer = import_viewing_manifest if args.kind == "viewing" else import_regen_document
    imported = importer(document, args.material)
    _write_json(args.out, imported.record(), force=args.force)
    print(f"wrote {len(imported.sources)} draft {args.material} sources to {args.out} (unreviewed)")
    return 0


def plan_build(args: argparse.Namespace) -> int:
    from diff.job_plan import PlanError, build_phase1_plan, build_phase2_plan

    source_set = _sources(args.sources)
    try:
        if args.phase == 1:
            plan = build_phase1_plan(source_set.sources, tuple(args.conditions.split(",")), args.purpose)
        else:
            decision = _read_json(args.g0_decision) if args.g0_decision else None
            plan = build_phase2_plan(source_set.sources, args.condition, args.lattice, args.purpose, decision)
    except PlanError as exc:
        raise SystemExit("cannot plan:\n  " + "\n  ".join(exc.problems)) from exc
    _write_json(args.out, plan.record(), force=args.force)
    print(f"{plan.plan_id}: {len(plan.jobs)} jobs → {args.out}")
    return 0


def _tokenizer(model_dir: Path, profile_id: str) -> tuple[Any, dict[str, Any]]:
    """A verified tokenizer and the record of the files it came from. [generator]"""
    from diff.model_profile import profile, verify_snapshot

    model = profile(profile_id)
    record = verify_snapshot(model_dir, model, weights=False)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=bool(model.remote_code), local_files_only=True
    )
    imported = _remote_tokenizer_file(tokenizer, model)
    return tokenizer, {
        "profile_id": profile_id,
        "files": {name: record["files"][name] for name in model.tokenizer_files},
        "imported_tokenizer_code": imported,
    }


def _remote_tokenizer_file(tokenizer: Any, model: Any) -> dict[str, str]:
    """Hash the tokenizer class file Python actually imported when it is remote code."""
    import inspect

    from diff.model_profile import file_sha256

    path = Path(inspect.getfile(type(tokenizer)))
    if path.name not in model.remote_code:
        return {}
    actual = file_sha256(path)
    if actual != model.files[path.name]:
        raise SystemExit(f"{path.name}: the imported copy ({actual}) is not the pinned file")
    return {path.name: actual}


def plan_freeze(args: argparse.Namespace) -> int:
    from diff.job_plan import PlanError, freeze_plan, plan_from_record
    from diff.model_runtime import chat_token_ids

    plan = plan_from_record(_read_json(args.plan))
    dirs = _model_dirs(args)
    needed = sorted({job.profile_id for job in plan.jobs})
    missing = [profile_id for profile_id in needed if profile_id not in dirs]
    if missing:
        raise SystemExit(f"pass a model directory for {missing}")
    tokenizers = {profile_id: _tokenizer(dirs[profile_id], profile_id) for profile_id in needed}
    try:
        frozen = freeze_plan(
            plan,
            lambda job: chat_token_ids(tokenizers[job.profile_id][0], job.prompt),
            {profile_id: record for profile_id, (_, record) in tokenizers.items()},
        )
    except PlanError as exc:
        raise SystemExit("refused at freeze:\n  " + "\n  ".join(exc.problems)) from exc
    _write_json(args.out, frozen.record(), force=args.force)
    lengths = [job.token_count for job in frozen.jobs]
    print(f"{plan.plan_id} frozen ({frozen.frozen_sha256[:12]}): prompt tokens {min(lengths)}–{max(lengths)}")
    return 0


def plan_tokens(args: argparse.Namespace) -> int:
    from diff.job_plan import source_token_report
    from diff.model_profile import DREAM_7B

    tokenizer, _ = _tokenizer(args.dream_dir, DREAM_7B.profile_id)
    report = source_token_report(
        _sources(args.sources).sources, lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def verification_record_path(model_dir: Path) -> Path:
    """``<dir>.verification.json`` beside the snapshot (``with_suffix`` would eat "2.5-…" in a name)."""
    return model_dir.parent / f"{model_dir.name}.verification.json"


def model_fetch(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    from diff.model_profile import profile, verify_snapshot

    model = profile(args.profile)
    snapshot_download(
        model.repo_id,
        revision=model.revision,
        local_dir=str(args.dest),
        allow_patterns=[*model.files, *model.weights],
        token=False,
    )
    record = verify_snapshot(args.dest, model, weights=True)
    _write_json(verification_record_path(args.dest), record, force=True)
    print(f"{model.profile_id}: fetched and verified into {args.dest}")
    return 0


def model_verify(args: argparse.Namespace) -> int:
    from diff.model_profile import ProfileMismatch, profile, verify_snapshot

    model = profile(args.profile)
    try:
        record = verify_snapshot(args.model_dir, model, weights=not args.no_weights)
    except ProfileMismatch as exc:
        raise SystemExit("snapshot does not match its profile:\n  " + "\n  ".join(exc.problems)) from exc
    if args.record:
        _write_json(args.record, record, force=True)
    print(f"{model.profile_id}: {len(record['files'])} files and {len(record['weights'])} shards match")
    return 0


def generate(args: argparse.Namespace) -> int:
    from diff.generation_worker import WorkerCommand, WorkerStartupError, run_plan
    from diff.job_plan import frozen_plan_from_record
    from diff.model_profile import profile, verify_snapshot

    frozen = frozen_plan_from_record(_read_json(args.plan))
    dirs = _model_dirs(args)
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    commands, verifications = {}, {}
    for profile_id in sorted({job.job.profile_id for job in frozen.jobs}):
        if profile_id not in dirs:
            raise SystemExit(f"pass a model directory for {profile_id}")
        print(f"verifying {profile_id} (hashing every weight shard) …", flush=True)
        verifications[profile_id] = record = verify_snapshot(
            dirs[profile_id], profile(profile_id), weights=True
        )
        slug = profile_id.replace("@", "-")
        record_path = run_dir / f"verification-{slug}.json"
        _write_json(record_path, record, force=True)
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_MODULES_CACHE": str((run_dir / "hf-modules" / uuid.uuid4().hex).resolve()),
            "TOKENIZERS_PARALLELISM": "false",
        }
        commands[profile_id] = WorkerCommand(
            argv=(
                sys.executable,
                "-m",
                "diff",
                "worker",
                "--profile",
                profile_id,
                "--model-dir",
                str(dirs[profile_id]),
                "--verification",
                str(record_path),
                "--device",
                args.device,
                "--lock",
                str(GENERATOR_LOCK),
            ),
            env=env,
            cwd=Path.cwd(),
            log_path=run_dir / f"worker-{slug}.log",
        )
    _write_json(run_dir / "plan.json", frozen.record(), force=True)
    try:
        summary = run_plan(frozen, run_dir, commands, verifications, startup_timeout=args.startup_timeout)
    except WorkerStartupError as exc:
        raise SystemExit(f"generation stopped: {exc}") from exc
    print(json.dumps(summary, indent=2))
    return 0


def worker(args: argparse.Namespace) -> int:
    """[generator, internal] One model behind the JSON-lines protocol on the original stdout."""
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)  # anything a library prints goes to the log, never into the protocol
    from diff.generation_worker import serve
    from diff.model_profile import profile

    model = profile(args.profile)
    verification = _read_json(args.verification)
    if model.runner == "dream":
        from diff import dream_runner

        def load_dream() -> tuple[Any, dict[str, Any]]:
            loaded = dream_runner.load(args.model_dir, verification, device=args.device, lock_path=args.lock)
            return loaded, dream_runner.warm_up(loaded)

        return serve(load_dream, dream_runner.run_job, protocol, sys.stdin)
    from diff import qwen_runner

    def load_qwen() -> tuple[Any, dict[str, Any]]:
        loaded = qwen_runner.load(args.model_dir, verification, device=args.device, lock_path=args.lock)
        return loaded, qwen_runner.warm_up(loaded)

    return serve(load_qwen, qwen_runner.run_job, protocol, sys.stdin)


def checks_run(args: argparse.Namespace) -> int:
    from diff.generation_worker import read_attempts
    from diff.job_plan import frozen_plan_from_record
    from diff.output_checks import check_output, output_from_record

    frozen = frozen_plan_from_record(_read_json(args.plan))
    source_set = _sources(args.sources)
    by_id = source_set.by_id()
    latest: dict[str, dict[str, Any]] = {}
    for attempt in read_attempts(args.run_dir / "attempts.jsonl"):
        latest[attempt["job_id"]] = attempt
    lexicon = sorted({term for s in source_set.sources for term in s.evidence_terms()["sources"]})
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"checked": 0, "excluded": 0, "not_completed": 0, "failures": {}, "flags": {}}
    for entry in frozen.jobs:
        job, final = entry.job, latest.get(entry.job.job_id)
        if final is None or final["status"] != "completed":
            summary["not_completed"] += 1
            continue
        source = by_id.get(job.source_id)
        if source is None:
            raise SystemExit(f"{job.job_id}: its source {job.source_id} is not in {args.sources}")
        result = check_output(
            source,
            output_from_record(final["output"]),
            template_phase=job.phase,
            cell=job.cell,
            lexicon=lexicon,
        )
        rows.append({"job_id": job.job_id, "attempt_id": final["attempt_id"], **result.record()})
        summary["checked"] += 1
        summary["excluded"] += int(result.excluded)
        for kind, findings in (("failures", result.failures), ("flags", result.flags)):
            for finding in findings:
                summary[kind][finding.code] = summary[kind].get(finding.code, 0) + 1
    out = args.out or args.run_dir / "checks.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def g0_plan(args: argparse.Namespace) -> int:
    from diff.feasibility_gate import build_g0_plan
    from diff.job_plan import PlanError

    try:
        plan = build_g0_plan(_sources(args.sources).sources, smoke=args.smoke)
    except PlanError as exc:
        raise SystemExit("cannot plan G0:\n  " + "\n  ".join(exc.problems)) from exc
    _write_json(args.out, plan.record(), force=args.force)
    print(f"{plan.plan_id} ({plan.purpose}): {len(plan.jobs)} jobs → {args.out}")
    return 0


def g0_report(args: argparse.Namespace) -> int:
    """Automatic report and blind packet; stops there, because the rating needs a person."""
    from diff import feasibility_gate as gate
    from diff.job_plan import frozen_plan_from_record

    frozen = frozen_plan_from_record(_read_json(args.plan))
    source_set = _sources(args.sources)
    latest = gate.latest_attempts(args.run_dir)
    manifest = gate.run_manifest(frozen, args.run_dir, source_set.sources)
    report = gate.g0_report(frozen, latest, source_set.by_id())
    rows, key = gate.blind_packet(frozen, latest, source_set.by_id(), manifest["manifest_sha256"])
    out: Path = args.out_dir
    _write_json(out / "researcher" / "run-manifest.json", manifest, force=args.force)
    _write_json(out / "researcher" / "report.json", report, force=args.force)
    _write_json(out / "researcher" / "key.json", key, force=args.force)
    _write_text(out / "rater" / "packet.csv", gate.packet_csv(rows), force=args.force)
    _write_text(out / "rater" / "ratings.csv", gate.ratings_template(rows), force=args.force)
    for condition, stats in report["conditions"].items():
        print(
            f"{condition}: {stats['completed']}/{stats['jobs']} completed, script pass "
            f"{stats['script_pass_share']:.0%}, incomplete {stats['incomplete']}, "
            f"technical {stats['technical_failures']}"
        )
    print(f"blind packet: {len(rows)} outputs → {out / 'rater'}; key and manifest → {out / 'researcher'}")
    print(
        "stop here: one fluent Korean reader rates naturalness 1–5 in rater/ratings.csv, "
        "then run `g0 decide`."
    )
    return 0


def g0_decide(args: argparse.Namespace) -> int:
    from diff import feasibility_gate as gate

    out: Path = args.out_dir
    manifest = _read_json(out / "researcher" / "run-manifest.json")
    report = _read_json(out / "researcher" / "report.json")
    key = _read_json(out / "researcher" / "key.json")
    try:
        ratings = gate.read_ratings(args.ratings.read_text(encoding="utf-8"), key)
        decision = gate.decide(
            report,
            key,
            ratings,
            manifest,
            decided_by=args.decided_by,
            decided_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            notes=args.notes,
        )
    except ValueError as exc:
        raise SystemExit(f"cannot decide G0: {exc}") from exc
    path = out / "researcher" / f"decision-{decision.decision_id}.json"
    _write_json(path, decision.record())
    print(f"G0 {decision.outcome}: {json.dumps(decision.evidence['criteria'])} → {path}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m diff", description=__doc__.split("\n\n")[0])
    commands = root.add_subparsers(dest="command", required=True)

    sources = commands.add_parser("sources", help="validate or import caption sources").add_subparsers(
        dest="action", required=True
    )
    validate = sources.add_parser("validate", help="validate a diff.caption-sources/v1 file")
    validate.add_argument("--sources", type=Path, required=True)
    validate.add_argument("--show", type=int, default=20, help="eligibility problems to print")
    validate.set_defaults(handler=sources_validate)
    for kind, help_text in (
        ("viewing", "regen viewing manifest → phase-2 drafts"),
        ("regen", "regen document → phase-1 drafts"),
    ):
        imported = sources.add_parser(f"import-{kind}", help=help_text)
        imported.add_argument("--input", type=Path, required=True)
        imported.add_argument("--out", type=Path, required=True)
        imported.add_argument("--material", choices=("development", "participant"), default="participant")
        imported.add_argument("--force", action="store_true")
        imported.set_defaults(handler=sources_import, kind=kind)

    plans = commands.add_parser("plan", help="build, freeze or measure generation plans").add_subparsers(
        dest="action", required=True
    )
    build = plans.add_parser("build", help="cue × condition (phase 1) or cue × cell (phase 2) jobs")
    build.add_argument("--phase", type=int, choices=(1, 2), required=True)
    build.add_argument("--sources", type=Path, required=True)
    build.add_argument("--purpose", choices=("development", "participant"), required=True)
    build.add_argument("--conditions", default="original,dream_k1,dream_k2,dream_k4")
    build.add_argument("--condition", default="dream_k1", help="phase 2: the one decoding condition")
    build.add_argument("--lattice", default="3x3")
    build.add_argument("--g0-decision", type=Path, help="phase 2: a passing G0 decision record (AC-20)")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--force", action="store_true")
    build.set_defaults(handler=plan_build)
    freeze = plans.add_parser("freeze", help="[generator] add prompt token hashes and apply the guards")
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--dream-dir", type=Path)
    freeze.add_argument("--qwen-dir", type=Path)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--force", action="store_true")
    freeze.set_defaults(handler=plan_freeze)
    tokens = plans.add_parser("tokens", help="[generator] CV-08 source token counts against the canvas")
    tokens.add_argument("--sources", type=Path, required=True)
    tokens.add_argument("--dream-dir", type=Path, required=True)
    tokens.set_defaults(handler=plan_tokens)

    models = commands.add_parser("model", help="fetch or verify pinned model snapshots").add_subparsers(
        dest="action", required=True
    )
    fetch = models.add_parser("fetch", help="[generator] download the pinned revision and verify it")
    fetch.add_argument("--profile", required=True)
    fetch.add_argument("--dest", type=Path, required=True)
    fetch.set_defaults(handler=model_fetch)
    verify = models.add_parser("verify", help="hash a snapshot against its profile")
    verify.add_argument("--profile", required=True)
    verify.add_argument("--model-dir", type=Path, required=True)
    verify.add_argument("--no-weights", action="store_true", help="skip hashing the weight shards")
    verify.add_argument("--record", type=Path)
    verify.set_defaults(handler=model_verify)

    run = commands.add_parser("generate", help="[generator, GPU] run a frozen plan under supervision")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--dream-dir", type=Path)
    run.add_argument("--qwen-dir", type=Path)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--startup-timeout", type=float, default=1800.0)
    run.set_defaults(handler=generate)

    child = commands.add_parser(
        "worker", help="[generator, internal] one model behind the JSON-lines protocol"
    )
    child.add_argument("--profile", required=True)
    child.add_argument("--model-dir", type=Path, required=True)
    child.add_argument("--verification", type=Path, required=True)
    child.add_argument("--device", default="cuda:0")
    child.add_argument("--lock", type=Path)
    child.set_defaults(handler=worker)

    checks = commands.add_parser("checks", help="output checks").add_subparsers(dest="action", required=True)
    check = checks.add_parser("run", help="fail/flag/measure every completed attempt of a run")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--run-dir", type=Path, required=True)
    check.add_argument("--sources", type=Path, required=True)
    check.add_argument("--out", type=Path)
    check.set_defaults(handler=checks_run)

    g0 = commands.add_parser("g0", help="G0 Korean feasibility gate").add_subparsers(
        dest="action", required=True
    )
    g0_build = g0.add_parser("plan", help="G0 jobs from development material (dream_k1/k2/k4, qwen)")
    g0_build.add_argument("--sources", type=Path, required=True)
    g0_build.add_argument(
        "--smoke", action="store_true", help="synthetic material: a pipeline smoke, never a G0"
    )
    g0_build.add_argument("--out", type=Path, required=True)
    g0_build.add_argument("--force", action="store_true")
    g0_build.set_defaults(handler=g0_plan)
    g0_out = g0.add_parser("report", help="automatic report and blind packet (stops for the human rating)")
    g0_out.add_argument("--plan", type=Path, required=True)
    g0_out.add_argument("--run-dir", type=Path, required=True)
    g0_out.add_argument("--sources", type=Path, required=True)
    g0_out.add_argument("--out-dir", type=Path, required=True)
    g0_out.add_argument("--force", action="store_true")
    g0_out.set_defaults(handler=g0_report)
    g0_rule = g0.add_parser("decide", help="apply the G0 rule to the returned ratings; record the decision")
    g0_rule.add_argument("--out-dir", type=Path, required=True)
    g0_rule.add_argument("--ratings", type=Path, required=True)
    g0_rule.add_argument("--decided-by", required=True)
    g0_rule.add_argument("--notes", default="")
    g0_rule.set_defaults(handler=g0_decide)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
