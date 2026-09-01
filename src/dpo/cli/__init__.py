"""The dpo command line: JSON in, JSON out, fail closed.

Every command prints exactly one JSON document. Exit codes: 0 success, 2
domain/usage error, 3 blocked pending an external operation (real GPU
training, live model scoring). Commands that publish take content-addressed
artifact ids as inputs and reject any input from a different contract.

Each command group lives in its own module; this package root only
assembles the parser and dispatches, so `ls src/dpo/cli/` reads as the
command list itself.
"""

from __future__ import annotations

import argparse

from dpo.cli._shared import DOMAIN_ERRORS, Handler
from dpo.cli.annotation import _annotation_export_tasks, _annotation_ingest, _annotation_serve
from dpo.cli.artifact import _artifact_gc, _artifact_rebuild_index, _artifact_trace, _artifact_verify
from dpo.cli.canary import _canary_run
from dpo.cli.candidates import _candidates_dedup, _candidates_generate
from dpo.cli.console import register as register_console
from dpo.cli.contract import _contract_lock, _contract_validate
from dpo.cli.corpus import _corpus_ingest, _corpus_lock_splits
from dpo.cli.report import _report_analyze, _report_show
from dpo.cli.select import _select_run
from dpo.cli.session import (
    DEFAULT_CONTRACT as SESSION_DEFAULT_CONTRACT,
)
from dpo.cli.session import _session_link_masks, _session_scaffold, _session_serve, _session_validate
from dpo.cli.stage import _stage_list
from dpo.cli.study import _study_export, _study_ingest, _study_serve
from dpo.cli.train import _train_run
from dpo.cli.views import _views_derive
from dpo.contracts.study_contract import AUDIO_PRESENTATIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dpo", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("contract", help="validate or lock the study contract")
    contract_actions = contract.add_subparsers(dest="action", required=True)
    validate = contract_actions.add_parser("validate")
    validate.add_argument("--contract", required=True)
    validate.set_defaults(handler=_contract_validate)
    lock = contract_actions.add_parser("lock")
    lock.add_argument("--contract", required=True)
    lock.add_argument("--out", required=True)
    lock.set_defaults(handler=_contract_lock)

    canary = commands.add_parser("canary", help="run the offline end-to-end canary")
    canary_actions = canary.add_subparsers(dest="action", required=True)
    canary_run = canary_actions.add_parser("run")
    canary_run.add_argument("--workspace", required=True)
    canary_run.add_argument("--contract", required=True)
    canary_run.set_defaults(handler=_canary_run)

    artifact = commands.add_parser("artifact", help="verify, trace, and collect artifacts")
    artifact_actions = artifact.add_subparsers(dest="action", required=True)
    verify = artifact_actions.add_parser("verify")
    verify.add_argument("--workspace", required=True)
    group = verify.add_mutually_exclusive_group(required=True)
    group.add_argument("--artifact-id")
    group.add_argument("--all", action="store_true")
    verify.set_defaults(handler=_artifact_verify)
    trace = artifact_actions.add_parser("trace")
    trace.add_argument("--workspace", required=True)
    trace.add_argument("--artifact-id", required=True)
    trace.set_defaults(handler=_artifact_trace)
    rebuild = artifact_actions.add_parser("rebuild-index")
    rebuild.add_argument("--workspace", required=True)
    rebuild.set_defaults(handler=_artifact_rebuild_index)
    gc = artifact_actions.add_parser("gc")
    gc.add_argument("--workspace", required=True)
    gc.add_argument("--lock", action="append")
    gc.add_argument("--execute", action="store_true")
    gc.set_defaults(handler=_artifact_gc)

    stage = commands.add_parser("stage", help="show the stage registry and lineage")
    stage_actions = stage.add_subparsers(dest="action", required=True)
    stage_list = stage_actions.add_parser("list")
    stage_list.set_defaults(handler=_stage_list)

    corpus = commands.add_parser("corpus", help="ingest clips and lock the immutable splits")
    corpus_actions = corpus.add_subparsers(dest="action", required=True)
    ingest = corpus_actions.add_parser("ingest")
    ingest.add_argument("--workspace", required=True)
    ingest.add_argument("--contract", required=True)
    ingest.add_argument("--input", required=True)
    ingest.set_defaults(handler=_corpus_ingest)
    lock_splits = corpus_actions.add_parser("lock-splits")
    lock_splits.add_argument("--workspace", required=True)
    lock_splits.add_argument("--contract", required=True)
    lock_splits.add_argument("--artifact-id", action="append", required=True)
    lock_splits.set_defaults(handler=_corpus_lock_splits)

    candidates = commands.add_parser(
        "candidates", help="generate, audit, pair, and freeze the offline C0 candidate pool"
    )
    candidates_actions = candidates.add_subparsers(dest="action", required=True)
    generate = candidates_actions.add_parser("generate")
    generate.add_argument("--workspace", required=True)
    generate.add_argument("--contract", required=True)
    generate.add_argument("--artifact-id", action="append", required=True)
    generate.add_argument("--track", required=True, choices=["visual", "audio"])
    generate.add_argument("--split", required=True, choices=["train", "validation"])
    generate.add_argument("--dataset-version", required=True)
    generate.add_argument("--backend-config")
    generate.add_argument("--media-dir")
    generate.set_defaults(handler=_candidates_generate)
    dedup = candidates_actions.add_parser("dedup")
    dedup.add_argument("--workspace", required=True)
    dedup.add_argument("--contract", required=True)
    dedup.add_argument("--artifact-id", action="append", required=True)
    dedup.add_argument("--dataset-version", required=True)
    dedup.set_defaults(handler=_candidates_dedup)

    annotation = commands.add_parser(
        "annotation", help="export collection tasks, serve the UI, ingest responses"
    )
    annotation_actions = annotation.add_subparsers(dest="action", required=True)
    export_tasks = annotation_actions.add_parser("export-tasks")
    export_tasks.add_argument("--workspace", required=True)
    export_tasks.add_argument("--contract", required=True)
    export_tasks.add_argument("--artifact-id", action="append", required=True)
    export_tasks.add_argument("--out-tasks", required=True)
    export_tasks.add_argument("--out-answers", required=True)
    export_tasks.add_argument(
        "--presentation",
        choices=sorted(AUDIO_PRESENTATIONS),
        help="export every audio-track clip under this presentation (one between-subjects condition)",
    )
    export_tasks.set_defaults(handler=_annotation_export_tasks)
    annotation_ingest = annotation_actions.add_parser("ingest")
    annotation_ingest.add_argument("--workspace", required=True)
    annotation_ingest.add_argument("--contract", required=True)
    annotation_ingest.add_argument("--artifact-id", action="append", required=True)
    annotation_ingest.add_argument("--split", required=True, choices=["train", "validation"])
    annotation_ingest.add_argument("--tasks", required=True)
    annotation_ingest.add_argument("--answers", required=True)
    annotation_ingest.add_argument("--responses", action="append", required=True)
    annotation_ingest.set_defaults(handler=_annotation_ingest)
    serve = annotation_actions.add_parser("serve")
    serve.add_argument("--tasks", required=True)
    serve.add_argument("--media-dir", required=True)
    serve.add_argument("--out", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(handler=_annotation_serve)

    views = commands.add_parser("views", help="derive every training view of one track")
    views_actions = views.add_subparsers(dest="action", required=True)
    derive = views_actions.add_parser("derive")
    derive.add_argument("--workspace", required=True)
    derive.add_argument("--contract", required=True)
    derive.add_argument("--artifact-id", action="append", required=True)
    derive.add_argument("--track", required=True, choices=["visual", "audio"])
    derive.set_defaults(handler=_views_derive)

    train = commands.add_parser("train", help="run and publish every experiment-matrix cell")
    train_actions = train.add_subparsers(dest="action", required=True)
    train_run = train_actions.add_parser("run")
    train_run.add_argument("--workspace", required=True)
    train_run.add_argument("--contract", required=True)
    train_run.add_argument("--artifact-id", action="append", required=True)
    train_run.add_argument("--checkpoint-dir", required=True)
    train_run.add_argument("--backend-config", action="append")
    train_run.add_argument("--media-dir")
    train_run.set_defaults(handler=_train_run)

    select = commands.add_parser("select", help="score every variant, select winners, and lock")
    select_actions = select.add_subparsers(dest="action", required=True)
    select_run = select_actions.add_parser("run")
    select_run.add_argument("--workspace", required=True)
    select_run.add_argument("--contract", required=True)
    select_run.add_argument("--artifact-id", action="append", required=True)
    select_run.add_argument("--checkpoint-dir", required=True)
    select_run.add_argument("--backend-config", action="append")
    select_run.add_argument("--media-dir")
    select_run.set_defaults(handler=_select_run)

    report = commands.add_parser("report", help="show published validation/selection/lock reports")
    report_actions = report.add_subparsers(dest="action", required=True)
    report_show = report_actions.add_parser("show")
    report_show.add_argument("--workspace", required=True)
    report_show.set_defaults(handler=_report_show)
    report_analyze = report_actions.add_parser("analyze")
    report_analyze.add_argument("--workspace", required=True)
    report_analyze.add_argument("--contract", required=True)
    report_analyze.add_argument("--artifact-id", action="append", required=True)
    report_analyze.set_defaults(handler=_report_analyze)

    study = commands.add_parser(
        "study", help="export the human-study stimuli, serve the study, ingest its responses"
    )
    study_actions = study.add_subparsers(dest="action", required=True)
    study_export = study_actions.add_parser("export")
    study_export.add_argument("--workspace", required=True)
    study_export.add_argument("--contract", required=True)
    study_export.add_argument("--artifact-id", action="append", required=True)
    study_export.add_argument("--track", required=True)
    study_export.add_argument("--checkpoint-dir", required=True)
    study_export.add_argument("--backend-config", action="append")
    study_export.add_argument("--media-dir")
    study_export.set_defaults(handler=_study_export)
    study_serve = study_actions.add_parser("serve")
    study_serve.add_argument("--export", required=True, help="a published dpo.study-export/v1 document")
    study_serve.add_argument("--media-dir", required=True)
    study_serve.add_argument("--out", required=True)
    study_serve.add_argument("--host", default="127.0.0.1")
    study_serve.add_argument("--port", type=int, default=8776)
    study_serve.set_defaults(handler=_study_serve)
    study_ingest = study_actions.add_parser("ingest")
    study_ingest.add_argument("--workspace", required=True)
    study_ingest.add_argument("--contract", required=True)
    study_ingest.add_argument("--artifact-id", action="append", required=True)
    study_ingest.add_argument(
        "--responses", action="append", required=True, help="one saved responses-<participant>.json"
    )
    study_ingest.set_defaults(handler=_study_ingest)

    session = commands.add_parser("session", help="validate, scaffold, and serve a caption session")
    session_actions = session.add_subparsers(dest="action", required=True)
    session_validate = session_actions.add_parser("validate")
    session_validate.add_argument("--session", required=True)
    session_validate.set_defaults(handler=_session_validate)
    session_scaffold = session_actions.add_parser("scaffold")
    session_scaffold.add_argument("--media-dir", required=True)
    session_scaffold.add_argument("--out", required=True)
    session_scaffold.add_argument("--clips", nargs="*", help="clip ids; default: every mp4 under --media-dir")
    session_scaffold.add_argument(
        "--mask-links",
        help="a dpo.caption-mask-link/v1 manifest from `session link-masks`; "
        "fills each shot's sources from it (audio on a shaped clip, visual on a control clip)",
    )
    session_scaffold.set_defaults(handler=_session_scaffold)
    session_link_masks = session_actions.add_parser(
        "link-masks", help="derive caption-parameter evidence from Sa2VA masks and audio tags"
    )
    session_link_masks.add_argument(
        "--mask-root", required=True, help="directory containing audio/ and visual/"
    )
    session_link_masks.add_argument(
        "--tidy-data", required=True, help="CSV with final_labels and top_level_parent_name"
    )
    session_link_masks.add_argument(
        "--ontology", help="AudioSet ontology JSON; default: ontology.json beside --tidy-data"
    )
    session_link_masks.add_argument("--out", required=True, help="output dpo.caption-mask-link/v1 JSON")
    session_link_masks.add_argument("--fps", type=float, default=60.0, help="source video frame rate")
    session_link_masks.add_argument(
        "--clips", nargs="*", help="optional clip ids; default: all discovered clips"
    )
    session_link_masks.set_defaults(handler=_session_link_masks)
    session_serve = session_actions.add_parser("serve")
    session_serve.add_argument("--session", required=True)
    session_serve.add_argument("--media-dir", required=True)
    session_serve.add_argument("--out", required=True)
    session_serve.add_argument("--writer", choices=["template", "gemma"], default="template")
    session_serve.add_argument("--backend-config", help="gemma: the audio backend config")
    session_serve.add_argument(
        "--checkpoint", help="gemma: a LoRA checkpoint directory; default: the base model"
    )
    session_serve.add_argument(
        "--contract", default=SESSION_DEFAULT_CONTRACT, help="gemma: the study contract for [tracks.audio]"
    )
    session_serve.add_argument("--host", default="127.0.0.1")
    session_serve.add_argument("--port", type=int, default=8777)
    session_serve.set_defaults(handler=_session_serve)

    # The console instrument (docs/v2-console) is a separate command over a separate
    # package, so whichever of the two is not adopted can be removed whole.
    register_console(commands)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler: Handler = arguments.handler
    try:
        return handler(arguments)
    except DOMAIN_ERRORS as exc:
        parser.exit(2, f"dpo: error: {exc}\n")
        return 2  # pragma: no cover
