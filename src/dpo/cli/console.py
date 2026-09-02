"""`dpo console`: preprocess, scaffold, validate, and serve the console instrument.

The commands follow the order the pipeline runs (docs/v2-console/spec-system.md §2):

* ``preprocess`` reads the Sa2VA mask tree, cuts shots on visual composition,
  groups audio labels by mask agreement, and writes ``r_g`` and ``p_g`` per
  shot per group;
* ``scaffold`` turns that manifest into a session document under a named
  calibration, leaving what only a researcher can supply empty;
* ``validate`` refuses a document that is not yet authored, naming the path;
* ``serve`` runs the kiosk.

``validate`` exists as its own command because a scaffold never validates. The
raw and default-policy captions are authored, and ``c_g`` and ``e_g`` cannot be
derived from a mask tree at all, so a scaffold that passed would let an
unauthored session reach a participant.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dpo.caption.ontology import OntologyError, TagRecord, load_tags
from dpo.caption.writer import CacheMismatch, CaptionWriter, ShotMedia
from dpo.cli._shared import _emit
from dpo.console.config import Calibration, Configuration
from dpo.console.document import (
    CONSOLE_SCHEMA,
    ConsoleDocumentError,
    configuration_of,
    load_console_document,
)
from dpo.console.masks import (
    MANIFEST_SCHEMA,
    MaskReadError,
    derive_manifest,
    write_manifest,
)

DEFAULT_CONTRACT = "configs/study/street-audio.toml"
# What crosses from the manifest into the document. The manifest's own
# ``review_required`` and ``frames`` stay behind: they are provenance for the
# researcher, not configuration the instrument reads.
SOURCE_FIELDS = ("id", "labels", "prose", "share", "presence", "confidence", "energy")


class ConsoleUsageError(ValueError):
    """The command line, not the document, is what is wrong."""


def _calibration(arguments: argparse.Namespace) -> Calibration:
    """The calibration in force: defaults, overridden by the named flags."""
    overrides = {
        name: getattr(arguments, name)
        for name in ("half_saturation", "iou_threshold", "cut_threshold", "minimum_shot_ms")
        if getattr(arguments, name, None) is not None
    }
    return Calibration(**overrides)


def _console_preprocess(arguments: argparse.Namespace) -> int:
    tags: Mapping[str, TagRecord] = {}
    if arguments.tidy_data:
        tidy = Path(arguments.tidy_data)
        ontology = Path(arguments.ontology) if arguments.ontology else tidy.with_name("ontology.json")
        try:
            tags = load_tags(tidy, ontology)
        except OntologyError as exc:
            _emit({"status": "invalid", "command": "console preprocess", "error": str(exc)})
            return 2
    elif arguments.provisional_salience:
        _emit(
            {
                "status": "error",
                "command": "console preprocess",
                "error": "--provisional-salience needs --tidy-data for the tag multiplicities",
            }
        )
        return 2
    root = Path(arguments.mask_root)
    clips = list(arguments.clips or [])
    if not clips:
        audio = root / "audio"
        clips = sorted(path.name for path in audio.iterdir() if path.is_dir()) if audio.is_dir() else []
    if not clips:
        _emit({"status": "error", "error": f"no clips named and none found under {root / 'audio'}"})
        return 2
    try:
        manifest = derive_manifest(
            root,
            clips,
            calibration=_calibration(arguments),
            fps=float(arguments.fps),
            downsample=int(arguments.downsample),
            tags=tags,
            provisional_salience=bool(arguments.provisional_salience),
            cache_dir=Path(arguments.cache_dir) if arguments.cache_dir else None,
        )
    except (MaskReadError, ValueError) as exc:
        _emit({"status": "invalid", "command": "console preprocess", "error": str(exc)})
        return 2
    out = write_manifest(manifest, arguments.out)
    _emit(
        {
            "status": "preprocessed",
            "out": str(out),
            "clips": len(manifest["clips"]),
            "shots": sum(len(clip["shots"]) for clip in manifest["clips"].values()),
            "sources": sum(
                len(clip["shots"][0]["sources"]) for clip in manifest["clips"].values() if clip["shots"]
            ),
            "provisional_salience": manifest["provisional_salience"],
        }
    )
    return 0


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsoleUsageError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ConsoleUsageError(f"{path}: not a {MANIFEST_SCHEMA} manifest")
    return manifest


def _console_scaffold(arguments: argparse.Namespace) -> int:
    manifest_path = Path(arguments.manifest)
    try:
        manifest = _load_manifest(manifest_path)
    except ConsoleUsageError as exc:
        _emit({"status": "error", "command": "console scaffold", "error": str(exc)})
        return 2
    # The manifest says whether c_g and e_g were measured or filled from tag
    # multiplicity for a dry run; the document carries that into its stamp.
    provisional = bool(manifest["provisional_salience"])
    # The thresholds that cut the shots and grouped the labels are the
    # manifest's, not a flag's: r_g, p_g and the grouping were computed under
    # them, and the stamp has to say so. r0 alone is applied at serve time.
    source = manifest.get("source") or {}
    preprocessing = {
        name: source[name]
        for name in ("iou_threshold", "cut_threshold", "minimum_shot_ms", "stride_ms")
        if name in source
    }
    if arguments.half_saturation is not None:
        preprocessing["half_saturation"] = arguments.half_saturation
    configuration = Configuration(
        study_id=arguments.study_id,
        corpus_id=arguments.corpus_id,
        provisional_salience=provisional,
        calibration=Calibration(**preprocessing),
    )
    wanted = list(arguments.clips or manifest["clips"])
    missing = [clip_id for clip_id in wanted if clip_id not in manifest["clips"]]
    if missing:
        _emit(
            {
                "status": "error",
                "command": "console scaffold",
                "error": f"{manifest_path}: no preprocessing for clip {missing[0]!r}",
            }
        )
        return 2
    clips = []
    for clip_id in wanted:
        clip = manifest["clips"][clip_id]
        clips.append(
            {
                "clip_id": clip_id,
                "opening": clip["opening"],
                "shots": [
                    {
                        "shot_id": shot["shot_id"],
                        "start_ms": shot["start_ms"],
                        "end_ms": shot["end_ms"],
                        "raw_caption": shot["raw_caption"],
                        "default_caption": shot["default_caption"],
                        "sources": [
                            {key: source[key] for key in SOURCE_FIELDS} for source in shot["sources"]
                        ],
                    }
                    for shot in clip["shots"]
                ],
            }
        )
    document = {
        "schema": CONSOLE_SCHEMA,
        "session_id": Path(arguments.out).stem,
        "config": configuration.artifact(),
        "clips": clips,
    }
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Nothing here validates yet, and the reason is not an oversight:
    # w_g = c_g * e_g needs an acoustic measurement no mask supplies.
    authoring = (
        "author raw_caption and default_caption; confidence and energy are provisional, "
        "so this document is a dry run and its stamp says so: never recruit on it"
        if provisional
        else "author raw_caption, default_caption, and every source's confidence and energy"
    )
    _emit(
        {
            "status": "scaffolded",
            "out": str(out),
            "config_hash": configuration.hash,
            "provisional_salience": provisional,
            "clips": len(clips),
            "shots": sum(len(clip["shots"]) for clip in clips),
            "next": f"{authoring}, then: dpo console validate --session {out}",
        }
    )
    return 0


def _console_validate(arguments: argparse.Namespace) -> int:
    try:
        document = load_console_document(arguments.session)
    except ConsoleDocumentError as exc:
        _emit({"status": "invalid", "session": str(arguments.session), "error": str(exc)})
        return 2
    configuration = configuration_of(document)
    _emit(
        {
            "status": "valid",
            "session": str(arguments.session),
            "session_id": document["session_id"],
            "config_hash": configuration.hash,
            "provisional_salience": configuration.provisional_salience,
            "clips": len(document["clips"]),
            "shots": sum(len(clip["shots"]) for clip in document["clips"]),
        }
    )
    return 0


def _gemma_writer(
    arguments: argparse.Namespace, document: Mapping[str, Any]
) -> tuple[CaptionWriter, ShotMedia] | None:
    """The Gemma writer and its shot-media resolver, or None when blocked (exit 3).

    Gate on CUDA before loading anything, then build the adapter from the
    backend config and the contract's audio track. The instruction comes from
    this instrument's own :class:`RequestBuilder`, so the model is told about
    grain and about what is out of frame, not about a skeleton.
    """
    if not arguments.backend_config:
        raise ConsoleUsageError("--writer gemma requires --backend-config")
    import torch

    if not torch.cuda.is_available():
        _emit(
            {
                "status": "blocked_pending_external_operation",
                "command": "console serve",
                "gate": "the Gemma writer requires a CUDA device; use --writer template without one",
                "side_effects": False,
            }
        )
        return None
    from dpo.caption.media import shot_audio
    from dpo.caption.writer import GemmaWriter
    from dpo.console.requests import ConsoleTemplateWriter, RequestBuilder
    from dpo.contracts.study_contract import load_contract
    from dpo.models.gemma4.adapter import GemmaCaptionAdapter
    from dpo.models.gemma4.backend_config import load_config

    config = load_config(arguments.backend_config)
    contract = load_contract(arguments.contract)
    if config.model.media_inputs != "audio" or "audio" not in contract.tracks:
        raise ConsoleUsageError(
            "--writer gemma needs an audio backend config and a contract with [tracks.audio]"
        )
    adapter = GemmaCaptionAdapter(
        config=config,
        contract=contract.tracks["audio"],
        media_resolver=lambda reference: reference,
        adapter_dir=None if arguments.checkpoint is None else str(arguments.checkpoint),
    )
    builder = RequestBuilder(configuration_of(document))
    media_dir = Path(arguments.media_dir)
    cache_dir = Path(arguments.out) / "media-cache"

    def resolve(clip_id: str, shot: Mapping[str, Any]) -> Path:
        return shot_audio(media_dir, cache_dir, clip_id, int(shot["start_ms"]), int(shot["end_ms"]))

    writer = GemmaWriter(
        adapter,
        instruction=builder.instruction,
        fallback=ConsoleTemplateWriter(),
    )
    return writer, resolve


def _console_serve(arguments: argparse.Namespace) -> int:
    from dpo.console.app import run_console_app

    try:
        # One read of the document serves the writer, the status line and the app.
        document = load_console_document(Path(arguments.session))
        if arguments.writer == "gemma":
            built = _gemma_writer(arguments, document)
            if built is None:
                return 3
            writer, shot_media = built
        else:
            from dpo.console.requests import ConsoleTemplateWriter

            writer, shot_media = ConsoleTemplateWriter(), None
    except ConsoleUsageError as exc:
        _emit({"status": "error", "command": "console serve", "error": str(exc)})
        return 2
    except ConsoleDocumentError as exc:
        _emit({"status": "invalid", "session": str(arguments.session), "error": str(exc)})
        return 2
    # Said once, on the console the operator is watching: the stamp this run
    # writes on every line, and whether it is a dry run's.
    configuration = configuration_of(document)
    _emit(
        {
            "status": "serving",
            "command": "console serve",
            "session": str(arguments.session),
            "config_hash": configuration.hash,
            "provisional_salience": configuration.provisional_salience,
            "writer": arguments.writer,
            "url": f"http://{arguments.host}:{int(arguments.port)}/",
        }
    )
    try:
        run_console_app(
            document,
            arguments.media_dir,
            arguments.out,
            writer=writer,
            shot_media=shot_media,
            host=arguments.host,
            port=int(arguments.port),
        )
    except CacheMismatch as exc:
        _emit({"status": "error", "command": "console serve", "error": str(exc)})
        return 2
    return 0


def register(subparsers: Any) -> None:
    """Attach ``dpo console`` and its four actions."""
    console = subparsers.add_parser("console", help="the scene-adaptive caption console (docs/v2-console)")
    actions = console.add_subparsers(dest="action", required=True)

    def calibration_flags(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--half-saturation", type=float, help="r0, the area counting as half visible")
        parser.add_argument("--iou-threshold", type=float, help="mask agreement above which labels merge")
        parser.add_argument("--cut-threshold", type=float, help="θ, the composition shift that cuts a shot")
        parser.add_argument("--minimum-shot-ms", type=int, help="the floor on shot length")

    preprocess = actions.add_parser(
        "preprocess", help="cut shots, group audio labels, and measure r_g and p_g"
    )
    preprocess.add_argument("--mask-root", required=True, help="directory containing audio/ and visual/")
    preprocess.add_argument("--out", required=True, help=f"output {MANIFEST_SCHEMA} JSON")
    preprocess.add_argument("--clips", nargs="*", help="clip ids; default: every clip under audio/")
    preprocess.add_argument("--fps", type=float, default=60.0, help="source video frame rate")
    preprocess.add_argument("--downsample", type=int, default=4, help="read masks at 1/N resolution")
    preprocess.add_argument(
        "--cache-dir",
        help="keep decoded masks here between runs, one .npz per clip, so a recalibration reads no PNG twice",
    )
    preprocess.add_argument(
        "--tidy-data",
        help="CSV with final_labels and top_level_parent_name; turns on family-aware grouping"
        " and enables --provisional-salience",
    )
    preprocess.add_argument(
        "--ontology", help="AudioSet ontology JSON; default: ontology.json beside --tidy-data"
    )
    preprocess.add_argument(
        "--provisional-salience",
        action="store_true",
        help="fill c_g and e_g from tag multiplicity for a dry run, and say so in the manifest",
    )
    calibration_flags(preprocess)
    preprocess.set_defaults(handler=_console_preprocess)

    scaffold = actions.add_parser("scaffold", help="turn a manifest into a session document")
    scaffold.add_argument("--manifest", required=True)
    scaffold.add_argument("--out", required=True)
    scaffold.add_argument("--study-id", required=True)
    scaffold.add_argument("--corpus-id", required=True)
    scaffold.add_argument("--clips", nargs="*", help="clip ids in session order; default: the manifest's")
    # The preprocessing thresholds come from the manifest, where they were
    # applied; only r0, applied when the document is served, is set here.
    scaffold.add_argument("--half-saturation", type=float, help="r0, the area counting as half visible")
    scaffold.set_defaults(handler=_console_scaffold)

    validate = actions.add_parser("validate", help="refuse a document that is not yet authored")
    validate.add_argument("--session", required=True)
    validate.set_defaults(handler=_console_validate)

    serve = actions.add_parser("serve", help="run the kiosk")
    serve.add_argument("--session", required=True)
    serve.add_argument("--media-dir", required=True)
    serve.add_argument("--out", required=True)
    serve.add_argument("--writer", choices=("template", "gemma"), default="template")
    serve.add_argument("--backend-config", help="Gemma 4 backend config; required by --writer gemma")
    serve.add_argument("--contract", default=DEFAULT_CONTRACT)
    serve.add_argument("--checkpoint", help="LoRA checkpoint directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8778)
    serve.set_defaults(handler=_console_serve)
