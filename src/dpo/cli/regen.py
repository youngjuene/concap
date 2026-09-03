"""`dpo regen`: scaffold, validate, and serve the regeneration instrument.

Three commands, in the order a study uses them (``docs/v3-regen/spec-behavior.md``):

* ``scaffold`` walks a staged media directory and writes a document with every
  asset §10 prepares already filled in, and the two caption tracks left as
  empty slots for a researcher to write;
* ``validate`` refuses a document that is not yet authored, naming the path;
* ``serve`` runs the instrument.

``validate`` is its own command because a scaffold never validates, and that is
deliberate. The prepared caption track is the study's stimulus and no tool can
write it; a scaffold that passed validation would let an unauthored document —
one whose captions are empty strings — reach a participant.

``serve`` says which item set it loaded before it binds a port. The default set
shipped in :mod:`dpo.regen.items` is placeholder wording, and a pilot run on it
is a legitimate thing to do; a study run on it by accident is not, so the
provenance is on the operator's console and on every response row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dpo.caption.writer import CacheMismatch, CaptionWriter
from dpo.cli._shared import _emit
from dpo.regen.config import Calibration, ConfigError, Configuration
from dpo.regen.document import (
    REGEN_SCHEMA,
    RegenDocumentError,
    configuration_of,
    load_regen_document,
    slug,
)
from dpo.regen.items import ItemsError, load_items

# Where ``scaffold`` looks under --media-dir. One directory per segment, named
# for the segment, so a document's references are readable as paths.
VIDEO = "clip.mp4"
FRAMES = "frames"
MASKS = "masks"
STEMS = "stems"
SEGMENTS = ("A", "B")


class RegenUsageError(ValueError):
    """The command line, not the document, is what is wrong."""


def _calibration(arguments: argparse.Namespace) -> Calibration:
    overrides = {
        name: getattr(arguments, name)
        for name in ("cue_slots", "minimum_points", "latency_ceiling_ms", "languages")
        if getattr(arguments, name, None) is not None
    }
    return Calibration(**overrides)


def _slots(count: int, duration_ms: int) -> list[dict[str, Any]]:
    """Empty cue slots spread evenly over the clip.

    The timings are a starting point a researcher moves, not a claim about the
    footage: §6 fixes them before generation, so where they land is a design
    decision about the stimulus. Even spacing is the one arrangement that
    encodes no assumption about where the interesting sound is.
    """
    span = duration_ms // count
    return [
        {"index": index, "start_ms": index * span, "end_ms": (index + 1) * span, "text": ""}
        for index in range(count)
    ]


def _segment(
    root: Path, segment: str, clip_id: str, calibration: Calibration, duration_ms: int
) -> dict[str, Any]:
    base = Path(segment)
    frames = sorted((root / segment / FRAMES).glob("*.png"))
    stems = sorted((root / segment / STEMS).glob("*.wav")) + sorted((root / segment / STEMS).glob("*.mp3"))
    if not frames:
        raise RegenUsageError(
            f"no frames under {root / segment / FRAMES}; §4 shows a strip of moments from the clip"
        )
    if not stems:
        raise RegenUsageError(f"no stems under {root / segment / STEMS}; §10 prepares the separated sources")
    strip = []
    step = duration_ms // (len(frames) + 1)
    for position, frame in enumerate(frames):
        masks = sorted((root / segment / MASKS / frame.stem).glob("*.png"))
        if not masks:
            raise RegenUsageError(
                f"no masks under {root / segment / MASKS / frame.stem}; "
                "each frame of the strip carries the masks cut from that frame"
            )
        strip.append(
            {
                # Evenly spaced, as a starting point: only the person who cut
                # the frames knows where in the clip each one came from, and a
                # scaffold that guessed a timestamp would be guessing.
                "at_ms": (position + 1) * step,
                "still": str(base / FRAMES / frame.name),
                "objects": [
                    {
                        "id": slug(path.stem),
                        "label": path.stem,
                        "mask": str(base / MASKS / frame.stem / path.name),
                    }
                    for path in masks
                ],
            }
        )
    return {
        "segment": segment,
        "clip_id": clip_id,
        "video": str(base / VIDEO),
        "duration_ms": duration_ms,
        "frames": strip,
        # Waveform, colour and gain cannot be derived from a file listing:
        # the envelope has to be computed and the gain measured against the
        # original mix (§5). Left at values the validator accepts so the
        # document loads, and named in the scaffold's own output so a
        # researcher knows they are placeholders.
        "stems": [
            {
                "id": slug(path.stem),
                "label": path.stem,
                "audio": str(base / STEMS / path.name),
                "colour": "#666666",
                "gain": 1.0,
                "waveform": [0.0] * 64,
            }
            for path in stems
        ],
        "prepared_track": _slots(calibration.cue_slots, duration_ms),
        "fallback_track": _slots(calibration.cue_slots, duration_ms),
    }


def _regen_scaffold(arguments: argparse.Namespace) -> int:
    root = Path(arguments.media_dir)
    clips = list(arguments.clips or [])
    if len(clips) != 2:
        _emit(
            {
                "status": "error",
                "command": "regen scaffold",
                "error": "--clips takes exactly two clip ids, A then B",
            }
        )
        return 2
    # The calibration flags are inside the guard with the media walk: a
    # --cue-slots the configuration refuses is the researcher's typo, and it
    # should read like the other refusals rather than like a crash.
    try:
        calibration = _calibration(arguments)
        configuration = Configuration(
            study_id=arguments.study_id, corpus_id=arguments.corpus_id, calibration=calibration
        )
        segments = {
            name: _segment(root, name, clip, calibration, int(arguments.duration_ms))
            for name, clip in zip(SEGMENTS, clips, strict=True)
        }
    except (RegenUsageError, ConfigError) as exc:
        _emit({"status": "error", "command": "regen scaffold", "error": str(exc)})
        return 2
    document = {
        "schema": REGEN_SCHEMA,
        "session_id": arguments.session_id,
        "config": configuration.artifact(),
        "segments": segments,
    }
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(
        {
            "status": "scaffolded",
            "command": "regen scaffold",
            "out": str(out),
            "config_hash": configuration.hash,
            "cue_slots": calibration.cue_slots,
            "authoring_required": [
                "segments.*.frames[*].at_ms",
                "segments.*.prepared_track[*].text",
                "segments.*.fallback_track[*].text",
                "segments.*.stems[*].waveform",
                "segments.*.stems[*].colour",
                "segments.*.stems[*].gain",
            ],
        }
    )
    return 0


def _regen_validate(arguments: argparse.Namespace) -> int:
    try:
        document = load_regen_document(Path(arguments.session))
        items = load_items(Path(arguments.items) if arguments.items else None)
    except (RegenDocumentError, ItemsError) as exc:
        _emit({"status": "invalid", "session": str(arguments.session), "error": str(exc)})
        return 2
    configuration = configuration_of(document)
    _emit(
        {
            "status": "valid",
            "command": "regen validate",
            "session": str(arguments.session),
            "config_hash": configuration.hash,
            "cue_slots": configuration.cue_slots,
            "items_provenance": items.provenance,
            "items_digest": items.digest,
        }
    )
    return 0


def _gemma_writer(arguments: argparse.Namespace, document: dict[str, Any]) -> CaptionWriter | None:
    """The participant-facing model writer, held to the pipeline's own pin.

    Assembled the same way ``dpo console serve`` assembles it, with this
    instrument's own instruction: the model is told what the participant
    reported, not what a console's controls are set to.
    """
    if not arguments.backend_config:
        raise RegenUsageError("--writer gemma requires --backend-config")
    from dpo.candidates.candidate_records import CandidateError
    from dpo.candidates.generation import verify_backend_pin
    from dpo.contracts.study_contract import load_contract
    from dpo.models.gemma4.backend_config import load_config

    config = load_config(arguments.backend_config)
    contract = load_contract(arguments.contract)
    if config.model.media_inputs != "audio" or "audio" not in contract.tracks:
        raise RegenUsageError(
            "--writer gemma needs an audio backend config and a contract with [tracks.audio]"
        )
    try:
        verify_backend_pin(contract, track="audio", backend_config_path=Path(arguments.backend_config))
    except CandidateError as exc:
        raise RegenUsageError(str(exc)) from exc

    import torch

    if not torch.cuda.is_available():
        _emit(
            {
                "status": "blocked_pending_external_operation",
                "command": "regen serve",
                "gate": "the Gemma writer requires a CUDA device; use --writer template without one",
                "side_effects": False,
            }
        )
        return None
    from dpo.caption.writer import GemmaWriter
    from dpo.core.safety import CheckpointSafetyError
    from dpo.models.gemma4.adapter import GemmaCaptionAdapter
    from dpo.regen.regeneration import RegenRequestBuilder, RegenTemplateWriter

    try:
        adapter = GemmaCaptionAdapter(
            config=config,
            contract=contract.tracks["audio"],
            media_resolver=lambda reference: reference,
            adapter_dir=None if arguments.checkpoint is None else str(arguments.checkpoint),
        )
    except CheckpointSafetyError as exc:
        raise RegenUsageError(str(exc)) from exc
    builder = RegenRequestBuilder(configuration_of(document))
    return GemmaWriter(adapter, instruction=builder.instruction, fallback=RegenTemplateWriter())


def _regen_serve(arguments: argparse.Namespace) -> int:
    from dpo.regen.app import run_regen_app
    from dpo.regen.regeneration import RegenTemplateWriter

    try:
        document = load_regen_document(Path(arguments.session))
        items = load_items(Path(arguments.items) if arguments.items else None)
        writer: CaptionWriter | None
        if arguments.writer == "gemma":
            writer = _gemma_writer(arguments, document)
            if writer is None:
                return 3
        else:
            writer = RegenTemplateWriter()
    except RegenUsageError as exc:
        _emit({"status": "error", "command": "regen serve", "error": str(exc)})
        return 2
    except (RegenDocumentError, ItemsError) as exc:
        _emit({"status": "invalid", "session": str(arguments.session), "error": str(exc)})
        return 2
    configuration = configuration_of(document)
    # Said once, on the console the operator is watching: the stamp every line
    # will carry, and whether the items are the placeholder set.
    _emit(
        {
            "status": "serving",
            "command": "regen serve",
            "session": str(arguments.session),
            "config_hash": configuration.hash,
            "cue_slots": configuration.cue_slots,
            "items_provenance": items.provenance,
            "items_digest": items.digest,
            "writer": arguments.writer,
            "url": f"http://{arguments.host}:{int(arguments.port)}/",
        }
    )
    try:
        run_regen_app(
            document,
            arguments.media_dir,
            arguments.out,
            writer=writer,
            items=items,
            host=arguments.host,
            port=int(arguments.port),
        )
    except CacheMismatch as exc:
        _emit({"status": "error", "command": "regen serve", "error": str(exc)})
        return 2
    return 0


def register(subparsers: Any) -> None:
    """Attach ``dpo regen`` and its three actions."""
    regen = subparsers.add_parser("regen", help="the caption regeneration instrument (docs/v3-regen)")
    actions = regen.add_subparsers(dest="action", required=True)

    scaffold = actions.add_parser("scaffold", help="turn a staged media directory into a session document")
    scaffold.add_argument("--media-dir", required=True, help="directory holding A/ and B/")
    scaffold.add_argument("--out", required=True)
    scaffold.add_argument("--session-id", required=True)
    scaffold.add_argument("--study-id", required=True)
    scaffold.add_argument("--corpus-id", required=True)
    scaffold.add_argument("--clips", nargs=2, required=True, metavar=("A", "B"), help="clip ids, A then B")
    scaffold.add_argument(
        "--duration-ms", type=int, default=10000, help="segment length; §10 matches the two"
    )
    scaffold.add_argument("--cue-slots", type=int, help="§9.4's single slot count, for both tracks")
    scaffold.add_argument("--minimum-points", type=int, help="§4's floor on the visual selection")
    scaffold.add_argument("--latency-ceiling-ms", type=int, help="§6's ceiling before the fallback track")
    scaffold.add_argument(
        "--languages",
        nargs="+",
        help="the language tags the study offers; the first is the one a session opens in",
    )
    scaffold.set_defaults(handler=_regen_scaffold)

    validate = actions.add_parser("validate", help="refuse a document that is not yet authored")
    validate.add_argument("--session", required=True)
    validate.add_argument("--items", help="items JSON; default: the placeholder set in dpo.regen")
    validate.set_defaults(handler=_regen_validate)

    serve = actions.add_parser("serve", help="run the instrument")
    serve.add_argument("--session", required=True)
    serve.add_argument("--media-dir", required=True)
    serve.add_argument("--out", required=True)
    serve.add_argument("--items", help="items JSON; default: the placeholder set in dpo.regen")
    serve.add_argument("--writer", choices=("template", "gemma"), default="template")
    serve.add_argument("--backend-config", help="Gemma 4 backend config; required by --writer gemma")
    serve.add_argument("--contract", default="configs/study/street-audio.toml")
    serve.add_argument("--checkpoint", help="LoRA checkpoint directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8779, help="one past the console's 8778")
    serve.set_defaults(handler=_regen_serve)
