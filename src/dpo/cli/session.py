"""`dpo session`: validate, scaffold, and serve a caption session document.

The document is authored by hand (docs/v1-session/runbook.md), so ``validate``
exists as its own command and ``scaffold`` writes a starting point that
``validate`` refuses until the sources are filled in — a scaffold that passed
would let an unauthored session be served to a participant.

``serve`` picks the writer. The template writer needs nothing and is the
default; the Gemma writer is constructed the way ``dpo study export`` builds
its adapter (a backend config, the study contract's audio track, an optional
checkpoint), behind a lazy import so choosing the template never imports torch.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dpo.caption.media import clip_duration_ms, find_clip_video
from dpo.caption.writer import CacheMismatch
from dpo.cli._shared import _emit
from dpo.session.document import (
    ATTENTION_POLES,
    ATTENTION_TEXT,
    AUDIO_ROLE_HEADS,
    CRITERION_TEXT,
    MAX_SOURCES,
    OPEN_ITEM_TEXT,
    POLES,
    SESSION_SCHEMA,
    SessionDocumentError,
    load_session_document,
)
from dpo.session.mask_parameters import MASK_LINK_SCHEMA, MaskParameterError, derive_mask_links
from dpo.session.writer import CaptionWriter, ShotMedia, TemplateWriter

DEFAULT_CONTRACT = "configs/study/street-audio.toml"
FALLBACK_CLIP_MS = 10000
# Placeholder visual roles for a scaffolded control clip (spec 4.6: "three role
# heads supplied with the visual inventory"); the researcher renames them.
SCAFFOLD_VISUAL_ROLES = {"backdrop": "Backdrop", "passing": "Passing through", "fixed": "Fixed here"}


class SessionUsageError(ValueError):
    """The command line, not the session document, is what is wrong.

    Kept apart from ``SessionDocumentError`` so ``serve`` never blames the
    document (``{"status": "invalid", "session": ...}``) for a missing flag.
    """


def _session_validate(arguments: argparse.Namespace) -> int:
    try:
        document = load_session_document(arguments.session)
    except SessionDocumentError as exc:
        _emit({"status": "invalid", "session": str(arguments.session), "error": str(exc)})
        return 2
    _emit(
        {
            "status": "valid",
            "session": str(arguments.session),
            "session_id": document["session_id"],
            "clips": len(document["clips"]),
            "shots": sum(len(clip["shots"]) for clip in document["clips"]),
        }
    )
    return 0


def _discover_clips(media_dir: Path) -> list[str]:
    found: dict[str, None] = {}
    for base in (media_dir / "unmuted_video", media_dir):
        if base.is_dir():
            for path in sorted(base.glob("*.mp4")):
                found.setdefault(path.stem, None)
    return list(found)


def _load_mask_links(path: Path) -> Mapping[str, Any]:
    """The ``dpo.caption-mask-link/v1`` manifest ``link-masks`` wrote."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionUsageError(f"cannot read mask links {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != MASK_LINK_SCHEMA:
        raise SessionUsageError(f"{path}: not a {MASK_LINK_SCHEMA} manifest")
    clips = document.get("clips")
    if not isinstance(clips, dict):
        raise SessionUsageError(f"{path}: manifest has no clips object")
    return clips


def _linked_sources(links: Mapping[str, Any], clip_id: str, task: str, path: Path) -> list[dict[str, Any]]:
    """The manifest's ``session_source`` rows for one clip, as the document wants them.

    A shaped clip takes the audio branch, a control clip the visual one. Only
    admission candidates cross over: on the visual branch a category whose mask
    is empty everywhere is not a thing in the shot, while on the audio branch
    every tag stays a candidate because an off-screen sound is still a sound
    (mask_parameters, ``_audio_entries``).

    Everything the manifest marks ``review_required`` comes across as it was
    derived. That is deliberate — the operator reviews it in the document,
    where the validator can refuse what is still unset — and it is why an
    audio source arrives with a null role and a scaffold does not validate.
    """
    clip = links.get(clip_id)
    if clip is None:
        raise SessionUsageError(f"{path}: no linked masks for clip {clip_id!r}")
    branch = "audio" if task == "shaped" else "visual"
    entries = clip.get(branch)
    if not isinstance(entries, list):
        raise SessionUsageError(f"{path}: clip {clip_id!r} has no {branch} branch")
    sources = [
        dict(entry["session_source"])
        for entry in entries
        if entry.get("parameters", {}).get("admission", {}).get("candidate")
    ]
    if not sources:
        raise SessionUsageError(
            f"{path}: clip {clip_id!r} has no admitted {branch} source; the skeleton needs at least one row"
        )
    if len(sources) > MAX_SOURCES:
        raise SessionUsageError(
            f"{path}: clip {clip_id!r} links {len(sources)} {branch} sources, "
            f"more than the {MAX_SOURCES} a shot may carry; re-run link-masks over fewer labels"
        )
    return sources


def _scaffold_clip(
    media_dir: Path,
    clip_id: str,
    index: int,
    links: Mapping[str, Any] | None = None,
    links_path: Path | None = None,
) -> dict[str, Any]:
    video = find_clip_video(media_dir, clip_id)
    end_ms = (clip_duration_ms(video) if video is not None else None) or FALLBACK_CLIP_MS
    task = "shaped" if index % 2 == 0 else "control"
    clip: dict[str, Any] = {
        "clip_id": clip_id,
        "task": task,
        "first_viewing_captions": index % 4 < 2,
        "balance_control": "orderings",
        "opening": {"level": "itemized", "balance": 0.5},
    }
    if task == "control":
        clip["visual_roles"] = dict(SCAFFOLD_VISUAL_ROLES)
    sources: list[dict[str, Any]] = []
    if links is not None and links_path is not None:
        sources = _linked_sources(links, clip_id, task, links_path)
    clip["shots"] = [
        {
            "shot_id": "s1",
            "start_ms": 0,
            "end_ms": end_ms,
            "automatic_caption": "",
            "scene": {"token": "", "prose": ""},
            "atmosphere": {"phrase": "", "prose": ""},
            "sources": sources,
        }
    ]
    return clip


def _session_scaffold(arguments: argparse.Namespace) -> int:
    media_dir = Path(arguments.media_dir)
    clip_ids = list(arguments.clips or []) or _discover_clips(media_dir)
    if not clip_ids:
        _emit({"status": "error", "error": f"no clips named and none found under {media_dir}"})
        return 2
    links_path = Path(arguments.mask_links) if arguments.mask_links else None
    try:
        links = _load_mask_links(links_path) if links_path is not None else None
        clips = [
            _scaffold_clip(media_dir, clip_id, index, links, links_path)
            for index, clip_id in enumerate(clip_ids)
        ]
    except SessionUsageError as exc:
        _emit({"status": "error", "command": "session scaffold", "error": str(exc)})
        return 2
    shaped = [clip["clip_id"] for clip in clips if clip["task"] == "shaped"]
    document = {
        "schema": SESSION_SCHEMA,
        "session_id": Path(arguments.out).stem,
        "poles": POLES,
        "role_heads": {"audio": AUDIO_ROLE_HEADS},
        "measures": {
            "items": [
                {"id": "attention", "text": ATTENTION_TEXT, "boxes": 7, "poles": ATTENTION_POLES},
                {"id": "criterion", "text": CRITERION_TEXT, "boxes": 2, "poles": ["No", "Yes"]},
            ],
            "open_item": OPEN_ITEM_TEXT,
        },
        "clips": clips,
        "followup": {
            "recognition": [{"clip_id": clip_id, "sounds": []} for clip_id in clip_ids],
            "sound_only": [],
            "check": [
                {"clip_id": clip_id, "a": "own" if i % 2 == 0 else "automatic"}
                for i, clip_id in enumerate(shaped)
            ],
        },
    }
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    linked = sum(len(clip["shots"][0]["sources"]) for clip in clips)
    _emit(
        {
            "status": "scaffolded",
            "clips": len(clips),
            "sources": linked,
            "mask_links": str(links_path) if links_path is not None else None,
            "out": str(out),
            # A scaffold never validates: the automatic caption, the scene and
            # atmosphere rows, the follow-up lists, and — on a shaped clip —
            # every source's role are the researcher's to author.
            "next": f"author the empty fields, then: dpo session validate --session {out}",
        }
    )
    return 0


def _session_link_masks(arguments: argparse.Namespace) -> int:
    """Summarize Sa2VA masks as auditable admission/balance/detail evidence."""
    tidy_data = Path(arguments.tidy_data)
    ontology = Path(arguments.ontology) if arguments.ontology else tidy_data.with_name("ontology.json")
    if arguments.clips is not None and not arguments.clips:
        # `--clips` with nothing after it selected nothing and wrote an empty
        # manifest that looked finished; naming no clip is a mistake to say.
        _emit(
            {
                "status": "error",
                "command": "session link-masks",
                "error": "--clips names no clip; give at least one id, or omit it to link every clip",
            }
        )
        return 2
    try:
        document = derive_mask_links(
            mask_root=Path(arguments.mask_root),
            tidy_data=tidy_data,
            ontology=ontology,
            fps=float(arguments.fps),
            clips=arguments.clips,
        )
    except MaskParameterError as exc:
        _emit({"status": "invalid", "command": "session link-masks", "error": str(exc)})
        return 2
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(
        {
            "status": "linked",
            "out": str(out),
            "clips": len(document["clips"]),
            "audio_sources": sum(len(clip["audio"]) for clip in document["clips"].values()),
            "visual_sources": sum(len(clip["visual"]) for clip in document["clips"].values()),
        }
    )
    return 0


def _gemma_writer(arguments: argparse.Namespace) -> tuple[CaptionWriter, ShotMedia] | None:
    """The Gemma writer and its shot-media resolver, or None when blocked (exit 3).

    Mirrors ``dpo.cli._backend``: gate on CUDA before loading anything, then
    build the adapter from the backend config and the contract's audio track.
    """
    if not arguments.backend_config:
        raise SessionUsageError("--writer gemma requires --backend-config")
    from dpo.candidates.candidate_records import CandidateError
    from dpo.candidates.generation import verify_backend_pin
    from dpo.contracts.study_contract import load_contract
    from dpo.models.gemma4.backend_config import load_config

    config = load_config(arguments.backend_config)
    contract = load_contract(arguments.contract)
    if config.model.media_inputs != "audio" or "audio" not in contract.tracks:
        raise SessionUsageError(
            "--writer gemma needs an audio backend config and a contract with [tracks.audio]"
        )
    # Ahead of the CUDA gate, so a config the contract does not pin is refused
    # on any machine rather than only on the one that would have loaded it.
    # The participant-facing writer is held to the same pin as the pipeline.
    try:
        verify_backend_pin(contract, track="audio", backend_config_path=Path(arguments.backend_config))
    except CandidateError as exc:
        raise SessionUsageError(str(exc)) from exc

    import torch

    if not torch.cuda.is_available():
        _emit(
            {
                "status": "blocked_pending_external_operation",
                "command": "session serve",
                "gate": "the Gemma writer requires a CUDA device; use --writer template without one",
                "side_effects": False,
            }
        )
        return None
    from dpo.caption.media import shot_audio, shot_still
    from dpo.core.safety import CheckpointSafetyError
    from dpo.models.gemma4.adapter import GemmaCaptionAdapter
    from dpo.session.writer import GemmaWriter

    try:
        adapter = GemmaCaptionAdapter(
            config=config,
            contract=contract.tracks["audio"],
            media_resolver=lambda reference: reference,
            adapter_dir=None if arguments.checkpoint is None else str(arguments.checkpoint),
        )
    except CheckpointSafetyError as exc:
        raise SessionUsageError(str(exc)) from exc
    media_dir = Path(arguments.media_dir)
    cache_dir = Path(arguments.out) / "media-cache"

    # A control clip's writer looks at the shot rather than listening to it
    # (spec 4.6): the resolver picks by the clip's task in the document.
    tasks = {
        str(clip["clip_id"]): str(clip["task"])
        for clip in load_session_document(Path(arguments.session))["clips"]
    }

    def resolve(clip_id: str, shot: Mapping[str, Any]) -> Path:
        cut = shot_still if tasks.get(clip_id) == "control" else shot_audio
        return cut(media_dir, cache_dir, clip_id, int(shot["start_ms"]), int(shot["end_ms"]))

    return GemmaWriter(adapter), resolve


def _session_serve(arguments: argparse.Namespace) -> int:
    from dpo.session.app import run_session_app

    try:
        if arguments.writer == "gemma":
            built = _gemma_writer(arguments)
            if built is None:
                return 3
            writer, shot_media = built
        else:
            writer, shot_media = TemplateWriter(), None
        run_session_app(
            session_path=Path(arguments.session),
            media_dir=Path(arguments.media_dir),
            out_dir=Path(arguments.out),
            writer=writer,
            shot_media=shot_media,
            host=arguments.host,
            port=arguments.port,
        )
    except SessionUsageError as exc:
        _emit({"status": "error", "command": "session serve", "error": str(exc)})
        return 2
    except CacheMismatch as exc:
        _emit({"status": "error", "command": "session serve", "error": str(exc)})
        return 2
    except SessionDocumentError as exc:
        _emit({"status": "invalid", "session": str(arguments.session), "error": str(exc)})
        return 2
    return 0
