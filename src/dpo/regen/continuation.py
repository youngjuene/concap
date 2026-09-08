"""Carry completed legacy calibration into the interactive viewing protocol."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from dpo.regen.config import FAMILY_OF_PARENT, SOUND_FAMILIES
from dpo.regen.log import EventLog, validate_participant
from dpo.regen.study_api import build_study_app
from dpo.regen.study_schema import compile_profile, digest
from dpo.regen.study_store import Conflict
from dpo.regen.study_worker import InferenceProcess


@dataclass(frozen=True)
class ViewingConfig:
    manifest: Path
    media: Path
    out: Path
    model: dict[str, Any] = field(default_factory=dict)
    inference_timeout: float = 30
    engine: InferenceProcess | None = None


class Continuation:
    def __init__(self, config: ViewingConfig, document: Mapping[str, Any], log: EventLog) -> None:
        self.document, self.log, self.engine = document, log, config.engine
        self.app = build_study_app(
            config.manifest,
            config.media,
            config.out,
            config.model,
            inference_timeout=config.inference_timeout,
            linked_only=True,
            engine=config.engine,
        )
        self.store = self.app.state.store
        self.namespace = digest([str(log.out_dir.resolve()), document["session_id"], log.config_hash])

    def _source(self, participant: str) -> str:
        return f"{self.namespace}:{validate_participant(participant)}"

    def enrol(self, participant: str, language: str, supplied: Any, *, issue: bool) -> str | None:
        token = self.store.link(
            self._source(participant), self.app.state.calibration_hash, language, create=issue
        )
        if token and (issue or isinstance(supplied, str) and secrets.compare_digest(token, supplied)):
            return str(token)
        return None

    def handoff(self, participant: str, token: Any, language: str) -> dict[str, Any]:
        held = self.enrol(participant, language, token, issue=False)
        if held is None:
            raise PermissionError("Continue from the browser tab that collected your calibration.")
        snapshot = self.log.snapshot(participant) or {}
        if snapshot.get("step") != "done":
            raise Conflict("Complete all six calibration pages before starting the viewing experience.")
        viewings, responses = self.log.viewings(participant), self.log.responses(participant)
        if len(viewings) != 2 or len(responses) != 2 or {r["page"] for r in responses} != {"art", "survey"}:
            raise Conflict("The completed calibration records are incomplete; contact the researcher.")
        if any(r.get("config_hash") != self.log.config_hash for r in viewings + responses):
            raise Conflict("Calibration belongs to another study configuration.")
        prepared = next(v for v in viewings if v["condition"] == "prepared")
        segment = self.document["segments"][prepared["segment"]]
        if "auditory_ids" not in snapshot or "visual_labels" not in snapshot:
            raise Conflict("The visual and sound calibration records are incomplete.")
        observation = {
            "clip": prepared["clip_id"],
            "labels": snapshot["visual_labels"],
            "points": snapshot.get("visual_points", []),
            "points_source": "server_snapshot"
            if "visual_points" in snapshot
            else "unavailable_before_upgrade",
            "heard": {family: family in snapshot["auditory_ids"] for family in SOUND_FAMILIES},
            "present": sorted(
                {
                    FAMILY_OF_PARENT[s["parent"]]
                    for s in segment["stems"]
                    if s.get("parent") in FAMILY_OF_PARENT
                }
            ),
        }
        source = {
            "schema": "dpo.legacy-calibration/v1",
            "participant": participant,
            "session_id": self.document["session_id"],
            "config_hash": self.log.config_hash,
            "language": language,
            "viewings": viewings,
            "responses": responses,
            "observations": [observation],
        }

        def freeze(state: dict[str, Any]) -> None:
            if state["stage"] != "awaiting-calibration":
                raise Conflict("This viewing session already has a frozen calibration.")
            profile = compile_profile({}, [observation], language)
            profile["calibration_source_hash"] = digest(source)
            profile.pop("hash")
            profile["hash"] = digest(profile)
            state.update(
                stage="ready",
                language=language,
                preferences={},
                observations=[observation],
                calibration_source=source,
                profile=profile,
                axes=profile["defaults"].copy(),
            )

        current = self.store.state(held)
        state = self.store.mutate(held, "legacy-calibration", current["revision"], digest(source), freeze)
        return {"url": f"/viewing/{state['session_id']}/", "token": held}

    def mount(self, parent: FastAPI) -> None:
        original = parent.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            try:
                async with original(app), self.app.router.lifespan_context(self.app):
                    yield
            finally:
                if self.engine is not None:
                    self.engine.close()

        parent.router.lifespan_context = lifespan
        parent.mount("/viewing/{continuation_id}", self.app)
