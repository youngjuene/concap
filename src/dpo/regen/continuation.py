"""Carry completed legacy calibration into the interactive viewing protocol."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from dpo.regen import progress
from dpo.regen.config import FAMILY_OF_PARENT, SOUND_FAMILIES, Scale
from dpo.regen.items import ItemSet, load_items
from dpo.regen.log import EventLog, validate_participant, view_id
from dpo.regen.study_api import build_study_app
from dpo.regen.study_instrument import instrument_snapshot
from dpo.regen.study_personalization import questionnaire_profile
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
    startup_timeout: float = 180


class Continuation:
    def __init__(
        self,
        config: ViewingConfig,
        document: Mapping[str, Any],
        log: EventLog,
        *,
        item_set: ItemSet | None = None,
        scale: Scale | None = None,
    ) -> None:
        self.document, self.log, self.engine = document, log, config.engine
        self.item_set = item_set if item_set is not None else load_items()
        self.scale = scale if scale is not None else Scale()
        self.app = build_study_app(
            config.manifest,
            config.media,
            config.out,
            config.model,
            inference_timeout=config.inference_timeout,
            linked_only=True,
            engine=config.engine,
            startup_timeout=config.startup_timeout,
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
        versioned = snapshot.get("flow_version") == progress.FLOW_VERSION
        if snapshot.get("document_hash", digest(self.document)) != digest(self.document):
            raise Conflict("Calibration belongs to another frozen study document.")
        raw_clip_count = snapshot.get("clip_count")
        clip_count = raw_clip_count if isinstance(raw_clip_count, int) and raw_clip_count > 0 else 1
        if versioned:
            if len(viewings) != clip_count * 2 or len(responses) != clip_count * 2 + 1:
                raise Conflict("The completed calibration records are incomplete; contact the researcher.")
            if {r.get("page") for r in responses} != {"art", "survey", "overall"}:
                raise Conflict("The completed calibration records are incomplete; contact the researcher.")
            by_id = {row.get("view_id"): row for row in viewings}
            expected_ids = {view_id(participant, index) for index in range(clip_count * 2)}
            expected_responses = {
                (view_id(participant, index), "art" if index % 2 == 0 else "survey")
                for index in range(clip_count * 2)
            } | {(f"{participant}-overall", "overall")}
            if (
                set(by_id) != expected_ids
                or {(r.get("view_id"), r.get("page")) for r in responses} != expected_responses
            ):
                raise Conflict("Calibration viewing identities are incomplete or duplicated.")
            order = snapshot.get("clip_order")
            for index in range(clip_count):
                original, updated = (
                    by_id[view_id(participant, index * 2)],
                    by_id[view_id(participant, index * 2 + 1)],
                )
                segment = original.get("segment")
                if segment not in self.document["segments"] or (order and segment != order[index]):
                    raise Conflict("Calibration clip does not match its frozen order.")
                for row, condition, scope in (
                    (original, "prepared", "clip_original"),
                    (updated, "regenerated", "clip_updated"),
                ):
                    if (
                        row.get("clip_index"),
                        row.get("segment"),
                        row.get("clip_id"),
                        row.get("condition"),
                        row.get("scope"),
                    ) != (index, segment, self.document["segments"][segment]["clip_id"], condition, scope):
                        raise Conflict("Calibration viewing does not match its clip and condition.")
        elif len(viewings) != 2 or len(responses) != 2 or {r["page"] for r in responses} != {"art", "survey"}:
            raise Conflict("The completed calibration records are incomplete; contact the researcher.")
        if any(r.get("config_hash") != self.log.config_hash for r in viewings + responses):
            raise Conflict("Calibration belongs to another study configuration.")
        if "auditory_ids" not in snapshot or "visual_labels" not in snapshot:
            raise Conflict("The visual and sound calibration records are incomplete.")
        observations: list[dict[str, Any]] = []
        if versioned:
            visual_by_segment = snapshot.get("visual_by_segment")
            auditory_by_segment = snapshot.get("auditory_by_segment")
            if not isinstance(visual_by_segment, Mapping) or not isinstance(auditory_by_segment, Mapping):
                raise Conflict("The visual and sound calibration records are incomplete.")
            original_viewings = sorted(
                [v for v in viewings if v.get("scope") == "clip_original"],
                key=lambda row: row.get("clip_index", 0),
            )
            if len(original_viewings) != clip_count:
                raise Conflict("The completed calibration records are incomplete; contact the researcher.")
            for viewing in original_viewings:
                segment_id = str(viewing["segment"])
                segment = self.document["segments"][segment_id]
                visual = visual_by_segment.get(segment_id)
                auditory = auditory_by_segment.get(segment_id)
                if not isinstance(visual, Mapping) or not isinstance(auditory, Mapping):
                    raise Conflict("The visual and sound calibration records are incomplete.")
                ids = auditory.get("ids")
                observations.append(
                    {
                        "clip": viewing["clip_id"],
                        "labels": list(visual.get("labels") or []),
                        "points": list(visual.get("points") or []),
                        "points_source": "server_snapshot",
                        "heard": {family: family in ids for family in SOUND_FAMILIES}
                        if isinstance(ids, list)
                        else {family: False for family in SOUND_FAMILIES},
                        "present": sorted(
                            {
                                FAMILY_OF_PARENT[s["parent"]]
                                for s in segment["stems"]
                                if s.get("parent") in FAMILY_OF_PARENT
                            }
                        ),
                    }
                )
        else:
            prepared = next(v for v in viewings if v["condition"] == "prepared")
            segment = self.document["segments"][prepared["segment"]]
            observations.append(
                {
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
            )
        source = {
            "schema": "dpo.legacy-calibration/v1",
            "participant": participant,
            "session_id": self.document["session_id"],
            "config_hash": self.log.config_hash,
            "language": language,
            "viewings": viewings,
            "responses": responses,
            "observations": observations,
        }

        def freeze(state: dict[str, Any]) -> None:
            if state["stage"] != "awaiting-calibration":
                raise Conflict("This viewing session already has a frozen calibration.")
            questionnaire = questionnaire_profile(responses, viewings, self.item_set, self.scale, language)
            profile = compile_profile({}, observations, language, questionnaire=questionnaire)
            profile["calibration_source_hash"] = digest(source)
            profile.pop("hash")
            profile["hash"] = digest(profile)
            state.update(
                stage="ready",
                flow_version=2,
                language=language,
                instrument=instrument_snapshot(language, item_set=self.item_set, scale=self.scale),
                preferences={},
                observations=observations,
                calibration_source=source,
                profile=profile,
                axes=profile["defaults"].copy(),
                video_surveys=[],
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
