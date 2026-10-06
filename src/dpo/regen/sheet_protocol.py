"""Spreadsheet questionnaire sessions; historical regen sessions remain readable.

Responses and page advancement share the existing SQLite transaction. The
workbook is frozen on enrollment; incomplete researcher materials never become
invented participant questions or implicit consent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from dpo.caption.writer import CaptionWriter
from dpo.regen.captions import cues_of, validate_generated
from dpo.regen.continuation import Continuation
from dpo.regen.document import configuration_of, frames_of, objects_of, segment_order, track_of
from dpo.regen.gate import Gate, is_local
from dpo.regen.phase1_analysis import codebook_for_clip
from dpo.regen.playback import record_playback
from dpo.regen.points import matched_labels, measure_points, parse_points
from dpo.regen.sheet_instrument import page_spec, snapshot, validate_answers
from dpo.regen.study_schema import compile_profile, digest, media_path
from dpo.regen.study_store import Conflict, StudyStore

POLICY = "interaction-only-sheet-v1"
PROTOCOL = "dpo.sheet-questionnaire/v1"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30
ORDER = ("P1", "P2", "P3", "P4", "P5", "generate", "P6", "P7", "P8", "P9", "P10", "P11")
VIEWS = {"P1", "P6"}


def _checkpoint() -> dict[str, Any]:
    return {"position_ms": 0, "sequence": -1, "coverage": [], "epoch": 0, "last_tick": None}


def _configured_page(
    page: str, options: Mapping[str, Any], *, session_key: str | None = None
) -> dict[str, Any]:
    spec = page_spec(page, options=options, session_key=session_key)
    missing = [item["id"] for item in spec["items"] if item.get("requires_options")]
    if missing:
        raise ValueError(f"Configure the per-video options for {', '.join(missing)}")
    return spec


class SheetQuestionnaire:
    def __init__(
        self,
        document: Mapping[str, Any],
        media: Path,
        out: Path,
        writer: CaptionWriter,
        continuation: Continuation | None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.document, self.media = document, media
        self.configuration = configuration_of(document)
        self.continuation = continuation
        self.config = deepcopy(dict(config or {}))
        self._store = continuation.store if continuation else None
        self.out = out
        self.cookie = "regen_questionnaire_" + digest(str(out.resolve()))[:12]
        self.document_hash = digest(document)

    @property
    def store(self) -> StudyStore:
        if self._store is None:
            self._store = StudyStore(self.out / "questionnaire.sqlite3")
        return self._store

    def segment(self, state: Mapping[str, Any]) -> str:
        return str(state["sheet_clip_order"][state["sheet_clip_index"]])

    def _clip_config(self, state: Mapping[str, Any], segment: str) -> dict[str, Any]:
        clips = state["sheet_config"].get("clips", {})
        entry = clips.get(segment, {}) if isinstance(clips, dict) else None
        if not isinstance(entry, dict):
            raise ValueError("Clip settings must be an object")
        return entry

    def options(self, state: Mapping[str, Any], segment: str | None = None) -> dict[str, Any]:
        options = self._clip_config(state, segment or self.segment(state)).get("options", {})
        if not isinstance(options, dict):
            raise ValueError("Questionnaire options must be an object")
        # page_spec already supplies A1s as the default for A2s.
        return deepcopy(options)

    def _practice(self, state: Mapping[str, Any]) -> dict[str, Any]:
        practice = state["sheet_config"].get("practice", {})
        if not isinstance(practice, dict) or practice.get("reviewed") is not True:
            raise ValueError("본실험과 별개인 연습 이미지·선택지를 설정하고 검토해야 합니다.")
        frames = practice.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("연습 이미지는 별도로 준비한 실제 파일이어야 합니다.")
        formal_frames = {
            (self.media / frame["still"]).resolve()
            for segment in self.document["segments"].values()
            for frame in segment["frames"]
        }
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("still"), str):
                raise ValueError("연습 이미지는 별도로 준비한 실제 파일이어야 합니다.")
            path = media_path(self.media, frame["still"])
            if path in formal_frames:
                raise ValueError("연습 이미지는 본실험 이미지와 별개여야 합니다.")
            if type(frame.get("at_ms", 0)) is not int or frame.get("at_ms", 0) < 0:
                raise ValueError("Practice frame times must be nonnegative integers")
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, SyntaxError) as exc:
                raise ValueError("연습 이미지는 읽을 수 있는 이미지 파일이어야 합니다.") from exc
        _configured_page("P3", {"A1s": practice.get("sound_options")})
        return practice

    def _validate_clip(self, state: Mapping[str, Any], segment: str) -> None:
        entry = self._clip_config(state, segment)
        if entry.get("options_status") != "reviewed":
            raise ValueError("소리·자막 선택지의 연구자 검토가 필요합니다.")
        options = self.options(state, segment)
        first_audio = _configured_page("P3", options)
        _configured_page("P8", options)
        _configured_page("P11", options)
        condition = entry.get("condition", {})
        if (
            not isinstance(condition, dict)
            or condition.get("audiovisual_congruence") not in ("congruent", "incongruent")
            or condition.get("description_depth") not in ("shallow", "deep")
        ):
            raise ValueError("시청각 일치도 × 묘사 깊이 조건을 명시해야 합니다.")
        strategy = entry.get("second_caption_strategy")
        if entry.get("stimulus_status") != "reviewed" or strategy not in ("maintain", "opposite"):
            raise ValueError("검토된 유지/반대 자막 조건이 필요합니다.")
        self.first_track(state, segment)
        if strategy == "opposite":
            tracks = entry.get("second_tracks", {})
            if not isinstance(tracks, dict):
                raise ValueError("Opposite captions must map response combinations to tracks")
            visuals = {"unclassified"} | {
                obj["id"] for frame in frames_of(self.document, segment) for obj in frame.get("objects", [])
            }
            sounds = {option["value"] for option in first_audio["items"][0]["options"]}
            missing = sorted(
                f"{visual}|{sound}"
                for visual in visuals
                for sound in sounds
                if f"{visual}|{sound}" not in tracks
            )
            if missing:
                raise ValueError("반대 자막 트랙이 없는 응답 조합: " + ", ".join(missing))
            for selection, track in tracks.items():
                try:
                    self._caption_track(segment, track)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{selection}: {exc}") from exc

    def setup_missing(self, state: Mapping[str, Any]) -> list[str]:
        missing = []
        config = state["sheet_config"]
        if not isinstance(config.get("consent_text"), str) or not config["consent_text"].strip():
            missing.append("연구자가 확정한 동의서 전문을 설정해야 합니다.")
        if not isinstance(config.get("debrief_text"), str) or not config["debrief_text"].strip():
            missing.append("연구자가 확정한 디브리핑 전문을 설정해야 합니다.")
        try:
            self._practice(state)
        except (ValueError, KeyError, TypeError) as exc:
            missing.append(f"연습: {exc}")
        for segment in state["sheet_clip_order"]:
            try:
                self._validate_clip(state, segment)
            except (ValueError, KeyError, TypeError) as exc:
                missing.append(f"{segment}: {exc}")
        return missing

    def create(self) -> str:
        calibration_hash = (
            self.continuation.app.state.calibration_hash if self.continuation else self.document_hash
        )

        def initialize(state: dict[str, Any], sequence: int) -> None:
            state.update(
                registration_sequence=sequence,
                stage="sheet-questionnaire",
                sheet_protocol=PROTOCOL,
                sheet_page="P0",
                sheet_clip_index=0,
                sheet_clip_order=list(segment_order(self.document, sequence)),
                sheet_instrument=snapshot(),
                sheet_config=self.config,
                sheet_document_hash=self.document_hash,
                sheet_responses=[],
                sheet_viewings=[],
                sheet_observations=[],
                sheet_regenerations={},
                sheet_drafts={},
                sheet_playback={},
                sheet_practice=[],
                survey_policy=POLICY,
                preferences={},
                poststudy={
                    "interview_prompts": page_spec("P12")["items"],
                    "debrief": {"text": self.config.get("debrief_text", "")},
                },
            )
            state.pop("instrument", None)
            self._record_event(state, "sheet-enrollment", {})

        return self.store.create(calibration_hash, "ko", initialize=initialize)

    def checked(self, token: str) -> dict[str, Any]:
        state = self.store.state(token)
        if state.get("sheet_protocol") != PROTOCOL:
            raise PermissionError("This is not a spreadsheet questionnaire session")
        if (
            state.get("sheet_page") == "P0"
            and "sheet_consent" not in state
            and state["sheet_config"] != self.config
        ):
            configuration = deepcopy(self.config)
            configuration_hash = digest(configuration)

            def refresh(draft: dict[str, Any]) -> None:
                if draft.get("sheet_page") != "P0" or "sheet_consent" in draft:
                    raise Conflict("Consented study materials are frozen")
                self._record_event(draft, "setup-refresh", {"configuration_hash": configuration_hash})
                draft["sheet_config"] = configuration
                draft["poststudy"]["debrief"] = {"text": configuration.get("debrief_text", "")}

            state = self.store.mutate(
                token,
                f"setup-{state['revision']}-{configuration_hash}",
                state["revision"],
                configuration_hash,
                refresh,
            )
        if state["sheet_document_hash"] != self.document_hash:
            raise Conflict("The session's media configuration changed; contact the researcher")
        if state["sheet_instrument"]["hash"] != snapshot()["hash"]:
            raise Conflict("The questionnaire version changed; resume with its original source version")
        return state

    def present(self, state: dict[str, Any]) -> dict[str, Any]:
        page = state["sheet_page"]
        result: dict[str, Any] = {
            "session_id": state["session_id"],
            "participant": state["session_id"],
            "revision": state["revision"],
            "protocol": PROTOCOL,
            "page": page,
            "clip_index": state["sheet_clip_index"],
            "clip_count": len(state["sheet_clip_order"]),
            "practice": page == "practice",
            "setup_missing": [],
            "items": [],
            "instrument_hash": state["sheet_instrument"]["hash"],
        }
        if page in {"generate", "handoff", "practice"}:
            result.update(kind={"generate": "generating", "handoff": "handoff", "practice": "practice"}[page])
            result["title"] = {
                "generate": "두 번째 자막 준비",
                "handoff": "1스테이지 완료",
                "practice": "연습 시행",
            }[page]
            result["instruction"] = "연습 응답은 분석에서 제외됩니다." if page == "practice" else ""
        else:
            spec = _configured_page(
                page, {} if page == "P0" else self.options(state), session_key=state["session_id"]
            )
            result.update(
                {key: spec[key] for key in ("id", "title", "kind", "instruction", "language", "items")}
            )
            result["page"] = page
        if page == "P0":
            result.update(kind="intro", consent_text=state["sheet_config"].get("consent_text", ""))
            result["setup_missing"] = self.setup_missing(state)
        segment = self.segment(state)
        result["clip_id"] = self.document["segments"][segment]["clip_id"]
        if page in {"P2", "P7"}:
            result["frames"] = [
                {
                    "id": str(index),
                    "index": index,
                    "at_ms": frame["at_ms"],
                    "url": f"/media/frame/{segment}/{index}",
                }
                for index, frame in enumerate(frames_of(self.document, segment))
            ]
        if page == "practice":
            practice = self._practice(state)
            result["frames"] = [
                {
                    "id": str(index),
                    "index": index,
                    "at_ms": frame.get("at_ms", 0),
                    "url": f"/api/questionnaire/practice-frame/{index}",
                }
                for index, frame in enumerate(practice["frames"])
            ]
            rating = deepcopy(page_spec("P4")["items"][0])
            rating["id"] = "practice_rating"
            rating["text"] = "연습: 선택지를 골라 응답 방법을 확인해 주세요."
            sound = _configured_page("P3", {"A1s": practice["sound_options"]})["items"][0]
            sound["id"] = "practice_sound"
            sound["text"] = "연습: 선택지 하나를 골라 주세요."
            point = deepcopy(page_spec("P2")["items"][0])
            point["id"] = "practice_visual"
            point["text"] = "연습 이미지에서 한 곳을 선택해 주세요."
            result["items"] = [rating, sound, point]
        if page in VIEWS:
            entry = self.document["segments"][segment]
            checkpoint = state["sheet_playback"].get(f"{state['sheet_clip_index']}:{page}", _checkpoint())
            captions = (
                self.first_track(state, segment)
                if page == "P1"
                else state["sheet_regenerations"][segment]["track"]
            )
            result.update(
                kind="view",
                media={
                    "url": f"/media/video/{segment}",
                    "duration_ms": entry["duration_ms"],
                    "captions": captions,
                    "playback": {k: v for k, v in checkpoint.items() if k != "last_tick"},
                },
            )
        result["draft"] = state["sheet_drafts"].get(f"{state['sheet_clip_index']}:{page}", {})
        item_fields = {
            "id",
            "text",
            "type",
            "required",
            "min",
            "max",
            "points",
            "labels",
            "options",
            "requires_options",
        }
        result["items"] = [
            {key: value for key, value in item.items() if key in item_fields} for item in result["items"]
        ]
        for item in result["items"]:
            if "options" in item:
                item["options"] = [
                    {k: v for k, v in option.items() if k in {"value", "label", "exclusive"}}
                    for option in item["options"]
                ]
        return result

    def first_track(self, state: Mapping[str, Any], segment: str) -> list[dict[str, Any]]:
        entry = self._clip_config(state, segment)
        authored = (
            entry["first_track"]
            if "first_track" in entry
            else [cue.record("ko") for cue in track_of(self.document, segment, "prepared_track")]
        )
        return self._caption_track(segment, authored)

    def _caption_track(self, segment: str, authored: Any) -> list[dict[str, Any]]:
        expected = track_of(self.document, segment, "prepared_track")
        if not isinstance(authored, list) or any(not isinstance(cue, dict) for cue in authored):
            raise ValueError("Caption tracks must contain cue objects")
        cues = cues_of(authored, "ko")
        validate_generated(cues, self.configuration.calibration, "ko")
        if [(c.start_ms, c.end_ms) for c in cues] != [(c.start_ms, c.end_ms) for c in expected]:
            raise ValueError("Authored caption timing must match this clip")
        return [cue.record("ko") for cue in cues]

    def _record_event(self, state: dict[str, Any], action: str, data: Mapping[str, Any]) -> None:
        page = state["sheet_page"]
        first = page in {"P1", "P2", "P3", "P4", "P5"}
        second = page in {"generate", "P6", "P7", "P8", "P9", "P10", "P11"}
        formal = first or second
        segment = self.segment(state) if formal else None
        configured = self._clip_config(state, segment) if segment else {}
        index = state["sheet_clip_index"] if formal else None
        view = "P1" if first else "P6"
        scoped = {
            **data,
            "phase": 0 if page in {"P0", "practice"} else 1,
            "page": page,
            "stage": page,
            "clip_index": index,
            "clip_id": self.document["segments"][segment]["clip_id"] if segment else None,
            "view_id": f"{state['session_id']}-{index}-{view}" if formal else None,
            "condition": "first" if first else "second" if second else None,
            "protocol": PROTOCOL,
            "session_id": state["session_id"],
            "analysis_excluded": page in {"P0", "practice"},
            "stimulus_assignment": configured.get("condition"),
            "caption_strategy": configured.get("second_caption_strategy"),
        }
        state["_event_request"] = json.dumps(
            {"action": action, "data": scoped}, sort_keys=True, ensure_ascii=False, allow_nan=False
        )

    def _advance(self, state: dict[str, Any]) -> None:
        page = state["sheet_page"]
        if page == "P11":
            if state["sheet_clip_index"] + 1 < len(state["sheet_clip_order"]):
                state["sheet_clip_index"] += 1
                state["sheet_page"] = "P1"
            else:
                state["sheet_page"] = "handoff"
        else:
            state["sheet_page"] = ORDER[ORDER.index(page) + 1]

    def mutate(self, token: str, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        current = self.checked(token)
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("Action data must be an object")

        def apply(state: dict[str, Any]) -> None:
            if state["stage"] != "sheet-questionnaire":
                raise Conflict("The short questionnaire is already complete")
            page, segment = state["sheet_page"], self.segment(state)
            if data.get("page") != page:
                raise Conflict("This answer belongs to another page")
            self._record_event(state, action, data)
            key = f"{state['sheet_clip_index']}:{page}"
            if action == "intro-complete":
                if page != "P0" or data.get("consented") is not True or self.setup_missing(state):
                    raise Conflict("Configured consent and study materials are required")
                state["sheet_consent"] = {
                    "accepted": True,
                    "at": time.time(),
                    "text": state["sheet_config"]["consent_text"],
                }
                state["sheet_page"] = "practice"
            elif action == "practice-complete":
                if page != "practice":
                    raise Conflict("Practice is only completed once")
                raw = data.get("answers", {})
                if not isinstance(raw, dict) or set(raw) != {
                    "practice_rating",
                    "practice_sound",
                    "practice_visual",
                }:
                    raise ValueError("Complete the three practice responses")
                rating = raw.get("practice_rating")
                if type(rating) is not int or not 0 <= rating <= 6:
                    raise ValueError("연습 척도에서 0~6 중 하나를 선택해 주세요.")
                validate_answers(
                    "P3",
                    {"A1s": raw.get("practice_sound")},
                    options={"A1s": state["sheet_config"]["practice"]["sound_options"]},
                )
                visual = validate_answers("P2", {"V1": raw.get("practice_visual")})["V1"]
                parse_points(
                    [dict(point, frame=int(point["frame_id"])) for point in visual],
                    len(state["sheet_config"]["practice"]["frames"]),
                )
                state["sheet_practice"].append({"answers": raw, "analysis_excluded": True, "at": time.time()})
                state["sheet_page"] = "P1"
            elif action == "draft":
                if not isinstance(data.get("answers"), dict):
                    raise ValueError("Draft answers must be an object")
                state["sheet_drafts"][key] = deepcopy(data["answers"])
            elif action in {"playback", "view-ended"}:
                if page not in VIEWS:
                    raise Conflict("This page has no viewing")
                duration = self.document["segments"][segment]["duration_ms"]
                checkpoint = state["sheet_playback"].setdefault(key, _checkpoint())
                if action == "playback":
                    record_playback(checkpoint, data, duration)
                    checkpoint.setdefault("started_at", time.time())
                else:
                    covered = sum(b - a for a, b in checkpoint["coverage"])
                    if (
                        covered < duration - min(300, duration * 0.05)
                        or checkpoint["position_ms"] < duration - 100
                    ):
                        raise Conflict("영상을 끝까지 시청한 뒤 진행해 주세요.")
                    detail = self.present(state)
                    state["sheet_viewings"].append(
                        {
                            "view_id": f"{state['session_id']}-{state['sheet_clip_index']}-{page}",
                            "clip_index": state["sheet_clip_index"],
                            "clip_id": detail["clip_id"],
                            "page": page,
                            "condition": "first" if page == "P1" else "second",
                            "caption_strategy": state["sheet_config"]["clips"][segment][
                                "second_caption_strategy"
                            ],
                            "captions": detail["media"]["captions"],
                            "coverage": checkpoint["coverage"],
                            "started_at": checkpoint.get("started_at"),
                            "ended_at": time.time(),
                            "stimulus_assignment": state["sheet_config"]
                            .get("clips", {})
                            .get(segment, {})
                            .get("condition"),
                        }
                    )
                    self._advance(state)
            elif action == "submit":
                if page not in {"P2", "P3", "P4", "P5", "P7", "P8", "P9", "P10", "P11"}:
                    raise Conflict("This page has no questionnaire")
                options = self.options(state)
                raw_answers = data.get("answers")
                if not isinstance(raw_answers, dict):
                    raise ValueError("Answers must be an object")
                submitted = validate_answers(page, raw_answers, options=options)
                rendered = _configured_page(page, options, session_key=state["session_id"])
                viewing = "P1" if page in {"P2", "P3", "P4", "P5"} else "P6"
                record = {
                    "page": page,
                    "clip_index": state["sheet_clip_index"],
                    "clip_id": self.document["segments"][segment]["clip_id"],
                    "view_id": f"{state['session_id']}-{state['sheet_clip_index']}-{viewing}",
                    "condition": "first" if viewing == "P1" else "second",
                    "caption_strategy": state["sheet_config"]["clips"][segment]["second_caption_strategy"],
                    "stimulus_assignment": state["sheet_config"]["clips"][segment]["condition"],
                    "answers": submitted,
                    "items": rendered["items"],
                    "presented_order": [item["id"] for item in rendered["items"]],
                    "instrument_hash": state["sheet_instrument"]["hash"],
                    "submitted_at": time.time(),
                    "client_submitted_at": data.get("submitted_at"),
                    "analysis_excluded": False,
                }
                if page in {"P2", "P7"}:
                    code = "V1" if page == "P2" else "V2"
                    points = [dict(p, frame=int(p["frame_id"])) for p in submitted[code]]
                    frames = frames_of(self.document, segment)
                    measured = measure_points(
                        parse_points(points, len(frames)),
                        objects_of(self.document, self.media, segment),
                        {index: media_path(self.media, frame["still"]) for index, frame in enumerate(frames)},
                    )
                    matches = tuple(measurement.match for measurement in measured)
                    record["matches"] = [match.record() for match in matches]
                    record["labels"] = list(matched_labels(matches))
                    record["visual_metrics"] = [measurement.metrics for measurement in measured]
                state["sheet_responses"].append(record)
                state["sheet_drafts"].pop(key, None)
                self._advance(state)
            elif action == "handoff":
                if page != "handoff" or not self.continuation:
                    raise Conflict("Complete the short clips and configure long videos first")
                observations = self._observations(state)
                profile = compile_profile({}, observations, "ko")
                # New source items are not the old placeholder ART/PRSS instrument.
                # Preserve raw answers for analysis; no new scoring/prompt policy was supplied.
                profile["questionnaire_source"] = {
                    "instrument": state["sheet_instrument"],
                    "responses": state["sheet_responses"],
                }
                profile["hash"] = digest({k: v for k, v in profile.items() if k != "hash"})
                state.update(
                    stage="ready",
                    profile=profile,
                    observations=observations,
                    axes=profile["defaults"].copy(),
                    video_surveys=[],
                    calibration_source={
                        "schema": PROTOCOL,
                        "participant": state["session_id"],
                        "instrument": state["sheet_instrument"],
                        "responses": state["sheet_responses"],
                        "viewings": state["sheet_viewings"],
                        "observations": observations,
                    },
                )
            else:
                raise ValueError("Unknown questionnaire action")
            # Invalid next-page materials must roll back the response and its
            # receipt, not leave a committed session whose state cannot render.
            self.present(state)

        if action == "generate":
            return self._generate(token, current, payload, data)
        updated = self.store.mutate(
            token,
            str(payload.get("key", "")),
            payload.get("revision", -1),
            json.dumps({"action": action, "data": data}, sort_keys=True),
            apply,
        )
        return self.present(updated)

    def _observations(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        observations = []
        for index, segment in enumerate(state["sheet_clip_order"]):
            records = {r["page"]: r for r in state["sheet_responses"] if r["clip_index"] == index}
            sound = records["P3"]["answers"]["A1s"]
            book = codebook_for_clip(self._clip_config(state, segment))
            sound_type = sound if book["identity_sound_types"] else book["sound_types"].get(sound)
            family = book["sound_families"].get(sound_type) if sound_type is not None else None
            observations.append(
                {
                    "clip": self.document["segments"][segment]["clip_id"],
                    "points": records["P2"].get("matches", []),
                    "labels": records["P2"].get("labels", []),
                    "heard": {str(family): True} if family else {},
                    "selected_sound": sound,
                    "observation_type": "single_most_attention",
                }
            )
        return observations

    def _generate(
        self, token: str, state: dict[str, Any], payload: Mapping[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        segment = self.segment(state)
        if state["sheet_page"] == "P6" and segment in state["sheet_regenerations"]:
            return self.present(state)
        if state["sheet_page"] != "generate" or data.get("page") != "generate":
            raise Conflict("Generation belongs after the first questionnaire")
        records = {
            r["page"]: r for r in state["sheet_responses"] if r["clip_index"] == state["sheet_clip_index"]
        }
        sound = records["P3"]["answers"]["A1s"]
        configured = state["sheet_config"]["clips"][segment]
        strategy = configured.get("second_caption_strategy")
        if configured.get("stimulus_status") != "reviewed":
            raise Conflict("Reviewed caption conditions are required")
        if strategy == "maintain":
            track = self.first_track(state, segment)
            selection_key = None
        elif strategy == "opposite":
            visual = records["P2"].get("matches", [{}])[0].get("object_id") or "unclassified"
            selection_key = f"{visual}|{sound}"
            authored = configured.get("second_tracks", {}).get(selection_key)
            if not authored:
                raise Conflict("This response combination has no reviewed opposite-caption track")
            track = self._caption_track(segment, authored)
        else:
            raise Conflict("Select the reviewed maintain or opposite caption strategy")
        result = {
            "track": track,
            "writer": "authored-" + strategy,
            "strategy": strategy,
            "selection_key": selection_key,
            "fallback": False,
            "duration_ms": 0,
            "condition": configured.get("condition"),
            "caption_hash": digest(track),
        }

        def commit(draft: dict[str, Any]) -> None:
            if draft["sheet_page"] != "generate":
                raise Conflict("This generation already advanced")
            self._record_event(draft, "generate", data)
            draft["sheet_regenerations"][segment] = result
            self._advance(draft)
            self.present(draft)

        changed = self.store.mutate(
            token,
            str(payload.get("key", "")),
            payload.get("revision", -1),
            json.dumps({"action": "generate", "data": data}, sort_keys=True),
            commit,
        )
        return self.present(changed)


def mount_questionnaire(
    app: FastAPI,
    document: Mapping[str, Any],
    media: Path,
    out: Path,
    writer: CaptionWriter,
    continuation: Continuation | None,
    gate: Gate,
    config: Mapping[str, Any] | None = None,
) -> None:
    protocol = SheetQuestionnaire(document, media, out, writer, continuation, config)
    app.state.sheet_questionnaire = protocol

    def response(request: Request, action: str, payload: Mapping[str, Any] | None = None) -> JSONResponse:
        try:
            if request.method == "POST" and request.headers.get("x-study-request") != "1":
                raise PermissionError("Same-origin study request required")
            token = request.cookies.get(protocol.cookie, "")
            if action == "session":
                if not token:
                    if gate.requires_code and (not gate.code() or (payload or {}).get("code") != gate.code()):
                        raise PermissionError("Study access code required")
                    token = protocol.create()
                result = protocol.present(protocol.checked(token))
            elif action == "state":
                result = protocol.present(protocol.checked(token))
            elif action == "export":
                if not is_local(request):
                    raise PermissionError("Research exports are available only on the study machine")
                protocol.checked(token)
                result = protocol.store.export(token)
            elif action == "handoff" and protocol.checked(token)["stage"] != "sheet-questionnaire":
                if protocol.checked(token)["sheet_page"] != "handoff" or not protocol.continuation:
                    raise Conflict("The short questionnaire is not complete")
                result = protocol.present(protocol.checked(token))
            else:
                result = protocol.mutate(token, action, payload or {})
            target = None
            if action == "handoff" and protocol.continuation:
                target = f"/viewing/{result['session_id']}/"
                result = {**result, "next_url": target}
            reply = JSONResponse(result)
            reply.headers["cache-control"] = "no-store"
            # Keep the same validated credential across normal browser restarts
            # and renew its lifetime while either questionnaire endpoint is used.
            reply.set_cookie(
                protocol.cookie,
                token,
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
                max_age=SESSION_COOKIE_MAX_AGE,
                path="/",
            )
            if target and protocol.continuation:
                reply.set_cookie(
                    protocol.continuation.app.state.cookie_name,
                    token,
                    httponly=True,
                    samesite="strict",
                    secure=request.url.scheme == "https",
                    max_age=SESSION_COOKIE_MAX_AGE,
                    path=target,
                )
            return reply
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=409 if isinstance(exc, Conflict) else 400)

    @app.post("/api/questionnaire/session")
    def enroll(request: Request, payload: dict[str, Any]) -> JSONResponse:
        return response(request, "session", payload)

    @app.get("/api/questionnaire/state")
    def current(request: Request) -> JSONResponse:
        return response(request, "state")

    @app.get("/api/questionnaire/export")
    def export(request: Request) -> JSONResponse:
        return response(request, "export")

    @app.get("/api/questionnaire/practice-frame/{index}")
    def practice_frame(request: Request, index: int) -> Any:
        try:
            state = protocol.checked(request.cookies.get(protocol.cookie, ""))
            frames = state["sheet_config"].get("practice", {}).get("frames", [])
            if state["sheet_page"] != "practice" or not 0 <= index < len(frames):
                raise PermissionError("Practice frame unavailable")
            return FileResponse(
                media_path(media, frames[index]["still"]), headers={"cache-control": "no-store"}
            )
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/questionnaire/{action}")
    def submit(request: Request, action: str, payload: dict[str, Any]) -> JSONResponse:
        return response(request, action, payload)
