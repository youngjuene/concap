"""Phase 1 in diff: regen's pages, one viewing per clip, for the live tuner's profile.

Researcher's decisions of 2026-10-18 (plan "Phase 1 — 2026-10-18"):
- **Pages:** P0, then practice, then for each clip ``P1 P2 P3 P4 P5 P10 P11``,
  then the handoff to live phase 2.
- **Clip order:** random per participant, from a seed recorded in the state.
- **One viewing:** the workbook's second viewing is dropped, so P10 (S1–S3)
  and P11 (R1, R2) follow P5 and bind to the P1 viewing.

What regen supplies and what is copied:
- **Items, option rules and answer validation** come from
  ``dpo.regen.sheet_instrument``; P2 points are matched against the masks by
  ``dpo.regen.points``. Every item therefore matches regen exactly.
- **The playback coverage rule** is copied from
  ``dpo.regen.playback.record_playback``, because importing it would load
  ``study_store``, ``study_schema`` and ``config``. A parity test runs regen's
  rule on the same ticks.
- **The actions** follow regen's ``sheet_protocol``: ``intro-complete``,
  ``practice-complete``, ``playback``, ``view-ended`` and ``submit``.

**Blindness:** a page payload carries no clip key, sound family or R2 answer
key. Media are named per page (``video``, ``frame-0``, …), and the server turns
each name into an opaque URL.
"""

from __future__ import annotations

import copy
import random
import secrets
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import dpo.regen.points as points
import dpo.regen.sheet_instrument as sheet_instrument
from diff.phase1_setup import stripped_options
from diff.session_store import Conflict
from diff.study_protocol import sha256_of

PROTOCOL: Final = "diff.phase1/v1"
CLIP_PAGES: Final = ("P1", "P2", "P3", "P4", "P5", "P10", "P11")
SUBMIT_PAGES: Final = frozenset({"P2", "P3", "P4", "P5", "P10", "P11"})
ITEM_FIELDS: Final = frozenset(
    {"id", "text", "type", "required", "min", "max", "points", "labels", "options", "requires_options"}
)
OPTION_FIELDS: Final = frozenset({"value", "label", "exclusive"})
# Regen's practice page, word for word (sheet_protocol.present).
PRACTICE_TITLE: Final = "연습 시행"
PRACTICE_NOTE: Final = "연습 응답은 분석에서 제외됩니다."
PRACTICE_TEXTS: Final = {
    "practice_rating": "연습: 선택지를 골라 응답 방법을 확인해 주세요.",
    "practice_sound": "연습: 선택지 하나를 골라 주세요.",
    "practice_visual": "연습 이미지에서 한 곳을 선택해 주세요.",
}
HANDOFF_TITLE: Final = "1스테이지 완료"
UNWATCHED: Final = "영상을 끝까지 시청한 뒤 진행해 주세요."
RATINGS: Final = ("S1", "S2", "S3")


def checkpoint() -> dict[str, Any]:
    return {"position_ms": 0, "sequence": -1, "coverage": [], "epoch": 0, "last_tick": None}


def merged(intervals: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _position(value: Any, duration: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a number")
    if value != value or not 0 <= value <= duration:
        raise ValueError(f"Expected a finite number between 0 and {duration}")
    return float(value)


def record_playback(state: dict[str, Any], data: Mapping[str, Any], duration: float, now: float) -> None:
    """Regen's coverage accounting (``dpo.regen.playback.record_playback``), with the clock passed in."""
    sequence = data.get("sequence")
    if type(sequence) is not int or sequence <= state["sequence"]:
        raise Conflict("Playback update is out of order")
    position = _position(data.get("position_ms"), duration)
    previous = state.get("last_tick")
    covered = sum(end - begin for begin, end in state["coverage"])
    clock = (
        previous.get("clock", {"at": previous["at"], "covered": covered})
        if previous and previous["playing"]
        else {"at": max(previous["at"], now - 2) if previous else now, "covered": covered}
    )
    if data.get("seek") is True:
        state["epoch"] += 1
        clock = {"at": max(previous["at"], now - 2) if previous else now, "covered": covered}
    elif previous and previous["playing"] and not data.get("hidden", False):
        delta = position - previous["position"]
        proposed = merged(state["coverage"] + [[previous["position"], position]])
        credited = sum(end - begin for begin, end in proposed) - clock["covered"]
        elapsed = (now - clock["at"]) * 1000
        if not (0 <= delta <= 7000 and credited <= elapsed * 1.25 + 300):
            raise Conflict("Resume playback from the last saved position after the interruption.")
        state["coverage"] = proposed
    state["position_ms"], state["sequence"] = position, sequence
    state["last_tick"] = {
        "position": position,
        "at": now,
        "playing": data.get("playing") is True and not data.get("hidden", False),
        "clock": clock,
    }


def clip_order(keys: Sequence[str], seed: int) -> list[str]:
    return random.Random(seed).sample(list(keys), len(keys))


class Phase1:
    """Pages, actions and records for one setup; the session's state lives in the store."""

    def __init__(self, setup: Mapping[str, Any], *, root: Path, instrument: Any = sheet_instrument) -> None:
        self.setup = setup
        self.root = root
        self.instrument = instrument
        self.clips: dict[str, dict[str, Any]] = {clip["key"]: clip for clip in setup["clips"]}

    def new_state(self, session_id: str, participant: str, seed: int | None = None) -> dict[str, Any]:
        seed = secrets.randbits(64) if seed is None else seed
        return {
            "protocol": PROTOCOL,
            "session_id": session_id,
            "participant": participant,
            "stage": "phase1",
            "page": "P0",
            "clip_index": 0,
            "order_seed": seed,
            "clip_order": clip_order(sorted(self.clips), seed),
            "instrument_hash": self.instrument.snapshot()["hash"],
            "setup_sha256": sha256_of(self.setup),
            "consent": None,
            "practice": [],
            "viewings": [],
            "responses": [],
            "playback": {},
            "revision": 0,
        }

    def clip(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return self.clips[state["clip_order"][state["clip_index"]]]

    def options(self, state: Mapping[str, Any]) -> dict[str, Any]:
        sounds = self.clip(state)["options"]
        return {"A1s": stripped_options(sounds["A1s"]), "R2": stripped_options(sounds["R2"])}

    def track(self, clip: Mapping[str, Any]) -> list[dict[str, Any]]:
        caption = clip["caption"]
        return [
            {
                "index": 0,
                "start_ms": caption["start_ms"],
                "end_ms": caption["end_ms"],
                "text": caption["text"],
                "language": "ko",
            }
        ]

    def media_files(self, state: Mapping[str, Any]) -> dict[str, Path]:
        """Every file the current page may fetch, by the name its URL is built from."""
        page = state.get("page")
        if state.get("stage") != "phase1":
            return {}
        if page == "practice":
            frames = self.setup["practice"]["frames"]
            return {f"practice-{i}": self.root / frame["still"] for i, frame in enumerate(frames)}
        if page == "P1":
            return {"video": self.root / self.clip(state)["video"]}
        if page == "P2":
            frames = self.clip(state)["frames"]
            return {f"frame-{i}": self.root / frame["still"] for i, frame in enumerate(frames)}
        return {}

    def _title(self, page: str, workbook: str) -> str:
        override = self.setup["wording"]["titles"].get(page)
        return str(override["shown"]) if override else workbook

    def _practice_items(self) -> list[dict[str, Any]]:
        page_spec = self.instrument.page_spec
        rating = copy.deepcopy(page_spec("P4")["items"][0])
        sound = page_spec("P3", options={"A1s": stripped_options(self.setup["practice"]["options"]["A1s"])})
        sound = sound["items"][0]
        point = copy.deepcopy(page_spec("P2")["items"][0])
        items = []
        for item, item_id in (
            (rating, "practice_rating"),
            (sound, "practice_sound"),
            (point, "practice_visual"),
        ):
            item.update(id=item_id, text=PRACTICE_TEXTS[item_id])
            items.append(item)
        return items

    def present(self, state: Mapping[str, Any], url: Callable[[str], str]) -> dict[str, Any]:
        """The page payload for the browser: no clip key, family or answer key."""
        if state.get("stage") != "phase1":
            raise Conflict("phase 1 is complete")
        page = state["page"]
        result: dict[str, Any] = {
            "revision": state["revision"],
            "protocol": PROTOCOL,
            "page": page,
            "clip_index": state["clip_index"],
            "clip_count": len(state["clip_order"]),
            "practice": page == "practice",
            "instrument_hash": state["instrument_hash"],
            "items": [],
        }
        if page == "practice":
            result.update(kind="practice", title=PRACTICE_TITLE, instruction=PRACTICE_NOTE)
            result["items"] = self._practice_items()
            result["frames"] = [
                {"id": str(i), "index": i, "at_ms": frame.get("at_ms", 0), "url": url(f"practice-{i}")}
                for i, frame in enumerate(self.setup["practice"]["frames"])
            ]
        elif page == "handoff":
            result.update(kind="handoff", title=HANDOFF_TITLE, instruction="")
        else:
            options = {} if page == "P0" else self.options(state)
            spec = self.instrument.page_spec(page, options=options, session_key=state["session_id"])
            result.update(
                kind=spec["kind"],
                title=self._title(page, spec["title"]),
                instruction=spec["instruction"],
                language=spec["language"],
                items=spec["items"],
            )
        if page == "P0":
            result.update(
                kind="intro",
                instruction=self.setup["wording"]["P0_instruction"]["shown"],
                consent_text=self.setup.get("consent_text"),
            )
        if page == "P1":
            clip = self.clip(state)
            saved = state["playback"].get(f"{state['clip_index']}:P1", checkpoint())
            result.update(
                kind="view",
                media={
                    "url": url("video"),
                    "duration_ms": clip["duration_ms"],
                    "captions": self.track(clip),
                    "playback": {key: value for key, value in saved.items() if key != "last_tick"},
                },
            )
        if page == "P2":
            result["frames"] = [
                {"id": str(i), "index": i, "at_ms": frame["at_ms"], "url": url(f"frame-{i}")}
                for i, frame in enumerate(self.clip(state)["frames"])
            ]
        result["items"] = [
            {key: value for key, value in item.items() if key in ITEM_FIELDS} for item in result["items"]
        ]
        for item in result["items"]:
            if item.get("options") is not None:
                item["options"] = [
                    {key: value for key, value in option.items() if key in OPTION_FIELDS}
                    for option in item["options"]
                ]
        return result

    def _view_id(self, state: Mapping[str, Any]) -> str:
        return f"{state['session_id']}-{state['clip_index']}-P1"

    def _advance(self, state: dict[str, Any]) -> None:
        page = state["page"]
        if page != "P11":
            state["page"] = CLIP_PAGES[CLIP_PAGES.index(page) + 1]
        elif state["clip_index"] + 1 < len(state["clip_order"]):
            state["clip_index"] += 1
            state["page"] = "P1"
        else:
            state["page"] = "handoff"

    def apply(
        self, state: dict[str, Any], action: str, data: Mapping[str, Any], now: float
    ) -> dict[str, Any]:
        """One browser action on ``state`` (in place); returns the event to store. Raises to roll back."""
        if state.get("stage") != "phase1":
            raise Conflict("phase 1 is complete")
        page = state["page"]
        if data.get("page") != page:
            raise Conflict("This answer belongs to another page")
        formal = page in CLIP_PAGES
        event = {
            "event": action,
            "phase": 1 if formal else 0,
            "page": page,
            "clip_index": state["clip_index"] if formal else None,
            "view_id": self._view_id(state) if formal else None,
            "analysis_excluded": not formal,
            "data": dict(data),
        }
        if action == "intro-complete":
            self._intro(state, data, now)
        elif action == "practice-complete":
            self._practice(state, data, now)
        elif action in {"playback", "view-ended"}:
            self._viewing(state, action, data, now)
        elif action == "submit":
            self._submit(state, data, now)
        else:
            raise ValueError("Unknown questionnaire action")
        # A next page that cannot render must roll the answer back (regen does the same).
        self.present(state, lambda name: name)
        return event

    def _intro(self, state: dict[str, Any], data: Mapping[str, Any], now: float) -> None:
        if state["page"] != "P0":
            raise Conflict("The introduction is already complete")
        consent = self.setup.get("consent_text")
        if consent is not None and data.get("consented") is not True:
            raise Conflict("Configured consent is required")
        state["consent"] = {
            "accepted": consent is not None,
            "text_sha256": None if consent is None else sha256_of(consent),
            "at": now,
        }
        state["page"] = "practice"

    def _practice(self, state: dict[str, Any], data: Mapping[str, Any], now: float) -> None:
        if state["page"] != "practice":
            raise Conflict("Practice is only completed once")
        raw = data.get("answers", {})
        if not isinstance(raw, dict) or set(raw) != set(PRACTICE_TEXTS):
            raise ValueError("Complete the three practice responses")
        rating = raw.get("practice_rating")
        if type(rating) is not int or not 0 <= rating <= 6:
            raise ValueError("연습 척도에서 0~6 중 하나를 선택해 주세요.")
        sounds = stripped_options(self.setup["practice"]["options"]["A1s"])
        self.instrument.validate_answers("P3", {"A1s": raw.get("practice_sound")}, options={"A1s": sounds})
        visual = self.instrument.validate_answers("P2", {"V1": raw.get("practice_visual")})["V1"]
        points.parse_points(
            [dict(point, frame=int(point["frame_id"])) for point in visual],
            len(self.setup["practice"]["frames"]),
        )
        state["practice"].append({"answers": copy.deepcopy(raw), "analysis_excluded": True, "at": now})
        state["page"] = "P1"

    def _viewing(self, state: dict[str, Any], action: str, data: Mapping[str, Any], now: float) -> None:
        if state["page"] != "P1":
            raise Conflict("This page has no viewing")
        clip = self.clip(state)
        duration = clip["duration_ms"]
        saved = state["playback"].setdefault(f"{state['clip_index']}:P1", checkpoint())
        if action == "playback":
            record_playback(saved, data, duration, now)
            saved.setdefault("started_at", now)
            return
        covered = sum(end - begin for begin, end in saved["coverage"])
        if covered < duration - min(300, duration * 0.05) or saved["position_ms"] < duration - 100:
            raise Conflict(UNWATCHED)
        captions = self.track(clip)
        state["viewings"].append(
            {
                "view_id": self._view_id(state),
                "clip_index": state["clip_index"],
                "clip": clip["key"],
                "page": "P1",
                "captions": captions,
                "caption_sha256": sha256_of(captions),
                "coverage": saved["coverage"],
                "started_at": saved.get("started_at"),
                "ended_at": now,
            }
        )
        self._advance(state)

    def _submit(self, state: dict[str, Any], data: Mapping[str, Any], now: float) -> None:
        page = state["page"]
        if page not in SUBMIT_PAGES:
            raise Conflict("This page has no questionnaire")
        raw = data.get("answers")
        if not isinstance(raw, dict):
            raise ValueError("Answers must be an object")
        options = self.options(state)
        submitted = self.instrument.validate_answers(page, raw, options=options)
        rendered = self.instrument.page_spec(page, options=options, session_key=state["session_id"])
        clip = self.clip(state)
        record: dict[str, Any] = {
            "page": page,
            "clip_index": state["clip_index"],
            "clip": clip["key"],
            "view_id": self._view_id(state),
            "answers": submitted,
            "items": rendered["items"],
            "presented_order": [item["id"] for item in rendered["items"]],
            "instrument_hash": state["instrument_hash"],
            "submitted_at": now,
            "client_submitted_at": data.get("submitted_at"),
            "analysis_excluded": False,
        }
        if page == "P2":
            frames = clip["frames"]
            measured = points.measure_points(
                points.parse_points(
                    [dict(p, frame=int(p["frame_id"])) for p in submitted["V1"]], len(frames)
                ),
                {
                    i: tuple(
                        points.MaskObject(item["id"], item["label"], self.root / item["mask"])
                        for item in frame["objects"]
                    )
                    for i, frame in enumerate(frames)
                },
                {i: self.root / frame["still"] for i, frame in enumerate(frames)},
            )
            matches = tuple(measurement.match for measurement in measured)
            record["matches"] = [match.record() for match in matches]
            record["labels"] = list(points.matched_labels(matches))
            record["visual_metrics"] = [measurement.metrics for measurement in measured]
        state["responses"].append(record)
        self._advance(state)

    def profile_inputs(self, state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Per clip: the heard family (A1s) and matched labels (P2), and the S1–S3 ratings (P10)."""
        observations: list[dict[str, Any]] = []
        ratings: list[dict[str, Any]] = []
        for index, key in enumerate(state["clip_order"]):
            records = {r["page"]: r for r in state["responses"] if r["clip_index"] == index}
            if not {"P2", "P3", "P10"} <= set(records):
                continue
            families = {o["value"]: o["family"] for o in self.clips[key]["options"]["A1s"]}
            heard = families.get(records["P3"]["answers"]["A1s"])
            observations.append({"heard": {heard: True} if heard else {}, "labels": records["P2"]["labels"]})
            ratings.append({item: records["P10"]["answers"][item] for item in RATINGS})
        return observations, ratings

    def hand_off(self, state: dict[str, Any], calibration: Mapping[str, Any], now: float) -> dict[str, Any]:
        """Close phase 1 with the calibration record built from it; returns the event to store."""
        if state.get("stage") != "phase1" or state["page"] != "handoff":
            raise Conflict("Complete the short clips first")
        state.update(stage="phase2", calibration=dict(calibration), handoff_at=now)
        return {"event": "handoff", "phase": 1, "calibration_sha256": calibration.get("sha256")}
