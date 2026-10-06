"""Questionnaire specifications derived from the four-tab research workbook.

The original cells are bundled with the instrument so edits, unresolved notes,
and references stay available alongside the exact displayed Korean wording.
This module does not alter the legacy pilot instrument or infer missing stimuli.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from typing import Any

SCHEMA = "dpo.regen-sheet-instrument/v1"
PROVENANCE = "google-sheet-1JP44YNkE2hgOTHUUJ69WrseeD8_fqXHDxYZ5_cMI-5Q-2026-10-04"
UNKNOWN_MEMORY = {"value": "memory_unknown", "label": "기억나지 않음", "exclusive": True}
PAGE_KINDS = {
    "P0": "intro",
    "P1": "video",
    "P2": "visual",
    "P3": "audio",
    "P4": "survey",
    "P5": "survey",
    "P6": "video",
    "P7": "visual",
    "P8": "audio",
    "P9": "survey",
    "P10": "survey",
    "P11": "survey",
    "P12": "interview",
    "P13": "debrief",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def _catalogue() -> dict[str, Any]:
    workbook = json.loads(files("dpo.regen").joinpath("items/sheet-workbook.json").read_text("utf-8"))
    sheets = {sheet["properties"]["title"]: sheet["values"] for sheet in workbook["sheets"]}
    labels = sheets["척도라벨"]
    recovery_labels = [str(row[2]) for row in labels[2:9]]
    bipolar_labels = [str(row[1]) for row in labels[20:27]]
    authored_labels = {row[0]: [row[1], "", "", row[2], "", "", row[3]] for row in labels[12:17]}
    headers = sheets["문항표"][0]
    pages: dict[str, Any] = {}
    notes = []
    for row_number, cells in enumerate(sheets["문항표"][1:], 2):
        row = dict(zip(headers, cells, strict=False))
        source_row = f"문항표!{row_number}"
        if row.get("확인 필요"):
            notes.append({"source_row": source_row, "text": row["확인 필요"]})
        page_id = row.get("페이지")
        if page_id not in PAGE_KINDS:
            continue
        page = pages.setdefault(
            page_id,
            {
                "id": page_id,
                "title": row["페이지명"],
                "stage": row["스테이지"],
                "kind": PAGE_KINDS[page_id],
                "instruction": "",
                "description": "",
                "language": "ko",
                "items": [],
                "source_rows": [],
                "notes": [],
            },
        )
        page["source_rows"].append(source_row)
        if row.get("확인 필요"):
            page["notes"].append(row["확인 필요"])
        if row["유형"] == "안내":
            page["instruction"] = row.get("한국어 (수정안)") or ""
        elif row["유형"] == "화면":
            page["description"] = row.get("한국어 (수정안)") or ""
        elif row["유형"] == "문항":
            item_id = row["코드"]
            item: dict[str, Any] = {
                "id": item_id,
                "text": row["한국어 (수정안)"],
                "required": True,
                "reverse": row.get("역채점") == "Y",
                "subscale": row.get("하위척도"),
                "source": row.get("출처"),
                "source_row": source_row,
                "source_row_hash": _digest(cells),
                "source_cells": row,
            }
            response_format = row.get("응답 형식") or ""
            if response_format.startswith("리커트"):
                item_labels = (
                    recovery_labels
                    if page_id in {"P4", "P9"}
                    else bipolar_labels
                    if item_id == "S3"
                    else authored_labels[item_id]
                )
                item.update(type="rating", points=7, min=0, max=6, labels=list(item_labels))
                item["scale_source"] = (
                    "척도라벨!C3:C9"
                    if page_id in {"P4", "P9"}
                    else ("척도라벨!B21:B27" if item_id == "S3" else "척도라벨!A13:E17")
                )
                if item_id == "S3":
                    item["neutral_value"] = 3
            elif item_id == "FA1":
                item.update(
                    type="choice",
                    options=[
                        {"value": "yes", "label": "있다"},
                        {"value": "no", "label": "없다"},
                        {"value": "unsure", "label": "잘 모르겠다"},
                    ],
                )
            elif item_id in {"A1s", "A2s"}:
                item.update(type="choice", options=[], requires_options=True)
            elif item_id == "R2":
                item.update(type="multi", options=[dict(UNKNOWN_MEMORY)], requires_options=True)
            elif item_id in {"V1", "V2"}:
                item.update(type="visual", min_selections=1, max_selections=1)
            else:
                item.update(type="oral", required=False)
                if item_id == "D2":
                    item["condition"] = "D1 reports a change"
            page["items"].append(item)
    pages["P0"]["requires_config"] = ["consent_text", "practice"]
    pages["P9"]["randomize_items"] = True
    pages["P12"]["after"] = "both-phases"
    pages["P12"]["response_mode"] = "oral-interview"
    pages["P13"]["requires_config"] = ["debrief_text"]
    references = [
        {"citation": row[0], "doi": row[1], "use": row[2], "source_row": f"참고문헌!{index}"}
        for index, row in enumerate(sheets["참고문헌"][1:], 2)
    ]
    return {
        "schema": SCHEMA,
        "provenance": PROVENANCE,
        "language": "ko",
        "source_url": workbook["source_url"],
        "workbook_hash": _digest(workbook),
        "source_workbook": workbook,
        "pages": pages,
        "references": references,
        "notes": notes,
        "translation_status": "revised-draft-not-back-translated",
        "phase1_pages": [f"P{index}" for index in range(1, 12)],
        "phase2": {
            "surveys": [],
            "collection": "interaction-logs-only",
            "decision_source": "user-2026-10-04",
            "source_note": sheets["문항표"][41][9],
        },
        "post_study_pages": ["P12", "P13"],
        "practice": {"required": True, "exclude_from_analysis": True},
    }


def catalogue() -> dict[str, Any]:
    """Return independently mutable specs and every source workbook cell."""
    return copy.deepcopy(_catalogue())


def snapshot() -> dict[str, Any]:
    """Freeze wording, source notes, labels and the phase-two user decision."""
    document = catalogue()
    return {**document, "hash": _digest(document)}


def page_item_order(page: str, session_key: str) -> list[str]:
    """Reproduce P9 order across reloads without Python PRNG/version coupling.

    Callers persist the returned order and snapshot with the participant. Include
    the video ID in session_key when each video needs its own fixed permutation.
    """
    spec = _catalogue()["pages"].get(page)
    if spec is None:
        raise ValueError(f"unknown page: {page}")
    ids = [str(item["id"]) for item in spec["items"]]
    if spec.get("randomize_items"):
        if not isinstance(session_key, str) or not session_key:
            raise ValueError("a nonempty session key is required for randomized item order")
        return sorted(ids, key=lambda item_id: _digest([PROVENANCE, session_key, page, item_id]))
    return ids


def _options(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise ValueError("per-video options must be a nonempty list")
    result = []
    seen = set()
    for value in values:
        option = {"value": value, "label": value} if isinstance(value, str) else copy.deepcopy(value)
        if not isinstance(option, dict) or not all(
            isinstance(option.get(key), str) and option[key].strip() for key in ("value", "label")
        ):
            raise ValueError("options require nonempty string value and label")
        if option["value"] in seen:
            raise ValueError("duplicate options are not allowed")
        seen.add(option["value"])
        if "exclusive" in option and not isinstance(option["exclusive"], bool):
            raise ValueError("option exclusive must be boolean")
        result.append(option)
    return result


def page_spec(
    page: str,
    language: str = "ko",
    *,
    session_key: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render source Korean; expose missing authoring instead of inventing it.

    Dynamic options are keyed by item ID and supplied by the video manifest.
    Source wording has no approved English translation, so other UI languages
    retain Korean questionnaire text and explicitly record the requested locale.
    """
    if page not in _catalogue()["pages"]:
        raise ValueError(f"unknown page: {page}")
    spec = copy.deepcopy(_catalogue()["pages"][page])
    spec["requested_language"] = language
    for item in spec["items"]:
        if not item.get("requires_options"):
            continue
        supplied = (options or {}).get(item["id"])
        if supplied is None and item["id"] == "A2s":
            supplied = (options or {}).get("A1s")
        if supplied is not None:
            item["options"] = _options(supplied)
            if item["id"] == "R2":
                item["options"] = [
                    option
                    for option in item["options"]
                    if option["value"] != UNKNOWN_MEMORY["value"]
                    and option["label"] != UNKNOWN_MEMORY["label"]
                ]
                if not item["options"]:
                    raise ValueError("R2 requires video-specific sound options")
                item["options"].append(dict(UNKNOWN_MEMORY))
            item["requires_options"] = False
    if session_key is not None:
        order = page_item_order(page, session_key)
        spec["items"] = sorted(spec["items"], key=lambda item: order.index(item["id"]))
    spec["item_order"] = [item["id"] for item in spec["items"]]
    return dict(spec)


def validate_answers(
    page: str,
    answers: Mapping[str, Any],
    *,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate strict page-complete responses; zero is an ordinary valid rating."""
    spec = page_spec(page, options=options)
    if not isinstance(answers, Mapping):
        raise ValueError("answers must be an object")
    items = {item["id"]: item for item in spec["items"]}
    unknown = set(answers) - set(items)
    if unknown:
        raise ValueError(f"unknown answers for {page}: {sorted(unknown)}")
    missing = {key for key, item in items.items() if item["required"] and key not in answers}
    if missing:
        raise ValueError(f"missing answers for {page}: {sorted(missing)}")
    result: dict[str, Any] = {}
    for key, value in answers.items():
        item = items[key]
        if item.get("requires_options"):
            raise ValueError(f"{key}: per-video options are not configured")
        kind = item["type"]
        if kind == "rating":
            if type(value) is not int or not item["min"] <= value <= item["max"]:
                raise ValueError(f"{key}: rating must be an integer from 0 to 6")
        elif kind in {"choice", "multi"}:
            allowed = {option["value"] for option in item["options"]}
            if kind == "choice":
                if not isinstance(value, str) or value not in allowed:
                    raise ValueError(f"{key}: select exactly one configured option")
            else:
                if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                    raise ValueError(f"{key}: select at least one configured option")
                if len(set(value)) != len(value) or not set(value) <= allowed:
                    raise ValueError(f"{key}: invalid or duplicated option")
                exclusive = {option["value"] for option in item["options"] if option.get("exclusive")}
                if len(value) > 1 and set(value) & exclusive:
                    raise ValueError(f"{key}: an exclusive option must be selected alone")
        elif kind == "visual":
            if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
                raise ValueError(f"{key}: select exactly one visual point")
            point = value[0]
            if not isinstance(point.get("frame_id"), str) or not point["frame_id"]:
                raise ValueError(f"{key}: frame_id is required")
            for axis in ("x", "y"):
                coordinate = point.get(axis)
                if (
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, (int, float))
                    or not 0 <= coordinate <= 1
                ):
                    raise ValueError(f"{key}: {axis} must be a normalized coordinate")
        elif kind == "oral":
            raise ValueError(f"{key}: collect this as a researcher-led oral interview, not a survey")
        result[key] = copy.deepcopy(value)
    return result
