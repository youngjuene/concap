"""Pure, source-preserving Phase 1 trend analysis; never a correctness score.

Inputs are public session payloads, not credentials or database connections.
Historical mask measurements are never reconstructed from mutable files.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from dpo.regen.study_schema import digest

SCHEMA = "dpo.phase1-analysis/v1"
SCOPE_FIELDS = [
    "clip_id",
    "instrument_hash",
    "codebook_version",
    "codebook_hash",
    "av_assigned_condition",
    "description_depth",
    "caption_strategy",
]
OBSERVATION_FIELDS = [
    "participant",
    "session_id",
    "clip_index",
    "round",
    "view_id",
    *SCOPE_FIELDS,
    "status",
    "issues",
    "visual_status",
    "visual_points",
    "selected_mask_id",
    "visual_label",
    "frame_id",
    "frame_sha256",
    "mask_sha256",
    "frame_width",
    "frame_height",
    "mask_width",
    "mask_height",
    "selected_mask_area_px",
    "frame_area_px",
    "selected_mask_area_ratio",
    "candidate_masks",
    "matching_rule",
    "threshold",
    "sound_raw",
    "sound_label",
    "sound_type",
    "sound_family",
    "sound_status",
    "offered_options",
    "offered_sound_types",
    "offered_sound_families",
    "unmapped_offered_sound_types",
    "av_selection_relation",
    "visual_submitted_at",
    "sound_submitted_at",
]
PAIRED_FIELDS = [
    "view_id",
    "sound_raw",
    "sound_label",
    "sound_type",
    "sound_family",
    "selected_mask_area_ratio",
    "frame_id",
    "frame_sha256",
    "selected_mask_id",
    "visual_label",
    "av_selection_relation",
]
PAIR_FIELDS = [
    "participant",
    "session_id",
    "clip_index",
    *SCOPE_FIELDS,
    "status",
    "issues",
    *[f"{field}_{round_}" for field in PAIRED_FIELDS for round_ in (1, 2)],
    "area_delta_pp",
    "type_changed",
    "family_changed",
    "frame_changed",
    "menu_changed",
    "order_changed",
]
PAGES = {"P2": (1, "visual"), "P3": (1, "sound"), "P7": (2, "visual"), "P8": (2, "sound")}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: Any) -> float | int | None:
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def codebook_for_clip(clipconfig: dict[str, Any]) -> dict[str, Any]:
    """Freeze explicit analysis codes; arbitrary research family names are valid.

    An absent type map means identity. An explicit partial/invalid type map
    never silently becomes identity. Legacy family maps use raw option IDs.
    """
    clipconfig = _mapping(clipconfig)
    issues: list[str] = []
    explicit = clipconfig.get("analysis_codebook", {})
    invalid = not isinstance(explicit, dict)
    if invalid:
        issues.append("analysis_codebook_not_mapping")
    book = _mapping(explicit)
    version = _text(book.get("version")) or "frozen-config-v1"
    if book and not _text(book.get("version")):
        issues.append("codebook_version_missing")

    def strings(raw: Any, name: str) -> dict[str, str]:
        if not isinstance(raw, dict):
            issues.append(f"{name}_not_mapping")
            return {}
        clean = {key: value for key, value in raw.items() if _text(key) and _text(value)}
        if len(clean) != len(raw):
            issues.append(f"{name}_invalid_entries")
        return clean

    identity = "sound_types" not in book and not invalid
    types = strings(book.get("sound_types", {}), "sound_types")
    families = strings(book.get("sound_families", clipconfig.get("source_families", {})), "sound_families")
    if "sound_families" not in book and not identity:
        converted: dict[str, str] = {}
        conflicts: set[str] = set()
        for raw, family in families.items():
            canonical = types.get(raw)
            if canonical is None:
                issues.append("legacy_family_type_unmapped")
            elif canonical in converted and converted[canonical] != family:
                conflicts.add(canonical)
            else:
                converted[canonical] = family
        for key in conflicts:
            converted.pop(key, None)
            issues.append("legacy_family_alias_conflict")
        families = converted
    relations: dict[str, dict[str, int]] = {}
    raw_relations = book.get("av_relations", {})
    if not isinstance(raw_relations, dict):
        issues.append("av_relations_not_mapping")
    else:
        for visual, values in raw_relations.items():
            if not _text(visual) or not isinstance(values, dict):
                issues.append("av_relations_invalid_entries")
                continue
            clean = {
                key: value
                for key, value in values.items()
                if _text(key) and type(value) is int and value in (0, 1)
            }
            if len(clean) != len(values):
                issues.append("av_relations_invalid_entries")
            relations[visual] = clean
    result = {
        "version": version,
        "sound_types": types,
        "identity_sound_types": identity,
        "sound_families": families,
        "av_relations": relations,
        "issues": sorted(set(issues)),
    }
    return {**result, "hash": digest(result)}


def _canonical(raw: str | None, book: dict[str, Any]) -> str | None:
    return raw if book["identity_sound_types"] else book["sound_types"].get(raw)


def _options(record: dict[str, Any], item_id: str) -> list[dict[str, str]] | None:
    items = record.get("items")
    if not isinstance(items, list):
        return None
    matching = [item for item in items if isinstance(item, dict) and item.get("id") == item_id]
    if len(matching) != 1 or not isinstance(matching[0].get("options"), list):
        return None
    result = []
    for option in matching[0]["options"]:
        if not isinstance(option, dict) or not _text(option.get("value")) or not _text(option.get("label")):
            return None
        result.append({"value": option["value"], "label": option["label"]})
    return result if len({o["value"] for o in result}) == len(result) and result else None


def _visual(row: dict[str, Any], record: dict[str, Any], round_: int) -> None:
    points = _mapping(record.get("answers")).get("V1" if round_ == 1 else "V2", [])
    row["visual_points"] = (
        [
            {"frame_id": _text(p.get("frame_id")), "x": _number(p.get("x")), "y": _number(p.get("y"))}
            for p in points
            if isinstance(p, dict)
        ]
        if isinstance(points, list)
        else []
    )
    metrics = record.get("visual_metrics")
    if not isinstance(metrics, list) or len(metrics) != 1 or not isinstance(metrics[0], dict):
        row["visual_status"] = "metrics_missing" if metrics is None else "metrics_ambiguous"
        row["issues"].append(row["visual_status"])
        if metrics is None:
            if len(row["visual_points"]) == 1:
                row["frame_id"] = row["visual_points"][0]["frame_id"]
            matches = record.get("matches")
            if isinstance(matches, list) and len(matches) == 1 and isinstance(matches[0], dict):
                match = matches[0]
                frame = match.get("frame")
                frame_id = str(frame) if type(frame) is int and frame >= 0 else None
                if frame_id is not None and row["frame_id"] in (None, frame_id):
                    row["frame_id"] = frame_id
                    row["selected_mask_id"] = _text(match.get("object_id"))
                    row["visual_label"] = _text(match.get("label"))
                elif frame_id is not None:
                    row["issues"].append("legacy_match_frame_conflict")
        return
    metric = metrics[0]
    for key in ("selected_mask_id", "frame_id", "frame_sha256", "mask_sha256", "matching_rule"):
        row[key] = _text(metric.get(key))
    row["visual_label"] = _text(metric.get("label"))
    row["visual_status"] = _text(metric.get("status")) or "metrics_invalid"
    for key in (
        "selected_mask_area_px",
        "frame_area_px",
        "threshold",
        "frame_width",
        "frame_height",
        "mask_width",
        "mask_height",
    ):
        row[key] = _number(metric.get(key))
    candidates = metric.get("candidate_masks", [])
    row["candidate_masks"] = (
        [
            {
                "object_id": _text(candidate.get("object_id")),
                "label": _text(candidate.get("label")),
                "mask_area_px": _number(candidate.get("mask_area_px")),
                "mask_area_ratio": _number(candidate.get("mask_area_ratio")),
                "mask_sha256": _text(candidate.get("mask_sha256")),
                "mask_width": _number(candidate.get("mask_width")),
                "mask_height": _number(candidate.get("mask_height")),
                "status": _text(candidate.get("status")),
            }
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        if isinstance(candidates, list)
        else []
    )
    area, total, ratio = (
        row["selected_mask_area_px"],
        row["frame_area_px"],
        _number(metric.get("selected_mask_area_ratio")),
    )
    valid = (
        row["visual_status"] == "matched"
        and row["selected_mask_id"] is not None
        and area is not None
        and total is not None
        and ratio is not None
        and total > 0
        and 0 < area <= total
        and 0 < ratio <= 1
        and math.isclose(area / total, ratio, abs_tol=1e-12)
    )
    if valid:
        row["selected_mask_area_ratio"] = ratio
    elif row["visual_status"] == "matched":
        row["visual_status"] = "metrics_invalid"
        row["issues"].append("metrics_invalid")
    if row["visual_status"] == "unclassified":
        row["selected_mask_id"] = None


def _observation(
    session: dict[str, Any],
    index: int,
    round_: int,
    records: dict[str, list[dict[str, Any]]],
    book: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(OBSERVATION_FIELDS)
    row.update(
        participant=_text(session.get("participant")) or session["session_id"],
        session_id=session["session_id"],
        clip_index=index,
        round=round_,
        issues=list(book["issues"]),
        codebook_version=book["version"],
        codebook_hash=book["hash"],
        candidate_masks=[],
        visual_points=[],
        offered_options=None,
        offered_sound_types=[],
        offered_sound_families=[],
        unmapped_offered_sound_types=[],
        visual_status="missing",
        sound_status="missing",
    )
    pages = ("P2", "P3") if round_ == 1 else ("P7", "P8")
    all_records = [record for page in pages for record in records.get(page, [])]
    identity_valid = True
    for field in ("clip_id", "instrument_hash", "view_id", "caption_strategy"):
        values = {_text(record.get(field)) for record in all_records}
        row[field] = next(iter(values)) if len(values) == 1 else None
        if len(values) != 1 or row[field] is None:
            row["issues"].append(f"conflicting_or_missing_{field}")
            identity_valid = False
    for field, source in (
        ("av_assigned_condition", "audiovisual_congruence"),
        ("description_depth", "description_depth"),
    ):
        values = {_text(_mapping(record.get("stimulus_assignment")).get(source)) for record in all_records}
        row[field] = next(iter(values)) if len(values) == 1 else None
        if len(values) > 1:
            row["issues"].append(f"conflicting_{field}")
            identity_valid = False
    for page, kind in zip(pages, ("visual", "sound"), strict=True):
        options = records.get(page, [])
        if len(options) != 1:
            row["issues"].append(f"{'duplicate' if options else 'missing'}_{page}")
            row[f"{kind}_status"] = "ambiguous" if options else "missing"
            continue
        record = options[0]
        row[f"{kind}_submitted_at"] = _number(record.get("submitted_at"))
        if not identity_valid:
            row[f"{kind}_status"] = "identity_conflict"
            continue
        if kind == "visual":
            _visual(row, record, round_)
            continue
        item_id = "A1s" if round_ == 1 else "A2s"
        raw = _text(_mapping(record.get("answers")).get(item_id))
        row["sound_raw"] = raw
        menu = _options(record, item_id)
        row["offered_options"] = menu
        if menu is None:
            row["sound_status"] = "menu_missing_or_invalid"
            row["issues"].append(row["sound_status"])
            continue
        row["offered_sound_types"] = sorted(
            {code for option in menu if (code := _canonical(option["value"], book))}
        )
        row["offered_sound_families"] = sorted(
            {
                book["sound_families"][code]
                for code in row["offered_sound_types"]
                if code in book["sound_families"]
            }
        )
        row["unmapped_offered_sound_types"] = [
            code for code in row["offered_sound_types"] if code not in book["sound_families"]
        ]
        selected = next((option for option in menu if option["value"] == raw), None)
        if selected is None:
            row["sound_status"] = "selection_not_offered"
            row["issues"].append(row["sound_status"])
            continue
        row["sound_label"] = selected["label"]
        row["sound_type"] = _canonical(raw, book)
        row["sound_family"] = book["sound_families"].get(row["sound_type"])
        row["sound_status"] = "valid" if row["sound_type"] else "type_unmapped"
        if row["sound_family"] is None:
            row["issues"].append("selected_family_unmapped")
    if row["selected_mask_id"] and row["sound_type"]:
        row["av_selection_relation"] = (
            book["av_relations"].get(row["selected_mask_id"], {}).get(row["sound_type"])
        )
    row["issues"] = sorted(set(row["issues"]))
    row["status"] = "complete" if not row["issues"] else "partial"
    return row


def _pair(first: dict[str, Any] | None, second: dict[str, Any] | None) -> dict[str, Any]:
    rows = [row for row in (first, second) if row is not None]
    row: dict[str, Any] = dict.fromkeys(PAIR_FIELDS)
    row.update({key: rows[0][key] for key in ("participant", "session_id", "clip_index")})
    row["issues"] = [f"round{item['round']}:{issue}" for item in rows for issue in item["issues"]]
    compatible = True
    for field in SCOPE_FIELDS:
        values = {item[field] for item in rows}
        row[field] = next(iter(values)) if len(values) == 1 else None
        if len(values) > 1 or field in ("clip_id", "instrument_hash") and row[field] is None:
            compatible = False
            row["issues"].append(f"pair_conflicting_{field}")
    for round_, item in enumerate((first, second), 1):
        if item is None:
            row["issues"].append(f"round{round_}:missing")
        for field in PAIRED_FIELDS:
            row[f"{field}_{round_}"] = item[field] if item else None
    if compatible and first and second:
        for metric, field in (
            ("type_changed", "sound_type"),
            ("family_changed", "sound_family"),
            ("frame_changed", "frame_id"),
        ):
            if first[field] is not None and second[field] is not None:
                row[metric] = (
                    (first[field] != second[field])
                    if metric == "frame_changed"
                    else int(first[field] != second[field])
                )
        a, b = first["selected_mask_area_ratio"], second["selected_mask_area_ratio"]
        if a is not None and b is not None:
            row["area_delta_pp"] = 100 * (b - a)
        if first["offered_options"] is not None and second["offered_options"] is not None:
            menus = [
                [(option["value"], option["label"]) for option in item["offered_options"]]
                for item in (first, second)
            ]
            row["menu_changed"] = set(menus[0]) != set(menus[1])
            # Order is comparable only with the same raw option IDs. Wording
            # differences belong to menu_changed, not order_changed.
            orders = [[code for code, _ in menu] for menu in menus]
            if set(orders[0]) == set(orders[1]):
                row["order_changed"] = orders[0] != orders[1]
    row["issues"] = sorted(set(row["issues"]))
    row["status"] = "complete" if not row["issues"] else "partial"
    return row


def _scope(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in [*SCOPE_FIELDS, "menu_changed"]}


def _aggregate(observations: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[digest(_scope(pair))].append(pair)
    summaries: list[dict[str, Any]] = []
    rates: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    by_key = {(row["session_id"], row["clip_index"], row["round"]): row for row in observations}
    for group in grouped.values():
        scope = _scope(group[0])
        summary: dict[str, Any] = dict(scope)
        for metric in ("type_changed", "family_changed", "area_delta_pp"):
            valid = [row[metric] for row in group if row[metric] is not None]
            values: dict[str, Any] = {
                "total_pairs": len(group),
                "valid_pairs": len(valid),
                "na_pairs": len(group) - len(valid),
            }
            if metric == "area_delta_pp":
                values["mean"] = sum(valid) / len(valid) if valid else None
            else:
                values.update(changed_pairs=sum(valid), rate=sum(valid) / len(valid) if valid else None)
            summary[metric] = values
        summaries.append(summary)
        selected = [
            by_key[key]
            for pair in group
            for round_ in (1, 2)
            if (key := (pair["session_id"], pair["clip_index"], round_)) in by_key
            and all(by_key[key][field] == scope[field] for field in SCOPE_FIELDS)
        ]
        for round_ in (1, 2):
            relevant = [row for row in selected if row["round"] == round_]
            total = sum((pair["session_id"], pair["clip_index"], round_) in by_key for pair in group)
            relations = [
                row["av_selection_relation"] for row in relevant if row["av_selection_relation"] is not None
            ]
            summary[f"av_selection_relation_{round_}"] = {
                "total_observations": total,
                "valid_observations": len(relations),
                "na_observations": total - len(relations),
                "relation_sum": sum(relations),
                "rate": sum(relations) / len(relations) if relations else None,
            }
            cross = Counter(
                (row["selected_mask_id"], row["sound_type"])
                for row in relevant
                if row["selected_mask_id"] is not None and row["sound_type"] is not None
            )
            transitions.extend(
                {
                    **scope,
                    "kind": "visual_sound",
                    "round": round_,
                    "from_code": codes[0],
                    "to_code": codes[1],
                    "count": count,
                }
                for codes, count in sorted(cross.items())
            )
        for kind, field, offered in (
            ("type", "sound_type", "offered_sound_types"),
            ("family", "sound_family", "offered_sound_families"),
        ):
            universe = sorted({code for row in selected for code in row[offered]})
            for round_ in (1, 2):
                relevant = [row for row in selected if row["round"] == round_]
                for code in universe:
                    eligible = [row for row in relevant if row[field] is not None and code in row[offered]]
                    numerator = sum(row[field] == code for row in eligible)
                    rates.append(
                        {
                            **scope,
                            "kind": kind,
                            "round": round_,
                            "code": code,
                            "numerator": numerator,
                            "denominator": len(eligible),
                            "rate": numerator / len(eligible) if eligible else None,
                            "unmapped_selections": sum(
                                row["sound_raw"] is not None and row[field] is None for row in relevant
                            ),
                        }
                    )
            counts = Counter(
                (row[f"{field}_1"], row[f"{field}_2"]) for row in group if row[f"{kind}_changed"] is not None
            )
            transitions.extend(
                {**scope, "kind": kind, "from_code": codes[0], "to_code": codes[1], "count": count}
                for codes, count in sorted(counts.items())
            )
    return {"summaries": summaries, "selection_rates": rates, "transitions": transitions}


def analyze_sessions(sessions: list[dict[str, Any]], *, clip_id: str | None = None) -> dict[str, Any]:
    """Analyze frozen public sessions without modifying them or reading files."""
    observations: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    warnings: list[str] = []
    merged: dict[str, dict[str, Any]] = {}
    conflicting_sessions: set[str] = set()
    for session in sessions:
        sid = _text(session.get("session_id"))
        if sid is None:
            warnings.append("session_id_missing")
            continue
        if sid in merged:
            if session.get("sheet_config") != merged[sid].get("sheet_config"):
                warnings.append(f"{sid}:conflicting_session_config:session_excluded")
                conflicting_sessions.add(sid)
                continue
            if any(session.get(key) != merged[sid].get(key) for key in ("participant", "sheet_clip_order")):
                warnings.append(f"{sid}:conflicting_session_identity:session_excluded")
                conflicting_sessions.add(sid)
                continue
            merged[sid]["sheet_responses"].extend(session.get("sheet_responses", []))
        else:
            merged[sid] = {**session, "sheet_responses": list(session.get("sheet_responses", []))}
    for sid, session in sorted(merged.items()):
        if sid in conflicting_sessions:
            continue
        clips: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for record in session["sheet_responses"]:
            if (
                not isinstance(record, dict)
                or record.get("page") not in PAGES
                or record.get("analysis_excluded")
            ):
                continue
            index = record.get("clip_index")
            if type(index) is not int or index < 0:
                warnings.append(f"{sid}:clip_index_invalid")
                continue
            existing = clips[index][record["page"]]
            if record not in existing:
                existing.append(record)
        for index, records in sorted(clips.items()):
            order = session.get("sheet_clip_order", [])
            segment = order[index] if isinstance(order, list) and index < len(order) else None
            config = (
                _mapping(_mapping(session.get("sheet_config")).get("clips")).get(segment, {})
                if isinstance(segment, str)
                else {}
            )
            book = codebook_for_clip(config)
            current = []
            for round_, pages in ((1, ("P2", "P3")), (2, ("P7", "P8"))):
                current.append(
                    _observation(session, index, round_, records, book)
                    if any(page in records for page in pages)
                    else None
                )
            pair = _pair(current[0], current[1])
            if clip_id is not None and pair["clip_id"] != clip_id:
                continue
            pairs.append(pair)
            observations.extend(row for row in current if row is not None)
            warnings.extend(f"{sid}:{index}:{issue}" for issue in pair["issues"])
    return {
        "schema": SCHEMA,
        "observations": observations,
        "pairs": pairs,
        **_aggregate(observations, pairs),
        "warnings": sorted(set(warnings)),
    }
