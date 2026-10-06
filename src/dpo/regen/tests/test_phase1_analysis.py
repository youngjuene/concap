"""Observational trends are neither correctness nor attention scores."""

from copy import deepcopy
from typing import Any

import pytest

from dpo.regen.phase1_analysis import analyze_sessions, codebook_for_clip


def session() -> dict[str, Any]:
    rows = []
    for round_, sound, ratio in ((1, "engine", 0.1), (2, "car", 0.25)):
        metadata = {
            "clip_index": 0,
            "clip_id": "clip",
            "instrument_hash": "instrument",
            "view_id": f"view-{round_}",
            "caption_strategy": "maintain",
            "stimulus_assignment": {"audiovisual_congruence": "congruent", "description_depth": "shallow"},
        }
        rows.append(
            {
                **metadata,
                "page": "P2" if round_ == 1 else "P7",
                "answers": {
                    "V1" if round_ == 1 else "V2": [{"frame_id": str(round_ - 1), "x": 0.2, "y": 0.3}]
                },
                "visual_metrics": [
                    {
                        "selected_mask_id": "road",
                        "label": "Road",
                        "frame_id": str(round_ - 1),
                        "selected_mask_area_px": ratio * 1000,
                        "frame_area_px": 1000,
                        "selected_mask_area_ratio": ratio,
                        "frame_sha256": f"frame-{round_}",
                        "mask_sha256": "mask",
                        "matching_rule": "smallest-containing-mask-v1",
                        "threshold": 127,
                        "candidate_masks": [
                            {
                                "object_id": "road",
                                "label": "Road",
                                "mask_area_px": ratio * 1000,
                                "mask_area_ratio": ratio,
                                "mask_sha256": "mask",
                                "status": "matched",
                            }
                        ],
                        "status": "matched",
                    }
                ],
            }
        )
        item_id = "A1s" if round_ == 1 else "A2s"
        rows.append(
            {
                **metadata,
                "page": "P3" if round_ == 1 else "P8",
                "answers": {item_id: sound},
                "items": [
                    {
                        "id": item_id,
                        "options": [{"value": "engine", "label": "엔진"}, {"value": "car", "label": "차량"}],
                    }
                ],
            }
        )
    return {
        "session_id": "session",
        "participant": "participant",
        "sheet_clip_order": ["A"],
        "sheet_config": {"clips": {"A": {"source_families": {"engine": "things", "car": "things"}}}},
        "sheet_responses": rows,
    }


def analyze(value: dict[str, Any] | None = None) -> dict[str, Any]:
    return analyze_sessions([session() if value is None else value])


def test_area_difference_and_frame_change() -> None:
    pair = analyze()["pairs"][0]
    assert pair["area_delta_pp"] == pytest.approx(15)
    assert pair["frame_changed"] is True


def test_type_changes_family_does_not() -> None:
    pair = analyze()["pairs"][0]
    assert pair["type_changed"] == 1 and pair["family_changed"] == 0


def test_missing_family_is_na_including_two_missing() -> None:
    value = session()
    value["sheet_config"]["clips"]["A"]["source_families"] = {}
    result = analyze(value)
    assert result["pairs"][0]["family_changed"] is None
    assert result["summaries"][0]["family_changed"]["valid_pairs"] == 0


def test_relation_absent_is_not_inferred_from_assignment() -> None:
    result = analyze()
    row = result["observations"][0]
    assert row["av_assigned_condition"] == "congruent"
    assert row["av_selection_relation"] is None
    assert result["summaries"][0]["av_selection_relation_1"]["rate"] is None
    cross = [row for row in result["transitions"] if row["kind"] == "visual_sound"]
    assert [(row["round"], row["from_code"], row["to_code"]) for row in cross] == [
        (1, "road", "engine"),
        (2, "road", "car"),
    ]


def test_relation_uses_explicit_class_and_sound_table() -> None:
    value = session()
    value["sheet_config"]["clips"]["A"]["analysis_codebook"] = {
        "version": "v1",
        "av_relations": {"road": {"engine": 1, "car": 0}},
    }
    result = analyze(value)
    assert [row["av_selection_relation"] for row in result["observations"]] == [1, 0]
    assert result["summaries"][0]["av_selection_relation_1"] == {
        "total_observations": 1,
        "valid_observations": 1,
        "na_observations": 0,
        "relation_sum": 1,
        "rate": 1.0,
    }
    assert result["summaries"][0]["av_selection_relation_2"]["rate"] == 0.0


def test_different_menus_preserved_and_compared() -> None:
    value = session()
    value["sheet_responses"][3]["items"][0]["options"] = [{"value": "car", "label": "차량"}]
    pair = analyze(value)["pairs"][0]
    assert pair["menu_changed"] is True and pair["type_changed"] == 1
    assert pair["order_changed"] is None


def test_menu_order_only_is_separate() -> None:
    value = session()
    value["sheet_responses"][3]["items"][0]["options"].reverse()
    pair = analyze(value)["pairs"][0]
    assert pair["menu_changed"] is False and pair["order_changed"] is True


def test_menu_label_change_counts_as_menu_change() -> None:
    value = session()
    value["sheet_responses"][3]["items"][0]["options"][0]["label"] = "다른 문구"
    pair = analyze(value)["pairs"][0]
    assert pair["menu_changed"] is True and pair["order_changed"] is False


def test_not_offered_denominator_zero_is_na() -> None:
    value = session()
    value["sheet_responses"][3]["items"][0]["options"] = [{"value": "car", "label": "차량"}]
    result = analyze(value)
    row = next(
        r
        for r in result["selection_rates"]
        if r["kind"] == "type" and r["round"] == 2 and r["code"] == "engine"
    )
    assert row["denominator"] == 0 and row["rate"] is None


def test_family_offered_once_even_if_two_types_in_family() -> None:
    rows = [row for row in analyze()["selection_rates"] if row["kind"] == "family"]
    assert all(row["denominator"] == 1 and row["numerator"] == 1 for row in rows)


def test_partial_pair_retained_with_metric_denominators() -> None:
    value = session()
    value["sheet_responses"] = value["sheet_responses"][:2]
    result = analyze(value)
    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["type_changed"] is None
    assert result["summaries"][0]["type_changed"] == {
        "total_pairs": 1,
        "valid_pairs": 0,
        "na_pairs": 1,
        "changed_pairs": 0,
        "rate": None,
    }


@pytest.mark.parametrize("field,value", [("clip_id", "other"), ("instrument_hash", "other")])
def test_conflicting_identity_is_na(field: str, value: str) -> None:
    data = session()
    data["sheet_responses"][1][field] = value
    result = analyze(data)
    assert result["pairs"][0]["type_changed"] is None
    assert result["warnings"]


def test_conflicting_duplicate_page_is_na() -> None:
    value = session()
    extra = deepcopy(value["sheet_responses"][1])
    extra["answers"]["A1s"] = "car"
    value["sheet_responses"].append(extra)
    assert analyze(value)["pairs"][0]["type_changed"] is None


def test_exact_mirror_duplicate_is_deduplicated() -> None:
    value = session()
    value["sheet_responses"] += deepcopy(value["sheet_responses"])
    assert analyze(value)["pairs"][0]["type_changed"] == 1


def test_two_sessions_never_cross_pair() -> None:
    first, second = session(), session()
    first["sheet_responses"] = first["sheet_responses"][:2]
    second["session_id"] = "other"
    second["sheet_responses"] = second["sheet_responses"][2:]
    result = analyze_sessions([first, second])
    assert len(result["pairs"]) == 2
    assert all(row["type_changed"] is None for row in result["pairs"])


def test_practice_excluded() -> None:
    value = session()
    value["sheet_responses"][1]["analysis_excluded"] = True
    assert analyze(value)["pairs"][0]["type_changed"] is None


@pytest.mark.parametrize("status", ["unclassified", "resolution_mismatch"])
def test_nonmeasured_area_is_na(status: str) -> None:
    value = session()
    value["sheet_responses"][0]["visual_metrics"][0]["status"] = status
    assert analyze(value)["pairs"][0]["area_delta_pp"] is None


def test_legacy_no_metrics_no_backfill() -> None:
    value = session()
    value["sheet_responses"][0].pop("visual_metrics")
    value["sheet_responses"][0]["matches"] = [{"frame": 0, "object_id": "road", "label": "Road"}]
    result = analyze(value)
    row = result["observations"][0]
    assert (row["selected_mask_id"], row["visual_label"], row["frame_id"]) == ("road", "Road", "0")
    assert row["frame_sha256"] is None and row["mask_sha256"] is None
    assert result["pairs"][0]["area_delta_pp"] is None
    assert result["pairs"][0]["frame_changed"] is True


def test_inconsistent_area_arithmetic_rejected() -> None:
    value = session()
    value["sheet_responses"][0]["visual_metrics"][0]["selected_mask_area_ratio"] = 0.9
    assert analyze(value)["pairs"][0]["area_delta_pp"] is None


def test_explicit_alias_mapping() -> None:
    value = session()
    value["sheet_config"]["clips"]["A"]["analysis_codebook"] = {
        "version": "v1",
        "sound_types": {"engine": "motor", "car": "motor"},
        "sound_families": {"motor": "transport"},
    }
    assert analyze(value)["pairs"][0]["type_changed"] == 0


@pytest.mark.parametrize("bad", [[], None, "bad"])
def test_malformed_codebook_maps_do_not_crash(bad: Any) -> None:
    result = codebook_for_clip({"analysis_codebook": {"version": "v1", "sound_families": bad}})
    assert result["sound_families"] == {} and result["issues"]


def test_legacy_alias_family_conflict_is_na() -> None:
    result = codebook_for_clip(
        {
            "analysis_codebook": {"version": "v1", "sound_types": {"a": "x", "b": "x"}},
            "source_families": {"a": "one", "b": "two"},
        }
    )
    assert "x" not in result["sound_families"] and result["issues"]


def test_filter_applied_before_denominators() -> None:
    assert analyze_sessions([session()], clip_id="absent")["pairs"] == []
    assert analyze_sessions([session()], clip_id="absent")["selection_rates"] == []


def test_output_allowlist_omits_credentials() -> None:
    import json

    value = session()
    value["token"] = "TOP_SECRET_SENTINEL"
    value["sheet_responses"][0]["visual_metrics"][0]["secret"] = "TOP_SECRET_SENTINEL"
    assert "TOP_SECRET_SENTINEL" not in json.dumps(analyze(value))


def test_input_is_not_modified() -> None:
    value = session()
    original = deepcopy(value)
    analyze(value)
    assert value == original


def test_condition_groups_remain_separate() -> None:
    first, second = session(), session()
    second["session_id"] = "second"
    for row in second["sheet_responses"]:
        row["stimulus_assignment"]["description_depth"] = "deep"
    assert len(analyze_sessions([first, second])["summaries"]) == 2


def test_conflicting_session_configs_do_not_choose_arbitrary_first() -> None:
    first, second = session(), session()
    second["sheet_config"]["clips"]["A"]["source_families"]["engine"] = "other"
    result = analyze_sessions([first, second])
    assert result["observations"] == result["pairs"] == result["selection_rates"] == []
    assert "session:conflicting_session_config:session_excluded" in result["warnings"]


def test_unmapped_family_selection_not_fake_zero() -> None:
    value = session()
    value["sheet_config"]["clips"]["A"]["source_families"].pop("engine")
    result = analyze(value)
    row = next(r for r in result["selection_rates"] if r["kind"] == "family" and r["round"] == 1)
    assert row["denominator"] == 0 and row["rate"] is None
    assert result["observations"][0]["unmapped_offered_sound_types"] == ["engine"]
