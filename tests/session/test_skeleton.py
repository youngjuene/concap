"""The orderings math: what the columns draw, and which requests the server believes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpo.session.skeleton import (
    MAX_ORDERINGS,
    MIN_SPAN,
    Settings,
    SettingsError,
    all_orderings,
    balance_of,
    head_weights,
    heads_present,
    members_in_order,
    orderings,
    resolve,
    settings_key,
    validate_settings,
)

FIXTURE = Path(__file__).parent / "fixtures" / "session.json"
ROLES = {"underneath": "Underneath", "stands_out": "Stands out", "only_here": "Only here"}


def _source(source_id: str, left: float, right: float, role: str = "underneath") -> dict[str, Any]:
    return {"id": source_id, "weights": [left, right], "role": role}


def _tram_shot() -> dict[str, Any]:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    shot = document["clips"][0]["shots"][0]
    assert isinstance(shot, dict)
    return shot


def test_one_source_is_one_ordering_over_the_whole_span() -> None:
    assert orderings([_source("a", 0.2, 0.9)]) == [{"order": ["a"], "span": [0.0, 1.0]}]


def test_equal_weights_never_cross_and_keep_document_order() -> None:
    result = orderings([_source("b", 0.5, 0.5), _source("a", 0.5, 0.5), _source("c", 0.5, 0.5)])
    assert result == [{"order": ["b", "a", "c"], "span": [0.0, 1.0]}]


def test_a_crossing_pair_gives_two_orderings_that_partition_the_span() -> None:
    result = orderings([_source("eye", 0.9, 0.1), _source("ear", 0.1, 0.9)])
    assert [entry["order"] for entry in result] == [["eye", "ear"], ["ear", "eye"]]
    assert result[0]["span"] == [0.0, 0.5]
    assert result[1]["span"] == [0.5, 1.0]


def test_the_eye_end_ranks_by_left_and_the_ear_end_by_right() -> None:
    shot = _tram_shot()
    result = orderings(shot["sources"])
    by_left = [s["id"] for s in sorted(shot["sources"], key=lambda s: -s["weights"][0])]
    by_right = [s["id"] for s in sorted(shot["sources"], key=lambda s: -s["weights"][1])]
    assert result[0]["order"] == by_left == ["tram", "pigeons", "footsteps", "chatter", "siren"]
    assert result[-1]["order"] == by_right == ["siren", "tram", "chatter", "footsteps", "pigeons"]
    # Spans partition [0, 1] and consecutive orderings are distinct.
    assert result[0]["span"][0] == 0.0 and result[-1]["span"][1] == 1.0
    for earlier, later in zip(result, result[1:], strict=False):
        assert earlier["span"][1] == later["span"][0]
        assert earlier["order"] != later["order"]
    assert len(result) > 2


def test_grouped_heads_take_the_elementwise_maximum_of_their_members() -> None:
    sources = [
        _source("a", 0.2, 0.9, "stands_out"),
        _source("b", 0.7, 0.1, "stands_out"),
        _source("c", 0.4, 0.4, "only_here"),
    ]
    heads = head_weights(sources, ROLES)
    assert [(h.id, h.left, h.right) for h in heads] == [("stands_out", 0.7, 0.9), ("only_here", 0.4, 0.4)]
    assert heads_present(sources, ROLES) == ["stands_out", "only_here"]


def test_all_orderings_covers_every_non_empty_subset_keyed_by_sorted_ids() -> None:
    shot = _tram_shot()
    table = all_orderings(shot, ROLES)
    assert set(table) == {"itemized", "grouped"}
    assert len(table["itemized"]) == 2 ** len(shot["sources"]) - 1
    assert "chatter+siren+tram" in table["itemized"]
    assert table["itemized"]["tram"] == [{"order": ["tram"], "span": [0.0, 1.0]}]
    # A single-role subset has one head and therefore one grouped ordering.
    assert table["grouped"]["footsteps+chatter".replace("footsteps+chatter", "chatter+footsteps")] == [
        {"order": ["underneath"], "span": [0.0, 1.0]}
    ]
    assert json.dumps(table)  # what the inventory sends: JSON-serializable, no weights
    assert "weights" not in json.dumps(table)


def test_resolve_picks_the_ordering_whose_span_holds_the_balance() -> None:
    result = orderings([_source("eye", 0.9, 0.1), _source("ear", 0.1, 0.9)])
    assert resolve(result, 0.25) == 0
    assert resolve(result, 0.75) == 1
    assert resolve(result, 0.5) == 0
    assert resolve(result, 1.0) == 1
    with pytest.raises(SettingsError):
        resolve([], 0.5)


def test_settings_key_shapes() -> None:
    assert settings_key("itemized", ["b", "a"], ["b", "a"]) == "itemized|b,a"
    assert (
        settings_key("grouped", ["b", "a"], ["stands_out", "underneath"])
        == "grouped|stands_out,underneath|a,b"
    )
    assert settings_key("scene", ["a"], ["a"]) == "scene"
    assert settings_key("atmospheric", [], []) == "atmospheric"
    with pytest.raises(SettingsError):
        settings_key("detailed", [], [])


def test_validate_settings_accepts_a_reachable_order_and_refuses_a_forged_one() -> None:
    shot = _tram_shot()
    reachable = orderings(shot["sources"])[1]["order"]
    settings = validate_settings(
        shot, ROLES, {"level": "itemized", "admitted": reachable, "order": reachable}
    )
    assert settings == Settings("itemized", tuple(reachable), tuple(reachable))
    assert settings.key == "itemized|" + ",".join(reachable)
    forged = list(reversed(sorted(reachable)))
    if forged in [o["order"] for o in orderings(shot["sources"])]:
        forged = [reachable[1], reachable[0], *reachable[2:]]
    with pytest.raises(SettingsError, match="settings.order is not an ordering"):
        validate_settings(shot, ROLES, {"level": "itemized", "admitted": reachable, "order": forged})
    with pytest.raises(SettingsError, match="settings.admitted names unknown id"):
        validate_settings(shot, ROLES, {"level": "itemized", "admitted": ["ghost"], "order": ["ghost"]})
    with pytest.raises(SettingsError, match="settings.admitted must not be empty"):
        validate_settings(shot, ROLES, {"level": "itemized", "admitted": [], "order": []})
    with pytest.raises(SettingsError, match="settings.order must be a permutation"):
        validate_settings(
            shot, ROLES, {"level": "itemized", "admitted": ["tram", "siren"], "order": ["tram"]}
        )
    with pytest.raises(SettingsError, match="settings.level"):
        validate_settings(shot, ROLES, {"level": "detailed", "admitted": ["tram"], "order": ["tram"]})


def test_validate_settings_grouped_orders_the_present_heads() -> None:
    shot = _tram_shot()
    admitted = ["tram", "siren", "footsteps", "chatter", "pigeons"]
    heads = orderings(head_weights(shot["sources"], ROLES))
    first = heads[0]["order"]
    settings = validate_settings(shot, ROLES, {"level": "grouped", "admitted": admitted, "order": first})
    assert settings.level == "grouped" and settings.order == tuple(first)
    assert settings.key.startswith("grouped|") and settings.key.endswith(
        "|chatter+footsteps+pigeons+siren+tram".replace("+", ",")
    )
    with pytest.raises(SettingsError, match="settings.order"):
        validate_settings(shot, ROLES, {"level": "grouped", "admitted": ["tram"], "order": ["underneath"]})
    # Scene and atmospheric ignore admission and order (contract §3).
    scene = validate_settings(shot, ROLES, {"level": "scene", "admitted": ["tram"], "order": ["siren"]})
    assert scene.key == "scene"
    assert validate_settings(shot, ROLES, {"level": "atmospheric"}).key == "atmospheric"


def test_members_sit_under_their_head_in_the_heads_itemized_order() -> None:
    shot = _tram_shot()
    settings = validate_settings(
        shot,
        ROLES,
        {
            "level": "grouped",
            "admitted": ["footsteps", "chatter", "tram"],
            "order": ["stands_out", "underneath"],
        },
    )
    balance = balance_of(shot, ROLES, settings)
    assert 0.0 <= balance <= 1.0
    admitted = [s for s in shot["sources"] if s["id"] in settings.admitted]
    eye_end = members_in_order(admitted, "underneath", 0.0)
    ear_end = members_in_order(admitted, "underneath", 1.0)
    assert [m["id"] for m in eye_end] == ["footsteps", "chatter"]
    assert [m["id"] for m in ear_end] == ["chatter", "footsteps"]


# ---- collapsing regimes nobody could choose (MIN_SPAN, MAX_ORDERINGS) --------


def _near_parallel(count: int) -> list[dict[str, Any]]:
    """Sources whose lines cross at many nearly-coincident points in (0, 1)."""
    return [_source(f"s{i}", 0.5 + i * 1e-4, 0.5 - i * 1e-4 + i * i * 1e-5) for i in range(count)]


def test_raw_partition_is_available_and_covers_the_axis() -> None:
    raw = orderings(_near_parallel(6), min_span=0.0, max_count=99)
    assert len(raw) > 1
    assert raw[0]["span"][0] == 0.0
    assert raw[-1]["span"][1] == 1.0
    assert min(o["span"][1] - o["span"][0] for o in raw) < MIN_SPAN


def test_collapsing_leaves_no_regime_narrower_than_min_span() -> None:
    kept = orderings(_near_parallel(6))
    assert all(o["span"][1] - o["span"][0] >= MIN_SPAN - 1e-12 for o in kept)


def test_collapsing_keeps_the_spans_a_partition_of_the_axis() -> None:
    kept = orderings(_near_parallel(6))
    assert kept[0]["span"][0] == 0.0
    assert kept[-1]["span"][1] == 1.0
    for earlier, later in zip(kept, kept[1:], strict=False):
        assert earlier["span"][1] == pytest.approx(later["span"][0])


def test_collapsing_keeps_the_eye_first_and_ear_first_orderings() -> None:
    sources = _near_parallel(6)
    raw = orderings(sources, min_span=0.0, max_count=99)
    kept = orderings(sources)
    assert kept[0]["order"] == raw[0]["order"]
    assert kept[-1]["order"] == raw[-1]["order"]


def test_no_shot_offers_more_columns_than_the_stage_was_sized_for() -> None:
    assert len(orderings(_near_parallel(8))) <= MAX_ORDERINGS


def test_a_kept_ordering_is_one_the_raw_partition_also_names() -> None:
    sources = _near_parallel(6)
    raw = [tuple(o["order"]) for o in orderings(sources, min_span=0.0, max_count=99)]
    for ordering in orderings(sources):
        assert tuple(ordering["order"]) in raw


def test_resolve_still_answers_for_every_balance_after_collapsing() -> None:
    kept = orderings(_near_parallel(6))
    for step in range(101):
        index = resolve(kept, step / 100.0)
        lo, hi = kept[index]["span"]
        assert lo - 1e-9 <= step / 100.0 <= hi + 1e-9


def test_wide_regimes_are_left_alone() -> None:
    # Two sources crossing once at the middle: one crossing, two wide regimes.
    sources = [_source("eye", 1.0, 0.0), _source("ear", 0.0, 1.0)]
    assert orderings(sources) == orderings(sources, min_span=0.0, max_count=99)


def test_collapsing_a_regime_never_moves_a_source_that_never_crosses() -> None:
    # A source that leads on both axes leads in every ordering, collapsed or not.
    sources = [_source("loud", 1.0, 1.0), *_near_parallel(5)]
    for ordering in orderings(sources):
        assert ordering["order"][0] == "loud"
