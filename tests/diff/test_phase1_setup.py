"""The phase-1 setup: drafting from metadata, the researcher's approval, and every refusal."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from phase1_fixtures import approved_setup, drafted, filled

import dpo.regen.sheet_instrument as sheet_instrument
from diff.__main__ import main
from diff.phase1_setup import (
    DRAFTED,
    P0_DELETED,
    SetupError,
    approve,
    approved,
    read_ontology,
    setup_problems,
    sound_options,
    specific_labels,
)


def problems(setup: dict, purpose: str, root: Path) -> list[str]:
    return setup_problems(setup, purpose, root=root, instrument=sheet_instrument)


def test_ancestors_go_as_in_regen_staging(tmp_path: Path) -> None:
    drafted(tmp_path)
    ids, below, _ = read_ontology(tmp_path / "ontology.json")
    assert specific_labels(["Vehicle", "Bus", "Speech"], ids, below) == (["Bus", "Speech"], ["Vehicle"])
    with pytest.raises(SetupError, match="absent from the ontology"):
        specific_labels(["Tuba"], ids, below)


def test_a_single_remaining_sound_keeps_its_ancestor(tmp_path: Path) -> None:
    drafted(tmp_path)
    ids, below, roots = read_ontology(tmp_path / "ontology.json")
    options, dropped = sound_options(["Vehicle", "Vehicle horn, car horn, honking"], ids, below, roots)
    assert [option["audioset"] for option in options] == ["Vehicle", "Vehicle horn, car horn, honking"]
    assert options[0]["kept_ancestor"] is True and dropped == []


def test_the_draft_holds_what_the_metadata_determines(tmp_path: Path) -> None:
    setup = drafted(tmp_path)
    assert [clip["key"] for clip in setup["clips"]] == ["clip_a", "clip_b", "clip_c"]
    clip_a = setup["clips"][0]
    assert [frame["at_ms"] for frame in clip_a["frames"]] == [1000, 3000, 5000]
    assert [item["id"] for item in clip_a["frames"][0]["objects"]] == ["road", "sky"]
    assert [option["audioset"] for option in clip_a["options"]["A1s"]] == ["Bus", "Male speech, man speaking"]
    assert {option["family"] for option in clip_a["options"]["A1s"]} == {"things", "human"}
    assert clip_a["dropped_labels"] == ["Vehicle", "Speech"]
    assert clip_a["caption"] == {
        "text": "",
        "start_ms": 0,
        "end_ms": 10000,
        "orig_text": "a bus idles while a man speaks.",
        "provenance": DRAFTED,
    }
    wind = setup["clips"][2]["options"]["A1s"][1]
    assert wind["family"] == "natural" and wind["family_ambiguous"] == ["Natural sounds", "Sounds of things"]
    assert setup["practice"]["frames"] == [{"still": "artifacts/practice.png", "at_ms": 5000}]
    assert setup["instrument_hash"] == sheet_instrument.snapshot()["hash"]


def test_the_wording_draft_makes_only_the_researcher_s_edits(tmp_path: Path) -> None:
    wording = drafted(tmp_path)["wording"]
    intro = wording["P0_instruction"]
    assert intro["workbook"] == sheet_instrument.page_spec("P0")["instruction"]
    assert all(sentence not in intro["shown"] for sentence in P0_DELETED)
    assert intro["shown"] == intro["workbook"].replace(" " + P0_DELETED[0], "").replace(
        " " + P0_DELETED[1], ""
    )
    assert wording["titles"]["P10"] == {"workbook": "2차 자막 평가", "shown": "자막 평가"}
    assert wording["titles"]["P2"]["shown"] == "시각 지각" and "P1" not in wording["titles"]


def test_empty_korean_fields_block_even_smoke_sessions(tmp_path: Path) -> None:
    found = problems(drafted(tmp_path), "smoke", tmp_path)
    assert "clip_a.caption: no Korean text" in found
    assert any(problem.startswith("clip_a.R2") for problem in found)
    assert problems(filled(drafted(tmp_path)), "smoke", tmp_path) == []


def test_participant_sessions_need_every_part_approved(tmp_path: Path) -> None:
    setup = filled(drafted(tmp_path))
    found = problems(setup, "participant", tmp_path)
    assert len(found) == 5 and all("not approved" in problem for problem in found)
    assert problems(approved_setup(tmp_path / "again"), "participant", tmp_path / "again") == []


def test_an_edit_after_approval_voids_it(tmp_path: Path) -> None:
    setup = approved_setup(tmp_path)
    assert approved(setup["clips"][0])
    setup["clips"][0]["caption"]["text"] = "버스가 공회전하고 남자가 말한다."
    assert not approved(setup["clips"][0])
    assert problems(setup, "participant", tmp_path) == [
        "clip_a: not approved by the researcher (or edited since approval)"
    ]


def test_a_changed_workbook_refuses_participant_sessions(tmp_path: Path) -> None:
    setup = approved_setup(tmp_path)
    setup["instrument_hash"] = "0" * 64
    assert "the workbook instrument changed since this setup was drafted" in problems(
        setup, "participant", tmp_path
    )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda s: s["clips"][0]["caption"].update(text="버스 " * 40), "characters, over 96"),
        (lambda s: s["clips"][0]["caption"].update(text="버스\n남자\n바람"), "more than 2 lines"),
        (lambda s: s["clips"][0]["caption"].update(text="bus 소리"), "Latin letters"),
        (lambda s: s["clips"][0]["caption"].update(end_ms=9000), "must cover the clip"),
        (lambda s: s["clips"][0]["options"]["R2"][1].update(mentioned=True), "one distractor"),
        (lambda s: s["clips"][0]["options"]["R2"][1].pop("mentioned"), "the answer key"),
        (lambda s: s["clips"][0]["options"]["A1s"][1].update(value="s1"), "duplicate"),
        (lambda s: s["clips"][0]["options"]["A1s"][1].update(family="vehicles"), "family must be one of"),
        (lambda s: s["clips"][0]["frames"].pop(), "P2 needs 3–9"),
        (
            lambda s: s["practice"]["frames"][0].update(still=s["clips"][0]["frames"][0]["still"]),
            "must not be",
        ),
        (lambda s: s["wording"]["titles"].update(P7={"shown": "시각"}), "does not show"),
        (lambda s: s.update(consent_text="  "), "consent_text"),
    ],
)
def test_refusals(tmp_path: Path, change, expected: str) -> None:
    setup = filled(drafted(tmp_path))
    change(setup)
    assert any(expected in problem for problem in problems(setup, "smoke", tmp_path)), problems(
        setup, "smoke", tmp_path
    )


def test_a_missing_media_file_is_named(tmp_path: Path) -> None:
    setup = filled(drafted(tmp_path))
    (tmp_path / setup["clips"][1]["frames"][0]["objects"][0]["mask"]).unlink()
    assert any(
        "clip_b.frame 0 road: missing file" in problem for problem in problems(setup, "smoke", tmp_path)
    )


def test_approval_needs_a_name_and_known_clips(tmp_path: Path) -> None:
    setup = filled(drafted(tmp_path))
    with pytest.raises(SetupError, match="reviewer's name"):
        approve(setup, by=" ", at="now", clips=["clip_a"])
    with pytest.raises(SetupError, match="no such clips"):
        approve(setup, by="researcher", at="now", clips=["clip_z"])
    original = copy.deepcopy(setup)
    approve(setup, by="researcher", at="now", clips=["clip_a"])
    assert setup == original


def test_the_cli_shows_and_approves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(filled(drafted(tmp_path)), ensure_ascii=False), encoding="utf-8")
    assert main(["phase1", "show", "--setup", str(path)]) == 0
    shown = capsys.readouterr().out
    assert "[clip_a]" in shown and "5 problems for participant sessions" in shown
    approval = ["--clip", "clip_a", "--clip", "clip_b", "--clip", "clip_c", "--practice", "--wording"]
    assert main(["phase1", "approve", "--setup", str(path), "--by", "researcher", *approval]) == 0
    assert "0 problems left" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="name what you approve"):
        main(["phase1", "approve", "--setup", str(path), "--by", "researcher"])
