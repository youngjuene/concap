"""The phase-1 setup: drafting from metadata, the researcher's approval, and every refusal."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import phase1_fixtures
import pytest
from phase1_fixtures import PREPARED, approved_setup, drafted, filled

import dpo.regen.sheet_instrument as sheet_instrument
from diff.__main__ import main
from diff.phase1_setup import (
    P0_DELETED,
    SetupError,
    approve,
    approved,
    label_families,
    read_ontology,
    setup_problems,
)


def problems(setup: dict, purpose: str, root: Path) -> list[str]:
    return setup_problems(setup, purpose, root=root, instrument=sheet_instrument)


def test_every_audioset_label_is_kept_with_its_family(tmp_path: Path) -> None:
    drafted(tmp_path)
    ids, roots = read_ontology(tmp_path / "ontology.json")
    assert label_families(["Vehicle", "Bus", "Vehicle", "Speech"], ids, roots) == [
        {"label": "Vehicle", "family": "things"},
        {"label": "Bus", "family": "things"},
        {"label": "Speech", "family": "human"},
    ]
    with pytest.raises(SetupError, match="absent from the ontology"):
        label_families(["Tuba"], ids, roots)


def test_the_draft_holds_what_the_metadata_determines(tmp_path: Path) -> None:
    setup = drafted(tmp_path)
    assert [clip["key"] for clip in setup["clips"]] == ["kyoto_101", "lima_202", "oslo_303"]
    first_clip = setup["clips"][0]
    assert [frame["at_ms"] for frame in first_clip["frames"]] == [1000, 3000, 5000]
    assert [item["id"] for item in first_clip["frames"][0]["objects"]] == ["road", "sky"]
    assert [entry["label"] for entry in first_clip["audioset_labels"]] == [
        "Vehicle",
        "Bus",
        "Speech",
        "Male speech, man speaking",
    ]
    assert first_clip["options"] == {"A1s": [], "R2": []}  # written from orig_text, never from labels
    assert first_clip["caption"] == {
        "text": "버스가 공회전하는 소리",
        "start_ms": 0,
        "end_ms": 10000,
        "source": "scripts/captions.json",
        "provenance": "transcribed-from-video",
        "orig_text": "a bus idles while a man speaks.",
    }
    wind = setup["clips"][2]["audioset_labels"][1]
    assert wind["family"] == "natural" and wind["family_ambiguous"] == ["Natural sounds", "Sounds of things"]
    assert setup["practice"]["orig_text"] == "people talk near a bus." and setup["practice"]["options"] == {
        "A1s": []
    }
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


def test_the_prepared_captions_are_copied_raw(tmp_path: Path) -> None:
    setup = drafted(tmp_path)
    assert {clip["key"]: clip["caption"]["text"] for clip in setup["clips"]} == PREPARED
    assert setup["clips"][1]["caption"]["text"].count("\n") == 2  # three lines, kept and allowed
    assert problems(filled(setup), "smoke", tmp_path) == []


def test_a_clip_without_a_prepared_caption_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase1_fixtures, "PREPARED", {"kyoto_101": "버스 소리", "lima_202": "경적 소리"})
    with pytest.raises(SetupError, match=r"no prepared caption .* \['oslo_303'\]"):
        drafted(tmp_path)


def test_what_the_prepared_material_needs_no_longer_refuses(tmp_path: Path) -> None:
    setup = filled(drafted(tmp_path))
    setup["clips"][0]["caption"]["text"] = "버스 " * 60  # long and raw
    setup["clips"][0]["options"]["A1s"] = setup["clips"][0]["options"]["A1s"][:1]  # one sound
    setup["clips"][0]["options"]["R2"][1].update(mentioned=True, orig_phrase="a man speaks")  # no distractor
    setup["practice"]["options"]["A1s"][1]["label"] = "버스 소리"  # a label the clips also use
    assert problems(setup, "smoke", tmp_path) == []


def test_unwritten_options_block_even_smoke_sessions(tmp_path: Path) -> None:
    found = problems(drafted(tmp_path), "smoke", tmp_path)
    assert any(problem.startswith("kyoto_101.R2") for problem in found)
    assert not any("caption" in problem for problem in found)


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
        "kyoto_101: not approved by the researcher (or edited since approval)"
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
        (lambda s: s["clips"][0]["caption"].update(text=" "), "caption: empty"),
        (lambda s: s["clips"][0]["caption"].update(end_ms=9000), "must cover the clip"),
        (lambda s: s["clips"][0]["options"]["R2"][1].pop("mentioned"), "the answer key"),
        (lambda s: s["clips"][0]["options"]["A1s"][1].update(value="s1"), "duplicate"),
        (lambda s: s["clips"][0]["options"]["A1s"][1].update(family="vehicles"), "family must be one of"),
        (lambda s: s["clips"][0]["options"]["A1s"][1].pop("orig_phrase"), "no orig_phrase"),
        (
            lambda s: s["clips"][0]["options"]["A1s"][1].update(orig_phrase="a trumpet plays"),
            "'a trumpet plays' is not in orig_text",
        ),
        (lambda s: s["clips"][0]["options"]["R2"][0].pop("orig_phrase"), "kyoto_101.R2 m1: no orig_phrase"),
        (lambda s: s["practice"]["options"]["A1s"][0].update(orig_phrase="a man"), "not in orig_text"),
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
        "lima_202.frame 0 road: missing file" in problem for problem in problems(setup, "smoke", tmp_path)
    )


def test_approval_needs_a_name_and_known_clips(tmp_path: Path) -> None:
    setup = filled(drafted(tmp_path))
    with pytest.raises(SetupError, match="reviewer's name"):
        approve(setup, by=" ", at="now", clips=["kyoto_101"])
    with pytest.raises(SetupError, match="no such clips"):
        approve(setup, by="researcher", at="now", clips=["clip_z"])
    original = copy.deepcopy(setup)
    approve(setup, by="researcher", at="now", clips=["kyoto_101"])
    assert setup == original


def test_the_cli_shows_and_approves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(filled(drafted(tmp_path)), ensure_ascii=False), encoding="utf-8")
    assert main(["phase1", "show", "--setup", str(path)]) == 0
    shown = capsys.readouterr().out
    assert "[kyoto_101]" in shown and "5 problems for participant sessions" in shown
    approval = ["--clip", "kyoto_101", "--clip", "lima_202", "--clip", "oslo_303", "--practice", "--wording"]
    assert main(["phase1", "approve", "--setup", str(path), "--by", "researcher", *approval]) == 0
    assert "0 problems left" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="name what you approve"):
        main(["phase1", "approve", "--setup", str(path), "--by", "researcher"])
