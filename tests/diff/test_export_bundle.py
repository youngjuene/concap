"""The research export: complete records, flat answers with the answer keys, and a manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from test_live_server import drain
from test_participant_server import Lab, through_phase_one

from diff.__main__ import main
from diff.export_bundle import PHASE1_COLUMNS, R2_CODING_COLUMNS, SECOND_CAPTION_COLUMNS, caption_id, export
from diff.phase1_tweak import ELEMENTS_KO, RESTORATIVE
from diff.session_store import SessionStore


@pytest.fixture
def finished(tmp_path: Path) -> Lab:
    lab = Lab(tmp_path)
    client = lab.client()
    lab.enter(client)
    lab.act(client, through_phase_one(lab, client), "handoff")
    client.post("/watch/api/session", json={})
    drain(lab.service)
    client.post("/watch/api/start")
    client.post("/watch/api/video-ended")
    client.post("/watch/api/ratings", json={"answers": {"L1": 4}})
    lab.enter(lab.client(), "P2")  # a second, unfinished session
    return lab


def test_the_export_holds_every_session_and_its_answer_keys(finished: Lab, tmp_path: Path) -> None:
    out = tmp_path / "export"
    store = SessionStore(finished.root / "sessions.sqlite")
    manifest = export(
        store, [finished.root / "logs"], finished.setup, out, generated_at="2026-10-18T12:00:00"
    )
    assert manifest["counts"] == {
        "sessions": 2,
        "finished": 1,
        "phase1_rows": 102,
        "second_caption_rows": 3,
        "post_watch_rows": 1,
        "r2_coding_rows": 6,  # three shown captions, two R2 options each
    }

    records = [
        json.loads(line) for line in (out / "participants.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first = records[0]
    assert first["participant"] == "P1" and first["state"]["practice"][0]["analysis_excluded"] is True
    assert [event["event"] for event in first["live"]["events"]][:2] == ["session_started", "watch_started"]
    assert first["live"]["attempts"] and first["store_events"][0]["event"] == "session_created"

    with (out / "phase1.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == PHASE1_COLUMNS
    pages = {"P2", "P3", "P4", "P5", "P7", "P8", "P9", "P10", "P11"}
    assert {row["page"] for row in rows} == pages  # practice excluded
    assert {row["viewing"] for row in rows if row["page"] in ("P2", "P5")} == {"P1"}
    assert {row["viewing"] for row in rows if row["page"] in ("P7", "P11")} == {"P6"}
    a1s = next(row for row in rows if row["item"] == "A1s")
    clip = next(c for c in finished.setup["clips"] if c["key"] == a1s["clip"])
    assert a1s["answer"] == "s1" and a1s["sound_family"] == clip["options"]["A1s"][0]["family"]
    assert a1s["sound_orig_phrase"] == clip["options"]["A1s"][0]["orig_phrase"]
    assert a1s["sound_orig_phrase"].casefold() in clip["caption"]["orig_text"]
    a2s = next(row for row in rows if row["item"] == "A2s" and row["clip"] == a1s["clip"])
    assert a2s["answer"] == "s2" and a2s["sound_label"] == clip["options"]["A1s"][1]["label"]
    r2 = next(row for row in rows if row["item"] == "R2")
    assert json.loads(r2["r2_intended_key"]) == ["m1"] and json.loads(r2["answer"]) == ["m1"]
    for item in ("V1", "V2"):
        assert json.loads(next(row for row in rows if row["item"] == item)["point_object"]) == ["sky"]

    with (out / "second_captions.csv").open(encoding="utf-8") as handle:
        seconds = list(csv.DictReader(handle))
    assert tuple(seconds[0]) == SECOND_CAPTION_COLUMNS and len(seconds) == 3
    row = next(row for row in seconds if row["clip"] == a1s["clip"])
    second = f"{clip['options']['A1s'][0]['label']}가 또렷하게 들린다."
    assert (row["status"], row["second_caption"], row["first_caption"]) == (
        "ready",
        second,
        clip["caption"]["text"],
    )
    assert (row["rung"], row["wording"]) == ("0", "every-sound")
    assert row["emphasis"] in RESTORATIVE and row["emphasis_tied"] in ("True", "False")
    assert all(row[f"{name}_centred"] != "" for name in RESTORATIVE)
    assert row["attended_sound"] == clip["options"]["A1s"][0]["label"]
    assert (row["caption_source"], row["approved_from"]) == ("trida", "") and len(row["variant_id"]) == 12
    assert row["noticed_audioset_label"] == clip["options"]["A1s"][0]["audioset_label"]
    assert row["on_screen"] == str(clip["scene"]["sounds"]["s1"]["visibility"]["on_screen"])
    assert 0 < float(row["tendency"]) < 1 and row["congruence_centre"] != ""
    assert row["v2_attention"] in ("hit", "miss", "source not seen")  # V2 on the noticed source
    assert row["caption_id"] == caption_id(row["clip"], second)

    # One blind row per distinct shown caption and R2 option: no participant, mode or key.
    with (out / "r2_coding.csv").open(encoding="utf-8") as handle:
        coding = list(csv.DictReader(handle))
    assert tuple(coding[0]) == R2_CODING_COLUMNS and {r["named"] for r in coding} == {""}
    assert {(r["caption_id"], r["option"]) for r in coding if r["clip"] == row["clip"]} == {
        (row["caption_id"], "m1"),
        (row["caption_id"], "m2"),
    }
    assert row["assignment"] in ("follow", "counter") and 0 < float(row["p_bound"]) < 1
    assert row["mode"] == (row["recommended"] if row["assignment"] == "follow" else row["mode"])
    assert row["caption_type"].startswith(row["mode"]) and row["seen"] == (
        ELEMENTS_KO[clip["scene"]["distinctive"]] if row["mode"] == "scene" else ""
    )
    assert row["intended_r2_key"] == '["m1"]' and row["job_id"].startswith("job-")
    assert row["template_provisional"] == "True"

    with (out / "post_watch.csv").open(encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {
                "participant": "P1",
                "session_id": first["session_id"],
                "finished": "True",
                "item": "L1",
                "answer": "4",
            }
        ]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_the_export_never_overwrites(finished: Lab, tmp_path: Path) -> None:
    out = tmp_path / "export"
    out.mkdir()
    (out / "old.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        export(SessionStore(finished.root / "sessions.sqlite"), [], finished.setup, out, generated_at="now")


def test_the_cli_exports(finished: Lab, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup = tmp_path / "setup.json"
    setup.write_text(json.dumps(finished.setup, ensure_ascii=False), encoding="utf-8")
    argv = [
        "export",
        "--store",
        str(finished.root / "sessions.sqlite"),
        "--log-dir",
        str(finished.root / "logs"),
    ]
    assert main([*argv, "--phase1-setup", str(setup), "--out", str(tmp_path / "out")]) == 0
    assert "exported 2 sessions (1 finished)" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no session store"):
        main(
            [
                "export",
                "--store",
                str(tmp_path / "missing.sqlite"),
                "--log-dir",
                "x",
                "--phase1-setup",
                str(setup),
                "--out",
                str(tmp_path / "o2"),
            ]
        )
