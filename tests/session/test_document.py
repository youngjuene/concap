"""The session document: every rule names its path, and the browser sees only its share."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from dpo.session.document import (
    SessionDocumentError,
    load_session_document,
    participant_document,
    validate_session_document,
)

FIXTURE = Path(__file__).parent / "fixtures" / "session.json"


def fixture_document() -> dict[str, Any]:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_the_fixture_validates() -> None:
    document = load_session_document(FIXTURE)
    assert document["session_id"] == "demo-session"
    assert [clip["task"] for clip in document["clips"]].count("shaped") == 3


def _set(path: list[str | int], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(document: dict[str, Any]) -> None:
        target: Any = document
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value

    return mutate


def _delete(path: list[str | int]) -> Callable[[dict[str, Any]], None]:
    def mutate(document: dict[str, Any]) -> None:
        target: Any = document
        for step in path[:-1]:
            target = target[step]
        del target[path[-1]]

    return mutate


def _remove_item(item_id: str) -> Callable[[dict[str, Any]], None]:
    def mutate(document: dict[str, Any]) -> None:
        document["measures"]["items"] = [i for i in document["measures"]["items"] if i["id"] != item_id]

    return mutate


def _nine_sources(document: dict[str, Any]) -> None:
    shot = document["clips"][0]["shots"][0]
    base = shot["sources"][0]
    shot["sources"] = [{**base, "id": f"s{index}"} for index in range(9)]


def _all_shaped(document: dict[str, Any]) -> None:
    shaped = [clip for clip in document["clips"] if clip["task"] == "shaped"]
    document["clips"] = shaped
    kept = {clip["clip_id"] for clip in shaped}
    followup = document["followup"]
    followup["recognition"] = [entry for entry in followup["recognition"] if entry["clip_id"] in kept]
    followup["sound_only"] = [entry for entry in followup["sound_only"] if entry["clip_id"] in kept]


CASES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("schema", _set(["schema"], "dpo.study-export/v1"), "schema"),
    ("session id", _set(["session_id"], ""), "session_id"),
    ("poles verbatim", _set(["poles", "audio"], ["Left", "Right"]), "poles"),
    ("role heads verbatim", _set(["role_heads", "audio", "only_here"], "Only Here"), "role_heads.audio"),
    ("attention item present", _remove_item("attention"), "measures.items"),
    ("attention boxes", _set(["measures", "items", 4, "boxes"], 5), "measures.items[attention].boxes"),
    (
        "attention poles",
        _set(["measures", "items", 4, "poles"], ["Image", "Sound"]),
        "measures.items[attention].poles",
    ),
    (
        "attention text",
        _set(["measures", "items", 4, "text"], "Where were you looking?"),
        "measures.items[attention].text",
    ),
    (
        "criterion text",
        _set(["measures", "items", 5, "text"], "Do you like it?"),
        "measures.items[criterion].text",
    ),
    ("criterion boxes", _set(["measures", "items", 5, "boxes"], 3), "measures.items[criterion].boxes"),
    ("item boxes minimum", _set(["measures", "items", 0, "boxes"], 1), "measures.items[0].boxes"),
    ("duplicate item id", _set(["measures", "items", 1, "id"], "match"), "measures.items[1].id"),
    ("open item", _set(["measures", "open_item"], "Anything else?"), "measures.open_item"),
    ("clips non-empty", _set(["clips"], []), "clips"),
    ("both tasks occur", _all_shaped, "clips"),
    ("clip id charset", _set(["clips", 0, "clip_id"], "tram stop"), "clips[0].clip_id"),
    ("duplicate clip id", _set(["clips", 1, "clip_id"], "demo_tram_stop"), "clips[1].clip_id"),
    ("task", _set(["clips", 0, "task"], "practice"), "clips[0].task"),
    (
        "first viewing captions",
        _set(["clips", 0, "first_viewing_captions"], "yes"),
        "clips[0].first_viewing_captions",
    ),
    ("balance control", _set(["clips", 0, "balance_control"], "slider"), "clips[0].balance_control"),
    ("opening level", _set(["clips", 0, "opening", "level"], "detailed"), "clips[0].opening.level"),
    ("opening balance", _set(["clips", 0, "opening", "balance"], 1.5), "clips[0].opening.balance"),
    (
        "shaped clip has no visual roles",
        _set(["clips", 0, "visual_roles"], {"a": "A"}),
        "clips[0].visual_roles",
    ),
    (
        "control clip needs three roles",
        _set(["clips", 1, "visual_roles"], {"a": "A", "b": "B"}),
        "clips[1].visual_roles",
    ),
    ("control clip needs visual roles", _delete(["clips", 1, "visual_roles"]), "clips[1].visual_roles"),
    ("shots contiguous", _set(["clips", 0, "shots", 1, "start_ms"], 4300), "clips[0].shots[1].start_ms"),
    ("shots start at zero", _set(["clips", 0, "shots", 0, "start_ms"], 100), "clips[0].shots[0].start_ms"),
    ("shot end after start", _set(["clips", 0, "shots", 0, "end_ms"], 0), "clips[0].shots[0].end_ms"),
    ("duplicate shot id", _set(["clips", 0, "shots", 1, "shot_id"], "s1"), "clips[0].shots[1].shot_id"),
    (
        "automatic caption",
        _set(["clips", 0, "shots", 0, "automatic_caption"], 3),
        "clips[0].shots[0].automatic_caption",
    ),
    # Authored captions are placed or returned verbatim, so a line over the
    # two-line budget would clip in both positions with nothing refusing it.
    (
        "automatic caption within the two-line budget",
        _set(["clips", 0, "shots", 0, "automatic_caption"], "A tram brakes " * 8),
        "clips[0].shots[0].automatic_caption",
    ),
    (
        "scene prose within the two-line budget",
        _set(["clips", 0, "shots", 0, "scene", "prose"], "A wide street " * 8),
        "clips[0].shots[0].scene.prose",
    ),
    (
        "atmosphere prose within the two-line budget",
        _set(["clips", 0, "shots", 0, "atmosphere", "prose"], "Steady, then " * 8),
        "clips[0].shots[0].atmosphere.prose",
    ),
    (
        "scene token",
        _set(["clips", 0, "shots", 0, "scene", "token"], "[TRAM STOP]"),
        "clips[0].shots[0].scene.token",
    ),
    ("scene prose", _set(["clips", 0, "shots", 0, "scene", "prose"], ""), "clips[0].shots[0].scene.prose"),
    (
        "atmosphere phrase",
        _set(["clips", 0, "shots", 0, "atmosphere", "phrase"], "2 rises"),
        "clips[0].shots[0].atmosphere.phrase",
    ),
    ("sources non-empty", _set(["clips", 0, "shots", 0, "sources"], []), "clips[0].shots[0].sources"),
    ("sources at most eight", _nine_sources, "clips[0].shots[0].sources"),
    (
        "source id",
        _set(["clips", 0, "shots", 0, "sources", 0, "id"], "tr am"),
        "clips[0].shots[0].sources[0].id",
    ),
    (
        "duplicate source id",
        _set(["clips", 0, "shots", 0, "sources", 1, "id"], "tram"),
        "clips[0].shots[0].sources[1].id",
    ),
    (
        "token brackets",
        _set(["clips", 0, "shots", 0, "sources", 0, "token"], "[TRAM]"),
        "clips[0].shots[0].sources[0].token",
    ),
    (
        "token uppercase",
        _set(["clips", 0, "shots", 0, "sources", 0, "token"], "Tram braking"),
        "clips[0].shots[0].sources[0].token",
    ),
    (
        "prose",
        _set(["clips", 0, "shots", 0, "sources", 0, "prose"], ""),
        "clips[0].shots[0].sources[0].prose",
    ),
    (
        "two phrases",
        _set(["clips", 0, "shots", 0, "sources", 0, "phrases"], ["in frame"]),
        "clips[0].shots[0].sources[0].phrases",
    ),
    (
        "phrases carry no numbers",
        _set(["clips", 0, "shots", 0, "sources", 0, "phrases", 1], "3 times"),
        "clips[0].shots[0].sources[0].phrases[1]",
    ),
    (
        "two weights",
        _set(["clips", 0, "shots", 0, "sources", 3, "weights"], [0.1, 0.2, 0.3]),
        "clips[0].shots[0].sources[3].weights",
    ),
    (
        "weights in unit range",
        _set(["clips", 0, "shots", 0, "sources", 3, "weights", 0], 1.2),
        "clips[0].shots[0].sources[3].weights[0]",
    ),
    (
        "weights numeric",
        _set(["clips", 0, "shots", 0, "sources", 3, "weights", 1], "high"),
        "clips[0].shots[0].sources[3].weights[1]",
    ),
    (
        "audio role known",
        _set(["clips", 0, "shots", 0, "sources", 0, "role"], "backdrop"),
        "clips[0].shots[0].sources[0].role",
    ),
    (
        "visual role known",
        _set(["clips", 1, "shots", 0, "sources", 0, "role"], "underneath"),
        "clips[1].shots[0].sources[0].role",
    ),
    ("recognition covers every clip", _set(["followup", "recognition"], []), "followup.recognition"),
    (
        "recognition clip known",
        _set(["followup", "recognition", 0, "clip_id"], "demo_zoo"),
        "followup.recognition[0].clip_id",
    ),
    (
        "recognition sounds are tokens",
        _set(["followup", "recognition", 0, "sounds", 0], "[TRAM]"),
        "followup.recognition[0].sounds[0]",
    ),
    (
        "sound only clip known",
        _set(["followup", "sound_only", 0, "clip_id"], "demo_zoo"),
        "followup.sound_only[0].clip_id",
    ),
    (
        "sound only within the clip",
        _set(["followup", "sound_only", 0, "end_ms"], 12000),
        "followup.sound_only[0].end_ms",
    ),
    (
        "duplicate excerpt id",
        _set(["followup", "sound_only", 1, "excerpt_id"], "x1"),
        "followup.sound_only[1].excerpt_id",
    ),
    ("check covers every shaped clip", _set(["followup", "check"], []), "followup.check"),
    (
        "check only shaped clips",
        _set(["followup", "check", 0, "clip_id"], "demo_market"),
        "followup.check[0].clip_id",
    ),
    ("check side", _set(["followup", "check", 0, "a"], "mine"), "followup.check[0].a"),
]


@pytest.mark.parametrize(("name", "mutate", "path"), CASES, ids=[case[0] for case in CASES])
def test_each_rule_fails_naming_its_path(
    name: str, mutate: Callable[[dict[str, Any]], None], path: str
) -> None:
    document = copy.deepcopy(fixture_document())
    mutate(document)
    with pytest.raises(SessionDocumentError) as caught:
        validate_session_document(document)
    assert str(caught.value).startswith(f"{path}:"), (name, str(caught.value))


def test_a_file_that_is_not_json_names_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "session.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionDocumentError, match="session.json"):
        load_session_document(broken)


def test_narrowing_keeps_the_participant_blind() -> None:
    document = fixture_document()
    narrowed = participant_document(document)
    serialized = json.dumps(narrowed)
    # Weights are measurements (spec 2: never shown as numbers); prose is the
    # writer's business; the whole sources/scene/atmosphere block is what the
    # gated inventory delivers after the listing is submitted (spec 7).
    for leaked in ("weights", "prose", "sources", "scene", "atmosphere", "phrases", "token"):
        assert leaked not in serialized, leaked
    # A control clip's visual role heads are inventory text (spec 4.6: "three
    # role heads supplied with the visual inventory"): they travel only as the
    # gated inventory's role_heads, never in the pre-gate document.
    assert "visual_roles" not in serialized
    for head in document["clips"][1]["visual_roles"].values():
        assert head not in serialized, head
    for clip, original in zip(narrowed["clips"], document["clips"], strict=True):
        assert clip["clip_id"] == original["clip_id"]
        assert "visual_roles" not in clip
        for shot in clip["shots"]:
            assert set(shot) == {"shot_id", "start_ms", "end_ms", "automatic_caption"}
    # The follow-up answers: which clip an excerpt came from, which caption
    # set is A. Both stay on the server.
    assert all(set(entry) == {"excerpt_id"} for entry in narrowed["followup"]["sound_only"])
    assert all(set(entry) == {"clip_id"} for entry in narrowed["followup"]["check"])
    assert narrowed["followup"]["recognition"] == document["followup"]["recognition"]
    assert narrowed["measures"] == document["measures"]
    assert narrowed["poles"] == document["poles"]
