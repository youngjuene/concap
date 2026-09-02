"""`dpo session serve`: a wrong command line is reported as such, never as a wrong document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from dpo.cli.session import (
    SessionUsageError,
    _gemma_writer,
    _session_scaffold,
    _session_serve,
)
from dpo.session.document import SessionDocumentError, validate_session_document

FIXTURE = Path(__file__).parent / "fixtures" / "session.json"


def _arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "session": str(FIXTURE),
        "media_dir": "media",
        "out": "out",
        "writer": "gemma",
        "backend_config": None,
        "contract": "configs/study/street-audio.toml",
        "checkpoint": None,
        "host": "127.0.0.1",
        "port": 8777,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_gemma_without_a_backend_config_is_a_usage_error_not_an_invalid_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SessionUsageError, match="--backend-config"):
        _gemma_writer(_arguments())
    assert _session_serve(_arguments()) == 2
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == "error" and emitted["command"] == "session serve"
    assert "--writer gemma requires --backend-config" in emitted["error"]
    # The document was never blamed: no "invalid" status, no session path.
    assert "session" not in emitted


# ---- scaffolding a document from a mask-link manifest ------------------------


def _manifest(path: Path, clip_id: str = "clip_001") -> Path:
    """A minimal ``dpo.caption-mask-link/v1`` manifest, shaped like link-masks writes."""

    def audio(source_id: str, token: str, weights: list[float]) -> dict[str, object]:
        return {
            "label": token.title(),
            "parameters": {"admission": {"candidate": True}},
            "session_source": {
                "id": source_id,
                "token": token,
                "prose": token.lower(),
                "phrases": ["in frame", "comes and goes"],
                "weights": weights,
                "role": None,
                "review_required": ["prose", "phrases[1]", "role", "weights[1]"],
            },
        }

    def visual(source_id: str, token: str, weights: list[float], *, candidate: bool) -> dict[str, object]:
        return {
            "label": token.title(),
            "parameters": {"admission": {"candidate": candidate}},
            "session_source": {
                "id": source_id,
                "token": token,
                "prose": token.lower(),
                "phrases": ["in frame", "visible throughout"],
                "weights": weights,
                "role": "fixed",
                "review_required": ["prose", "phrases[1]"],
            },
        }

    path.write_text(
        json.dumps(
            {
                "schema": "dpo.caption-mask-link/v1",
                "clips": {
                    clip_id: {
                        "audio": [audio("siren", "SIREN", [0.2, 0.9]), audio("idling", "IDLING", [0.9, 0.3])],
                        "visual": [
                            visual("road", "ROAD", [0.9, 0.2], candidate=True),
                            visual("sky", "SKY", [0.3, 0.9], candidate=True),
                            visual("person", "PERSON", [0.0, 0.0], candidate=False),
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _scaffold_arguments(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "media_dir": str(tmp_path / "media"),
        "out": str(tmp_path / "session.json"),
        "clips": ["clip_001"],
        "mask_links": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_link_masks_with_an_empty_clips_flag_refuses_rather_than_linking_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dpo.cli.session import _session_link_masks

    arguments = argparse.Namespace(
        mask_root=str(tmp_path / "masks"),
        tidy_data=str(tmp_path / "tidy.csv"),
        ontology=None,
        fps=60.0,
        out=str(tmp_path / "links.json"),
        clips=[],
    )
    assert _session_link_masks(arguments) == 2
    assert "names no clip" in json.loads(capsys.readouterr().out)["error"]
    assert not (tmp_path / "links.json").exists()


def test_scaffold_without_mask_links_leaves_the_sources_empty(tmp_path: Path) -> None:
    assert _session_scaffold(_scaffold_arguments(tmp_path)) == 0
    document = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert document["clips"][0]["shots"][0]["sources"] == []


def test_scaffold_fills_a_shaped_clip_from_the_audio_branch(tmp_path: Path) -> None:
    links = _manifest(tmp_path / "links.json")
    assert _session_scaffold(_scaffold_arguments(tmp_path, mask_links=str(links))) == 0
    document = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    clip = document["clips"][0]

    assert clip["task"] == "shaped"
    sources = clip["shots"][0]["sources"]
    assert [source["id"] for source in sources] == ["siren", "idling"]
    assert [source["weights"] for source in sources] == [[0.2, 0.9], [0.9, 0.3]]
    # The audio role is the one field link-masks refuses to guess.
    assert all(source["role"] is None for source in sources)


def test_scaffold_fills_a_control_clip_from_the_visual_branch_and_drops_empty_masks(
    tmp_path: Path,
) -> None:
    links = _manifest(tmp_path / "links.json", clip_id="clip_002")
    arguments = _scaffold_arguments(tmp_path, clips=["clip_001", "clip_002"], mask_links=str(links))
    _manifest(tmp_path / "links.json", clip_id="clip_001")
    both = json.loads((tmp_path / "links.json").read_text(encoding="utf-8"))
    both["clips"]["clip_002"] = both["clips"]["clip_001"]
    (tmp_path / "links.json").write_text(json.dumps(both), encoding="utf-8")

    assert _session_scaffold(arguments) == 0
    document = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    control = document["clips"][1]

    assert control["task"] == "control"
    sources = control["shots"][0]["sources"]
    # PERSON has an empty mask everywhere, so it is not a thing in this shot.
    assert [source["id"] for source in sources] == ["road", "sky"]
    assert all(source["role"] is not None for source in sources)


def test_a_scaffold_from_mask_links_still_does_not_validate(tmp_path: Path) -> None:
    links = _manifest(tmp_path / "links.json")
    _session_scaffold(_scaffold_arguments(tmp_path, mask_links=str(links)))
    document = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))

    with pytest.raises(SessionDocumentError):
        validate_session_document(document)


def test_the_unset_role_error_names_the_source_and_says_why(tmp_path: Path) -> None:
    links = _manifest(tmp_path / "links.json")
    _session_scaffold(_scaffold_arguments(tmp_path, mask_links=str(links)))
    document = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    # Author everything the scaffold leaves blank except the role.
    shot = document["clips"][0]["shots"][0]
    shot["automatic_caption"] = "A siren passes while an engine idles."
    shot["scene"] = {"token": "A CITY STREET", "prose": "A city street."}
    shot["atmosphere"] = {"phrase": "steady, with one rise", "prose": "Steady, with one rise."}
    document["followup"]["recognition"] = [{"clip_id": "clip_001", "sounds": ["BUS BRAKES"]}]

    with pytest.raises(SessionDocumentError) as raised:
        validate_session_document(document)
    message = str(raised.value)
    assert message.startswith("clips[0].shots[0].sources[0].role")
    assert "no mask measures it" in message


def test_a_clip_missing_from_the_manifest_is_a_usage_error(tmp_path: Path) -> None:
    links = _manifest(tmp_path / "links.json")
    arguments = _scaffold_arguments(tmp_path, clips=["clip_404"], mask_links=str(links))

    assert _session_scaffold(arguments) == 2


def test_a_manifest_of_the_wrong_schema_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": "something/v1", "clips": {}}), encoding="utf-8")

    assert _session_scaffold(_scaffold_arguments(tmp_path, mask_links=str(other))) == 2
