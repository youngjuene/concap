"""`dpo session serve`: a wrong command line is reported as such, never as a wrong document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from dpo.cli.session import SessionUsageError, _gemma_writer, _session_serve

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
