"""Shot media cut for the writers: every cut lands whole, however many ask at once."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from dpo.caption import media


def _fake_ffmpeg(monkeypatch: pytest.MonkeyPatch, started: threading.Barrier) -> None:
    """Stand in for ffmpeg: wait for the other cut to start, then write the output path."""

    def run(tool: str, arguments: list[str]) -> str:
        started.wait(5)
        Path(arguments[-1]).write_bytes(b"RIFF" + b"\0" * 64)
        return ""

    monkeypatch.setattr(media, "_run", run)


def test_two_cuts_of_one_shot_at_once_do_not_share_a_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Barrier(2)
    _fake_ffmpeg(monkeypatch, started)
    destination = tmp_path / "shots" / "clip-0-1000.wav"
    results: list[Path] = []
    workers = [
        threading.Thread(
            target=lambda: results.append(media._cut_audio(tmp_path / "clip.mp4", destination, 0, 1000))
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    assert results == [destination, destination]
    assert destination.read_bytes().startswith(b"RIFF")
    # Both partials were replaced onto the destination; none is left, and
    # neither cut wrote into the other's file.
    assert not list((tmp_path / "shots").glob("*.part.wav"))
