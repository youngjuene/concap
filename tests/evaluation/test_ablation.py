"""The ablation twin: a clip's shape with none of its content."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from dpo.evaluation.ablation import AblationError, _geometry, gray_counterpart, gray_counterparts

WIDTH, HEIGHT, FRAMES, RATE = 64, 48, 12, 10


def _clip(path: Path) -> Path:
    """A tiny video whose picture actually changes, so a twin cannot be a copy."""
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=RATE)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        for number in range(FRAMES):
            picture = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            picture[:, (number * 5) % WIDTH :] = 255
            frame = av.VideoFrame.from_ndarray(picture, format="rgb24")
            frame.pts = number
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _frames(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def test_the_twin_keeps_the_geometry_that_sets_the_token_count(tmp_path: Path) -> None:
    source = _clip(tmp_path / "clip.mp4")
    twin = gray_counterpart(source, tmp_path / "gray.mp4")
    # Width, height, rate and frame count all feed how many soft tokens the
    # vision tower emits. If any of them moved, the ablation would put back the
    # context-length difference it exists to remove.
    assert _geometry(twin) == _geometry(source)


def test_the_twin_carries_no_picture(tmp_path: Path) -> None:
    source = _clip(tmp_path / "clip.mp4")
    twin = gray_counterpart(source, tmp_path / "gray.mp4")
    assert _frames(source).std() > 0.0
    ablated = _frames(twin)
    assert ablated.std() == 0.0
    # Flat, and flat in the middle: a black or white field is a stimulus of its
    # own, and would ask the model to explain a caption against one.
    assert 100 < float(ablated.flat[0]) < 160


def test_the_twin_is_reproducible(tmp_path: Path) -> None:
    source = _clip(tmp_path / "clip.mp4")
    first = gray_counterpart(source, tmp_path / "a.mp4").read_bytes()
    second = gray_counterpart(source, tmp_path / "b.mp4").read_bytes()
    # The measurement derived from these twins goes into a content-addressed
    # artifact, so the twins themselves may not drift between runs.
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_counterparts_are_named_for_their_clip(tmp_path: Path) -> None:
    sources = {name: _clip(tmp_path / f"{name}.mp4") for name in ("clip-b", "clip-a")}
    twins = gray_counterparts(sources, tmp_path / "ablation")
    assert set(twins) == {"clip-a", "clip-b"}
    assert twins["clip-a"].name == "clip-a.mp4"
    assert all(path.is_file() for path in twins.values())


def test_a_file_with_no_picture_is_refused(tmp_path: Path) -> None:
    not_a_video = tmp_path / "notes.txt"
    not_a_video.write_text("no video stream here")
    with pytest.raises(AblationError, match="could not read the video geometry"):
        gray_counterpart(not_a_video, tmp_path / "gray.mp4")
