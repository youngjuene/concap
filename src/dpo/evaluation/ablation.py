"""A content-free twin of a clip's video: same shape, nothing to see.

The congruency axis is the difference between explaining a caption with the
video and explaining it without. Dropping the video entirely does not measure
that. It also removes ~2500 soft tokens of context, and a language model's
absolute log-probability moves with how much context precedes the completion
regardless of what that context contains — so a per-clip offset of unknown size
rides on every score. Within one clip the offset is shared by every candidate,
so rung ORDER and spacing survive it; the claim that does not survive is the
one about the axis's zero, which is exactly what a negative congruency asserts.

Replacing the video with a same-length one that carries no information keeps
the token count, the frame count and the positions identical, so the only thing
that differs between the two passes is what is in the picture. This is the
ablation that visual contrastive decoding uses (Leng et al., CVPR 2024) in place
of the deletion used by context-aware decoding and M3ID.

Mid-gray rather than noise: it is content-free without being a stimulus of its
own, and it needs no seed, so the twin — and therefore the measurement — is
reproducible without one.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

GRAY = 0x80
"""Mid-gray in every plane: neutral luma, neutral chroma, no edges."""


class AblationError(RuntimeError):
    """Raised when a clip's video cannot be read or its twin cannot be written."""


def _geometry(source: Path) -> tuple[int, int, Fraction, int]:
    """(width, height, frame rate, frame count) of a clip's video stream."""
    import av

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise AblationError(f"{source} has no video stream to ablate")
            stream = container.streams.video[0]
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            rate = stream.average_rate or Fraction(24, 1)
            frames = int(stream.frames or 0)
            if frames <= 0:
                # Some containers do not record a frame count; the decode is the
                # only honest answer, and a twin with the wrong length would put
                # back the very context-length difference this removes.
                frames = sum(1 for _ in container.decode(video=0))
    except AblationError:
        raise
    except Exception as exc:  # noqa: BLE001 - av raises a wide family of decode errors
        raise AblationError(f"could not read the video geometry of {source}: {exc}") from exc
    if width <= 0 or height <= 0 or frames <= 0:
        raise AblationError(f"{source} reports an empty video stream ({width}x{height}, {frames} frames)")
    return width, height, Fraction(rate), frames


def gray_counterpart(source: Path, destination: Path) -> Path:
    """Write ``source``'s geometry with none of its content, and return the path."""
    import av

    width, height, rate, frames = _geometry(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with av.open(str(destination), mode="w") as container:
            stream = container.add_stream("libx264", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame(width, height, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes([GRAY]) * plane.buffer_size)
            for number in range(frames):
                frame.pts = number
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:  # noqa: BLE001 - av raises a wide family of encode errors
        raise AblationError(f"could not write a gray counterpart for {source}: {exc}") from exc
    return destination


def gray_counterparts(videos: dict[str, Path], directory: Path) -> dict[str, Path]:
    """One gray twin per clip, named for the clip so a run is inspectable."""
    return {
        clip_id: gray_counterpart(source, directory / f"{clip_id}.mp4")
        for clip_id, source in sorted(videos.items())
    }
