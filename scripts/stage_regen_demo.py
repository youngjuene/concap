"""Synthetic media for a regeneration document, so the instrument runs without footage.

Nothing here imports from ``dpo``: the script reads the document's own fields,
so it stages whatever a researcher's document declares.

Per segment under ``<out>/<segment>/``:

``audio.wav``
    The clip's sound on its own, which is what §6's model is given: a caption
    model's audio stack reads wav and little else.

``clip.mp4``
    A ten-second 640x360 clip with sound. The two segments get different
    patterns and different tones, because the study asks a participant to
    report on one and then watch the other; identical clips would make the
    second viewing a repeat of the first even in a demo.

``frames/<n>.png``
    One picture per frame of §4's strip, pulled from the clip at the moment the
    document says that frame is from — extracted rather than drawn, so the
    strip and the footage agree.

``masks/<n>/<object>.png``
    One binary mask per object per frame, laid out as horizontal bands in
    document order with the last object a small square inside the others, and
    the square shifted along the strip so a mark on one frame is not a mark on
    another. §4's rule is that the smallest containing mask wins, so a demo
    where nothing overlaps would never exercise it.

``stems/<source>.wav``
    One tone per source, amplitude-shaped by that source's own ``waveform``
    array from the document, so the envelope drawn in the §5 lane is the
    envelope the participant hears. A demo where the picture and the sound
    disagreed would teach the wrong thing about the screen.

    uv run python scripts/stage_regen_demo.py \
        --session tests/regen/fixtures/regen.json --out data/regen-demo/media

Idempotent: existing files are skipped. Needs ffmpeg with libx264 and aac.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

WIDTH, HEIGHT, RATE = 640, 360, 25
AUDIO_RATE = 16000
PATTERNS = ("testsrc2", "smptebars")
BASE_HZ = (220.0, 330.0)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(command[:3])}… failed:\n{result.stderr.strip()}")


def stage_clip(path: Path, seconds: float, pattern: str, hertz: float) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"{pattern}=size={WIDTH}x{HEIGHT}:rate={RATE}:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={hertz}:sample_rate={AUDIO_RATE}:duration={seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )


def stage_still(clip: Path, path: Path, at_seconds: float) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{at_seconds:.3f}",
            "-i",
            str(clip),
            "-frames:v",
            "1",
            str(path),
        ]
    )


def stage_mask(path: Path, index: int, total: int, drift: int = 0) -> None:
    """A band per object, and a square inside them for the last one.

    ``drift`` moves the square along the frame, so the strip's frames do not
    all carry the same masks: a mark placed on one frame must be matched
    against that frame's masks, and a demo where every frame agreed could not
    show the difference.

    Written as a minimal 8-bit greyscale PNG by hand: the mask is the one asset
    a demo cannot fake with ffmpeg, and pulling in an image library for four
    rectangles would put a dependency in a staging script.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last = index == total - 1
    top, bottom = (HEIGHT * index) // total, (HEIGHT * (index + 1)) // total
    left = max(0, min(WIDTH - 120, 200 + drift))
    rows = []
    for y in range(HEIGHT):
        if last:
            inside = 120 <= y < 220
            row = bytes(255 if inside and left <= x < left + 120 else 0 for x in range(WIDTH))
        else:
            row = bytes(255 if top <= y < bottom else 0 for x in range(WIDTH))
        rows.append(b"\x00" + row)
    _write_png(path, b"".join(rows))


def _write_png(path: Path, raw: bytes) -> None:
    import binascii
    import zlib

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def stage_stem(path: Path, seconds: float, hertz: float, waveform: list[float]) -> None:
    """A tone shaped by the lane's own envelope, so the two agree on screen."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(AUDIO_RATE * seconds)
    samples = bytearray()
    for index in range(frames):
        position = index / max(1, frames - 1)
        bin_index = min(len(waveform) - 1, int(position * len(waveform)))
        amplitude = waveform[bin_index] * 0.6
        value = int(32767 * amplitude * math.sin(2 * math.pi * hertz * index / AUDIO_RATE))
        samples += struct.pack("<h", value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_RATE)
        handle.writeframes(bytes(samples))


def stage_sound(clip: Path, path: Path) -> None:
    """The clip's own audio as a mono wav, which is what §6's model reads."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clip),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_RATE),
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )


def stage(document: dict[str, Any], out: Path) -> None:
    for index, (name, segment) in enumerate(sorted(document["segments"].items())):
        seconds = segment["duration_ms"] / 1000
        clip = out / segment["video"]
        stage_clip(clip, seconds, PATTERNS[index % len(PATTERNS)], BASE_HZ[index % len(BASE_HZ)])
        stage_sound(clip, out / segment["audio"])
        for step, frame in enumerate(segment["frames"]):
            stage_still(clip, out / frame["still"], frame["at_ms"] / 1000)
            objects = frame["objects"]
            for position, entry in enumerate(objects):
                stage_mask(out / entry["mask"], position, len(objects), drift=step * 60)
        for position, stem in enumerate(segment["stems"]):
            stage_stem(
                out / stem["audio"],
                seconds,
                BASE_HZ[index % len(BASE_HZ)] * (position + 1),
                list(stem["waveform"]),
            )
        print(f"staged segment {name} ({segment['clip_id']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="the regen document to stage for")
    parser.add_argument("--out", required=True, help="media directory to write into")
    arguments = parser.parse_args(argv)
    document = json.loads(Path(arguments.session).read_text(encoding="utf-8"))
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    stage(document, out)
    print(f"media staged under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
