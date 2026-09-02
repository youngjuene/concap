"""Synthetic media for either instrument's document, so it runs without footage.

Nothing here is imported from ``dpo``: the script reads two fields, so it
stages for the skeleton instrument and the console alike.

Every clip in the document gets a 10 s (or the clip's own length) 640x360 mp4
with sound under ``<out>/unmuted_video/<clip_id>.mp4`` — the layout the
session app looks in first. The visuals and the tones DIFFER per clip on
purpose: the follow-up asks which clip a sound came from and shows stills of
all of them, so identical clips would make that screen unanswerable even in a
demo. Each tone also changes amplitude mid-clip so a shot boundary has
something audible on either side of it.

    uv run python scripts/stage_session_demo.py \
        --session tests/session/fixtures/session.json --out data/session-demo/media

Idempotent: existing files are skipped. Needs ffmpeg with libx264 and aac.
``gradients`` arrived in ffmpeg 4.4; on an older ffmpeg (4.2 here) the script
substitutes ``mandelbrot`` so the list of distinct patterns stays six long.

``--container webm`` writes VP9 + Opus ``.webm`` instead (needs libvpx-vp9 and
libopus). The tablet plays mp4; the open-source Chromium that Playwright ships
decodes neither H.264 nor AAC, so the headless walk of the kiosk runs on a webm
staging. The session app accepts both suffixes (``VIDEO_SUFFIXES``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

WIDTH, HEIGHT, RATE = 640, 360, 25
AUDIO_RATE = 16000
BASE_HZ = 220.0
# Six distinct lavfi video sources; the last is testsrc with its hue rotated.
PATTERNS = ("testsrc2", "smptebars", "rgbtestsrc", "gradients", "testsrc", "testsrc2:hue")
# Container -> (video codec, audio codec, extra output flags).
CONTAINERS = {
    "mp4": ("libx264", "aac", ["-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
    "webm": ("libvpx-vp9", "libopus", ["-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p"]),
}


def _has_filter(name: str) -> bool:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=False
    )
    return any(line.split()[1:2] == [name] for line in completed.stdout.splitlines() if line.strip())


def _video_source(pattern: str, duration: float, *, gradients_available: bool) -> tuple[str, list[str]]:
    name, _, variant = pattern.partition(":")
    if name == "gradients" and not gradients_available:
        name = "mandelbrot"
    if name == "mandelbrot":
        source = f"mandelbrot=size={WIDTH}x{HEIGHT}:rate={RATE}"
    elif name == "gradients":
        source = f"gradients=size={WIDTH}x{HEIGHT}:rate={RATE}:speed=0.05"
    else:
        source = f"{name}=size={WIDTH}x{HEIGHT}:rate={RATE}"
    filters = ["-vf", "hue=h=140:s=1.4"] if variant == "hue" else []
    return source, filters


def _audio_source(index: int, duration: float) -> str:
    # Distinct pitch per clip (a major-third ladder), plus a louder burst from
    # 40% to 55% of the clip so the middle shot boundary is audible.
    hz = BASE_HZ * (1.25**index)
    burst_lo, burst_hi = 0.4 * duration, 0.55 * duration
    expression = (
        f"0.35*sin(2*PI*{hz:.2f}*t)*(1+1.2*gt(t,{burst_lo:.2f})*lt(t,{burst_hi:.2f}))"
        f"+0.08*sin(2*PI*{hz * 1.5:.2f}*t)"
    )
    return f"aevalsrc='{expression}':s={AUDIO_RATE}:d={duration:.3f}"


def stage(session_path: Path, out: Path, container: str = "mp4") -> list[Path]:
    document = json.loads(session_path.read_text(encoding="utf-8"))
    video_codec, audio_codec, codec_flags = CONTAINERS[container]
    target = out / "unmuted_video"
    target.mkdir(parents=True, exist_ok=True)
    gradients_available = _has_filter("gradients")
    written: list[Path] = []
    for index, clip in enumerate(document["clips"]):
        destination = target / f"{clip['clip_id']}.{container}"
        if destination.is_file():
            continue
        duration = max(shot["end_ms"] for shot in clip["shots"]) / 1000.0
        video, filters = _video_source(
            PATTERNS[index % len(PATTERNS)], duration, gradients_available=gradients_available
        )
        partial = destination.with_name(f"{destination.name}.part.{container}")
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            video,
            "-f",
            "lavfi",
            "-i",
            _audio_source(index, duration),
            *filters,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            video_codec,
            "-c:a",
            audio_codec,
            "-b:a",
            "96k",
            *codec_flags,
            str(partial),
        ]
        subprocess.run(command, check=True)
        partial.replace(destination)
        written.append(destination)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session",
        required=True,
        help="any document with clips[].clip_id and clips[].shots[].end_ms "
        "(dpo.caption-session/v1 or dpo.caption-console/v1)",
    )
    parser.add_argument("--out", required=True, help="media directory; clips go under unmuted_video/")
    parser.add_argument(
        "--container",
        choices=sorted(CONTAINERS),
        default="mp4",
        help="mp4 (H.264 + AAC, the tablet's format) or webm (VP9 + Opus, for a Chromium without H.264)",
    )
    arguments = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH", file=sys.stderr)
        return 2
    written = stage(Path(arguments.session), Path(arguments.out), arguments.container)
    print(json.dumps({"status": "staged", "written": [str(path) for path in written], "out": arguments.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
