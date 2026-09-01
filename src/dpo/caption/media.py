"""Media the session serves that is not on disk as such: shot audio, stills, excerpts.

The clip itself is served as the file it is (the same lookup as
``dpo.userstudy.app``: ``unmuted_video/`` first, then the media dir — the
participant must hear the clip, so a muted corpus render is the fallback of
last resort and never the first choice). Everything else is cut from it:

* a shot's audio, 16 kHz mono, for the Gemma writer to condition on;
* one still per clip, for the follow-up's recognition and sound-only screens
  (spec 8: "still frame of the clip", "stills of all six");
* an excerpt's audio, for the sound-only screen, cut by the SERVER so the
  browser never receives the clip the excerpt belongs to (that is the answer).

All of it is cached under ``cache_dir`` and skipped when present: the cuts are
deterministic functions of the file and the bounds, and a second run of the
same session should cost nothing. ffmpeg is the one tool that does all three
without a Python decoder dependency; it fails loudly with its own stderr.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from dpo.annotation.webapp import VIDEO_SUFFIXES

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
AUDIO_RATE = 16000
STILL_AT_MS = 1000


class MediaError(RuntimeError):
    """ffmpeg is missing, or it failed; the message carries its stderr."""


def find_clip_video(media_dir: Path, clip_id: str) -> Path | None:
    for base in (media_dir / "unmuted_video", media_dir):
        for suffix in VIDEO_SUFFIXES:
            candidate = base / f"{clip_id}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def require_clip_video(media_dir: Path, clip_id: str) -> Path:
    path = find_clip_video(media_dir, clip_id)
    if path is None:
        raise MediaError(f"no video for clip {clip_id!r} under {media_dir}")
    return path


def _run(tool: str, arguments: list[str]) -> str:
    if shutil.which(tool) is None:
        raise MediaError(f"{tool} is not on PATH")
    completed = subprocess.run(
        [tool, "-loglevel", "error", "-y", *arguments] if tool == FFMPEG else [tool, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise MediaError(f"{tool} failed ({completed.returncode}): {completed.stderr.strip()}")
    return completed.stdout


def clip_duration_ms(path: Path) -> int | None:
    """The container duration from ffprobe, or None when it cannot be read."""
    try:
        output = _run(
            FFPROBE,
            ["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        )
        return int(round(float(output.strip()) * 1000))
    except (MediaError, ValueError):
        return None


def _cut_audio(source: Path, destination: Path, start_ms: int, end_ms: int) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part.wav")
    _run(
        FFMPEG,
        [
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-t",
            f"{(end_ms - start_ms) / 1000:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_RATE),
            "-c:a",
            "pcm_s16le",
            str(partial),
        ],
    )
    partial.replace(destination)
    return destination


def shot_audio(media_dir: Path, cache_dir: Path, clip_id: str, start_ms: int, end_ms: int) -> Path:
    """``<cache_dir>/shots/<clip_id>-<start>-<end>.wav`` — for the Gemma writer."""
    source = require_clip_video(media_dir, clip_id)
    return _cut_audio(source, cache_dir / "shots" / f"{clip_id}-{start_ms}-{end_ms}.wav", start_ms, end_ms)


def excerpt_audio(
    media_dir: Path, cache_dir: Path, excerpt_id: str, clip_id: str, start_ms: int, end_ms: int
) -> Path:
    """``<cache_dir>/excerpts/<excerpt_id>.wav`` — named by the excerpt, not the clip."""
    source = require_clip_video(media_dir, clip_id)
    return _cut_audio(source, cache_dir / "excerpts" / f"{excerpt_id}.wav", start_ms, end_ms)


def clip_still(media_dir: Path, cache_dir: Path, clip_id: str) -> Path:
    """One jpeg at 1000 ms, or at the midpoint of a clip too short for that.

    A seek at or past the last frame makes ffmpeg exit 0 having written
    nothing, so the frame is taken no later than the midpoint and the absence
    of an output file is an error rather than a missing still.
    """
    destination = cache_dir / "stills" / f"{clip_id}.jpg"
    if destination.is_file():
        return destination
    source = require_clip_video(media_dir, clip_id)
    duration = clip_duration_ms(source)
    at_ms = STILL_AT_MS if duration is None else min(STILL_AT_MS, duration // 2)
    return _frame(source, at_ms, destination)


def shot_still(media_dir: Path, cache_dir: Path, clip_id: str, start_ms: int, end_ms: int) -> Path:
    """One jpeg from the middle of a shot: what the control task's writer looks at.

    The middle rather than the first frame, because a shot boundary is the
    frame most likely to be a transition and least representative of the shot.
    """
    destination = cache_dir / "stills" / f"{clip_id}-{start_ms}-{end_ms}.jpg"
    if destination.is_file():
        return destination
    return _frame(require_clip_video(media_dir, clip_id), (start_ms + end_ms) // 2, destination)


def _frame(source: Path, at_ms: int, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part.jpg")
    _run(
        FFMPEG,
        ["-ss", f"{at_ms / 1000:.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "3", str(partial)],
    )
    if not partial.is_file():
        raise MediaError(f"ffmpeg produced no frame at {at_ms} ms from {source}")
    partial.replace(destination)
    return destination
