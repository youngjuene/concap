"""Cached browser delivery copies; study source media remains immutable."""

from __future__ import annotations

import asyncio
import fcntl
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from starlette.responses import FileResponse
from starlette.types import Message, Receive, Scope, Send

from dpo.regen.study_schema import bound_media, fingerprint

VIEWING_VIDEO_RATE = 1_600_000


class PacedVideoResponse(FileResponse):
    """Leave relay capacity for controls instead of filling its queue with video."""

    chunk_size = 32 * 1024
    bytes_per_second = 256 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        loop = asyncio.get_running_loop()
        next_send = loop.time()

        async def paced_send(message: Message) -> None:
            nonlocal next_send
            body = message.get("body", b"")
            if message["type"] == "http.response.body" and body:
                await asyncio.sleep(max(0, next_send - loop.time()))
                next_send = max(next_send, loop.time()) + len(body) / self.bytes_per_second
            await send(message)

        # A server-side sendfile extension would bypass body pacing.
        scope = {
            **scope,
            "extensions": {
                key: value
                for key, value in scope.get("extensions", {}).items()
                if key != "http.response.pathsend"
            },
        }
        await super().__call__(scope, receive, paced_send)


def _delivery_path(source: Path, cache: Path, video_rate: int | None) -> Path:
    if source.suffix.lower() != ".mp4":
        return source
    variant = f"h264-{video_rate}-v1" if video_rate else "faststart"
    return cache / f"{fingerprint(source)}-{variant}.mp4"


def prepared_video(source: Path, cache: Path, *, video_rate: int | None = None) -> Path:
    """Request-time lookup only: encoding belongs to preparation before entry."""
    target = _delivery_path(source, cache, video_rate)
    if not target.is_file():
        raise ValueError("Video delivery has not been prepared; prepare the study before participant entry")
    return target


def streamable_video(source: Path, cache: Path, *, video_rate: int | None = None) -> Path:
    """Move MP4 metadata first; optionally constrain video bitrate, preserving audio."""
    target = _delivery_path(source, cache, video_rate)
    if target == source:
        return source
    cache.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_file():
            return target
        descriptor, name = tempfile.mkstemp(suffix=".mp4", dir=cache)
        os.close(descriptor)
        temporary = Path(name)
        try:
            encode = (
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "24",
                    "-maxrate",
                    str(video_rate),
                    "-bufsize",
                    str(video_rate * 2),
                    "-threads",
                    "4",
                ]
                if video_rate
                else []
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0",
                    "-c",
                    "copy",
                    *encode,
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
                timeout=1800 if video_rate else 120,
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def prepare_delivery_videos(manifest: dict[str, Any], media: Path, cache: Path) -> list[dict[str, Any]]:
    """Prepare every available viewing asset once and record reuse and delivered hashes."""
    records = []
    for kind, entries in (
        ("calibration", manifest.get("calibration_clips", [])),
        ("watch", manifest.get("viewing_videos", [])),
    ):
        for index, entry in enumerate(entries):
            if kind == "watch" and entry.get("status") != "ready":
                continue
            started = time.monotonic()
            source = bound_media(media, entry, "video")
            rate = VIEWING_VIDEO_RATE if kind == "watch" else None
            hit = _delivery_path(source, cache, rate).is_file()
            target = streamable_video(source, cache, video_rate=rate)
            records.append(
                {
                    "kind": kind,
                    "index": index,
                    "id": entry["id"],
                    "source": str(source),
                    "source_sha256": fingerprint(source),
                    "path": str(target),
                    "sha256": fingerprint(target),
                    "cache_hit": hit,
                    "uses_original": target == source,
                    "video_rate": rate,
                    "preparation_ms": round((time.monotonic() - started) * 1000),
                }
            )
    return records
