"""Cached browser delivery copies; study source media remains immutable."""

from __future__ import annotations

import asyncio
import fcntl
import os
import subprocess
import tempfile
from pathlib import Path

from starlette.responses import FileResponse
from starlette.types import Message, Receive, Scope, Send

from dpo.regen.study_schema import fingerprint


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


def streamable_video(source: Path, cache: Path, *, video_rate: int | None = None) -> Path:
    """Move MP4 metadata first; optionally constrain video bitrate, preserving audio."""
    if source.suffix.lower() != ".mp4":
        return source
    cache.mkdir(parents=True, exist_ok=True)
    variant = f"h264-{video_rate}-v1" if video_rate else "faststart"
    target = cache / f"{fingerprint(source)}-{variant}.mp4"
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
