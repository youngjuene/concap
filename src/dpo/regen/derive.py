"""Web-sized copies of the staged media, made once and served in its place.

The media a study is staged from is archival: §4's stills are full-resolution
PNGs because the masks were cut from those pixels, and the clip is one file
carrying both tracks because that is what the participant watches. Neither is
what a browser should be asked to download. A strip of five stills is 6.4 MB
of PNG for a picture drawn 880 px wide, and §5's reference mix is the clip's
6.2 MB fetched a second time for the 244 KB of sound inside it. On campus that
is invisible. Over the published link of ``deploy/edge`` it is the wait the
participant reads as the instrument being slow.

So nothing here is re-staged. The files under ``--media-dir`` stay exactly as
``scripts/stage_regen_media.py`` wrote them — they are the study's record —
and this module keeps a second, derived copy beside them for the browser:

*The still becomes a WebP at its own size.* Same pixels, same dimensions, no
crop and no downscale, because §4 asks a participant to find a small thing in
a street and a resample would take that decision for them. Only the encoding
changes, and at quality 88 that is a twelfth of the bytes. Coordinates are
normalised to the delivered image and the masks are matched on the server
against the original PNGs, so what §4 records is unmoved by any of this.

*The clip's sound becomes an audio-only container.* Copied, not re-encoded:
the AAC bitstream is the same bytes it is in the mp4, so §5's mix is the
clip's own audio at the clip's own level, which is the one thing that file is
required to be. It is the video track that is dropped, and §5 never showed it.

*A derivative that cannot be made is not an error.* Every entry point falls
back to the source file, once and quietly, and the instrument serves what it
always served. A missing encoder, an unwritable cache or a file ffmpeg will
not open costs the participant some bandwidth, never the session.

Derivatives are keyed by what they were made from — the source's path, size
and modification time, plus the recipe's own version — so re-staging a clip
produces a new key rather than a stale picture, and nothing has to be swept.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

# What :meth:`Derivatives._derive` calls: source in, target written.
Maker = Callable[[Path, Path], None]

# The directory of derived copies, under the media directory so it travels with
# the media it was made from and is obvious to anyone looking at the staging.
CACHE_DIR = ".derived"

# Bumped when a recipe changes what it produces, which retires every key it
# ever wrote without anyone having to delete a file.
STILL_RECIPE = "still-webp-q88-v1"
SOUND_RECIPE = "sound-m4a-copy-v1"

# §4's stills are marked on, not glanced at: the participant is asked to put a
# point on a particular thing in a street scene. 88 is where these frames stop
# changing to the eye — 12x smaller than the PNG, and still 25% larger than the
# quality below it, which is the direction to err in for a picture being
# measured against.
STILL_QUALITY = 88

STILL_TYPE = "image/webp"
SOUND_TYPE = "audio/mp4"


class Derivatives:
    """The derived copies of one media directory, made on demand and kept.

    Construction never fails and never touches the disk. Each accessor returns
    a path to serve: the derivative when there is one, the source otherwise.
    """

    def __init__(self, media_dir: Path, *, cache_dir: Path | None = None, enabled: bool = True) -> None:
        self._media = Path(media_dir).resolve()
        self._cache = Path(cache_dir) if cache_dir is not None else self._media / CACHE_DIR
        self._enabled = enabled
        # One lock per key, so two requests for the same frame encode once, and
        # requests for different frames do not queue behind each other. Guarded
        # by its own lock because the route runs in a thread pool.
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        # Sources whose recipe has already failed. A file ffmpeg will not open
        # will not open on the next request either, and retrying would put the
        # cost of the failure on every participant instead of the first.
        self._refused: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def still(self, source: Path) -> tuple[Path, str | None]:
        """§4's frame as WebP, or the source PNG if it cannot be made."""
        return self._derive(source, STILL_RECIPE, ".webp", _encode_still, STILL_TYPE)

    def sound(self, source: Path) -> tuple[Path, str | None]:
        """The clip's audio alone, or the whole clip if it cannot be made."""
        return self._derive(source, SOUND_RECIPE, ".m4a", _extract_sound, SOUND_TYPE)

    def warm(self, stills: Iterable[Path] = (), sounds: Iterable[Path] = ()) -> tuple[int, int, int]:
        """Make every derivative now, so no participant waits for the first one.

        Returns ``(stills made, sounds made, sources still served whole)``. The
        last is the honest one: it is the count of files for which a derivative
        could not be made, and the operator is told rather than left to notice
        the bandwidth.
        """
        ready = [0, 0]
        whole = 0
        for index, (sources, derive) in enumerate(((stills, self.still), (sounds, self.sound))):
            for source in sources:
                path, _ = derive(source)
                if path == source:
                    whole += 1
                else:
                    ready[index] += 1
        return ready[0], ready[1], whole

    def _derive(
        self,
        source: Path,
        recipe: str,
        suffix: str,
        make: Maker,
        media_type: str,
    ) -> tuple[Path, str | None]:
        if not self._enabled:
            return source, None
        try:
            key = self._key(source, recipe)
        except OSError:
            return source, None
        target = self._cache / f"{key}{suffix}"
        if target.is_file():
            return target, media_type
        if key in self._refused:
            return source, None
        with self._lock_for(key):
            # Another thread may have finished it while this one waited.
            if target.is_file():
                return target, media_type
            if key in self._refused:
                return source, None
            try:
                self._cache.mkdir(parents=True, exist_ok=True)
                # Written under a private name and moved into place, so a
                # reader never opens a half-encoded file and a crash mid-encode
                # leaves nothing to mistake for a derivative. The suffix stays
                # last: ffmpeg reads the container to write from the extension,
                # and a name ending in a thread id is a format it does not know.
                pending = self._cache / f".{key}.{os.getpid()}.{threading.get_ident()}{suffix}"
                try:
                    make(source, pending)
                    os.replace(pending, target)
                finally:
                    pending.unlink(missing_ok=True)
            except Exception as error:  # noqa: BLE001 - any failure serves the source
                self._refused.add(key)
                logger.warning("serving %s whole: %s could not be made (%s)", source.name, recipe, error)
                return source, None
        return target, media_type

    def _key(self, source: Path, recipe: str) -> str:
        """What identifies a derivative: the file, not the spelling of its path.

        Resolved first, because the same still is reached by two names. Warming
        walks the document from ``--media-dir`` as it was typed, usually a
        relative path; the routes resolve every reference to an absolute one
        before serving it. Keyed on the spelling, those are two keys for one
        file, and the warm pass would fill a cache the routes never read —
        leaving the encode to be paid by the first participant through §4,
        which is the one thing warming exists to prevent.
        """
        resolved = source.resolve()
        stat = resolved.stat()
        material = f"{recipe}\0{resolved}\0{stat.st_size}\0{stat.st_mtime_ns}"
        return hashlib.sha256(material.encode()).hexdigest()[:32]

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())


def _encode_still(source: Path, target: Path) -> None:
    """The frame at its own size, WebP rather than PNG.

    Pillow is already a dependency of §4's matching, so this adds nothing to
    the environment the instrument needs.
    """
    from PIL import Image

    with Image.open(source) as opened:
        opened.load()
        # WebP has no palette or 16-bit mode; a still is a photograph either
        # way, and the masks that matter are read elsewhere from the originals.
        keep = "RGBA" if "A" in opened.getbands() else "RGB"
        image = opened if opened.mode == keep else opened.convert(keep)
        image.save(target, "WEBP", quality=STILL_QUALITY, method=5)


def _extract_sound(source: Path, target: Path) -> None:
    """The clip's audio track, copied out of the container rather than re-encoded.

    ``-c:a copy`` is the point: §5's mix has to be the sound the participant
    just heard in §3, and a re-encode would make it a second, quieter thing
    that a listening task is then asked to compare against a memory of the
    first. The bytes of the AAC stream are unchanged.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-c:a",
            "copy",
            # The moov atom in front, so the browser can start on the first
            # bytes instead of seeking to the end of the file first.
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("ffmpeg wrote nothing")
