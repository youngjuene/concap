"""Stage a condition directory of clips into track media plus corpus clip rows.

The collected stimulus set is organised by presentation condition (video with
audio, muted video, and the two caption-burned-in variants), while the pipeline
wants one media file per clip per track and a JSONL of clip rows to ingest.
This bridges the two without touching the originals.

    visual track <- an mp4, symlinked as-is (or restripped with --mute)
    audio  track <- 16 kHz mono wav demuxed from that mp4

``media_hash`` identifies the pristine source clip, so every condition render
of the same footage shares one media identity; the staged rendering is recorded
as its derivative. Run it once per (condition directory, track).

    uv run python scripts/stage_media.py --condition-dir data/video/c3_video \
        --source-dir /path/to/source --track visual \
        --out data/live/visual --rows data/live/visual_clips.jsonl

Conditions that burn a caption into the frame cannot be shown to an annotator
who is judging captions, so ``--media-source source`` stages the pristine
footage the condition was rendered from instead of the render itself. The
condition directory still selects *which* clips are staged. Pair it with
``--mute`` on the visual track to drop the audio stream, so modality isolation
holds in the file and not only in the player.

``--track both`` stages a muted mp4 AND a wav per clip into one directory, and
emits one clip row naming both as derivatives. The training matrix trains every
track from a single registry, so it needs one corpus in which each clip carries
both modalities; no single condition directory can supply that, because c1
(video+audio) and c3 (muted video) sample disjoint footage. Pass
``--condition-dir`` once per condition to collect the full strata-labelled clip
set, and it implies ``--media-source source``: only the pristine clip has both
a caption-free frame and usable audio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpo.core.identity import sha256_file  # noqa: E402

# The audio tower's working rate; the processor resamples anyway, but staging
# at the target rate keeps the files small and the decode trivial.
AUDIO_RATE = 16000


def _duration_ms(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return round(float(completed.stdout.strip()) * 1000)


def _source_video_id(clip_id: str) -> str:
    """``hh_mfcc_amsterdam_006`` -> ``amsterdam_006``: the footage it came from.

    The leading fields are sampling strata (cluster type, feature space), not
    identity; grouping by them would leak the same footage across splits.
    """
    parts = clip_id.split("_")
    if len(parts) < 2:
        raise SystemExit(f"clip id {clip_id!r} has no city_index suffix to group on")
    return "_".join(parts[-2:])


def _stage_visual(source: Path, destination: Path, *, mute: bool) -> None:
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    if not mute:
        destination.symlink_to(source.resolve())
        return
    # Stream-copied, so the frames are bit-identical to the input; only the
    # audio stream is dropped.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-an", "-c:v", "copy", str(destination)],
        check=True,
    )


def _stage_audio(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_RATE),
            str(destination),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition-dir",
        required=True,
        action="append",
        type=Path,
        help="condition directory selecting which clips to stage; repeatable",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="pristine source clips, named {source_video_id}.mp4",
    )
    parser.add_argument("--track", required=True, choices=("visual", "audio", "both"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument(
        "--audio-presentation",
        choices=("audio_only", "unmuted_video"),
        help="audio-track presentation for these clips; omit for the audio_only default",
    )
    parser.add_argument(
        "--media-source",
        choices=("condition", "source"),
        default="condition",
        help="stage the condition render (default) or the pristine source it was rendered from",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="drop the audio stream from staged visual media (frames are stream-copied)",
    )
    arguments = parser.parse_args()

    both = arguments.track == "both"
    if both and arguments.media_source != "source":
        # Only the pristine clip has a caption-free frame AND usable audio.
        raise SystemExit("--track both stages from the source clip; pass --media-source source")
    clips = sorted(clip for directory in arguments.condition_dir for clip in directory.glob("*.mp4"))
    if not clips:
        listed = ", ".join(str(directory) for directory in arguments.condition_dir)
        raise SystemExit(f"no mp4 clips under {listed!r}")
    seen = {clip.stem for clip in clips}
    if len(seen) != len(clips):
        raise SystemExit("the same clip id appears in two condition directories; identities must be unique")
    arguments.out.mkdir(parents=True, exist_ok=True)
    arguments.rows.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for clip in clips:
        clip_id = clip.stem
        source_video_id = _source_video_id(clip_id)
        source = arguments.source_dir / f"{source_video_id}.mp4"
        if not source.is_file():
            raise SystemExit(f"clip {clip_id!r} has no source clip at {str(source)!r}")
        origin = source if arguments.media_source == "source" else clip
        derivatives = []
        if arguments.track in {"visual", "both"}:
            video = arguments.out / f"{clip_id}.mp4"
            _stage_visual(origin, video, mute=arguments.mute or both)
            derivatives.append(sha256_file(video))
            timed = video
        if arguments.track in {"audio", "both"}:
            audio = arguments.out / f"{clip_id}.wav"
            _stage_audio(origin, audio)
            derivatives.append(sha256_file(audio))
            if arguments.track == "audio":
                timed = audio
                if arguments.audio_presentation == "unmuted_video":
                    # The model reads the wav, but the UI asks this presentation
                    # for a video file; stage both or the session 404s on the clip.
                    companion = arguments.out / f"{clip_id}.mp4"
                    _stage_visual(origin, companion, mute=False)
                    derivatives.append(sha256_file(companion))
        row: dict[str, object] = {
            "clip_id": clip_id,
            "source_video_id": source_video_id,
            "media_hash": sha256_file(source),
            "start_ms": 0,
            "end_ms": _duration_ms(timed),
            "derivative_hashes": derivatives,
        }
        if arguments.track in {"audio", "both"} and arguments.audio_presentation:
            row["audio_presentation"] = arguments.audio_presentation
        rows.append(row)

    with arguments.rows.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"staged {len(rows)} {arguments.track} clips into {arguments.out} -> {arguments.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
