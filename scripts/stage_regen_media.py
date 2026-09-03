"""Stage the study's own footage for `dpo regen`, and write the document for it.

The regeneration instrument's §10 asks for six things per segment. This turns
what the corpus already holds into five of them, from three inputs and nothing
else:

``--videos``   the uncaptioned 10 s pool (``avmask/data/videos_10s``)
``--masks``    the Sa2VA run's mask tree (``runs/60fps-windowed/masks``)
``--tidy``     the response table (``avmask/data/tidy_data.csv``)

Four decisions are worth stating, because each one is a place a staging script
could quietly change what the study measures.

*The clip is the uncaptioned source, not a condition deliverable.* The
``c2_video_audio_caption`` files carry a burnt-in Korean caption. The
instrument draws its own caption band, so a c2 clip would show two captions at
once, one of them the wrong provenance. ``videos_10s`` is the only pool that is
uncaptioned for all 24 and it is the footage the masks were computed on.

*Loudness is normalised here.* Measured across the corpus the clips run
-34.0 to -14.9 LUFS. §10 asks for the two segments to be matched on loudness
and they are not, so this normalises to one target (EBU R128) as it copies. A
19 dB difference between the two viewings would sit inside every ART and PRSS
answer the study collects.

*The still and the masks come from one frame index, not one timestamp.* §4
matches a participant's points against masks cut from a particular frame. Frame
``00300`` at 60 fps is the five-second frame §10 names; the still is that frame
and the mask for each object is that object's PNG at that index. Asking for a
timestamp twice would be right until the day a clip's timebase differs.

*Sound sources are de-confused against the AudioSet ontology.* The response
table's labels are what annotators chose, and they nest: ``Speech`` beside
``Male speech, man speaking``, ``Vehicle`` beside ``Bus``. A participant asked
to pick between a label and its own parent is being asked about the vocabulary
rather than about what they heard, so where one label is an ancestor of
another in the same clip the ancestor is dropped and the specific one kept.
Each surviving source carries its top-level family and the palette's colour for
it, so the lanes are coloured by family rather than arbitrarily.

What this cannot produce is the audio itself. §5 plays a separated stem per
source and draws its envelope; no separation exists for this corpus, and the
mask run measures "neither loudness nor onset" by its own account. The stems
are declared with everything except the audio, and ``--report`` says exactly
which files are missing.

    uv run python scripts/stage_regen_media.py --pool
    uv run python scripts/stage_regen_media.py \\
        --segments amsterdam_006 amsterdam_012 \\
        --out data/live/regen-media --document data/live/regen.json

Idempotent: existing staged files are left alone. Needs ffmpeg with libx264.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dpo.regen.config import Calibration, Configuration
from dpo.regen.document import REGEN_SCHEMA, slug

AVMASK = Path("/mnt/hdd/research/2026/Sa2VA/avmask")
STILL_FRAME = 300  # five seconds at 60 fps — the frame §10 names, and §4 matches against
LOUDNESS_TARGET = -23.0  # LUFS, EBU R128
# The same threshold dpo.regen.points reads a mask at, so an object this keeps
# is an object a point can hit.
MASK_INSIDE = 127
SEGMENTS = ("A", "B")
# The condition that showed video, sound and a caption together — the one whose
# stimulus the regeneration instrument's §2 reproduces. c3 and c4 carry no audio
# stream at all, and c1 carried sound without a caption, so neither is a clip a
# participant has ever seen captioned. Widen it with --conditions if a study
# decides otherwise; the default is the twelve that match §2 as specified.
STIMULUS_CONDITIONS = ("c2_video_audio_caption",)


class StagingError(RuntimeError):
    """Something the corpus does not hold, named where a researcher can act on it."""


# ---- the sources ------------------------------------------------------------


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise StagingError(f"{' '.join(command[:4])}… failed:\n{result.stderr.strip()[:800]}")


def read_clips(tidy: Path) -> dict[str, dict[str, str]]:
    """One row per clip from the response table: its labels, families and caption.

    The table is one row per participant per clip and the three fields taken
    here are properties of the clip, identical across its rows. That is checked
    rather than assumed: a table where they differ is a table this cannot read.
    """
    rows: dict[str, dict[str, str]] = {}
    with tidy.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clip = row["file_index"]
            entry = {
                "condition": row["file_path"].split("/")[0],
                "labels": row["final_labels"],
                "parents": row["top_level_parent_name"],
                "caption": row["orig_text"].strip(),
            }
            if clip in rows and rows[clip] != entry:
                raise StagingError(f"{clip}: the response table disagrees with itself about this clip")
            rows[clip] = entry
    return rows


def stimulus_clips(
    clips: Mapping[str, Mapping[str, str]], conditions: Sequence[str] = STIMULUS_CONDITIONS
) -> list[str]:
    """The clips shown under the named conditions, in id order."""
    return sorted(clip for clip, row in clips.items() if row["condition"] in conditions)


def read_ontology(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Label to AudioSet id, and each id's full set of descendants."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    children = {entry["id"]: list(entry["child_ids"]) for entry in entries}
    ids = {entry["name"]: entry["id"] for entry in entries}

    def below(node: str, seen: set[str]) -> set[str]:
        for child in children.get(node, []):
            if child not in seen:
                seen.add(child)
                below(child, seen)
        return seen

    return ids, {node: below(node, set()) for node in children}


def sources_of(
    clip: str,
    row: Mapping[str, str],
    palette: Mapping[str, Any],
    ids: Mapping[str, str],
    descendants: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    """§5's lanes for one clip: distinct sources, the specific ones kept.

    An annotator may name a label and one of its own descendants for the same
    clip. Both are true, and showing both asks the participant to tell
    ``Vehicle`` from ``Bus`` — a question about AudioSet, not about the street.
    The descendant is the informative claim, so the ancestor goes.
    """
    labels: list[str] = []
    for label in row["labels"].split("|"):
        if label and label not in labels:
            labels.append(label)
    unknown = [label for label in labels if label not in ids]
    if unknown:
        raise StagingError(f"{clip}: labels absent from the ontology: {unknown}")
    dropped = {
        ancestor
        for ancestor in labels
        for other in labels
        if other != ancestor and ids[other] in descendants[ids[ancestor]]
    }
    kept = [label for label in labels if label not in dropped]
    sources = []
    for label in kept:
        entry = palette.get(label)
        if entry is None:
            raise StagingError(f"{clip}: {label!r} has no colour in the palette")
        sources.append(
            {
                "id": slug(label),
                "label": label,
                "parent": entry["parent"],
                "audio": None,  # filled in by the caller, which knows the segment
                "colour": "#{:02X}{:02X}{:02X}".format(*entry["rgb"]),
            }
        )
    return sources


# ---- staging ----------------------------------------------------------------


def stage_clip(source: Path, target: Path) -> None:
    """Copy the clip to one loudness, leaving the picture untouched.

    Two passes: ``loudnorm`` measures, then corrects. One pass would gate on a
    running estimate and land somewhere near the target rather than on it, and
    "near" is the thing §10 is trying to remove.
    """
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    measured = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-af",
            f"loudnorm=I={LOUDNESS_TARGET}:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if measured.returncode != 0:
        raise StagingError(f"{source}: loudness measurement failed:\n{measured.stderr.strip()[:600]}")
    report = json.loads(measured.stderr[measured.stderr.rindex("{") : measured.stderr.rindex("}") + 1])
    corrected = (
        f"loudnorm=I={LOUDNESS_TARGET}:TP=-1.5:LRA=11"
        f":measured_I={report['input_i']}:measured_TP={report['input_tp']}"
        f":measured_LRA={report['input_lra']}:measured_thresh={report['input_thresh']}"
        f":offset={report['target_offset']}:linear=true"
    )
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            corrected,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(target),
        ]
    )


def stage_still(source: Path, target: Path) -> None:
    """The five-second frame, selected by index so it is the masks' frame."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select=eq(n\\,{STILL_FRAME})",
            "-frames:v",
            "1",
            "-vsync",
            "0",
            str(target),
        ]
    )


def stage_masks(masks: Path, clip: str, out: Path) -> tuple[list[dict[str, str]], list[str]]:
    """One PNG per object, taken from the frame the still is.

    The run stores a mask per object per frame; the document wants the object's
    mask, once. Which frame that is is not a detail — it is the frame the
    participant is looking at.

    An object whose mask has no pixels in that frame is left out, and returned
    as the second value. It is in the clip somewhere but not in the picture the
    participant marks, so declaring it would put a label in the document that
    no point could ever reach and that §6 could never be given.
    """
    import numpy as np
    from PIL import Image

    root = masks / "visual" / clip
    if not root.is_dir():
        raise StagingError(f"{clip}: no visual masks under {root}")
    out.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, str]] = []
    empty: list[str] = []
    for label in sorted(path.name for path in root.iterdir() if path.is_dir()):
        frame = root / label / f"{STILL_FRAME:05d}.png"
        if not frame.is_file():
            raise StagingError(f"{clip}/{label}: no mask at frame {STILL_FRAME:05d}")
        with Image.open(frame) as handle:
            if not bool((np.asarray(handle.convert("L")) > MASK_INSIDE).any()):
                empty.append(label)
                continue
        identifier = slug(label)
        target = out / f"{identifier}.png"
        if not target.exists():
            shutil.copyfile(frame, target)
        objects.append({"id": identifier, "label": label, "mask": target.name})
    if not objects:
        raise StagingError(f"{clip}: no object is visible in the five-second frame")
    return objects, empty


def duration_ms(path: Path) -> int:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise StagingError(f"{path}: cannot be probed")
    return int(round(float(probe.stdout.strip()) * 1000))


def cue_track(text: str, span: int, slots: int) -> list[dict[str, Any]]:
    """The prepared track: the study's own caption over the fixed slots.

    At one slot this is the stimulus the corpus already carries, whole. At more
    than one the text goes in the first slot and the rest are left empty, which
    the document validator refuses — deliberately, because splitting a written
    caption across slots is an authoring decision and not a staging one.
    """
    step = span // slots
    return [
        {
            "index": index,
            "start_ms": index * step,
            "end_ms": (index + 1) * step if index < slots - 1 else span,
            "text": text if index == 0 else "",
        }
        for index in range(slots)
    ]


def stage_one(clip: str, videos: Path, masks: Path, out: Path) -> dict[str, Any]:
    """Everything §10 prepares for one clip that this can produce, under ``<out>/<clip>/``.

    Filed by clip id rather than by segment letter, so one staged pool serves
    every pairing: which clip is A and which is B is a per-document decision
    (and §1 flips it per participant anyway), and re-pairing should not mean
    re-encoding.
    """
    source = videos / f"{clip}.mp4"
    if not source.is_file():
        raise StagingError(f"{clip}: no uncaptioned source at {source}")
    home = out / clip
    stage_clip(source, home / "clip.mp4")
    stage_still(home / "clip.mp4", home / "still.png")
    objects, empty = stage_masks(masks, clip, home / "masks")
    return {
        "clip_id": clip,
        "video": f"{clip}/clip.mp4",
        "still": f"{clip}/still.png",
        "duration_ms": duration_ms(home / "clip.mp4"),
        "objects": [{**entry, "mask": f"{clip}/masks/{entry['mask']}"} for entry in objects],
        "not_in_the_frame": empty,
    }


def build_segment(
    name: str,
    staged: Mapping[str, Any],
    row: Mapping[str, str],
    sources: Sequence[Mapping[str, Any]],
    slots: int,
    fallback: str,
) -> dict[str, Any]:
    clip = staged["clip_id"]
    span = int(staged["duration_ms"])
    stems = [
        {
            **{key: value for key, value in stem.items() if key != "audio"},
            "audio": f"{clip}/stems/{stem['id']}.wav",
            # §5 needs the envelope drawn from the stem and the level measured
            # against the original mix. Neither can be computed without the
            # stem, so both are placeholders the validator accepts and the
            # closing report names. A flat envelope is the honest placeholder:
            # it claims nothing about when the source is loud.
            "gain": 1.0,
            "waveform": [0.0] * 64,
        }
        for stem in sources
    ]
    return {
        "segment": name,
        "clip_id": clip,
        "video": staged["video"],
        "still": staged["still"],
        "duration_ms": span,
        "objects": list(staged["objects"]),
        "stems": stems,
        "prepared_track": cue_track(row["caption"], span, slots),
        "fallback_track": cue_track(fallback, span, slots),
    }


# ---- the command ------------------------------------------------------------


def _pool(
    clips: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, list[dict[str, Any]]],
    budget: int,
) -> None:
    print(f"{len(sources)} clips were shown with video, sound and a caption together.")
    print(f"`chars` is the prepared caption against the {budget}-character slot budget.\n")
    print(f"{'clip':<16}{'sources':<8}{'chars':<7}{'families':<34}caption")
    over = []
    for clip in sorted(sources):
        families = ", ".join(sorted({stem["parent"] for stem in sources[clip]}))
        caption = clips[clip]["caption"]
        mark = f"{len(caption)}" + ("!" if len(caption) > budget else " ")
        if len(caption) > budget:
            over.append(clip)
        print(f"{clip:<16}{len(sources[clip]):<8}{mark:<7}{families[:32]:<34}{caption[:52]}")
    if over:
        print(
            f"\n{len(over)} captions run past the {budget}-character slot and would be refused by "
            "`dpo regen validate`:"
        )
        print(f"  {', '.join(over)}")
        print(
            "  The caption is the study's stimulus, so this shortens it or raises the budget — "
            "a decision, not a staging step."
        )


def _staged_report(staged: Mapping[str, Mapping[str, Any]], out: Path) -> None:
    """What the pool now holds, and where the segmentation left gaps."""
    print(f"\nStaged {len(staged)} clips under {out}: clip, five-second still, one mask per object.")
    print(f"\n{'clip':<16}{'objects':>8}   objects the five-second frame does not show")
    for clip in sorted(staged):
        entry = staged[clip]
        gone = ", ".join(entry["not_in_the_frame"]) or "—"
        print(f"{clip:<16}{len(entry['objects']):>8}   {gone}")


def _report(document: Mapping[str, Any], out: Path) -> list[str]:
    """Every staged file the document names that is not on disk yet."""
    missing: list[str] = []
    for name in SEGMENTS:
        segment = document["segments"][name]
        references: Iterable[str] = [
            segment["video"],
            segment["still"],
            *[entry["mask"] for entry in segment["objects"]],
            *[stem["audio"] for stem in segment["stems"]],
        ]
        missing.extend(str(reference) for reference in references if not (out / reference).is_file())
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--videos", type=Path, default=AVMASK / "data" / "videos_10s")
    parser.add_argument("--masks", type=Path, default=AVMASK / "runs" / "60fps-windowed" / "masks")
    parser.add_argument("--tidy", type=Path, default=AVMASK / "data" / "tidy_data.csv")
    parser.add_argument("--palette", type=Path, default=AVMASK / "data" / "palette.json")
    parser.add_argument("--ontology", type=Path, default=AVMASK / "data" / "ontology.json")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(STIMULUS_CONDITIONS),
        help="the condition directories to draw candidates from (default: the captioned audio one)",
    )
    parser.add_argument("--pool", action="store_true", help="list the candidate clips and stop")
    parser.add_argument("--segments", nargs=2, metavar=("A", "B"), help="the two clip ids to stage")
    parser.add_argument("--out", type=Path, help="media directory to stage into")
    parser.add_argument("--document", type=Path, help="session document to write")
    parser.add_argument("--session-id", default="street-regen")
    parser.add_argument("--study-id", default="street2026")
    parser.add_argument("--corpus-id", default="wtour-24")
    parser.add_argument("--cue-slots", type=int, default=1, help="§9.4's slot count; 1 matches the corpus")
    parser.add_argument("--language", default="en", help="the language the prepared track is written in")
    parser.add_argument(
        "--fallback-text",
        default="",
        help="what §6 shows when generation fails; empty leaves it for the researcher, and "
        "`dpo regen validate` refuses the document until it is written",
    )
    arguments = parser.parse_args(argv)

    try:
        clips = read_clips(arguments.tidy)
        palette = json.loads(arguments.palette.read_text(encoding="utf-8"))["audio"]
        ids, descendants = read_ontology(arguments.ontology)
        candidates = stimulus_clips(clips, arguments.conditions)
        sources = {clip: sources_of(clip, clips[clip], palette, ids, descendants) for clip in candidates}

        budget = Calibration().slot_max_chars
        if arguments.pool:
            _pool(clips, sources, budget)
            return 0
        if not arguments.out:
            raise StagingError("--out names the media directory to stage into")
        out = Path(arguments.out)

        chosen = list(arguments.segments or [])
        unknown = [clip for clip in chosen if clip not in sources]
        if unknown:
            raise StagingError(f"not clips that carried sound: {unknown}; run --pool for the candidates")
        if len(chosen) == 2 and chosen[0] == chosen[1]:
            raise StagingError("the two segments must be different footage")

        # Stage the whole pool unless a pair was named: every clip that carried
        # sound, filed by its own id, so any pairing can be documented later
        # without re-encoding anything.
        wanted = chosen or candidates
        staged = {}
        for index, clip in enumerate(wanted, 1):
            print(f"[{index}/{len(wanted)}] {clip}", flush=True)
            staged[clip] = stage_one(clip, arguments.videos, arguments.masks, out)

        if not chosen:
            _staged_report(staged, out)
            print("\nPass --segments A B --document <file> to write a session document for a pair.")
            return 0
        if not arguments.document:
            raise StagingError("--segments needs --document, the session document to write")

        configuration = Configuration(
            study_id=arguments.study_id,
            corpus_id=arguments.corpus_id,
            calibration=Calibration(cue_slots=arguments.cue_slots, language=arguments.language),
        )
        document = {
            "schema": REGEN_SCHEMA,
            "session_id": arguments.session_id,
            "config": configuration.artifact(),
            "segments": {
                name: build_segment(
                    name,
                    staged[clip],
                    clips[clip],
                    sources[clip],
                    arguments.cue_slots,
                    arguments.fallback_text,
                )
                for name, clip in zip(SEGMENTS, chosen, strict=True)
            },
        }
        arguments.document.parent.mkdir(parents=True, exist_ok=True)
        arguments.document.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except StagingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    missing = _report(document, out)
    long_captions = [
        (name, clip, len(clips[clip]["caption"]))
        for name, clip in zip(SEGMENTS, chosen, strict=True)
        if len(clips[clip]["caption"]) > configuration.calibration.slot_max_chars
    ]
    for name, clip in zip(SEGMENTS, chosen, strict=True):
        stems = document["segments"][name]["stems"]
        print(
            f"staged {name} = {clip}: {len(document['segments'][name]['objects'])} objects, "
            f"{len(stems)} sources ({', '.join(stem['label'] for stem in stems)})"
        )
    print(f"\ndocument: {arguments.document}")
    print(f"media:    {out}")
    if long_captions:
        budget = configuration.calibration.slot_max_chars
        print(f"\nThe prepared caption runs past the {budget}-character slot on:")
        for name, clip, length in long_captions:
            print(f"  {name} = {clip}: {length} characters")
        print("`dpo regen validate` will refuse the document until the text fits or the budget moves.")
    if missing:
        print(f"\n{len(missing)} files the document names are not staged:")
        for reference in missing:
            print(f"  {reference}")
        print("Each also needs its envelope in `waveform` and its level in `gain`; both are")
        print("placeholders until the audio exists, and neither can be measured without it.")
    if not arguments.fallback_text:
        print("\nStill to author, in the document:")
        print("  segments.*.fallback_track[*].text — what a participant reads when §6 fails.")
        print("  It has to be defensible on its own, so it is left empty and `dpo regen validate`")
        print("  refuses the document until it is written. That refusal is the tool working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
