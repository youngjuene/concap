"""Stage the study's own footage for `dpo regen`, and write the document for it.

The regeneration instrument's §10 asks for six things per segment. This turns
what the corpus already holds into five of them, from three inputs and nothing
else:

``--videos``   the uncaptioned 10 s pool (``data/corpus/videos``)
``--tidy``     the response table (``data/corpus/tidy_data.csv``)
``--masks``    the Sa2VA run's mask tree

The first two live in the project. They were read across the disk from the
Sa2VA working tree until a staging run depended on a directory nobody had
copied, so the clips and the table a study is actually run on now sit under
``data/corpus/`` and the defaults point there. ``/data/`` is gitignored, so
this costs the repository nothing and gains it a corpus that travels with the
checkout. The mask tree is the exception and stays external: it is 1.7 GB of
per-frame PNGs, it belongs to the Sa2VA run rather than to this study, and
``--masks`` names it. Copy it in too if a machine needs to stage offline.

Four decisions are worth stating, because each one is a place a staging script
could quietly change what the study measures.

*The model is given a wav, not the clip.* §6 conditions the regenerated
captions on the segment's sound, and the audio stack a caption model brings
with it reads wav and little else — an mp4 needs ``torchcodec``, whose
published wheels all link against a symbol this project's torch does not
export. A container the model cannot open is not an error worth showing a
participant; it is four failed slots and a fallback track. So the sound is
extracted once, here, from the clip *after* loudness correction, and the model
hears what the participant hears.

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
        --segments amsterdam_181 bangkok_034 \\
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
# The clips and the response table, in the project. scripts/ sits beside src/,
# so the root is one hop up whatever the working directory is.
CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
# Mono at the rate the caption model's feature extractor wants, so nothing
# downstream resamples a clip nobody meant to resample.
AUDIO_RATE = 16000
# The Korean each c2 clip carried, transcribed by eye from the burnt-in text.
# It exists nowhere as data — see the file's own note — so it is beside this
# script rather than read out of the corpus, and it is the one input here a
# researcher should check against the footage before running on it.
KOREAN = Path(__file__).with_name("c2-korean-captions.json")
# §4 shows a strip of moments rather than one still. Five frames at equal
# intervals across the clip; with 599 frames the middle one lands on 300, the
# five-second frame §10 names, so the strip is centred on it.
FRAME_COUNT = 5
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


def _require(path: Path, what: str, name: str) -> None:
    """Refuse a missing corpus by name, before anything is encoded.

    The two project-local inputs are gitignored, so a fresh checkout has the
    code and not the footage. That is the right trade for 286 MB of mp4, but it
    means the first run on a new machine fails — and it should fail here,
    saying which directory and where it came from, rather than four minutes in
    on a clip whose source could not be opened.
    """
    if path.exists():
        return
    raise StagingError(
        f"{what} is not at {path}. It is not in the repository — /data/ is gitignored — "
        f"so copy {name} into {CORPUS} from the Sa2VA working tree "
        f"({AVMASK / 'data'}), or name another with the flag."
    )


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


def frame_indices(total: int, count: int = FRAME_COUNT) -> list[int]:
    """``count`` frame indices at equal intervals across ``total`` frames.

    Spaced at the midpoints of equal slices rather than from the first frame to
    the last: neither end of a cut is a moment anyone chose, and the middle
    index lands on the middle of the clip — frame 300 of 599, the five-second
    frame §10 names.
    """
    return [round((2 * index + 1) * total / (2 * count)) for index in range(count)]


def frame_count(path: Path) -> int:
    """How many frames the clip has, counted rather than derived from a rate."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip().isdigit():
        raise StagingError(f"{path}: cannot count frames")
    return int(probe.stdout.strip())


def stage_audio(clip: Path, target: Path) -> None:
    """The clip's sound as a mono wav, for §6's model to read.

    Taken from the staged clip rather than the source, so it carries the same
    loudness correction the participant hears; the two viewings are matched on
    loudness and a model listening to the uncorrected mix would be listening to
    a different clip from the one being rated.
    """
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
            str(clip),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_RATE),
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )


def stage_frame(source: Path, index: int, target: Path) -> None:
    """One frame of the strip, selected by index so it is exactly the masks' frame."""
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
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-vsync",
            "0",
            str(target),
        ]
    )


def stage_masks(masks: Path, clip: str, index: int, out: Path) -> tuple[list[dict[str, str]], list[str]]:
    """One PNG per object, taken from one frame of the strip.

    The run stores a mask per object per frame; a frame of the strip wants the
    masks of that frame and no other. Objects move, and a mask from two seconds
    away would put a click on empty road where a person was standing.

    An object with no pixels in this frame is left out of it, and returned as
    the second value. It may well be in the next frame along; here it is not in
    the picture being marked, and a label no point could reach is a label §6
    could never be given.
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
        frame = root / label / f"{index:05d}.png"
        if not frame.is_file():
            raise StagingError(f"{clip}/{label}: no mask at frame {index:05d}")
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
        raise StagingError(f"{clip}: no object is visible in frame {index:05d}")
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


def cue_track(text: Mapping[str, str], span: int, slots: int) -> list[dict[str, Any]]:
    """The prepared track: the study's own caption over the fixed slots.

    ``text`` is keyed by language, and every language the study offers goes
    into the slot, so a participant reading either finds a caption there.

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
            "text": dict(text) if index == 0 else dict.fromkeys(text, ""),
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
    stage_audio(home / "clip.mp4", home / "audio.wav")
    span = duration_ms(home / "clip.mp4")
    total = frame_count(home / "clip.mp4")
    frames: list[dict[str, Any]] = []
    missing: list[str] = []
    for index in frame_indices(total):
        still = home / "frames" / f"{index:05d}.png"
        stage_frame(home / "clip.mp4", index, still)
        objects, empty = stage_masks(masks, clip, index, home / "masks" / f"{index:05d}")
        missing.extend(f"{index:05d}:{label}" for label in empty)
        frames.append(
            {
                # The frame's own position in the clip, in the units the
                # document speaks: the page shows it, and the log needs to say
                # which moment a mark was placed on.
                "at_ms": round(index * span / total),
                "still": f"{clip}/frames/{still.name}",
                "objects": [
                    {**entry, "mask": f"{clip}/masks/{index:05d}/{entry['mask']}"} for entry in objects
                ],
            }
        )
    return {
        "clip_id": clip,
        "video": f"{clip}/clip.mp4",
        "audio": f"{clip}/audio.wav",
        "duration_ms": span,
        "frames": frames,
        "not_in_the_frame": missing,
    }


def build_segment(
    name: str,
    staged: Mapping[str, Any],
    row: Mapping[str, str],
    sources: Sequence[Mapping[str, Any]],
    slots: int,
    fallback: Mapping[str, str],
    captions: Mapping[str, str],
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
        "audio": staged["audio"],
        "duration_ms": span,
        "frames": list(staged["frames"]),
        "stems": stems,
        "prepared_track": cue_track(captions, span, slots),
        "fallback_track": cue_track(fallback, span, slots),
    }


# ---- the command ------------------------------------------------------------


def fits_slot(text: Mapping[str, str], calibration: Calibration) -> bool:
    """Whether this clip's prepared caption fits the slot in every language.

    The caption is the stimulus and shortening it is the researcher's call, so
    a clip that overruns is not trimmed here — it is left out of the pool and
    named on the way past, so a sample run can be built from the clips that are
    ready without deciding anything about the ones that are not.
    """
    return all(
        len(said) <= calibration.slot_max_chars and len(said.splitlines()) <= calibration.slot_max_lines
        for said in text.values()
    )


def _fallback(entries: Sequence[str], languages: Sequence[str]) -> dict[str, str]:
    """§6's default track, one text per language the study offers.

    One string across two languages would show a Korean session an English
    fallback under its own language tag, which is worse than showing nothing:
    the row would say ``ko`` and read as English.
    """
    if not entries:
        return dict.fromkeys(languages, "")
    if len(entries) == 1 and "=" not in entries[0]:
        if len(languages) > 1:
            raise StagingError(
                f"--fallback-text needs one text per language ({list(languages)}), "
                'as --fallback-text en="..." ko="..."'
            )
        return {languages[0]: entries[0]}
    written: dict[str, str] = {}
    for entry in entries:
        tag, _, text = entry.partition("=")
        if tag not in languages:
            raise StagingError(f"--fallback-text names {tag!r}; the study offers {list(languages)}")
        written[tag] = text
    missing = [tag for tag in languages if tag not in written]
    if missing:
        raise StagingError(f"--fallback-text has nothing for {missing}")
    return {tag: written[tag] for tag in languages}


def _captions(
    clip: str, row: Mapping[str, str], korean: Mapping[str, str], languages: Sequence[str]
) -> dict[str, str]:
    """The prepared caption in every language the study offers.

    English is the response table's ``orig_text``. Korean is the transcription
    of what the clip actually showed, and the two are not always the same
    claim about the sound — three of the twelve describe different events
    entirely. Both are staged as they are; reconciling them is the study's
    decision and not a staging step.
    """
    available = {"en": row["caption"], "ko": korean.get(clip, "")}
    missing = [tag for tag in languages if not available.get(tag)]
    if missing:
        raise StagingError(f"{clip}: no prepared caption in {missing}")
    return {tag: available[tag] for tag in languages}


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
    print(
        f"\nStaged {len(staged)} clips under {out}: the clip, {FRAME_COUNT} frames at equal "
        "intervals, and one mask per object per frame."
    )
    print(f"\n{'clip':<16}{'frames':>7}{'objects':>10}   objects absent from at least one frame")
    for clip in sorted(staged):
        entry = staged[clip]
        counts = [len(frame["objects"]) for frame in entry["frames"]]
        gone = ", ".join(sorted({item.split(":", 1)[1] for item in entry["not_in_the_frame"]})) or "—"
        span = f"{min(counts)}-{max(counts)}" if min(counts) != max(counts) else str(counts[0])
        print(f"{clip:<16}{len(counts):>7}{span:>10}   {gone}")


def _report(document: Mapping[str, Any], out: Path) -> list[str]:
    """Every staged file the document names that is not on disk yet."""
    missing: list[str] = []
    for name in SEGMENTS:
        segment = document["segments"][name]
        references: Iterable[str] = [
            segment["video"],
            *[frame["still"] for frame in segment["frames"]],
            *[entry["mask"] for frame in segment["frames"] for entry in frame["objects"]],
            *[stem["audio"] for stem in segment["stems"]],
        ]
        missing.extend(str(reference) for reference in references if not (out / reference).is_file())
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # The clips and the table come from the project; the mask tree does not.
    parser.add_argument("--videos", type=Path, default=CORPUS / "videos")
    parser.add_argument("--tidy", type=Path, default=CORPUS / "tidy_data.csv")
    parser.add_argument("--masks", type=Path, default=AVMASK / "runs" / "60fps-windowed" / "masks")
    parser.add_argument("--palette", type=Path, default=AVMASK / "data" / "palette.json")
    parser.add_argument("--ontology", type=Path, default=AVMASK / "data" / "ontology.json")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(STIMULUS_CONDITIONS),
        help="the condition directories to draw candidates from (default: the captioned audio one)",
    )
    parser.add_argument(
        "--fits-slot",
        action="store_true",
        help="keep only the clips whose prepared caption fits the slot in every language offered",
    )
    parser.add_argument("--pool", action="store_true", help="list the candidate clips and stop")
    parser.add_argument("--segments", nargs=2, metavar=("A", "B"), help="the two clip ids to stage")
    parser.add_argument("--out", type=Path, help="media directory to stage into")
    parser.add_argument("--document", type=Path, help="session document to write")
    parser.add_argument("--session-id", default="street-regen")
    parser.add_argument("--study-id", default="street2026")
    parser.add_argument("--corpus-id", default="wtour-24")
    parser.add_argument("--cue-slots", type=int, default=1, help="§9.4's slot count; 1 matches the corpus")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "ko"],
        help="the languages the prepared track carries; the first is the one a session opens in",
    )
    parser.add_argument(
        "--fallback-text",
        nargs="+",
        default=[],
        help="what §6 shows when generation fails, as <lang>=<text> per language offered "
        "(a bare string is accepted only by a single-language study); left out, the slots stay "
        "empty and `dpo regen validate` refuses the document until they are written",
    )
    arguments = parser.parse_args(argv)

    try:
        _require(arguments.tidy, "the response table", "tidy_data.csv")
        _require(arguments.videos, "the uncaptioned clip pool", "videos/")
        clips = read_clips(arguments.tidy)
        languages = tuple(arguments.languages)
        korean = json.loads(KOREAN.read_text(encoding="utf-8"))["captions"] if "ko" in languages else {}
        palette = json.loads(arguments.palette.read_text(encoding="utf-8"))["audio"]
        ids, descendants = read_ontology(arguments.ontology)
        candidates = stimulus_clips(clips, arguments.conditions)
        if arguments.fits_slot:
            calibration = Calibration(cue_slots=arguments.cue_slots, languages=languages)
            dropped = [
                clip
                for clip in candidates
                if not fits_slot(_captions(clip, clips[clip], korean, languages), calibration)
            ]
            candidates = [clip for clip in candidates if clip not in dropped]
            if dropped:
                print(
                    f"--fits-slot leaves out {len(dropped)}: {', '.join(dropped)}\n"
                    "  Their prepared caption runs past the slot in one language or both;\n"
                    "  shortening it is the study's decision, so they are left out rather than cut.",
                    file=sys.stderr,
                )
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
            calibration=Calibration(cue_slots=arguments.cue_slots, languages=tuple(arguments.languages)),
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
                    _fallback(arguments.fallback_text, languages),
                    _captions(clip, clips[clip], korean, languages),
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
    caps = configuration.calibration
    long_captions = [
        (f"{name} = {clip}", tag, len(text), len(text.splitlines()))
        for name, clip in zip(SEGMENTS, chosen, strict=True)
        for tag, text in _captions(clip, clips[clip], korean, languages).items()
        if len(text) > caps.slot_max_chars or len(text.splitlines()) > caps.slot_max_lines
    ]
    for name, clip in zip(SEGMENTS, chosen, strict=True):
        segment = document["segments"][name]
        stems = segment["stems"]
        counts = [len(frame["objects"]) for frame in segment["frames"]]
        print(
            f"staged {name} = {clip}: {len(counts)} frames carrying {min(counts)}-{max(counts)} "
            f"objects, {len(stems)} sources ({', '.join(stem['label'] for stem in stems)})"
        )
    print(f"\ndocument: {arguments.document}")
    print(f"media:    {out}")
    if long_captions:
        print(
            f"\nThe prepared caption does not fit the slot ({caps.slot_max_chars} chars, "
            f"{caps.slot_max_lines} lines) on:"
        )
        for where, tag, length, lines in long_captions:
            print(f"  {where} [{tag}]: {length} characters, {lines} lines")
        print("`dpo regen validate` will refuse the document until the text fits or the budget moves.")
    if missing:
        print(f"\n{len(missing)} files the document names are not staged:")
        for reference in missing:
            print(f"  {reference}")
        print("Each also needs its envelope in `waveform` and its level in `gain`; both are")
        print("placeholders until the audio exists, and neither can be measured without it.")
    if not any(_fallback(arguments.fallback_text, languages).values()):
        print("\nStill to author, in the document:")
        print("  segments.*.fallback_track[*].text — what a participant reads when §6 fails.")
        print("  It has to be defensible on its own, so it is left empty and `dpo regen validate`")
        print("  refuses the document until it is written. That refusal is the tool working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
