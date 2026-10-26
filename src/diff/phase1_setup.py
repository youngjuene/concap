"""Phase-1 setup material: the short clips, their first captions, options and practice.

Researcher's decisions of 2026-10-18 (plan "Phase 1 — 2026-10-18"): phase 1
collects each participant's responses to the prepared captions, so that phase 2
can be tuned to them. Each of regen's 12 c2 clips is shown twice over regen's
uncaptioned ``videos_10s`` media (``data/live/regen-media/<clip>/``), following
the questionnaire spreadsheet:
- first with its prepared Korean caption shown raw: the text burnt into the
  delivered stimulus, as transcribed in ``scripts/c2-korean-captions.json``,
  with its line breaks and no length or line limit;
- then with a second caption Trida writes from that caption and the
  participant's first-step answers (``diff.phase1_tweak``), never stored here.

The file (``diff.phase1-setup/v1``) is server-side only: it holds the real clip
keys, ``orig_text``, each A1s option's sound family and the R2 answer key
(``mentioned``), none of which may reach the participant's browser. The
page texts are the spreadsheet's, word for word (``diff.workbook_instrument``),
so the setup carries no wording of its own beyond an optional ``consent_text``.

:func:`draft_setup` fills in what the prepared metadata determines:
- media, durations and frames, with each frame's per-object masks;
- the practice clip, a ``c1_video_audio`` clip outside the 12, staged beside
  them at their loudness (:func:`level_audio`; researcher, 2026-10-26: the
  corpus file is about 3 dB quieter, so a volume set during practice would
  play the clips louder);
- the prepared Korean captions, copied verbatim;
- ``orig_text``;
- each clip's AudioSet labels, every one kept, with its family: a reference
  for the options' families, never a source of options.

The options are then written by hand into the draft, marked :data:`DRAFTED`.
They come from ``orig_text`` (researcher, 2026-10-18): every A1s option and
every mentioned R2 option renders a sound ``orig_text`` names and quotes those
words (``orig_phrase``); nothing it names is dropped and no sound is added from
the labels. R2 is asked after the second viewing, about Trida's caption, so
its key is taken from the prompt (researcher, 2026-10-18): the mentioned
options' labels are the sounds the second-caption prompt names, and so count
as mentioned. Each A1s option also names the AudioSet label that stands for its
sound (``audioset_label``), whose Sa2VA audio masks the clip's scene part reads
(``python -m diff phase1 scene``, :mod:`diff.phase1_scene`, 2026-10-23); the
label keys the masks and never words an option. Whether its source is on
screen is measured from those masks, never judged (researcher, 2026-10-24).
Each part is approved by the researcher (:func:`approve`) against a hash of its
content, so any later edit voids the approval.
:func:`setup_problems` refuses a participant setup until every part is approved,
and until every second caption it can show is (:mod:`diff.phase1_approved`,
2026-10-24).
"""

from __future__ import annotations

import copy
import csv
import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from diff.caption_sources import CATEGORIES_KO
from diff.phase1_approved import approval_problems
from diff.phase1_scene import scene_problems
from diff.study_protocol import sha256_of

SETUP_SCHEMA: Final = "diff.phase1-setup/v1"
DRAFTED: Final = "assistant-draft-unreviewed"
PURPOSES: Final = ("participant", "smoke")
# AudioSet's top-level categories, as the tuner's sound families.
FAMILY_OF_ROOT: Final[dict[str, str]] = {
    "Human sounds": "human",
    "Animal": "animal",
    "Music": "music",
    "Natural sounds": "natural",
    "Sounds of things": "things",
    "Source-ambiguous sounds": "other",
    "Channel, environment and background": "other",
}
# A label under two top-level categories takes the first of these, flagged for review.
FAMILY_ORDER: Final = ("human", "animal", "music", "natural", "things", "other")
OPTION_RULE: Final = (
    "each A1s option and each mentioned R2 option renders a sound orig_text names and quotes those "
    "words (orig_phrase); the mentioned R2 labels are the sounds the second-caption prompt names; "
    "AudioSet labels only inform the families and, as each A1s option's audioset_label, key its audio masks"
)
# Regen's document rule for a segment's annotated frames (dpo.regen.document).
FRAMES_MIN: Final = 3
FRAMES_MAX: Final = 9
PRACTICE_AT_MS: Final = 5000
# The loudness scripts/stage_regen_media.py staged the 12 clips at (EBU R128, LUFS).
LOUDNESS: Final = -23.0
# The staged clips' frames are 1280 px wide; the practice still matches them.
FRAME_WIDTH: Final = 1280
LATIN = re.compile(r"[A-Za-z]")
HANGUL = re.compile(r"[가-힣]")

Probe = Callable[[Path], tuple[int, float]]
Extract = Callable[[Path, int, Path], None]
Level = Callable[[Path, Path], None]


class SetupError(ValueError):
    """Material the drafter cannot build from, named where the researcher can act on it."""


def read_ontology(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Label → AudioSet id, and each id's top-level category names."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    children = {entry["id"]: list(entry["child_ids"]) for entry in entries}
    ids = {entry["name"]: entry["id"] for entry in entries}
    names = {entry["id"]: entry["name"] for entry in entries}
    parents: dict[str, set[str]] = {}
    for node, below in children.items():
        for child in below:
            parents.setdefault(child, set()).add(node)

    def roots(node: str) -> set[str]:
        above = parents.get(node)
        if not above:
            return {names[node]}
        return set().union(*(roots(parent) for parent in above))

    return ids, {node: roots(node) for node in children}


def family_of(label: str, ids: Mapping[str, str], roots: Mapping[str, set[str]]) -> tuple[str, list[str]]:
    """The label's sound family and every top-level category it sits under."""
    under = sorted(roots[ids[label]])
    families = {FAMILY_OF_ROOT[name] for name in under if name in FAMILY_OF_ROOT}
    if not families:
        raise SetupError(f"{label!r} is under no known top-level category: {under}")
    return next(family for family in FAMILY_ORDER if family in families), under


def corpus_clips(path: Path, condition: str) -> dict[str, dict[str, Any]]:
    """``orig_text``, labels and video path per clip of one corpus condition (every row must agree)."""
    clips: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["file_path"].startswith(f"{condition}/"):
                continue
            seen = {
                "orig_text": row["orig_text"],
                "labels": row["final_labels"].split("|"),
                "file_path": row["file_path"],
            }
            known = clips.setdefault(row["file_index"], seen)
            if known != seen:
                raise SetupError(f"{row['file_index']}: corpus rows disagree on orig_text or labels")
    return clips


def provided_captions(path: Path) -> tuple[dict[str, str], str]:
    """The prepared Korean caption of each clip, verbatim, and the file's provenance."""
    document = json.loads(path.read_text(encoding="utf-8"))
    captions = document.get("captions")
    if not isinstance(captions, dict) or not all(isinstance(text, str) for text in captions.values()):
        raise SetupError(f"{path}: no captions object of clip → text")
    return captions, str(document.get("provenance", path.name))


def probe_video(path: Path) -> tuple[int, float]:
    """The container's duration in ms and the video stream's frame rate, from ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate:format=duration", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    probed = json.loads(out)
    numerator, denominator = probed["streams"][0]["r_frame_rate"].split("/")
    return round(float(probed["format"]["duration"]) * 1000), int(numerator) / int(denominator)


def level_audio(source: Path, target: Path) -> None:
    """A copy of the video with its sound at the clips' loudness and its picture untouched.

    The two passes of ``scripts/stage_regen_media.py``'s ``stage_clip``, which staged the 12
    clips: ffmpeg's loudnorm measures the file, then corrects it to :data:`LOUDNESS`.
    """
    loudnorm = f"loudnorm=I={LOUDNESS}:TP=-1.5:LRA=11"
    measured = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(source), "-af", f"{loudnorm}:print_format=json"]
        + ["-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(measured.stderr[measured.stderr.rindex("{") : measured.stderr.rindex("}") + 1])
    corrected = (
        f"{loudnorm}:measured_I={report['input_i']}:measured_TP={report['input_tp']}"
        f":measured_LRA={report['input_lra']}:measured_thresh={report['input_thresh']}"
        f":offset={report['target_offset']}:linear=true"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-af", corrected]
        + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(target)],
        check=True,
    )


def extract_frame(video: Path, at_ms: int, out: Path) -> None:
    """One PNG still at ``at_ms``, scaled to the clips' frame width."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{at_ms / 1000:.3f}",
            "-i",
            str(video),
            "-vf",
            f"scale={FRAME_WIDTH}:-2",
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )


def _frames(clip_dir: Path, fps: float, root: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for still in sorted((clip_dir / "frames").glob("*.png")):
        masks = sorted((clip_dir / "masks" / still.stem).glob("*.png"))
        frames.append(
            {
                "index": len(frames),
                "at_ms": round(int(still.stem) * 1000 / fps),
                "still": str(still.relative_to(root)),
                "objects": [
                    {
                        "id": mask.stem,
                        "label": mask.stem.replace("_", " ").capitalize(),
                        "mask": str(mask.relative_to(root)),
                    }
                    for mask in masks
                ],
            }
        )
    return frames


def label_families(
    labels: Iterable[str], ids: Mapping[str, str], roots: Mapping[str, set[str]]
) -> list[dict[str, Any]]:
    """Each distinct AudioSet label with its family, every label kept: the reviewer's reference."""
    distinct = [label for label in dict.fromkeys(labels) if label]
    unknown = [label for label in distinct if label not in ids]
    if unknown:
        raise SetupError(f"labels absent from the ontology: {unknown}")
    entries = []
    for label in distinct:
        family, under = family_of(label, ids, roots)
        entry: dict[str, Any] = {"label": label, "family": family}
        if len(under) > 1:
            entry["family_ambiguous"] = under
        entries.append(entry)
    return entries


def draft_setup(
    *,
    root: Path,
    media_root: Path,
    corpus: Path,
    captions: Path,
    ontology: Path,
    video_root: Path,
    practice_clip: str,
    practice_still: Path,
    instrument: Any,
    probe: Probe = probe_video,
    extract: Extract = extract_frame,
    level: Level = level_audio,
) -> dict[str, Any]:
    """Every part the prepared metadata determines; the options are left to write."""
    ids, roots = read_ontology(ontology)
    shown = corpus_clips(corpus, "c2_video_audio_caption")
    prepared, provenance = provided_captions(captions)
    missing = sorted(set(shown) - set(prepared))
    if missing:
        raise SetupError(f"no prepared caption in {captions} for {missing}")
    practice_rows = corpus_clips(corpus, "c1_video_audio")
    if practice_clip not in practice_rows:
        raise SetupError(f"practice clip {practice_clip!r} is not a c1_video_audio clip in {corpus}")
    corpus_video = video_root / practice_rows[practice_clip]["file_path"]
    if not corpus_video.is_file():
        raise SetupError(f"practice clip video missing: {corpus_video}")
    practice_video = media_root / "practice" / f"{practice_clip}.mp4"
    level(corpus_video, practice_video)
    clips = []
    for key in sorted(shown):
        clip_dir = media_root / key
        video = clip_dir / "clip.mp4"
        if not video.exists():
            raise SetupError(f"{key}: no staged media at {clip_dir}")
        duration_ms, fps = probe(video)
        clips.append(
            {
                "key": key,
                "video": str(video.relative_to(root)),
                "duration_ms": duration_ms,
                "frames": _frames(clip_dir, fps, root),
                "caption": {
                    "text": prepared[key],
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "source": str(captions.relative_to(root)),
                    "provenance": provenance,
                    "orig_text": shown[key]["orig_text"],
                },
                "audioset_labels": label_families(shown[key]["labels"], ids, roots),
                "options": {"A1s": [], "R2": []},
                "review": None,
            }
        )
    extract(practice_video, PRACTICE_AT_MS, practice_still)
    return {
        "schema": SETUP_SCHEMA,
        "instrument_hash": instrument.snapshot()["hash"],
        "consent_text": None,
        "practice": {
            "clip": practice_clip,
            "video": str(practice_video.relative_to(root)),
            "frames": [{"still": str(practice_still.relative_to(root)), "at_ms": PRACTICE_AT_MS}],
            "orig_text": practice_rows[practice_clip]["orig_text"],
            "audioset_labels": label_families(practice_rows[practice_clip]["labels"], ids, roots),
            "options": {"A1s": []},
            "provenance": DRAFTED,
            "review": None,
        },
        "clips": clips,
        "sources": {
            "media_root": str(media_root.relative_to(root)),
            "corpus": str(corpus.relative_to(root)),
            "captions": str(captions.relative_to(root)),
            "ontology": str(ontology),
            "option_rule": OPTION_RULE,
        },
    }


def content_sha256(part: Mapping[str, Any]) -> str:
    """The hash an approval binds to: the part without its review."""
    return sha256_of({key: value for key, value in part.items() if key != "review"})


def approved(part: Mapping[str, Any]) -> bool:
    review = part.get("review")
    return isinstance(review, Mapping) and review.get("content_sha256") == content_sha256(part)


def approve(
    setup: Mapping[str, Any],
    *,
    by: str,
    at: str,
    clips: Sequence[str] = (),
    practice: bool = False,
) -> dict[str, Any]:
    """A copy with the named parts approved by ``by`` against their current content."""
    if not by.strip():
        raise SetupError("an approval needs the reviewer's name")
    updated = copy.deepcopy(dict(setup))
    by_key = {clip["key"]: clip for clip in updated["clips"]}
    unknown = sorted(set(clips) - set(by_key))
    if unknown:
        raise SetupError(f"no such clips: {unknown}")
    parts = [by_key[key] for key in clips]
    if practice:
        parts.append(updated["practice"])
    for part in parts:
        part["review"] = {"by": by, "at": at, "content_sha256": content_sha256(part)}
    return updated


def stripped_options(options: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """What the participant may see of an option list: value and label only."""
    return [{"value": option["value"], "label": option["label"]} for option in options]


def _korean_problems(text: str, where: str) -> list[str]:
    problems = []
    if not HANGUL.search(text):
        problems.append(f"{where}: no Korean text")
    if LATIN.search(text):
        problems.append(f"{where}: contains Latin letters (participant-facing text is Korean only)")
    return problems


def _option_problems(
    page: str, item: str, options: Sequence[Mapping[str, Any]], page_spec: Callable[..., Any], where: str
) -> list[str]:
    problems = []
    try:
        page_spec(page, options={item: stripped_options(options)})
    except (ValueError, KeyError, TypeError) as exc:
        problems.append(f"{where}: {exc}")
    for option in options:
        problems += _korean_problems(str(option.get("label", "")), f"{where} {option.get('value')}")
    return problems


def _phrase_problems(option: Mapping[str, Any], orig_text: str, where: str) -> list[str]:
    """An option must quote the words of ``orig_text`` it renders."""
    phrase = option.get("orig_phrase")
    if not (isinstance(phrase, str) and phrase.strip()):
        return [f"{where}: no orig_phrase (the words of orig_text this option renders)"]
    if phrase.casefold() not in orig_text.casefold():
        return [f"{where}: orig_phrase {phrase!r} is not in orig_text"]
    return []


def _caption_problems(clip: Mapping[str, Any], where: str) -> list[str]:
    """The prepared caption is shown raw: it only has to exist and cover the clip."""
    caption = clip["caption"]
    problems = []
    if not str(caption.get("text", "")).strip():
        problems.append(f"{where}.caption: empty")
    if (caption.get("start_ms"), caption.get("end_ms")) != (0, clip["duration_ms"]):
        problems.append(f"{where}.caption: the cue must cover the clip, 0–{clip['duration_ms']} ms")
    return problems


def _file_problems(root: Path, relative: str, where: str) -> list[str]:
    return [] if (root / relative).is_file() else [f"{where}: missing file {relative}"]


def _clip_problems(clip: Mapping[str, Any], root: Path, page_spec: Callable[..., Any]) -> list[str]:
    where = str(clip.get("key"))
    problems = _file_problems(root, clip["video"], f"{where}.video")
    if not (isinstance(clip.get("duration_ms"), int) and clip["duration_ms"] > 0):
        problems.append(f"{where}: duration_ms must be a positive integer")
        return problems
    frames = clip.get("frames", [])
    if not FRAMES_MIN <= len(frames) <= FRAMES_MAX:
        problems.append(f"{where}: {len(frames)} frames; P2 needs {FRAMES_MIN}–{FRAMES_MAX}")
    for frame in frames:
        at = f"{where}.frame {frame.get('index')}"
        problems += _file_problems(root, frame["still"], at)
        if not 0 <= frame["at_ms"] <= clip["duration_ms"]:
            problems.append(f"{at}: at_ms outside the clip")
        if not frame.get("objects"):
            problems.append(f"{at}: no annotated objects")
        for item in frame.get("objects", []):
            problems += _file_problems(root, item["mask"], f"{at} {item.get('id')}")
    problems += _caption_problems(clip, where)
    orig_text = str(clip["caption"].get("orig_text", ""))
    sounds = clip["options"]["A1s"]
    problems += _option_problems("P3", "A1s", sounds, page_spec, f"{where}.A1s")
    for option in sounds:
        problems += _phrase_problems(option, orig_text, f"{where}.A1s {option.get('value')}")
        if option.get("family") not in CATEGORIES_KO:
            problems.append(
                f"{where}.A1s {option.get('value')}: family must be one of {sorted(CATEGORIES_KO)}"
            )
        labels = {entry["label"] for entry in clip.get("audioset_labels", [])}
        if option.get("audioset_label") not in labels:
            problems.append(
                f"{where}.A1s {option.get('value')}: audioset_label must be one of {sorted(labels)}"
            )
    recall = clip["options"]["R2"]
    problems += _option_problems("P11", "R2", recall, page_spec, f"{where}.R2")
    if any(not isinstance(option.get("mentioned"), bool) for option in recall):
        problems.append(f"{where}.R2: every option needs mentioned true or false (the answer key)")
    for option in recall:
        if option.get("mentioned") is True:
            problems += _phrase_problems(option, orig_text, f"{where}.R2 {option.get('value')}")
    return problems + scene_problems(clip, root)


def _practice_problems(
    practice: Mapping[str, Any], formal: set[str], root: Path, page_spec: Any
) -> list[str]:
    problems = []
    frames = practice.get("frames", [])
    if not frames:
        problems.append("practice: no frame")
    for frame in frames:
        problems += _file_problems(root, frame["still"], "practice")
        if frame["still"] in formal:
            problems.append("practice: the frame must not be one of the clips' frames")
    sounds = practice["options"]["A1s"]
    problems += _option_problems("P3", "A1s", sounds, page_spec, "practice.A1s")
    orig_text = str(practice.get("orig_text", ""))
    for option in sounds:
        problems += _phrase_problems(option, orig_text, f"practice.A1s {option.get('value')}")
    return problems


def setup_problems(setup: Mapping[str, Any], purpose: str, *, root: Path, instrument: Any) -> list[str]:
    """Every reason this setup cannot serve sessions of ``purpose``; empty when it can."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if setup.get("schema") != SETUP_SCHEMA:
        return [f"schema must be {SETUP_SCHEMA}"]
    clips = setup.get("clips", [])
    if not clips:
        return ["no clips"]
    problems = []
    keys = [clip.get("key") for clip in clips]
    if len(set(keys)) != len(keys):
        problems.append("clip keys repeat")
    consent = setup.get("consent_text")
    if consent is not None and not (isinstance(consent, str) and consent.strip()):
        problems.append("consent_text must be null or non-empty text")
    for clip in clips:
        problems += _clip_problems(clip, root, instrument.page_spec)
    formal = {frame["still"] for clip in clips for frame in clip.get("frames", [])}
    problems += _practice_problems(setup["practice"], formal, root, instrument.page_spec)
    sound = not problems  # every clip's scene part is there to list its second captions from
    if purpose == "participant":
        if setup.get("instrument_hash") != instrument.snapshot()["hash"]:
            problems.append("the workbook instrument changed since this setup was drafted")
        if not approved(setup["practice"]):
            problems.append("practice: not approved by the researcher (or edited since approval)")
        for clip in clips:
            if not approved(clip):
                problems.append(f"{clip['key']}: not approved by the researcher (or edited since approval)")
        if sound:
            problems += approval_problems(setup)
    return problems


def read_setup(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SetupError(f"{path}: not a setup object")
    return value
