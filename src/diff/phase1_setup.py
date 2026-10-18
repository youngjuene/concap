"""Phase-1 setup material: the short clips, their captions and options, practice and wording.

Researcher's decisions of 2026-10-18 (plan "Phase 1 — 2026-10-18"): phase 1
only profiles the participant for the live tuner and shows no model output.
Each of regen's 12 c2 clips is shown once over regen's uncaptioned
``videos_10s`` media (``data/live/regen-media/<clip>/``), with a Korean
rendering of the corpus ``orig_text`` as its caption, because the burnt-in
원문 contradicts the audio on four clips.

The file (``diff.phase1-setup/v1``) is server-side only: it holds the real clip
keys, each A1s option's sound family and the R2 answer key (``mentioned``),
none of which may reach the participant's browser.

:func:`draft_setup` fills in what the prepared metadata determines:
- media, durations and frames, with each frame's per-object masks;
- ``orig_text``;
- each clip's AudioSet labels, with ancestors dropped by the rule
  ``scripts/stage_regen_media.py`` uses;
- each label's family;
- the minimal wording edits the researcher asked for: the P0 sentences about a
  second viewing and an interview are deleted, and "1차"/"2차" is dropped from
  the titles.

The Korean captions, option labels and R2 options are then written by hand
into the draft, marked :data:`DRAFTED`. Each part is approved by the
researcher (:func:`approve`) against a hash of its content, so any later edit
voids the approval. :func:`setup_problems` refuses a participant setup until
every part is approved.
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
from diff.study_protocol import MAX_CHARS, MAX_LINES, sha256_of

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
# Regen's document rule for a segment's annotated frames (dpo.regen.document).
FRAMES_MIN: Final = 3
FRAMES_MAX: Final = 9
PRACTICE_AT_MS: Final = 5000
# The staged clips' frames are 1280 px wide; the practice still matches them.
FRAME_WIDTH: Final = 1280
# The workbook pages diff shows; their titles are the only workbook text it may override.
SHOWN_PAGES: Final = ("P0", "P1", "P2", "P3", "P4", "P5", "P10", "P11")
# The two P0 sentences that describe the dropped second viewing and interview.
P0_DELETED: Final = (
    "같은 영상을 두 번 보시는 경우가 있는데, 이는 서로 다른 버전의 자막을 비교하기 위한 것입니다.",
    "모든 영상이 끝난 뒤에는 2~3분 정도 간단한 인터뷰가 있습니다.",
)
LATIN = re.compile(r"[A-Za-z]")
HANGUL = re.compile(r"[가-힣]")

Probe = Callable[[Path], tuple[int, float]]
Extract = Callable[[Path, int, Path], None]


class SetupError(ValueError):
    """Material the drafter cannot build from, named where the researcher can act on it."""


def read_ontology(path: Path) -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    """Label → AudioSet id, each id's descendants, and each id's top-level category names."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    children = {entry["id"]: list(entry["child_ids"]) for entry in entries}
    ids = {entry["name"]: entry["id"] for entry in entries}
    names = {entry["id"]: entry["name"] for entry in entries}
    parents: dict[str, set[str]] = {}
    for node, below in children.items():
        for child in below:
            parents.setdefault(child, set()).add(node)

    def descendants(node: str, seen: set[str]) -> set[str]:
        for child in children.get(node, []):
            if child not in seen:
                seen.add(child)
                descendants(child, seen)
        return seen

    def roots(node: str) -> set[str]:
        above = parents.get(node)
        if not above:
            return {names[node]}
        return set().union(*(roots(parent) for parent in above))

    return (
        ids,
        {node: descendants(node, set()) for node in children},
        {node: roots(node) for node in children},
    )


def specific_labels(
    labels: Iterable[str], ids: Mapping[str, str], descendants: Mapping[str, set[str]]
) -> tuple[list[str], list[str]]:
    """Distinct labels with any ancestor of another label dropped (regen staging's rule)."""
    distinct: list[str] = []
    for label in labels:
        if label and label not in distinct:
            distinct.append(label)
    unknown = [label for label in distinct if label not in ids]
    if unknown:
        raise SetupError(f"labels absent from the ontology: {unknown}")
    dropped = [
        ancestor
        for ancestor in distinct
        if any(other != ancestor and ids[other] in descendants[ids[ancestor]] for other in distinct)
    ]
    return [label for label in distinct if label not in dropped], dropped


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


def sound_options(
    labels: Sequence[str],
    ids: Mapping[str, str],
    descendants: Mapping[str, set[str]],
    roots: Mapping[str, set[str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """A1s options from AudioSet labels: opaque values, family kept, Korean label left to write.

    Ancestors are dropped as regen's staging does, unless that leaves fewer than
    two sounds: a single option is no choice, so they are then kept and marked.
    """
    kept, dropped = specific_labels(labels, ids, descendants)
    restored = []
    if len(kept) < 2 and dropped:
        restored, dropped = dropped, []
        kept = [label for label in dict.fromkeys(labels) if label]
    options = []
    for number, label in enumerate(kept, 1):
        family, under = family_of(label, ids, roots)
        option: dict[str, Any] = {"value": f"s{number}", "label": "", "family": family, "audioset": label}
        if len(under) > 1:
            option["family_ambiguous"] = under
        if label in restored:
            option["kept_ancestor"] = True
        options.append(option)
    return options, dropped


def draft_wording(page_spec: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
    """The researcher's minimal edits: two P0 sentences deleted, "1차"/"2차" dropped from titles."""
    intro = str(page_spec("P0")["instruction"])
    missing = [sentence for sentence in P0_DELETED if sentence not in intro]
    if missing:
        raise SetupError(f"the workbook P0 text no longer contains {missing}")
    edited = intro
    for sentence in P0_DELETED:
        edited = edited.replace(" " + sentence, "").replace(sentence + " ", "").replace(sentence, "")
    titles = {}
    for page in SHOWN_PAGES:
        title = str(page_spec(page)["title"])
        stripped = re.sub(r"^[12]차\s*", "", title)
        if stripped != title:
            titles[page] = {"workbook": title, "shown": stripped}
    return {
        "P0_instruction": {"workbook": intro, "shown": edited, "deleted": list(P0_DELETED)},
        "titles": titles,
        "provenance": DRAFTED,
        "review": None,
    }


def draft_setup(
    *,
    root: Path,
    media_root: Path,
    corpus: Path,
    ontology: Path,
    video_root: Path,
    practice_clip: str,
    practice_still: Path,
    instrument: Any,
    probe: Probe = probe_video,
    extract: Extract = extract_frame,
) -> dict[str, Any]:
    """Every part the prepared metadata determines; Korean text fields are left empty to write."""
    ids, descendants, roots = read_ontology(ontology)
    shown = corpus_clips(corpus, "c2_video_audio_caption")
    practice_rows = corpus_clips(corpus, "c1_video_audio")
    if practice_clip not in practice_rows:
        raise SetupError(f"practice clip {practice_clip!r} is not a c1_video_audio clip in {corpus}")
    practice_video = video_root / practice_rows[practice_clip]["file_path"]
    if not practice_video.is_file():
        raise SetupError(f"practice clip video missing: {practice_video}")
    clips = []
    for key in sorted(shown):
        clip_dir = media_root / key
        video = clip_dir / "clip.mp4"
        if not video.exists():
            raise SetupError(f"{key}: no staged media at {clip_dir}")
        duration_ms, fps = probe(video)
        options, dropped = sound_options(shown[key]["labels"], ids, descendants, roots)
        clips.append(
            {
                "key": key,
                "video": str(video.relative_to(root)),
                "duration_ms": duration_ms,
                "frames": _frames(clip_dir, fps, root),
                "caption": {
                    "text": "",
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "orig_text": shown[key]["orig_text"],
                    "provenance": DRAFTED,
                },
                "options": {"A1s": options, "R2": []},
                "dropped_labels": dropped,
                "review": None,
            }
        )
    extract(practice_video, PRACTICE_AT_MS, practice_still)
    practice_options, _ = sound_options(practice_rows[practice_clip]["labels"], ids, descendants, roots)
    return {
        "schema": SETUP_SCHEMA,
        "instrument_hash": instrument.snapshot()["hash"],
        "wording": draft_wording(instrument.page_spec),
        "consent_text": None,
        "practice": {
            "clip": practice_clip,
            "video": str(practice_video.relative_to(root)),
            "frames": [{"still": str(practice_still.relative_to(root)), "at_ms": PRACTICE_AT_MS}],
            "options": {"A1s": practice_options},
            "provenance": DRAFTED,
            "review": None,
        },
        "clips": clips,
        "sources": {
            "media_root": str(media_root.relative_to(root)),
            "corpus": str(corpus.relative_to(root)),
            "ontology": str(ontology),
            "label_rule": "an AudioSet ancestor of another label of the clip is dropped (stage_regen_media)",
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
    wording: bool = False,
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
    if wording:
        parts.append(updated["wording"])
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


def _caption_problems(clip: Mapping[str, Any], where: str) -> list[str]:
    caption = clip["caption"]
    text = str(caption.get("text", ""))
    problems = _korean_problems(text, f"{where}.caption")
    if len(text) > MAX_CHARS[1]:
        problems.append(f"{where}.caption: {len(text)} characters, over {MAX_CHARS[1]}")
    if len(text.split("\n")) > MAX_LINES[1]:
        problems.append(f"{where}.caption: more than {MAX_LINES[1]} lines")
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
    sounds = clip["options"]["A1s"]
    problems += _option_problems("P3", "A1s", sounds, page_spec, f"{where}.A1s")
    if len(sounds) < 2:
        problems.append(f"{where}.A1s: fewer than two sounds to choose from")
    for option in sounds:
        if option.get("family") not in CATEGORIES_KO:
            problems.append(
                f"{where}.A1s {option.get('value')}: family must be one of {sorted(CATEGORIES_KO)}"
            )
    recall = clip["options"]["R2"]
    problems += _option_problems("P11", "R2", recall, page_spec, f"{where}.R2")
    marks = [option.get("mentioned") for option in recall]
    if any(not isinstance(mark, bool) for mark in marks):
        problems.append(f"{where}.R2: every option needs mentioned true or false (the answer key)")
    elif not (any(marks) and not all(marks)):
        problems.append(f"{where}.R2: needs at least one mentioned sound and one distractor")
    return problems


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
    problems += _option_problems("P3", "A1s", practice["options"]["A1s"], page_spec, "practice.A1s")
    return problems


def _wording_problems(wording: Mapping[str, Any]) -> list[str]:
    problems = []
    if not str(wording.get("P0_instruction", {}).get("shown", "")).strip():
        problems.append("wording: the P0 text is empty")
    unknown = sorted(set(wording.get("titles", {})) - set(SHOWN_PAGES))
    if unknown:
        problems.append(f"wording: titles for pages diff does not show: {unknown}")
    for page, title in wording.get("titles", {}).items():
        problems += _korean_problems(str(title.get("shown", "")), f"wording.titles.{page}")
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
    problems += _wording_problems(setup["wording"])
    for clip in clips:
        problems += _clip_problems(clip, root, instrument.page_spec)
    formal = {frame["still"] for clip in clips for frame in clip.get("frames", [])}
    problems += _practice_problems(setup["practice"], formal, root, instrument.page_spec)
    if purpose == "participant":
        if setup.get("instrument_hash") != instrument.snapshot()["hash"]:
            problems.append("the workbook instrument changed since this setup was drafted")
        for name, part in (("wording", setup["wording"]), ("practice", setup["practice"])):
            if not approved(part):
                problems.append(f"{name}: not approved by the researcher (or edited since approval)")
        for clip in clips:
            if not approved(clip):
                problems.append(f"{clip['key']}: not approved by the researcher (or edited since approval)")
    return problems


def read_setup(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SetupError(f"{path}: not a setup object")
    return value
