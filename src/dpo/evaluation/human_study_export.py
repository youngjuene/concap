"""Blinded incomplete-block human study export (PRD section 29).

Nine models yield 36 unordered pairs. Every pair receives the contract's
exposure count, clips are balanced across pairs, A/B position is randomized
deterministically, participant blocks are clip-disjoint, and everything a
participant can see is scanned against a blinding lexicon. Model identity
lives only in the restricted randomization manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from dpo.contracts.study_contract import EXPERIMENT_IDS, OBJECTIVE_NAMES
from dpo.core.identity import semantic_hash

# Derived from the code-owned vocabularies so a renamed or added experiment can
# never silently escape the blinding scan; the extras cover generic phrasing.
BLINDING_TOKENS = tuple(
    sorted(
        {experiment_id.casefold() for experiment_id in EXPERIMENT_IDS}
        | set(OBJECTIVE_NAMES)
        | {"model", "baseline", "warm"}
    )
)


class StudyExportError(ValueError):
    """Raised when a study export would leak identity or break the design."""


@dataclass(frozen=True)
class StudyTask:
    task_id: str
    block: int
    clip_id: str
    track: str
    stimulus_a: str
    stimulus_b: str


@dataclass(frozen=True)
class StudyExport:
    tasks: tuple[StudyTask, ...]
    candidate_text: dict[str, str]  # stimulus_id -> caption text (blind)
    media_manifest: dict[str, str]  # clip_id -> media hash
    randomization: dict[str, dict[str, str]]  # RESTRICTED: task_id -> identities
    attention_checks: tuple[StudyTask, ...]

    def public_documents(self) -> dict[str, list[dict[str, object]]]:
        return {
            "study_task_manifest": [
                {
                    "task_id": task.task_id,
                    "block": task.block,
                    "clip_id": task.clip_id,
                    "track": task.track,
                    "stimulus_a": task.stimulus_a,
                    "stimulus_b": task.stimulus_b,
                }
                for task in self.tasks
            ],
            "study_candidate_text": [
                {"stimulus_id": stimulus_id, "text": text}
                for stimulus_id, text in sorted(self.candidate_text.items())
            ],
            "study_media_manifest": [
                {"clip_id": clip_id, "media_hash": media_hash}
                for clip_id, media_hash in sorted(self.media_manifest.items())
            ],
            "study_attention_checks": [
                {
                    "task_id": task.task_id,
                    "block": task.block,
                    "clip_id": task.clip_id,
                    "track": task.track,
                    "stimulus_a": task.stimulus_a,
                    "stimulus_b": task.stimulus_b,
                }
                for task in self.attention_checks
            ],
        }


def _scan_blinding(text: str, context: str) -> None:
    lowered = text.casefold()
    for token in BLINDING_TOKENS:
        if token in lowered.split() or f"{token}-" in lowered or f"-{token}" in lowered:
            raise StudyExportError(f"{context} leaks a blinding token {token!r}")


def build_study_export(
    *,
    track: str,
    captions: Mapping[str, Mapping[str, str]],  # model_id -> clip_id -> caption
    media_hashes: Mapping[str, str],
    exposures_per_pair: int,
    block_size: int = 8,
    salt: str = "dpo-study/v1",
    attention_pairs: Sequence[tuple[str, str, str]] = (),  # (clip_id, good, bad)
) -> StudyExport:
    models = tuple(sorted(captions))
    if models != EXPERIMENT_IDS:
        raise StudyExportError(f"the study compares exactly the nine-condition suite; got {list(models)}")
    clip_sets = {model: set(captions[model]) for model in models}
    common_clips = sorted(set.intersection(*clip_sets.values()))
    if not common_clips:
        raise StudyExportError("every model must caption the same study clips")
    missing_media = sorted(set(common_clips) - set(media_hashes))
    if missing_media:
        raise StudyExportError(f"study clip {missing_media[0]!r} has no media hash")
    pairs = list(combinations(models, 2))
    usage = dict.fromkeys(common_clips, 0)
    tasks: list[StudyTask] = []
    randomization: dict[str, dict[str, str]] = {}
    candidate_text: dict[str, str] = {}
    for left, right in pairs:
        ordered_clips = sorted(
            common_clips,
            key=lambda clip: (
                usage[clip],
                semantic_hash({"salt": salt, "pair": [left, right], "clip": clip}),
            ),
        )
        if exposures_per_pair > len(ordered_clips):
            raise StudyExportError("not enough study clips for the requested exposures per model pair")
        for clip in ordered_clips[:exposures_per_pair]:
            usage[clip] += 1
            flip = (
                int(
                    semantic_hash({"salt": salt, "flip": [left, right, clip]}).removeprefix("sha256:")[:2],
                    16,
                )
                % 2
                == 1
            )
            first, second = (right, left) if flip else (left, right)
            task_id = (
                "task-"
                + semantic_hash({"salt": salt, "pair": [left, right], "clip": clip}).removeprefix("sha256:")[
                    :12
                ]
            )
            stimuli = {}
            for slot, model in (("a", first), ("b", second)):
                stimulus_id = (
                    "stim-"
                    + semantic_hash(
                        {"salt": salt, "task": task_id, "slot": slot, "model": model}
                    ).removeprefix("sha256:")[:12]
                )
                text = captions[model][clip]
                _scan_blinding(text, f"caption for task {task_id}")
                candidate_text[stimulus_id] = text
                stimuli[slot] = (stimulus_id, model)
            tasks.append(
                StudyTask(
                    task_id=task_id,
                    block=0,
                    clip_id=clip,
                    track=track,
                    stimulus_a=stimuli["a"][0],
                    stimulus_b=stimuli["b"][0],
                )
            )
            randomization[task_id] = {
                "model_a": stimuli["a"][1],
                "model_b": stimuli["b"][1],
                "clip_id": clip,
            }
    # Clip-disjoint blocks with bounded burden: a greedy pass that opens a new
    # block whenever the current one saw the clip or is full.
    ordered_tasks = sorted(tasks, key=lambda task: task.task_id)
    blocks: list[tuple[set[str], list[StudyTask]]] = []
    assigned: list[StudyTask] = []
    for task in ordered_tasks:
        placed = False
        for index, (clips_in_block, members) in enumerate(blocks):
            if task.clip_id in clips_in_block or len(members) >= block_size:
                continue
            clips_in_block.add(task.clip_id)
            members.append(task)
            assigned.append(
                StudyTask(
                    task_id=task.task_id,
                    block=index,
                    clip_id=task.clip_id,
                    track=task.track,
                    stimulus_a=task.stimulus_a,
                    stimulus_b=task.stimulus_b,
                )
            )
            placed = True
            break
        if not placed:
            blocks.append(({task.clip_id}, [task]))
            assigned.append(
                StudyTask(
                    task_id=task.task_id,
                    block=len(blocks) - 1,
                    clip_id=task.clip_id,
                    track=task.track,
                    stimulus_a=task.stimulus_a,
                    stimulus_b=task.stimulus_b,
                )
            )
    attention_tasks = []
    for index, (clip_id, good, bad) in enumerate(attention_pairs):
        task_id = (
            "task-attn-" + semantic_hash({"salt": salt, "attention": index}).removeprefix("sha256:")[:12]
        )
        good_id = (
            "stim-"
            + semantic_hash({"salt": salt, "attn": index, "slot": "good"}).removeprefix("sha256:")[:12]
        )
        bad_id = (
            "stim-" + semantic_hash({"salt": salt, "attn": index, "slot": "bad"}).removeprefix("sha256:")[:12]
        )
        _scan_blinding(good, f"attention caption {task_id}")
        _scan_blinding(bad, f"attention caption {task_id}")
        candidate_text[good_id] = good
        candidate_text[bad_id] = bad
        attention_tasks.append(
            StudyTask(
                task_id=task_id,
                block=index % max(1, len(blocks)),
                clip_id=clip_id,
                track=track,
                stimulus_a=good_id,
                stimulus_b=bad_id,
            )
        )
    return StudyExport(
        tasks=tuple(assigned),
        candidate_text=candidate_text,
        media_manifest={clip: media_hashes[clip] for clip in common_clips},
        randomization=randomization,
        attention_checks=tuple(attention_tasks),
    )
