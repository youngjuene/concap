"""The AudioSet ontology, resolved against the corpus's tidy CSV.

Both instruments need the same two facts about an audio label: how often the
audio-language model tagged it on a clip, and which top-level AudioSet family it
belongs to. The first is a count in ``final_labels``; the second is a walk up
the ontology's DAG, which is ambiguous exactly where it matters — *Bell* is a
child of both *Music* and *Sounds of things* — and the CSV's own
``top_level_parent_name`` column carries the per-clip family counts that
resolve the tie.

Neither instrument owns this. It is corpus knowledge, so it lives with the
other corpus-shaped machinery in ``dpo.caption`` rather than in either
package, and each imports it.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict


class OntologyError(ValueError):
    """The ontology or the tidy CSV cannot be read, or the two disagree."""


class TagRecord(TypedDict):
    """One clip's audio tags: multiplicities, resolved families, declared totals."""

    counts: Counter[str]
    parents: dict[str, str]
    family_counts: Counter[str]


def split_labels(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


def parse_family_counts(value: str, *, clip_id: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for raw in split_labels(value):
        family, separator, count = raw.rpartition(":")
        if not separator or not family.strip() or not count.isdigit():
            raise OntologyError(f"{clip_id}: top_level_parent_name entry {raw!r} must be '<family>:<count>'")
        result[family.strip()] += int(count)
    return result


def ontology_parents(path: Path) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    """``(id by name, parents by id, root order)`` from the ontology JSON."""
    try:
        nodes = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OntologyError(f"cannot read AudioSet ontology {path}: {exc}") from exc
    if not isinstance(nodes, list):
        raise OntologyError(f"{path}: ontology must be a JSON list")
    by_id: dict[str, str] = {}
    id_by_name: dict[str, str] = {}
    parents: dict[str, list[str]] = defaultdict(list)
    children: set[str] = set()
    for index, raw in enumerate(nodes):
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("id"), str)
            or not isinstance(raw.get("name"), str)
        ):
            raise OntologyError(f"{path}: ontology[{index}] must carry string id and name")
        node_id, name = raw["id"], raw["name"]
        by_id[node_id] = name
        id_by_name[name] = node_id
        child_ids = raw.get("child_ids", [])
        if not isinstance(child_ids, list) or not all(isinstance(child, str) for child in child_ids):
            raise OntologyError(f"{path}: ontology[{index}].child_ids must be strings")
        for child in child_ids:
            parents[child].append(node_id)
            children.add(child)
    roots = [node_id for node_id in by_id if node_id not in children]
    root_order = {node_id: index for index, node_id in enumerate(roots)}
    return id_by_name, parents, root_order


def top_parent_candidates(
    name: str,
    *,
    ontology_path: Path,
    id_by_name: Mapping[str, str],
    parents: Mapping[str, Sequence[str]],
    root_order: Mapping[str, int],
) -> tuple[str, ...]:
    """Every top-level family the label can reach, nearest first."""
    try:
        start = id_by_name[name]
    except KeyError as exc:
        raise OntologyError(f"audio label {name!r} is absent from {ontology_path}") from exc
    name_by_id = {node_id: node_name for node_name, node_id in id_by_name.items()}
    seen = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    found: dict[str, int] = {}
    while queue:
        node_id, depth = queue.popleft()
        if node_id in root_order:
            found.setdefault(node_id, depth)
            continue
        for parent in parents.get(node_id, []):
            if parent not in seen:
                seen.add(parent)
                queue.append((parent, depth + 1))
    if not found:
        raise OntologyError(f"audio label {name!r} has no top-level parent in {ontology_path}")
    roots = sorted(found, key=lambda node_id: (found[node_id], root_order[node_id]))
    return tuple(name_by_id[root] for root in roots)


def assign_declared_parents(
    counts: Counter[str],
    candidates: Mapping[str, Sequence[str]],
    declared: Counter[str],
    *,
    clip_id: str,
) -> dict[str, str]:
    """Resolve ontology DAG ties so per-label families satisfy the CSV counts."""
    labels = list(counts)

    def solve(index: int, remaining: Counter[str], assigned: dict[str, str]) -> dict[str, str] | None:
        if index == len(labels):
            return dict(assigned) if not +remaining else None
        label = labels[index]
        occurrences = counts[label]
        for parent in candidates[label]:
            if remaining[parent] < occurrences:
                continue
            next_remaining = remaining.copy()
            next_remaining[parent] -= occurrences
            if not next_remaining[parent]:
                del next_remaining[parent]
            assigned[label] = parent
            resolved = solve(index + 1, next_remaining, assigned)
            if resolved is not None:
                return resolved
        assigned.pop(label, None)
        return None

    resolved = solve(0, declared.copy(), {})
    if resolved is None:
        choices = {label: list(parents) for label, parents in candidates.items()}
        raise OntologyError(
            f"{clip_id}: top_level_parent_name {dict(declared)} cannot be assigned "
            f"to final_labels through ontology candidates {choices}"
        )
    return resolved


def load_tags(tidy_data: str | Path, ontology: str | Path) -> dict[str, TagRecord]:
    """Every clip's tags with their families resolved.

    The real CSV carries one row per participant per clip, all repeating the
    same ``final_labels`` cell; the cell is read once and rows are never
    summed. Rows that disagree are a data problem the caller must see.
    """
    tidy_path, ontology_path = Path(tidy_data), Path(ontology)
    try:
        with tidy_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise OntologyError(f"cannot read tidy data {tidy_path}: {exc}") from exc
    required = {"file_index", "final_labels", "top_level_parent_name"}
    if not rows or not required.issubset(rows[0]):
        raise OntologyError(f"{tidy_path}: required columns are {sorted(required)}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["file_index"].strip()].append(row)

    id_by_name, parents, root_order = ontology_parents(ontology_path)
    records: dict[str, TagRecord] = {}
    for clip_id, clip_rows in grouped.items():
        label_cells = {row["final_labels"].strip() for row in clip_rows}
        family_cells = {row["top_level_parent_name"].strip() for row in clip_rows}
        if len(label_cells) != 1 or len(family_cells) != 1:
            raise OntologyError(
                f"{clip_id}: repeated tidy rows disagree on final_labels or top_level_parent_name"
            )
        counts = Counter(split_labels(next(iter(label_cells))))
        declared = parse_family_counts(next(iter(family_cells)), clip_id=clip_id)
        candidates = {
            label: top_parent_candidates(
                label,
                ontology_path=ontology_path,
                id_by_name=id_by_name,
                parents=parents,
                root_order=root_order,
            )
            for label in counts
        }
        resolved = assign_declared_parents(counts, candidates, declared, clip_id=clip_id)
        records[clip_id] = {"counts": counts, "parents": resolved, "family_counts": declared}
    return records


__all__ = [
    "OntologyError",
    "TagRecord",
    "assign_declared_parents",
    "load_tags",
    "ontology_parents",
    "parse_family_counts",
    "split_labels",
    "top_parent_candidates",
]
