"""The orderings a set of sources can take between the eye and the ear.

Balance is "a choice among the distinct orderings the admitted sources can
take between an eye-first list and an ear-first list" (spec 4.2), and the
columns beside the rows draw exactly those orderings. So balance is not a
number the participant sets — it is which of a few rankings the list shows —
and this module computes the rankings so the browser never needs the weights.

Each source carries ``weights = [left, right]``: on the audio skeleton
visibility and salience, on the control skeleton nearness and prominence at a
distance. Sweep λ from 0 (eye / near) to 1 (ear / far) with
``score(λ) = (1−λ)·left + λ·right`` and rank by score. The ranking only changes
where two scores cross, so the distinct orderings are found exactly by
evaluating the ranking between consecutive crossings. Every ordering carries
the λ interval in which it holds, and the browser keeps λ (the midpoint of the
selected span) so that when admission changes the count of orderings the
selection carries to the ordering whose span contains it — the "nearest"
ordering of spec 4.2 — rather than resetting.

The grouped level (spec 4.3) ranks the role heads instead: a head's weights are
the element-wise maximum over its admitted members, so a group leans where its
strongest member leans, and the same sweep applies.

``all_orderings`` precomputes the table for every non-empty subset of a shot's
sources; that table, and never the weights, is what the inventory route sends.
Everything here is pure and stays free of the document module so it can be
tested on three-line inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, TypedDict

from dpo.session.document import LEVELS

_EPSILON = 1e-9


class Ordering(TypedDict):
    order: list[str]
    span: list[float]


@dataclass(frozen=True)
class Weighted:
    """One ranked thing — a source or a role head — with its two weights."""

    id: str
    left: float
    right: float

    def score(self, balance: float) -> float:
        return (1.0 - balance) * self.left + balance * self.right


class SettingsError(ValueError):
    """A caption request's settings name something the shot cannot produce."""


def weighted(sources: Iterable[Mapping[str, Any]]) -> list[Weighted]:
    return [Weighted(str(s["id"]), float(s["weights"][0]), float(s["weights"][1])) for s in sources]


def _ranking(items: Sequence[Weighted], balance: float) -> list[str]:
    # Stable sort: ties keep document order, so a tie is not a crossing and
    # two sources that agree everywhere never swap.
    return [item.id for item in sorted(items, key=lambda item: -item.score(balance))]


def _crossings(items: Sequence[Weighted]) -> list[float]:
    points: set[float] = set()
    for a, b in combinations(items, 2):
        slope = (a.right - a.left) - (b.right - b.left)
        if abs(slope) < _EPSILON:
            continue
        crossing = (b.left - a.left) / slope
        if _EPSILON < crossing < 1.0 - _EPSILON:
            points.add(round(crossing, 12))
    return sorted(points)


def orderings(sources: Sequence[Mapping[str, Any]] | Sequence[Weighted]) -> list[Ordering]:
    """The distinct rankings from λ=0 to λ=1, each with the span it holds over.

    Spans partition [0, 1]. One item, or items whose lines never cross in
    (0, 1), give exactly one ordering with span [0, 1].
    """
    items = [
        s if isinstance(s, Weighted) else Weighted(str(s["id"]), *map(float, s["weights"])) for s in sources
    ]
    if not items:
        return []
    boundaries = [0.0, *_crossings(items), 1.0]
    result: list[Ordering] = []
    for lo, hi in zip(boundaries, boundaries[1:], strict=False):
        order = _ranking(items, (lo + hi) / 2.0)
        if result and result[-1]["order"] == order:
            result[-1]["span"][1] = hi
        else:
            result.append({"order": order, "span": [lo, hi]})
    return result


def resolve(candidates: Sequence[Ordering], balance: float) -> int:
    """The index of the ordering whose span contains ``balance``."""
    if not candidates:
        raise SettingsError("no orderings to resolve against")
    for index, candidate in enumerate(candidates):
        lo, hi = candidate["span"]
        if lo - _EPSILON <= balance <= hi + _EPSILON:
            return index
    return len(candidates) - 1


def heads_present(sources: Iterable[Mapping[str, Any]], roles: Mapping[str, str]) -> list[str]:
    """The role ids present among the given sources, in role-head order."""
    present = {str(source["role"]) for source in sources}
    return [role for role in roles if role in present]


def head_weights(sources: Sequence[Mapping[str, Any]], roles: Mapping[str, str]) -> list[Weighted]:
    """One weighted item per present head: element-wise MAX over its members."""
    heads: list[Weighted] = []
    for role in heads_present(sources, roles):
        members = [source for source in sources if source["role"] == role]
        heads.append(
            Weighted(
                role,
                max(float(member["weights"][0]) for member in members),
                max(float(member["weights"][1]) for member in members),
            )
        )
    return heads


def subset_key(ids: Iterable[str]) -> str:
    return "+".join(sorted(ids))


def all_orderings(shot: Mapping[str, Any], roles: Mapping[str, str]) -> dict[str, dict[str, list[Ordering]]]:
    """The itemized and grouped orderings for every non-empty admitted subset."""
    sources = list(shot["sources"])
    itemized: dict[str, list[Ordering]] = {}
    grouped: dict[str, list[Ordering]] = {}
    for size in range(1, len(sources) + 1):
        for subset in combinations(sources, size):
            key = subset_key(str(source["id"]) for source in subset)
            itemized[key] = orderings(list(subset))
            grouped[key] = orderings(head_weights(list(subset), roles))
    return {"itemized": itemized, "grouped": grouped}


def members_in_order(
    sources: Sequence[Mapping[str, Any]], role: str, balance: float
) -> list[Mapping[str, Any]]:
    """A head's admitted members in the head's itemized order at ``balance``."""
    members = [source for source in sources if source["role"] == role]
    ranked = _ranking(weighted(members), balance)
    by_id = {str(source["id"]): source for source in members}
    return [by_id[source_id] for source_id in ranked]


def settings_key(level: str, admitted: Sequence[str], order: Sequence[str]) -> str:
    """The identity of a caption's settings; freshness is key equality (spec 4.4).

    Itemized names the order of the admitted sources; grouped names the head
    order and the admitted set (member order is a function of both); scene and
    atmospheric have no free parameter, so every such request is one key and
    identical settings return the identical caption (spec 7).
    """
    if level == "itemized":
        return "itemized|" + ",".join(order)
    if level == "grouped":
        return "grouped|" + ",".join(order) + "|" + ",".join(sorted(admitted))
    if level in ("scene", "atmospheric"):
        return level
    raise SettingsError(f"settings.level must be one of {list(LEVELS)}")


@dataclass(frozen=True)
class Settings:
    level: str
    admitted: tuple[str, ...]
    order: tuple[str, ...]

    @property
    def key(self) -> str:
        return settings_key(self.level, self.admitted, self.order)


def _id_list(value: object, field: str, known: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SettingsError(f"settings.{field} must be a list of ids")
    ids = tuple(str(item) for item in value)
    known_set = set(known)
    for item in ids:
        if item not in known_set:
            raise SettingsError(f"settings.{field} names unknown id {item!r}")
    if len(set(ids)) != len(ids):
        raise SettingsError(f"settings.{field} repeats an id")
    return ids


def validate_settings(
    shot: Mapping[str, Any], roles: Mapping[str, str], settings: Mapping[str, Any]
) -> Settings:
    """Refuse settings the skeleton could not have produced (contract §3).

    The browser can only reach an ordering by tapping a column, so an order
    that is not one of the precomputed orderings for the admitted subset is a
    forged request, not a participant's choice.
    """
    level = settings.get("level")
    if not isinstance(level, str) or level not in LEVELS:
        raise SettingsError(f"settings.level must be one of {list(LEVELS)}")
    source_ids = [str(source["id"]) for source in shot["sources"]]
    if level in ("scene", "atmospheric"):
        admitted_any = settings.get("admitted", [])
        admitted = _id_list(admitted_any, "admitted", source_ids) if isinstance(admitted_any, list) else ()
        return Settings(level=level, admitted=admitted, order=())
    admitted = _id_list(settings.get("admitted"), "admitted", source_ids)
    if not admitted:
        raise SettingsError(f"settings.admitted must not be empty at the {level} level")
    subset = [source for source in shot["sources"] if source["id"] in admitted]
    if level == "itemized":
        order = _id_list(settings.get("order"), "order", admitted)
        if set(order) != set(admitted):
            raise SettingsError("settings.order must be a permutation of settings.admitted")
        reachable = orderings(subset)
    else:
        present = heads_present(subset, roles)
        order = _id_list(settings.get("order"), "order", present)
        if set(order) != set(present):
            raise SettingsError("settings.order must be a permutation of the role heads present")
        reachable = orderings(head_weights(subset, roles))
    if list(order) not in [candidate["order"] for candidate in reachable]:
        raise SettingsError("settings.order is not an ordering the admitted set can take")
    return Settings(level=level, admitted=admitted, order=order)


def balance_of(shot: Mapping[str, Any], roles: Mapping[str, str], settings: Settings) -> float:
    """The λ midpoint of the settings' ordering: what grouped member order uses."""
    subset = [source for source in shot["sources"] if source["id"] in settings.admitted]
    if settings.level == "itemized":
        reachable = orderings(subset)
    elif settings.level == "grouped":
        reachable = orderings(head_weights(subset, roles))
    else:
        return 0.5
    for candidate in reachable:
        if candidate["order"] == list(settings.order):
            lo, hi = candidate["span"]
            return (lo + hi) / 2.0
    return 0.5
