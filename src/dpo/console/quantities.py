"""The measured half of scene-adaptive caption curation.

Everything in ``docs/v2-console/spec-system.md`` between the masks
and the prompt, as pure functions over numbers. The mask tree reader lives in
:mod:`dpo.console.masks`; this module never opens a file, so the whole chain
can be tested on three-source inputs.

The chain, from §2::

    masks → IoU grouping → {r_g, p_g, w_g} → v_g(r0) → filter
          → rank(α) → band + register → grain → prompt → caption

The grouping step is :func:`dpo.console.masks.group_audio_labels`, which is
where the mask windows and the AudioSet family constraint live; everything
from the quantities rightward is here.

Two properties hold it together and both are easy to break by accident.

Normalization happens over the *full* source set and filtering happens after
(§6). Normalizing over the admitted subset would make removing one source
redistribute weight across the rest and reorder rows the participant never
touched, so no control would be attributable. Every function here that takes
an admitted set takes the field it was filtered from as well.

Ordering saturates. Both normalized axes are built on the saturating
visibility ``v_g``, not on raw area, so coherence and ranking rest on one
perceptual assumption applied consistently (§6). The specification records
this as a revisable theory choice (§13, last row); ``Field`` keeps ``r`` beside
``v`` so the reverse resolution can be measured against it later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, TypedDict

_EPSILON = 1e-9

# Grain runs from itemized through grouped and scene-level to atmospheric
# (§7). At the atmospheric step nothing is named, so the admitted set and α
# have no surface to act on.
GRAINS = ("itemized", "grouped", "scene", "atmospheric")
UNNAMED_GRAIN = "atmospheric"


class QuantityError(ValueError):
    """A measured quantity is outside the range its definition allows."""


# ---- §5 quantities ----------------------------------------------------------


def mean_visual_share(areas: Sequence[float]) -> float:
    """``r_g``: the mean of ``a_g(t)`` over every frame of the window, zeros included."""
    if not areas:
        return 0.0
    if any(area < 0.0 or area > 1.0 for area in areas):
        raise QuantityError("an area ratio must lie in [0, 1]")
    return sum(areas) / len(areas)


def presence_rate(areas: Sequence[float]) -> float:
    """``p_g``: the fraction of frames in which the source occupies anything."""
    if not areas:
        return 0.0
    return sum(1 for area in areas if area > 0.0) / len(areas)


def salience(confidence: float, energy: float) -> float:
    """``w_g = c_g · e_g``: detection times acoustic prominence (§5.1).

    The product form separates the two. Confidence reports that a label is
    present; energy reports how much of the acoustic field it occupies, and a
    distant siren can carry high confidence with negligible energy. Weighting
    by confidence alone would make the audio axis detectional rather than
    acoustic — which is why :class:`Source` refuses to invent either factor.
    """
    if confidence < 0.0 or energy < 0.0:
        raise QuantityError("confidence and energy are non-negative")
    return confidence * energy


def visibility(share: float, half_saturation: float) -> float:
    """``v_g = r_g / (r_g + r0)``: saturating visibility (§5.3).

    Visibility saturates rather than thresholding, since a hard cutoff makes a
    small distant source flip between states on noise. ``r0`` is the area a
    source must occupy to count as half visible — an empirical claim about
    pedestrian-scale visual salience, calibrated once per corpus and then
    frozen, never a per-session knob.
    """
    if half_saturation <= 0.0:
        raise QuantityError("the half-saturation constant r0 must be positive")
    if share < 0.0:
        raise QuantityError("a mean visual share is non-negative")
    return share / (share + half_saturation)


# ---- §6 normalization -------------------------------------------------------


def normalize(values: Sequence[float]) -> list[float]:
    """Share of the total, or all zeros when the total is zero.

    Sum-normalization, not max-normalization. Both preserve the ranking within
    an axis, but the two axes are afterwards mixed by α, so the ratio between
    their normalizers decides where the two lines cross — which regimes exist
    and how wide each is. The specification says sum (§6), and the regime
    widths are the recorded variable (§7), so the choice is not cosmetic.
    """
    total = sum(values)
    if total <= 0.0:
        return [0.0 for _ in values]
    return [value / total for value in values]


# ---- the field of one shot --------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One grouped sound source in one shot, with the quantities of Table 1.

    ``confidence`` and ``energy`` are separate and required. A caller with no
    acoustic measurement must say so by passing the provisional pair its own
    provenance record explains, rather than letting this module quietly stand
    one in for the other.
    """

    id: str
    labels: tuple[str, ...]
    prose: str
    share: float  # r_g
    presence: float  # p_g
    confidence: float  # c_g
    energy: float  # e_g

    @property
    def salience(self) -> float:
        """``w_g``."""
        return salience(self.confidence, self.energy)


class Regime(TypedDict):
    """One ordering with the α interval over which it holds."""

    order: list[str]
    span: list[float]


@dataclass(frozen=True)
class Field:
    """Every source of one shot, normalized over the full set, ready to filter.

    Built once per shot from the calibration in force. ``admitted`` arguments
    below select from this field; nothing recomputes a normalizer.
    """

    sources: tuple[Source, ...]
    half_saturation: float
    min_regime: float
    visibility: dict[str, float] = field(init=False)  # v_g
    salience: dict[str, float] = field(init=False)  # w_g
    visibility_hat: dict[str, float] = field(init=False)  # v̂_g
    salience_hat: dict[str, float] = field(init=False)  # ŵ_g

    def __post_init__(self) -> None:
        ids = [source.id for source in self.sources]
        if len(set(ids)) != len(ids):
            raise QuantityError("two sources of one shot share an id")
        raw_v = [visibility(source.share, self.half_saturation) for source in self.sources]
        raw_w = [source.salience for source in self.sources]
        object.__setattr__(self, "visibility", dict(zip(ids, raw_v, strict=True)))
        object.__setattr__(self, "salience", dict(zip(ids, raw_w, strict=True)))
        object.__setattr__(self, "visibility_hat", dict(zip(ids, normalize(raw_v), strict=True)))
        object.__setattr__(self, "salience_hat", dict(zip(ids, normalize(raw_w), strict=True)))

    # -- §7 rank -------------------------------------------------------------

    def score(self, source_id: str, alpha: float) -> float:
        """``score_g = α·ŵ_g + (1−α)·v̂_g``, linear in α by construction."""
        return alpha * self.salience_hat[source_id] + (1.0 - alpha) * self.visibility_hat[source_id]

    def _ranking(self, admitted: Sequence[str], alpha: float) -> list[str]:
        # Stable: sources that agree everywhere keep document order and never
        # swap, so a tie is not a crossing.
        return sorted(admitted, key=lambda source_id: -self.score(source_id, alpha))

    def _crossings(self, admitted: Sequence[str]) -> list[float]:
        points: set[float] = set()
        for left, right in combinations(admitted, 2):
            slope = (self.salience_hat[left] - self.visibility_hat[left]) - (
                self.salience_hat[right] - self.visibility_hat[right]
            )
            if abs(slope) < _EPSILON:
                continue
            crossing = (self.visibility_hat[right] - self.visibility_hat[left]) / slope
            if _EPSILON < crossing < 1.0 - _EPSILON:
                points.add(round(crossing, 12))
        return sorted(points)

    def regimes(self, admitted: Sequence[str]) -> list[Regime]:
        """The reachable orderings of ``admitted``, each with its α interval.

        The score is linear in α, so the induced ordering changes only where
        two sources cross, and at most ``n(n−1)/2 + 1`` orderings partition
        [0, 1] (§7). Within a regime α is unidentifiable because no output
        distinguishes values inside it, so the regime — not a number — is what
        the interface offers and what the log records.

        Regimes narrower than ``min_regime`` are folded into their neighbours.
        The interface draws segments at widths proportional to these spans, and
        a segment of a thousandth of the axis is a target no hand can hit and a
        censoring interval no analysis should be handed.
        """
        selected = [source_id for source_id in (s.id for s in self.sources) if source_id in set(admitted)]
        if not selected:
            return []
        boundaries = [0.0, *self._crossings(selected), 1.0]
        result: list[Regime] = []
        for low, high in zip(boundaries, boundaries[1:], strict=False):
            order = self._ranking(selected, (low + high) / 2.0)
            if result and result[-1]["order"] == order:
                result[-1]["span"][1] = high
            else:
                result.append({"order": order, "span": [low, high]})
        return _fold_narrow(result, self.min_regime)

    # -- §9 measurements, never shown to a participant -----------------------

    def anchoring(self) -> float | None:
        """``C_anch``: what proportion of what is heard can be seen.

        None when nothing is heard, since the ratio has no denominator then.
        Computed on the raw quantities, not the normalized ones: it is a
        salience-weighted mean of a visibility in [0, 1], and normalizing
        would leave a number that no longer reads as a proportion.
        """
        total = sum(self.salience.values())
        if total <= 0.0:
            return None
        return sum(self.salience[key] * self.visibility[key] for key in self.salience) / total

    def divergence(self) -> float | None:
        """``D``: whether the modalities agree on relative weighting.

        Total variation between the two normalized axes — the same measure
        §3.1 cuts shots with, so one metric performs both jobs. Undefined when
        nothing is visible, "which is a meaningful state rather than an error"
        (§9), so the caller gets None and must record it as such.
        """
        if sum(self.visibility.values()) <= 0.0 or sum(self.salience.values()) <= 0.0:
            return None
        return 0.5 * sum(abs(self.salience_hat[key] - self.visibility_hat[key]) for key in self.salience_hat)

    def residuals(self) -> dict[str, float]:
        """``ρ_g = ŵ_g − v̂_g``: which source the disagreement sits on."""
        return {key: self.salience_hat[key] - self.visibility_hat[key] for key in self.salience_hat}

    def measurements(self) -> dict[str, Any]:
        """§9 in one object, for the log and the model — never for a screen."""
        return {
            "anchoring": self.anchoring(),
            "divergence": self.divergence(),
            "residuals": self.residuals(),
        }


def subsets_of(ids: Sequence[str]) -> list[tuple[str, ...]]:
    """Every non-empty admitted set, smallest first, in document order within a size.

    The regimes of all of these are precomputed and served with the shot, so
    muting a source redraws the crossfader with no round trip. At the eight
    sources a shot may carry that is 255 entries, each a short list of ids.
    """
    return [tuple(subset) for size in range(1, len(ids) + 1) for subset in combinations(ids, size)]


def _fold_narrow(result: list[Regime], min_regime: float) -> list[Regime]:
    """Drop regimes below ``min_regime``, keeping the spans a partition of [0, 1].

    The narrowest goes first and its span passes to its neighbours, so folding
    never opens a gap. The endpoints survive whatever happens between them:
    the visibility-first and salience-first orderings hold at α=0 and α=1.
    """
    while len(result) > 1:
        widths = [regime["span"][1] - regime["span"][0] for regime in result]
        narrowest = min(range(len(result)), key=widths.__getitem__)
        if widths[narrowest] >= min_regime:
            break
        low, high = result[narrowest]["span"]
        del result[narrowest]
        if narrowest == 0:
            result[0]["span"][0] = low
        elif narrowest == len(result):
            result[-1]["span"][1] = high
        else:
            middle = (low + high) / 2.0
            result[narrowest - 1]["span"][1] = middle
            result[narrowest]["span"][0] = middle
        index = 0
        while index + 1 < len(result):
            if result[index]["order"] == result[index + 1]["order"]:
                result[index]["span"][1] = result[index + 1]["span"][1]
                del result[index + 1]
            else:
                index += 1
    return result


# ---- §3.1 shot segmentation -------------------------------------------------


def composition(shares: Mapping[str, float]) -> dict[str, float]:
    """``c(t)``: the frame's normalized composition over visual classes."""
    total = sum(shares.values())
    if total <= 0.0:
        return {key: 0.0 for key in shares}
    return {key: value / total for key, value in shares.items()}


def total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """``½ Σ_k |c_k(t) − c_k(t−Δ)|``, the distance §3.1 cuts on.

    The same measure returns in §9 as the emphasis divergence, so one metric
    performs both jobs: how much the image changed, and how far the two
    modalities disagree.
    """
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def cut_points(
    compositions: Sequence[Mapping[str, float]],
    *,
    threshold: float,
    stride: int,
    minimum_frames: int,
) -> list[int]:
    """Frame indices where a shot boundary falls.

    A cut goes where composition shifts by more than ``threshold`` across
    ``stride`` frames, subject to a floor on shot length. The floor is set by
    audio reliability rather than by any visual criterion (§5.2): audio windows
    align to these cuts, so a siren sweeping across a cut splits into two
    weaker detections, and lowering the threshold shortens windows and degrades
    every ``w_g`` measured in them.

    The floor binds on every shot, the last one included: a cut close enough
    to the end that the tail would fall under the floor is not taken, because
    the tail is a shot too and its audio window would be the short one. A clip
    that cannot be split into shots that all clear the floor stays whole.

    Boundaries are edited by hand after this runs, and the threshold never
    appears again downstream (§3.1).
    """
    if stride < 1:
        raise QuantityError("stride must be at least one frame")
    if minimum_frames < 1:
        raise QuantityError("a shot must be at least one frame long")
    cuts: list[int] = []
    last = 0
    for index in range(stride, len(compositions)):
        if index - last < minimum_frames:
            continue
        if len(compositions) - index < minimum_frames:
            break
        if total_variation(compositions[index], compositions[index - stride]) > threshold:
            cuts.append(index)
            last = index
    return cuts


def segments(frame_count: int, cuts: Sequence[int]) -> list[tuple[int, int]]:
    """``[start, end)`` frame ranges for the cuts, covering the whole clip."""
    edges = [0, *[cut for cut in cuts if 0 < cut < frame_count], frame_count]
    return [(low, high) for low, high in zip(edges, edges[1:], strict=False) if high > low]
