"""The human study's analysis: placement and match rating on the measured axis.

Every number here is a pure function of the validated responses and the
export they answer, so ``dpo.study-results/v1`` regenerates bit-identically
from its parents. Intervals come from the cluster bootstrap in
``dpo.analysis.bootstrap`` twice over — resampling whole clips, then whole
participants — because both are random effects of this design and an
interval clustered on one of them understates the other. No mixed model is
fitted here; a downstream fit reads the persisted rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from dpo.analysis.bootstrap import clip_cluster_bootstrap
from dpo.analysis.bradley_terry import AnalysisError
from dpo.userstudy.responses import StudyResponse

Statistic = Callable[[Sequence[StudyResponse]], float]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Pearson's r, or None when either side is constant (r is undefined there)."""
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x, mean_y = _mean(xs), _mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return covariance / math.sqrt(var_x * var_y)


def summarize_study(
    responses: Sequence[StudyResponse],
    *,
    rungs: int,
    bootstrap_samples: int,
    seed: int = 0,
) -> dict[str, object]:
    """Placement, match rating, and their relation, with clip- and participant-clustered intervals."""
    if not responses:
        raise AnalysisError("the study analysis requires at least one response")
    if any(response.congruency_index >= rungs for response in responses):
        raise AnalysisError("a response names a rung beyond the ladder width")

    def intervals(statistic: Statistic) -> dict[str, list[float]]:
        by_clip = clip_cluster_bootstrap(
            responses, lambda response: response.clip_id, statistic, samples=bootstrap_samples, seed=seed
        )
        by_participant = clip_cluster_bootstrap(
            responses,
            lambda response: response.participant_hash,
            statistic,
            samples=bootstrap_samples,
            seed=seed,
        )
        return {"by_clip": list(by_clip[1]), "by_participant": list(by_participant[1])}

    def mean_of(field: Callable[[StudyResponse], float]) -> Statistic:
        return lambda rows: _mean([field(row) for row in rows])

    def histogram(rows: Sequence[StudyResponse]) -> list[int]:
        counts = [0] * rungs
        for row in rows:
            counts[row.congruency_index] += 1
        return counts

    position = mean_of(lambda row: row.congruency_position)
    congruency = mean_of(lambda row: row.congruency)
    rating = mean_of(lambda row: float(row.match_rating))

    def correlation(rows: Sequence[StudyResponse]) -> float:
        value = _pearson([(row.congruency, float(row.match_rating)) for row in rows])
        # A resample with a constant side has no correlation; report it as 0
        # inside the bootstrap so the interval still covers that outcome.
        return 0.0 if value is None else value

    point_correlation = _pearson([(row.congruency, float(row.match_rating)) for row in responses])
    by_rung: dict[str, dict[str, object]] = {}
    for index in range(rungs):
        members = [row for row in responses if row.congruency_index == index]
        by_rung[str(index)] = {
            "responses": len(members),
            "mean_match_rating": _mean([float(row.match_rating) for row in members]) if members else None,
        }
    presentation_indices = sorted({row.presentation_index for row in responses})
    per_clip: dict[str, dict[str, object]] = {}
    for clip_id in sorted({row.clip_id for row in responses}):
        members = [row for row in responses if row.clip_id == clip_id]
        per_clip[clip_id] = {
            "responses": len(members),
            "mean_position": position(members),
            "mean_congruency": congruency(members),
            "mean_match_rating": rating(members),
            "rung_histogram": histogram(members),
        }
    per_participant: dict[str, dict[str, object]] = {}
    for identity in sorted({row.participant_hash for row in responses}):
        members = [row for row in responses if row.participant_hash == identity]
        per_participant[identity] = {
            "responses": len(members),
            "mean_position": position(members),
            "mean_match_rating": rating(members),
        }
    return {
        "responses": len(responses),
        "participants": len(per_participant),
        "clips": len(per_clip),
        "rungs": rungs,
        "bootstrap_samples": bootstrap_samples,
        "placement": {
            "mean_position": position(responses),
            "position_ci95": intervals(position),
            "mean_congruency": congruency(responses),
            "congruency_ci95": intervals(congruency),
            "rung_histogram": histogram(responses),
        },
        "match_rating": {
            "mean": rating(responses),
            "ci95": intervals(rating),
            "by_rung": by_rung,
        },
        "congruency_rating_correlation": {
            "pearson_r": point_correlation,
            "ci95": intervals(correlation) if point_correlation is not None else None,
        },
        "order": {
            "mean_position_by_presentation_index": [
                position([row for row in responses if row.presentation_index == index])
                for index in presentation_indices
            ],
            "presentation_indices": presentation_indices,
        },
        "per_clip": per_clip,
        "per_participant": per_participant,
    }
