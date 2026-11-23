"""The chain's tests, fitted from a research export (``docs/v4-diff/chain-spec.md`` §5).

Researcher, 2026-11-23: H1 (follow beats counter) and R (binding is a listener's trait) are
confirmatory; H2, H3 and D are registered estimates with their direction stated, since at N = 48
between-person links from three slider settings each are underpowered (the chain simulation,
``artifacts/diff/chain-sim``). Every test reports an estimate, its central 90% interval and the
one-sided p of the predicted direction.

| | Outcome | Predictors | Prediction |
|---|---|---|---|
| H1 | P9's F + BF per participant × clip | P4's F + BF, follow ±½, participant, clip | follow > 0 |
| H2 | context slider's settled stop per participant × video | ``B_i``, start, video | ``B`` > 0 |
| H3 | texture slider's settled stop per participant × video | ``Φ_i``, start, video | ``Φ`` > 0 |
| D | H2 and H3 with both traits | ``B_i`` and ``Φ_i`` | cross terms near 0 |
| R | the source block's intercept ``a_i`` | ``B_i`` | correlation > 0 |

The registered models are Bayesian multilevel (Kay et al., CHI 2016). This module is their
pre-specified check, in least squares, with intervals that hold at 48 participants (a percentile
bootstrap over participants missed zero in 20% of null simulations against a nominal 10%):
- H1 with the participant and the clip as fixed effects, which is what the multilevel fit
  converges to under flat priors and this balanced design, and a standard error cluster-robust by
  participant on N − 1 degrees of freedom;
- H2, H3 and D on one row per participant: each video's settled stop net of the slider's start
  and the video (estimated on all rows), averaged per participant, then regressed on the trait,
  so a participant's three videos count once;
- R as Pearson's r with its Fisher-z interval. ``a_i`` is estimated the way ``B_i`` is
(:func:`diff.phase1_binding.binding_trait`): the posterior mean of ``logit π`` from the block's
reports, each weighing ``±ln(ρ/(1−ρ))`` against its cue's measured visibility.

:func:`simulate` writes the tables :func:`fit` reads, from known effects, so the analysis can be
checked end to end before any session (spec §8, step 4).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import numpy as np

from diff.phase1_binding import BASELINE, REPORT_K, binding_trait

ANALYSIS_POLICY: Final = "diff.chain-analysis/v1"
COMPOSITE: Final = (*BASELINE["fascination"], *BASELINE["being_away_from"])  # F + BF, the primary outcome
INTERVAL: Final = (0.05, 0.95)
ON_SCREEN: Final = "on_screen"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


# -- the tables, as the tests need them ----------------------------------------------------------


def restoration(export: Path) -> list[dict[str, Any]]:
    """One row per participant × clip with a second viewing: P4's and P9's F + BF means and the plan."""
    import numpy as np

    scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in _rows(export / "phase1.csv"):
        if row["page"] in ("P4", "P9") and row["item"] in COMPOSITE:
            scores[(row["participant"], row["clip"], row["page"])].append(float(row["answer"]))
    found = []
    for row in _rows(export / "second_captions.csv"):
        before = scores.get((row["participant"], row["clip"], "P4"))
        after = scores.get((row["participant"], row["clip"], "P9"))
        if row.get("assignment") in ("follow", "counter") and before and after:
            found.append(
                {
                    "participant": row["participant"],
                    "unit": row["clip"],
                    "y": float(np.mean(after)),
                    "baseline": float(np.mean(before)),
                    "follow": 0.5 if row["assignment"] == "follow" else -0.5,
                }
            )
    return found


def settled(export: Path, axis: str) -> list[dict[str, Any]]:
    """One row per participant × video for one slider: its settled stop, start and the traits."""
    found = []
    for row in _rows(export / "phase2_sliders.csv"):
        value, b, phi = _number(row["settled"]), _number(row["binding_B"]), _number(row["fascination_Phi"])
        if row["axis"] == axis and value is not None and b is not None and phi is not None:
            found.append(
                {
                    "participant": row["participant"],
                    "unit": row["video_id"],
                    "y": value,
                    "start": int(row["start"]),
                    "B": b,
                    "Phi": phi,
                }
            )
    return found


def traits(export: Path) -> dict[str, dict[str, float]]:
    """Each participant's ``B`` (phase 1) and ``a`` (the source block), where both exist."""
    b: dict[str, float] = {}
    for row in _rows(export / "second_captions.csv"):
        if (value := _number(row.get("binding_B", ""))) is not None:
            b[row["participant"]] = value
    history: dict[str, list[tuple[float, float]]] = defaultdict(list)
    path = export / "source_check.csv"
    for row in _rows(path) if path.exists() else []:
        sign = 1 if row["answer"] == ON_SCREEN else -1
        history[row["participant"]].append((float(row["visibility"]), sign * REPORT_K))
    return {
        participant: {"B": b[participant], "a": binding_trait(answers)["B"]}
        for participant, answers in history.items()
        if participant in b
    }


# -- estimation ----------------------------------------------------------------------------------


def _design(
    rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], dummies: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Intercept, the predictors, then one-hot columns (first level dropped) for each of ``dummies``."""
    import numpy as np

    columns = [np.ones(len(rows))] + [np.array([float(r[p]) for r in rows]) for p in predictors]
    for name in dummies:
        levels = sorted({r[name] for r in rows})
        columns += [np.array([float(r[name] == level) for r in rows]) for level in levels[1:]]
    return np.column_stack(columns), np.array([float(r["y"]) for r in rows])


def _interval(estimate: float, se: float, df: int, n: int) -> dict[str, Any]:
    """The estimate, its central 90% t interval and the one-sided p of the predicted direction (> 0)."""
    from scipy import stats  # type: ignore[import-untyped]

    t = stats.t.ppf(INTERVAL[1], df)
    return {
        "estimate": round(float(estimate), 4),
        "se": round(float(se), 4),
        "interval_90": [round(float(estimate - t * se), 4), round(float(estimate + t * se), 4)],
        "p_one_sided": round(float(stats.t.sf(estimate / se, df)), 4) if se > 0 else None,
        "n_participants": n,
    }


def regression(
    rows: Sequence[Mapping[str, Any]],
    predictors: Sequence[str],
    dummies: Sequence[str],
    *,
    cluster: bool,
) -> dict[str, dict[str, Any]]:
    """Each predictor's least-squares coefficient. With ``cluster`` the standard error is
    cluster-robust by participant (CR1) on N − 1 degrees of freedom, for rows that repeat a
    participant; otherwise it is the ordinary one on N − k, for one row per participant."""
    import numpy as np

    X, y = _design(rows, predictors, dummies)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    participants = sorted({r["participant"] for r in rows})
    n, k = len(rows), int(np.linalg.matrix_rank(X))
    if cluster:
        meat = np.zeros((X.shape[1], X.shape[1]))
        for p in participants:
            mine = np.array([r["participant"] == p for r in rows])
            score = X[mine].T @ residual[mine]
            meat += np.outer(score, score)
        g = len(participants)
        cov = bread @ meat @ bread * (g / (g - 1)) * ((n - 1) / (n - k))
        df = g - 1
    else:
        cov = bread * float(residual @ residual) / (n - k)
        df = n - k
    return {
        name: _interval(beta[j + 1], math.sqrt(max(cov[j + 1, j + 1], 0.0)), df, len(participants))
        for j, name in enumerate(predictors)
    }


def _participant_level(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """For a between-person predictor: each row's outcome net of the start and the video, then each
    participant's mean, beside their traits. The start and video effects are estimated once, on all rows."""
    import numpy as np

    X, y = _design(rows, (), ("start", "unit"))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    groups: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, residual, strict=True):
        groups[row["participant"]].append(float(value))
    first = {row["participant"]: row for row in rows}
    return [
        {"participant": p, "y": float(np.mean(values)), "B": first[p]["B"], "Phi": first[p]["Phi"]}
        for p, values in groups.items()
    ]


def correlation(b: Sequence[float], a: Sequence[float]) -> dict[str, Any]:
    """Pearson's r with its Fisher-z 90% interval and the one-sided p of r > 0."""
    import numpy as np
    from scipy import stats

    n = len(b)
    r = float(np.corrcoef(b, a)[0, 1])
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    half = stats.norm.ppf(INTERVAL[1]) * se
    return {
        "estimate": round(r, 4),
        "interval_90": [round(math.tanh(z - half), 4), round(math.tanh(z + half), 4)],
        "p_one_sided": round(float(stats.norm.sf(z / se)), 4),
        "n_participants": n,
    }


def _shares(export: Path, axis: str) -> dict[str, float]:
    import numpy as np

    rows = [r for r in _rows(export / "phase2_sliders.csv") if r["axis"] == axis and r["share_0"] != ""]
    return {f"share_{k}": round(float(np.mean([float(r[f"share_{k}"]) for r in rows])), 4) for k in range(3)}


def fit(export: Path) -> dict[str, Any]:
    """Every test of the chain from one export directory."""
    report: dict[str, Any] = {"policy": ANALYSIS_POLICY}
    rows = restoration(export)
    if rows:
        found = regression(rows, ("baseline", "follow"), ("participant", "unit"), cluster=True)
        report["H1"] = {"role": "confirmatory", "predicted": "> 0", **found["follow"], "rows": len(rows)}
    for test, axis, trait in (("H2", "context", "B"), ("H3", "texture", "Phi")):
        sliders = settled(export, axis)
        if len({row["participant"] for row in sliders}) < 5:
            continue
        level = _participant_level(sliders)
        report[test] = {
            "role": "estimate",
            "predicted": "> 0",
            **regression(level, (trait,), (), cluster=False)[trait],
            "rows": len(sliders),
            "time_at_stop": _shares(export, axis),
        }
        other = "Phi" if trait == "B" else "B"
        both = regression(level, ("B", "Phi"), (), cluster=False)[other]
        report.setdefault("D", {"role": "estimate", "predicted": "cross terms near 0"})[f"{axis}_{other}"] = {
            key: both[key] for key in ("estimate", "se", "interval_90")
        }
    pairs = traits(export)
    if len(pairs) >= 5:
        report["R"] = {
            "role": "confirmatory",
            "predicted": "> 0",
            **correlation([v["B"] for v in pairs.values()], [v["a"] for v in pairs.values()]),
        }
    return report


# -- simulated exports ---------------------------------------------------------------------------


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate(
    out: Path,
    *,
    participants: int = 48,
    follow: float = 0.3,
    gamma_context: float = 0.3,
    gamma_texture: float = 0.3,
    recovery: float = 0.65,
    seed: int = 20261123,
) -> dict[str, Any]:
    """Write ``phase1.csv``, ``second_captions.csv``, ``phase2_sliders.csv`` and ``source_check.csv``
    with the columns :func:`fit` reads, from known effects (the chain simulation's model):

    - ``follow``: the follow−counter difference on P9's composite, in scale points;
    - ``gamma_*``: the standardised link from each trait to its slider's latent preference;
    - ``recovery``: the correlation of the exported ``B`` with the true trait (.65 with the report).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    clips, videos = [f"c{j:02d}" for j in range(12)], ["viewing-1", "viewing-2", "viewing-3"]
    orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    lam = np.log((p := rng.beta(2, 2, participants)) / (1 - p))
    phi_true = rng.normal(0, 1, participants)
    noise = lam.std() * math.sqrt(1 / recovery**2 - 1)
    b_hat = lam + rng.normal(0, noise, participants)
    clip_effect = rng.normal(0, 0.5, 12)
    phase1, seconds, sliders, block = [], [], [], []
    for i in range(participants):
        who = f"P{i + 1}"
        fascination = np.clip(3 + 0.8 * phi_true[i] + clip_effect + rng.normal(0, 0.8, 12), 0, 6)
        phi = round((float(fascination.mean()) - 3) / 3, 4)
        plan = rng.permutation(["follow"] * 6 + ["counter"] * 6)
        person = rng.normal(0, 0.7)
        for j, clip in enumerate(clips):
            p4 = np.clip(3 + person + clip_effect[j] + rng.normal(0, 0.8), 0, 6)
            p9 = np.clip(p4 + (follow if plan[j] == "follow" else 0.0) + rng.normal(0, 1.0), 0, 6)
            for page, value in (("P4", p4), ("P9", p9)):
                for item in COMPOSITE:
                    phase1.append(
                        {
                            "participant": who,
                            "clip": clip,
                            "page": page,
                            "item": item,
                            "answer": round(value, 3),
                        }
                    )
            seconds.append(
                {
                    "participant": who,
                    "clip": clip,
                    "assignment": plan[j],
                    "binding_B": round(float(b_hat[i]), 4),
                    "fascination_Phi": phi,
                }
            )
        for axis, gamma, trait, order in (
            ("context", gamma_context, lam, orders[i % 6]),
            ("texture", gamma_texture, phi_true, orders[(i + i // 6) % 6]),
        ):
            z = (trait[i] - trait.mean()) / trait.std()
            preference = gamma * z + math.sqrt(1 - gamma**2) * rng.normal()
            for v, video in enumerate(videos):
                start = order[v]
                stop = int(np.digitize(preference + rng.normal(0, 0.5) + (v - 1) * 0.2, [-0.5, 0.5]))
                moved_at = min(rng.exponential(45), 300) if rng.random() < 0.7 else 300.0
                held = [0.0, 0.0, 0.0]
                before = min(max(moved_at - 30, 0), 270) / 270
                held[start] += before
                held[stop] += 1 - before
                sliders.append(
                    {
                        "participant": who,
                        "video_id": video,
                        "axis": axis,
                        "start": start,
                        "settled": round(sum(k * h for k, h in enumerate(held)), 6),
                        **{f"share_{k}": round(h, 6) for k, h in enumerate(held)},
                        "binding_B": round(float(b_hat[i]), 4),
                        "fascination_Phi": phi,
                    }
                )
        drifted = lam[i] + rng.normal(0, 0.5)
        for q in np.r_[rng.uniform(0.55, 0.95, 10), rng.uniform(0.05, 0.45, 11)]:
            bound = rng.random() < 1 / (1 + math.exp(-(drifted + math.log(q / (1 - q)))))
            said = bound if rng.random() < 0.8 else not bound
            block.append(
                {
                    "participant": who,
                    "visibility": round(float(q), 4),
                    "answer": "on_screen" if said else "off_screen",
                }
            )
    for name, rows in (
        ("phase1", phase1),
        ("second_captions", seconds),
        ("phase2_sliders", sliders),
        ("source_check", block),
    ):
        _write(out / f"{name}.csv", rows)
    return {
        "participants": participants,
        "follow": follow,
        "gamma_context": gamma_context,
        "gamma_texture": gamma_texture,
        "recovery": recovery,
        "seed": seed,
    }
