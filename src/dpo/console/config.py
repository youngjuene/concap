"""The versioned configuration artifact, and the hash every result carries.

Every adjustable quantity in the instrument is sorted by one criterion (§10).
If varying it within a session is the point of the session, it is an
experimental variable. If varying it between sessions would invalidate
comparison, it is a method constant. If it must be tuned to the corpus once and
then hold still, it is a calibration.

    | Quantity                        | Tier                  | Cadence        |
    |---------------------------------|-----------------------|----------------|
    | saturation applied to ordering  | method constant       | per study      |
    | w_g = c_g · e_g                 | method constant       | per study      |
    | IoU grouping threshold          | method constant       | per study      |
    | register from p_g               | method constant       | per study      |
    | r0                              | calibration           | per corpus     |
    | band cut points                 | calibration           | per corpus     |
    | θ, minimum duration, boundaries | calibration           | per corpus     |
    | admitted set, α, grain          | experimental variable | per shot, run  |

The first two tiers freeze into the artifact this module builds, and its hash
stamps every caption, every endpoint, and every log entry, so any result is
reproducible from its stamp. The third tier is what a participant moves and
never appears here.

The method constants are recorded as *declarations*, not as executable
configuration. Nothing reads them back to decide what to compute — the formulas
live in :mod:`dpo.console.quantities` — but they are hashed, so changing a
formula without changing the declaration beside it leaves two studies sharing
one stamp, which is the failure the stamp exists to prevent. Change both.

Section numbers cite ``spec-system.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

CONFIG_SCHEMA = "dpo.caption-console-config/v1"
HASH_LENGTH = 12

# Declared, hashed, never dispatched on. Each line states what
# ``dpo.console.quantities`` does, so a study's stamp changes when the method
# changes. §13 records the first as a revisable theory choice rather than a
# fact: ordering inherits the saturating visibility on the assumption that
# visual salience saturates with area, and linear coherence is equally
# consistent with the rest of the specification.
METHOD_CONSTANTS: Mapping[str, Any] = {
    "ordering_saturates": True,
    "salience": "w_g = c_g * e_g",
    "visibility": "v_g = r_g / (r_g + r0)",
    "normalization": "sum over the full source set, applied before filtering",
    "score": "score_g = alpha * w_hat_g + (1 - alpha) * v_hat_g",
    "register_from": "p_g",
    "grouping": "labels merged above the IoU threshold; area once per group, energy summed",
}


class ConfigError(ValueError):
    """The configuration is not one a study could be run under."""


@dataclass(frozen=True)
class Band:
    """One cut point on a computed quantity, and the phrase above it.

    Bands turn ``v_g`` into a visibility band and ``p_g`` into a temporal
    register (§8). Both are per-corpus calibrations (§10) and both are read
    aloud as words: no measured value is ever shown to a participant as a
    number, so the phrase — not the cut point — is what reaches a screen.
    """

    at_least: float
    phrase: str


# English placeholders, in the interface's voice. §13 leaves the Korean wording
# of the criterion sentence a pending construct decision, and these are in the
# same position: they are copy, they are calibrated per corpus, and they are
# hashed so a study cannot silently change how a source reads between sessions.
DEFAULT_VISIBILITY_BANDS: tuple[Band, ...] = (
    Band(0.66, "fills much of the frame"),
    Band(0.33, "in frame"),
    Band(0.05, "small in the frame"),
    Band(0.0, "out of frame"),
)
DEFAULT_REGISTERS: tuple[Band, ...] = (
    Band(0.85, "throughout"),
    Band(0.5, "for much of the moment"),
    Band(0.15, "in passing"),
    Band(0.0, "once, briefly"),
)


def phrase_for(value: float, bands: Sequence[Band]) -> str:
    """The phrase of the highest band the value reaches."""
    for band in sorted(bands, key=lambda band: -band.at_least):
        if value >= band.at_least:
            return band.phrase
    raise ConfigError(f"no band covers {value}; the lowest band must start at zero")


@dataclass(frozen=True)
class Calibration:
    """What is tuned to the corpus once and then holds still (§10).

    ``half_saturation`` is ``r0``: the area a source must occupy to count as
    half visible. It is an empirical claim about pedestrian-scale visual
    salience, set through the admin calibration protocol and then frozen, not
    a number anyone tunes between participants.

    ``cut_threshold`` is ``θ`` and ``minimum_shot_ms`` the floor on shot
    length. Both belong to audio reliability rather than to any visual
    criterion (§5.2): audio windows align to visual cuts, so lowering θ
    shortens windows and degrades every ``w_g`` measured in them. Measure them
    against audio before fixing θ.
    """

    half_saturation: float = 0.02
    iou_threshold: float = 0.5
    min_regime: float = 0.02
    cut_threshold: float = 0.25
    minimum_shot_ms: int = 2000
    stride_ms: int = 250
    visibility_bands: tuple[Band, ...] = DEFAULT_VISIBILITY_BANDS
    registers: tuple[Band, ...] = DEFAULT_REGISTERS

    def __post_init__(self) -> None:
        if not 0.0 < self.half_saturation < 1.0:
            raise ConfigError("r0 must lie strictly between 0 and 1")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ConfigError("the IoU threshold must lie in [0, 1]")
        if not 0.0 <= self.min_regime < 1.0:
            raise ConfigError("the minimum regime width must lie in [0, 1)")
        if not 0.0 < self.cut_threshold <= 1.0:
            raise ConfigError("θ must lie in (0, 1]")
        if self.minimum_shot_ms < 1 or self.stride_ms < 1:
            raise ConfigError("the shot floor and the stride are positive durations")
        for name, bands in (("visibility_bands", self.visibility_bands), ("registers", self.registers)):
            if not bands:
                raise ConfigError(f"{name} must not be empty")
            if min(band.at_least for band in bands) > 0.0:
                raise ConfigError(f"{name} must have a band starting at zero")


@dataclass(frozen=True)
class Configuration:
    """The frozen artifact, and its hash.

    ``study_id`` and ``corpus_id`` are in the hash on purpose. Two corpora
    calibrated to identical numbers are still two corpora, and a result should
    not claim to be reproducible from the other's footage.

    ``provisional_salience`` is in the hash for the same reason, and it has no
    default so that nothing can build a configuration without saying. A dry
    run fills ``c_g`` and ``e_g`` from tag multiplicity (``console preprocess
    --provisional-salience``) and the audio axis it yields is detectional, the
    failure §5.1 names; a result computed on it must never share a stamp with
    one computed on an acoustic measurement, and a document must say which it
    is at serve time, not only in the manifest it was scaffolded from.
    """

    study_id: str
    corpus_id: str
    provisional_salience: bool = field(kw_only=True)
    calibration: Calibration = field(default_factory=Calibration)

    def artifact(self) -> dict[str, Any]:
        """The hashable document: everything frozen, nothing computed."""
        return {
            "schema": CONFIG_SCHEMA,
            "study_id": self.study_id,
            "corpus_id": self.corpus_id,
            "provisional_salience": self.provisional_salience,
            "method_constants": dict(METHOD_CONSTANTS),
            "calibration": asdict(self.calibration),
        }

    @property
    def hash(self) -> str:
        """The stamp: sha256 over the canonical artifact, truncated.

        Canonical means sorted keys and no incidental whitespace, so the stamp
        depends on the values and not on how the file was written.
        """
        payload = json.dumps(self.artifact(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LENGTH]

    def stamped(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """``document`` with the stamp on it. Every result leaves through here."""
        return {**document, "config_hash": self.hash}

    def visibility_phrase(self, value: float) -> str:
        return phrase_for(value, self.calibration.visibility_bands)

    def register_phrase(self, value: float) -> str:
        return phrase_for(value, self.calibration.registers)


def load_configuration(raw: Mapping[str, Any]) -> Configuration:
    """Rebuild a configuration from an artifact, refusing a changed method.

    A stored artifact whose ``method_constants`` differ from this build's was
    written by a different instrument. Loading it would produce results stamped
    with a hash that no longer describes how they were computed, so it is an
    error rather than a warning.
    """
    if raw.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"not a {CONFIG_SCHEMA} artifact")
    declared = raw.get("method_constants")
    if declared != dict(METHOD_CONSTANTS):
        raise ConfigError(
            "the artifact's method constants are not this build's; "
            "results computed here could not carry its hash honestly"
        )
    calibration = dict(raw.get("calibration") or {})
    try:
        for name in ("visibility_bands", "registers"):
            if name in calibration:
                calibration[name] = tuple(Band(**band) for band in calibration[name])
        provisional = raw["provisional_salience"]
        if not isinstance(provisional, bool):
            raise ConfigError("provisional_salience must be true or false")
        return Configuration(
            study_id=str(raw["study_id"]),
            corpus_id=str(raw["corpus_id"]),
            provisional_salience=provisional,
            calibration=Calibration(**calibration),
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"the artifact is missing or misnames a field: {exc}") from exc
