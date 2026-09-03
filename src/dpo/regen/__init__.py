"""The regeneration instrument of ``docs/v3-regen/``.

A participant watches one segment of a street scene under a prepared sound
caption track, answers the ART sub-factors, marks what they saw on the
five-second still and picks out what they heard from the separated stems, and
then watches the other segment under a caption track regenerated from exactly
those two reports. The same ART items are asked again afterwards, beside the
caption measures and the PRSS.

The contrast is provenance and nothing else: both tracks have the same number
of cues at the same moments, validated against one calibration
(:mod:`dpo.regen.config`), stored under one schema (:mod:`dpo.regen.captions`),
and logged into one row shape (:mod:`dpo.regen.log`). Which segment carries
which condition alternates by sequence number
(:mod:`dpo.regen.assignment`), so segment and condition are not the same
variable.

Throughout this package a bare ``§N`` cites ``docs/v3-regen/spec-behavior.md``;
every module says so at the end of its docstring.

Separate from :mod:`dpo.session` and :mod:`dpo.console`, which build the
instruments of ``docs/v1-session/`` and ``docs/v2-console/`` against
specifications this one does not share a control surface, a screen sequence or
a measure with. The three share :mod:`dpo.caption` and nothing else, so
whichever are not adopted can be deleted whole.
"""

from dpo.regen.assignment import PREPARED, REGENERATED, Assignment, assign
from dpo.regen.captions import Cue, TrackError, validate_generated, validate_track
from dpo.regen.config import Calibration, Configuration, Scale
from dpo.regen.document import (
    REGEN_SCHEMA,
    RegenDocumentError,
    load_regen_document,
    validate_regen_document,
)
from dpo.regen.items import ItemsError, ItemSet, load_items
from dpo.regen.regeneration import Regeneration, Report, regenerate

__all__ = [
    "PREPARED",
    "REGENERATED",
    "REGEN_SCHEMA",
    "Assignment",
    "Calibration",
    "Configuration",
    "Cue",
    "ItemSet",
    "ItemsError",
    "RegenDocumentError",
    "Regeneration",
    "Report",
    "Scale",
    "TrackError",
    "assign",
    "load_items",
    "load_regen_document",
    "regenerate",
    "validate_generated",
    "validate_regen_document",
    "validate_track",
]
