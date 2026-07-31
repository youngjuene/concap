"""`dpo stage`: the pipeline stage registry and lineage."""

from __future__ import annotations

import argparse

from dpo.cli._shared import _emit
from dpo.pipeline.stages import STAGES, lineage_edges


def _stage_list(arguments: argparse.Namespace) -> int:
    del arguments
    _emit(
        {
            "status": "ok",
            "stages": {
                name: {
                    "inputs": list(stage.input_artifact_types),
                    "outputs": list(stage.output_artifact_types),
                    "description": stage.description,
                }
                for name, stage in sorted(STAGES.items())
            },
            "lineage": [list(edge) for edge in lineage_edges()],
        }
    )
    return 0
