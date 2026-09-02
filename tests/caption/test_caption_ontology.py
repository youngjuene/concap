"""The AudioSet ontology and the tidy CSV, read through the shared loader.

``dpo.caption.ontology`` is shared by both instruments and outlives whichever
is archived, so its tests live beside it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dpo.caption.ontology import OntologyError, load_tags


class TestTags:
    """The tidy CSV and the ontology, read through the shared loader."""

    def _write(self, tmp_path: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
        tidy = tmp_path / "tidy.csv"
        with tidy.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["file_index", "final_labels", "top_level_parent_name"]
            )
            writer.writeheader()
            writer.writerows(rows)
        ontology = tmp_path / "ontology.json"
        ontology.write_text(
            json.dumps(
                [
                    {"id": "things", "name": "Sounds of things", "child_ids": ["siren", "traffic"]},
                    {"id": "animal", "name": "Animal", "child_ids": ["bird"]},
                    {"id": "siren", "name": "Siren", "child_ids": []},
                    {"id": "traffic", "name": "Traffic noise", "child_ids": []},
                    {"id": "bird", "name": "Bird", "child_ids": []},
                ]
            ),
            encoding="utf-8",
        )
        return tidy, ontology

    def test_repeated_rows_do_not_multiply_a_count(self, tmp_path: Path) -> None:
        tidy, ontology = self._write(
            tmp_path,
            [
                {
                    "file_index": "c",
                    "final_labels": "Siren|Siren|Bird",
                    "top_level_parent_name": "Sounds of things:2|Animal:1",
                }
            ]
            * 3,
        )
        tags = load_tags(tidy, ontology)
        assert dict(tags["c"]["counts"]) == {"Siren": 2, "Bird": 1}
        assert tags["c"]["parents"] == {"Siren": "Sounds of things", "Bird": "Animal"}

    def test_rows_that_disagree_are_a_data_problem_not_an_average(self, tmp_path: Path) -> None:
        tidy, ontology = self._write(
            tmp_path,
            [
                {"file_index": "c", "final_labels": "Siren", "top_level_parent_name": "Sounds of things:1"},
                {"file_index": "c", "final_labels": "Bird", "top_level_parent_name": "Animal:1"},
            ],
        )
        with pytest.raises(OntologyError, match="disagree"):
            load_tags(tidy, ontology)
