"""Dependency-free tests executing the production exposure validation branch.

Run directly with Python. Browser integration and model inference are outside
this test; the real numeric guard and API branch are compiled from their AST.
"""

import ast
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
schema = ast.parse((ROOT / "study_schema.py").read_text())
api = ast.parse((ROOT / "study_api.py").read_text())
number = next(node for node in schema.body if isinstance(node, ast.FunctionDef) and node.name == "number")
branch = next(
    node
    for node in ast.walk(api)
    if isinstance(node, ast.If)
    and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name)
    and node.test.left.id == "action"
    and any(isinstance(value, ast.Constant) and value.value == "exposures" for value in node.test.comparators)
)
namespace = {"Any": object, "math": math, "json": json}
exec(compile(ast.Module(body=[number], type_ignores=[]), str(ROOT / "study_schema.py"), "exec"), namespace)
validate = compile(ast.Module(body=branch.body, type_ignores=[]), str(ROOT / "study_api.py"), "exec")


class DisplayValidation(unittest.TestCase):
    def setUp(self):
        self.state = {"coverage": [[0, 1000]], "position_ms": 1000, "epoch": 0}
        self.video = {
            "id": "v",
            "duration_ms": 10000,
            "cues": [{"start_ms": 0, "end_ms": 5000}, {"start_ms": 5000, "end_ms": 10000}],
        }
        self.entry = {
            "id": "e",
            "cue": 0,
            "start_ms": 4900,
            "end_ms": 5215,
            "cue_end": 5000,
            "timing": "display-v1",
            "episode_id": "episode-a",
        }

    def submit(self, entry):
        env = {**namespace, "state": self.state, "video": self.video, "data": {"entries": [entry]}}
        exec(validate, env)
        return self.state["_exposure_batch"][0]

    def test_actual_replacement_can_cross_nominal_end_without_awarding_playback(self):
        try:
            saved = self.submit(self.entry)
        except ValueError as error:
            self.fail(f"A valid observed replacement was rejected: {error}")
        self.assertEqual(saved["end_ms"], 5215)
        self.assertEqual(saved["cue_end"], 5000)
        self.assertEqual(self.state["coverage"], [[0, 1000]])
        self.assertEqual(self.state["position_ms"], 1000)
        self.assertEqual(self.state["epoch"], 0)

    def test_nominal_activation_and_video_bounds_are_still_enforced(self):
        for patch in (
            {"start_ms": 5200},
            {"start_ms": -1},
            {"end_ms": 10001},
            {"end_ms": 4800},
            {"end_ms": float("nan")},
            {"cue": 3},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.submit({**self.entry, **patch})

    def test_display_contract_requires_matching_cue_metadata_and_episode(self):
        for patch in (
            {"cue_end": 6000},
            {"cue_end": None},
            {"episode_id": ""},
            {"episode_id": None},
            {"timing": "future-unknown"},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.submit({**self.entry, "start_ms": 100, "end_ms": 1000, **patch})

    def test_legacy_intervals_keep_their_original_cue_bounds(self):
        legacy = {"id": "old", "cue": 0, "start_ms": 100, "end_ms": 1000}
        self.assertEqual(self.submit(legacy)["end_ms"], 1000)
        with self.assertRaises(ValueError):
            self.submit({**legacy, "end_ms": 5215})


if __name__ == "__main__":
    unittest.main(verbosity=2)
