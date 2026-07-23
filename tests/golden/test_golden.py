"""Golden tests (PRD section 32.4): one training step per trained objective on a
small synthetic fixture, loss/margin snapshots, checkpoint save/load,
deterministic generation, and artifact-manifest regeneration.

Regenerate snapshots with: uv run python scripts/regenerate_golden.py
Snapshots must only change with an intentional, reviewed objective change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ArtifactStore, RequestSpec
from dpo.core.identity import canonical_bytes, sha256_bytes
from dpo.models.base import CompletionBatch
from dpo.models.tiny import TinyAdapter, synthetic_media
from tests.conftest import PreferenceWorld
from tests.golden.harness import golden_training_steps

GOLDEN_PATH = Path(__file__).parent / "golden_values.json"


def _load_golden() -> dict[str, dict[str, float]]:
    if not GOLDEN_PATH.is_file():
        pytest.fail("golden_values.json is missing; run `uv run python scripts/regenerate_golden.py`")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_one_training_step_per_objective_matches_the_snapshot(
    world: PreferenceWorld, contract: StudyContract
) -> None:
    del contract
    expected = _load_golden()
    actual = golden_training_steps(world)
    assert set(actual) == set(expected), "experiment coverage changed; regenerate deliberately"
    for experiment_id, values in actual.items():
        for key, value in values.items():
            assert value == pytest.approx(expected[experiment_id][key], abs=1e-6), (
                f"{experiment_id}.{key} drifted from the golden snapshot; if this change is"
                " intentional, regenerate with scripts/regenerate_golden.py and review the diff"
            )


def test_checkpoint_save_and_load_roundtrip(world: PreferenceWorld, tmp_path: Path) -> None:
    adapter = TinyAdapter(track="visual", prompt=world.contract.tracks["visual"].prompt, media_dim=16, seed=3)
    adapter.save(tmp_path / "checkpoint")
    restored = TinyAdapter.load(tmp_path / "checkpoint")
    assert restored.state_signature() == adapter.state_signature()
    media = synthetic_media("visual", ["clip-000"], media_dim=16)
    completions = CompletionBatch(texts=("a caption to score",))
    with torch.no_grad():
        original = adapter.completion_logps(media, completions)
        reloaded = restored.completion_logps(media, completions)
    assert torch.allclose(original, reloaded, atol=1e-6)


def test_generation_is_deterministic(world: PreferenceWorld) -> None:
    adapter = TinyAdapter(track="audio", prompt=world.contract.tracks["audio"].prompt, media_dim=16, seed=3)
    media = synthetic_media("audio", ["clip-000", "clip-001"], media_dim=16)
    first = adapter.generate(media, temperature=0.0, top_p=1.0, max_new_tokens=12, seed=9)
    second = adapter.generate(media, temperature=0.0, top_p=1.0, max_new_tokens=12, seed=9)
    assert first == second


def test_artifact_manifest_regeneration_is_bit_identical(tmp_path: Path) -> None:
    request = RequestSpec(
        artifact_type="dpo.fixture/v1",
        parents=(),
        contract_id=sha256_bytes(b"golden-contract"),
        model_pins={},
        prompt_pins={},
        parameters={"operation": "golden"},
        seed=1,
        lock_id=sha256_bytes(b"golden-lock"),
    )
    payload = canonical_bytes({"rows": [1, 2, 3]})
    first_store = ArtifactStore.create(tmp_path / "store-a")
    second_store = ArtifactStore.create(tmp_path / "store-b")
    first = first_store.publish(request, payload)
    second = second_store.publish(request, payload)
    assert first.artifact_id == second.artifact_id
    assert first_store.manifest_path(first.artifact_id).read_bytes() == (
        second_store.manifest_path(second.artifact_id).read_bytes()
    )
