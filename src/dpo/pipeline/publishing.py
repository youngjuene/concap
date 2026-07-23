"""The one artifact-publishing home for pipeline stages.

Every producer (the CLI handlers and the offline canary) publishes through
``ArtifactPublisher`` so request construction cannot fork: one payload shape,
one lock identity, and one cache-keying rule. Each publish is tagged with its
pipeline stage; the artifact's ``contract_id`` becomes the semantic hash of
exactly the contract slice that stage declares in the registry
(``dpo.pipeline.stages``), so tweaking a contract section only invalidates
the stages that read it. A ``slice_override`` narrows further — the matrix
runner keys each training cell on its resolved variant hyperparameters, so
extending a sweep axis leaves sibling variants' identities untouched.
"""

from __future__ import annotations

from collections.abc import Mapping

from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ArtifactStore, ParentEdge, RequestSpec
from dpo.core.identity import canonical_bytes, semantic_hash
from dpo.pipeline.stages import FULL_CONTRACT, contract_slice_hash, stage


class ArtifactPublisher:
    def __init__(self, store: ArtifactStore, contract: StudyContract, lock_id: str) -> None:
        self.store = store
        self.contract = contract
        self.lock_id = lock_id

    def contract_id_for(self, stage_name: str, slice_override: Mapping[str, object] | None) -> str:
        if slice_override is not None:
            return semantic_hash(dict(slice_override))
        if stage_name == "*":
            return self.contract.contract_hash
        sections = stage(stage_name).contract_sections
        if sections == FULL_CONTRACT:
            return self.contract.contract_hash
        return contract_slice_hash(self.contract.raw, self.contract.contract_hash, sections)

    def publish(
        self,
        artifact_type: str,
        payload_document: object,
        *,
        stage: str,
        slice_override: Mapping[str, object] | None = None,
        parents: tuple[ParentEdge, ...] = (),
        parameters: Mapping[str, object] | None = None,
        seed: int = 0,
        row_count: int = 0,
        clips: set[str] | None = None,
        role_exposure: set[str] | None = None,
        fit_exposure: set[str] | None = None,
        selection_exposure: set[str] | None = None,
        test_exposure: set[str] | None = None,
        attributes: Mapping[str, object] | None = None,
        purpose: str = "research",
    ) -> str:
        request = RequestSpec(
            artifact_type=artifact_type,
            parents=parents,
            contract_id=self.contract_id_for(stage, slice_override),
            model_pins={},
            prompt_pins={},
            parameters=dict(parameters or {"operation": artifact_type}),
            seed=seed,
            lock_id=self.lock_id,
        )
        manifest = self.store.publish(
            request,
            canonical_bytes(payload_document),
            row_count=row_count,
            clips=clips,
            role_exposure=role_exposure,
            fit_exposure=fit_exposure,
            selection_exposure=selection_exposure,
            test_exposure=test_exposure,
            attributes=attributes,
            purpose=purpose,
        )
        return manifest.artifact_id
