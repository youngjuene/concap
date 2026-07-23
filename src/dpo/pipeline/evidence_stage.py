"""Shared claims-stage implementation: publishing the audited claim ledger.

Every producer records a finished, human-audited claim ledger through this
function, so the payload document, the parsed-evidence parent edge, and the
exposure tags have exactly one home. Constructing the ledger — evidence
extraction, claim proposal, and the audit decisions — happens upstream: the
offline canary builds it from fixtures, a live runner from real provider
artifacts and real auditor decisions.
"""

from __future__ import annotations

from dpo.core.artifacts import ParentEdge
from dpo.evidence.claim_ledger import ClaimLedger
from dpo.pipeline.publishing import ArtifactPublisher


def publish_claim_ledger(
    publisher: ArtifactPublisher,
    *,
    ledger: ClaimLedger,
    parsed_artifact_id: str,
    role: str,
) -> str:
    """Publish one clip's audited claim ledger against its parsed-evidence parent."""
    return publisher.publish(
        "dpo.claim-ledger/v1",
        ledger.document(),
        parents=(ParentEdge(parsed_artifact_id, "parsed-evidence"),),
        stage="claims",
        parameters={"operation": "claims", "clip_id": ledger.clip_id, "track": ledger.track},
        row_count=len(ledger.claims),
        clips={ledger.clip_id},
        role_exposure={role},
    )
