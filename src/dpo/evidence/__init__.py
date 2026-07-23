"""Modality-isolated evidence layer and the per-clip claim ledger.

The evidence layer never defines a unique reference caption. It records, per
track, whether individual caption claims are supported by the target modality:
`records` defines the typed evidence items, `visual`/`audio` own the per-track
schemas and parsers, `claim_ledger` owns claims, statuses, and human audits,
and `providers` is the fail-closed pinned boundary for external evidence
models (SED, detectors, VLMs). Automated outputs retain provenance and are
never silently promoted to human-verified ground truth.
"""
