"""Human preference collection records, aggregation, and reliability.

Raw annotations are append-only scientific data: they are validated on entry,
never overwritten by aggregates, and every derived view (SFT, strict pairs,
metadata-rich pairs, noise calibration) is computed from them and versioned
separately. Choices are recorded against the displayed order and resolved to
canonical candidate identity here, exactly once.
"""
