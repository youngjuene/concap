"""Immutable splits and the derived training views over one frozen preference set.

`split` computes the group-level split manifest before any candidate, pair, or
annotation exists. `derive_sft` and `derive_pairs` regenerate every training
view (D_sft, D_pair_strict, D_pair_all) from the frozen pool plus raw
annotations. `noise` owns natural-noise calibration and the shared synthetic
flip manifest, `weighting` owns clip-level weighting, and `leakage_audit`
verifies the group/split/leakage rules end to end.
"""
