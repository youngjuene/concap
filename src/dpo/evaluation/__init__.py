"""Per-model measurement, applied identically to every experiment in the matrix.

``preference_accuracy`` scores held-out pairs from the model's own
log-probabilities; ``congruency`` selects the measured audiovisual ladder the
human study is built from; ``ablation`` is the gray-video control; and
``caption_reuse`` is the memorization gate — a validation caption that
byte-matches a frozen training candidate is a wiring bug or a memorized one.

Inferential statistics over what these produce live in ``dpo.analysis``, and
human preference remains the primary criterion.
"""
