"""Model and processor abstraction with hard modality isolation.

`logprob` is the one shared completion-only sequence log-probability
implementation every trainer and evaluator uses. `base` defines the
modality-isolated media/batch contracts and the adapter protocol;
`visual_media`/`audio_media` are the per-track processor boundaries;
`tiny` is the deterministic CPU backend that makes golden training-step tests
possible without checkpoints or GPUs; `gemma4/` is the pinned real-model
backend (adapter, backend config, tokenization safety).
"""
