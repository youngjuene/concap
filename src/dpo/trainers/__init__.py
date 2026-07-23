"""Training loops shared by every trained experiment.

One preference trainer serves every preference objective through the
registry; one SFT trainer serves SFT. Both compute completion
log-probabilities through the shared implementation in
``dpo.models.logprob`` (via the model adapters), freeze references
structurally, and emit the full shared diagnostics schema per step.
"""
