"""Dream-7B caption pipeline behind the regen spreadsheet flow (docs/v4-diff/plan.md).

Caption text is written offline, under fixed and instrumented settings, by a
generator that runs in its own locked environment (``envs/diff-generator``):
Dream-v0-Instruct-7B, with Qwen2.5-7B-Instruct as the review-stage baseline.
Its outputs are checked, reviewed blind and frozen into a hashed stimulus
bundle; the participant server only ever reads that bundle and never calls a
model. Nothing beyond G0 (Korean feasibility) runs until G0 passes.

This package imports nothing when it is imported. The generator environment
must never pull in the web stack, and the server environment must never pull in
torch; each command imports what it needs inside ``python -m diff``.
"""
