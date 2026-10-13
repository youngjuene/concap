"""Trida-7B caption pipeline behind the regen spreadsheet flow (docs/v4-diff/plan.md).

Caption text is written under fixed and instrumented settings by a generator
that runs in its own locked environment (``envs/diff-generator``): Trida-7B
(the researcher's model since 2026-10-12, replacing Dream), with
Qwen2.5-7B-Instruct as the review-stage baseline. Phase-1 captions are written
offline; phase-2 captions are rewritten live while a participant watches, by
supervised GPU workers the participant server talks to but never imports.

This package imports nothing when it is imported. The generator environment
must never pull in the web stack, and the server environment must never pull in
torch; each command imports what it needs inside ``python -m diff``.
"""
