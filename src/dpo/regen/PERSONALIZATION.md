# Phase 1 ratings → Phase 2 system prompt

New calibration handoffs include Phase 1 questionnaire ratings in the captioner's
system prompt. Long videos are still watched once, with live acoustic-detail and
source/scene-detail controls.

## Data flow

`Continuation.handoff` reads server-recorded responses and their matching viewings.
`questionnaire_profile` in `study_personalization.py` validates and freezes the
item wording, item provenance/digest, response scale, raw answers, and viewing scope.
`compile_profile` in `study_schema.py` adds this snapshot and its writing guidance
to the personalized template. The scheduler combines that template with current
control levels and cue evidence. `study_worker.py` sends the resulting instruction
to Gemma as the actual system message alongside the audio excerpt.

Every submitted rating is represented in the prompt, including custom items.
Changing a rating changes the template and the instruction-based caption cache
key. Clip filenames and research hashes remain in the stored profile rather than
being shown to the model as context for the current scene.

## Versioned writing policy

Policy identifier: `phase1-caption-guidance/v1`. Profile version: `2`.
`personalization_source=phase1_questionnaire` identifies the new behavior.

Ratings are normalized as `(rating - 1) / (points - 1)` and inverted for items
marked `reverse`. Recognized items within each group are equally weighted across
viewings. A normalized score of 0.5 is the scale midpoint; the thresholds below
are explicit writing heuristics, not validated psychometric cutoffs.

| Measure | Prompt guidance |
| --- | --- |
| Caption credibility | Below midpoint: prefer conservative labels for uncertain sources. Otherwise: retain clear, supported source labels within the current detail setting. |
| Reading distraction | Above midpoint for the negatively worded distraction item: use compact syntax and omit parenthetical/redundant narration. Otherwise: allow concise descriptive phrasing within requested detail. |
| Caption restfulness | Lower ratings favor neutral, unobtrusive event wording; higher ratings retain scene-focused phrasing without calling the current scene restful. |
| ART original/updated difference | A negative paired difference favors less narration beyond requested details. Pair only corresponding viewings of the same clip, and do not infer causation. |
| Overall PRSS | Lower ratings favor direct, unevaluative language; higher ratings favor continuity of supported ongoing events. Neither permits invented calming sounds or suppressed warnings. |

The complete item definitions and ratings accompany these rules as data, not
instructions. Built-in semantic rules apply only when the item's full wording and
reverse flag match the packaged definition. Custom wording is retained with its
actual scale and marked in `unmapped_item_ids`; built-in meanings are not assigned
solely from a reused item ID. A changed item digest, invalid rating, missing answer,
duplicate response, or mismatched viewing prevents reinterpretation at handoff.

## Controls and compatibility

The live axes continue to determine acoustic descriptor budget and source/scene
specificity; their instructions take precedence over inferred writing guidance.
Ratings do not manufacture an explicit acoustic/source-detail preference:
`preferences={}` and `preference_source=not_collected` remain accurate, and the
axes start at 50%. Personalization is carried separately in `profile.questionnaire`.

Already-frozen Phase 2 profiles remain unchanged on resume. The new behavior
applies when Phase 1 is newly handed over. Historical different-clip calibration
can contribute its ratings, but its ART responses are never paired as a same-clip
effect. Standalone preference-only calibration retains its existing behavior.

## Verification

- `tests/test_study_personalization.py`: rating sensitivity, reverse coding,
  five/seven-point scales, different-clip exclusion, custom wording, invalid
  records, immutable profiles, and live-control precedence.
- Mounted API handoff → scheduling → actual worker dispatch is tested with a
  recording model adapter; the system message equals the personalized instruction.
- **504 Regen tests pass**, including existing continuation and legacy tests.
- Six real-GPU requests on GPU 1 used synthetic low/high ratings, English/Korean
  question wording, real study audio, and an additional control adjustment.
  All six succeeded without fallback and recorded the expected system prompt.
  Cold request: 16,952 ms; warm requests: 932–1,088 ms.
- The sampled low/high rating profiles produced different captions at identical
  control settings in both languages. This is a bounded integration check, not
  proof that every rating change produces distinct prose or improves restoration.

GPU evidence: `/tmp/regen-questionnaire-personalization/gpu-results.json`.
Changed implementation files: `study_personalization.py`, `study_schema.py`, and
`continuation.py`; supporting tests and documentation describe the policy.

## Activation

Activated on **2026-09-14 at 00:36:10 UTC** on `127.0.0.1:8779`, GPU 1,
process `3401437`. The verified idle service was restarted with its existing
model, media, access, and output flags. Aggregate fingerprints confirm existing
calibration files and viewing database records were preserved. No additional
caption-cache rotation was needed; the Phase 2 cache keys include the complete
personalized instruction. Inference workers still load lazily on first use.

Activation and record-preservation evidence:
`/tmp/regen-questionnaire-personalization/activation.json` and
`record-fingerprints-{before,after}.json`. This update's GPU check was bounded to
six generation requests; the prior full timed playback evidence remains in
`FRESH_GPU_QA.md` and is not presented as a new full playback run.

Future policy changes should receive a new policy identifier. Do not relabel or
rebuild a participant's frozen profile during an active Phase 2 session.
