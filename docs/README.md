# Documentation map

Three directories, because the repository builds two participant instruments
against two specifications that disagree, on top of one shared stack. Anything
naming a *version* belongs to exactly one instrument and dies with it; anything
in `shared/` outlives whichever is archived.

| Directory | Covers | Code |
|---|---|---|
| [`shared/`](shared/) | the pipeline and the study both instruments sit inside, and the caption machinery they both call | `dpo.caption`, and the training/annotation/study stack |
| [`v1-session/`](v1-session/) | the **skeleton** instrument: the ordered list of what a caption could mention is the only control surface | `dpo.session`, `dpo session …`, port 8777 |
| [`v2-console/`](v2-console/) | the **console** instrument: three controls drive a read-only numbered skeleton | `dpo.console`, `dpo console …`, port 8778 |

Within a version directory the filename says the role:

* `runbook.md` — how an operator runs it: author, validate, serve, and what
  lands on disk. Start here.
* `spec-*.md` — the specification it is built to, as written. Not a summary of
  the code; the code is measured against these.
* anything else — supporting material for that instrument alone.

## shared/

| File | What it is |
|---|---|
| [`caption-stack.md`](shared/caption-stack.md) | `dpo.caption`: the writers, the cache, the budget, and the seam that keeps the two instruments detachable |
| [`proposal.md`](shared/proposal.md) | the research plan and work programme |
| [`pipeline.md`](shared/pipeline.md) | pipeline invariants, gates, and claim limits |
| [`study-runbook.md`](shared/study-runbook.md) | the congruency-slider study (`dpo study serve`), with real artifact ids |
| [`TODO.md`](shared/TODO.md) | what is deliberately deferred, and whose call each one is |

## v1-session/

| File | What it is |
|---|---|
| [`runbook.md`](v1-session/runbook.md) | operator steps for `dpo session` |
| [`spec-behavior.md`](v1-session/spec-behavior.md) | what to build: screens, components, gestures, copy |
| [`spec-identity.md`](v1-session/spec-identity.md) | how it looks: palette, type, surfaces, motion |
| [`design-prompts.md`](v1-session/design-prompts.md) | the prompts the design was generated from |

## v2-console/

| File | What it is |
|---|---|
| [`runbook.md`](v2-console/runbook.md) | operator steps for `dpo console`, and the corpus calibration measured so far |
| [`spec-system.md`](v2-console/spec-system.md) | the measured half: masks → grouping → quantities → regimes → prompt |
| [`spec-uiux.md`](v2-console/spec-uiux.md) | the surfaces, as a design brief |

## The two specifications contradict each other

This is not drift to be tidied away. `v2-console/spec-uiux.md` reverses several
of `v1-session/spec-behavior.md`'s *fixed decisions* — a console against no
console, graying at the last detent against graying nothing, a numbered
skeleton against no numeral anywhere, the crossfader as the control against the
crossfader as a comparison condition — and drops the listing gate, the measures
screen, the control task and the follow-up entirely.

Both are built so the choice can be made by looking at them. The comparison
table is at the top of [`v2-console/runbook.md`](v2-console/runbook.md).
Whichever is not adopted is removed by deleting its package, its tests, and its
directory here; nothing in `shared/` needs untangling, and
[`shared/caption-stack.md`](shared/caption-stack.md) says why.
