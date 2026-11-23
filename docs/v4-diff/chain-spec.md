# The response–caption chain: specification

Drafted 2026-11-18 from the researcher's decisions of that day; **approved by the researcher on
2026-11-23** (decisions below). Nothing here is built yet. It replaces
the separate rules of the two phases (the emphasis sentence, the tuner sentence, sliders that enter
no equation) with one chain from what a participant answers to the caption they are shown and on to
the analysis. Sessions wait until it is built.

**Decided (researcher, 2026-11-18):**
- the paper claims both the formulation and its test, as one framework;
- phase 2 tests a prediction made from phase 1;
- the survey measures *perception* and the sliders measure *preference*; the link between them is a result;
- phase 1 asks directly where the noticed sound seemed to come from;
- phase 1's emphasis sentence is replaced by a texture level, so both phases write in one caption space;
- context is predicted from binding and texture from fascination;
- anything in the study may change for this.

**Approved (researcher, 2026-11-23), each as drafted:**
- H2 is registered one-sided, `B > 0`; the rival direction (§7) is reported descriptively, through
  the share of time at each stop, not tested;
- phase 1's texture follows fascination by the rule in §4, under both modes alike;
- the tuner sentence goes: one fixed role sentence for everyone, so the sliders are phase 2's only
  personal input and H2/H3 cannot be circular;
- the sliders' starting stops are counterbalanced 0/1/2 across the three videos;
- the direct report comes right after the noticed-sound item, before the second viewing;
- `ρ`, `δ`, the settling window and N are set by the simulation (§8 step 1) and reported to the
  researcher before pre-registration; the values below are its starting points;
- the cited sources were checked on 2026-11-23 (Sources).

**Set by the simulation (2026-11-23, `artifacts/diff/chain-sim/`; researcher's choice of strategy):**
- `ρ = 0.8`, `δ = 0.5` (texture levels 0/1/2 on 28/43/29% of clips), settling window 30 s (power
  did not change between 0 and 120 s, since the start is counterbalanced and in the model);
- the report lifts the recovery of the binding trait from r = .41 (click and E1/E3) to .65;
- **H1 and R are confirmatory; H2, H3 and D are registered estimates** (direction stated, posterior
  and 90% interval reported, the small sample stated), because between-person links from three
  slider settings each are underpowered: at N = 48 and γ = .3, H2 .26 and H3 .45, against H1 .97
  and R .94; 80% would need N ≈ 130 for H3 and over 200 for H2;
- **N = 48** (a multiple of the six start orders), which gives H1 and R over .9.

## 1. The idea in one paragraph

A caption for a sound can say two different things: *what makes the sound and where it is*, and *what
the sound itself is like*. These are the two ways of listening the literature already separates:
everyday or causal listening, which attends to the source, and musical or reduced listening, which
attends to the sound's own qualities (Gaver 1993; Chion 1994). The study's two sliders are these two
axes. The chain says: how a participant *perceives* a clip's sounds on each axis predicts the caption
they *prefer* on that axis. Phase 1 measures perception and tests the first axis by experiment;
phase 2 measures preference and tests the prediction on both.

## 2. One caption space

Every caption in both phases is a point `c = (m, x, t)`:

| Symbol | Meaning | Values |
|---|---|---|
| `m` | mode: is the sound tied to what is on screen | acousmatic −½, scene-aware +½ |
| `x` | context: how far the source is specified | 0 category, 1 specific source, 2 location and relation |
| `t` | texture: how much of the sound's own quality is described | 0 none, 1 some, 2 much |

Phase 1 sets `m` and `t` and holds `x` at 1. Phase 2 sets `m` from the masks and lets the participant
set `x` and `t`. The writer receives the cue, `m`, `x`, `t` and nothing about the participant: the
tuner sentence (rating means and sound families) goes, so two participants at the same point read
the same kind of caption and the sliders are the only personal input in phase 2.

## 3. Three terms, one currency

Every term is a log-odds or a centred score, and each has a value that means "says nothing".

**Scene term, per sound (unchanged).** `v = logit q`, where `q` is the share of the sound's audio mask
on the visual categories its source can be seen in. `v = 0` is a source on the line.

**Binding term, per participant and clip.** `b` = the participant hears the noticed sound as coming
from the scene. Its prior is `logit P(b) = logit π_i + v`, with `π_i` the participant's tendency. The
evidence, as log likelihood ratios that add:

| Evidence | Log ratio | Says nothing when |
|---|---|---|
| **direct report** `r` (new): "이 소리는 어디에서 나는 것처럼 들렸나요?" | `+ln(ρ/(1−ρ))` for on screen, `−ln(ρ/(1−ρ))` for not | not asked |
| V1 point on the sound's click mask (as now) | `ln((0.2 + 0.8a)/a)` hit, `ln 0.8` miss | source not on screen |
| E1/E3 against the participant's own mean (as now) | `ln 3 · (C − C̄)/3` | at their own mean |

`ρ` is how reliably a report reflects the percept; proposed 0.8, so a report carries `ln 4`, the size
of a hit on a small source. As now, `u = logit P(b | evidence) − logit P₀(b)` is what the answers add
to the scene.

**Participant binding, `B_i`.** `B_i = logit π̂_i`, the tendency learnt from all twelve clips'
evidence: how much more (or less) this participant binds sounds to the scene than the masks alone
predict. `B_i = 0` is a participant who follows the masks.

**Fascination, per participant and clip.** `F_ij` = mean of the three fascination items at P4 (0–6).
`f_ij = F_ij − F̄_i` is the clip against the participant's own level, and `Φ_i = (F̄_i − 3)/3` is the
participant's level on a −1 to +1 scale. Both are 0 at "says nothing".

## 4. From terms to the caption shown

**Phase 1, second caption of clip j.**
- mode: recommended `m* = +½` when `P(b) ≥ ½`, else `−½`; then followed or countered by the
  stratified plan (unchanged). This is the experiment.
- texture: `t = 1 + [f_ij ≥ δ] − [f_ij ≤ −δ]`, with δ proposed at 0.5 scale points: a clip the
  participant found more fascinating than their usual gets more of the sound's quality described, a
  less fascinating one gets none. Applied the same way under both modes, so it cannot bias the mode
  comparison. It is not tested in phase 1.
- context: `x = 1`.

**Phase 2, cue k.**
- mode: from the masks alone (the 2 s rule, unchanged).
- context and texture: where the participant has set the sliders at that moment.
- starting stops: each slider starts at 0 on one video, 1 on one and 2 on one, counterbalanced across
  participants and unrelated to phase 1. This is the method of adjustment with varied starts, so the
  pull of the starting point is measured and removed instead of being mistaken for preference.

**Recorded at the handoff, before phase 2 begins:** `B_i` and `Φ_i`. These are the predictions.

## 5. From terms to the analysis

The same symbols are the model terms. Mode is coded ±½; participants and clips or videos are random
effects; the models are Bayesian multilevel, as already planned.

| | Test | Model | Prediction |
|---|---|---|---|
| H1 | a caption in the matching mode restores more (phase 1, experiment) | `P9 ~ P4 + follow + (1 \| i) + (1 \| j)` | `follow > 0` |
| H1′ | and its two sources | `P9 ~ P4 + m × (v + u) + …` (as planned) | `m × u > 0`, `m × v > 0` |
| H2 | perception predicts preference, source axis (phase 2; estimate) | `X_iv ~ B_i + start + (1 \| i) + (1 \| video)` | `B > 0` |
| H3 | perception predicts preference, quality axis (phase 2; estimate) | `T_iv ~ Φ_i + start + (1 \| i) + (1 \| video)` | `Φ > 0` |
| D | the two axes are distinct (estimate) | add `Φ_i` to H2 and `B_i` to H3 | cross terms near 0 |
| R | binding is a stable trait of the listener | source-block intercept `a_i` (phase 2) against `B_i` (phase 1) | positive correlation |

`X_iv` and `T_iv` are the stops participant `i` settled on in video `v`: the time-weighted mean stop
over the video after its first 30 s. The source block's own model (`answer ~ a_i + λ·v_k + video`) is
unchanged; its intercept `a_i` is the same quantity as `B_i`, measured again by direct report on
other material. H2, H3 and R are what make the chain a chain: a phase-1 term reappears as a phase-2
outcome's predictor.

## 6. What changes in the study

| Where | Change |
|---|---|
| Phase 1, after the noticed-sound item | one new item per clip: where the sound seemed to come from (two answers, wording already approved for the source block) |
| Phase 1, second-caption prompts | the three emphasis sentences go; the prompt takes a texture level, in the wording of phase 2's texture stops |
| Binding model | one more evidence term (the report); policy version, sensitivity run and approval again |
| Handoff | records `B_i` and `Φ_i`; no rating means or sound families are sent to the writer |
| Phase 2 writer | a fixed role sentence for everyone; cell prompts unchanged |
| Phase 2 sliders | counterbalanced starting stops instead of the midpoint |
| Export | the report, `B_i`, `Φ_i`, the texture level and its rule; a flat table of slider stops per video and per cue |
| Analysis | a script that fits the table in section 5 from an export |

Unchanged: the clips and masks, the mode's follow/counter plan, the P4/P9 scales, the long videos,
the source block, the 2 s on-screen rule.

## 7. What could be wrong, stated now

- **A rival direction for H2.** Unseen sources make listeners work harder to identify the cause
  (Chion on acousmatic sound). So a participant who binds little may want the *source named* more,
  while one who binds much wants its *place in the scene*. Both predict a stop above 0; they differ
  between stops 1 and 2. The registered direction is `B > 0`; the share of time at each stop is
  reported so this rival can be seen.
- **Mode and the context slider overlap.** A scene-aware caption already places a sound on screen, and
  stop 2 adds location. H2 is therefore tested with the cue's mode in the model.
- **The report may sensitise.** Asking where a sound comes from before the second viewing draws
  attention to it, equally for follow and counter clips. It cannot bias H1 but may enlarge it.
- **Fascination is rated for the clip's sounds as a whole**, the noticed sound among them.
- **Open parameters, to be set before any session:** `ρ` (0.8), `δ` (0.5), the 30 s settling window,
  and the number of participants. Each needs a simulation of whether its link can be recovered; the
  earlier click-only tendency could not be (r ≤ 0.18), which is why the report is added.

## 8. Order of work once this is approved

1. Simulate H1–H3, D and R at realistic numbers of participants, and set `ρ`, `δ` and N from it.
2. Change the binding model, the phase-1 item and the prompts; bench the new second-caption prompts.
3. Change phase 2's starts, the handoff record and the writer's role sentence.
4. Write the export tables and the analysis script, and run it on simulated exports.
5. Pre-register sections 3 to 5 and 7.

## Sources

Checked on 2026-11-18 and 2026-11-23 (bibliographic record and abstract; the full texts of Gaver
1993 and the books were not read):
- Gaver, W. W. (1993). What in the world do we hear? An ecological approach to auditory event
  perception. *Ecological Psychology* 5(1), 1–29. Its abstract defines everyday listening as hearing
  events in the world rather than sounds per se. The contrast with *musical* listening (attributes
  of the sound itself: pitch, loudness, timbre) is stated in Gaver's Institute of Acoustics paper on
  auditory icons (1989); cite that for the contrast.
- Chion, M. (1994). *Audio-Vision: Sound on Screen*. Causal, semantic and reduced listening; reduced
  listening and "acousmatic" come from Pierre Schaeffer.
- Kaplan, S. (1995). The restorative benefits of nature: Toward an integrative framework. *Journal
  of Environmental Psychology* 15(3), 169–182. Attention Restoration Theory, of which fascination is
  a component.
- Payne, S. R. (2013). The production of a Perceived Restorativeness Soundscape Scale. *Applied
  Acoustics* 74(2), 255–263.
- Gescheider, G. A. (1997). *Psychophysics: The Fundamentals* (3rd ed.). Lawrence Erlbaum. The
  method of adjustment, where the observer controls the stimulus.
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*
  185(4157), 1124–1131. Adjustment from an anchor.
- Johnson, E. J., & Goldstein, D. (2003). Do defaults save lives? *Science* 302(5649), 1338–1339.
  A default's pull on a choice.
- Chen, Y.-C., & Spence, C. (2017). Assessing the role of the 'unity assumption' on multisensory
  integration: A review. *Frontiers in Psychology* 8, 445. Proposes the prior of a common cause as
  the measure of the unity assumption, which is what `π_i` is here.
- Kay, M., Nelson, G. L., & Hekler, E. B. (2016). Researcher-centered design of statistics: Why
  Bayesian statistics better fit the culture and incentives of HCI. CHI 2016, 4521–4532.
- Cockburn, A., Gutwin, C., & Dix, A. (2018). HARK no more: on the preregistration of CHI
  experiments. CHI 2018.
