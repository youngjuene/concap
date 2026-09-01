# Scene-Adaptive Audio Caption Curation

## System description

Companion document, covering surfaces and interaction, is
`spec-uiux.md`.

---

## 1. Purpose

The system elicits audio caption preferences for cityscape video. It does not
train a captioner. It takes the output of an existing audio-language model,
re-emphasizes it through a small set of participant-operated parameters, and
records the settings participants converge on as data. The settings, modeled
against measured scene structure, yield a policy that maps scenes to caption
parameters. That fitted policy is the deliverable. The study does not evaluate
the tool so much as manufacture its default behavior.

## 2. Architecture

The chain has a deterministic middle and a stochastic end.

```
masks → IoU grouping → {r_g, p_g, w_g} → v_g(r0) → filter
      → rank(α) → band + register → grain → prompt → caption
```

Everything through the prompt is deterministic and recomputes instantly on any
parameter change. Only the final prose generation is slow, paid, and sampled.
The interface is split along the same line. The deterministic portion provides
live feedback, and generation occurs only on explicit request. Section 8 covers
how sampling is made stable.

## 3. Preprocessing

### 3.1 Shot segmentation

Segmentation masks are generated per frame from visual label prompts. Each
frame yields a normalized composition vector `c(t)` over visual classes by
pixel share, and a cut is placed where composition shifts by more than a
threshold.

$$\tfrac{1}{2}\sum_k \left| c_k(t) - c_k(t-\Delta) \right| > \theta$$

The same total variation distance reappears in Section 9 as the emphasis
divergence measure, so one metric performs both jobs. Splitting enforces a
minimum shot duration. The floor on shot length is set by audio reliability
rather than by any visual criterion, for the reason given in Section 5.2. After
splitting runs, boundaries are edited by hand rather than recomputed, and `θ`
never appears again downstream.

### 3.2 Audio labeling

The audio-language model runs once per shot-aligned window. For each label it
returns a confidence score and prose. Because the window is defined by the
visual cut, the temporal grid of audio analysis is set by image change.

### 3.3 Source localization

For every audio label in a shot, the segmenter is prompted with that label text
to produce a per-frame mask. The label text itself bridges the two modalities,
so no fixed correspondence table between visual and audio vocabularies exists
anywhere in the system. Preprocessing therefore runs two independent prompt
paths. Visual label prompts drive shot segmentation, and audio label prompts
drive source localization.

## 4. Source grouping

Prompt-based segmentation returns overlapping masks for labels that share a
physical source. Prompting with *speech* and with *footsteps* recovers the same
pedestrian pixels, and summing area across labels would credit that source
twice. Labels whose masks agree above an IoU threshold are therefore merged
into one sound source `g`. Area is computed once per group, and salience is
summed within it. The sound source, not the label, is the unit everything
downstream operates on.

## 5. Quantities

### 5.1 Definitions

Table 1 lists the quantities computed for shot `s` and source `g`, where `F`
is the frame, `M_g(t)` the mask at frame `t`, and `T` the frame count.

**Table 1. Computed quantities**

| Symbol | Definition | Meaning |
|---|---|---|
| `a_g(t)` | \|M_g(t)\| / \|F\| | area ratio at one frame |
| `r_g` | mean of `a_g(t)` over `T` | mean visual share |
| `p_g` | fraction of frames with `a_g(t) > 0` | presence rate |
| `c_g` | max confidence in group | detection certainty |
| `e_g` | band-limited energy over window | acoustic prominence |
| `w_g` | `c_g · e_g` | audio salience |
| `v_g` | `r_g / (r_g + r0)` | saturating visibility |

The product form of `w_g` separates detection from prominence. Confidence
reports that a label is present, while energy reports how much of the acoustic
field it occupies, and a distant siren can carry high confidence with
negligible energy. Weighting by confidence alone would make the audio axis
detectional rather than acoustic.

### 5.2 Window coupling

Shots are cut on visual change and audio windows align to those cuts, so
visual events set the analysis window for audio. A siren sweeping across a cut
splits into two weaker detections. Lowering `θ` therefore shortens windows and
degrades `w_g`, which is why the floor on `θ` and the minimum shot duration
belong to audio reliability and should be measured before `θ` is fixed.

### 5.3 Visibility

Visibility saturates rather than thresholding, since a hard cutoff makes a
small distant source flip between states on noise. The half-saturation
constant `r0` is the area a source must occupy to count as half visible. It is
an empirical claim about pedestrian-scale visual salience, set once per corpus
through the calibration protocol described in the companion document, then
frozen.

## 6. Normalization and filtering

Normalized weights are computed over the full source set before any filtering.

$$\hat{w}_g = \frac{w_g}{\sum_h w_h}, \qquad \hat{v}_g = \frac{v_g}{\sum_h v_h}$$

Filtering applies afterward. Normalizing over the admitted subset instead
would make removing one source redistribute weight across the rest and reorder
rows the participant never touched, so no control would be attributable.
Computing first and filtering second keeps removal subtractive. Both
normalizations use saturating visibility, so coherence and ordering rest on
one perceptual assumption applied consistently.

## 7. Parameter layer

Table 2 lists the three participant-facing controls. They correspond
one-to-one with the middle column of the original design sketch.

**Table 2. Experimental variables**

| Control | Governs | Form |
|---|---|---|
| admitted set | which sources enter the prompt | subset toggles |
| `α` | audio against visual weighting in rank | regime selection |
| grain | descriptive granularity | four steps |

Rank is assigned by a weighted score.

$$\text{score}_g = \alpha\,\hat{w}_g + (1-\alpha)\,\hat{v}_g$$

The score is linear in `α`, so the induced ordering changes only where two
sources cross. The reachable orderings partition `[0, 1]` into at most
`n(n-1)/2 + 1` regimes, and within a regime `α` is unidentifiable because no
output distinguishes values inside it. The recorded variable is therefore the
regime interval, and `α` enters analysis as interval-censored data. Onset
ordering and shuffled ordering are not participant options. They are assigned
conditions, since a temporal prior is not a point on the modal axis and the
shuffle exists as an experimental control.

Grain runs from itemized through grouped and scene-level to atmospheric. At
the atmospheric step nothing is named, so the admitted set and `α` have no
surface to act on and the interface visibly disables them. Conditions at that
step are not comparable to conditions elsewhere.

## 8. Generation

The prompt assembles the ordered admitted sources, each carrying a visibility
band on `v_g` and a temporal register from `p_g`, followed by the grain
instruction. The generating model is told to honor the order and is never told
what produced it. Visible sources are described as seen producing their sound,
and non-visible sources as arriving from outside the frame.

Generation is made stable by caching on the tuple of shot, regime, admitted
set, and grain. Identical settings always return identical prose within a
session, so participants search a stable function, revisits cost nothing, and
revisits are visible in the log. There is no reroll. The sampled prose is part
of the object being preferred, and a reroll would shift the measured variable
from parameter preference to sample curation. Captions are generated per shot
independently, with no memory across cuts.

## 9. Measurements

Three quantities are computed for analysis and never feed generation. They are
hidden from participants entirely.

Anchoring ratio reports what proportion of what is heard can be seen.

$$C_{\text{anch}} = \frac{\sum_g w_g v_g}{\sum_g w_g}$$

Emphasis divergence reports whether the modalities agree on relative
weighting, and is undefined when nothing is visible, which is a meaningful
state rather than an error.

$$D = \tfrac{1}{2}\sum_g \left| \hat{w}_g - \hat{v}_g \right|$$

The per-source residual locates the disagreement.

$$\rho_g = \hat{w}_g - \hat{v}_g$$

Under the preference study these stop being diagnostics and become the
predictor side of the main model, as Section 12 describes.

## 10. Roles and configuration

Every adjustable quantity is sorted by one criterion. If varying it within a
session is the point of the session, it is an experimental variable. If
varying it between sessions would invalidate comparison, it is a method
constant. If it must be tuned to the corpus once and then hold still, it is a
calibration. Table 3 gives the assignment.

**Table 3. Authority tiers**

| Quantity | Tier | Cadence |
|---|---|---|
| saturation applied to ordering | method constant | per study |
| `w_g = c_g · e_g` | method constant | per study |
| IoU grouping threshold | method constant | per study |
| register from `p_g` | method constant | per study |
| `r0` | calibration | per corpus |
| band cut points | calibration | per corpus |
| `θ`, minimum duration, boundaries | calibration | per corpus |
| admitted set, `α`, grain | experimental variables | per shot, per run |

Constants and calibrations freeze into a versioned configuration artifact with
a hash. Every caption, endpoint, and log entry carries that hash, so any
result is reproducible from its stamp.

## 11. Study design

Participants operate the parameter layer, so the paradigm is the method of
adjustment and the interface is a measurement apparatus. The commit criterion
is preference. The participant affirms the caption they would want, which
makes the participant the authority on the measurement and scales the claim to
what a curation system can honestly support. Fidelity would require an answer
to faithful-to-what that remediated footage does not give, and adequacy for
viewers who cannot hear, measured without deaf and hard-of-hearing
participants, would not survive review.

Table 4 gives the trial structure per clip.

**Table 4. Trial phases**

| Phase | Activity | Data captured |
|---|---|---|
| 1 watch | clip plays with raw model captions | replays, dwell |
| 2 author | per-shot adjustment, commit per shot | trajectories, generations, provisional endpoints |
| 3 watch again | clip plays with committed captions, revision in place | revisions, final endpoints |
| check | committed against default-policy caption, two-alternative | choice per shot |
| probe | free text on what the controls could not change | text |

The phase 2 commit is provisional. Because consoles remain live during the
second viewing, the endpoint is whatever survives phase 3, and the difference
between the phase 2 snapshot and the phase 3 endpoint is itself a measurement,
authoring preference against viewing preference. The check and probe follow
the endpoint rather than the provisional commit.

Guards against known failure modes of adjustment are structural. Start states
are randomized per trial, counterbalanced, and recorded as covariates. Commit
is disabled whenever displayed prose is stale relative to current settings, so
every endpoint was witnessed as a caption. The full event stream is logged,
since trajectories in adjustment paradigms routinely carry more information
than endpoints.

The phase 1 baseline is the raw model prose, which is not a point in the
console's reachable space. A participant who preferred it cannot steer back to
it, and that frustration lands in the probe, which is the instrument for
bounding the reachable set empirically. The raw caption, the default-policy
caption, and the committed caption form a three-object gradient per shot.

Phase 3 is also where the absence of a cross-shot persistence control is
adjudicated. Re-viewing is the first experience of the captions as a sequence,
and independent generation re-describes whatever persists across cuts. If
probes fill with redundancy complaints, that is the evidence for a persistence
parameter in the next iteration.

## 12. Modeling target

Committed settings are the outcomes. Scene measurements, including
`C_anch`, `D`, `ρ_g`, and the composition vector, are the predictors. The
model is hierarchical across shots and participants, with `α` entering as
interval-censored. The fitted function is a policy from scene structure to
caption parameters. Deployed, it becomes the system's default, with the
instrument surviving as the override layer. Scene-adaptive curation is that
regression, not an aspiration.

## 13. Open items

Table 5 lists what remains unsettled. None blocks the prototype.

**Table 5. Open items**

| Item | Status |
|---|---|
| `r0`, band cut points, IoU threshold values | set in admin calibration per corpus |
| Korean wording of the commit criterion | construct decision, pending |
| real stimuli with audio | required before settings are interpreted |
| cross-shot persistence | deferred to phase 3 evidence |
| saturation direction | ordering saturates, revisable theory choice |

The last row records a claim rather than a fact. The specification has
ordering inherit the saturating visibility on the assumption that visual
salience saturates with area. The reverse resolution, linear coherence, is
equally consistent and states a different perceptual claim.
