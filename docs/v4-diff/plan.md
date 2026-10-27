# v4-diff plan — Trida-7B captions behind the regen participant flow

Phase A deliverable, revised 2026-10-09 for Dream and again on 2026-10-12 back
to Trida-7B (both model changes below). Written against `main` at `3842302`.
Where the brief and the PRD disagree, the brief wins.

## Phase 1's second caption — a Bayesian binding model (researcher, 2026-10-23)

**Asked:** the second caption takes one mode per clip, scene-aware or acousmatic,
and its wording varies with the clip's scene. A Bayesian algorithm chooses the
mode from the response scales, the visual mask proportions, and the match
between the Sa2VA audio mask and the sound the participant noticed (AudioSet
label, grouped). The goal is restoration. The researcher asked for the most
HCI-fitting way to find the restorative direction. The answer, built here, is a
theory-driven model, validated by a randomised, balanced within-subject
comparison and analysed with Bayesian multilevel models. Phase 2 is untouched.

**The model** (`diff.phase1_binding`, `diff.phase1-binding/v3`, provisional;
v1 of 2026-10-23 was revised twice the next day, below):
- `b` says whether the participant hears the noticed sound (A1s) as part of the
  scene on screen. `π` is their tendency to bind, uniform on a 100-point grid.
  The clip's prior is `logit P(b) = logit π + logit q`, where `q` is the
  noticed sound's measured visibility (v3, below): the Beta posterior mean
  of the share of its audio mask on the visual labels its source can be.
- **When:** every clip's mode is decided once, at the last P5, since stage A
  holds every clip's first step before any second viewing. The 12 jobs are
  then queued in stage-B order; the interlude covers the first.
- **Attention evidence:** is the V1 point inside the noticed sound's click
  mask (its Sa2VA audio mask, widened by 1.5% of the frame's width) on its
  frame? Chance is the click mask's share `a` of the frame, floored at 0.05. A
  bound viewer's eye is drawn to the source with probability 0.2.
  - A hit gives `(0.2 + 0.8a)/a`; a miss gives 0.8.
  - The ratio is 1 when the source is not on screen on that frame, or not
    on screen on average.
- **Congruence evidence:** E1 and E3, the sheet's axis-1 check items, against
  the participant's own mean over their 12 clips:
  `log ratio = ln 3 · (C − C̄)/3`.
- **Tendency:** each clip's `π` is updated by the other 11 clips' attention
  evidence only.
- **Decision:** scene-aware when P(b) ≥ 0.5, otherwise acousmatic. A
  caption that matches how the sound is heard should cost no directed
  attention (ART's compatibility).
- **Assignment:** a balanced plan per participant drawn from the order seed
  (`diff.phase1-assignment/balanced-v1`). Six clips follow the
  recommendation and six take the other mode. Because follow vs counter is
  independent of the recommendation, each mode is shown with probability 1/2
  whatever P(b) is, so P(b) is fixed before the outcome. The analysis takes it
  as two moderators, the scene's and the participant's (below).
- **Not steering:** F, BF, E, PRS, FA2 and FA3 are recorded as the baseline
  (`model.baseline`), the first measure of the outcome. FA1 is unused.
- **Manipulation check:** P7's record tests the V2 point against the noticed
  sound's click mask in the same way (`source_attention`, exported as
  `v2_attention`). S2 does not check the mode, since an acousmatic caption
  can still fit the scene.

**Design review, 2026-10-24** (before any Trida run; what v1 got wrong):
- **Rating style built up in `π`.** v1 fed E1/E3 into `π` clip by clip, so a
  participant's rating level accumulated. Simulated on the 12 clips: rating
  E1/E3 at 4.5 rather than 3 took the scene-aware share from 23% to 89% by the
  twelfth clip. The fix is the whole-stage decision with centred E1/E3 and a
  tendency learnt from the other clips' points.
- **Visibility depended on an arbitrary rule.** v1's `q` was the Beta mean of
  the share of frames whose audio mask lay at least half inside the plausible
  categories. Under the other rule (less than half outside them), 10 of the 27
  seeable sounds moved by more than 0.1, and 3 crossed 0.5. Sa2VA's visual
  masks miss things: singapore_067's speech mask sits on two distant talkers
  that the Person mask does not cover, which gave 0.08. amsterdam_012's siren
  mask sits on the passing tram, so a mask shows "a plausible source on
  screen", not the true one. The fix is the researcher's judgement per option
  (drafted from the share) and the second rule for the per-frame test.
- **The click evidence was over-weighted.** V1 asks what first caught the eye,
  which visual salience drives, so v1's 0.5 for a bound viewer clicking the
  source made most points misses and pushed toward acousmatic. At neutral
  ratings, 0.5 vs 0.2 moved the scene-aware share from 0.27 to 0.41. Several
  click masks covered under 0.5% of the frame with no tolerance.
- **Caption types:**
  - The sky median split put 0.084 (enclosed) and 0.094 (open) on opposite
    sides; it is now three levels with a neutral band.
  - Two distinctive picks had lifts of 1.12 and 1.18; under 1.5 the largest
    element stands instead.
  - The scene-aware prompt asked for two centres, which invited invented
    links; v2 sets the V1 object as the setting and ties each sound only to
    its own source.

**v3: visibility measured, one setting per clip** (researcher, 2026-10-24,
after the second captions were drafted for review):
- **Visibility follows the masks strictly, with no judgement.** It is the
  share of the noticed sound's audio mask (its AudioSet label, the audio
  caption's category match) that lies on the visual labels its source can
  be (`PLAUSIBLE`), averaged over the sampled frames. Unlabelled pixels count
  against it. `q` is the share's Beta posterior mean (0.08–0.92 for a 10 s
  clip). A source is on screen from a share of 0.5 (`ON_SCREEN`): on average
  for the caption and for counting a click, and per frame for a click.
  - **What `q` measures: candidate-source availability** (design review,
    2026-10-25). Sa2VA never hears the audio. It draws each audio mask from
    the frames and the label's text alone ("segment all possible sources of
    {label} sound in the scene", in
    `/mnt/hdd/research/2026/Sa2VA/avmask/bin/sa2va.py`). So `q` says
    whether a plausible source of the sound's kind is in view, not whether
    the actual source is, and nothing about whether anything on screen moves
    with the sound. "Visibility" in this plan and in the code means this; the
    code's names are unchanged.
  - Of the three readings compared, this one changed 7 of the 29 v2 drafts.
    For example, singapore_067's speech went from 0.32 to 0.01: the distant
    talkers lie outside the Person mask. amsterdam_202's engine went from
    0.90 to 0.24: its mask lies mostly on unlabelled pixels.
  - The 29 drafted AudioSet labels stand as the category match.
- **One setting per clip** (`diff.phase1-tweak/v5`). The scene-aware setting
  is the clip's distinctive element, not the V1 object. A second caption
  then depends only on the mode and the attended sound, so the 12 clips have
  58 possible captions, and every one is reviewed before the study.

**Caption types:**
- `scene/<distinctive element>` names the distinctive element and the largest
  other one, and takes the distinctive element as its setting. It lists the
  sounds on screen, and the noticed sound when the point hit its mask.
- `acousmatic` describes the sounds as heard, keeping every name so that R2's
  key holds, and names nothing on screen. Until 2026-10-26 it was
  `acousmatic/<open|neutral|enclosed>`: an open or enclosed scene added one
  request about how the sound spreads (dropped, see "Decisions of
  2026-10-26").

The prompts are `rewrite_prompts.TWEAK_PROMPTS` (`tweak-p1`, one text per
mode, provisional: scene-aware v3, acousmatic v4 since 2026-10-26). The four % degrees (`tweak-p1` v0) are gone.

**Prompt v3 and the R2 key** (after the hypothesis team's report,
`artifacts/diff/hypotheses/report-20261024.md`; the researcher said "go ahead"
on 2026-10-24):
- **One off-screen sentence.** The scene-aware prompt names the sounds not on
  screen in one shared sentence, which asks Trida to write them as heard
  somewhere here though not seen. It is left out when there are none, and
  never includes microphone wind (`RECORDING_ARTEFACTS`). v2 told Trida to
  spread every unseen sound "in this scene", mic wind included. The wording
  avoids the word for off screen, which would echo E1's place on screen.
- **80 characters.** `TWEAK_MAX_CHARS` is 80, and the prompts ask for no more.
  160 characters take about 13 s to read at 12 characters per second, longer
  than the 10 s clip. A longer output fails the check and the clip's second
  step is skipped, so the bench must show how often that happens. The fake
  worker cuts its echoed caption to fit.
- **R2 is scored against the shown caption.** The intended key stays as
  decided on 2026-10-18. The export adds a blind coding sheet
  (`r2_coding.csv`) and a rough automatic hint (`r2_named_auto`). The reason:
  amsterdam_057's first caption names a car horn, one of its R2
  distractors.
- **A fifth contradicting first caption.** singapore_067's names a horn and a
  traffic jam, where orig_text has a talking crowd and an idling truck.
- **The bench gate**, per mode:
  - near-copies of the first caption (similarity ≥ 0.9);
  - wrong first-caption sounds kept (train, horn, talk, motorbike and
    traffic jam, across the four contradicting clips);
  - R2 distractors named;
  - questionnaire wording echoed (the items' words for the place on screen,
    belonging, blending in and standing apart);
  - lists with no sentence.

  On the 2026-10-20 bench, which ran the old `tweak-p1` v0, 10 of 36 second
  captions copied the first one, and the train, horn and motorbike survived
  3/3. v3 has not yet run on Trida.
- **Not done; the researcher's to decide:**
  - sound qualities written after listening, for the acousmatic prompt;
  - unchanged anchor clips (the sheet's line keeping some clips as they are,
    against the 10-18 decision to rewrite every clip);
  - the counter-centring pilot.

**Analysis, to pre-register** (moderators split by the researcher,
2026-10-25):
`P9 ~ P4 + mode × (v + u) + (1|participant) + (1|clip)` (superseded on
2026-10-26: the model under "Decisions of 2026-10-26" adds mode slopes by
clip and first-caption fidelity), Bayesian multilevel
(Kay et al., CHI 2016), with the P4 score as a covariate rather than change
scores. `mode` is coded scene-aware +½, acousmatic −½. The binding model's
`P(b)` enters as two moderators, both fixed before the outcome and both
exported (`second_captions.csv`, export v6):
- **`v`, the scene's part** (`moderator_scene`): `logit q`, the noticed
  sound's measured visibility, i.e. whether a plausible source of its kind is
  in view (v3, above). It depends on the participant only through which
  sound they noticed. `v = 0` is a source on the line
  (`q` = 0.5).
- **`u`, the participant's part** (`moderator_participant`):
  `logit P(b) − logit P₀(b)`, where `P₀` is the model with no answers (a flat
  tendency and no evidence from the clip). It holds what the answers add: the
  tendency from the other clips' points, the clip's own point, and E1/E3.
  `u = 0` when the answers say nothing. `logit P(b) − logit q` would not do:
  a flat tendency pulls P(b) toward 1/2 (by up to 0.8 in log odds at the
  setup's `q`), so part of `q` would stay in it.

Why two: P(b) mostly follows `q`. In a simulation on the 12 clips with
assumed answer patterns, the recommendation was the scene-only rule
(scene-aware when the source is on screen) in 80–96% of clips. One
`mode × P(b)` term could not tell a participant's binding from a source's
visibility. In the same simulation, `u`'s SD was 0.59 against `v`'s 1.39,
and they correlated at 0.05.

How the estimates are read:
- **`mode × u` is the compatibility effect of binding,** the test of the
  model's claim: answers that say "bound" should favour the scene-aware
  caption beyond what visibility predicts.
- **`mode × v` is the scene's part, not evidence of binding.** It also
  carries how much the two modes differ, which grows with visibility: a
  scene-aware caption ties a visible source to the screen, but can only
  place an unseen one "here, though unseen". This is the visibility-only
  rival. What it tests is semantic: a source of the right kind in view, the
  semantic route of the unity assumption (Chen and Spence 2017). `q` holds no
  temporal or spatial coincidence with the sound, so this term is not a test
  of synchrony.
- `β(mode × u) − β(mode × v)` is reported. Equal coefficients give back the
  one-moderator model `mode × logit P(b)`, kept as a secondary analysis.
- `u`'s observed spread is reported before any estimate. A narrow `u` makes
  `mode × u` weakly informative, so a null there is not evidence against
  binding. A wide one is not evidence that `u` measures binding, though: V1
  asks what first caught the eye, which visual salience also drives, and in
  simulation `u` tracked the true tendency at r ≤ 0.18 (below).

**`q` keeps its full weight** (researcher, 2026-10-25). A weaker pull, so that
the answers move more recommendations, was simulated on the real setup. The
prior was `logit π + λ·logit q`, with answers generated by the model's own
assumptions, under three truths about binding: driven by the scene, by the
participant, or by both equally (the model's own).
- At λ = 0.5 the answers decided 18–22% of recommendations instead of
  9–11%; at λ = 0, about half.
- `u`'s SD stayed at 0.61–0.67, and its correlation with the participant's
  true tendency at 0.18 or below, so `mode × u` gains no power. The limit is
  how little the answers say, not `q`.
- Whether the recommendations got more accurate depended on the truth.
  Under scene-driven binding, accuracy fell from 81% to 69% at λ = 0. Under
  participant-driven binding, it rose from 65% to 72% at λ = 0.25.
- "Follow" clips with a scene-aware caption for a source that isn't on screen
  rose from 1% of clips to 13% at λ = 0. Such a caption can only say the
  sound is heard here, though unseen.

A stronger binding measure would be the lever for the participant's part, for
example pointing to where the noticed sound came from. That would change the
sheet's instrument, so it is not planned.

E1/E3 feed `u`, so the main effect of `u` on the E change carries regression
to the mean; the mode is randomised, so its interactions do not. They are not
free of ceiling effects, though: a clip rated high on E1/E3 has a high `u` and
leaves the E items little room to rise, which can shrink `mode × u` on them
(design review, 2026-10-25). Since the
mode's propensity is 1/2 for every clip, the same data also compare three
policies: the model's, always scene-aware and always acousmatic. The primary
outcome subscale is still to be named. The five clips whose first caption
contradicts their audio (amsterdam_012, amsterdam_057, bangkok_034,
singapore_040, singapore_067) get a correction in both modes, so they enter
as a covariate.

**Measured on the 12 c2 clips** (setup draft, rebuilt 2026-10-24 from the
windowed Sa2VA run, which stays the measure after 2026-10-26's comparison
with the full-clip run; see "Decisions of 2026-10-26"):
- Measured visibility (v3): 13 of 29 sounds on screen and 16 off. Most
  vehicles are on screen. Speech is on screen only at bangkok_209 (0.61).
  - In four clips no source is on screen, so their scene-aware caption
    places every sound "here, though unseen": amsterdam_057, amsterdam_202,
    bangkok_328 and singapore_067.
  - Before v3 the drafts were 14 visible, 8 partly and 7 not.
- Distinctive elements: traffic 3, people 3, built 3, nature 2 and open 1
  (amsterdam_181). amsterdam_012 and bangkok_209 have no lift of 1.5, so their
  largest element, built, stands; 43% of amsterdam_012's frame is unlabelled
  café furniture. Openness: 3 open, 4 neutral, 5 enclosed.
- **The judgement dominated (v2, before measured visibility).** In the
  pseudo walk, every visible source was
  recommended scene-aware and every invisible one acousmatic. The answers
  decided only the partly visible ones (binders 5/5 scene-aware, detachers
  0/6). At 0.75/0.5/0.25, the participant's own points override the judgement
  in both directions: detachers got 8/10 scene-aware on visible sources, and
  binders 7/7 on invisible ones. This is the researcher's call.

**Verified (2026-10-24):**
- 6 pseudo participants walked phase 1 on the rebuilt setup, with the fake
  worker, over the app's HTTP API. P(bound) had a median of 0.92 for binders
  and 0.43 for detachers. The tendency was 0.64–0.72 vs 0.40–0.46. The plan
  split follow/counter 36/36, the modes went 37/35, and all 8 caption types
  occurred.
- `scripts/diff_session_audit.py` recomputes the whole-stage model, the plan,
  the slots and V2's test independently. It found 0 mismatches and caught a
  tampered P(bound) and a tampered V2 status. The export audit passed.
- `make check` passed (1903 tests, after prompt v3). The v3 walk and audit
  were also clean: 0 mismatches, and the export check covers the coding sheet.
- The bench (`scripts/diff_prompt_bench.py`) runs both modes per clip and
  reports per mode:
  - length;
  - place or on-screen words outside the sounds' names, and those the prompt
    did not give;
  - prompt sounds kept;
  - the four contradicting clips side by side.

**Real Trida, 2026-10-24 (GPU 1, once the card was free).**
- The six pseudo participants walked phase 1 again with the real worker, so
  each one's 12 second captions ran through the app's queue.
  - The audit found 0 mismatches.
  - The token re-hash matched 72/72 prompts.
  - The bench, given that walk's answers, reproduced every planned caption
    exactly (72/72).
- **The check failed 9 of 72 second captions**, so those clips' second
  viewings were skipped. Seven were scene-aware, and singapore_286 failed 3
  of 3 scene-aware captions. Trida either:
  - copied the scene prompt's labelled slots (the distinctive element, the
    background, the sounds whose source is seen) until it ran out of room;
  - looped, repeating the word for car;
  - or stopped after one word (road, with its particle).
- **What passes is mostly a list of sound names.** On the bench's 144
  captions (both modes):
  - 100 have no sentence (scene-aware 54/72, acousmatic 46/72).
  - The modes always differ in wording (72/72) but rarely in kind. 30 of 72
    scene-aware captions name a place or object. A word for how a sound
    sounds (distance, loudness, faintness, continuity; a rough count, often
    carried over from the first caption) appears in 28 of 72 acousmatic and
    13 of 72 scene-aware ones.
  - Near-copies of the first caption: acousmatic 11/72, scene-aware 2/72.
  - A wrong sound from the first caption survives in 32 of the 48 captions
    of the four clips that have one (train, horn, motorbike, traffic jam).
  - Nine name an R2 distractor.
  - Trida wrote a caption label (the word for caption and a colon) in 35
    (17 of the 72 planned), and words are visibly broken: a doubled
    syllable, a doubled word, two words for sound run together.

  tweak-p1 v3 therefore does not yet deliver the manipulation. The report is
  `artifacts/diff/bench/tweak-v3-20261024.json`, and the walk is in
  `artifacts/diff/walk-v3-real-20261024/` (both gitignored).

**Wording round, 2026-10-24** (researcher: "draft and run the candidates").
Six scratch candidates ran on the same inputs; none is a proposal. Everything
is in `artifacts/diff/bench/candidates-20261024/`: the candidate files, their
reports, `score.py`, and `RULE.md`. The selection rule was written before each
round ran.
- A caption counts as **delivered** when it:
  - passes the server's check;
  - is a sentence;
  - is not a near-copy of the first caption;
  - keeps no wrong first-caption sound and names no R2 distractor;
  - has no visibly broken text.

  A wording's score is the delivered count of its weaker mode, out of 72.

| Wording | Scene-aware | Acousmatic | Weaker mode | With the stricter damage check |
|---|---|---|---|---|
| v3 (checkout) | 11 | 9 | 9 | 9 |
| A, prose (slots inside sentences, no "label: value" lines) | 19 | 15 | 15 | 9 |
| B, A plus concrete expressions | 38 | 46 | **38** | 28 |
| C, A plus an unrelated example | 25 | 26 | 25 | 19 |
| D, short | 40 | 30 | 30 | 20 |
| E, D's scene-aware with where-to phrases, B's acousmatic | 35 | 47 | 35 | 27 |
| F, B with where-to phrases | 12 | 47 | 12 | 10 |

- **The rule picks B.** Its scene-aware text gives where-phrases (heard from
  a place, heard beyond something); its acousmatic text gives hearing adverbs
  (from afar, low, unceasingly, faintly, clearly). Most of its captions are
  real sentences, e.g. a siren heard from afar while passing cars are heard
  clearly.
- **Costs of B:**
  - In 10 of its 38 delivered scene-aware captions, a sound comes from
    something that doesn't make it (a presenter's voice ringing from the
    sky).
    E's scene-aware text has none of these (0 of 35), but E names fewer
    places (manipulation 24 vs 32), and the rule's tie-break goes to B.
  - The acousmatic qualities come from the prompt's word list, not from the
    audio: the adverb for clearly appears in 42 of 46. These need the
    researcher's listened qualities.
- **Visible damage is under-counted.** Cut verbs (stems that stop mid-word) and
  doubled phrases slip past the bench's broken-text check. The last column
  counts them as damage (a check added after the run, so it doesn't change
  the pick). On it, B and E are level, about 28 of 72.
- **Second captions could be prepared ahead.** A prompt depends on the
  participant only through:
  - the mode;
  - the attended sound (A1s);
  - the object under the V1 point (scene-aware only);
  - whether that point hit an unseen source (scene-aware only).

  That gives 29 acousmatic prompts and at most a few hundred scene-aware
  ones. All of them could be generated and approved by a fluent reader
  before the study, instead of the server showing whatever passes the
  check. This is the researcher's decision.

**All second captions drafted for review (researcher, 2026-10-24).** This
first run, with the V1 object as the setting, is superseded by v3's 58
variants (one setting per clip, measured visibility); it is kept in the
folder's `superseded-277/`.
- `scripts/diff_second_captions.py` lists every reachable prompt (runbook
  §11): 277 variants, 29 acousmatic and 248 scene-aware. The list contains
  every slot set from the bench and the real walk.
- Trida drafted each one with both finalists, B and E. The drafts reproduce
  the bench's outputs exactly (144/144 per wording).
- 130 of the 277 rows have at least one draft with no flagged issue:
  - 20 of the 29 acousmatic rows;
  - 42% of the scene-aware rows, weighted by how often a V1 point lands on
    each setting.
- E fails the check on 120 of 248 scene-aware variants, against 16 of 72 on
  the bench: it does worse on the settings the bench did not sample.
- The sheet and the reviewer's guide are in
  `artifacts/diff/phase1/second-captions-20261024/` (`review.csv`,
  `README.md`).
- After the review, the app has to serve the approved text by `variant_id`
  instead of generating it live (built the same day, below).

**Approved second captions served** (researcher, 2026-10-24: every row must
be approved before participants; retired on 2026-10-26, when captions became
live: see "Second captions: responsive, written live" below).
- `python -m diff phase1 import-second-captions` checks the filled sheet and
  writes the approved texts into the setup's `second_captions` block, with
  the reviewer, the time and a hash (`diff.phase1_approved`, runbook §11).
- Participant sessions are refused until every caption the setup can show
  is approved. The second-caption prompt's own provisional flag no longer
  blocks them, since a participant never sees Trida's text.
- At the last P5 the flow (`diff.phase1/v6`) takes each clip's approved
  caption, ready at once, with no Trida job. A smoke session falls back to
  Trida.
- Export v5 marks each caption's source, variant and `approved_from`. The
  audit checks the approved text against the recomputed slots, the P6
  viewing, and that no Trida call was made.
- **Verified:** six stand-in participants walked the real setup with 58
  stand-in approvals. All 72 captions came from the approved set, with no
  Trida call, and the audit found 0 mismatches. A changed approved text was
  flagged in both sessions that showed it.

**Design review, 2026-10-25** (a reviewer panel; the report is in
`artifacts/diff/reviews/20261025-caption-design/`). Three fixes needed no
decision:
- **The sound-name hint.** `named_in` matched any word of an option's label
  by its first two syllables anywhere in the caption. The siren's name matched
  the word for "between", and the label's word for "heard" matched nearly
  every caption, so dropped sounds looked kept. It now matches the label's
  sound-naming word (`phase1_tweak.sound_name`) at the start of a caption's
  word. `scripts/diff_second_captions.py rescore` scored the 116 drafts again
  without Trida: 36 show no issue, against 44 before.
- **`q` is described as candidate-source availability** (v3, above).
- **Stale text corrected:** §11 and §12 marked superseded; the reading of
  `u`'s spread and the ceiling caveat (above); the primary-outcome note
  below.

The panel's other proposals wait on the researcher's decisions.

**Decisions of 2026-10-26** (researcher, after four checks; the report is
`artifacts/diff/reviews/20261025-caption-design/checks-20261026.md`):
- **Visibility stays with the windowed Sa2VA run, over the whole clip.** The
  full-clip run (`artifacts/diff/sa2va-60fps-video/masks`, each label grounded
  once with `sa2va.py --mode propagate --window 0`) was adopted in the morning.
  It was reverted the same day after the filter audit found that it loses
  tracks. The scene parts and click masks are again those of 2026-10-24, and
  the levelled practice clip stays.
  - **Why the full-clip run was reverted.** It grounds each label on the
    first frames and then only tracks it, so a lost track is never found
    again.
    - bangkok_034's passing car is masked only in the first sampled frame,
      and singapore_286's engine disappears after frame 90. The windowed run,
      which grounds each label again every 96 frames, finds both later.
    - 6 of the 9 sounds on which the runs disagree are lost tracks.
    - Under the full-clip run, 8 of 29 sounds would be on screen instead of
      13.
  - **Where the runs agree.** The comparison uses 26 usable sounds: it leaves
    out the two winds, which have no visual category, and counts bangkok_209's
    duplicate label once.
    - **Over the first 96 frames (16% of each clip) the runs are the same
      computation.** The windowed run's first window and the full-clip run
      start from the same grounding, and their masks there are identical
      (IoU 1.00; all 26 flags agree). Agreement in that span is built in and
      is not evidence of reliability.
    - **On the frames grounded independently** (from frame 96), the runs
      agree on 18 of 26 on-screen flags: kappa 0.34, ICC of logit q 0.38,
      IoU between the runs' visual masks 0.56. Over frames 96–149 alone,
      kappa is 0.49 and ICC 0.51.
    - The full-clip run's detection rate falls from 0.64 in the first 10% of
      the clip to about 0.4 after 30%. The windowed run's stays between 0.54
      and 0.69. 9 of the full-clip run's 14 lost tracks are lost within the
      first 20% of the clip.
    - **Corrected the same day.** A first-25% comparison (kappa 0.85, ICC
      0.93) had been read as evidence that the grounding is reproducible.
      The window curve showed that 16 of its 25 points are the identical
      span. So no early-window percentage supports a reliability claim
      (`checks-20261026/window_curve.py` in the review folder).
  - **What q measures.** Over the 29 sounds' sampled frames, 40% of the
    windowed run's audio masks are empty, and an empty frame counts as not on
    screen.
    - Within the run, a sound's share follows how often it is detected
      (r = 0.86).
    - Over the whole clip the runs agree on 20 of 29 flags. None of the other
      rules tried (empty frames skipped, unlabelled pixels left out, every
      frame) raised that above 22.
  - **Rationale for the window: 100% of the clip.** q covers the span over
    which the experiment shows and asks about the sound.
    - The second caption is on screen from 0 to 10 s.
    - The five P2 stills, on which the V1 point is placed and tested, sit at
      1, 3, 5, 7 and 9 s (10–90% of the clip).
    - The 16% boundary is not a design choice: it is where the windowed and
      full-clip runs stop being the same computation. Their comparison is
      reported on the remaining 84%. The reliability the paper cites comes
      from the independent 1 fps run (below).
    - A kappa of at least 0.80 holds only in windows from the clip's start
      that the shared frames dominate (up to 29%; ICC of at least 0.90 up to
      27%). It never holds on independent frames, where kappa peaks at 0.57
      and ICC at 0.54 (plot:
      `checks-20261026/kappa_window.png` in the review folder).
    - **Against the independent 1 fps run, no interval beats the whole clip.**
      Every contiguous interval of whole seconds was checked, on two
      disjoint sets: the 26 study sounds, and a held-out set of 99 corpus
      labels from the 36 videos outside the study. The figure is
      `checks-20261026/interval_choice.png`: ICC and kappa against interval
      length, each the mean over start positions with a 95% bootstrap band
      over sounds. The full grid is `interval_1fps.png`.
      - ICC, which uses the continuous q the moderator is built from, rises
        with interval length in both sets and is highest over the whole clip:
        0.89 [0.77, 0.95] on the study sounds, and 0.77 [0.66, 0.85] held
        out. The held-out curve flattens after about 6 s.
      - Kappa on the binary flag does not favour any shorter interval. On the
        study sounds it rises to about 0.8 and the whole clip sits within
        noise (0.77 [0.52, 1.00]). Held out, it stays flat at 0.50–0.59
        whatever the length (whole clip 0.55 [0.37, 0.71]).
      - Single best intervals (3–7 s on the study sounds, 4–9 s across the
        corpus) do not replicate between sets, and picking one would fit
        noise. The whole clip is both the most reliable interval and the one
        the experiment uses.
      - **By city, the same shape and no plot.** ICC rises with length and
        is highest at the whole clip in all three cities, in both sets.
        - Held-out whole-clip ICC: Amsterdam 0.67 [0.43, 0.84], Bangkok 0.81
          [0.65, 0.90], Singapore 0.82 [0.63, 0.92]. The intervals overlap,
          and the study set has 7–10 sounds per city, too few to compare.
        - City is confounded with content, so the per-city figures go in the
          supplement's stimulus table only. Amsterdam's lower reliability is
          covered by the per-clip reliability column and the leave-one-clip-out
          refits.
    - **Split by mask type, the disagreement is mostly in the audio masks.**
      `kappa_window_av.png` varies one kind of mask at a time, with the
      windowed run fixed for the other, as coloured lines in one plot
      (`kappa_window_split.png` has the same as panels, with mask IoU). On
      independent frames:
      - when only the audio masks differ, kappa never reaches 0.80 (peak
        0.65); over the whole clip it is 0.26, with ICC 0.28 and IoU 0.20
        between the runs' audio masks;
      - when only the visual masks differ, kappa is at least 0.80 for
        windows of 19–40% (peak 0.92); over the whole clip it is 0.59, with
        ICC 0.80 and IoU 0.56.
      So the scene measures from the visual masks hold up across runs, and
      the weak link in q is Sa2VA's audio grounding. The visual column reuses
      the windowed audio masks, whose empty frames score 0 under either
      visual map, which inflates its agreement somewhat.
  - **An independent third run supports the windowed q.** The 1 fps run
    grounds each second from scratch and shares no computation with the
    windowed run beyond the model. It was compared on seconds 0–9
    (`checks-20261026/onefps_compare.py` in the review folder).
    - On the 26 study sounds the windowed run and the 1 fps run agree:
      kappa 0.77, ICC 0.89.
    - The full-clip run agrees with neither: kappa 0.14 against the 1 fps
      run, and 0.28 against the windowed run.
    - Of the 12 sounds on which the three runs disagree, the full-clip run is
      the odd one out in 9, the 1 fps run in 3, and the windowed run in none.
    - Across all 48 videos the visual scene measures reproduce: composition
      ICC 0.96, distinctive element the same in 41 of 48 videos, openness in
      47 of 48. Audio q over 127 corpus labels: kappa 0.61, ICC 0.80; on the
      99 labels from the 36 videos outside the study: kappa 0.55, ICC 0.77.
    - The windowed q from these 10 frames gives the same flags as from its 60
      sampled frames (kappa 1.00).
    - Both runs share the model, its text-only grounding and the `PLAUSIBLE`
      table, so this is reliability, not validity.
  - **Clip reliability over all 48 corpus clips, pooled** (a diagnostic, not
    a selection rule: the adopted design excludes no clip for measurement).
    The bars were fixed before looking (`checks-20261026/clip_filter.png`,
    `clip_filter.json`):
    - **Per sound:** the same on-screen flag in the windowed and 1 fps runs.
    - **Per clip:** every sound passes, and the distinctive element (the
      caption's setting) is the same in both runs.
    - A second sound bar, |q_w − q_o| within q's own 95% Beta uncertainty,
      removed no sound the flag bar keeps, so it is left out.

    Results:
    - 103 of 127 sounds and 24 of 48 clips pass. The passing clips spread
      evenly over conditions (5–7 each) and cities (7–9 each). Failures spread
      over sound groups (engine 5/22, vehicle 7/38, alarm 4/23, human voice
      8/30).
    - **7 of the 12 phase-1 clips pass.** bangkok_209 and singapore_134 fail
      on the setting (built vs open; traffic vs built). The same holds
      against the study's own 12-clip reference.
    - amsterdam_057, bangkok_328 and singapore_067 fail on the one sound
      each that only the 1 fps run puts on screen: male speech, engine and
      idling truck.
    - Only the 12 condition-2 clips have prepared captions and options. Any
      other clip would need both before phase 1 could use it.
  - **A q from the clip's start alone is not used** (researcher, 2026-10-26).
    It would describe only the first seconds of a 10-second clip.
    - It misses sources that arrive later: bangkok_034's passing car scores
      0.20 in the first 25% and 0.78 after it, in the windowed run.
    - It keeps sources that leave: singapore_286's speech scores 0.67 and 0.
    - Its only rationale was the early agreement, which is built in.
  - The blind human check of visibility was offered twice and not chosen.
- **The acousmatic `space` slot is dropped** (`diff.phase1-tweak/v6`;
  acousmatic text `tweak-p1` v4). The prompt used to ask for an open feel
  when the scene's sky share was at least 0.15, and an enclosed one when it
  was at most 0.07.
  - **What changes.** The acousmatic prompt takes nothing from the picture.
    Its inputs are the first caption, the sounds `orig_text` names and the
    attended sound. Openness is still measured and recorded but reaches no
    caption, so a change in sky share no longer changes an acousmatic
    caption's key. The export's `caption_type` for this mode is `acousmatic`
    (export v7).
  - **What it costs.** The acousmatic mode loses its one systematic,
    scene-specific element. What remains to set it apart is how the sounds
    are heard (distance, loudness, continuity), which Trida takes from the
    prompt's word list rather than from the audio (wording round, above). So
    the two modes may differ less in kind.
  - **Measured on the drafts.** The request was rarely realised.
    - Of the 40 acousmatic drafts made with an open or enclosed request, 1
      used a word for spreading or space, and none used the words for open or
      enclosed.
    - Without the request, acousmatic drafts are shorter (59 characters on
      average, against 72), more of them pass the review hints (32 of 58,
      against 25), and fewer name a place or object (2, against 6).
    - The 18 drafts whose prompt did not change (neutral scenes) came out
      identical. So the differences follow from the prompt change, not from
      decoding noise.
  - **Length.** Among drafts that pass the server's check, the modes are
    about the same length (median 48 characters scene-aware, 50 acousmatic).
    - 12 scene-aware and 4 acousmatic drafts are over the 80-character limit,
      mostly looping drafts that cannot be approved.
    - Approved texts must be 80 characters or fewer. Length is not otherwise
      matched between the modes.
    - The all-draft means first reported for the full-clip drafts (88 against
      59 characters) were inflated by looping drafts; corrected after the
      filter audit.
- **The practice clip is staged at the clips' loudness.** `draft-setup` now
  passes it through the same two loudnorm passes that staged the 12 clips
  (`phase1_setup.level_audio`), into `data/live/regen-media/practice/`. The
  corpus file is untouched, and the picture stream is copied.
  - On loudnorm's meter the copy is at −23.0 LUFS, like the clips (−22.5 to
    −23.1).
  - On ffmpeg's ebur128 meter it is at −24.1, against −22.0 to −23.3 for the
    clips; it was at −25.7.
- **Stereo stays as recorded.** In 9 of the 12 clips the right channel is
  1.3–7.5 dB louder; amsterdam_181 leans the most. This looks like a
  recording-rig imbalance. Over headphones it places sounds to the right,
  wherever their source is on screen. It is reported as a limitation.
- **The playback level is left to the participant.** The app has no volume
  control and records none, and participants may change the system volume at
  any time. Loudness at the ear is therefore neither controlled nor known,
  and the paper reports it as such.
- **Second captions drafted again,** on the windowed setup with v6.
  - Of the 58 variants, the 29 scene-aware keep their 2026-10-24 keys and
    drafts, with identical text. The 29 acousmatic are new, since they have
    no space slot.
  - Trida drafted all 58 with the two wordings, minus the space request
    (`artifacts/diff/bench/candidates-20261026/`). 43 of the 116 drafts show
    no issue: 11 scene-aware and 32 acousmatic. At least one clean draft
    exists for 9 of the 29 scene-aware rows and 21 of the 29 acousmatic rows.
  - The sheet is `artifacts/diff/phase1/second-captions-20261026/review.csv`.
    The full-clip drafts are in its `superseded-full-clip/`, and the
    2026-10-24 folder is superseded.
  - The setup's clips and practice need approving again.
- **Clip quality: the reviewer panel's design, adopted in full.** Six
  reviewers audited the first, tiered draft
  (`artifacts/diff/reviews/20261025-caption-design/filter-audit-20261026.md`).
  Five judged it acceptable with changes, one not acceptable, and all six
  rejected its tiers. To pre-register:
  - **All 12 clips, no tiers.** No clip is excluded, down-weighted or replaced
    for run agreement, mask overlap, within-clip contrast, draft quality or
    audio.
    - The tiered "robust six" subset is dropped. It had no rule, and it skewed
      the sound types.
    - It also halved power: in the audit's simulation at N = 120, H1 fell
      from 0.87 to 0.59.
  - **The only exclusion rule: the approved captions deliver their mode.**
    - Two fluent Korean raters who did not approve the captions classify each
      approved caption as scene-aware or acousmatic, blind to its intended
      mode.
    - They also flag any caption that ties a sound to a source the video
      contradicts.
    - Captions that fail are rewritten and checked again. A clip is excluded
      only if, after rewriting, fewer than 80% of its captions are classified
      correctly by both raters.
    - Kappa is reported.
  - **Model.** `P9 ~ P4 + mode × (v + u) + fidelity + mode:fidelity +
    (1 + mode | participant) + (1 + mode | clip)`.
    - `(1 + mode | clip:sound)` is added if it converges, and dropped first
      if not.
    - **Merged levels.** Sounds count as one `clip:sound` level if they share
      an AudioSet label, or if their audio masks overlap with mean IoU of at
      least 0.8 over 20 or more shared sampled frames in the windowed run.
      Provisional, the researcher's to confirm, this merges:
      - bangkok_209's two women's voices;
      - amsterdam_012's siren and traffic;
      - amsterdam_181's vehicle and horn;
      - bangkok_034's engine, passing car and horn.
    - **Why the slopes.** The earlier model had no mode slope by clip. In the
      audit's simulation, a clip-level slope SD of 0.4 raised false
      positives for H1 and mode × v to 13–17%, against a nominal 2.5% one-sided.
  - **v and its measurement error.** v = logit q from the windowed run over
    the whole clip, as the primary measure.
    - **One alternative is pre-registered: v from the 1 fps run**
      (researcher, 2026-10-26). That run grounds one frame per second from
      scratch, at seconds 0–9 (`avmask/releases/sa2va_{audio,visual}_mask_2026.zip`).
      - Its per-sound q is computed now and frozen with the stimulus table
        (`checks-20261026/onefps_compare.json`).
      - It replaces the full-clip run, which was the odd one out in 9 of the
        12 sounds on which the three runs disagree.
    - A mode × v conclusion is drawn only if its sign holds in both versions,
      and both estimates are reported.
    - **What the paper cites.** The windowed run against the 1 fps run, at
      their ten shared frames:
      - on the 26 study sounds: 23 of 26 flags agree, kappa 0.77, ICC of
        logit q 0.89;
      - as a held-out replication, on the 99 corpus labels from the 36 videos
        outside the study: kappa 0.55, ICC 0.77 (all 127 labels: 0.61 and
        0.80).
    - **Reported as methods notes:** the full-clip run's lost tracks, and the
      3 sounds that only the 1 fps run puts on screen (amsterdam_057's male
      speech, bangkok_328's engine, singapore_067's idling truck).
  - **The former filters become a stimulus table.** Per sound: q in the
    windowed and 1 fps runs, detection rate, shared-mask pairs, and whether its
    sound group has a visual category. Per clip: loudness range, channel
    level difference and first-caption fidelity.
  - **First-caption fidelity replaces the contradiction flag.**
    - Two Korean readers listen to each clip and code its first caption: the
      share of A1s sounds it names, and the number of sounds it names that
      are not in the audio. Kappa is reported.
    - It enters the model as a covariate (above).
    - bangkok_034's first caption leaves the horn out rather than naming a
      wrong sound.
  - **Draft quality is dropped as a clip measure.** It measures the generator
    and its review hints, and every caption shown is approved.
  - **Influence.** Leave-one-clip-out refits of every confirmatory term are
    shown as a forest plot. A clip whose removal flips a sign is reported.
  - **Next:** re-run the power check with mode slopes by clip before N is
    fixed.
- **The 12 condition-2 clips are confirmed as the phase-1 materials**
  (researcher, 2026-10-26).
  - **What the set is.** The caption-bearing condition of the earlier corpus,
    one clip per cell of city (3) × feature space (2) × cluster type (2). It
    is a stratified sample, not a powered one.
  - **Why no subset is drawn.** The clips' scene profiles over the full 10 s
    (`checks-20261026/profiles.png` in the review folder) give no
    measurement-based reason to drop any clip.
    - The sounds fall into three temporal types: persistent, intermittent,
      and mostly undetected. The types mix within clips: only 2 of the 12
      clips contain a single type.
    - No profile measure has a gap that survives the comparison with the
      1 fps run.
    - A bar from the design's own logic (every sound keeps its on-screen
      status for at least 80% of the clip, in both runs) would keep only 3
      clips.
    - So a subset would select sound types, not measurement quality.
  - **What the paper states.** The stimulus set and its stratification; that
    the count of 12 was inherited; and the mode-by-clip slope variance the
    pilot estimates.
    - That variance bounds generalisation: the SE of the mode terms cannot
      fall below (clip-slope SD) / √12, for example 0.087 at an SD of 0.3.
    - Claims are restricted to these scenes accordingly.
  - **Each sound's temporal profile** (persistent, intermittent, mostly
    undetected) goes into the stimulus table as a descriptor. So does the
    unlabelled share of each frame, which counts against visibility. Neither
    is used to select.

**Second captions: responsive, written live** (researcher, 2026-10-26,
after the check of the scene-aware rewriting, §11 of `checks-20261026.md` in
the review folder).
- **The goal.** The rewrite steers the audio caption from what the scene
  holds to what this participant perceived and preferred, in the
  attention-restorative direction. It is not a scene-descriptive rewrite.
- **What steers the text.** Two answers, in both modes alike:
  - the noticed sound (A1s), which the caption centres;
  - the restorative appraisal (P4).

  V1 and FA3 do not steer the text. V1 still enters the binding model.
- **Reflect.** The emphasis is the soundscape subscale on which the clip
  stands highest for this participant. The subscales are fascination, being
  away from, and extent and coherence (Payne, 2013). Each subscale's mean on
  the clip is taken minus the participant's own mean of it over their clips,
  so neither scale use nor a subscale rated high everywhere moves it. A tie
  goes to the higher raw mean, then to the questionnaire's order.
  - The place's being away (P1–P2, Hartig et al., 1997) is left out: it
    rates the picture, not the sounds the caption describes.
  - Each subscale has one sentence, the same under both modes, and none
    uses the items' own wording (`phase1_tweak.EMPHASIS_KO`).
- **Live, made robust by engineering.** Every caption is written when stage
  A ends. The approval of every caption before the study (2026-10-24) is
  retired, with `diff.phase1_approved` and `phase1 import-second-captions`.
  - A gate (`diff.phase1_usable`, v4) adds to the technical check the
    defects the 2026-10-24 bench and the 2026-10-26 drafts showed it letting
    through:
    - decoding damage: loops, cut or doubled words, a caption cut off
      mid-clause;
    - near-copies, and lists of names: two or more comma-separated phrases
      with no predicate (v2; researcher, 2026-10-26: a list is no rewrite);
    - an unnamed attended sound, with the bare syllable for "car" counted among the vehicles (v3);
    - a named R2 distractor;
    - the items' wording: every P4/P9 item's own words since v3, against
      four stems before; and the emphasis sentence's own words (v3), since a
      caption should carry the quality, not state it;
    - a place in an acousmatic caption, or one the prompt did not give in a
      scene-aware caption.
  - The caption's ending stays free (2026-10-18), so a caption may end on a
    noun but not on a connective.
  - A refused caption is written again in the next wording of
    `rewrite_prompts.TWEAK_LADDER` (researcher: alternate wordings), since a
    greedy call repeats itself. This replaces "no quality retries"
    (2026-10-18). A first version varied Trida's τ instead; see the
    measurements below.
  - The clip's second step is skipped only when the last rung is refused.
- **What phase 1 tests.** Following or countering the recommended mode, as
  before. The emphasis is applied the same way under either mode and does
  not depend on the plan, so the mode contrast stays clean of it. Testing the
  modulation is phase 2's role, under the same rule (researcher). Nothing in
  phase 2 is applied yet; that waits until phase 1 is confirmed.
- **For the analysis.**
  - The emphasis is chosen from P4, the first measure of the P4→P9 change.
    The emphasised subscale is the clip's highest for the participant, so it
    will tend to fall back toward their mean at P9 whatever the caption
    does. A per-subscale reading must allow for that. The mode contrast is
    not affected.
  - Export v8 records the emphasis, whether it was a tie, the three centred
    means, and the delivering rung and τ. The delivering τ is a covariate.
- **Built:**
  - tweak policy v7; scene-aware prompt v4 and acousmatic prompt v5, each
    ending with `{emphasis}`;
  - the gate; flow v7 (the rung in each entry);
  - the server queues the next rung as soon as one is refused;
  - the audit recomputes the emphasis and checks the whole ladder;
  - the delivery bench (runbook §11) runs the 174 reachable variants down
    the ladder.

  Participant sessions wait again for the second-caption prompts' approval
  (`TWEAK_PROVISIONAL`), now with the gate's.
- **Verified:**
  - `make check` passed (1920).
  - Six stand-in participants walked the real setup with the fake worker.
    A third of the first-rung captions came back as copies of the first
    caption, which the gate refused.
  - All 72 captions were delivered: 47 at the first rung and 25 at the
    second.
  - The audit found 0 mismatches over 194 ladder checks, and flagged a
    tampered emphasis and rung.
  - Each stand-in's emphases spread over all three subscales.
- **Measured the same day** (delivery bench, 174 captions, 87 per mode;
  `artifacts/diff/bench/candidates-20261026-live/RESULTS.md`):
  - Trida does not deliver sentence rewrites at temperature 0. Asked for
    sentences, it cuts verbs mid-word and loops; asked tersely, it writes
    lists of names.
  - With lists refused, its best first-call delivery in the weaker mode was
    17 of 87, and τ 1.0 reproduced τ 0.9's text in 61 of 86 retries.
  - The researcher then asked for a different model sized for rewriting
    speed (Qwen3.5), with alternate wordings as the retries, so that no
    fallback is needed.
  - On the same gate (v3), Qwen3.5 at bf16 with thinking off, greedy, one
    call (weaker mode of 87; median time per caption; peak memory):

    | size | first call | median time | peak memory |
    |---|---|---|---|
    | 2B | 45–50 | 0.9 s | 4.2 GiB |
    | 4B | 71 (75/71) | 1.2 s | 8.6 GiB |
    | 9B | 72 (84/72) | 1.4 s | 17.7 GiB |

    The 4B and 9B rows use the prose wording; the 9B result is under its
    shorter wording, since it overruns under prose. The times are without
    the linear-attention kernels.
  - Two wordings on 4B (the prose one, then the short one) would deliver an
    estimated 84 of 87 in each mode.
  - The emphasis sentences were rewritten the same day: until then they used
    F2's, BF2's and E2's own words.
- **The writer** (researcher, 2026-10-26): Qwen3.5-9B replaces Trida for
  rewriting Korean captions. Phase 1's second captions switch now; phase 2's
  live captions stay on Trida until phase 1 is confirmed. The writer's
  footprint ruled out 4B's memory advantage, since only one model is meant to
  be loaded in the end.
  - **The profile** is pinned at revision c2022362
    (`model_profile.QWEN35_9B`). Every file and shard was hashed and matched
    the Hub.
  - **The call** is greedy with thinking off (`diff.qwen35_runner`). It runs
    in the repository environment: Qwen3.5 needs transformers 5, and Trida's
    code pins the generator environment at 4.57.3.
  - **The ladder** has three wordings: every-sound (the bench's R1), then
    checklist (R3), then prose (P2). Each was picked on the bench by the
    pre-written rule (`RULE.md`, round 3).
  - **The session's GPUs:** the writer runs on its own card (`live serve
    --writer-gpu`), since 9B and Trida do not fit one 24 GB card. A session
    holds the writer from its entry to the handoff.
  - **Integrated bench** (`scripts/diff_second_captions.py draft`, 174
    reachable captions): 173 delivered, 87 of 87 scene-aware and 86 of 87
    acousmatic.
    - By rung: 159 at the first, 11 at the second, 3 at the third.
    - The first calls reproduce the scratch bench's text in 174 of 174.
    - The one undelivered caption, bangkok_209's acousmatic one, names three
      long sound labels and overruns 80 characters in every wording.
    - The gate's shouting stem now matches its conjugated form; the bench
      found one false refusal of that kind.
  - **Through the server:** three stand-in participants walked phase 1 with
    the real writer, and every one of the 36 captions was delivered (35 at
    the first rung). The median time per caption was 1.2 s (p90 1.5 s). The
    audit found 0 mismatches, and all 37 logged prompts re-tokenize to the
    worker's hash.
- **The writer moves to the Claude API** (researcher, 2026-10-27). After the
  trials of 2026-10-24 to 26, phase 1's second captions are rewritten by Claude
  Sonnet 5.5 (`claude-sonnet-5-5`) through the Claude API, not by a local model.
  Phase 2's live captions stay on Trida until phase 1 is confirmed (moved to
  the same writer later that day; below).
  - **Why** (`artifacts/diff/bench/candidates-20261026-live/MODELS.md`, every
    run judged again by gate v4):
    - Trida damaged words, looped and wrote lists. Its best first call
      reached 16 of 87 in the weaker mode.
    - Every Qwen drifted into another script: Chinese paragraphs and English
      or Vietnamese words from Qwen2.5-7B, and Cyrillic or Chinese sound words
      from Qwen3.5-9B in 4–9 of 174 first calls per wording.
    - Qwen3.5-2B echoed the prompt's instructions.
    - To stay short, the Qwen models dropped listed sounds: 74 of 9B's 159
      delivered first calls under its best wording. The gate only flags this.
    - 9B needed its own GPU and a second Python environment.
  - **The call** (`diff.claude_writer`) runs inside the app's process, with no
    GPU.
    - Its settings are part of each job's identity: low effort, adaptive
      thinking, up to 2,048 output tokens, and Anthropic's server-side
      fallback (`fallbacks: "default"`).
    - API version 2026-10-21 takes no sampling parameters, so the output is
      not reproducible. Every request and response is logged instead, with
      the request's SHA-256, the served model, the request ID and the tokens.
    - A busy or unreachable API (timeout, 429, 5xx) counts as transient and is
      tried once more.
    - The prompts, the three-wording ladder and the gate are unchanged.
  - **Built:**
    - protocol v8, tweak policy v9;
    - tweak attempts v3, which log the API record;
    - export v10, whose `writer_model` column names the model that wrote each
      caption;
    - one writer per live card, so the writer never limits concurrent
      sessions;
    - the audit checks each logged request against the recomputed job and
      the stored writer model against the log;
    - the delivery bench drafts through the API (`--jobs` at once);
    - `diff.qwen35_runner`, the Qwen3.5 profile and `--writer-gpu` are removed.
  - **Verified:**
    - `make check` passed.
    - Three stand-in participants walked phase 1 through the app with the
      writer's real SDK client against a local stand-in for the Messages API.
      The stand-in scripted 429s and copied captions, and all 36 captions
      were delivered (24 at the first rung, 12 at the second).
    - Every one of the 61 requests carried the pinned model, API version and
      fallback header, and the audit found 0 mismatches.
    - This checks the integration, not the captions.
  - **Measured on the API** (delivery bench, the 174 reachable captions,
    `artifacts/diff/phase1/second-captions-api-20261027/`):
    - 169 delivered, 168 of them at the first wording, with no output in
      another script.
    - 32 delivered captions were flagged for a dropped sound, against 78 for
      Qwen3.5-9B. 48 were flagged for a spatial relation, against 21.
    - The median call took 1.5 s, the slowest 2.9 s. The run cost about $0.40,
      with no fallbacks.
    - Two of the five misses were the matcher's errors, not Claude's:
      - the idiom for "without pause" contains the one-syllable word for bird,
        so it read as naming the bird distractor;
      - the noun and the plain form of "shout" were not counted as naming the
        attended shout.
    - The two later wordings ran over 80 characters in 8 of their 10 calls.
  - **Then, the same day** (researcher: fix the matcher, and retry the first
    wording):
    - **Gate v5:** a one-syllable sound name counts only as a noun, followed
      by a plural, the word for "sound", or a particle; the shout stems gain
      the noun and the plain form.
    - **Tweak policy v10:** the ladder is every-sound twice, then checklist,
      then prose. The second try is the same job at the next rung, so the
      writer services tell queued tweaks apart by job and rung.
    - **A race fixed.** A page poll reads the pending attempts and then queues
      them; an attempt that finished in between was queued and run again. The
      walk caught it: a delivered caption's retry ran twice. A service now
      refuses any attempt that has already run for the session.
    - **The bench again:** 174 of 174 delivered, 171 at the first try and 3 at
      the second; checklist and prose were never reached. No output in
      another script. 24 delivered captions were flagged for a dropped sound,
      45 for a spatial relation. The median call took 1.5 s and the p90
      2.1 s; 6 of 177 calls took over 3 s, the slowest 9.1 s.
    - **Through the server,** with the stand-in API: two walks of three
      participants each. Every caption was delivered, the first wording's
      second try was queued as the same job in 14 clips, no attempt ran
      twice, and the audit found 0 mismatches.
- **Phase 2 moves to the same writer** (researcher, 2026-10-27, the same
  day). Phase 2's live captions are written by the same Claude Sonnet 5.5
  call as phase 1's second captions, not by Trida. Nothing in the participant
  app runs on a GPU any more.
  - **What changed:**
    - a live job takes the writer's call (`LIVE_CONDITION` is the writer's
      condition), with the session's tuner text as the system prompt;
    - `live serve` drops `--trida-dir` and `--gpus` and takes `--sessions N`,
      one writer service per concurrent session, which writes that
      session's second captions and then its live captions. The separate
      phase-1 writer pool is gone;
    - `live preflight` checks the API with one short call instead of
      Trida's snapshot and the GPUs' memory;
    - live attempts (v5) log the API record, and the audit checks each live
      request against its recomputed job. A busy or unreachable API is
      retried once by the scheduler, so the audit no longer counts it as a
      mismatch;
    - the prompt bench's `run` writes through the API.
  - **What did not change:** the scheduler, the nine cell prompts, the tuner,
    the checks and the display rule. The offline G0 and plan commands
    (`generate`, `g0`) still run Trida and Qwen2.5-7B in the generator
    environment; no participant session uses them.
  - **The cell bench** (`artifacts/diff/cell-prompts/`): the same 30 cues
    under all nine cells and the default tuner, scored by `cell_metrics_v4.py`.

    | | Trida (v4b, 2026-10-22) | Claude Sonnet 5.5 (2026-10-27) |
    |---|---|---|
    | copies of the starting caption | 9% | 0% |
    | unusable (check failure or repetition) | 7.8% | 0% |
    | visibly broken | 6.9% | 1.5% |
    | distinct captions per cue, of 9 | 9 | 9 |
    | cells passing the texture rule | 2 of 9 | 6 of 9 |
    | cells passing the context rule | 4 of 9 | 6 of 9 |
    | time per caption, median / p90 / slowest | 0.7 / 1.1 / 3.2 s | 1.6 / 2.3 / 6.3 s |

    - The v4 rule is still not met. On texture, texture 1's relation and
      specific cells have exactly one sound word in 43% of rewrites (the
      rule asks 60%), and texture 0's relation cell has none in 73% (80%).
      On context, the three broad cells pass in 57–67% (70%).
    - The 270 calls cost about $0.66, with no fallbacks.
  - **The cost of the change:** a slider move shows about 2–3 s after it is
    made, against about 1–1.5 s with Trida. The scheduler aims at the
    playing window one latency ahead, the p90 of the last 20 calls.
- **Open for the researcher:**
  - the three emphasis sentences and the gate's rules (provisional);
  - the writer's three wordings (provisional), after a native read of
    Claude's drafts (the sheet in
    `artifacts/diff/phase1/second-captions-api-20261027/ladder-v10/`). Two
    things to look at: Claude adds spatial words such as "beyond" more often,
    and one delivered caption (amsterdam_057, scene-aware, fascination)
    paraphrases the emphasis sentence, which the echo check, built on its
    exact stem, lets through;
  - what leaves the machine: each prompt, with the attended sound and the
    emphasis but no ID, now goes to Anthropic. Whether the consent text must
    say so;
  - the nine cell prompts and the tuner, written for Trida, on Claude's
    output: a native read of the cell bench, and whether to tune texture 1
    and broad context against the v4 rule;
  - whether the slower response to a slider move is acceptable;
  - what the scene-aware prompt keeps of the scene. The setting is a
    between-clip contrast that misnames some scenes, and its "here, though
    unseen" sentence was never followed. Four clips have no sound with a
    source on screen, so a countered scene-aware caption there can only bind
    to the scene itself;
  - a gate for sounds the first caption names but the audio lacks. It needs
    those words per clip in the setup; the bench lists them for four clips.

**Open for the researcher:**
- the on-screen line (0.5) and the `PLAUSIBLE` table, which now decide
  visibility on their own;
- the parameters: 0.2 drawn, the 0.05 floor, the 1.5% click tolerance, ln 3,
  the uniform tendency prior, the 0.5 threshold, the sky cut-offs (which
  reach no caption since 2026-10-26), the 1.5
  lift, and the `PLAUSIBLE` table;
- the 29 drafted `audioset_label`s;
- the two prompt texts and the Korean element and off-screen phrases;
- the balanced plan, and the primary outcome subscale. The hypothesis team
  suggested the sign of scene × centred E1/E3 on BF1_2, with the built
  model's positive sign as the rival. That was before the moderator split,
  under which centred E1/E3 is part of `u`. The sheet also notes that BF1's
  loading shifted with environment type in Payne (2013) and may be unstable
  for urban video.

Until then `BINDING_PROVISIONAL` and `TWEAK_PROVISIONAL` refuse participant
sessions, and the setup's clips need approving again.

## Readiness fixes (2026-10-20)

The 2026-10-19 smoke run with three agent pseudo participants on real Trida
found the wiring correct and the captions weak. The researcher asked for the
engineering fixes found on the way to the user test.

- **The watch log names the video.** Every live event that carries
  `video_index` (`watch_started`, `settings`, `playback`, `exposures`,
  `video_ended`) now also carries `video_id`. The watch order is the serve's
  `--video` order, so the index alone did not say which video an event
  belonged to. `scripts/diff_session_audit.py` uses the ID, and for older logs
  infers it from the order of the attempts.
- **The display rule** (researcher's decision, relaxing "shown as written" of
  the live redesign of 2026-10-11). Trida often opens with a prompt label
  (the word for caption and a colon, 7 of 163 live calls in the smoke
  baseline) or wraps the caption in
  「」 or quotes (7 of 36 second captions; 87 of 163 live calls under one probe
  wording). What a participant sees of a live caption or a second caption is
  now Trida's text under `output_checks.DISPLAY_RULE` (`diff.display/v1`, v2 since 2026-10-22):
  - a leading label is removed (the words for caption, rewritten caption,
    answer or output, followed by colons);
  - one pair of quotes or 「」『』 wrapping the whole text is removed, only
    when that pair appears nowhere inside it;
  - nothing inside the text changes, so a broken word is still shown;
  - version 2 (researcher, 2026-10-22): a run of commas inside (",,", ", ,")
    becomes one. ",," was two-thirds of the visible damage in the nine-prompt
    bench; a stuttered word is still shown.

  The checks of those captions run on the shown text, so a quoted copy is
  still flagged as a copy. The attempt logs (`diff.live-attempt/v2`,
  `diff.tweak-attempt/v2`) keep Trida's text and add `shown_text` and
  `display_rule`. The offline G0 path is unchanged. The audit checks every
  shown text against the rule, and the bench scores copies and texture words
  on the shown text but still counts the quotes and labels in Trida's own.
- **A caption never goes blank** (researcher, 2026-10-20). On 13,216 real
  Trida outputs from 40 synthetic participants and all nine slider cells,
  Trida twice wrote only the prompt's caption label and stopped, which the
  display rule left empty. Nothing makes Trida keep the caption it was given.
  - A live caption with nothing readable left (no letter or digit) shows the
    cue's starting caption, the text Trida was given to rewrite. The attempt
    log (`diff.live-attempt/v3`) records `shown_from`: `trida` or
    `starting caption`.
  - A second caption with nothing readable fails ("nothing readable after the
    display rule", or the check's `empty_text`), so that clip's second step is
    skipped, as for any failed second caption.

  Replayed through Trida (deterministic), both cases write the label again; the
  live one now shows its starting caption and the second caption fails. A
  15-session run of the app on both GPUs (unattended pseudo participants,
  seven answer profiles, seven slider styles) passed the session audit, and
  every caption the browser showed was the logged shown text.

- **Participant sessions need reviewed starting captions** (researcher,
  2026-10-20). The phase-2 starting captions come from models (an audio
  captioner, then a translation), and `live serve` checked the tuner, the
  second-caption prompt and the setup, but not them.
  `caption_sources.starting_caption_review_problems` now refuses a cue whose
  status is not `reviewed` or whose text provenance is not human or reviewed,
  and `live serve --purpose participant` refuses while any cue it would serve
  has a problem. Unlike a bundle, a live cue needs no evidence events. Smoke
  runs still serve drafts.
- **Phase-2 starting captions from a captions table.** The researcher writes
  the starting captions per 10 s window with an audio captioner and a
  translation, as a CSV. `python -m diff sources import-captions` turns that
  table into the cue file, refusing a gap between windows or a caption over 96
  characters, and marks the cues reviewed only with `--reviewed-by`. The first
  table (2026-10-20) failed on length: its 90 Korean captions were 173–667
  characters, so the researcher is re-running the pipeline for one sentence of
  at most 96 characters per window.

- **One list of what stops the study.** `live serve` stopped at the first
  unmet participant gate, so an unready setup showed its problems one serve
  at a time. The gates now come from one function, `serve` reports all of
  them, and `python -m diff live preflight` lists, without serving:
  - the gates;
  - each video's cues against its real length;
  - the post-watch items and the Trida snapshot;
  - each GPU's free memory.

  On 2026-10-20, with the long videos and the M1 captions, it listed: the
  13 unapproved setup parts, the provisional tuner and second-caption
  prompt, and 90 unreviewed starting captions.

- **Every window shows a caption about its own audio** (researcher,
  2026-10-20). A move in a window's last ~1.4 s (the 300 ms debounce plus
  Trida's ~1.1 s) lands one window later, and the browser then kept the
  previous window's caption over the next window's audio for its whole
  10 s: 16 of 360 windows in the M1 rehearsal. Live captions whose check
  failed or looped were shown as written (3.8% loops on M1). Now
  `LiveScheduler.ready` serves, for every window, the first of:
  - Trida's usable caption under the current settings (`current`);
  - its latest usable caption for that window under earlier settings;
  - the window's starting caption (`starting-caption:<video>:<cue>`).

  Usable (`live_generation.usable`) means something readable is left under the
  display rule, and the live check found no failure and no repetition. On the
  M1 timing run, 74 of 810 rewrites (9%) are not usable. This replaces the
  blank-only fallback. Start waits until the first window's attempt is
  `settled`, so a video still opens with Trida's rewrite at the middle
  settings. The browser keeps its cached captions after a move and marks a
  change applied only when its window's caption is `current`. The waiting
  chip now says that the earlier setting's captions show while the new ones
  are prepared (wording for the researcher's approval). Attempt logs move to
  `diff.live-attempt/v4` (`usable`), and the exposure log names the job or
  the starting caption each window showed. The audit checks that every
  window showed its own caption.

- **A change shows as soon as it is written** (researcher, 2026-10-21,
  replacing "a caption changes only at a cue boundary"). Trida writes a
  rewrite in about 0.6 s, but the page waited up to a whole 10 s window to
  show it.
  - The scheduler writes a tweak first for the cue playing at `p + L`.
  - The page swaps a caption written under the current settings into the cue
    on screen as soon as it arrives, polling every 0.5 s instead of 1 s.
  - Every caption stays on screen at least 3 s (researcher, 2026-10-21: "for
    proper reading, and preventing instant rewrite by mistake"). The page
    swaps only after the caption on screen has had 3 s and while 3 s of its
    cue are left (`MIN_READ_MS`), and never while a newer slider move is
    still on its way to the server.
  - If that cue would end within `MIN_SHOWN_MS` (3.5 s: the 3 s plus one
    poll) of `p + L`, the tweak goes to the next cue.
  - A swapped cue logs two exposures, before and after the swap.

  The chip no longer counts down to a boundary: it says the caption is being
  rewritten with the new settings, then "applied". The handoff briefing now
  says a changed setting reaches the captions shortly, and its figure turns the
  caption the change was made in. The researcher approved both wordings on
  2026-10-22, on condition that neither covers the video: the chip has its
  own row below the sliders (checked at 1280 and 390 px wide), and the
  briefing is on the handoff page, before the watch.

- **Nine live prompts, one per slider cell** (researcher, 2026-10-21: "9
  different prompts for rewriting the original caption to have noticeable
  semantical adjustments", then "polish the prompts to be kind enough to
  deliver as soundscape descriptors"). `live-p2` v3 was one prompt with a
  texture rule and a context rule; on the final captions 38 of 54 rewrites
  under the default tuner were copies, so the sliders barely showed.
  - `rewrite_prompts.LIVE_CELL_PROMPTS` (`live-p2` v4, rules version 3): each
    cell writes a short description of the sound scene. Context: broad groups
    the sounds into general kinds but keeps what they do; specific names each
    source and what it does; relation places the sounds in space around the
    first. Texture: how they sound, on no sound (0), the first sound (1),
    every sound (2).
  - The cue comes last, after three invented examples (park, forest, market)
    shown with their rewrite for that same cell.
  - Chosen on the 90 final captions × 9 cells at the study's settings, by a
    rule written before the first run (`artifacts/diff/cell-prompts/RULE.md`,
    tightened twice, never loosened). Bench of 2026-10-21 on every third cue:
    an example after the cue was copied into unrelated cues (human and animal
    sounds, for traffic); no example left 56% unchanged; the soundscape form
    brought copies to 10% and gave every cell its own caption.
  - The nine prompts do not pass that rule yet: 17% of rewrites show broken
    text, two-thirds of it ",," (collapsed on display since 2026-10-22,
    leaving 7%); broad keeps source names, and texture counts drift. They are
    provisional (`LIVE_PROMPTS_PROVISIONAL`), so participant sessions are
    refused; smoke runs use them.
  - The audit checks each call against its own cell's template.

## UI review fixes (2026-10-19, "Diff Flow Review and Fixes")

The researcher supplied a review of the participant flow (14 findings) and an
improved design, and asked for its fixes to be applied. The weak spots were
the seams: where the stage changes, where the app changes, and where the
screens end before the protocol does.

**Applied:**
- **One caption band (F1).** The watch uses phase 1's black band below the
  video (24/38 px, line breaks kept, grows), not a text-shadow overlay on the
  frame. Phase 1's S1–S3 seed the tuner, so both phases now judge the same
  presentation.
- **Stepped sliders (F4).** Each slider has the three stops its axis really
  has: texture 0–2 and context broad/specific/relation. The browser sends 0,
  0.5 or 1, only when the stop changes, so the log no longer carries
  meaningless 0.01 moves. `slider_steps` is unchanged.
  - A chip follows each change: a countdown to the next cue boundary, then
    "applied" for 2 s, or "the previous caption stays" when the boundary came
    before the new caption.
  - Once a change is accepted, the browser drops the cached captions for
    later cues written under the old settings, so a boundary keeps the
    previous caption, as the scheduler intends.
- **Where am I (F2, F14).** Every page except the watch has a six-segment
  session progress bar (introduction, practice, first viewing, second
  viewing, long videos, interview), sized by expected time.
  - Clip pages name stage 1, the first or second viewing and the clip's
    number out of N, with the count repeated as dots.
  - The watch header shows the long video's number out of V and the elapsed
    time.
- **Stage-B interstitial (F2).** A new page, `interlude`, sits between the
  last P5 and the first P6, and is phase 1's one natural break. It says the
  first viewing is over and the N clips will be watched once more, and has a
  start button.
  - It does not say that the captions differ: P0 already does, and saying it
    again would prime D1.
  - It is closed by `interlude-complete`, which is analysis-excluded.
  - The page order changed, so the protocol is now `diff.phase1/v3`.
- **The end of the screens (F3).** The last screen says the on-screen tasks
  are over and asks the participant to take off the headphones where they
  sit and call the experimenter, because a short interview follows.
  - The interview (P12) and debrief (P13) stay outside the app, as decided on
    2026-10-18. The session still finishes when the ratings are saved.
  - Holding the researcher's next-participant button for 1 s returns to the
    entry.
- **One theme (F5):** `live.html` links `participant_theme.css` and has no
  palette of its own.
- **Briefing (F6):** the handoff is phase 2's briefing:
  - what the sliders change;
  - that a change shows from a later caption;
  - that the video keeps playing while they steer.

  The long videos' count and length come from the videos served.
- **Hidden tab (F9):** the watch pauses behind phase 1's overlay, which covers
  the sliders. The playback log records `hidden`.
- **Ratings (F10):** the post-watch items use phase 1's scale cards, answered
  count and sticky bar.
- **Waiting (F11):** while a second caption is written, P6 shows an
  indeterminate bar (still under reduced motion). After 20 s it adds that
  this is taking a while and asks the participant to tell the researcher.
- **Items in cards of three (F12)**, with no subscale names. P9 keeps its
  shuffled order.
- **Validation (F13):** the missing-item message moved into the sticky bar,
  with a link that jumps to the first missing item.
- **Found by the walk:** on the second long video, Start stayed enabled from
  the first video. A click before cue 0 was ready got a silent 409. Start and
  the caption cache are now reset for each video.

**Not applied:**
- **F7, practice viewing.** The practice clip has no prepared caption, and
  the researcher chose to remove the practice viewing pages.
- **F8, clip order (lag-1 interleave).** This is the researcher's decision,
  not a UI change.
- **The in-app interview checklist (design H6).** It would contradict "P12
  and P13 happen outside the app".
- **The briefing's optional 20-second slider practice.** It needs a non-study
  clip.

**Wording, decided by the researcher on 2026-10-19:**
- **Slider labels approved:** none, a little and a lot for texture (step 0
  forbids texture words, so the design's "less" became "none") and broad
  kinds, specific sources and places and relations for context.
- **No steering goal:** the design's line saying there is no right answer and
  asking for the easiest caption to read stays out, so participants are given
  no goal for steering.
- **The watch's change chip stays as designed.**
- **Shorter screens:** the design's interstitial, briefing and end screen
  were too verbose for the user test, so each is cut to one or two lines:
  - The interstitial has no remaining-steps list and no line saying a short
    rest is fine.
  - The briefing has a one-line lead (each video lasts about N minutes, and
    the captions can be adjusted while watching) and three one-line points:
    two sliders change the caption; a change shows from the next caption; the
    video keeps playing without stopping (F6's "no pausing"). The hidden-tab
    pause is not briefed, because phase 1 already pauses the same way and the
    overlay explains itself.
  - The end screen has its title and one notice. The notice still asks the
    participant to stay where they sit, because keeping them in place for D1
    is F3's point.

**Verified:**
- the `tests/diff` suite, ruff and mypy;
- a headless Chromium walk with the fake worker (3 clips, two long videos of
  30 s and 15 s), with no page errors. Every chip state showed as intended:
  - pending, then applied at the next boundary;
  - a move 1.2 s before a boundary kept the previous caption, then applied
    one cue later;
  - a hidden tab paused behind the overlay;
  - the settings carried over to the second video;
  - a short press of the end button did nothing, and a held press returned to
    the entry.

## Phase 1 — two viewings per clip, the second caption by Trida (researcher, 2026-10-18, later)

The researcher read the questionnaire spreadsheet (regen's copy was
taken from the same sheet on 2026-10-04) and asked for phase 1 to follow it:
each 10-second clip has two steps. This supersedes "one viewing per clip"
and "it shows no model output" in the section below. The rest of that
section stands: the raw first captions, the options from `orig_text`, entry
by `P{n}`, the store and the export.

**Decided by the researcher:**
- **Two stages.** Stage A runs every clip's first step: P1 (viewing with the
  raw first caption), P2 V1, P3 A1s, P4 (11 ratings) and P5 FA1–FA3. Stage B
  runs every clip's second step **in the same order**:
  - P6, a viewing with the second caption;
  - P7 V2, P8 A2s and P9 (the 11 ratings with `_2` ids, shuffled as regen
    does);
  - P10 S1–S3 and P11 R1/R2.

  Then the handoff to the live watch, as before. The split gives Trida the
  rest of stage A to write each clip's second caption.
- **Second caption.** Trida rewrites ("tweaks") the clip's raw first caption,
  steered by that participant's step-1 answers. Every clip is tuned; no clip
  keeps its first caption. The answers are processed into the slots of a
  prepared prompt template, which names each direction and its degree in
  percent.
- **Failure.** If the output fails the technical checks or the worker fails,
  that clip's step 2 is skipped and the reason recorded. There are no quality
  retries; one identical retry follows a lost worker or device error.
- **R2's key comes from the prompt.** The sounds the prompt names count as
  mentioned. This is recorded as the intended key, not checked against what
  Trida wrote.
- **P0 and titles:** the sheet's text, word for word, including the
  two-viewings and interview sentences. The interview (P12) and debrief (P13)
  happen out loud, outside the app.
- **Long-video stage:** unchanged. The sheet has no items for it (marked
  undecided), so
  the post-watch items still come from the researcher.

**The sheet as of 2026-10-18.** Since regen's copy, one cell changed: R1 now
asks how far the participant read the captions while watching. diff now has
its own copy of the instrument:
- `workbook_instrument`, regen's `sheet_instrument` code unchanged;
- over `src/diff/items/workbook-20261018.json`, which is regen's snapshot
  with that cell updated, checked against the sheet's export on all 886
  cells.

This is plan Stage 4's detach for this module, done early.
`dpo.regen.sheet_instrument` is no longer imported.

**The tweak prompt** (`rewrite_prompts.TWEAK_P1`, `tweak-p1` v0; replaced on 2026-10-23 by the
binding model's two modes, above) was
**provisional**. `live serve --purpose participant` refuses until the
researcher approves or rewrites it. `phase1_tweak` fills the slots from that
clip's step-1 answers:

| Slot | From | In the prompt |
|---|---|---|
| `{caption}` | the raw first caption | the text to tweak |
| `{sounds}` | the labels of the mentioned R2 options (`orig_text` sounds) | the sounds heard in this video; also R2's intended key |
| `{attended_sound}` | the P3 A1s answer's label | the sound the viewer attended to |
| `{attended_object}` | the object under the P2 point, in Korean (the word for none, off every mask) | what the viewer saw first |
| `{fascination}` % | the F1–F3 mean | make the sounds' appeal and interest stand out |
| `{being_away}` % | the BF1–BF3 and P1–P2 mean | bring out a sense of getting away from routine |
| `{coherence}` % | the E1–E3 mean | show how the sounds belong to and fit the place |
| `{familiarity}` % | FA2 | name the sound sources concretely |

The degree is `round(mean / 6 × 100)`: it *reflects* the answers. FA1 and FA3
are recorded, not used.

**Open for the researcher:**
- **A. Degree direction:** reflect, as drafted, or compensate,
  `(6 − mean) / 6`. It is one constant, `phase1_tweak.DEGREE_RULE`.
- **B. The sound list in the prompt:** it grounds the tweak in the audio and
  defines R2's key. On the four clips whose raw caption contradicts the audio,
  the prompt then holds both.
- **C. The prompt's wording.**

**Built (commits `5f4837a` … `c2d69cc`):**
- **`workbook_instrument`:** diff's copy of the questionnaire, with a parity
  test (every page equals regen's except R1). The setup has no wording part
  any more; P0 and the titles are the sheet's.
- **`phase1_tweak`:** the slots, the job and the check.
  - The job runs at the live condition (τ 0.9) on a 128-token canvas.
  - The check fails only what cannot be shown as Korean text: empty,
    incomplete, broken characters, Latin letters, or over 160 characters.
  - Sentence count, lines and endings are not checked, because the captions
    it tweaks are raw.
- **`LiveService`:** runs a queue of tweaks on the session's claimed GPU
  whenever no video needs it. It applies the one-retry rule, and logs every
  attempt in `live-<session>.jsonl` with `kind: "tweak"`. A worker that
  cannot start now counts as a lost worker instead of ending the service
  thread.
- **`phase1_flow` (protocol `diff.phase1/v2`):** the two stages.
  - P5 records the clip's slots and marks its second caption pending.
  - The server queues the job after the answer commits and writes the outcome
    back.
  - P6 shows regen's line saying the next video is being prepared while it
    waits, and a failed
    clip is skipped (`skip-second`) without a participant step.
  - Pending captions are queued again at entry and at every phase-1 request,
    so a restart loses none.
  - P7–P11 bind to the P6 viewing.
- **Export (`diff.export/v2`):**
  - `phase1.csv` gains `viewing` and annotates A2s and V2;
  - `r2_intended_key` replaces `r2_mentioned`;
  - a new `second_captions.csv` gives one row per participant × clip.

**Verified:**
- the `tests/diff` suite, and a full check in a clean worktree for each
  commit;
- a headless Chromium walk with the fake worker (1280×800):
  - stage A, then stage B, then the handoff, in 5 minutes;
  - all 12 first captions equal the provided text;
  - all 12 second captions were the fake's placeholder built on the clip's
    caption;
  - nothing clipped, and no page errors (only the pre-entry 401 the start
    screen expects);
- a second walk with a fake that held one caption back 200 s and timed out
  another:
  - the waiting page showed on that clip's P6 and moved on by itself when the
    caption arrived;
  - the failed clip's step 2 was skipped with no participant step;
  - the walk reached the handoff, with 23 viewings.
- the same walk with real Trida on GPU 0 (smoke purpose, 2026-10-19). The
  walk answered every rating at the midpoint, so every degree was 50%.
  - Trida took 0.53 s per second caption (median; 2.2 s at most). The slowest
    was ready 2.25 s after its P5, and no waiting page appeared.
  - 11 of 12 passed the check. On `singapore_067`, Trida repeated the caption
    and then copied lines of the prompt, which failed the 160-character limit.
    That clip's step 2 was skipped, as designed.
  - What the provisional prompt produced (not a quality rating):
    - three outputs are bare lists of sound names where the caption was a
      sentence or a single phrase (`singapore_040`, `amsterdam_012`,
      `bangkok_034`). For example, `amsterdam_012`'s phrase about a running
      train became a list of train, police siren and vehicle sounds;
    - `amsterdam_181` copied its caption unchanged (flagged);
    - `bangkok_034` starts with a broken word (a word cut to its last
      syllable);
    - `amsterdam_202` repeats the word for engine;
    - `bangkok_316` kept the prompt's 「」 quotes.

    This bears on decision C.

**Not yet:** the researcher's decisions A–C, the approval of the setup draft,
and any human participant.

## Phase 1 — 2026-10-18: profiling only, rebuilt in diff (researcher's decision)

Phase 1 exists to build the participant's profile for the live phase-2 tuner.
It shows no model output, so Trida appears only in live phase 2 (superseded:
see the section above). This
supersedes the plan's phase 1 with Trida-written P6 tracks (§5–§6.3): there is
no τ comparison in phase 1, and so no blind review, G1/G2 gates or phase-1
stimulus bundle for it.

**Decided by the researcher:**
- **Where:** in diff, one app and one participant session. Phase 1 hands off
  to live phase 2 inside the app; the tuner system message is built from the
  phase-1 answers at that moment, with no researcher step in between.
- **Items:** regen's phase-1 instrument (`sheet_instrument`, the source
  workbook frozen on 2026-10-04): P0 consent and practice; per clip P1–P5.
  Validation and order rules follow regen's, item for item.
- **Superseded by the section above — one viewing per clip.** The workbook's second viewing (P6 with an opposite
  or maintained caption, and the repeated P7–P9) is dropped. P10 (S1–S3) and
  P11 (R1/R2) follow P5 and refer to the first caption. The tuner reads A1s
  and S1–S3 as before.
- **First caption: the prepared Korean caption, shown raw** (corrected later
  on 2026-10-18).
  - Each clip shows the Korean caption burnt into the delivered stimulus, as
    transcribed in `scripts/c2-korean-captions.json`.
  - It is shown word for word, with its line breaks and no length or line
    limit.
  - This holds for all 12 clips, including the four where it contradicts the
    audio (`amsterdam_012`, `singapore_040`, `amsterdam_057`, `bangkok_034`).
  - The assistant had read "use Korean orig_text for all" as its own Korean
    rendering of the English `orig_text`. Those renderings were shown in the
    first walks and are withdrawn.

  The video is regen's uncaptioned `videos_10s` footage, so only diff's
  caption band shows. There is no 2×2 congruence × depth assignment.
- **Clips:** regen's 12 c2 clips. Six of them were G0's development cues. The
  rule "development material is never a participant clip" protected Trida's
  phase-1 generation, and phase 1 now generates nothing, so the researcher
  accepts this overlap.
- **Ending:** the session ends after the post-watch items, with no P12
  interview prompts and no P13 debrief.

**Build decisions (researcher, 2026-10-18):**
- **Clips and order:** all 12 per participant, in a random order per
  participant from a recorded seed.
- **Entry:** the experimenter types the participant ID (`P1`, `P2`, …) on the
  start screen and the participant's pages appear. The same ID resumes an
  unfinished session; a finished one is refused.
- **Lab only:** the server listens on loopback or the lab network, with no
  off-campus edge.
- **Export:** a command on this machine (`python -m diff export`), with no
  export reachable from a browser.
- **Look:** regen's questionnaire pages.
- **P0:** the workbook's introduction text, as regen's P0, until the
  researcher adds consent text.
- **Setup material drafted from the prepared metadata, then reviewed by the
  researcher.** Participant sessions are refused until every clip is marked
  reviewed. Drafts:
  - the captions: copied raw from `scripts/c2-korean-captions.json`;
  - A1s options and R2 options, from `orig_text` (corrected below);
  - P2 objects, from the per-frame masks;
  - practice material, from one clip outside the 12 (`c1_video_audio`).

**Options come from `orig_text` (researcher, 2026-10-18, correcting the
draft):** phase 1 collects the responses to the prepared questionnaire about
the `orig_text` caption, so its options keep `orig_text`.
- The first draft built A1s options from each clip's AudioSet labels and
  dropped any label that is an ancestor of another (regen media staging's
  rule). That is not a study decision. It let the options drift from the
  caption. For example, `amsterdam_057` offered a trumpet, which
  `orig_text` never names, and `bangkok_316` lost the traffic it names.
- **A1s now has one option per sound `orig_text` names, in its order.**
  Nothing is dropped and nothing is added from the labels. Each option quotes
  the words it renders (`orig_phrase`), and the setup check refuses a phrase
  that is not in `orig_text`.
- **R2:** the mentioned options are the same sounds with the same phrases,
  and the distractors may quote none.
- **AudioSet labels** are kept, every one, only as the reference for each
  option's family.
- **Practice:** the options come from the practice clip's `orig_text`.
- **Export:** `phase1.csv` carries the chosen sound's `orig_phrase`.
- **Kept with the raw captions (researcher, 2026-10-18):** the options stay
  from `orig_text`. R2's answer key (`mentioned`) is therefore relative to
  `orig_text`. On the four clips whose caption contradicts the audio, the
  options name sounds the shown caption does not.

**Setup checks (researcher, 2026-10-18):** phase 1's goal is to collect the
participants' responses to the prepared captions, to tune phase 2. Checks the
assistant had invented were removed:
- the 2-line and 96-character caption limits, and the Korean-only check on
  captions;
- at least two A1s options (regen accepts one);
- at least one mentioned R2 option and one distractor;
- "two options quote the same words" and "a distractor may not quote
  `orig_text`";
- practice labels differing from the clips' (regen documents it but does not
  enforce it).

What remains:
- regen's own validation of every option list and answer;
- the media, frame and mask files, and 3–9 annotated frames for P2;
- one non-empty caption cue covering the clip;
- for each A1s option, a family the tuner knows and its `orig_text` words;
- R2's answer key on every option, with `orig_text` words on the mentioned
  ones;
- Korean-only option labels and titles;
- the researcher's approval, and an unchanged workbook.

**The caption band** copied from regen collapsed line breaks and had a fixed
height for two lines. It now keeps the caption's line breaks and grows past
two lines while the video gives way.

**Wording (researcher, 2026-10-18; superseded, the sheet's text is shown as is):** minimal edits for review.
- The workbook's P0 told participants about a second viewing and an
  interview, and its titles numbered the first and second rounds. Both no
  longer fit.
- The two sentences are deleted and the prefixes dropped. Item texts stay the
  workbook's.

### Built, 2026-10-18 (commits `cb70787` … `060cb23`)

- **`phase1_setup`** drafts the setup: media, frames, masks, `orig_text`,
  every AudioSet label with its family, and the wording edits.
  - `phase1 approve` binds an approval to a hash of each part's content.
  - The captions and options were drafted by the assistant into
    `artifacts/diff/phase1/setup-draft.json`, marked
    `assistant-draft-unreviewed`:
    - the 12 captions, copied raw from `scripts/c2-korean-captions.json`
      (1–3 lines);
    - A1s: every sound `orig_text` names, 2–3 per clip, each with its
      `orig_phrase` and family;
    - R2: those sounds as mentioned, plus 1–2 distractors absent from both
      `orig_text` and the clip's labels;
    - practice options from `c1` `amsterdam_006`'s `orig_text`.
- **`session_store`** keeps sessions in SQLite, keyed by participant ID, with
  regen's revision and idempotency-key contract. Sessions survive restarts.
- **`phase1_flow`** implements the pages:
  - items and validation come from `sheet_instrument`, and P2 matching from
    `points`;
  - regen's playback coverage rule is copied, with a parity test over 300
    random tick sequences;
  - payloads carry no clip key, family or R2 answer key;
  - `profile_inputs` gives the tuner its observations and ratings.
- **`participant_server`** is one app:
  - the start screen and phase 1 at `/`, with regen's pages copied at
    `dc06c7f`;
  - the handoff, which builds the calibration;
  - the watch at `/watch/`, with the GPU claimed at entry;
  - security headers and per-session opaque media tokens.

  `live_server` now serves sessions it is handed. `live serve` takes
  `--phase1-setup` and `--store`, and the hand-written profiles file is
  gone.
- **`export_bundle`** (`python -m diff export`) writes:
  - `participants.jsonl`;
  - `phase1.csv`, with clip keys, families, the chosen sound's
    `orig_phrase` and the R2 answer key;
  - `post_watch.csv`;
  - a manifest with file hashes.

**Verified:**
- unit and HTTP tests (the `diff` suite), and a clean-worktree full check on
  each commit;
- a headless Chromium walk as `P1` and `P2` with the fake worker:
  - entry, P0, practice, then 12 clips at about 10.2 s each;
  - a reload resumed on the same page;
  - handoff, watch, items and done, then the finished ID was refused;
  - no page errors;
- the walk's export: 2 sessions, 504 phase-1 rows.
- the same walk as `P3` with real Trida on GPU 0 (smoke purpose, synthetic
  60-second watch, 2026-10-18):
  - from the handoff to Start becoming available took 1.1 s, the time for
    Trida to write cue 0;
  - the system message carried this walk's phase-1 profile (the heard
    families and the S1–S3 means);
  - 12 calls at a median of 0.4 s, with fluent Korean captions;
  - no page errors.
- after the switch to raw captions, a headless walk of phase 1 with the fake
  worker (1280×800, 2026-10-18):
  - all 12 rendered captions equal `scripts/c2-korean-captions.json` exactly;
  - each renders as many lines as it has;
  - `singapore_067`'s three lines fill a 146 px band;
  - no caption is clipped, leaves the screen or covers the video.

**Not yet:**
- the researcher's review and approval of the drafts;
- any human participant.

## Model change — 2026-10-12: Dream 7B → Trida-7B (researcher's decision)

**Why.** Live phase 2 worked on Dream, but its Korean broke: about 2 in 5 live
captions on held-out material were broken (U+FFFD), looping or copied. At the
researcher's request, three other diffusion LMs ran the same comparison
(`artifacts/diff/candidate-compare-20261012/`):
- the same 11 held-out cues;
- the same live prompt, live-p2 v2 cells;
- the same checks.

| Model | Bad / 55 live | Median latency | Broken characters | Loops | Copies |
|---|---|---|---|---|---|
| Fast-dLLM v2 7B | 20 | 0.8 s | 18 | 2 | 0 |
| Dream (then current) | 22 | 5.8 s | 11 | 2 | 8 |
| Trida-7B, README system message (diagnostic) | 23 | 0.36 s | **0** | 10 | 13 |
| Trida-7B, generic system message | 32 | 0.36 s | 0 | 12 | 18 |
| LLaDA-8B-Instruct | 44 | 11.1 s (ineligible) | 16 | 10 | 13 |

The researcher chose **Trida-7B** (`trillionlabs/Trida-7B` @ `c37755e8…`). It
has a Korean-native tokenizer (no broken characters) and is about 15× faster
than Dream, and it is the brief's original model.

**Decisions (researcher, 2026-10-12):**
- **Dream's code is removed.** One diffusion path; git keeps Dream's code and
  runs. A guard test forbids any Dream module, profile, condition or call.
- **Decoding factor:** back to the brief's original threshold τ, with
  `trida_tau_070`, `trida_tau_090` and `trida_tau_100`. Live phase 2 runs at
  τ 0.9, the README setting.
- **Live delay follows the device:** no 6 s floor, and `L` starts at 1 s.
- **System message:** our own "audio caption tuner" role. It is built once per
  session at handoff from the participant's phase-1 answers (heard sound
  families) and caption ratings (S1–S3 means), and stays fixed through the
  watch. **The researcher writes the role text.** Until then the wording is
  provisional (`rewrite_prompts.TUNER_SYSTEM`, `TUNER_PROVISIONAL`) and
  `live serve` refuses participant sessions.
- **Build first;** prompt work on Trida's copies and repeated lines comes
  later.

**What changed (protocol v6):**
- **Environment:** `envs/diff-generator` moves to transformers 4.57.3 and
  einops, torch 2.5.1. Qwen runs there unchanged.
- **Profile:** `TRIDA_7B` is pinned, with every text file and shard hashed.
  The text hashes are those of §2.3.
- **Runner:** `trida_runner` replaces `dream_runner`. The call is exactly
  §4.4's:
  - blocks of 32 in small blocks of 8;
  - temperature 0 (`top_p` not passed);
  - no block cache;
  - mask/stop/pad 128012/128001/128004.
- **Canvas:** whole blocks, 96 tokens for phase 1 and live and 128 for phase 2.
  Measured Trida tokens per Korean character: 0.43 on average, 0.75 at most.
- **Randomness:** none, so the "RNG unchanged" check (AC-07) is back and
  Dream's seeds are gone.
- **Guard:** a prompt of exactly 32 tokens is refused, because Trida's
  `generate` does not prefill it. The first warm-up prompt hit exactly that
  length and the guard stopped it.
- **Qwen** has its own token constants.

**The reading rule changes back.** Wherever this plan still says k, Dream,
canvas ÷ steps, `alg` or seeds, read τ, Trida, whole 32-token blocks and "no
randomness". §2.3, §4.4 and the AC-05 boundary suite describe Trida again and
are current.

**Measured with Trida (RTX 3090):**
- **Automatic G0 run** on the six development cues
  (`artifacts/diff/g0-trida-20261013`, 72 jobs):
  - every job completed, with no technical failures;
  - the RNG state was unchanged on every call;
  - no broken characters and no empty outputs;
  - about 0.6 s per caption.

  | Condition | Script check |
  |---|---|
  | τ 0.7 | 18/18 |
  | τ 0.9 (rated) | 17/18 (one exact copy) |
  | τ 1.0 | 17/18 |
  | Qwen | 13/18 |

  G0's automatic script share at the rated τ is therefore 94.4%, against 95%:
  it fails by one copy. The researcher waived G0 for Trida on 2026-10-13.
  The decision is `g0/researcher/decision-dec-907de0dcf279ab025d45dcc7.json`,
  with outcome `waived` and evidence that keeps the 17/18 and the unmet script
  criterion.
- **Live serve with real Trida** (synthetic 60 s stream, headless walk):
  - median 0.4 s per caption, 1.6 s at most;
  - a tweak at 1.8 s showed at the 5 s boundary;
  - captions changed only at boundaries; ratings, then done.
  - The captions read as fluent Korean with no broken characters. One cue
    repeated its sentence three times.

**Open for the researcher:**
- the tuner role text;
- the re-prepared videos with their starting captions;
- the post-watch items;
- prompt work on copies and repeated lines.

## Model change — 2026-10-09: Trida-7B → Dream 7B (researcher's decision)

> **Superseded on 2026-10-12** by the change above: Dream is no longer used.
> This section and the run write-ups below stay as the record of what Dream did.

The researcher replaced Trida-7B with Dream 7B mid-way through Stage 1. The
follow-up message amends the brief.

**The models.**
- **Main:** `Dream-org/Dream-v0-Instruct-7B` @
  `05334cb9faaf763692dcf9d8737c642be2b2a6ae`, loaded with `AutoModel`
  (`DreamModel`), bfloat16, `trust_remote_code=True`. Like Trida it is
  text-only, so the evidence stays the only grounding.
- **Baseline:** `Qwen/Qwen2.5-7B-Instruct` @
  `a09a35458c702b33eeacc393d103063234e8bc28`, in Tri-7B's former role (review
  stage only by default).

**The decoding factor.** Dream has no unmasking threshold, so τ becomes
parallelism **k = canvas ÷ steps**. The phase-1 conditions are `original`,
`dream_k1`, `dream_k2` and `dream_k4` (plus optional `qwen`). Wherever this plan
below still says τ, Trida, Tri-7B, block, small block or threshold, read the
decoding condition k, Dream, Qwen, canvas and steps as defined here.

**What was verified against the pinned files.** All files were downloaded at
the exact revisions, and their git blob IDs matched the Hub.
- **Hashes:** SHA-256 for every file the follow-up lists is in
  `src/diff/model_profile.py` (Appendix B).
- **Token IDs:** `<|mask|>` 151666; `<|endoftext|>` 151643 (EOS and PAD);
  `<|im_end|>` 151645. BOS is 151643 in `config.json` and `<|beginoftext|>`
  151665 in the tokenizer; `add_bos_token` is false.
- **Context:** `config.json` says `max_position_embeddings` 131,072; the GitHub
  README at `31f94a6` gives a 2,048-token context, transformers 4.46.2, torch
  2.5.1 and a GPU of at least 20 GB.
- **Call behaviour in `generation_utils.py`:** every behaviour the follow-up's
  §3 lists holds.
  - Config built from `config.json` when no `generation_config` is passed (line 228).
  - Unknown keys dropped silently by `.update()` (236).
  - Mask padding to prompt + `max_new_tokens` (388).
  - `steps` full forwards with no cache and no early exit, logits shifted by
    one (409–412).
  - `origin` ordering by `torch.rand` (424).
  - top-p/top-k filtering before the confidence (62–65).
  - `int(n_mask·(1−s/t))` per step and every remaining position at the last
    step (437).
  - The mask search covers the prompt (410).
- **Legacy defaults:** in transformers 4.46.2, `PretrainedConfig.to_dict()`
  carries `temperature` 1.0, `top_k` 50 and `top_p` 1.0. Checked by running
  library code over `config.json`, with no remote code executed.
- **Qwen's shipped defaults:** `generation_config.json` turns on sampling
  (0.7 / 0.8 / 20) with `repetition_penalty` 1.05.

**Gating.** G0 (Korean feasibility) now gates everything else in Stage 1 (§7 of
the follow-up). AC-20: the bundle builder and the lattice generator refuse
without a passing G0 decision for the current Dream profile.

### Changelog entry — 2026-10-09, reason: researcher's decision

**Kept** (model-agnostic):
- `caption_sources` (schema, closed vocabulary, importers), plus a new
  `material` field: development, participant or synthetic;
- the job-plan structure (content-addressed jobs, plans, freeze);
- `generation_worker` (supervisor, deadline, one retry);
- the output-check rules other than token IDs;
- the CLI skeleton and the synthetic fixture (relabelled `synthetic`);
- the participant-flow, review, bundle, assignment and store designs (§5–§7).

**Changed:**
- `study_protocol` (Dream contract, canvas, k conditions, G0, open decisions);
- `model_profile` (Dream and Qwen pins and hashes);
- `rewrite_prompts` (explicit system message "You are a helpful assistant." for
  both models, and `render_cell` for G0);
- `job_plan` (k and Qwen settings, prompt guards for 2,048 / mask / BOS, AC-20);
- `output_checks` (cut at the first `<|endoftext|>` or `<|im_end|>` with the
  tail kept, `incomplete`, leftover masks, repetition and exact-copy measures);
- the lattice axis renamed `quality_limit`, cell IDs now `t*`, so it is never
  confused with decoding k;
- the generator env, now `envs/diff-generator` with transformers 4.46.2 and
  torch 2.5.1+cu124.

**Redone:**
- `trida_runner.py` → `dream_runner.py`;
- `tri_runner.py` → `qwen_runner.py`;
- new `model_runtime.py` (shared loaded-model, refusal, chat and RNG helpers),
  `analysis_decisions.py` and `feasibility_gate.py` (G0).

**Removed:**
- the Trida decode trace (`decode_trace.py`); Dream's trace follows a passing G0;
- `envs/trida-generator`.

No Trida path, alias or switch remains, and a test enforces it.

**Trida artifacts produced:**
- no Trida outputs exist, because nothing was generated;
- the Trida-7B snapshot, downloaded at 13:10–13:23 into
  `artifacts/diff/models/trida-7b-c37755e8` (15 GB, gitignored), never entered
  a bundle, fixture or test. It was deleted on 2026-10-09 at the researcher's
  request, together with its download log (Q16).

### GPU smoke, 2026-10-09 (synthetic prompts; not G0, not a quality result)

8 jobs over the synthetic fixture, run on GPU 1 (RTX 3090) in
`envs/diff-generator` (Python 3.11.15, torch 2.5.1+cu124, transformers
4.46.2).

**Outcomes:**
- **Loading:** every pinned file and shard verified. All four remote-code
  files Python imported matched their pins; so did the remote tokenizer at
  freeze.
- **Caps:** all 8 attempts completed. Forward calls equalled `steps` in all 5
  Dream calls, new positions equalled the canvas, and no call changed an RNG
  state. This is AC-07/AC-08 observed on real hardware for these calls, not the
  full AC suites.
- **Per-step commits:** exactly as predicted. k1 runs 0, 1, …, 1, 2 and k4 runs
  3, 4, …, 4, 5.
- **Latency:** Dream k1 took about 14 s on a 128-token canvas and 25 s on 192
  (with about 210 prompt tokens); k4 took 3.5 s; Qwen 0.5–2.8 s. The 30 s
  phase-2 timeout is close and is finalised at M0.

**Findings:**
- **Empty output:** one Dream k1 output was empty; it committed
  `<|endoftext|>` at the first position.
- **Exact copies:** two outputs copied the source exactly.
- **Split character:** one Dream k4 output began with U+FFFD. Byte-level BPE
  had committed the continuation byte of a Hangul syllable without its lead
  byte. The checks missed this until a new fail rule, `undecodable_bytes`, was
  added; it also counts against G0's script check.

**What was exercised:** the G0 report and blind packet were run on this smoke
run, with no identifier leaks in the packet, and `g0 decide` refused it.

These observations come from 3 synthetic prompts and say nothing about Dream's
Korean ability; that is G0's job.

### First G0 run, 2026-10-09 (not rated, not decided)

**Setup:**
- 72 jobs: 6 development cues (Q15) with the phase-1 prompt and 12 corner
  prompts, each under `dream_k1`, `dream_k2`, `dream_k4` and `qwen`.
- Run on GPU 1 (RTX 3090). Prompts were 216–315 tokens.
- Run directory: `artifacts/diff/g0-20261009`. The report and packet are in
  `g0/`.
- Every job was terminal, with no technical failures; caps and the RNG check
  held.

**Findings:**
- **Timeouts:** all 12 `dream_k1` corner jobs timed out against the 30 s
  placeholder. No other job did.
  - `dream_k2` on the same 192 canvas took 15.3–16.2 s for 96 steps.
  - Latency scales with steps (k1/k2 at 128 measured 1.97–1.98), so k1 needs
    about 31–33 s.
  - These 12 jobs produced nothing.
- **Empty outputs:** the canvas was all `<|endoftext|>` from position 0.
  - `dream_k1`: 4 of its 6 completed outputs.
  - `dream_k2`: 9 of 18.
  - `dream_k4`: never.
  - The six source captions measure 31 tokens at the median and 45 at most
    (Dream tokenizer). Completed outputs stopped after 8–46 tokens. Most of
    each 128/192 canvas is therefore end-of-text, and the canvas has not yet
    been set from measured counts (CV-08, M0).
- **Script check:** `dream_k1` 2/18 under the implemented rule, `dream_k2`
  9/18, `dream_k4` 18/18, `qwen` 15/18.
  - The implemented rule counts empty output, U+FFFD, leftover masks and jobs
    without output as failures, beyond the brief's Han/kana/Latin and
    `incomplete`. The brief's literal reading gives `dream_k1` 6/18, or 6/6
    over completed outputs only.
- **Qwen:**
  - It wrote Chinese once, behind a "-automatic" prefix that appears in three
    `bangkok_209` corner outputs. The prompt renders as the template
    specifies.
  - It broke the phase-1 sentence-ending rule twice.

**Status:** the packet exists, but it is not fit to rate. It holds 6 Dream
outputs, 4 of them empty, against 18 from Qwen, and the k1 corner prompts
produced nothing.

**Decided by the researcher on 2026-10-10 (M0; protocol
`diff.study-protocol/v3`):**
- **Timeout:** 60 s.
- **Canvas:** 80 tokens in phase 1 and 128 in phase 2, sized to the 96- and
  160-character limits.
- **Empty outputs:** an empty output, or a job with no output, fails the script
  check. The share is over every `dream_k1` job.
- **Evidence:** the six drafts are used as they are.

The second G0 run, under v3, is in `artifacts/diff/g0-20261010`.

### Second G0 run, 2026-10-10 (protocol v3; fails the automatic rule)

**Setup:** the same six development cues and 72 jobs, on GPU 1, with a canvas
of 80/128 and a 60 s timeout. All 72 jobs completed, with no timeouts and no
technical failures. Latency medians: `dream_k1` 16.1 s, `k2` 8.1 s, `k4`
4.0 s, Qwen 0.6 s.

**Empty outputs and script pass:**

| Condition | Empty | Script pass |
|---|---|---|
| `dream_k1` | 16/18: all 6 phase-1 cues and 10 of 12 corners | 2/18 |
| `dream_k2` | 8/18 | 10/18 |
| `dream_k4` | 0/18 | 17/18 (one U+FFFD) |
| `qwen` | 0/18 | 15/18 (the same outputs as run 1) |

**Findings:**
- **The smaller canvas did not cure the end-of-text fill at k1; it made it
  worse.** On the phase-1 cues, run 1 had 4/6 empty at canvas 128 and run 2
  had 6/6 at canvas 80.
- **The two non-empty `dream_k1` outputs** are two-word sentences about a
  car or vehicle sounding, the first misspelt.

**G0 outcome:** under the rule decided on 2026-10-10, `dream_k1`'s script
share is 11 % against 95 %. G0 fails whatever the rating, so the packet
(36 rows, 16 of them empty Dream outputs) is not worth a rater's time. The
fail is recorded without a rating as
`g0/researcher/decision-dec-95ec20c38bda25e5b83db947.json`, with naturalness
`null` (unrated), on run manifest `9982c8c4…`.

**What remains for the researcher (brief §7):** choose between revised
prompts on development material (a new protocol version) and another model.
The diagnostic below was run first, at the researcher's request, to inform
that choice.

### k1 diagnostic, 2026-10-10 (out of protocol; never G0)

**Setup.** The researcher approved this run. It used the six development cues,
all at k1 (steps = canvas), and was never G0 evidence.
- A scratch script, `artifacts/diff/diagnostic-20261010/k1_diagnostic.py`,
  called `diffusion_generate` directly. It bypassed `checked_call`, so it could
  try settings the protocol refuses.
- The script is gitignored, like its results (`gpu0.jsonl`, `gpu1.jsonl` and
  `summary.json`).
- The prompts, the cut at the first terminator and the output checks are the
  pipeline's own.

| Variant | Settings | Empty | Exact copy of the source caption | Script pass | Median latency |
|---|---|---|---|---|---|
| control | greedy, canvas 80 (run 2's call) | 6/6 | 0 | 0/6 | 8.9 s |
| greedy256 | greedy, canvas 256 | 5/6 | 1 | 1/6 | 49.6 s (max 66.6 s) |
| english80 | greedy, canvas 80, the same task and evidence in English | 4/6 | — | — | 7.1 s |
| readme80 | temperature 0.2, `top_p` 0.95, canvas 80, seeds 0, 1, 2 | 0/18 | 9/18 | 18/18 | 19.8 s |
| readme512 | temperature 0.2, `top_p` 0.95, canvas 512, seed 0 (the README call) | 0/6 | 3/6 | 6/6 | 122 s |

**Measured:**
- **The harness matches the pipeline.** The control's output token IDs are
  identical to run 2's on all six cues.
- **The empty captions come from the greedy decoding order, not from
  Korean.**
  - Under greedy decoding, every token committed in the first quarter of the
    steps is end-of-text.
  - Position 0 is the last position committed: step 79 of 80, or 232–255 of
    256.
  - The English prompt shows the same pattern and comes back empty on 4 of 6
    cues.
- **A larger canvas does not help.** It was still empty on 5 of 6 cues at
  256, at about 50 s a job.
- **The README's sampling settings remove the empties.** Position 0 is
  committed at step 1 or 2. At canvas 80, no end-of-text is committed in the
  first quarter.
- **What comes back is mostly the source caption.**
  - Exact copies: 9 of 18 at canvas 80, and 3 of 6 at 512.
  - The rest are near-copies, three of them prefixed with the caption label.
- **The seeds made no difference here, but the call still uses randomness.**
  - The three seeds gave identical text for every cue.
  - The call consumes RNG, so two protocol rules would have to change: the
    no-RNG cap (AC-07) and the refusal of non-zero temperature.
- **The README call is slow.** At canvas 512 it takes about 2 minutes a job,
  twice the 60 s deadline.

**Inferred from the pinned `sample_tokens`; not traced:**
- Logits are divided by 0.2 and truncated at `top_p` 0.95 before the entropy
  confidence is computed.
- A position whose top token holds 95 % of the mass therefore becomes one-hot,
  with confidence ≈ 0, the maximum.
- Many positions then tie, and `torch.topk` breaks the tie. That fits position
  0 being committed early, but the tie order on CUDA was not checked.
- If it holds, tie-breaking rather than confidence decides much of the
  unmasking order under these settings. That would change what k manipulates.

**RNG-free follow-up, 2026-10-11.** The researcher asked whether a
deterministic setting could do what the README's settings did. The run used
the same six cues at k1 and canvas 80, with the same script
(`rngfree.jsonl`). Every call left the CPU and CUDA RNG state unchanged (6/6
per variant).

| Variant | Settings | Empty | Copies of the source caption | Script pass | Median latency |
|---|---|---|---|---|---|
| control, rerun | run 2's call | 6/6 | 0 | 0/6 | 8.9 s |
| greedytopp80 | temperature 0, `top_p` 0.95, `entropy` | 4/6 | 1 exact, plus a fragment (the opening words of a sentence about the clip's main sounds) | 2/6 | 9.5 s |
| maskgit80 | temperature 0, `alg="maskgit_plus"` | 4/6 | 1 exact, 1 near | 2/6 | 10.1 s |
| margin80 | temperature 0, `alg="topk_margin"` | 2/6 | 2 exact, 2 near (one with U+FFFD) | 3/6 | 11.0 s |

- **The rerun of run 2's call is token-identical to run 2 again.**
- **No deterministic setting removes the empties.**
  - `top_p` does apply at temperature 0. In the pinned code it is not gated on
    sampling.
  - Without the division by 0.2, too few positions become one-hot, and
    end-of-text still dominates the first quarter.
- **Every non-empty output across these variants copies the source caption,
  nearly copies it, or is a fragment of it.** Copying is not specific to the
  sampling settings.

**What this means for the choice:**
- **Revised prompts under the fixed greedy call are unlikely to help on their
  own.** The canvas fills with end-of-text before any content is committed.
  The same happens in English, and with every deterministic setting tried.
- **Only the README's sampling settings made Dream write at k1.** They cost:
  - a new protocol version;
  - the no-RNG cap, which would become a recorded seed per job; the three
    seeds gave identical text for every cue;
  - possibly the meaning of k as confidence-ordered parallelism.
- **Dream copies the source caption under every setting that produces text.**
  - Any Dream path would need a revised prompt.
  - G0's rule does not count copies against Dream: `identical_to_source` is a
    flag, not a script failure. A rater shown human source captions could
    therefore pass the gate on copies.
- **Another model stays open.**

### Protocol v4, 2026-10-11 (researcher's decision)

The researcher chose seeded sampling plus a revised rewrite prompt, and ruled
that copies count against Dream. Protocol v4 (`diff.study-protocol/v4`,
`c3688aeb3765…`; prompt templates are hashed per job, not in the protocol)
implements the parts that do not depend on the prompt:

- **Decoding.** Dream samples at the README's temperature 0.2 with `top_p`
  0.95. `checked_call` requires exactly these two values and still refuses
  `top_k` and a non-zero `alg_temp`.
- **Seed.** Each Dream job's seed is the first 8 hex digits of its job ID, as
  an integer. `torch.manual_seed(seed)` runs immediately before the call, so a
  job is reproducible and its one allowed retry is identical. Every attempt
  records its seed.
- **Reproducibility replaces the no-RNG cap for Dream (AC-07, revised).**
  - The worker's warm-up makes two calls with seed 0, and the worker refuses
    to start if they differ.
  - It passed on the RTX 3090 on 2026-10-11.
  - Qwen stays greedy and keeps the no-RNG check.
- **Copies.** An output identical to its cue's source caption fails G0's
  script check, like an empty one (`script_check_failing_flags`). Near-copies
  stay flags.
- **Prompt.** `rewrite-p1` is now version 2. The brief's text stays in code as
  `PHASE1_REWRITE_V1`. The corner-cell prompt (`lattice-p2`) shows no source
  caption and is unchanged.

**How version 2 was chosen (2026-10-11, out of protocol; never G0).** The
researcher asked the assistant to prepare the prompt from the captions on
hand.
- **Drafts:** two, written from the c2 captions' styles.
  - `v2a` asks for a new sentence with the same meaning, forbids repeating the
    original sentence, and asks for new words and word order.
  - `v2b` is `v2a` plus one invented worked example (a distant dog barking).
- **Test set:** the drafts and version 1 ran at k1 under protocol v4 on 11
  held-out captions, never G0's six cues:
  - three c2 clips excluded from G0 (`amsterdam_012`, `amsterdam_057`,
    `singapore_040`), with evidence drafted from their Korean captions;
  - the eight synthetic phase-1 test cues.
- **Rule, fixed beforehand:** the most passes of G0's script check, where an
  empty output or an exact copy fails.

| Draft | Script pass | Empty | Exact copies | Copies apart from spacing |
|---|---|---|---|---|
| v1 (brief) | 9/11 | 0 | 1 | 2 |
| v2a | 8/11 | 0 | 0 | 0 |
| v2b | **10/11** | 0 | 0 | 1 |

`v2b` became version 2. The script check misses most of what is wrong with
these outputs, under every draft:
- echoed evidence lines (the prompt's numbered sound and source fields);
- echoed labels (the caption label, and under `v2b` four of 11 start with
  the rewritten-caption label);
- broken characters (U+FFFD) and stray Latin letters (an English word inside
  a Korean verb).

Output is never trimmed (brief), so the naturalness rating has to catch
these. Script and results: `artifacts/diff/prompt-precheck-20261011/`
(gitignored). The run started at 17:05 on GPU 0, once the researcher's
midsentence render had freed it, and warm-up's seeded calls agreed.

### Third G0 run, 2026-10-11 (protocol v4, `rewrite-p1` v2; passes the automatic rule, awaits the rating)

**Setup:**
- The same six development cues and 72 jobs, in `artifacts/diff/g0-20261011`.
- Frozen protocol `c3688aeb3765`; plan `plan-3194424c…`.
- GPU 0, 17:13–17:24. Every Dream attempt records its seed.
- All 72 jobs completed, with no timeouts and no technical failures.
- Latency medians: `dream_k1` 15.6 s, `k2` 7.8 s, `k4` 3.9 s, Qwen 0.7 s.

| Condition | Script pass | Empty | Exact copies | Starts with a prompt label (rewritten caption / caption) | U+FFFD |
|---|---|---|---|---|---|
| `dream_k1` | 18/18 | 0 | 0 | 10/18 | 0 |
| `dream_k2` | 18/18 | 0 | 0 | 10/18 | 0 |
| `dream_k4` | 12/18 | 0 | 0 | 4/18 | 5 |
| `qwen` | 15/18 | 0 | 0 | 0 | 0 |

**Findings:**
- **The empties and the copies are gone.** `dream_k1` passes the script check
  on every job, so G0's automatic criteria are met for the first time.
- **The rating decides G0.** It needs ≥ 3/4 of `dream_k1` rated 3 or higher.
- **Visible defects, in the output as written (never trimmed):**
  - 10 of 18 `dream_k1` outputs start with a label echoed from the prompt;
  - a dropped first syllable (car, missing its first syllable);
  - misspelt endings (a polite verb ending spelt wrongly);
  - 3 sentence-ending mismatches.
- **The corner cells barely respond to their cell.** For `amsterdam_181`,
  three of the four corners gave the identical sentence. That matters for G2,
  not for G0.

**Outcome:** the researcher waived the human rating on 2026-10-11, because
participants rate the live captions instead (see "Live phase 2"). The waiver
is recorded as `g0/researcher/decision-dec-c48cd5efa9c2ec1e87522dec.json`:
outcome `waived`, naturalness `null`. A waiver is never a pass, and only a step
that says so accepts it.

### Live phase 2 (researcher's redesign, 2026-10-11)

**What changed.** Phase 2 no longer looks captions up in a pre-generated
lattice.
- During the watch, Dream rewrites each cue's starting caption live, as the
  participant moves the texture and context sliders.
- There is no Gemma anywhere and no separate rater: G0's rating is waived.
- Participants answer the researcher's items after the watch.
- Phase 1 stays pre-generated.

The approved plan, with every decision: `~/.claude/plans/sunny-brewing-haven.md`.

**Decisions:**
- **Decoding:** `dream_k2`.
- **Timing:**
  - A tweak can be sent at any time.
  - It is written first for the first cue that starts at or after
    `position + L`, the earliest boundary where it can appear. `L` is the
    90th percentile of the worker's last 20 latencies, floor 6 s, start 7 s.
  - Captions change only at cue boundaries, and the previous caption stays
    when nothing new is ready.
- **Opening:** each video opens with Dream's rewrite at the calibrated
  settings, and Start waits for cue 0.
- **Calibration:** regen's sheet-handoff rule (`live_calibration`, policy
  `diff.live-calibration/v1`).
  - The sliders start at the midpoint.
  - The heard sound families from phase 1 become a Korean guidance sentence
    in the prompt.
  - Visual labels stay out of the prompt, because Dream echoes Latin letters.
- **Failures:** shown as written, never retried for quality. One identical
  retry is allowed after a transient worker failure.
- **Material:** long videos that the researcher re-prepares, with per-cue
  Korean starting captions. Regen's are in `data/live/regen-media/viewing/`.

**Built:**
- `live_generation`: the scheduler, latency estimator, cache, retry rule,
  service thread and attempt log `diff.live-attempt/v1`.
- `generation_worker`: a `live` message; the worker freezes the job with
  Dream's tokenizer.
- `dream_runner.freeze_live`.
- `job_plan.live_job`.
- `rewrite_prompts.render_live` (`live-p2` v1).
- `live_calibration`.
- `live_server` and `live_static/`: the participant app, event log
  `diff.live-event/v1`, opaque tokens, one session per GPU.
- `python -m diff live serve`.
- `python -m diff g0 waive`.
- `output_checks` v3, which adds the `label_echo` and `format_echo` measures.

**Prompt choice (held-out pre-check, `artifacts/diff/prompt-precheck-live-20261011/`).**
- **Material:** k2 at canvas 128 for live and k1 at canvas 80 for phase 1, on
  the 11 held-out cues: three excluded c2 clips and eight synthetic cues.
- **Rule, fixed beforehand:** the fewest outputs with a failure or a prompt
  leak.
- **`rewrite-p1` v3** (p3b) had 5/11 bad, against 10/11 for v2. Label echo
  went from 4 to 0.
- **`live-p2` v1** (lc) had 19/55 bad, against 20/55 for la.

**Live quality is the open problem.**

| Texture step | Exact copies | Broken characters (U+FFFD) |
|---|---|---|
| 0 | 6–9 of 22 | 1–2 of 22 |
| 2 (with guidance) | — | 5 of 22 |
| 4 | — | 5–6 of 22 |

- At texture 4 many outputs that pass the checks are still unusable. Dream
  dumps the quality list or loops on one syllable.
- The prompt cannot remove broken characters, because they come from
  decoding.

**GPU smoke (2026-10-11, RTX 3090, synthetic 60 s stream, headless walk).**
The machinery held up:
- Start waited for cue 0, and captions changed only at boundaries.
- Ratings, then done, with no console errors.
- RNG reproducibility held at warm-up.

Measured:
- **Latency:** k2 at canvas 128 took 8.9 s per caption.
- **A tweak at 1.8 s first showed at the 20 s boundary.** The worker was
  still writing cue 1 under the opening settings (a call cannot be
  cancelled). Then `position + L` = 17.5 s, so the first boundary was 20 s.
  The reveal delay therefore runs from `L` (about 9 s) to `L` plus one cue
  plus one in-flight call, about 23 s at worst.
- **The captions after the tweak (texture 4, relation) were mostly broken,**
  as in the pre-check.

**Re-measured on 2026-10-12 with protocol v5** (canvas 80, texture capped at
step 2), on the same walk:
- **Latency:** 5.4 s per caption, so the worker writes almost every cue
  fresh.
- **A tweak at 1.9 s first showed at the 15 s boundary, 13 s later.** That
  was about 3 s waiting for the call in flight, then `L` = 6 s, then the
  boundary.
- **Expected range:** about 6–17 s.
- **Quality is still the weak point:** for example, repeated syllables broken
  by U+FFFD, and an English word inside a Korean phrase.

**Decided by the researcher on 2026-10-12 (protocol v5):**
- **Live canvas: 80 tokens** (`LIVE_CANVAS_TOKENS`), at k2 with 40 steps.
  - That takes about 5.7 s per caption against 8.9 s at 128, so `L` settles
    near its 6 s floor.
  - Starting captions and rewrites must fit 96 characters (`LIVE_MAX_CHARS`).
    `live serve` refuses a starting caption that does not.
- **Wording approved:** the slider labels and the page text, as drafted.
  - Texture slider: how the sounds' texture is described, from less to more.
  - Source slider: how the sources are described, from broad kinds to places
    and relations.
- **Texture quality:** fixes were tested on held-out captions first.
  - Material: `artifacts/diff/prompt-precheck-texture-20261012/`, at canvas
    80, the same 11 held-out cues, 55 outputs per variant.
  - Rule, fixed beforehand: among the k2 variants, the fewest outputs with a
    failure, a prompt leak, an exact copy or a repetition loop. Ties go to the
    full range, then to keeping the vocabulary list.

  | Variant | Bad of 55 | Broken characters | Loops | Copies | Median latency |
  |---|---|---|---|---|---|
  | texture 0–4, vocabulary list, k2 | 25 | 14 | 6 | 7 | 5.5 s |
  | texture 0–4, no list, k2 | 30 | 19 | 9 | 7 | 4.9 s |
  | **texture 0–2, vocabulary list, k2 (chosen)** | **22** | 11 | 2 | 8 | 5.8 s |
  | texture 0–2, no list, k2 | 22 | 10 | 2 | 10 | 5.4 s |
  | texture 0–4, vocabulary list, k1 (not eligible) | 19 | 6 | 3 | 8 | 17.0 s |

  - **Adopted:** the texture slider spans steps 0–2 (`LIVE_TEXTURE_MAX`, live
    rules version 2), and `live-p2` v2 asks for at most 96 characters.
  - **Still broken:** about 2 in 5 live captions are broken, looping or copied
    on this material (U+FFFD in 1 in 5). k1 was best but too slow for the
    reveal delay.

**Still open for the researcher:**
- the post-watch items;
- the long videos with their starting captions.

### G0 and AC-20 in the traceability

| ID | Content (follow-up) | Modules | Tests |
|---|---|---|---|
| G0 | Korean feasibility before any other Dream work: development cues (phase-1 prompt) + ≥ 12 corner cell prompts; `dream_k1/k2/k4` + `qwen`; automatic report; blind packet (`dream_k1` + `qwen`) rated 1–5 by one fluent Korean reader; pass = all `dream_k1` jobs terminal with no unresolved technical failure, ≥ 95 % script pass over every `dream_k1` job (incomplete, empty, missing and, since v4, exact copies of the source caption count as failures), ≥ 3/4 naturalness ≥ 3; recorded as an `AnalysisDecision` with the run-manifest hash | `feasibility_gate`, `analysis_decisions`, `output_checks`, `dream_runner`, `qwen_runner`, `__main__ g0 …` | `test_feasibility_gate` |
| AC-20 | the bundle builder and lattice generator refuse unless a passing G0 decision exists for a run manifest with the current model profile | `job_plan.build_phase2_plan` (lattice generator), `analysis_decisions.gate_problems`; bundle builder in Stage 2 | `test_job_plan_identity` (AC-20 cases) |
| AC-01 | synthetic logits: recomputed confidences, counts and selected positions equal the pinned code's | after G0: vendored `dream_generation_utils.py` + decode trace | pending G0 |
| AC-02 | the §3 refusal list, including each missing required keyword | `dream_runner.checked_call`, `qwen_runner.checked_call` | `test_dream_runner_contract`, `test_qwen_runner_contract` |
| AC-03 | 3 inputs × 3 conditions: identical token IDs with and without hooks and history; hook selections = history unmasking + logged `selected_but_masked` | after G0 (decode trace) | pending G0 |
| AC-05 | context and canvas edges: 2,047 and 2,048 accepted, 2,049 refused before the call; invalid steps refused; very short prompts | `dream_runner.context_problems`, `checked_call` | `test_dream_runner_contract` |
| AC-07 | revised in v4 (2026-10-11): each Dream call is seeded from its job ID immediately before it runs, and the worker's warm-up refuses a device where two seeded calls differ; Qwen calls still leave the RNG states unchanged (`rng_consumed`) | `dream_runner.seed_all`, `dream_runner.warm_up`; `model_runtime.rng_snapshot` for Qwen | `test_dream_runner_contract` (fake); warm-up passed on an RTX 3090 on 2026-10-11 |
| AC-08 | forward-hook count equals `steps` | read-only forward hook in `dream_runner.call_with_caps` (`count_mismatch`) | `test_dream_runner_contract` (fake); GPU pending |

## 0. Read this first

1. **The PRD is not in this workspace.** `trida-korean-hci-prd.md` is not on
   this machine (full-disk search), not in git history on any branch, and not in
   the connected Google Drive or Notion. Every PRD section, FR, AC, MC, CV and IV
   cited below comes from the brief. §9 traces the brief's constraints in full
   and gives every FR-01–FR-24 and AC-01–AC-19 a row, but rows whose wording
   only the PRD holds say *pending PRD* rather than guess. Add the PRD
   (suggested path `docs/v4-diff/trida-korean-hci-prd.md`) and complete §9
   before Stage 1.
2. **The allowed regen imports load Gemma code transitively.** Importing any
   `dpo.regen.*` module runs `dpo/regen/__init__.py`, which imports
   `dpo.regen.regeneration` and through it `dpo.caption.writer`, the module that
   defines `GemmaWriter`. Measured at `3842302`: importing `sheet_instrument`,
   `playback`, `points`, `study_media` and `study_store` loads
   `dpo.caption.{writer,background}`, `dpo.core.atomic` and
   `dpo.regen.{assignment,captions,config,document,items,regeneration,study_instrument,study_schema}`.
   It does **not** load `torch`, `transformers`, `numpy`, `dpo.models`,
   `dpo.candidates` or `dpo.contracts` (`dpo.models.gemma4` is imported inside a
   method, never at module load). `src/dpo/` may not be edited, so diff cannot
   avoid this before Stage 4. Default: accept it, pin it with an
   import-boundary test, remove it at Stage 4. Alternative: detach the five
   modules at Stage 1 (Q2).
3. **Superseded by the model change.** This item recorded a defect in Trida's
   native `generate` (a prompt of exactly one block skipped prefill). Trida is
   no longer used. Dream's call contract, verified against its pinned
   `generation_utils.py`, is in the Model change section at the top.
4. **The stimulus materials do not yet have the shape diff needs.**
   - **Phase 1:** the live regen document has one cue per clip
     (`calibration.cue_slots: 1`). Diff needs four authored cues per clip with
     structured evidence, for the two participant clips and for separate
     development clips.
   - **Phase 2:** each of the 180 windows' evidence is a free-text string, and
     both the evidence and the fallbacks are unreviewed Gemma drafts
     (`REAL_VIDEO_QA.md`).

   Generation can be built and tested with fakes before the materials exist. G1
   and G2 cannot be measured until they do.
5. **The working tree is not the baseline.** Sixteen regen files are modified
   and nine are untracked. The changes add an optional per-video "experience"
   questionnaire policy, caption provenance on viewings and responses, and
   job-execution timing. This plan targets `3842302`. The parity target is
   regen's spreadsheet path under its default policy,
   `interaction-only-sheet-v1`.

## 1. Goal and scope

A new top-level package `src/diff` reproduces the regen spreadsheet protocol's
participant response order exactly and replaces only how caption text is
produced:

- **Offline generation:** Trida-7B, and optionally Tri-7B, writes caption text
  under fixed, instrumented settings.
- **Freeze:** outputs are checked, reviewed blind and frozen into a hashed
  stimulus bundle.
- **Serving:** the participant server reads the bundle and never calls a model.

In scope: generator (M0–M1), blind review and freeze (M2), participant and
review server (M3), detachment from `dpo` (Stage 4).

Out of scope: regen's v3 A/B flow, protocol-3 v4 and the standalone `study_api`
v2 entry. Also out of scope: any edit to `src/dpo/`, `data/` or `deploy/`, live
services or participant records, and deleting the Gemma/DPO dependencies (Stage
4 only proposes it).

## 2. What was verified

### 2.1 Code facts in the brief, re-checked at `3842302`

Every line reference in the brief matches the baseline. The working tree shifts
`sheet_protocol.py` by up to +66 lines; diff edits none of these files.

| Fact | Where | Status |
|---|---|---|
| Transition never calls a model; returns the first track (`maintain`) or an authored track keyed `<object_id>\|<A1s>` (`opposite`) | `sheet_protocol.py` 611–665 | confirmed |
| `writer` argument unused | `sheet_protocol.py` 60 | confirmed |
| 2×2 condition and strategy fixed per clip; only clip order rotates (`segment_order(document, sequence)`, 195) | 132–141 | confirmed |
| Whole instrument snapshot (91,787 bytes measured) and full config copied into every session; each response stores its item definitions | 196–197, 524; `study_store.py` 131–177 | confirmed |
| Instrument/document hash change blocks running sessions | 244–247 | confirmed |
| Real `clip_id` in every participant payload | 285 | confirmed |
| No path to P12/P13 without phase 2 (`handoff` requires a continuation) | 547–549 | confirmed |
| Export opens with the participant cookie from loopback | 694–698 | confirmed |
| Registration sequence = `SELECT count(*) FROM sessions` | `study_store.py` 97 | confirmed |
| Session created at page load | `sheet_questionnaire.js` 574 | confirmed |
| Viewing manifest re-read at `start-viewing`, shown in manifest order | `study_api.py` 366, 473 | confirmed |
| One repair retry on validation failure | `study_worker.py` 91–108 | confirmed |
| `context < 0.34` switches to a JSON-template path | `caption_controls.py` 66–87, 98 | confirmed |
| Two-decimal axes give a distinct prompt per 1 % step; 335 tells the model to listen to audio | `study_schema.py` 323–359 | confirmed |
| Any Hangul character passes the language check | `captions.py` 121–130 | confirmed |
| Page entry and client submit times never sent (`entered_at` stays in `sessionStorage`; server reads an absent `submitted_at`) | `sheet_questionnaire.js` 73; `sheet_protocol.py` 528 | confirmed |

### 2.2 Further regen findings that shape diff

- **Drafts:** the sheet client never sends server drafts. Drafts live in
  `sessionStorage`, scoped by session, instrument, practice flag, clip index
  and page. The server `draft` action exists but this client does not call it;
  a server draft is used only by a fresh tab.
- **Order and options:** P9 order is
  `page_item_order("P9", session_id)` = sort by
  `sha256([PROVENANCE, session_key, page, item_id])`. It is the same for every
  clip in a session. R2 correctness flags live in the clip config and are
  stripped from API options. Validation messages from `validate_answers` are
  English, and the client shows them to participants.
- **Phase-2 wording to drop:** `study-ko.json` tells participants that captions
  are "shaped by your responses" and that "settings carry into the next
  video". Neither is true in diff: phase-1 answers are not used, and every video
  starts at the centre cell.
- **Status codes differ by phase:** a missing `x-study-request` header is 401 in
  the questionnaire app and 403 in the viewing app's middleware.
- **Unused legacy state:** `StudyStore._initial_state` embeds the legacy
  instrument snapshot in every session; the sheet path pops it.
- **Workbook constants:** workbook hash
  `b23eb462bfc6e3fd4a14c3616a0b7bd12e703fdda8fba418258b981f5a7fb2ce`;
  instrument snapshot hash
  `6c97d64a8796fa09e4fa3d82d4d2680d220cb15faeaa0774bc60a08cd176ad8b`.
  P4 and P9 have 11 items each, with P9 IDs suffixed `_2`. Two clips give 34
  answers per clip over nine submit pages; regen's own end-to-end test asserts
  18 responses and 68 answers.
- **Live materials:** clips A = `bangkok_034` and B = `singapore_286`, each
  10,032 ms with one cue and five frames of 7–8 masks. Three long videos run
  299,679 / 300,259 / 300,142 ms, with 60 cues each.
- **Shared GPU:** GPU 0 holds a 15.5 GB python process; GPU 1 is free. Nothing
  was run.

### 2.3 Model facts (Hub API and the pinned files, read 2026-10-09)

> **Superseded by the Model change section and Appendix B.** The Trida/Tri facts
> below record what was verified before the switch; no code uses them.

**Trida-7B @ `c37755e81222de0aff69ee481f5299a99928b105`** (Hub last-modified
2026-01-29, not gated, Apache-2.0):

- **Architecture:** `TridaForDLM`, custom code (`auto_map`), bf16 shards
  totalling 15.05 GB.
- **Config versus card:** `config.json` was saved with
  `transformers_version` 4.57.3. It sets `max_position_embeddings` 4096,
  `vocab_size` 128256 and `bd_size` 4 (`bd_size` is used only by training
  paths). The model card states a context length of 8,192.
- **Generation defaults are never read:** `generation_config.json` holds
  `do_sample: true`, `temperature: 0.6` and `top_p: 0.9`. The custom `generate`
  overrides `GenerationMixin.generate` and never reads it.
- **`generate` signature:** `generate(input_ids, max_new_tokens, mask_id=128012,
  threshold=1, small_block_size=8, block_size=32, stop_token=128001,
  stopping_criteria=None, top_p=0.95, temperature=0, use_block_cache=False,
  pad_token_id=128004, return_dict_in_generate=False, **kwargs)`.
  `**kwargs` swallows anything unknown. The model card's own quickstart passes
  `tokenizer=`, which is silently ignored. `stopping_criteria` and
  `return_dict_in_generate` are accepted and unused.
- **Block arithmetic:** `num_blocks = max_new_tokens // block_size`. A
  `max_new_tokens` below 32 decodes no block; at most the post-prefill argmax
  token below is appended. After every completed block one token is appended by
  plain `argmax`; it is not subject to `threshold`. When the prompt length is a
  multiple of 32 greater than 32, one argmax token is appended after prefill.
- **Unmasking:** `threshold` unmasks every masked position whose top-1
  probability exceeds it, plus the single most confident one. At
  `threshold=1` that is exactly one position per step.
- **`temperature=0`:** `sample_with_top_p` returns `softmax` + `argmax` before
  any nucleus filtering, so `top_p` is not applied.
- **Hard-coded IDs:** `sample_with_top_p` always suppresses IDs 128012 and
  128004, whatever `mask_id` or `pad_token_id` was passed to `generate`. The
  per-step `forward` calls inside `generate` do not pass `block_size`, so they
  use the default 32. Any `block_size` other than 32, or token IDs other than
  the defaults, are internally inconsistent in this revision.
- **Output:** truncated after the first stop token; positions after it are set
  to the pad ID.
- **Runtime dependencies:** `fused_flex_attention` is
  `torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")`, so the
  first call compiles. The module also imports `einops`, which is not a repo
  dependency.
- **Special tokens:** `<|endoftext|>` 0 (bos), `<|im_start|>` 128000,
  `<|im_end|>` 128001 (stop), `<|pad|>` 128004, `<|mask|>` 128012.
- **Chat template:** without a system message, the template inserts "You are
  Trillion, created by TrillionLabs. You are a helpful assistant."

**Tri-7B pin (proposed): `94e2cc881102428ab9be3df2366a09ac928c2c82`** (current
`main`; Hub last-modified 2025-07-25, not gated, Apache-2.0):

- **Architecture:** `LlamaForCausalLM`, no remote code. Shards are fp32
  (30.1 GB) and load as bf16.
- **Config:** saved with transformers 4.53.2; `max_position_embeddings` 32768.
- **Tokens:** same special-token IDs as Trida, without `<|mask|>`.
- **Chat template:** differs from Trida's in whitespace and `<|im_end|>`
  placement, so Tri prompt token IDs are hashed separately.

SHA-256 of the pinned text files (downloaded at the exact revision; git blob
IDs matched the Hub's):

| File | Trida-7B @ c37755e8 | Tri-7B @ 94e2cc88 |
|---|---|---|
| `modeling_trida.py` | `100558ff8e0e7d37e24bb8a3cdfcea508ed9db5c81eeb64a026ba76f4b8ec949` | — |
| `configuration_trida.py` | `105a4722bc7fd3ed5f3b67e18cda91aae404135835f9e31037435d8c2f78ffe1` | — |
| `tokenizer.json` | `87522bb349f1b34dc65b11fbc841cf6d6499da80c375f142ea4c4a7923011321` | `de39a11436696c5cb69359e688e3acf8ed4af3711ec78e5328ea17e4c77eea85` |
| `tokenizer_config.json` | `e9477f4ef3120bc2b001c7b482207690c84cfc4a79a5ad557b114a7549769190` | `b2feed86404a118d02084149e794ddabf5c964df16ad5b615fc262efe582f031` |
| `special_tokens_map.json` | `ea3dfcfe26148c30aa296b54083a66531a28c58a0f880625a59439fe1d67baa7` | `20909c4754098a276d107d8080a930ac0c9a713d3bb296b92444f473eedad3fa` |
| `chat_template.jinja` | `8fe304e54dbef96520a19f0640ddf421c3a2188cf85352b09789c28f4c788238` | `c704db189c305fbc925629b77ce6fd437e42d9ccc65fcb7fc98268872461abe6` |
| `config.json` | `ed7d5a45a81e50883e3a67651b3b8bd999715221238770011837744a893566e5` | `2ec149fc6aeedf90a870562bfa5bd11073b0dfd5d2a0a2f07696473db310537f` |
| `generation_config.json` | `12603b65aab9b66affeafb91cbc5c0b3afe4e0c40c71492bce9aa564024efb86` | `4ed67772c746f5019b48337d838aaace2db308e23939d9eb657d3458d2a34b97` |

Weight shard hashes, the LFS SHA-256 values reported by the Hub, are in Appendix A.

## 3. Architecture

### 3.1 Modules

One flat package. Every basename was checked against `git ls-files` (no
collision) and against `sys.stdlib_module_names` (no shadowing). `diff` itself
is not a stdlib module; `difflib` is.

| Module | Stage | Env | Responsibility |
|---|---|---|---|
| `__init__.py` | 1 | both | docstring only; imports nothing |
| `__main__.py` | 1 | both | `python -m diff <command>`; lazy imports so the generator env never imports FastAPI and the server env never imports torch |
| `study_protocol.py` | 1 | both | protocol data and its hash: page order, gates, budgets, caps, open-decision defaults (§12), CV/MC constants |
| `caption_sources.py` | 1 | both | `CaptionSource` schema, the closed quality vocabulary (copied from `caption_controls.QUALITIES` at `3842302`, hash-pinned), provenance gate, read-only importers for the regen document and viewing manifest |
| `rewrite_prompts.py` | 1 | both | versioned Korean templates, deterministic evidence rendering, template and text hashes |
| `model_profile.py` | 1 | gen | Dream and Qwen pins, file SHA-256, token IDs, context limit, verification before `trust_remote_code`, post-load hash of imported remote code |
| `job_plan.py` | 1 | both | content-addressed jobs (cue × condition, cue × cell), caps, plan freeze (token IDs added in the generator env) |
| `generation_worker.py` | 1 | gen | *added*: subprocess isolation, the job's deadline (60 s since M0), transient-failure classification, the one identical retry |
| `dream_runner.py` | 1 | gen | native `diffusion_generate`, batch 1, every keyword required; refusals before the call; caps (forward calls = steps, new positions = canvas); each call seeded from its job ID (v4) and a reproducibility check at warm-up; realized per-step counts from history |
| `qwen_runner.py` | 1 | gen | greedy baseline, `repetition_penalty=1.0`, sampling knobs unset, resolved config recorded |
| `model_runtime.py` | 1 | gen | *added*: loaded-model record, refusal error, chat rendering with the system message, token-ID check, RNG snapshots |
| `analysis_decisions.py` | 1 | both | *added*: `AnalysisDecision` records and the gate check (AC-20) |
| `feasibility_gate.py` | 1 | both | *added*: G0 inputs, report, blind packet, ratings, decision |
| `decode_trace.py` | after G0 | gen | Dream decode trace (logits hook recomputing `sample_tokens` + `topk`; tokens hook); not built yet |
| `output_checks.py` | 1 | both | fail/flag/measure rules; no repair, truncation or shortening |
| `review_app.py` | 2 | server | blind double review and adjudication (researcher port) |
| `freeze_gates.py` | 2 | server | *added*: G1/G2 computation and recorded hold decisions |
| `stimulus_bundle.py` | 2 | server | bundle builder, verifier, `bundle_sha256`, version registry |
| `assignment_schedule.py` | 3 | server | slot table generation, balance report, release policy |
| `session_store.py` | 3 | server | transactional store: sessions, receipts, events, slots, responses, viewings, exposures, control events, page entries |
| `phase1_flow.py` | 3 | server | P0–P11, practice, transition, handoff |
| `phase2_viewing.py` | 3 | server | ready → watch / break → P12 → P13; lattice delivery; controls; exposures |
| `participant_server.py` | 3 | server | *added*: app factory, opaque media tokens, security headers, static allowlist, blindness boundary |
| `export_bundle.py` | 3 | server | researcher-authenticated export (CLI and researcher port) |

Static assets, also collision-checked: `diff_participant.html`,
`phase1_pages.js`, `phase2_player.js`, `participant_theme.css`,
`review_console.html`, `review_console.js`. The participant assets are Korean
only; the review console is researcher-facing.

`pyproject.toml` (Stage 1, first commit): add `"src/diff"` to
`[tool.hatch.build.targets.wheel] packages` and to `[tool.mypy] files`. Add no
`[project.scripts]` entry; the system `diff` stays reachable inside the venv.
`testpaths` already covers `tests/diff/`.

### 3.2 Environments

| | Server env (participant and review) | Generator env |
|---|---|---|
| Lives in | the shared `pyproject.toml` / `uv.lock` | `envs/diff-generator/pyproject.toml` + its own `uv.lock`, `package = false` |
| Needs | FastAPI, uvicorn, Pillow, system ffmpeg/ffprobe (4.2.7 present) | `torch==2.5.1` (cu124), `transformers==4.46.2`, `safetensors`, `huggingface_hub` < 1.0 (Dream README pins); Python 3.11 |
| Imports torch | never (asserted by test) | yes |
| Runs diff via | `uv run python -m diff …` (diff is in the editable wheel) | `PYTHONPATH=src uv run --project envs/diff-generator --frozen python -m diff …` from the repo root |
| Accepted when | `make check` passes (see §10 on the pre-existing `.omx` lint) | AC-03 and AC-07 pass on a GPU with ≥ 20 GB in this env |

The generator env cannot be a dependency group of the shared project:
`transformers==5.13.1` and `torch==2.10.0` are main dependencies, and a group
cannot downgrade them. diff is never installed there; it runs from the source
tree. Both runners (Dream and Qwen) use it. `make locks` checks its lock too.
Until Stage 4, torch stays a dependency of the shared project as the brief
requires.

### 3.3 Data flow and workspace

```
authored sources (human / reviewed drafts)
  → caption_sources validate
  → job_plan build (text prompts)
  → job_plan freeze  [generator env: token IDs, lengths, guards]
  → dream_runner / qwen_runner  [GPU]  → attempts (+ decode traces after G0)
  → output_checks  → review_app (blind, 2 reviewers + adjudication)
  → freeze_gates (G1 / G2 or a recorded hold)
  → stimulus_bundle build + verify  → bundle registry (by hash)
  → assignment_schedule  → slots for that bundle
  → participant_server (no model)  → session_store
  → export_bundle (researcher only)
```

Runs, attempts, traces, reviews, bundles and session databases live under a
workspace given on the command line, default `artifacts/diff/` (already
gitignored). Real media is referenced read-only from wherever it is staged,
including `data/live/regen-media/`, and is never copied or modified. Protocol
constants are versioned in `src/diff`.

### 3.4 Import boundary

- **Static check.** Until Stage 4, `src/diff` may import only
  `dpo.regen.{sheet_instrument,playback,points,study_media,study_store}`. An AST
  test enforces this.
- **Runtime check.** A subprocess test imports `diff.participant_server` and
  `diff.review_app` and asserts that `torch`, `transformers`, `dpo.models`,
  `dpo.candidates`, `dpo.contracts` and `dpo.regen.caption_controls` are absent
  from `sys.modules`.
- **Known exception.** `dpo.caption.writer` is listed explicitly as a known
  transitive load until Stage 4 (§0.2, Q2).
- **Planned uses.** `sheet_instrument`: `page_spec`, `page_item_order`,
  `validate_answers`, `snapshot`. `playback`: `record_playback`, `merged`.
  `points`: `MaskObject`, `parse_points`, `measure_points`, `matched_labels`.
  `study_media`: delivery copies and paced responses. `study_store`: `Conflict`
  only, because `record_playback` raises it. Diff's store is its own.

## 4. Generator (Stage 1, PRD M0–M1)

### 4.1 Model profile and verification

`model_profile.py` defines one frozen profile per model: repo, revision, every
file's SHA-256 (§2.3 and Appendix A), dtype `bfloat16`, expected
transformers / torch / einops versions, token IDs, the kwargs allowlist, and
recorded facts. The facts include `max_position_embeddings` 4096 against the
card's 8,192, the context guard, `top_p` not applied, and the prompt-length
guard.

Loading sequence:

1. **Snapshot.** `snapshot_download(revision=<pin>)` into the workspace.
2. **Verify before code runs.** SHA-256 every remote-code, tokenizer, config,
   template and weight file against the profile. Any mismatch stops the load.
3. **Load from the verified directory, offline.** Call
   `from_pretrained(<verified local dir>, trust_remote_code=True)` with
   `HF_HUB_OFFLINE=1` and a fresh per-run `HF_MODULES_CACHE`, so a stale module
   cache cannot substitute code.
4. **Verify after load.** Hash `inspect.getfile(type(model))` and the config
   class's file, and require them to equal the pinned
   `modeling_trida.py` / `configuration_trida.py`.
5. **Record.** The attempt record carries the environment fingerprint: lock
   hash, Python, torch, CUDA, GPU name and driver.

### 4.2 Prompts (`rewrite_prompts.py`)

The two Korean templates in the brief are stored verbatim as development
drafts. Each template carries an id, a version and a template SHA-256. Every
job records three hashes:

- **Text hash:** the rendered text.
- **Token hash:** the token IDs after the model's own
  `apply_chat_template([{"role": "user", "content": text}],
  add_generation_prompt=True)`, computed in the generator env at plan freeze.
- **Chat-template hash:** the template file used for that tokenization.

No system message is passed, so the template's built-in Trillion system prompt
is part of the hashed input (Q9).

`{cue_evidence}` is a deterministic, versioned Korean rendering of the
structured evidence. Events appear in order, each with its source (category),
count, action, qualities in their Korean final forms, location or relation, and
certainty. The phase-1 instruction's kind, count, order and certainty
therefore each have a field to come from.

Phase 2 renders three more slots:

- **`{qualities}`:** the cue's evidence qualities, comma-joined.
- **`{k}`:** the texture step's value.
- **`{context_rule}`:** one of three draft sentences, to be approved (Q10):
  - broad: write sources only as broad kinds (human, animal, traffic and
    machine, music, nature), with no specific source names or places;
  - specific: name the sources in the evidence, but not their places or
    relations to other things;
  - relation: name the sources in the evidence with the places or relations
    it gives.

Two edge cases are left as open questions (Q10): `k = 0` renders "up to 0",
and evidence with no qualities renders an empty list.

### 4.3 Job plan and caps

- **Phase-1 job:** one cue × one second-track condition. `original` is not
  generated; it is the source track repeated exactly.
- **Phase-2 job:** one cue × one lattice cell, at the fixed phase-2 τ.
- **Identity:** `job_id = sha256(canonical spec)`, and the plan id is the hash
  of its sorted job IDs. Re-running a plan skips terminal jobs.
- **Development jobs:** they share the schema but are flagged
  `purpose: development`, and use development clips that are never participant
  clips.
- **Budgets (CV-08):** 128 tokens in phase 1 and 192 in phase 2 to start. At
  M0, the budget is re-set from measured source token counts with the Trida
  tokenizer and recorded in the protocol. Character limits live in the output
  checks, not in the budget.
- **Caps (§6.2):** `max_new_tokens = budget + 32` (160 or 224, both multiples of
  32) and a 30 s timeout per cue, finalised after the M0 smoke test.
  - **Recorded new positions:** the native loop's actual new positions depend on
    the prompt length mod 32 and on the argmax tokens appended after each
    block. Each attempt records the real count.
  - **Guards at freeze:** `prompt_tokens + max_new_tokens + 1 ≤ 4096` (the
    config value, not the card's), and `prompt_tokens ≠ 32`.
- **Counts with the current materials:**

  | Set | Formula | Jobs |
  |---|---|---|
  | Phase-1 Trida | 2 clips × 4 cues × 3 τ | 24 |
  | Phase-1 Tri | 2 × 4 | 8 |
  | Phase 2, 3×3 | 3 videos × 60 cues × 9 cells | 1,620 |
  | Phase 2, 5×3 | 3 × 60 × 15 | 2,700 |
  | G1 development, suggested 8 clips | 8 × 4 × 3, plus 32 Tri | 96 |

### 4.4 Runners, isolation, retries

> **Protocol v4 (2026-10-11) supersedes the values in this box:**
> `temperature=0.2` and `top_p=0.95` are required, not refused. Each call is
> seeded from its job ID, and the worker's warm-up checks that two seeded calls
> agree, instead of the RNG-unchanged cap. See "Protocol v4" near the top.
>
> **Revised for Dream (2026-10-09).** The Trida call described below is gone.
> `dream_runner.checked_call` requires every keyword of the follow-up's §3 call —
> `attention_mask`, `max_new_tokens` (the canvas), `steps` (canvas ÷ k),
> `temperature=0.0`, `top_p=None`, `top_k=None`, `alg="entropy"`,
> `alg_temp=0.0`, `eps=1e-3`, `mask_token_id=151666`, `output_history=True` and
> `return_dict_in_generate=True` — and allows only the two read-only hooks
> besides.
>
> **Refused before the call:** a missing keyword; any other keyword; `alg="origin"`;
> non-zero `temperature` or `alg_temp`; any `top_p` or `top_k`; `steps` larger
> than the canvas or not dividing it; prompt plus canvas over 2,048; a prompt
> containing 151666 or a BOS.
>
> **Every call** records the resolved generation config. It checks the caps:
> forward calls equal `steps` (a read-only forward hook), new positions equal
> the canvas, and the CPU and CUDA RNG states are unchanged. A violation is a
> non-transient technical failure (`count_mismatch`, `rng_consumed`,
> `leftover_mask`). Isolation, the deadline and the one retry are unchanged.
>
> **Qwen** (`qwen_runner`) is called with exactly `do_sample=False`,
> `repetition_penalty=1.0`, `max_new_tokens` = the Dream canvas and
> `eos_token_id=[151645, 151643]`. Both models get the system message "You are
> a helpful assistant."

**Trida call.** Exactly one call shape, with batch size 1 and no attention
mask:

```
model.generate(input_ids, max_new_tokens=…, threshold=τ, block_size=32,
               small_block_size=8, temperature=0, use_block_cache=False,
               mask_id=128012, stop_token=128001, pad_token_id=128004)
```

- **Not passed:** `top_p`; the profile records it as `not_applied`.
- **Allowlist check:** a pure function runs before every call. It compares the
  argument dict against the allowlist and the protocol values, and refuses any
  extra key (`steps`, `seed`, `do_sample`, `tokenizer`, `top_p`, `top_k`,
  `stopping_criteria`, `return_dict_in_generate`, `generation_config`, …) or
  any differing value.

**Tri call.**

```
model.generate(input_ids, attention_mask, max_new_tokens=…, do_sample=False,
               eos_token_id=128001, pad_token_id=128004,
               output_scores=True, return_dict_in_generate=True)
```

The last two arguments are instrumentation, and AC-03-style equivalence
covers them.

**Isolation.** `generation_worker.py` runs the model in a subprocess. The parent
enforces the job's deadline (`caps.timeout_s`, 60 s since M0) and kills the
worker on expiry.

**Warm-up.** A fixed non-study warm-up generation runs first and is logged but
never recorded as an attempt, so that `torch.compile` time is not charged to a
cue. Compile and recompile counts go into each attempt.

**Outcomes.**
- **Completed:** then checked.
- **Cap outcomes**, terminal and excluded: `timeout`, `no_stop_within_cap`.
- **Transient failures:** CUDA error or OOM, worker death, IPC loss. These get
  exactly one identical retry with the same spec, a fresh worker and
  `retry_of` set. A second transient failure is an unresolved technical
  failure.

No outcome triggers a quality-driven retry, a repair prompt, truncation or
shortening.

### 4.5 Decode trace and instrumentation equivalence

> **Revised for Dream (2026-10-09); follows a passing G0.** The Trida walk below
> was removed. Dream's trace (follow-up §4) works as follows.
> - **Logits hook:** recomputes `sample_tokens` with the same arguments, then
>   `torch.topk` over a full row of −inf outside the masked positions. It
>   records `selected`, top-1, top-2, entropy and `selected_but_masked`, and
>   returns the logits untouched.
> - **Tokens hook:** records only.
> - **Recorded per step:** `t` and `s`, the masked positions and, for each, its
>   rank and selection.
> - **Size limit:** `trace_level: selected` applies if full phase-2 traces would
>   exceed 5 GB compressed.
>
> AC-01/AC-03-hooks/AC-04/AC-05 run on CPU against a vendored
> `dream_generation_utils.py` whose SHA-256 the tests check. Before G0, only
> realized per-step counts are recorded, from `output_history`.

**Observers.** `decode_trace.py` replaces `model.forward` and
`model.sample_with_top_p` on the instance with pass-through observers. Native
`generate` calls `self.forward` / `self.sample_with_top_p`, so instance
attributes take effect, and the observers change no argument and no return
value. They record:

- each forward's input IDs, which are at most one block, and whether it updates
  the cache;
- for each sampling call, the top-1 token and its probability per position.
  The values are gathered on device and only small tensors are copied.

**Reconstruction.** From successive forward inputs, the trace derives which
positions were unmasked at each step and to what. It also records the argmax
token appended at each block commit, the prefill length and whether prefill
ran, and the stop position. It recomputes the expected unmask set
(`p > threshold` plus forced argmax) and cross-checks it against what was
observed.

**Replay check.** Replaying the events must reproduce the native output exactly.
A mismatch is an instrumentation defect: it halts the run and is not eligible
for retry.

**AC-03 (generator env acceptance).** For every development prompt, native
`generate` with observers must equal native `generate` without them, token for
token, on GPU. If GPU nondeterminism appears, deterministic mode
(`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`) is tried, and
whether it is required is recorded.

**AC-05 boundary suite, with B = `block_size` = 32 (A3).** Synthetic token
prompts of length 31, 32, 33, 63, 64 and 65 go through native `generate`, with
and without observers. For each: trace replay equals output, and the observed
branches are recorded (no prefill; prefill only; prefill plus appended
argmax). At 32 the trace records the no-prefill branch. The study guard refuses
such prompts.

**Tests without a GPU.** A fake model reproduces the native loop's control flow
on a tiny deterministic logit table. It is used for observer transparency,
replay, kwargs refusal and branch coverage. A CPU smoke test of the real remote
code with a tiny random config is attempted in the generator env at M0; flex
attention may not compile on CPU, and the result is recorded either way.

### 4.6 Output checks (`output_checks.py`)

**Text extraction.** Take the new IDs up to, not including, the first stop
token. Decode them without skipping special tokens to get `raw_text`. The only
transformation is stripping leading and trailing whitespace to get `text`;
both strings are stored.

**Fail rules** exclude the output and record the reason. These are technical
and format rules:

- no stop token within the cap;
- any special token (mask, pad, reserved, chat markers) inside the text;
- empty text;
- length: phase 1 > 96 characters or > 2 lines, phase 2 > 160 characters;
- more than one sentence;
- script: any letter that is not Hangul, unless it appears in that cue's
  source or evidence. This is stricter than regen's "any Hangul passes".
- CV-10 sentence-ending class differs from the source's, using a closed,
  versioned set: declarative, nominalised, noun-phrase ending, polite.

**Flag rules** send the output to double review. These are content rules:

- a sound-source term absent from the evidence;
- a quality term outside the cell's allowed list, or more than `k` of them;
- a source name at context step 0;
- a relation term below step 2;
- phase-1 output identical to the source;
- phase-2 text identical to a lattice neighbour.

**Measures** (§5.3 additions):

- character and token length;
- quality-term count against the closed vocabulary's Korean forms;
- specific-source count and relation count against the evidence's own surface
  forms;
- whether the text equals the source.

Fidelity and length are recorded as outcomes of the condition, never as
covariates.

## 5. Review and freeze (Stage 2, PRD M2)

### 5.1 Blind review (`review_app.py`)

**Access.** A researcher-only app on loopback (default port 8783) with one
token per reviewer.

**What reviewers see.** Each item shows the source text (phase 1), the
rendered evidence and one candidate text, under an opaque item ID and in
randomised order. No τ, model, condition, cell or run identifier is shown.
Identical texts for the same cue are reviewed once and mapped back.

**Phase 1.** Every output, including Tri's if Tri is run, is rated by two
independent reviewers on:

- naturalness, 1–5 (scale A5);
- sound-event fidelity: preserved, or changed kind / count / order /
  certainty, or a sound added or dropped;
- R2 mentions: which R2 options this track mentions. These become the
  shown-track key; the source track gives the source key.

**Phase 2.** Automated checks run on everything. Double review covers a
stratified sample (default 10 % per video × cell, at least 2 per stratum) plus
every flagged output, for naturalness and "adds a sound absent from the
evidence".

**Adjudication.** A reviewer cannot see the other reviewer's ratings. An
adjudicator, still blind to condition, sees both and decides when:

- naturalness differs by ≥ 2;
- naturalness crosses a gate threshold (3 or 4);
- the reviewers disagree on any categorical judgement.

Otherwise the final naturalness is the lower of the two ratings (A6).
Agreement is reported: weighted κ for naturalness, κ for the categorical
judgements.

### 5.2 Gates (`freeze_gates.py`)

These are provisional operating thresholds, not validated cut-offs. Track
naturalness is the minimum over its cues (A6).

| Gate | Rule |
|---|---|
| G1 technical | every planned development job is terminal, with zero unresolved technical failures |
| G1 readability | in ≥ 3/4 of development clips, every second-track condition has naturalness ≥ 3 |
| G1 difference | in ≥ half of those complete clips, ≥ 2 τ tracks differ as strings, both with naturalness ≥ 4 and sound events preserved; otherwise τ leaves the participant conditions (option C) |
| G2 monotonicity | ≥ 80 % of cues are non-decreasing in quality-term count along acoustic detail, and in specific-source + relation count along scene detail |
| G2 distinctness | identical-neighbour rate reported per axis; an axis above 20 % has its steps reduced; a cell that adds a sound absent from the evidence is substituted, and the count is reported |

A gate that is not met produces a recorded hold decision, with who decided,
when, and what was decided. Nothing is silently relaxed. Phase-2 τ is chosen
after G1 and recorded as protocol data.

### 5.3 Phase-2 lattice

> **Superseded for participants by "Live phase 2" (2026-10-11).** Phase-2
> captions are written live, so no lattice is pre-generated or reviewed for
> sessions. The lattice generator stays for development use and keeps AC-20.

- **Steps.** Texture `k ∈ {0, 2, 4}` (0–4 after G2). Context ∈ {broad,
  specific, relation}.
- **Generation path.** One template for every cell. Regen's JSON-template branch
  is not reproduced.
- **Missing or substituted cells.** By default the nearest-cell substitute is
  used: nearest by L1 distance in step space, ties toward the same context step,
  then toward lower texture. The bundle records both the requested and the
  served cell. The alternative is excluding the video.
- **5×3.** Moving to 5×3 is a new plan and a new bundle, made only after G2
  passes.

### 5.4 Bundle (`stimulus_bundle.py`)

**Build.** Inputs: frozen plans, attempts, check results, reviews and
adjudications, sources (which must have reviewed provenance), R2 options and
keys, the lattice, τ, and gate decisions. Output: canonical JSON
`diff.stimulus-bundle/v1` with `bundle_sha256` over everything else (§8.1).

**Track completeness.** A condition track for a clip exists only if all N cues
passed the checks and the review. Otherwise the condition is listed as an
exclusion with its reasons, and slots cannot assign it.

**Verifier checks.**
- schema and hash;
- media hashes;
- each clip has a first track and a full track for every assigned condition;
- R2 options are identical across a clip's conditions;
- every shown track has both keys;
- every cue × cell has a text or a recorded substitute (AC-16);
- no unreviewed provenance;
- option values are neutral slugs (no `correct`, `distractor` or condition
  substrings).

**Registry.** The registry holds many bundles by hash, with one active. Changing
the active bundle never changes a bound session's bundle.

## 6. Participant server (Stage 3, PRD M3)

### 6.1 Session store (`session_store.py`)

Kept from regen:

- `BEGIN IMMEDIATE` transactions;
- the revision check, where every accepted mutation increments `revision`;
- idempotent receipts keyed `(session, key)`: the same request returns the
  latest state, a different request with the same key is 409;
- an append-only events table.

Dropped: the `jobs` and `captions` tables.

Added tables: `slots`, `responses`, `viewings`, `exposures`, `control_events`,
`page_entries`, `instruments` (snapshot by hash, stored once) and `bundles`
(registry).

**Session state stays small.** It holds ids, the protocol hash, the instrument
hash, `bundle_sha256`, `slot_id`, stage and page, clip index, video position,
consent reference, the current view's playback checkpoint, and at most one
draft, capped at 16 KiB. Responses and viewings move to their own rows written
in the same transaction. That keeps a 1 s playback tick from rewriting tens of
kilobytes (AC-18).

### 6.2 Assignment (`assignment_schedule.py`)

**Slot table.** A pre-generated table per bundle; each slot holds the clip
order, the second-track condition per clip, and the long-video order. It
replaces regen's `count(*)` sequence.

**Design.**
- **Latin square** when clips = conditions, using a Williams sequence for clip
  order.
- **BIB with orientation** when clips < conditions. Each participant gets
  distinct conditions on their clips. With 2 clips and 4 conditions this is 12
  ordered condition pairs × 2 clip orders = a 24-slot cycle. In it each
  (clip, condition) appears 6 times, each condition 6 times per viewing
  position, and each unordered pair 4 times (twice in each orientation).
- **Video order** cycles through all 3! = 6 orders, a Williams design for
  n = 3, 4 times per cycle.
- **Five conditions** (with Tri) give a 40-slot phase-1 cycle. The cycle is
  lcm(40, 6) = 120, or video-order balance is reported as partial (Q8).
- **Clips > conditions** is refused.

**Binding.** A slot binds in the same transaction as consent: the lowest free
slot of the active bundle, released slots first. The cookie session created at
page load holds no slot.

**Abandoned slots (default).** A slot is reusable only after a researcher
releases it with a reason. It is reissued only if its session had no formal
response; otherwise it stays consumed and the imbalance is reported. A
released session becomes `withdrawn` and cannot continue. The alternatives are
in §12.

**Status report.** `slots status` reports balance for every recruited prefix,
so a study that stops mid-cycle knows its imbalance.

### 6.3 Phase-1 flow (`phase1_flow.py`, `phase1_pages.js`)

The participant sees the same pages, items, orders, validation and recovery
behaviour as regen; §7 is the parity map. Changes:

- **Payload.** No `clip_id`. Media and frame URLs are opaque per-session tokens
  (`/m/<token>`). The response carries `consent_sha256` so that consent binds
  to the text actually shown.
- **Transition page.** Its id is `transition` instead of `generate`, with the
  same title (preparing the second caption) and the same button (prepare the
  next video). It looks up
  the slot's condition and binds the bundle's track for P6, with no model
  call.
- **Handoff.** It moves to phase 2 inside the same app (no continuation mount,
  no second cookie) and compiles no profile, because phase-1 answers are not
  used.
- **Responses.** Each stores page, view id, viewing (first/second),
  `stimulus_id`, answers, presented order, instrument hash, server and client
  timing, and the point matches for P2/P7. Item definitions are not copied into
  each response; the instrument is stored once by hash.
- **Errors.** Participants get Korean messages; the English validator detail
  goes to the event log.

### 6.4 Phase-2 viewing (`phase2_viewing.py`, `phase2_player.js`)

> **Superseded by "Live phase 2" (2026-10-11).** The built app is
> `live_server.py` with `live_static/`: the browser polls captions written
> live instead of fetching a lattice. The controls, telemetry and playback
> gates below are the starting point it follows.

**Video order** comes from the slot, not the manifest.

**Lattice delivery.** On entering a video, the browser fetches that video's
lattice once: cues, the cells with their step coordinates, and per cue × cell
an opaque stimulus token and its text. That is about 540 texts for 3×3.

**Controls.**
- Two sliders snap to steps. The pad, presets (the four corners) and reset (the
  centre) remain.
- The overlay shows steps, not percentages.
- Every video starts at the centre cell.
- A control change sets the requested cell, and its text appears at the next
  cue boundary.

**Telemetry.** Control events (origin `slider:texture`, `slider:context`, `pad`,
`preset:<corner>` or `reset`; from and to cell) and exposures (stimulus and
cell, cue, displayed interval) are uploaded through append-only idempotent
endpoints. Like regen, a durable outbox in `localStorage` and Web-Lock tab
ownership keep them recoverable.

**Unchanged from regen.**
- **Playback gates.** Long videos need coverage ≥ duration − 1,500 ms and
  position ≥ duration − 500 ms, through regen's `record_playback`.
- **Pages.** Break between videos, then P12 (`interview-complete`
  {researcher_confirmed}) and P13 (`debrief-complete` {researcher_confirmed,
  participant_acknowledged}).

**Wording changes.** The "ready" and "break" pages drop "shaped by your
responses" and "settings carry into the next video" (Q11).

### 6.5 Blindness boundary

**Inventory.** Everything the browser can fetch is enumerable: one HTML shell,
three static files, the participant API, media under `/m/<token>`, and the
cookie.

**What it never contains.**
- **Opaque tokens:** `HMAC(server_secret, session_id ∥ internal_id)`. Item and
  option values are neutral slugs, and option lists carry only
  value/label/exclusive.
- **Neutral strings:** the cookie name and every error message are neutral.
- **Nothing that reveals a condition:** bundle hashes, condition names, cell
  internal IDs, model names, τ and real clip IDs never leave the server. Even
  the `original` condition is not detectable from P1/P6 tokens, because tokens
  are view-scoped.

**Exports.** These, and the researcher port, are a separate app that never
reads the participant cookie.

### 6.6 Timing

- **Per page load.** The client creates a random `clock_session_id` per page
  load and timestamps with `performance.now()`.
- **Page entry.** Sent as an append-only beacon: page, view, clock id,
  monotonic time and wall time. It does not bump `revision`, so it cannot
  disturb response order.
- **Submit.** Every submit carries `{entry_clock_session_id, entered_mono_ms,
  clock_session_id, submitted_mono_ms, submitted_wall}`.
- **Playback.** Ticks carry the clock id and a monotonic stamp.
- **Validity.** An interval whose two endpoints have different clock ids is
  stored with `timing_valid: false`, and no duration is computed for it (FR-19).

### 6.7 Export (`export_bundle.py`)

- **Access.** `python -m diff export` reads the store directly, so it requires
  filesystem access. A kiosk export (FR-22) is also served on the researcher
  port with a researcher token. The participant app has no export route, and
  presenting the participant cookie anywhere on the researcher app is 401.
- **Contents.** Exports resolve opaque IDs to bundle, condition, cell and real
  clip IDs (FR-18). They carry long-format P4/P9 tables for the §9 primary
  contrast. Fitting the models is not part of these modules (Q14).

## 7. Response-order parity map (AC-19)

Regen columns describe `3842302`: `sheet_protocol.py` and
`sheet_questionnaire.js` before the handoff, `study_api.py` and `study.js`
after it. "Same" means the same page id, kind, title, instruction, item IDs and
order, options, validation and transition.

| # | Regen page → action `data` | Regen stored | diff page → action `data` | diff stored | Parity rule |
|---|---|---|---|---|---|
| 0 | page load → `POST session {code?}`; clip order from `count(*)` | session + `sheet-enrollment` event | page load → `POST session {code?}`; no slot, no bundle | session row (pre-consent) | diff-only: no order is fixed at load |
| 1 | P0 intro: consent text, `setup_missing` → `intro-complete {page:"P0", consented:true}` | `sheet_consent {accepted, at, text}`; page → practice | P0 same, + `consent_sha256` → `intro-complete {page, consented, consent_sha256, timing}` | consent row; slot and bundle bound in the same transaction; page → practice | same page and gate; 409 while setup is missing |
| 2 | practice: `practice_rating` (0–6), `practice_sound`, `practice_visual` → `practice-complete {page, answers}` | `sheet_practice[] {answers, analysis_excluded:true, at}` | same | practice row, `analysis_excluded` | same; practice frames ≠ formal frames |
| 3 | P1 view (first track, no seek UI) → `playback {page, position_ms, playing, hidden, sequence}` …, `view-ended {page}`; gate cov ≥ d − min(300, 5 %), pos ≥ d − 100 | checkpoint; `sheet_viewings[] {view_id, clip_index, clip_id, condition:"first", captions, coverage, started_at, ended_at, …}` | P1 same; ticks + `clock_session_id`, `mono_ms` | checkpoint in state; viewing row {view_id, clip, viewing:"first", stimulus_id, coverage, …} | same gate (shared `record_playback`), same 409 text class |
| 4–7 | P2 V1 → P3 A1s → P4 (11) → P5 FA1–FA3, each `submit {page, answers}` | `sheet_responses[] {page, clip_index, clip_id, view_id, condition, caption_strategy, stimulus_assignment, answers, items, presented_order, instrument_hash, submitted_at, client_submitted_at:null, …}` (+ matches, labels, visual_metrics for P2) | same pages → `submit {page, answers, timing}` | response row {page, view_id, viewing, stimulus_id, answers, presented_order, instrument_hash, submitted_at, client timing, timing_valid} (+ matches, labels, visual_metrics) | same items, order, validation; diff stores items by instrument hash |
| 8 | `generate` (kind `generating`, titled preparing the second caption, button to prepare the next video) → `generate {page:"generate"}` | `sheet_regenerations[seg] {track, writer, strategy, selection_key, condition, caption_hash}` | `transition` (same title and button) → `transition {page:"transition"}` | second-track binding {view P6 → stimulus_id from slot} | page-id map `generate ↔ transition`; no model in either |
| 9 | P6 view (second track) | viewing, `condition:"second"` | P6 same | viewing row, `viewing:"second"` | same gate; caption text equal by construction (harness) |
| 10–14 | P7 V2 → P8 A2s (A1s options) → P9 (11 `_2`, `page_item_order(P9, session_key)`) → P10 S1–S3 → P11 R1, R2 (+ an exclusive "don't remember") | as 4–7 | same | as 4–7 | same; P9 order is the same function of each system's session key |
| 15 | after P11: next clip's P1, or `handoff` (stage 1 done, button to start stage 2) → `handoff {page:"handoff"}` → `next_url` | stage `ready`; profile, observations, `calibration_source` | same → `handoff {page}` | stage `phase2/ready`; no profile | same sequence; diff stays in one app |
| 16 | ready: a line saying the earlier answers are saved + interaction-only text, button to start watching → `start-viewing {}` (manifest re-read) | `viewing` (manifest order), `viewing_hash` | same → `start-viewing {}` | video order from slot | same page; order source differs by design |
| 17 | watch: `settings {video_id, position_hint_ms, origin, texture, context}`, `playback {video_id, position_ms, sequence, playing, hidden, seek}`, `exposures {video_id, entries}`, then the watching-done and on-to-the-interview buttons → `video-ended {video_id}`; gate cov ≥ d − 1,500, pos ≥ d − 500 | axes, `settings_revision`, jobs, exposures, `completions[]` | watch: lattice once; control events and exposures (append-only); `playback`; `video-ended {video}` | control_events, exposures, completions | same gate and transition; caption mechanics differ by design |
| 18 | break (a line asking the participant to rest) → `continue {}` | stage `watch` | same (text revised) | — | same position |
| 19 | P12 interview prompts D1/D2 → `interview-complete {researcher_confirmed:true}` | `poststudy.interview_completion` | same | same | same |
| 20 | P13 debrief text → `debrief-complete {researcher_confirmed, participant_acknowledged}` | `debrief_completion`; stage `done` | same | same; slot `completed` | same |
| 21 | done: thanks + receipt (`session_id`) | — | same (opaque id) | — | same |

**Errors that must match.**

| Case | Status |
|---|---|
| Missing item (validator) | 400 |
| Answer for another page | 409 |
| Stale revision | 409 |
| Replayed key, same input | 200, latest state |
| Replayed key, different input | 409 |
| `view-ended` before the gate | 409 |
| Consent before setup | 409 |
| No cookie | 401 |
| Missing `x-study-request` | 401 in phase 1, 403 in phase 2 (kept per phase) |

**Stored-field comparison.** Compared fields: page, `clip_index`, view-id
pattern, first/second, answers, presented order, instrument hash,
`analysis_excluded`, matches, labels and `visual_metrics`, and the reconstructed
item definitions. Excluded as provenance: `caption_strategy`,
`stimulus_assignment`, `caption_provenance`, writer, condition and cell IDs,
`stimulus_id`, real `clip_id` versus diff's internal clip key (mapped), and
diff-only timing.

**Harness** (`tests/diff/test_response_order_parity.py`, the one test allowed
to import `dpo.regen.app`).

1. **Build both systems.** Construct regen with `build_app(…,
   viewing=ViewingConfig(…, model={}), questionnaire_config=…)` over a 4-cue
   fixture document. In regen, each clip's second track is filled from the
   bundle's text for the scripted participant's condition: `maintain` for
   `original`; otherwise `opposite`, with every `<object>|<sound>` key mapped
   to that one track. Diff is built over the same media and the same option
   values.
2. **Run one script on both.** Valid submits; a missing item on P4; P5 answers
   sent while on P4; a reload mid-P4 that restores the draft; a replayed key
   with the same and with a different body; then both viewing phases through
   P13.
3. **Compare.** Page sequence, item IDs, presented order (structurally for P9,
   and exactly given the same key), status codes, and the stored fields above.

## 8. Schemas

The examples show shape. Every record carries its `schema` id. Hashes are
SHA-256 over canonical JSON (sorted keys, no NaN, UTF-8, compact separators).

### 8.1 Stimulus bundle — `diff.stimulus-bundle/v1` (server-side only)

```json
{
  "schema": "diff.stimulus-bundle/v1",
  "bundle_sha256": "<sha256 of this object without this field>",
  "protocol_sha256": "…",
  "run_manifests": {"plan_ids": ["plan-…"], "attempt_set_sha256": "…",
                    "review_set_sha256": "…", "model_profiles": ["trida-7b@c37755e8"]},
  "phase1": {
    "conditions": ["original", "trida_tau_070", "trida_tau_090", "trida_tau_100"],
    "clips": {
      "<clip_key>": {
        "media": {"video": "rel/path.mp4", "video_sha256": "…", "duration_ms": 10032,
                  "frames": [{"at_ms": 1000, "still": "…", "still_sha256": "…",
                              "objects": [{"id": "…", "label": "…", "mask": "…", "mask_sha256": "…"}]}]},
        "a1s_options": [{"value": "s1", "label": "…"}],
        "first_track": {"stimulus_id": "st-…", "source_ids": ["src-…"],
                        "cues": [{"index": 0, "start_ms": 0, "end_ms": 2508, "text": "…"}]},
        "second_tracks": {
          "original": {"stimulus_id": "st-…", "same_as_first": true},
          "trida_tau_090": {"stimulus_id": "st-…",
                            "cues": [{"index": 0, "start_ms": 0, "end_ms": 2508, "text": "…",
                                      "attempt_id": "att-…"}]}},
        "r2": {"options": [{"value": "r1", "label": "…"}],
               "keys": {"st-…": {"source": ["r1", "r3"], "shown": ["r1"]}}}
      }
    }
  },
  "phase2": {
    "tau": 0.9,
    "lattice": {"texture_steps": [0, 2, 4], "context_steps": ["broad", "specific", "relation"],
                "cells": [{"cell_id": "k2-specific", "texture_index": 1, "k": 2, "context_index": 1}],
                "centre": "k2-specific",
                "presets": {"both_brief": "k0-broad", "texture": "k4-broad",
                            "context": "k0-relation", "both_detailed": "k4-relation"}},
    "videos": {"<video_key>": {"media": {"video": "…", "video_sha256": "…", "duration_ms": 299679},
               "cues": [{"index": 0, "start_ms": 0, "end_ms": 5000,
                         "cells": {"k2-specific": {"stimulus_id": "st-…", "text": "…",
                                                   "served_cell": "k2-specific", "attempt_id": "att-…"}}}]}}
  },
  "exclusions": [{"unit": "<clip_key>", "cue": 2, "condition": "trida_tau_070",
                  "reason": "length_over_96", "attempt_id": "att-…"}],
  "substitutions": [{"video": "<video_key>", "cue": 7, "requested_cell": "k4-relation",
                     "served_cell": "k4-specific", "reason": "adds_sound_absent_from_evidence"}]
}
```

### 8.2 CaptionSource — `diff.caption-source/v1`

```json
{"schema": "diff.caption-source/v1", "source_id": "src-<sha256[:16]>", "phase": 1,
 "unit": {"kind": "clip", "key": "<clip_key>"}, "cue_index": 0, "start_ms": 0, "end_ms": 2508,
 "text": "…",
 "properties": {"audiovisual_congruence": "congruent", "description_depth": "shallow"},
 "evidence": {"events": [{"event_id": "e1", "order": 0, "source": "<source>", "category": "things",
                          "count": "one", "action": "<action>", "qualities": ["loud", "low_pitched"],
                          "location": "<location>", "relation": null, "certainty": "clear"}]},
 "provenance": {"text": "human", "evidence": "human", "origin": "…",
                "reviewed_by": ["R1"], "reviewed_at": "2026-…"},
 "status": "reviewed"}
```

`provenance.text` / `.evidence` ∈ {`human`, `gemma-draft-reviewed`,
`gemma-draft-unreviewed`, `assistant-draft-reviewed`, …}. A bundle refuses any
`*-unreviewed` value. Qualities must come from the closed vocabulary.

### 8.3 LatticeCell

`{"cell_id": "k2-specific", "texture_index": 1, "k": 2, "context_index": 1,
"context": "specific", "is_centre": true, "corner": null}`. The browser
receives only an opaque token and the two indices.

### 8.4 Job — `diff.job/v2` (revised for Dream)

```json
{"schema": "diff.job/v2", "job_id": "job-…", "phase": 1, "purpose": "development",
 "source_id": "src-…", "source_sha256": "…", "unit_key": "…", "cue_index": 0,
 "condition": "dream_k2", "cell": null,
 "model": {"profile_id": "dream-v0-instruct-7b@05334cb9", "runner": "dream"},
 "prompt": {"template_id": "rewrite-p1", "template_version": 1, "template_sha256": "…",
            "evidence_renderer_version": 1, "context_rules_version": null,
            "system_message": "You are a helpful assistant.", "text": "…", "text_sha256": "…"},
 "generate": {"max_new_tokens": 128, "steps": 64, "temperature": 0.0, "top_p": null, "top_k": null,
              "alg": "entropy", "alg_temp": 0.0, "eps": 0.001, "mask_token_id": 151666,
              "output_history": true, "return_dict_in_generate": true},
 "caps": {"canvas_tokens": 80, "steps": 40, "expected_forward_calls": 40, "timeout_s": 60},
 "checks_profile": "p1-v2"}
```

A `qwen` job's `generate` is `{"max_new_tokens": 128, "do_sample": false,
"repetition_penalty": 1.0, "eos_token_id": [151645, 151643]}`. A phase-2 cell
carries `{"cell_id": "t2-specific", "quality_limit": 2, "context": "specific",
…}`; `quality_limit` is the template's `{k}`, never the decoding k.

`token_ids_sha256` and `token_count` are written at freeze; `job_id` excludes
them so it can be computed before freeze.

### 8.5 Attempt — `diff.attempt/v1`

```json
{"schema": "diff.attempt/v1", "attempt_id": "att-…", "job_id": "job-…", "attempt_number": 1,
 "retry_of": null, "retry_reason": null,
 "environment": {"lock_sha256": "…", "python": "…", "torch": "…", "transformers": "4.57.3",
                 "cuda": "…", "gpu": "…", "driver": "…", "deterministic_mode": false},
 "verified_files": {"modeling_trida.py": "100558ff…", "configuration_trida.py": "105a4722…"},
 "warmup": {"done": true, "compile_events": 0},
 "started_at": "…", "finished_at": "…", "wall_ms": 0,
 "status": "completed",
 "output": {"prompt_tokens": 0, "new_token_ids": [], "stop_index": 0, "new_positions": 0,
            "raw_text": "…", "text": "…"},
 "trace": {"path": "traces/att-….jsonl", "sha256": "…", "events": 0, "steps": 0,
           "forward_calls": 0, "prefill": "prefill_only", "replay_matches": true},
 "checks": {"profile": "p1-v1", "failures": [], "flags": [], "measures": {}},
 "terminal": true, "excluded": false, "exclusion_reason": null}
```

`status` ∈ {`completed`, `timeout`, `refused`, `worker_lost`, `device_error`,
`runner_error`, `trace_inconsistent`, `count_mismatch`, `rng_consumed`,
`leftover_mask`}. Revised for Dream, attempts also carry `resolved_config`,
`caps` (forward calls, steps, canvas), `rng_unchanged` (Qwen), `seed` (Dream, since v4) and `latency_ms`. The
output records `completion` (`terminated` or `incomplete`), `terminator`
(`endoftext` or `im_end`), the trimmed tail as a `postprocess` record, and
`leftover_mask_count`.

### 8.6 Decode event — `diff.decode-event/v2` (Dream; built after G0)

```json
{"attempt_id": "att-…", "step_index": 3, "t": 0.977, "s": 0.969,
 "masked": [{"position": 214, "token_id": 1234, "p_top1": 0.91, "p_top2": 0.04,
             "neg_entropy": -0.41, "confidence": -0.41, "rank": 0, "selected": true}],
 "selected_count": 1, "unmasked_count": 1, "selected_but_masked": [], "remaining_masks": 124}
```

Positions are absolute. A `postprocess` event records the cut and the trimmed
tail. The block, small-block, boundary-argmax and fallback fields are gone.
Before G0 only realized per-step counts are stored, from `output_history`
(`trace.level: history-counts`).

### 8.7 Assignment slot — `diff.slot/v1`

```json
{"schema": "diff.slot/v1", "slot_id": "slot-0007", "bundle_sha256": "…",
 "design": "bib-oriented-v1", "cycle": 0, "index": 7,
 "assignments": [
   {"phase": 1, "position": 0, "clip": "<clip_key>", "role": "second-track", "condition": "trida_tau_090"},
   {"phase": 1, "position": 1, "clip": "<clip_key>", "role": "second-track", "condition": "original"},
   {"phase": 2, "position": 0, "clip": "<video_key>", "role": "long-video", "condition": null}],
 "state": "free", "bound_session": null, "bound_at": null, "releases": []}
```

`state` ∈ {`free`, `bound`, `completed`, `released`, `consumed`}. The first
track is implicit for every clip (`role: first-track`, the source).

### 8.8 Response — `diff.response/v1`

```json
{"schema": "diff.response/v1", "response_id": "…", "session": "…", "phase": 1,
 "page": "P4", "view_id": "<session>-0-P1", "clip_index": 0, "clip": "<clip_key>",
 "viewing": "first", "stimulus_id": "st-…",
 "answers": {"F1": 0}, "presented_order": ["F1", "F2"],
 "instrument_hash": "6c97d64a…", "submitted_at": 0.0,
 "client": {"entry_clock_session_id": "…", "entered_mono_ms": 0.0,
            "clock_session_id": "…", "submitted_mono_ms": 0.0, "submitted_wall": "…"},
 "timing_valid": true, "analysis_excluded": false,
 "matches": [], "labels": [], "visual_metrics": []}
```

### 8.9 Exposure — `diff.exposure/v1`

Phase 2, reported by the client:

```json
{"schema": "diff.exposure/v1", "exposure_id": "<client uuid>", "session": "…", "phase": 2,
 "video_position": 0, "video": "<video_key>", "cue_index": 12, "cue_start_ms": 60000,
 "cue_end_ms": 65000, "stimulus_id": "st-…", "cell_id": "k2-specific",
 "start_ms": 60012.0, "end_ms": 65020.0, "incomplete": false,
 "client": {"clock_session_id": "…", "start_mono_ms": 0.0, "end_mono_ms": 0.0},
 "received_at": 0.0}
```

As in regen's display-interval v2, a displayed interval may run past
`cue_end_ms` until the browser replaces the text. Phase-1 exposures are derived
server-side as playback coverage ∩ cue spans of the view's fixed track, with
`"source": "coverage"`; no extra client traffic.

### 8.10 Control event — `diff.control-event/v1`

```json
{"schema": "diff.control-event/v1", "event_id": "<client uuid>", "session": "…",
 "video_position": 0, "video": "<video_key>", "origin": "slider:texture",
 "from_cell_id": "k2-specific", "requested_cell_id": "k4-specific", "position_ms": 61234.0,
 "client": {"clock_session_id": "…", "mono_ms": 0.0}, "received_at": 0.0}
```

`origin` ∈ {`slider:texture`, `slider:context`, `pad`, `preset:both_brief`,
`preset:texture`, `preset:context`, `preset:both_detailed`, `reset`}. Video
start at the centre is recorded as `origin: initial`.

### 8.11 PlaybackCoverage — `diff.playback-coverage/v1`

```json
{"schema": "diff.playback-coverage/v1", "session": "…", "phase": 1, "view_id": "…",
 "duration_ms": 10032, "coverage": [[0, 10032]], "position_ms": 10032, "epochs": 0,
 "gate": {"rule": "short", "coverage_min_ms": 9732, "position_min_ms": 9932, "passed_at": 0.0},
 "clock_session_ids": ["…"], "timing_valid": true}
```

`rule: long` uses `d − 1500` and `d − 500`.

## 9. Traceability

### 9.1 The brief's constraints (B-IDs) → modules and tests

All test files live in `tests/diff/`.

| B | Constraint | Modules | Tests |
|---|---|---|---|
| B-01 | Response order P0 → practice → per clip P1…P11 with transition → handoff → long videos with controls and breaks → P12 → P13 | `phase1_flow`, `phase2_viewing`, `study_protocol` | `test_response_order_parity`, `test_phase1_flow_pages` |
| B-02 | Item IDs, wording, workbook hash; 0–6 with zero valid; page validation; drafts; receipts; recovery; practice separation; setup gate | `phase1_flow`, `session_store` (imports `sheet_instrument`) | `test_phase1_flow_pages`, `test_session_store_receipts` |
| B-03 | Playback gates (short and long) | `phase1_flow`, `phase2_viewing` (`record_playback`) | `test_phase1_flow_pages`, `test_phase2_viewing_lattice` |
| B-04 | No model calls in the server; transition and controls are lookups | `participant_server`, `phase*` | `test_diff_import_boundary` |
| B-05 | Blindness of everything browser-fetchable; opaque IDs | `participant_server`, `phase*`, `stimulus_bundle` | `test_participant_blindness_audit` |
| B-06 | Export needs a researcher credential; never the participant cookie | `export_bundle`, `review_app` | `test_export_bundle_auth` |
| B-07 | No quality retries, repair, truncation or shortening; one identical retry after a transient failure | `generation_worker`, `output_checks` | `test_generation_worker_retry`, `test_output_checks_rules` |
| B-08 | Dream native `diffusion_generate`, batch 1, every keyword required and explicit; refusal before the call (follow-up §3) | `dream_runner`, `qwen_runner`, `model_profile` | `test_dream_runner_contract`, `test_qwen_runner_contract` |
| B-09 | SHA-256 of remote code and tokenizer before `trust_remote_code` | `model_profile` | `test_model_profile_hashes` |
| B-10 | No edits to `src/dpo` / `data` / `deploy` / live; only the five regen modules; never `dpo.candidates` / `contracts` / `models` or Gemma; run from the repo root | all | `test_diff_import_boundary`; review checklist |
| B-11 | Unique basenames, no stdlib shadowing, `python -m diff`, no `diff` console script | all, `pyproject.toml` | `test_diff_import_boundary` |
| B-12 | Separate locked generator env (transformers 4.46.2, torch 2.5.1), established by AC-03 and AC-07; server never imports torch | `envs/diff-generator`, `dream_runner` | `test_diff_import_boundary`; GPU AC-03/AC-07 pending |
| B-13 | Participant text Korean only | static assets, `phase*` | `test_participant_blindness_audit` (script scan) |
| B-14 | Phase-1 units, conditions, job = cue × condition | `caption_sources`, `job_plan` | `test_job_plan_identity` |
| B-15 | Phase-2 units, job = cue × cell, fixed τ | `job_plan`, `stimulus_bundle` | `test_job_plan_identity` |
| B-16 | Source = text + structured evidence + provenance; unreviewed drafts excluded | `caption_sources`, `stimulus_bundle` | `test_caption_sources_schema`, `test_stimulus_bundle_integrity` |
| B-17 | Text-only; acoustic detail only from evidence | `rewrite_prompts`, `output_checks` | `test_rewrite_prompts_versions`, `test_output_checks_rules` |
| B-18 | Lattice rules (steps, one path, snapping, step overlay, corner presets, centre start, deliver once, switch at boundary, IDs recorded) | `phase2_viewing`, `phase2_player.js`, `stimulus_bundle` | `test_phase2_viewing_lattice`, `phase2_player_vm.mjs` |
| B-19 | Versioned prompts; token-ID hash after the chat template | `rewrite_prompts`, `job_plan` | `test_rewrite_prompts_versions` |
| B-20 | Bundle contents; state holds bundle hash and slot; versions; finish on own version | `stimulus_bundle`, `session_store` | `test_stimulus_bundle_integrity`, `test_session_state_size` |
| B-21 | Slot table, Latin square / BIB, bind at consent, abandoned-slot rule | `assignment_schedule` | `test_assignment_schedule_balance` |
| B-22 | Store kept / dropped / added; client monotonic and `clock_session_id`; `timing_valid`; control events; exposures | `session_store`, `phase*` | `test_session_store_receipts`, `test_phase2_viewing_lattice` |
| B-23 | Module layout, `tests/diff`, hatch and mypy entries | `pyproject.toml` | `make check` |
| B-24 | Four stages; GPU code tested with fakes; nothing fabricated | all | fakes; reports state what ran |
| B-25 | G1 / G2 provisional gates | `freeze_gates` | `test_freeze_gates_thresholds` |
| B-26 | Open decisions as protocol data with defaults | `study_protocol` | `test_freeze_gates_thresholds`, `test_assignment_schedule_balance` |

### 9.2 PRD IDs (as amended)

"Brief" means the wording or scope is given by the brief. "Stage exit" means the
brief places the ID in a stage's exit criterion but does not give its wording.

| ID | Known from | Content | Maps to |
|---|---|---|---|
| FR-01 … FR-15 | — | *pending PRD* (presumably §6.1–6.3 generation, instrumentation, review) | provisionally Stages 1–2 |
| FR-16 | brief §6.4 | single protocol with a slot table | B-21, `assignment_schedule` |
| FR-17 | brief | exposure rule: each clip twice, P1 first track, P6 the one assigned second track; no other condition; reconnecting never changes it | B-21, `phase1_flow`; `test_phase1_flow_pages` |
| FR-18 | brief | covers `clip_id` | records and exports carry the real clip key; payloads never (B-05) |
| FR-19 | brief | covers page entry and submit times | §6.6, B-22 |
| FR-20, FR-21 | — | *pending PRD* | — |
| FR-22 | brief | covers kiosk export | §6.7, B-06 |
| FR-23, FR-24 | — | *pending PRD* (possibly §7–§9 materials and analysis) | — |
| AC-01, AC-02, AC-04, AC-06 … AC-09 | stage exit | *pending PRD*; Stage-1 exit on development clips | Stage 1 |
| AC-03 | brief + follow-up | 3 inputs × 3 conditions: identical token IDs with and without hooks and history; hook selections = history unmasking + `selected_but_masked`; with AC-07 it establishes the env | §4.5 (after G0), B-12 |
| AC-05 | follow-up | context and canvas edges replace block edges: 2,047 and 2,048 accepted, 2,049 refused; invalid steps refused; very short prompts | `dream_runner.context_problems` / `checked_call`; `test_dream_runner_contract` |
| AC-01, AC-07, AC-08 | follow-up | synthetic-logit recomputation; RNG states unchanged; forward-hook count = steps | AC-07/08 checked on every call (`model_runtime`, `dream_runner`); AC-01 after G0 |
| AC-20 | follow-up | bundle builder and lattice generator refuse without a passing G0 for the current profile | `analysis_decisions`, `job_plan.build_phase2_plan`; `test_job_plan_identity` |
| G0 | follow-up | Korean feasibility gate (see the Model change section) | `feasibility_gate`; `test_feasibility_gate` |
| AC-10, AC-12, AC-13, AC-14 | stage exit | *pending PRD*; Stage-3 exit | Stage 3 |
| AC-11 | brief | balance of clip × condition and of video order | B-21 |
| AC-15 | brief | blindness | B-05 |
| AC-16 | brief | lattice completeness | §5.4 verifier |
| AC-17 | brief | playback gates and recovery | B-03 |
| AC-18 | brief | session-state size and bundle versioning | B-20; bound: state ≤ 8 KiB excluding the draft, draft ≤ 16 KiB (A7) |
| AC-19 | brief | response-order parity | §7 |
| MC-08 | brief + follow-up | text-only; evidence the only grounding; `config.json` 131,072 against the README's 2,048, and 2,048 is used; remote code hashed | B-09, B-17, `model_profile` |
| MC-01 … MC-07 | — | *pending PRD* (§2 model contract) | `model_profile` |
| CV-08 / CV-10 / CV-13 / CV-14 / CV-15 | brief + follow-up | the canvas (80 / 128, set at M0 on 2026-10-10); sentence ending; caption band; two viewings; token IDs 151666 / 151643 / 151645 (BOS 151643 and 151665 recorded, refused in prompts) | `study_protocol`, `output_checks`, `model_profile` |
| CV-01 / CV-05 / CV-09 / CV-12 | follow-up | Dream pin; `top_p` and `top_k` None; ≤ 2,048 tokens; no RNG consumed (CV-02, CV-03, CV-06 deleted) | `model_profile`, `dream_runner` |
| IV-01 … IV-05 | brief | see changelog §11 | `study_protocol`, `assignment_schedule` |

## 10. Stages and exit criteria

Each stage lands as one-concern commits with why-bodies, and `make check` passes
on every intermediate tree.

### Stage 1 — Generator (PRD M0–M1), G0 first

**Built for G0 (follow-up §7 order):**
- the profile and hashes;
- prompt rendering with the system message;
- both runners with the §3 contract and their AC-02 tests;
- output checks;
- `python -m diff g0 plan|report|decide`;
- `envs/diff-generator` with its lock;
- the importers.

Tests run with fakes, never weights. The CPU tests against a vendored copy of
`generation_utils.py` come after G0.

**G0, in order:**
1. **Development sources:** at least 3 development-clip cues, with evidence and
   `material: development`. Six c2 clips, with their evidence as unreviewed
   drafts (Q15).
2. `g0 plan`, then `plan freeze` and `generate` in the generator env on a GPU
   with ≥ 20 GB.
3. `g0 report` writes the automatic report and the blind packet, and stops.
4. One fluent Korean reader rates naturalness 1–5.
5. `g0 decide` records the `AnalysisDecision`.

If G0 fails, stop and report. The researcher chooses between revised prompts on
development material (a new protocol version) and another model.

**After a passing G0:**
- the decode trace (logits and tokens hooks) and the vendored-loop CPU tests;
- AC-01, AC-03, AC-04 and AC-09.

M0's canvas and timeout were settled earlier than planned, on 2026-10-10,
between the first and second G0 runs: the first run showed that both decide
G0's outcome.

**Exit:** a passing G0, then AC-01–AC-09 as amended on development clips.

### Stage 2 — Review and freeze (PRD M2)

**Build:** `review_app` and `review_console.*`, `freeze_gates`,
`stimulus_bundle` (builder, verifier, registry), and phase-2 job plans.

**Prerequisites:**
- reviewers named;
- phase-2 evidence structured and reviewed;
- R2 options authored.

**GPU:** development and participant phase-1 jobs; after G1 fixes τ, the phase-2
lattice (1,620 jobs).

**Exit:** G1 and G2 met, or a recorded hold decision.

### Stage 3 — Participant server (PRD M3)

**Build:** `session_store`, `assignment_schedule`, `phase1_flow`,
`phase2_viewing`, `participant_server`, `export_bundle`, and the static assets.

**Tests:**
- the parity harness (§7);
- the blindness crawler: scan every response, header, cookie and URL for τ
  values, condition IDs, model names (`trida`, `tri-7b`, `trillion`, `gemma`),
  answer-key fields and real clip and media stems;
- AC-17 recovery: reload mid-view, lost reply, stale revision;
- AC-18 state size across a full scripted session, and two-bundle versioning;
- AC-16 completeness;
- AC-11 balance;
- export authentication;
- a browser render audit that every bundle text fits the fixed caption band in
  at most 2 rendered lines (CV-13);
- a headless walk of both phases with webm media, because Playwright's Chromium
  lacks H.264.

**Exit:** parity test, AC-10–AC-19 and usability-pilot readiness. Readiness is a
written checklist: Korean-only walk on the kiosk hardware, consent and debrief
texts approved, slot table generated for the active bundle, and an export
round-trip.

### Stage 4 — Detach

**Copy.** Copy the regen code diff still imports into `src/diff` under new
unique basenames:

| From | To |
|---|---|
| `sheet_instrument` + workbook JSON | `workbook_instrument.py` |
| `playback` + `number` | `playback_coverage.py` |
| `points` | `frame_points.py` |
| `study_media` + `bound_media` / `fingerprint` | `video_delivery.py` |
| `Conflict` | diff's own |

`SCHEMA` and `PROVENANCE` are kept verbatim, so the workbook hash `b23eb462…` and
the snapshot hash `6c97d64a…` are unchanged; a test asserts both.

**Remove.** Remove every `dpo` import from `src/diff`. The parity test keeps
importing regen and is skipped if `dpo` is absent.

**Propose, do not perform.** Drop `peft`, `bitsandbytes`, `accelerate`, `torch`,
`torchvision`, `librosa`, `av` and `transformers` from the shared project once
`dpo` is retired, leaving diff's server needing FastAPI, uvicorn and Pillow.

### Phase C — review

- **Static checks:** `ruff check`, `ruff format --check`, strict `mypy`, and
  `pytest tests/diff`.
- **Regen unchanged:** the regen suites (`tests/regen`,
  `src/dpo/regen/tests`) must pass with regen untouched.
- **Parity and blindness:** the AC-19 parity run and the AC-15 audit.
- **Independent review:** a reviewer who did not write the code checks the
  change against the brief and the PRD.
- **Report:** commands with test counts, what was not verified, and every
  deviation.

## 11. PRD v1.1 changelog (from the brief's amendments)

**Superseded for phase 1** (noted 2026-10-25). Since 2026-10-18 every clip's
second caption is rewritten, so there is no `original` control and no τ
comparison in phase 1. Since 2026-10-23 the second caption's mode comes from
the binding model, and the analysis is the moderator model in the section at
the top. The phase-1 parts of §1's research question, §5.1's IV-02 and §9's
primary contrast below are therefore history. Phase 2 is still descriptive.

**Amendments, by PRD section.**

| PRD | Change |
|---|---|
| §1 | **Research question:** how do Trida rewrites under τ differ from the source caption, and how does the difference relate to the second viewing's change in restoration ratings (P4→P9), visual and auditory selections (V1→V2, A1s→A2s), caption ratings (S1–S3) and recall (R2)? How do participants move through the pre-generated lattice in phase 2? No directional hypothesis for τ. |
| §2 | **MC-08 added:** text-only model; evidence text is the only grounding. Record `max_position_embeddings` 4096 (config) against 8,192 (model card). Hash the remote code. |
| §3 | **Scope:** video stimulus validation and the participant server are in scope. The calibrated slider becomes a pre-generated prompt lattice, never τ. |
| §5.1 | **Independent variables:** IV-01 τ becomes either the second-track condition (option A) or fixed (option C). IV-02 is the second-track condition, with `original` as the control. IV-03 is removed. IV-04 becomes a sound-event fidelity annotation. New IV-05 is the lattice cell, chosen by the participant, not assigned. The first-caption 2×2 condition becomes a source property. |
| §5.2 | **Controlled variables:** CV-08 sets the budget from measured source token counts at M0 (initially 128 tokens in phase 1 and 192 in phase 2); character limits live in the output checks. CV-10 keeps the source caption's sentence ending. CV-13 is a caption band over the video, with fixed font and position. CV-14 is two fixed viewings per clip. |
| §5.3 | **Measures:** human measures become the regen items. Sound-event fidelity, quality-term count and specific-source count join the text measures. |
| §6.2 | **Per-cue caps:** new positions = budget + 32 and a 30 s timeout, finalised after the M0 smoke test. |
| §6.3 | **Review:** phase 1 gets double review of every output. Phase 2 gets automated checks on all outputs, plus double review on a stratified sample and on everything flagged. |
| §6.4 | **Server requirements:** FR-16 is a single protocol with a slot table. FR-17 is the exposure rule. FR-18 covers `clip_id`. FR-19 covers page entry and submit times. FR-22 covers kiosk export. |
| §7 | **Materials:** caption sources, the two prompts, and R2 with two answer keys (source-based and shown-track-based). |
| §8 | **Data contracts:** add CaptionSource, LatticeCell, Exposure, ControlEvent and PlaybackCoverage. Assignment gains phase, clip and role; Response gains page and view ID. |
| §9 | **Analysis:** the primary contrast is the P9−P4 change by second-track condition against `original`, using ordinal models with participant and clip effects (clip fixed when there are only two). Fidelity and length are outcomes of the condition, not covariates. Phase 2 is descriptive. |
| §10 | **Acceptance:** AC-11 balances clip × condition and video order. Add AC-15 blindness, AC-16 lattice completeness, AC-17 playback gates and recovery, AC-18 session-state size and bundle versioning, and AC-19 response-order parity. |
| §11 | **Gates:** G1 is replaced by the caption gates (technical, readability, difference) and G2 is added (monotonicity, distinctness); the thresholds are provisional. |
| §12 | **Risks added:** transformers version; loss of audio grounding; lattice review load; unreviewed phase-2 sources; fixed video order; incomplete disclosure and debrief. |

**New items:**
- **FR-17 text:** a participant sees each clip twice, at P1 with the first
  track and at P6 with the one assigned second track. No other condition of
  that clip is shown to them, and reconnecting never changes the assignment.
- **§11 gate wording:** exactly as in §5.2 of this plan.
- **Open decisions:** recorded as protocol data, as in §12 of this plan.

**Engineering additions this plan proposes for v1.1** (not in the brief;
confirm):
- **§2 facts:** record the `generate` facts in §2.3 — `**kwargs` swallowing,
  hard-coded suppression of 128012/128004, the per-step `block_size` default,
  block flooring, the argmax tokens after prefill and after each block, and the
  32-token prefill skip.
- **§6.2 guards:** the prompt-length guard (`≠ block_size`), the context guard
  against 4096, and a warm-up compile that is not charged to the timeout.
- **§8 attempt statuses:** `refused` and `trace_inconsistent`.

The three Trida-specific additions above lapse with the model change.

**Follow-up amendments (Dream, 2026-10-09).** Each row merges into the
same-section row above; where they conflict, this row wins.

| PRD | Change |
|---|---|
| Header, §4, §13 | **Model:** `Dream-org/Dream-v0-Instruct-7B` @ `05334cb…`; baseline Qwen2.5-7B-Instruct @ `a09a354…`; remote-code and tokenizer SHA-256s (Appendix B); sources: the pinned HF files, the `HKUNLP/Dream` README at `31f94a6`, arXiv 2508.15487. |
| §1, §9 | **Factor:** τ becomes the decoding condition k (parallelism); still no directional hypothesis. |
| §2 | **Contract:** §2.1–2.2 are replaced by the follow-up's §3 (call contract) and §4 (instrumentation). MC-08: text-only; record `config.json`'s 131,072 against the README's 2,048, and use 2,048. |
| §3 | **Scope:** decoding-condition generation replaces τ generation; the block-size and cache row is deleted; G0 is added as P0. |
| §5.1 | **IVs:** IV-01 is the decoding condition k (option A) or fixed (option C); IV-02 levels are `original`, `dream_k1`, `dream_k2`, `dream_k4`. |
| §5.2 | **CVs:** CV-01 is the Dream pin. CV-02, CV-03 and CV-06 (block, sub-block, block cache) are deleted. CV-08 becomes the canvas. `alg`, `alg_temp` and `eps` are added. CV-05: `top_p` and `top_k` None. CV-09: ≤ 2,048 tokens. CV-12: no RNG consumed. CV-15: 151666 / 151643 / 151645. |
| §5.3 | **DVs:** DV-M01: `steps` is set, not measured. DV-M02: NFE = `steps`. DV-M03: realized per-step counts. DV-M04 is deleted. |
| §6.1, §8 | **Requirements:** FR-03 to FR-06 and DecodeEvent as in the follow-up's §3–§4; the §8.1 manifest example is rewritten. |
| §7 | **Prompts:** both models get the system message "You are a helpful assistant."; the prompt texts are unchanged. |
| §10, §11 | **Acceptance:** the AC changes in the Model change section; G0 comes before G1; G1 difference reads "two or more decoding-condition tracks". |
| §12 | **Risks added:** unverified Korean ability; the 2,048-token context; preprint status; an old pinned stack; canvas-size effects; a GPU of at least 20 GB. |

**M0 decisions (researcher, 2026-10-10; protocol v3).** These follow the first
G0 run, and each row wins over the rows above.

| PRD | Change |
|---|---|
| §5.2 | **CV-08:** the canvas is 80 tokens in phase 1 and 128 in phase 2. It covers the 96- and 160-character limits at about 0.75 tokens per character, measured on the development captions. |
| §6.2 | **Timeout:** 60 s per job, about twice the slowest measured job. Generation is offline, so the timeout only catches a stuck worker. |
| §11 | **G0 script check:** besides Han/kana, stray Latin and `incomplete`, it fails on empty text, U+FFFD and leftover masks. The share is over every `dream_k1` job, so a job with no output counts as a failure. |

**Protocol v4 decisions (researcher, 2026-10-11).** These follow the second
G0 run and the k1 diagnostic, and each row wins over the rows above.

| PRD | Change |
|---|---|
| §5.2 | **CV-05 / CV-12:** Dream samples at temperature 0.2 with `top_p` 0.95, the README's settings, because greedy decoding returned empty captions. Each call is seeded from its job ID, and the warm-up refuses a device where two seeded calls differ. Qwen stays greedy. |
| §7 | **Prompts:** `rewrite-p1` version 2, aimed at Dream copying the source caption. The assistant drafted it at the researcher's request and chose it on held-out captions by a rule fixed beforehand (see "Protocol v4"). |
| §11 | **G0 script check:** an exact copy of the cue's source caption fails it. Near-copies stay flags. |

**Live phase 2 decisions (researcher, 2026-10-11).** These follow G0 run 3,
and each row wins over the rows above.

| PRD | Change |
|---|---|
| §4, §6 | **Phase 2 is live:** Dream (`dream_k2`) rewrites each cue's starting caption during the watch, following the participant's sliders. No Gemma, no pre-generated phase-2 lattice or bundle. The participant server still imports no torch; one Dream worker per GPU runs in the generator environment. |
| §6.2 | **Timing:** a tweak targets the first boundary after `position + L` (adaptive latency); the previous caption stays until a new one is ready; each video opens with Dream's rewrite. |
| §7 | **Prompts:** `live-p2` v1 for live rewrites and `rewrite-p1` v3 for phase 1, both chosen on held-out captions by a rule fixed beforehand. |
| §9, §11 | **G0:** the human rating is waived (recorded as `waived`). Participants answer the researcher's post-watch items. |

## 12. Open decisions (protocol data; default unless you decide otherwise)

**Superseded** (noted 2026-10-25). This table predates the return to Trida
(2026-10-12) and the phase-1 decisions of 2026-10-18 and 2026-10-23. Its
phase-1 manipulation (`dream_k1`/`k2`/`k4` against `original`) and its
decoding rows no longer apply. The open decisions are under "Open for the
researcher" at the end of the section at the top.

| Decision | Default | Alternative |
|---|---|---|
| Phase-1 manipulation | A: `dream_k1`/`k2`/`k4` with `original` as the control | C: steering tracks at one decoding condition |
| Decoding factor | parallelism k at `alg="entropy"` | remasking order (`entropy`, `topk_margin`, `maskgit_plus`) at k = 1 |
| Phase-2 decoding | fixed, chosen after G1 (default `dream_k1`, the README's setting) | assigned per video |
| Lattice | 3×3 | 5×3 after G2 |
| Baseline | Qwen2.5-7B-Instruct, review stage only | participant condition (40-slot phase-1 cycle) |
| Phase-2 sources | reviewed Gemma drafts with provenance | re-authored by people |
| Missing lattice cell | nearest-cell substitute, recorded with the original cell | exclude the video |
| Abandoned slot | reissue only after researcher release, and only if no formal response exists | reissue after release regardless; or auto-release after inactivity |

## 13. Assumptions

- **A1:** the parity target is regen at `3842302` with the default sheet policy.
  The uncommitted experience policy is not reproduced.
- **A2:** `top_p` is not passed to Trida and is recorded as `not_applied`.
- **A3:** `B` in AC-05 is `block_size` = 32.
- **A4:** a timeout or a missing stop within the cap is a terminal cap outcome,
  not a transient failure, so it is not retried.
- **A5:** naturalness uses a 1–5 scale, since the gates use 3 and 4.
- **A6:** track naturalness is the minimum over its cues; the final rating is the
  adjudicated value, or the lower of the two.
- **A7:** the AC-18 bounds are state ≤ 8 KiB excluding the draft and the draft
  ≤ 16 KiB, pending the PRD's numbers.
- **A8:** the page id `transition` replaces `generate`; the parity harness maps
  the two.
- **A9:** drafts behave as in regen: `sessionStorage` first, server draft only
  for a fresh tab, and the client does not post drafts.
- **A10:** the instrument is the one at `3842302`. Until Stage 4 a workbook
  change blocks running sessions, as in regen; after Stage 4, sessions render
  from their stored snapshot.
- **A11:** ports are 8782 (participant) and 8783 (researcher and review), both
  bound to loopback. Public exposure is out of scope, and `deploy/` is not
  touched.
- **A12:** development clips come from the c2 pool minus the participant clips
  (10 candidates); the suggestion is 8, so "3/4 of clips" is meaningful.

## 14. Open questions

1. **Q1 — PRD.** Please add `trida-korean-hci-prd.md`. §9.2 is completed against
   it before Stage 1.
2. **Q2 — transitive Gemma import.** Accept the transitive `dpo.caption.writer`
   load until Stage 4 (default), or detach the five modules at Stage 1?
3. **Q3 — generator env path.** *(Answered by default; renamed for the model change to
   `envs/diff-generator/`.)* `envs/trida-generator/` with its own lock (new
   top-level directory)?
4. **Q4 — phase-1 source text.** The source caption for each clip's four cues
   must be authored. Is the Korean burnt-in caption, the English `orig_text`
   or new copy the starting point? They disagree for at least three c2 clips.
5. **Q5 — phase-2 evidence.** Who restructures and reviews the 180 windows'
   evidence into events with source, action, qualities, location/relation and
   certainty, and who reviews the fallbacks?
6. **Q6 — R2.** Who authors the R2 options per clip (3–4 mentioned sounds plus
   1–2 distractors) and confirms the two keys?
7. **Q7 — reviewers.** Reviewer and adjudicator roster, and whether the
   phase-2 sample fraction (10 %) is acceptable.
8. **Q8 — Tri as a participant condition.** If Tri becomes one, accept a
   120-slot full cycle, or partial video-order balance?
9. **Q9 — system message.** Trida's template injects "You are Trillion…".
   Keep it (default), or pass an explicit system message?
10. **Q10 — phase-2 wording.** Approve the three `{context_rule}` drafts, the
    `k = 0` wording, and the rendering of an empty quality list.
11. **Q11 — participant wording.** Approve the revised "ready" and "break"
    texts. Confirm that the debrief text discloses the automatic rewriting and
    its conditions (§12 risk; IRB).
12. **Q12 — GPU timing.** Resolved in practice: the smoke and both G0 runs
    used GPU 1. GPU 0 holds the xart project's GPU API container
    (`xart-gpu-api-1`, 15.5 GB), not regen. Stopping it, as the researcher
    asked on 2026-10-10, needs `docker stop xart-gpu-api-1` run by the
    researcher.
13. **Q13 — first-caption 2×2.** The 2×2 becomes a source property. Must the
    participant clips cover particular cells of it?
14. **Q14 — analysis code.** Are the §9 analysis models part of this change, or
    only the export tables?
15. **Q15 — development evidence for G0.** G0 needs development-clip cues,
    never participant clips. Each needs its source caption, structured
    evidence and `material: development`; at least 3 cues give the 12 corner
    prompts. A cue is one timed caption line, and regen's 10 s clips carry one
    each (`cue_slots: 1`).

    **Drafted 2026-10-09 from regen's prepared c2 material** (the earlier "none
    exist" was wrong):
    - **Participant material, never G0:** `bangkok_034` and `singapore_286`
      (the live pair) and the three viewing videos.
    - **Development set (6):** `amsterdam_181`, `amsterdam_202`,
      `bangkok_209`, `bangkok_316`, `bangkok_328` and `singapore_134`.
      - The source caption is the Korean caption transcribed in
        `scripts/c2-korean-captions.json`, copied verbatim.
      - The evidence was drafted by the assistant from each clip's
        `tidy_data.csv` AudioSet labels, its `orig_text` and the Korean
        caption, not by listening. It is marked
        `assistant-draft-from-annotations-unreviewed`, so G0 may read it but no
        bundle ever can.
      - Qualities appear only where a description states one (constant →
        steady, faint → soft), so the texture corners have little to draw on.
      - `bangkok_316` was in regen's live pair until 2026-09-07. It is not in
        diff's flow.
    - **Excluded:**
      - `amsterdam_012` and `singapore_040`: the Korean caption names a
        different sound from the labels (train against police siren;
        motorcycle against car, wind and bird).
      - `amsterdam_057`: the caption adds car horns the labels lack.
      - `singapore_067`: three lines, against a two-line limit.
    - **File:** `artifacts/diff/sources/c2-development-draft.json`
      (gitignored).

    Still open: the researcher reviews the evidence by listening, and any
    change means a new G0 run before rating.
16. **Q16 — the Trida snapshot.** Resolved 2026-10-09: the researcher had
    `artifacts/diff/models/trida-7b-c37755e8` (15 GB, gitignored, unused
    since the model change) deleted, along with its download log.

## 15. Risks

From the §12 amendments, plus what the reading found:

- **Dream (follow-up §12).**
  - **Korean is unverified:** the training data and reported evaluations are
    mostly English and code; G0 exists for this.
  - **Context:** 2,048 tokens.
  - **Status:** preprint (arXiv 2508.15487).
  - **Stack:** an old pinned transformers 4.46.2 / torch 2.5.1.
  - **Canvas-size effects:** the model fills the whole canvas, so the canvas is
    a protocol value, not a cap.
  - **GPU:** at least 20 GB.
  - **Timeout (resolved):** the first G0 run measured about 32 s for k = 1
    on the 192 canvas, over the 30 s placeholder. M0 (2026-10-10) set 60 s and
    a 128 phase-2 canvas.

- **Transformers version.** Trida's config is from 4.57.3. Its remote code
  imports helpers that moved in 5.x. The generator env pins 4.57.3 and is
  accepted only through AC-03.
- **Compile behaviour.** `max-autotune` compile and recompiles on new shapes
  could exceed the 30 s cue timeout. Mitigations: warm-up, recording compile
  counts, and a timeout finalised at M0.
- **GPU nondeterminism.** It could fail AC-03 spuriously. Mitigation:
  deterministic mode, recorded.
- **Loss of audio grounding.** The model is text-only, so acoustic detail
  exists only where the evidence has it. Evidence quality bounds caption
  quality, and identical-neighbour rates will rise where qualities are sparse.
- **Lattice review load.** 1,620 outputs; a 10 % sample plus flags, double
  reviewed.
- **Unreviewed phase-2 sources** block the phase-2 bundle entirely until
  reviewed.
- **Video order.** Regen fixes it in manifest order; the slot table removes
  this.
- **Incomplete disclosure and debrief** need approved text.
- **Uncommitted regen work** landing could change the parity baseline; the
  harness pins the default policy.

## Appendix A — weight shard SHA-256 (LFS values reported by the Hub API)

Trida-7B @ `c37755e8`:
`model-00001-of-00004` `e5e9a4589f938a297ead260acedab905c6dc4dc84e190628c58bdc70a560192f`,
`00002` `624ad919451158c5bc7229c6346dca79631abfedcd112f9bf128447ca92125c8`,
`00003` `ae69bf99459c60fb06138805a4ed8cd8e4ad58a76e2d61220a15634947d23858`,
`00004` `5f0a7f453efe5cea64bc17ef717b4177030167f0d724da0d8bda1fd2f89d4a29`;
`model.safetensors.index.json`
`0ecd88d572d5d263671d0a725369538b5d598e88c6523dc3ddee8f6becb3f461` (downloaded).

Tri-7B @ `94e2cc88`:
`model-00001-of-00007` `642ccdb85f37ea9da6a994ab69f5bf1161d175cdde1a5da1a9c18450735ca401`,
`00002` `8911da53d5abc25fd16632e112d2eeae6847d8e8fa510a8174e7bc540c66d75d`,
`00003` `c6736729000c393742c1ca66170f9a6814fa776a9d261bb3a0a9a27b23a68ff5`,
`00004` `a6c1d7b2c07e3a2e95a203da0177e6ab859719fefff9084441964468a60cea52`,
`00005` `9d401836b166724f99bfa3f4d5a7bb707f261957314bf73c49613fa5c4781e3b`,
`00006` `38cdf11e0adca5ab9c1302400423d4b11bac24cb7e0d194d34da4ea2a333502c`,
`00007` `b066ac2e2d862d570b6fddfa27306aa2597f2c8c06b7b7e2309e7291dba18532`.
The shards were not downloaded. These hashes are the Hub's, and are re-verified
locally at M0.

## Appendix B — Dream and Qwen pins (SHA-256 of the pinned files)

Dream-v0-Instruct-7B @ `05334cb9faaf763692dcf9d8737c642be2b2a6ae` (downloaded
at the exact revision; git blob IDs matched the Hub):

| File | SHA-256 |
|---|---|
| `configuration_dream.py` | `24d038962ccf16361595494461000ec0c1b043653c7494d7fc835c96f121f0b9` |
| `modeling_dream.py` | `3166e789f0d69beb1f4fbdee2317953d1c1c3b9bc21479b109afc5dd059de5b3` |
| `generation_utils.py` | `7f8ad01484898946b3c9c9d5ebc85e5ac726061d0fa7b4a522c4b8110c13a921` |
| `tokenization_dream.py` | `1d7f04104fa673831f48256e3cd42cd0bbd1bddd801b4abea5c74cc47d98bdd2` |
| `config.json` | `11afc7fe7a1881e73dcd70cca6868aaaf8eb89ad45ea5fc6426c3fb82dec9340` |
| `generation_config.json` | `e4fa42a31ef22740804534395c7a18f335a3c1f9abd9db83a6941b1c8748b2ee` |
| `tokenizer_config.json` | `b0e9c1a6299d8832a1362d6b24f007de8a2acfe0243d8a3a1d1b57aa8250f7ee` |
| `special_tokens_map.json` | `fa14641fdc084244b0a530556b1d5bfcb9fa0ed9151a780ffec3b44ba96a0dfe` |
| `added_tokens.json` | `32ad9d49fafda64e737853737fe3be3e33cce3f719a159ed73f04cbd81a690ce` |
| `vocab.json` | `87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d` |
| `merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `model.safetensors.index.json` | `998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb` |

Dream weight shards (LFS SHA-256 from the Hub API): `3014d849…a3177aec993`,
`78fc44a7…7222aa57`, `eac5afaf…92c378e48`, `398da65f…259712b3cb7` (in full in
`model_profile.py`).

Qwen2.5-7B-Instruct @ `a09a35458c702b33eeacc393d103063234e8bc28`:

| File | SHA-256 |
|---|---|
| `config.json` | `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c` |
| `generation_config.json` | `3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f` |
| `tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tokenizer_config.json` | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `model.safetensors.index.json` | `624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028` |
