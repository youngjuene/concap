# Regen interface review and user-test response analysis plan

Review date: 14 September 2026. Scope: the prepared integrated `regen` interface, its researcher dashboard, and the current two-chapter protocol. This report reviews the implementation and proposes a human-test and analysis procedure; it does not report participant efficacy results.

**Overall assessment:** Regen provides an implemented, traceable workflow for collecting caption-experience ratings, visual and sound observations, control interactions, and caption-display records. It supports a formative user study. The present design can describe usability and within-person changes in reported experience, but cannot isolate a causal benefit of personalization. Questionnaire validation and the human-study protocol remain separate from software readiness.

Throughout this report, **Evidence** means directly inspected implementation, configuration, or verification artifacts; **Interpretation** means a conclusion from that evidence; **Proposed** means a study or analysis procedure that is not automatically implemented.

## 1. Review findings

| Priority | Finding | Basis and confidence | Consequence for the study |
| --- | --- | --- | --- |
| 1 | The current integrated flow separates clip-specific answers from chapter-wide answers. | **Evidence; high.** Short-clip scopes in `app.py:299`; long-video/final submissions in `study_api.py:517`. | Pair original/updated ART within the same short clip; analyze PRSS at chapter level. |
| 2 | Response provenance and exposure records support an auditable analysis. | **Evidence; high.** Frozen instruments, profile/media hashes, transactional records, and read-only dashboard exports. | Preserve version and stimulus identity through every derived table. |
| 3 | The design is a sequential, adaptive experience study. | **Interpretation; high.** Prepared captions always precede updated captions; Phase 1 answers inform Phase 2 generation; participants choose their controls. | Report associations and descriptive paired changes. Neither phase establishes a randomized treatment effect. |
| 4 | The built-in ART/PRSS questions are pilot wording, with draft Korean translations. | **Evidence; high.** `items/default.json:3` explicitly declares placeholder provenance. | Do not label resulting totals validated ART/PRSS scores or use them as evidence of clinical restoration. |
| 5 | Software QA is substantially stronger than the available human-response evidence. | **Evidence; high.** Fresh checks below passed; the inspected non-QA cohort contains no completed sessions. | Treat the analysis below as a prospective plan, not a finding that participants benefited. |

The code takes precedence over older prose where they differ. For example, some documentation still describes pending long videos or an earlier different-clip protocol. The inspected `data/live/viewing-study.json` has three ready videos; the new integrated flow repeats each short clip. Historical sessions retain their original protocol and must remain identifiable.

## 2. What the participant experiences

### Chapter 1: short clips and caption regeneration

**Evidence.** The inspected prepared configuration contains two 10.032-second clips. Participant sequence parity determines whether clip A or B comes first. This alternates **clip order**, not caption-condition order. Each clip follows the same sequence:

1. Watch the clip with prepared captions.
2. Answer eight scene-experience items organized around being away, fascination, extent, and compatibility.
3. Mark meaningful objects/features in selected still frames. At least one mark is required in the inspected configuration; marks can be moved, removed, or entered with keyboard controls.
4. Report, from memory, whether each of five sound families was heard: human, animal, things, music, and natural sounds. Every family requires an explicit yes/no answer.
5. Wait while captions are regenerated from the submitted observations, then watch the **same clip** with updated captions.
6. Answer the same eight scene-experience items and six caption items covering credibility and restfulness/distraction.

After both clips, participants answer eight PRSS items once, considering the whole short-clip chapter. There is no separate prepared-caption credibility questionnaire. Therefore, credibility cannot be compared before versus after regeneration using the current questions.

The interface uses a forward progress rail, explicit remaining-answer feedback, saved drafts, and recovery after interrupted playback or requests. Language is selected before playback and retained through the handoff. [Sources: `app.py:267`, `app.py:299`, `copy.py:134`, `copy.py:176`, `regen.js:94`, `STUDY.md`.]

### Chapter 2: longer videos and live caption controls

**Evidence.** Participants watch three approximately five-minute videos in manifest order. The inspected durations are 299.679, 300.259, and 300.142 seconds. No long-video randomization is implemented in this path (`study_api.py:436`).

Two controls remain available during viewing:

| Participant label | Stored axis | Meaning |
| --- | --- | --- |
| Acoustic detail | `texture` | Brief descriptions through more detailed acoustic descriptions. |
| Source and scene detail | `context` | General sources through more specific sources and scene relationships. |

Both begin at 50% in the integrated flow, with 1% increments. Native sliders provide keyboard access; an optional two-dimensional pad adjusts both axes. A temporary overlay shows the **selected** levels. A separate caption status identifies the levels actually applied or states that a prepared fallback is showing. Updates apply at caption boundaries, keeping text stable within a cue. [Sources: `study.js:333`, `study_api.py:445`, `STEERING_METER_QA.md`.]

The standalone v2 entry is a separate protocol: it collects two explicit preference ratings and can initialize the controls from those answers. The neutral-start description and response counts in this report apply to integrated handoff sessions.

After each video, four questions ask about caption accuracy, usefulness of acoustic detail, identifying sources, and matching the participant's information needs. After all three, the final questionnaire asks chapter-wide PRSS, success using each control, effort, perceived update speed, and an optional comment.

**Interpretation.** The layout makes the two detail dimensions and local/global survey scopes visible. Human testing should establish whether participants understand them. In particular, the integrated flow starts at neutral control values even though the UI says “Your calibrated settings are ready” and offers “Reset to calibration.” Participants could interpret those labels as individually estimated slider preferences; the present questionnaires do not collect such preferences.

### How earlier responses affect later captions

**Evidence.** New handoffs freeze Phase 1 ratings, wording, scales, provenance, and observations into a profile. Version `phase1-caption-guidance/v1` uses credibility, distraction, restfulness, same-clip ART differences, and overall PRSS as writing guidance. Explicit controls override that inferred guidance. This is prompt conditioning, not participant-specific model-weight training. Custom wording is retained without assigning built-in meanings solely from reused item IDs. [Source: `PERSONALIZATION.md`.]

**Interpretation.** Phase 2 ratings evaluate an adaptive system whose captions already depend on Phase 1 responses. A relationship between early and late ratings is not independent validation that the inferred preferences were correct.

## 3. Response inventory and participant burden

Counts below assume the packaged items, two short clips, and the current integrated protocol. Customized instruments require a new inventory from their saved snapshots.

| Collection point | Responses per participant | Scale/type | Analysis unit |
| --- | ---: | --- | --- |
| Original short-clip ART items | 8 × 2 = 16 | Configured Phase 1 scale; currently 1–7 | Participant × clip × item |
| Updated short-clip ART items | 8 × 2 = 16 | Same configured scale | Participant × clip × item |
| Updated-caption questions | 6 × 2 = 12 | **Also 1–7 in the inspected Phase 1 configuration** | Participant × clip × item |
| Short-chapter PRSS | 8 | Configured 1–7 | Participant × chapter × item |
| Visual observations | Variable; at least one mark per clip | Frame/time, coordinates, matched labels, unmatched marks | Participant × clip × frame/mark |
| Sound observations | 5 × 2 = 10 | Explicit heard/not-heard | Participant × clip × family |
| Long-video caption questions | 4 × 3 = 12 | 1–5 agreement | Participant × video × item |
| Long-chapter PRSS | 8 | Imported Phase 1 scale; currently 1–7 | Participant × chapter × item |
| Overall controls and effort | 3 | 1–5; the two control-success items allow N/A | Participant × item |
| Perceived update speed | 1 | Five categorical choices, including no update noticed and no control use | Participant |
| Final comment | 1 optional | Free text, up to 2,000 characters | Participant |
| Interaction and delivery records | Variable | Settings, playback, exposure, job and event records | Participant × video × event/cue |

There are **nine questionnaire submissions**, comprising 75 rating slots, one timing choice, and an optional comment; two rating slots may contain N/A. The ten sound-family judgments and visual marks are additional. Required media playback totals approximately **15 minutes 40 seconds**, excluding reading, answering, generation waits, interruptions, and breaks. A 30–40-minute appointment is a **proposed planning allowance**, to be revised using observed human completion times.

The scale distinction is essential: `app.py:589` serves one configured scale for all Phase 1 blocks, whereas `study_instrument.py:68` defaults Phase 2 caption/control items to five points and `study_instrument.py:82` imports the PRSS scale. Do not infer scale length from a question's label or normalize seven- and five-point items into an interchangeable construct.

## 4. Proposed user-test collection procedure

### Preparation and recruitment

**Proposed.** First run a formative round of 8–12 participants, including both supported languages if the planned study is bilingual. This is a practical starting round for finding comprehension and workflow problems, not a powered efficacy sample. Cover the intended range of caption familiarity, language proficiency, and input/device needs. Define the target population explicitly; the repository does not establish it.

Before that round, record the study's consent/recruitment procedure, participant eligibility, and handling of withdrawals. These procedures are not established by the inspected interface. Use pseudonymous study IDs; keep any contact/consent linkage separately. Record language, relevant hearing/vision and caption-use background where appropriate to the research question, device/browser, headphones, and assistance provided in a separate researcher form. These are **proposed extra fields**, not existing automated measurements.

Freeze the protocol version, item wording/translations and anchors, media/cue hashes, model/checkpoint configuration, and personalization policy. Review the draft instrument with the intended language groups. Rehearse the actual devices and connection conditions before relying on earlier Chromium automation. Determine the later main-study sample size from a specified primary outcome and desired precision, accounting for repeated observations within people. A feasibility sample should be justified by its feasibility objectives; randomized-pilot guidance provides a useful methodological analogy, although this interface is not a randomized trial. [Method reference: CONSORT pilot/feasibility extension](https://www.bmj.com/content/355/bmj.i5239).

### Session procedure

1. **Orientation.** Establish comfortable headphone volume, preferred language, and how to obtain assistance. Explain both control dimensions and selected-versus-applied feedback using a standardized script. Avoid suggesting that more detail or regenerated captions should be better.
2. **Chapter 1.** Let the participant follow the assigned sequence. Observe instruction rereading, marking difficulties, unanswered-item navigation, and recovery problems. Do not coach the answers.
3. **Chapter 2.** Ask the participant to adjust captions when useful, allowing non-use as a meaningful outcome. Do not require a fixed number of adjustments in the main observational protocol. Log researcher assistance and interruptions separately.
4. **Questionnaires and receipt.** Allow the participant to submit the per-video and final answers without interpreting their answers aloud. Verify that the final receipt appears and the backend records completion.
5. **Retrospective debrief.** Ask what each control changed, whether updates were noticed, whether captions matched audible events, what was distracting, and whether the chapter-wide questions were clear. Record comprehension errors and concrete examples alongside the optional in-app comment.

Use retrospective discussion instead of concurrent think-aloud during the measured videos: speaking and problem-solving would change the listening/attention task. If a separate usability session requires forced control tasks or think-aloud, label it as a different protocol and do not pool its experience ratings with natural-use sessions.

### Collection safeguards and daily checks

**Evidence.** Phase 1 saves responses/viewings/events as JSONL with participant snapshots. Phase 2 uses SQLite for authoritative session state, receipts, events, jobs, cache and exposures. Submission revisions and stable request/exposure identities protect against duplicate writes; drafts remain distinct from submitted answers. The long-video completion gate requires near-full played coverage and a position near the end; seeking straight to the end is insufficient. Display telemetry does not establish gaze or comprehension. [Sources: `log.py:97`, `study_store.py:25`, `study_store.py:130`, `study_api.py:478`, `STUDY.md`.]

**Proposed.** After each session, reconcile the expected five short-chapter and four long-chapter questionnaire submissions, two sets of observations, four short viewings, three video completions, and final receipt. Verify item/profile/media hashes and inspect missing exposure windows, fallbacks, interruptions, and any researcher intervention. Use protocol-specific expected counts for historical records. Preserve partial sessions and the stage where they ended; do not convert an absent answer into “not heard,” N/A, zero, or a midpoint.

## 5. Dataset preparation and analysis plan

### A. Freeze and reconcile the analysis cohort

**Proposed.** Take a dated, consistent snapshot of the collection, retaining raw files and database provenance. Analyze derived copies. Define the cohort by study/configuration, protocol, instrument, language, and personalization policy. Exclude QA/synthetic sessions using both the dashboard flag and the recruitment/session ledger; a naming convention alone does not establish research participation.

Use the dashboard for monitoring and case inspection. Export its normalized response CSV and sanitized participant JSON for analysis. The CSV covers the whole cohort under the selected QA include/exclude setting, **not just the currently visible search/item filters**; QA is excluded by default. Reapply documented cohort filters in the analysis. JSON/raw records are also needed for observations, playback and exposure measures; the answer CSV is not a complete behavioral dataset. [Sources: `DASHBOARD.md`, `DASHBOARD_API.md`.]

Maintain linked tables for participants, questionnaire submissions/items, visual/sound observations, playback/completions, control events, and caption exposures/jobs. Key response rows by study/session, phase, flow version, clip/video, viewing/condition, instrument identity, and item. Join the two phases using the stored calibration handoff identity, rather than approximate timestamps or labels alone. Deduplicate imported Phase 1 records against the original submission identity.

Retain each Phase 2 session's frozen instrument with its per-video answers: those rows contain instrument/item hashes rather than a self-contained wording snapshot. The dashboard derives Phase 2 submission timestamps from mutation events; survey state alone does not supply a submission time. Missing events therefore limit timing analyses even when answers survive. Do not substitute a job timestamp or export time for an unavailable survey timestamp. [Sources: `study_api.py:531`, `dashboard_data.py:549`.]

For every output, disclose the number of participants as well as clips/videos, eligible observations, valid answers, missing answers, and N/A values. Treat warnings about unreadable/truncated sources as an unresolved data-quality condition, not proof of zero responses.

### B. Primary formative outcomes: can people complete and understand the study?

Report completion and dropout by stage, task/session duration, assistance frequency, failed/retried submissions, and recovery success. Define completion rate as eligible enrolled participants reaching the final receipt divided by eligible enrolled participants who started. Report a separate rate for completed Phase 1.

From moderator notes, tabulate misunderstandings of the two axes, selected/applied status, fallback status, scale anchors, and clip-versus-chapter reference periods. Rank issues by whether they prevent completion, compromise the meaning of a response, or cause recoverable inconvenience. Record which language/device conditions produced each issue. Report observed counts and uncertainty, without treating software test passes as human usability results.

### C. Short-clip experience: paired descriptive changes

For each participant, clip, and ART item, join the original and updated answers and compute `updated − original`. Show the paired distribution and proportions increasing, unchanged, or decreasing. Summarize each clip separately, then use equal participant weighting for any combined estimate. Repeated items and clips from one person are not additional independent participants.

Treat the four two-item ART domains as exploratory content groupings. Report item-level results first; form domain/composite scores only under a documented scoring rule supported by the selected instrument. For reverse-coded items on an L-point scale, use `L + 1 − response` only where the saved definition specifies reversal. Preserve the raw values. The negative credibility and distraction items require that care; higher raw effort in Phase 2 means more burden.

Use participant-level resampling for uncertainty intervals around pooled descriptive paired changes when the sample supports it. For a sufficiently sized later study, an ordinal mixed model could represent response category as a function of viewing condition and clip, with participant effects and appropriate item handling. With only two clips, treat clip as a fixed stimulus factor; do not claim generalization to all scenes. Cumulative-link mixed models are designed for ordered responses with grouped effects. [Method reference: official `ordinal` documentation](https://stat.ethz.ch/CRAN/web/packages/ordinal/refman/ordinal.html).

The estimate remains an **original-to-updated sequence contrast**. Repetition, familiarity, elapsed time, and the intervening observation tasks are inseparable from regeneration here. Alternating clip order does not remove those confounds. Caption credibility/restfulness ratings describe only the updated condition because the corresponding baseline items are absent.

### D. Visual and sound observations

Summarize marked object categories, marks per frame, spatial distributions, and the proportion unmatched by available masks. Label these explicit selections, not eye tracking or objective attention. An unmatched point is retained evidence, not automatically participant error.

For each sound family, compare the reported heard/not-heard answer with the authored presence annotation. Report hits, misses, false alarms, and correct rejections with their separate eligible denominators. Review annotation coverage before interpreting disagreement: “present” means annotated in the material, not guaranteed perceptibility for every participant. With only five families per clip, avoid unstable individual sensitivity scores. These observations were collected after the prepared viewing, so they cannot measure a before/after improvement in detection. [Sources: `app.py:882`, `config.py`, `points.py`.]

### E. Long-video experience and control use

Report all four five-point caption items per video and language, alongside the overall control-success, effort, and timing distributions. Keep “did not use the controls,” “did not notice an update,” N/A, and missingness separate. Non-use alone is not evidence of satisfaction.

Derive behavioral summaries per participant/video: whether each axis was used, number of acknowledged setting changes, range of selected values, final setting, presets/resets, and time at settings where timestamps and playback state support reconstruction. Count server-acknowledged changes rather than raw pointer events. Treat backward seeks and replays explicitly; do not assume playback position increases monotonically or infer dwell time across missing observations.

Separate three quantities: **requested settings**, **settings attached to displayed captions**, and **generated jobs that may never be displayed**. Report generated/fallback exposure proportions by count and, where complete intervals permit, displayed duration. Identify incomplete exposure records and unknown-duration windows. Do not treat model generation time as perceived response latency: caption-boundary waiting, queueing, delivery, and visibility are different components.

Inspect comments or low ratings alongside the actual captions, control timeline, and technical conditions for that video. Analyze setting–rating associations as exploratory, because participants self-select settings and may change them in response to difficult scenes or poor captions. The three long videos also appear in fixed order, so scene and order/fatigue effects cannot be separated. Keep the all-session description primary; any low-fallback subset is a sensitivity analysis with an explicit threshold and exclusion count, not a way to discard unfavorable experiences.

### F. Chapter-wide PRSS and qualitative explanation

Report PRSS item distributions separately for the short and long chapters. Even with identical items, these reference different stimuli, exposure lengths, caption systems, and stages. A chapter difference is descriptive; it is neither a repeated measure for each long video nor an isolated personalization effect. Do not copy one chapter score into three video rows and count it three times.

Code the final comments and proposed debrief notes for recurring concerns: grounding/accuracy, excess or insufficient detail, reading distraction, control comprehension, update timing, and perceived fit. Allow new themes and contradictory cases. Keep participants as the counting unit, preserve original-language text, and have bilingual interpretation checked before quoting translated material. Relate themes to that participant's exposure evidence without presenting temporal proximity as causation.

### G. Missingness, versions, and inferential limits

Include partial sessions in the collection-flow report and use complete pairs only for each paired contrast. Report who is missing from each analysis and why. Do not automatically impute forced-form omissions; a completed form does not imply a completed study. Keep unversioned legacy, different-clip, standalone preference-based, and current integrated protocols separate. Check exact wording/anchors before combining any instrument variants or languages.

Predefine one primary endpoint for a later hypothesis-testing study and any multiple-comparison correction for secondary families. This formative round should prioritize comprehension, feasibility, distributions, and uncertainty. The dashboard does not implement this statistical plan, psychometric validation, qualitative coding, or sample-size justification.

## 6. Verification and present collection status

**Fresh evidence from this review:**

- **42 Python tests passed:** survey hierarchy (5), instrument records (3), personalization (20), dashboard normalization (10), and playback timing (4). The run produced one existing Starlette/httpx deprecation warning.
- The existing Chromium hierarchy harness completed both English and Korean journeys, including three per-video surveys, final `done`, and the legacy progress-rail check. Evidence: `/tmp/regen-report-review-retry-20260914/browser-survey-hierarchy-result.json`. This harness simulates media completion and seeds playback coverage in an isolated QA server; it verifies forms, APIs, drafts, instruments, exports and localization, not live listening or model quality.
- The first browser attempt outside the sandbox failed while capturing a screenshot; one unchanged retry passed. No application fix was made. The initial restricted runs also encountered local-network/SQLite access limitations; successful verification used the required environment access.
- Existing screenshots of the per-video form, steering overlay, and linked researcher inspector were visually inspected. They show distinct survey scope, labeled controls, and selected/applied feedback; this is not a new physical-device accessibility study.
- A read-only dashboard aggregation at **2026-09-14 04:56:20 UTC** returned **9 non-QA participant records, 2 submitted questionnaires, 16 answer values, and no completed Phase 1 or started/completed Phase 2 sessions**, with no reader warnings. These are stored-record counts, not independently verified recruited-person counts. They do not support an outcome analysis yet.

**Historical evidence, not rerun here:** the 12 September real-media/GPU EN/KO journeys each completed four short viewings, five short-chapter surveys, three long videos and per-video surveys, and the final questionnaire. Of 362 caption exposures, 349 (96.4%) displayed generated captions. Independent audits reconciled records. The 14 September personalization checks and steering/dashboard QA cover subsequent bounded changes. These checks do not establish human caption quality, language equivalence, concurrent participant capacity, or current end-to-end GPU health. See [FRESH_GPU_QA.md](FRESH_GPU_QA.md), [PERSONALIZATION.md](PERSONALIZATION.md), [STEERING_METER_QA.md](STEERING_METER_QA.md), and [DASHBOARD.md](DASHBOARD.md).

## 7. Expected study report outputs

**Proposed.** Produce a participant-flow table with stage-specific exclusions; a versioned instrument/data dictionary; paired short-clip plots; per-video caption and chapter PRSS distributions; control and fallback summaries; and a qualitative issue table linked to anonymized examples. Include a reproducible cohort/analysis script and a record of all scoring and exclusion decisions when analysis is implemented.

Before interpreting results beyond a formative study, resolve the target population and sample-size rationale, review item wording and Korean translations, specify the primary outcome, and verify comprehension and caption grounding with people. A claim that personalization itself improves experience would require a separately designed comparator or randomized manipulation; the prepared interface alone does not provide that comparison.

This review adds only `USER_TEST_REPORT.md`. Application code and collection records were not edited. The report makes no code simplifications; its main clarification is to separate what the interface collects, what the data can support, and what the proposed study still needs to establish.
