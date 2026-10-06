# Phase 1 observational analysis

This analysis describes changes in selected visual mask area and attended sound
categories. Masks and sound families are **observational codes, not correct or
incorrect answers**. First- and second-view sound menus may differ. The outputs
are descriptive records; they do not measure gaze, attention intensity, audibility
accuracy, preference change or the causal effect of captions.

Use the **Phase 1 trends** dashboard view, its Observation CSV and Paired CSV.
The selected participant search, language, progress, QA and short-clip filters
apply to every analysis table and both exports. See [the dashboard guide](DASHBOARD.md)
and [HTTP contract](DASHBOARD_API.md) for access and endpoint details.

## Source and analysis units

The pure Python entry points in `phase1_analysis.py` are:

```python
codebook_for_clip(clipconfig: dict) -> dict
analyze_sessions(sessions: list[dict], *, clip_id: str | None = None) -> dict
```

Each session input contains only public `session_id`, `participant`,
`sheet_responses`, frozen `sheet_config` and `sheet_clip_order`. A public session
ID is a join key, not its bearer credential. Never pass tokens or cookies.
The analyzer performs no file reads, database writes, inference or GPU work.

An observation is one `session_id × clip_index × round`:

| Round | Visual response | Sound response | Viewing |
| --- | --- | --- | --- |
| 1 | P2 / V1 | P3 / A1s | first / P1 |
| 2 | P7 / V2 | P8 / A2s | second / P6 |

A pair joins the two rounds of the same `session_id × clip_index`, checking
`clip_id`, instrument and other grouping metadata. The rounds intentionally have
different `view_id` values. Incomplete observations and pairs remain visible;
availability is assessed separately for each derived metric. Practice records
with `analysis_excluded=true` are excluded.

Exact response mirrors are deduplicated. Conflicting duplicate pages or identity
fields produce issues and unavailable derived values. Repeated session payloads
with different frozen configurations or session identities are excluded as a
whole with an explicit warning; the analyzer never arbitrarily chooses one.

## Visual mask area

For a submitted point, let `H_ir` contain the masks hit in its selected frame,
where `i` identifies the pair and `r` the round. The existing selection rule is
unchanged: choose the hit mask with the smallest **whole original binary mask**
foreground area; ties use document order. Binarization is `pixel > 127`.

```text
j*       = argmin over j in H_ir of foreground_pixels(M_irj)
A_ir     = foreground_pixels(M_irj*) / (frame_width_ir × frame_height_ir)
delta_pp = 100 × (A_i2 - A_i1)
```

`selected_mask_area_ratio` stores `A` as a fraction from 0 to 1. For example,
100 foreground pixels in a 1,000-pixel frame gives `A=0.1` or 10%; a change from
0.1 to 0.25 gives **+15 percentage points**, not a 15% relative increase.

The numerator includes every foreground pixel of the selected mask. If one
mask represents a class union, the numerator is that union's area. It is not a
connected-object area or a winner-map area after subtracting other masks.
`candidate_masks` retains every hit mask separately. Their areas may overlap;
their ratios must not be summed or expected to total 100%.

No hit gives `A=NA`, not zero. A frame/mask resolution mismatch preserves the
recorded selection but makes the primary area ratio unavailable. The analyzer
also rejects missing or internally inconsistent saved area measurements. If
either round's area is unavailable, `area_delta_pp` is `NA`.

The observation retains frame/mask dimensions and SHA-256 hashes, selected ID
and label, `matching_rule="smallest-containing-mask-v1"`, threshold, candidate
measurements and `visual_status`. `frame_changed` compares the selected frame
IDs when both are known. Examine it alongside the two saved frame hashes: an
area difference between frames also reflects scene composition. A stable class
ID such as `building` is not evidence of a tracked physical building.

## Sound type, family and menu changes

Let `raw_ir` be the exact selected option code, `h_v` the declared canonical
type mapping and `g_v` the family mapping in frozen codebook version `v`:

```text
S_ir        = h_v(raw_ir)                 # canonical sound type
G_ir        = g_v(S_ir)                  # observational family
D_type_i    = 1[S_i2 != S_i1]
D_family_i  = 1[G_i2 != G_i1]
```

When no `sound_types` map is declared, `h_v` is identity. An explicitly supplied
partial map does not guess the meaning of missing codes. Missing type or family
values make the corresponding change indicator `NA`, including when both are
missing. No name similarity is used to merge codes. Original sound code,
displayed label, option menu and order remain available alongside derived codes.

For example, `engine → car_passing_by` yields `D_type=1`; if both map to
`transport`, it yields `D_family=0`. These are different questions.

`menu_changed` compares sets of original `(option code, displayed label)` pairs.
`order_changed` compares ordered original option codes **only when the code sets
are identical**. A wording-only change can therefore give `menu_changed=true`
and `order_changed=false`. Different code sets give `order_changed=NA`, while
type/family changes can still be calculated for valid selected codes.

## Declaring an analysis codebook

The following is an illustrative fragment under the frozen questionnaire
configuration. Its categories and relationship codes are research definitions,
not independently established facts about the media:

```json
{
  "clips": {
    "A": {
      "analysis_codebook": {
        "version": "observational-codes-2026-10-v1",
        "sound_types": {
          "engine": "engine",
          "car_passing_by": "car_passing_by",
          "speech": "speech"
        },
        "sound_families": {
          "engine": "transport",
          "car_passing_by": "transport",
          "speech": "human"
        },
        "av_relations": {
          "road_transport": {"engine": 1, "car_passing_by": 1, "speech": 0}
        }
      }
    }
  }
}
```

`sound_types` maps raw option codes to canonical IDs. `sound_families` maps
canonical IDs to arbitrary declared family IDs; there is no acoustic-correctness
allowlist. `av_relations` maps selected visual **mask IDs**, not display labels,
to canonical sound IDs and integer `0`/`1` codes. Undeclared combinations stay
unknown; omission does not imply zero.

If `sound_families` is absent, the helper uses existing `source_families` from
the clip configuration. Those legacy keys are raw option IDs and are converted
through `sound_types` when supplied. Conflicting family assignments for aliases
are omitted with an issue. Malformed maps produce issues and unavailable codes,
not a handoff crash. An absent version uses the descriptive fallback
`frozen-config-v1`; a nonempty explicit codebook without a version is also warned.
Use an explicit version for authored analysis definitions.

The helper returns `version`, `hash`, `sound_types`, `identity_sound_types`,
`sound_families`, `av_relations` and `issues`. The digest of the normalized
codebook distinguishes different mappings, even when a version label is reused.
The analysis records `codebook_version` and `codebook_hash`; it does not rewrite
a participant's frozen source configuration.

## Denominators and aggregation

Summaries are separated by clip, instrument, codebook version/hash, assigned AV
condition, description depth, caption strategy and `menu_changed`. No aggregate
silently pools these groups. A participant can contribute several clip-level
pairs; a trial count is not a unique participant count.

For each change metric separately:

```text
valid_pairs  = number of pairs where that metric is defined
na_pairs     = total_pairs - valid_pairs
change_rate  = sum(D_i over valid pairs) / valid_pairs
mean_delta_pp = sum(delta_pp_i over valid pairs) / valid_pairs
```

Zero valid pairs gives `NA`. Type, family and area can have different valid
denominators. The returned summary objects include total, valid and NA counts;
change summaries also include `changed_pairs`, while area summaries include
`mean`. Incomplete pairing is not counted as unchanged.

For offered-choice-conditioned selection rates, let `O_ir` be the set of
canonical types actually offered in that round and `V_ir` mean that the selected
type is valid and belongs to the stored menu:

```text
eligible_r(s) = sum_i 1[V_ir and s in O_ir]
selected_r(s) = sum_i 1[V_ir and s in O_ir and S_ir = s]
P_r(s)       = selected_r(s) / eligible_r(s)
```

The family calculation uses the set of mapped offered families and requires a
mapped selected family. Two offered types in one family create one opportunity
for that family in the trial. Missing selected mappings are excluded from that
metric's eligible denominator, retained in observation issues and reported as
`unmapped_selections`. `unmapped_offered_sound_types` identifies offered canonical
types lacking a family map.

`selection_rates` returns `kind`, `round`, `code`, `numerator`, `denominator`,
`rate` and `unmapped_selections` for each group. Codes offered in either round
are represented in both rounds; a code never offered in one round has
denominator zero and rate `NA`. Different codes can have different denominators,
so these conditional percentages need not sum to 100%. Missing or invalid sound
answers are not treated as non-selections. Different menus remain a limit on
interpreting selected-code changes as preference changes.

## Assigned AV condition versus selected-code relationship

`av_assigned_condition` copies `stimulus_assignment.audiovisual_congruence`.
It is an experimental assignment, separate from `description_depth`,
`caption_strategy` and first/second viewing order. It is never inferred from
responses.

Only a declared crossmodal relationship table defines:

```text
C_ir = R_v(selected_mask_id_ir, S_ir)
relationship_rate_r = sum(C_ir where defined) / number of defined C_ir
```

`av_selection_relation` stores `C`: 1 means the declared table codes the pair as
corresponding; 0 means it codes the pair as non-corresponding. Missing table,
unclassified visual selection or undeclared combination gives `NA`. Equality of
visual and sound strings is never used to infer a relationship.

Each summary's `av_selection_relation_1` and `_2` includes
`total_observations`, `valid_observations`, `na_observations`, `relation_sum` and
`rate`. These describe available observation rows, not all potential viewers;
the pair summaries separately expose incomplete pairs. Zero defined relations
gives `rate=NA`.

`transitions` contains `kind="type"` and `kind="family"` rows with `from_code`,
`to_code` and `count` for valid pairs. It also contains
`kind="visual_sound"`, `round`, `from_code=selected_mask_id`, `to_code=sound_type`
and `count`. Those rows are **selection cross-counts**, not temporal transitions
or accuracy scores, and do not require a relationship table. Rows missing either
code do not enter that cross-count; they remain visible in observations.

## Output and historical records

The core result has `schema="dpo.phase1-analysis/v1"`, `observations`, `pairs`,
`summaries`, `selection_rates`, `transitions` and `warnings`. HTTP adds source
warnings, filter metadata, available clips and an update timestamp.
`OBSERVATION_FIELDS` and `PAIR_FIELDS` are the authoritative CSV column lists.
Observation rows retain one round's source selection and measurement provenance;
paired fields use `_1` and `_2` suffixes plus derived changes and issue status.
Output uses explicit field allowlists, excluding credentials, prompts and media
filesystem paths.

JSON uses `null` for `NA`; CSV uses an empty cell. Numeric zero, a defined
unchanged indicator and a declared relationship code 0 are retained as zero.
Nested arrays/objects are JSON-encoded in CSV, which uses UTF-8 BOM and the
dashboard's spreadsheet-formula neutralization.

Older spreadsheet-protocol rows without saved `visual_metrics` keep an
unambiguous stored raw match ID/label and selected frame where available, but
their area, dimensions and hashes stay unavailable. The analyzer never opens
present-day images or masks to backfill historical measurements. Records outside
the spreadsheet P2/P3/P7/P8 protocol are not converted into invented paired
observations. Original source data are not modified.

This document defines the implementation and interpretation contract. It does
not certify recruitment materials or report completion of any ongoing full QA.
