# Scene-Adaptive Audio Caption Curation

## UI and UX description, prepared as a design brief

---

## 1. Brief

Design the participant-facing session view of a research instrument. A
participant watches a short street-scene video clip, then shapes its
AI-generated audio captions using three controls, then watches the clip again
with their captions running, and finally answers two short checks. The
captions they settle on are research data. The interface must feel like a
calm, precise instrument, never like a dashboard or a consumer app.

Scope covers the seven session screens listed in Table 2, designed for a
tablet in landscape as a kiosk, around 1180 by 820, tolerant of desktop
widths. Two further views exist, a researcher bench and an admin calibration
view. Sections 8 and 9 describe them as context only. Do not design them.

The system behind the interface, in one paragraph. Preprocessing turns the
clip into shots, each with a list of sound sources. Each source carries a
visibility level and a temporal note, both computed, never shown as numbers.
The participant's three controls decide which sources a caption mentions, how
the balance between what is seen and what is heard ranks them, and how
detailed the caption is. Prose is generated on request from those settings,
identical settings always return the identical caption, and the commit button
stays disabled until the participant has seen the caption their current
settings produce.

## 2. Governing principle

No detached settings panel exists. Every control is embedded in a
representation of its own consequence, and affordance follows the response
shape of the parameter. A control whose effect is discrete gets a discrete
surface, a quantity that is measured rather than set gets no control at all,
and anything that looks live must respond.

## 3. Session screens

### 3.1 Layout

```
[ phase rail: 1 watch · 2 author · 3 watch again · 4 check ]
[ stage: 16:9 video frame                                  ]
[ progress bar, full-clip position                         ]
[ caption band, display type                               ]
[ console: sources / balance / detail                      ]
[ skeleton: ordered source list                            ]
[ Show caption · Keep this caption                         ]
```

The caption band is the hero. The object of preference is prose, so prose
gets the display treatment, set in the display face at reading size with
reserved height so the layout never jumps between generations. The stage sits
above it and stays neutral, showing only the footage, a small shot label, and
the progress bar. No overlays, mask colors, or scene annotations appear,
since scene structure must not cue responses.

The phase rail is numbered because the trial genuinely is a sequence, and the
numbers tell the participant where they are.

### 3.2 Screens and states

Table 2 lists every screen with the states each must be designed in.

**Table 2. Screens and required states**

| Screen | Key elements | States to design |
|---|---|---|
| intro | instruction card, Begin | default |
| 1 watch | stage, raw captions | playing, paused, ended |
| author intro | instruction card, Start | default |
| 2 author | looping shot, console, skeleton | no caption yet, writing, caption fresh, caption stale, atmospheric grayed |
| 3 watch again | stage with kept captions | playing, paused, revising, finish disabled until one full playthrough |
| check | two caption cards per shot | preparing comparison, ready |
| probe and done | free-text field, summary, log download | empty, filled, complete |

During authoring the current shot loops so the moment stays present while its
caption is shaped. During the second viewing, Revise pauses playback and
opens the console for the current shot, and the revised commit replaces the
kept caption. In the check, the participant's caption and a default-policy
caption render as two equal cards with side assignment randomized per shot.

## 4. Components

### 4.1 Sources

Sources render as a token strip in the mono face. Mute is latching and shown
as a strike-through, so an excluded source remains legible rather than
vanishing. Solo is hold-to-activate and releases on pointer up, a momentary
inspection gesture that cannot become a resting state. While held, the token
takes the sodium accent and other skeleton rows dim.

### 4.2 Balance

Balance renders as a segmented crossfader with EYE and EAR poles set as small
mono text, no numerals anywhere. Segments have varying widths per shot,
clicking a segment selects it, and the selected segment fills with ink. A
continuous slider is wrong here because the underlying effect is discrete,
and dead spans where dragging changes nothing read as breakage.

### 4.3 Detail

Detail renders as four equal detents labeled itemized, grouped, scene-level,
and atmospheric. At atmospheric the token strip and crossfader visibly gray
and stop responding, because nothing is named at that step and controls that
look live but do nothing corrode trust in the instrument.

### 4.4 Skeleton

Below the controls, the admitted sources render as a numbered list in their
current order, each row carrying its visibility phrase and temporal phrase as
a right-aligned eyebrow. The skeleton is the zero-latency feedback surface.
Every gesture reorders, re-bands, or re-admits it instantly, while prose
waits for an explicit request.

### 4.5 Actions

Show caption requests prose for the current settings and reads as the
secondary, outlined action. Keep this caption is the single filled sodium
action, disabled whenever the displayed prose is stale relative to current
settings, so a participant cannot keep a caption they have not read. The
criterion sentence sits under the buttons. No reroll affordance exists
anywhere.

## 5. Copy deck

Table 3 gives the exact strings. Buttons are written in sentence case and
styled uppercase by the type treatment. All copy is an English placeholder
pending the Korean construct check, and the criterion sentence in particular
is a construct decision, not a localization task.

**Table 3. Copy**

| Element | String |
|---|---|
| app title | Caption session |
| phase rail | 1 watch · 2 author · 3 watch again · 4 check |
| criterion sentence | Is this the caption you'd want for this moment? |
| primary actions | Begin · Show caption · Keep this caption · Revise this caption · Finish viewing · Submit · Download session log |
| author helpers | Replay shot · Pause loop · Adjust, then show the caption |
| busy states | Writing… · Preparing comparison… |
| check heading | Which caption would you rather have? |
| probe question | Across the clip, what, if anything, would you still change about the captions that the controls didn't let you change? |
| viewing eyebrows | First viewing · automatic captions / Second viewing · your captions · revise any shot as it plays |
| error | Caption request failed. Check the connection and try again. |

Errors state what happened and the way forward, in the interface's voice,
without apology. Every action keeps one name through the whole flow.

## 6. Visual identity

The identity is an instrument for street-level material. Table 4 gives the
palette.

**Table 4. Palette**

| Token | Hex | Role |
|---|---|---|
| ground | #DFE2DE | app background, concrete |
| panel | #F1F2EF | surfaces, cards |
| ink | #191D1C | text, active states |
| dim | #6E7674 | secondary text |
| rule | #BFC5C0 | hairlines, borders |
| sodium | #B96F14 | primary action, streetlight amber |
| teal | #1F6663 | audio accent, bench only |

Type pairs Archivo for display with IBM Plex Mono for labels, tokens, and
eyebrows, and IBM Plex Sans for body text. The caption band is Archivo 600 at
19 to 22, headings run 20 to 26, body is 14 on a 1.7 line, tokens and buttons
are mono at 11 to 12 with tracked uppercase, and eyebrows are mono 10,
letter-spaced, uppercase, in dim.

Surfaces are flat. Zero border radius, 1px hairline borders in rule, no
shadows, no gradients, no icons, and nothing meaningful carried by color
alone. Spacing works on a 4 and 8 scale with a content column around 760.
Sodium appears only on the primary action and the momentary solo state.

## 7. Behavior notes

The deterministic layer, everything up to the skeleton, responds instantly to
every gesture. Prose generation is slow and explicit, always behind Show
caption, with the writing state on the button itself rather than a global
spinner. Identical settings return the identical cached caption, and repeated
generation of the same settings is free.

The session runs as a kiosk, fullscreen with navigation suppressed, every
interaction autosaved to an event log, resuming after a crash, gated by a
participant identifier. The log itself has no participant-facing UI beyond
the single download action on the final screen.

Focus is visible on every interactive element, reduced motion is respected,
touch targets meet tablet minimums, and contrast holds at the small mono
sizes, which are the risk point of this identity.

## 8. Context only, bench view

The researcher's bench adds what the session hides. A residual field scatters
every source by visibility against salience with mute, solo, and lasso acting
directly on it. The balance control gains drag-to-solve on the skeleton. A
contact sheet reads output, shots as columns, runs as rows, every cell a
caption stamped with its settings and config hash, hover-diff between cells,
repeats stacked. A live monitor mirrors a session read-only. None of this is
in scope.

## 9. Context only, admin view

Calibration surfaces are protocols rather than knobs. A saturation curve with
sources as dots sets the visibility constant by dragging the knee or by
clicking a source and declaring it half visible. Shot splitting renders a
distance signal under a filmstrip with a draggable threshold line and
exclusion shading, and committing freezes boundaries into hand-draggable
handles. Study configuration collects trial order, start-state scheme,
condition assignment, and criterion wording, then versions and hashes the
result. In the bench, frozen values are visible but inert, grayed with a
fixed-for-this-study note. The session view never shows them at all. None of
this is in scope.
