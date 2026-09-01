# Caption session design system

Read this before building anything in this project. It governs every frame, component, state, and export. The attached spec, spec-behavior.md, defines what to build. This file defines how it looks, reads, and moves. Use no external UI kit, icon set, or default component styling.

## Principle

The interface is typeset the way a caption track is typeset, and it reads as a calm, precise research instrument, never as a dashboard or a consumer app. Two registers share one typeface. The structure the participant works is in the SDH caption register, bracketed and uppercase. The sentence the model writes is sentence prose. Two materials share one screen. The clip and its caption are one dark object. The working surface and everything else sit on a light ground.

## Typeface

One typeface throughout, chosen for caption legibility at small sizes on a tablet, with unambiguous uppercase and a regular weight that reads well in sentence case. Choose it within that rule and use nothing else. Table 1 gives the scale.

**Table 1. Type scale**

| Element | Size | Treatment |
|---|---|---|
| token | 13 | bracketed, uppercase, lightly tracked |
| role head | 13 | as token, with members beneath at the same size in dim |
| caption prose | 22 on a 30 line | sentence case, regular, centered, two lines reserved |
| eyebrow | 10 | uppercase, letterspaced, dim |
| button | 12 | uppercase, sentence case in source |
| card and measure body | 14 on a 1.7 line | same face |
| card and measure heading | 20 | same face |

## Palette

Table 2 gives the seven tokens. There are no others.

**Table 2. Tokens**

| Token | Hex | Role |
|---|---|---|
| screen | #111311 | stage surround, parked strip, placed caption box at 80 percent |
| caption | #F2F2EC | caption text in both positions |
| bench | #E9EAE4 | working ground, cards, timeline, listing, measures, follow-up |
| ink | #1A1C1B | text, selected ordering column, filled lane segments, answered boxes |
| dim | #6F726E | eyebrows, group members, stale caption, struck rows |
| rule | #C3C6C0 | hairlines, unselected columns, empty lane segments, empty boxes |
| mark | #EAC93C | primary action, the lower-lane segment being shaped or revised |

mark is caption yellow, the color broadcast and teletext subtitling reserve for emphasis. It appears in exactly two places, the primary action and the lower-lane segment currently being shaped or revised, and nowhere else. Nothing meaningful is carried by color alone. Table 3 gives the cue beyond color for every state that has one.

**Table 3. States and their cues beyond color**

| State | Cue beyond color |
|---|---|
| struck row | struck through, dimmed, and moved beneath the fold handle under a hairline |
| selected ordering column | filled, and the skeleton rows align with it |
| stale caption | dimmed and labeled Written before your last change, with the moved rows marked in the skeleton |
| answered scale box | filled |

## Surfaces and spacing

Flat surfaces. Zero border radius. 1px hairlines in rule. No shadows, no gradients, no icons. Spacing on a 4 and 8 scale. The kiosk runs at 1180 by 820 landscape, tolerant of desktop widths, with the stage around 720 wide, the strip beneath it, and the working column taking the remaining width behind a 24 gutter. The follow-up is phone first, single column, tolerant of desktop widths.

## Materials

Table 4 gives the two materials.

**Table 4. Materials**

| Material | Ground | Text | Where |
|---|---|---|---|
| dark object | screen, and the placed box at 80 percent of screen | caption | stage surround, parked strip beneath the frame, placed caption box in the lower third of the frame |
| bench | bench | ink, with dim for secondary | working column, cards, timeline, listing, measures, follow-up |

The caption is one object in two positions, parked in the strip while shaped and placed on the frame while watched. Face, size, two-line box, and ground are identical in both. Only the vertical position differs.

## Register

Table 5 gives every kind of text and its form.

**Table 5. Register**

| Kind | Form | Example |
|---|---|---|
| token | bracketed, uppercase, lightly tracked | [DISTANT SIREN], [PARKED VAN] |
| listed sound | the same token form, in the participant's own words | a placeholder such as [BUS BRAKES] |
| role head | as token | Underneath, Stands out, Only here, uppercase by treatment |
| eyebrow | uppercase, letterspaced, dim | If only this were mentioned |
| caption prose | sentence case, centered, two lines reserved | a one- or two-line placeholder describing a street scene's sounds |
| button | sentence case in source, uppercase by treatment | Keep this caption |

Every quantity the system measures, visibility, temporal character, and role, appears as a phrase on a row and never as a number or a control. No numeral appears on the skeleton, its ordering columns, or its fold.

## Actions

One filled primary action per screen, in mark. Secondary actions are outlined, 1px. A busy action shows its busy string on the button itself, Writing… or Preparing comparison…, never a global spinner. Every action keeps one name through the whole flow. Errors state what happened and the way forward, in the interface's voice, without apology.

## Motion

Short linear motion when the caption moves between strip and frame on Keep and on Revise, and when rows move because the list reordered or folded. Under reduced motion every move cuts, and the rows that moved are marked with the same marking the stale state uses.

## Accessibility

Focus is visible on every interactive element. Rows are at least 44 tall, ordering columns at least 40 wide, and scale boxes at least 44 square. Contrast holds at the small uppercase sizes, which are the risk point of this identity. The caption on the dark ground is checked in both positions separately from the tokens on the light bench.

## Copy

All strings come from Table 6 of the spec, verbatim. They are English placeholders. Korean will replace them, so every text container holds both lengths.

## Never

No icons. No console and no detached widgets. No numeral on the skeleton, its columns, or its fold. No measured value shown as a number. No reroll. No history of read captions. No overlays, mask colors, or scene annotations on the stage. No shadows, gradients, or rounded corners. No screen, control, or state beyond the spec.
