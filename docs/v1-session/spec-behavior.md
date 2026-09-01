# Scene-Adaptive Audio Caption Curation

Participant session and follow-up screens

## 1. What to build

Build the participant-facing screens of a research instrument. A participant watches a short street-scene video clip, with or without captions, lists the sounds they noticed, shapes the clip's AI-generated audio captions by working an ordered list of the sounds each caption could mention, watches the clip again with their captions placed on it, and answers a short set of items. On some clips the shaping step is replaced by a control task. On a later day, in a browser, the participant completes a short follow-up. Everything reads as a calm, precise instrument, never as a dashboard or a consumer app.

Deliver every screen in Table 4 in every state listed, every component in section 4 in every state described, and every screen in Table 9 in every state listed. Use the exact strings in Table 6 and the type, palette, and surface rules in section 6 throughout. Lay out the kiosk session for a tablet in landscape at 1180 by 820, tolerant of desktop widths. Lay out the follow-up phone first, single column, tolerant of desktop widths. Show every screen state as its own frame. Put every component in section 4 on one components frame with its states side by side, the crossfader state of the orderings included. Where the prototype allows it, wire the gestures in Table 5 so the states connect. Do not add screens, controls, icons, or states beyond the ones listed here. The researcher bench and the admin calibration view are out of scope. Section 9 names them only so nothing from them leaks in.

Table 1 lists the decisions that are fixed. Build to them.

**Table 1. Fixed decisions**

| Decision | Setting |
|---|---|
| surface | the skeleton is the only control surface, no console, no detached widgets |
| commit | Keep this caption commits the caption currently shown, no history of read captions |
| comparison | own against default caption is judged on the placed caption, toggled while the shot loops, in the follow-up |
| register | interface in the SDH caption register, generated caption in sentence prose |
| roles | the same participant authors and watches |
| caption position | parked beneath the frame during shaping, placed in the lower third of the frame during viewing |
| identity | Grouped level by soundscape role, place-anchored naming, neither is a control |
| reroll | none, anywhere |

### The system behind the screens

Preprocessing splits each clip into shots, and each shot carries a list of sound sources. Each source carries a visibility level and a temporal note, both computed, and a soundscape role, annotated. None of these is ever shown as a number. Three parameters decide which sources a caption mentions, how the balance between what is seen and what is heard ranks them, and how detailed the caption is. Prose is generated on request from those settings. Identical settings always return the identical caption. The commit button stays disabled until the participant has seen the caption their current settings produce. Configuration assigns, per clip, whether the first viewing carries captions, whether the clip is shaped or given the control task, and the settings the skeleton opens at.

## 2. Rules on every screen

There is one control surface, and it is the description itself. The skeleton, the ordered list of what the caption will mention, is both the representation of the caption's structure and the only thing the participant touches. Each parameter is a gesture on that list. Admission strikes a row. Balance chooses among the orderings drawn beside the list. Detail folds the list. Table 2 gives the three parameters with their response shapes and gestures. Build nothing that acts on the caption from anywhere else.

**Table 2. Parameters and their gestures**

| Parameter | Response shape | Gesture | Instant consequence |
|---|---|---|---|
| Admission | discrete, per source | tap a row | the row is struck and leaves the ranking, or is restored |
| Balance | discrete, a few orderings | tap an ordering column | the list adopts that ordering |
| Detail | four structural levels | drag the fold handle, or pinch | the list folds or unfolds |

Show every consequence as description. Nothing on the surface refers to what is heard as signal, only to what will be said. Quantities that are measured rather than set, visibility, temporal character, and role, appear as phrases on the rows and never as controls. Disable nothing on the skeleton. At every fold level the only gestures left are the ones with something to act on.

Keep prose separate from structure. Prose is requested explicitly and is the only slow thing in the interface. The caption is one object in two positions. While it is being shaped it is parked beneath the frame, where it is read at rest. Once kept it is placed on the frame, where it is watched. Same face, same size, same two-line box, same ground. Only the vertical position differs. The participant only ever watches the placed form.

Show nothing the tool knows before the participant has said what they noticed. The skeleton never appears, and is never reachable, until the listing step is submitted.

## 3. Session

### 3.1 Layout

```
[ clip 2 of 6 · timeline: shots as segments · automatic lane above · your lane below ]
[ stage: 16:9 frame, shot label, placed caption   ][ working column                  ]
[ parked caption strip, two reserved lines        ][   skeleton                      ]
[                                                 ][   ordering columns beside rows  ]
[                                                 ][   fold handle at the foot       ]
[ criterion sentence · Show caption · Keep this caption                              ]
```

Hold the stage at the same size and position in every step. Put the placed caption in the lower third of the frame in a box at about 80 percent opacity of the screen color. Set the parked strip flush beneath the frame, at frame width, on the same color at full opacity. Reserve the strip's space in every step and leave it empty while the caption is placed, so the only thing that ever moves is the caption. Keep the stage neutral otherwise. It shows the footage, a small shot label, and nothing else. No overlays, mask colors, or scene annotations, since scene structure must not cue responses.

The working column beside the stage carries whatever the step needs. In the viewings it holds the step's instruction line, and after a viewing ends it holds the single attention item. In shaping it holds the skeleton. In the listing and measures steps, widen it to a single column across the layout, since there is no live frame to keep beside it. Its position never changes when it is present.

The timeline belongs to the current clip. Run the shots left to right as segments. Fill the upper lane with the automatic captions during the first viewing when that clip carries them. Fill the lower lane shot by shot as captions are kept. The second viewing plays the lower lane, and Revise reopens a segment. Put a small eyebrow at the left that reads the clip count. The timeline is a display, not a navigation. Segments are not tappable.

### 3.2 Steps per clip

Table 3 gives the fixed order of steps within a clip, what configuration assigns at each, and what each captures.

**Table 3. Steps per clip**

| Step | Screen | Assigned by configuration | Captured |
|---|---|---|---|
| first viewing | 1 watch | captions on or off | attention item |
| listing | list | none | the sounds the participant noticed, in their own words |
| shaping | 2 author, or control | shaped or control, and the opening settings | settings path, kept settings, timings |
| second viewing | 3 watch again | none | attention item, revisions |
| measures | measures | none | scaled items, open item, criterion answer |

A session runs six clips, three shaped and three control, in an order set by configuration. Show the intro card once at the start, the author intro once before the first shaped clip, the control intro once before the first control clip, and the probe and done screens once at the end.

### 3.3 Screens and states

Table 4 lists every screen with the states to build for each.

**Table 4. Screens and required states**

| Screen | Key elements | States to build |
|---|---|---|
| intro | instruction card, Begin | default |
| 1 watch | stage with placed captions or none, upper lane filling | playing, paused, ended with the attention item, captions off |
| list | entry field, listed tokens, Done listing | empty, one or more listed, confirming nothing more |
| author intro | instruction card, Start | default |
| 2 author | looping shot, skeleton, parked strip | opens unfolded, opens folded, no caption yet, writing, caption fresh, caption stale, auditioning, folded to scene, folded to atmospheric |
| control intro | instruction card, Start | default |
| control | looping shot, visual skeleton, parked strip | the same states as 2 author |
| 3 watch again | stage with placed captions, lower lane playing | playing, paused, revising, ended with the attention item, finish disabled until one full playthrough |
| measures | item rows with discrete scales, one open item, Continue | untouched, partly answered, complete |
| probe and done | free-text field, summary, log download | empty, filled, complete |

During shaping, loop the current shot so the moment stays present while its caption is shaped, and open the skeleton at the settings configuration assigns, which may be folded. During the second viewing, Revise pauses playback, drops the caption from the frame into the strip, and opens the skeleton for the current shot in the working column. The revised commit places it again and replaces the kept caption. A control clip follows the same steps with the visual skeleton in 4.6, and its second viewing shows the visual descriptions the participant kept, placed the same way.

## 4. Components

### 4.1 Skeleton

Build the skeleton as a list of rows, one per admitted source, in their current order of mention. Each row carries the source as a bracketed token in the caption register, [DISTANT SIREN], and to the right an eyebrow with its visibility phrase and temporal phrase. Rows are the touch targets. Tapping a row strikes it. A struck row falls beneath the fold handle, under a hairline, struck through, still legible, and still tappable to restore. Struck sources leave the ranking, so the ordering columns recompute around the sources that remain.

Holding a row auditions it. While held, the parked strip shows, ghosted, the caption that would be written if only that source were mentioned, under the eyebrow If only this were mentioned, and releases back to whatever the strip held. Holding a group head auditions the group. The auditioned sentence is not the caption for the current list and never enables Keep.

### 4.2 Orderings

Balance is a choice among the distinct orderings the admitted sources can take between an eye-first list and an ear-first list. Draw those orderings. Do not imply them. Beside the rows, run a set of narrow columns from a small Eye label on the left to a small Ear label on the right, one column per distinct ordering for this shot's admitted set. Fill the selected column with ink. Its ordering is the one the list shows, so the rows align with it. Draw the other columns as hairlines, and run a thin line for each source across the columns through the positions it would take, so the participant sees which sources trade places when the list leans toward the eye or the ear. A source that is loud and off-screen drops toward the Eye end, a source that fills the frame and barely sounds drops toward the Ear end, and sources on which seeing and hearing agree run level. Tapping a column selects it, and the rows move to match.

No numerals anywhere on this component. A shot whose admitted sources have one ordering shows one column and reads as such. When admission changes the number of orderings, carry the selection to the nearest ordering rather than resetting it.

Build a second required state of this component, the crossfader state, used as a comparison condition. It has the same segments, one per ordering, with only the selected segment filled and the Eye and Ear labels at the ends, and no drawn lines. Place it in the same spot as a state of the same component, not as a separate screen, so the two states differ in nothing else.

### 4.3 Fold

Detail is a change in the list's structure, and the control is the fold. Put a handle at the foot of the admitted rows. It drags upward to fold and downward to unfold, with a pinch on the tablet as an equivalent, and rests at four levels named in the eyebrow beside the handle. At Itemized every admitted source is a row. At Grouped, rows gather under three role heads. Underneath holds the sounds that run beneath everything, Stands out holds the sounds that rise as events, and Only here holds the sounds that belong to this place. The heads become the ranked rows, members sit indented and dim beneath them, and striking a head strikes its members. At Scene the list is one row naming the scene. At Atmospheric the row loses its nouns and carries only the temporal character of the moment as a whole as its phrase.

At Scene and Atmospheric there are no source rows to strike and one ordering. Collapse the ordering columns to their two labels. Hide the struck rows with the admitted rows, since there is no row for a restored source to return to, and bring them back on unfolding. Gray nothing, because nothing remains that could respond. When configuration opens a shot at Atmospheric, open the list folded to that one row with the handle at its top position, and let the participant unfold from there.

### 4.4 Track

The caption is one object in two positions. Parked, it occupies two reserved lines in the strip beneath the frame, centered as a caption is. Placed, it occupies the same two lines in a box in the lower third of the frame. Face, size, box, and ground are identical in both positions.

Parked states belong to shaping. Empty, the strip carries the helper Adjust, then show the caption in the eyebrow style. Fresh, it shows the caption in full ink. Stale, it keeps the caption but dims it, adds the eyebrow Written before your last change, and marks in the skeleton the rows that moved since the caption was written, so the participant can see what the sentence no longer describes. Auditioning, it shows the one-source sentence ghosted. The read caption never vanishes. Freshness is a comparison, not a flag. The caption is fresh whenever the list matches the list it was written for, so a participant who changes the list and then changes it back has a fresh caption again without requesting one.

Placed states belong to viewing. During the first viewing the box carries the automatic caption for the current shot, or is absent when configuration sets captions off. During the second viewing it carries the kept caption for the current shot. The box appears and clears with the shot's timing.

The move between positions is the commit. On Keep the caption slides up from the strip into the frame, and the shot advances. On Revise the caption drops from the frame into the strip as the skeleton opens. Under reduced motion it cuts rather than slides.

### 4.5 Listing

After the first viewing and its attention item, the participant lists the sounds they noticed. Lay the screen out as a single column with the heading, an entry field, and Done listing. Each submitted entry becomes a bracketed token in the participant's own words, in the same register the skeleton will later use, so the two lists are visibly the same kind of thing. No autocomplete, no suggestion, no count, and no reference to the inventory anywhere on the screen. An empty list is allowed. Done listing asks once, Nothing else you noticed, with Nothing else and Add more as the answers, and Nothing else submits. The list is not shown again during shaping, and rows the participant did not list are not marked when the skeleton opens.

### 4.6 Control task

The control clip replaces the audio skeleton with a visual one and changes nothing else. Tokens are visible things in the shot, [PARKED VAN], each with a size phrase and a motion phrase as its eyebrow in place of visibility and temporal character. The ordering columns run from a Near label to a Far label. The fold has the same four levels, with Grouped gathering rows under three role heads supplied with the visual inventory. The caption is a plain visual description in sentence prose, parked while shaped and placed when kept. Gestures, states, actions, copy, and identity are identical to the shaped clip. Build it as the same screen with different tokens and pole labels.

### 4.7 Measures

After the second viewing and its attention item, the participant answers a short set of items on a single-column screen. Build each item as a row with its text on the left and a discrete scale on the right, drawn as a row of equal boxes with the pole labels beneath the end boxes, and nothing continuous anywhere. Most items use five boxes. The attention item, which also appears alone after each viewing, uses seven boxes from Mostly the image to Mostly the sound. The criterion sentence is answered with two boxes. One open item, What makes this place sound like itself, takes free text. Continue stays disabled until every scaled item has an answer. Item wording is supplied separately. Build the row and the scale.

### 4.8 Actions

Show caption requests prose for the current list. Style it as the secondary, outlined action, with the writing state on the button itself. Keep this caption is the single filled action, disabled while the strip is empty or stale, so a participant cannot keep a caption they have not read. Keeping locks the list, places the caption, fills the shot's segment in the lower lane, and advances to the next shot. The criterion sentence sits under the buttons. No reroll affordance exists anywhere.

Table 5 gives every gesture on the shaping screen with its consequence on each side of the surface.

**Table 5. Gestures and consequences**

| Gesture | Skeleton | Caption |
|---|---|---|
| tap a row | struck or restored, columns recompute | stale if a caption is showing |
| hold a row | that row alone marked | ghosted one-source caption in the strip while held |
| tap a column | list adopts that ordering | stale |
| drag the fold handle, or pinch | list folds to the next level | stale |
| Show caption | unchanged | written into the strip for the current list |
| Keep this caption | locked, next shot loads | placed on the frame, committed to the lower lane |
| Revise this caption | reopened for the current shot | dropped from the frame into the strip |

## 5. Copy

Table 6 gives the exact strings. Use them verbatim. Write buttons in sentence case and let the type treatment set them uppercase. All copy is an English placeholder. Korean copy will be supplied separately, so every text container must hold both lengths.

**Table 6. Copy**

| Element | String |
|---|---|
| app title | Caption session |
| clip count | Clip 2 of 6 |
| timeline lanes | Automatic · Yours |
| viewing eyebrows | First viewing / Second viewing · your captions · revise any shot as it plays |
| attention item | Where was your attention? · Mostly the image · Mostly the sound |
| listing heading | What sounds did you notice? |
| listing field | Type a sound and press enter |
| listing confirm | Nothing else you noticed? · Nothing else · Add more |
| fold levels | Itemized · Grouped · Scene · Atmospheric |
| role heads | Underneath · Stands out · Only here |
| ordering poles | Eye · Ear |
| visual poles | Near · Far |
| strip, empty | Adjust, then show the caption |
| strip, stale | Written before your last change |
| strip, auditioning | If only this were mentioned |
| criterion sentence | Is this the caption you'd want for this moment? |
| open item | What makes this place sound like itself? |
| primary actions | Begin · Start · Done listing · Show caption · Keep this caption · Revise this caption · Finish viewing · Continue · Submit · Download session log |
| author helpers | Replay shot · Pause loop |
| busy states | Writing… · Preparing comparison… |
| follow-up headings | Which sounds were in this clip? · Which clip is this sound from? · Which caption would you rather have? |
| follow-up actions | Play · Show the other caption · I'd rather have this one · Next |
| follow-up eyebrows | Caption A · Caption B |
| probe question | Across the clips, what, if anything, would you still change about the captions that the controls didn't let you change? |
| error | Caption request failed. Check the connection and try again. |

Errors state what happened and the way forward, in the interface's voice, without apology. Every action keeps one name through the whole flow. The shaping commit and the follow-up choice are different actions with different names, so neither is mistaken for the other.

## 6. Visual identity

Typeset the interface the way a caption track is typeset. Captions have a folk typography, the bracketed non-speech descriptions of subtitles for the deaf and hard of hearing, and this instrument makes captions, so it borrows that register rather than the register of an instrument panel. The generated caption itself stays sentence prose. The two registers sit on one face and mark the difference between the structure the participant works and the sentence the model writes from it. The participant's own listed sounds take the same bracketed register, so what they noticed and what the tool knows are visibly the same kind of object.

Use one typeface throughout, chosen for caption legibility at small sizes on a tablet, with unambiguous uppercase and a regular weight that reads well in sentence case. The face is your choice within that rule. Table 7 sets the sizes.

**Table 7. Type**

| Element | Size | Treatment |
|---|---|---|
| tokens on the skeleton and the listing | 13 | bracketed, uppercase, lightly tracked |
| role heads | 13 | as tokens, with members beneath them at the same size in dim |
| caption prose, both positions | 22 on a 30 line | sentence case, regular, centered, two lines reserved |
| eyebrows | 10 | uppercase, letterspaced, dim |
| buttons | 12 | uppercase |
| instruction cards and measure items | 14 on a 1.7 line for body, 20 for headings | same face |

Two materials, given in Table 8. The clip and its caption are one dark object, caption text on a near-black ground, whether that ground is the strip beneath the frame or the box on it. The working column and everything else sit on a light ground. The thing being made and the surface that shapes it are visibly different materials.

**Table 8. Palette**

| Token | Hex | Role |
|---|---|---|
| screen | #111311 | stage surround, parked strip, placed box at 80 percent |
| caption | #F2F2EC | caption text in both positions |
| bench | #E9EAE4 | working ground, cards, timeline, listing, measures |
| ink | #1A1C1B | text, selected column, filled lane segments, answered boxes |
| dim | #6F726E | eyebrows, members, stale caption, struck rows |
| rule | #C3C6C0 | hairlines, unselected columns, empty lane segments, empty boxes |
| mark | #EAC93C | primary action, the lower-lane segment being shaped |

The one accent is caption yellow, the color broadcast and teletext subtitling reserve for emphasis. Use it only on the primary action and on the segment of the lower lane currently being shaped or revised. Keep every surface flat, with zero border radius, 1px hairlines in rule, no shadows, no gradients, and no icons. Carry nothing meaningful by color alone. Struck rows are struck through as well as dimmed, the selected column is filled and the rows align with it, stale is dimmed as well as labeled, and an answered box is filled as well as inked. Work spacing on a 4 and 8 scale. Make the stage around 720 wide with the strip beneath it, and give the working column the remaining width behind a 24 gutter.

## 7. Behavior, motion, and accessibility

The deterministic layer responds instantly to every gesture. Striking, restoring, selecting an ordering, and folding all update the skeleton and the columns before the finger lifts. Prose generation is slow and explicit, always behind Show caption, with the writing state on the button rather than a global spinner. Identical settings return the identical cached caption, and a settings state the participant has returned to costs nothing to show again. Single-source captions for the audition are precomputed, so the audition is instant. The inventory for a clip is not delivered to the session until the listing for that clip is submitted, so the skeleton cannot appear early by any path.

Move the caption between its two positions with a short linear motion on Keep and on Revise, and cut under reduced motion. Move rows with the same short linear motion when the list reorders or folds, so a reorder is legible as movement. Under reduced motion the list changes without animation and the rows that moved are marked instead, with the same marking the stale state uses.

The session runs as a kiosk, fullscreen with navigation suppressed, every interaction autosaved to an event log, resuming after a crash, gated by a participant identifier. The log has no participant-facing UI beyond the single download action on the final screen.

Make focus visible on every interactive element. Rows are at least 44 tall, ordering columns at least 40 wide, and scale boxes at least 44 square. Hold contrast at the small uppercase sizes, which are the risk point of this identity. Check the caption on the dark ground in both positions separately from the tokens on the light bench.

## 8. Follow-up session

On a later day the participant completes a short session in a browser, phone first, single column, in the same identity. Nothing here shows the skeleton. Table 9 lists its screens.

**Table 9. Follow-up screens and required states**

| Screen | Key elements | States to build |
|---|---|---|
| intro | instruction card, participant identifier field, Begin | default |
| recognition | still frame of the clip, list of sound names with boxes | untouched, some checked, complete |
| sound only | Play, excerpt, row of clip stills to choose from | not played, played, chosen |
| check | looping clip with placed caption, Show the other caption, I'd rather have this one | preparing comparison, caption A showing, caption B showing, choice enabled |
| done | summary | complete |

Recognition runs once per clip. The list mixes sound names supplied by configuration, in the bracketed register, and the participant checks the ones they remember hearing in that clip. Sound only runs once per excerpt. An audio excerpt plays with no image, and the participant picks the clip it came from among stills of all six. The check runs once per shaped clip on the placed caption, exactly as it looks in the kiosk. The clip loops, the caption box shows one of two captions, a single toggle switches to the other, which caption is A is set by configuration per clip, and the choice stays disabled until both captions have been on the frame.

## 9. Out of scope

Do not build the researcher bench or the admin calibration view, and let nothing from them appear on a participant screen. The bench adds a residual field that scatters every source by visibility against salience with mute, solo, and lasso acting on it, drag-to-solve on the balance control, a contact sheet of captions with shots as columns and runs as rows, and a read-only live monitor of a session. The admin view holds a saturation curve that sets the visibility constant, shot splitting on a distance signal with a threshold line and exclusion shading, role annotation into Underneath, Stands out, and Only here, and the configuration that sets clip order, the caption condition of each first viewing, shaped or control per clip, opening settings, naming policy, follow-up sound names, and criterion wording. Frozen calibration values are never shown in the session.

## 10. Check before delivering

Run every row of Table 10 before handing over.

**Table 10. Delivery checks**

| Check | Pass condition |
|---|---|
| screens | every screen in Table 4 and Table 9 exists in every listed state |
| components | skeleton, orderings with the crossfader state, fold, track, listing, control skeleton, measure row, and actions exist in every described state |
| one surface | nothing acts on the caption except a gesture on the skeleton, Show caption, Keep this caption, and Revise this caption |
| numerals | none on the skeleton, the ordering columns, or the fold, and no measured value shown as a number |
| commit | Keep this caption is disabled while the strip is empty or stale, and the audition never enables it |
| caption object | parked and placed captions share face, size, box, and ground and differ only in vertical position |
| gating | the skeleton is unreachable before Done listing is submitted |
| accent | mark appears only on the primary action and on the lower-lane segment being shaped or revised |
| surfaces | zero border radius, 1px hairlines in rule, no shadows, no gradients, no icons |
| redundancy | struck, selected, stale, and answered each read without color |
| copy | every string matches Table 6 and every container holds the longer Korean length |
| targets | rows at least 44 tall, columns at least 40 wide, scale boxes at least 44 square, focus visible everywhere |
| scope | no bench or admin element, no reroll, no history of read captions, no scene overlay on the stage |