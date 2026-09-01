# Caption session, prompts for Claude Design

Attach both files to the project before the first message, spec-identity.md as the design system and spec-behavior.md as the spec. Do not paste the spec into chat. Send one block per message, wait for the canvas, and send the next only when the current frames are right. Every section and table number below refers to the spec unless it says spec-identity.md.

## Prompt 0. Read first

```
Before building anything, read both attached files in full. spec-identity.md is the design system. spec-behavior.md is the spec, and its section and table numbers are the ones I will cite. Together they are the only source of screens, states, components, copy, and tokens for this project. Use no external UI kit, icon set, or default component styling. Do not add screens, controls, icons, or states the spec does not list, and do not build the researcher bench or the admin view named in spec section 9. If anything is unclear, ask every question you have now in one message, then stop. Build nothing yet.
```

## Prompt 1. Foundation

```
Build one foundation frame at 1180 by 820 that proves the design system before any screen exists. On the bench ground, set the type scale from spec-identity.md Table 1 as live samples, the seven palette tokens as flat swatches labeled with name and hex, one bracketed token row with a visibility phrase and a temporal phrase on its right, one eyebrow line, and the primary and secondary buttons in default, disabled, and busy states using Keep this caption, Show caption, and Writing…. Beside them, on a dark stage stub with a 16:9 frame, show the caption object twice with the same placeholder two-line sentence, once parked in the strip flush beneath the frame at frame width, once placed in a box in the lower third of the frame at 80 percent of screen. Face, size, box, and ground must be identical in both. Follow spec sections 4.4 and 6. Zero radius, no shadows, no icons.
```

## Prompt 2. Skeleton

```
Build the skeleton component from spec sections 4.1 and 4.3 on a components frame, states side by side, each labeled with an eyebrow. Show it Itemized with five admitted rows, then Itemized with two rows struck and sitting under the fold handle beneath a hairline, struck through, dim, and still legible, then Itemized with one row held for audition, then Itemized with two rows marked as moved since the caption was written, then Grouped with the role heads Underneath, Stands out, and Only here as the ranked rows and their members indented and dim beneath them, then Grouped with one head struck so its members strike with it, then Scene as one row naming the scene, then Atmospheric as one row carrying only a temporal phrase with the handle at its top position and the struck rows hidden. Every row is a bracketed uppercase token with its visibility phrase and temporal phrase on the right, at least 44 tall. The fold handle sits at the foot of the admitted rows with the current level named in the eyebrow beside it. Nothing on the skeleton is ever grayed or disabled.
```

## Prompt 3. Orderings

```
Add the orderings component from spec section 4.2 to the components frame, beside the skeleton, drawn four times. First, five admitted sources with four distinct orderings, as four narrow columns at least 40 wide between a small Eye label on the left and a small Ear label on the right, the selected column filled with ink, the other three as hairlines in rule, and one thin line per source running across the columns through the positions it would take, with the adjacent skeleton rows aligned to the selected column. Second, a shot with one ordering only, as a single column. Third, the collapsed state at Scene or Atmospheric, reduced to the two labels. Fourth, in exactly the same place and size, the crossfader state used as a comparison condition, with the same segments one per ordering, only the selected segment filled, Eye and Ear at the ends, and no drawn lines. No numerals anywhere on this component.
```

## Prompt 4. Shaping screen

```
Build the shaping screen, 2 author in spec Table 4, at 1180 by 820 on the layout in section 3.1. The timeline runs across the top with the clip count eyebrow Clip 2 of 6 at the left and two lanes labeled Automatic and Yours, shots as segments, and the segment being shaped in mark. The stage sits on the left with a looping 16:9 frame, a small shot label, and the parked strip flush beneath it at frame width. The working column sits to the right behind a 24 gutter and holds the skeleton with its ordering columns and fold handle. Beneath, the criterion sentence Is this the caption you'd want for this moment? sits under Show caption, outlined, and Keep this caption, filled in mark, with Replay shot and Pause loop as the only helpers. Produce the nine states from Table 4 as separate frames. Opens unfolded. Opens folded. No caption yet, with the strip reading Adjust, then show the caption. Writing, with Writing… on the button. Caption fresh, in full caption ink. Caption stale, with the strip dimmed under Written before your last change and the moved rows marked in the skeleton. Auditioning, with the ghosted one-source sentence under If only this were mentioned. Folded to scene. Folded to atmospheric. Keep this caption is disabled whenever the strip is empty or stale. Use sections 4.4 and 4.8 and Table 5.
```

## Prompt 5. Control screen

```
Duplicate the shaping screen as the control screen from spec section 4.6 and change only the tokens and the pole labels. Tokens are visible things in the shot, such as [PARKED VAN], each with a size phrase and a motion phrase as its eyebrow. The ordering columns run from a Near label to a Far label. The three role heads at Grouped come from the visual inventory, so use three placeholder visual role names. The parked and placed caption is a plain visual description in sentence prose. Produce the same nine states as separate frames. Gestures, actions, copy, and identity stay identical.
```

## Prompt 6. Viewing screens

```
Build the two viewing screens from spec Table 4 at 1180 by 820 on the same layout, with the strip's space reserved and empty on every frame. On 1 watch, the stage plays the clip with the automatic caption placed in the lower third of the frame, the upper lane fills shot by shot, and the working column holds the instruction line First viewing. Make four frames, playing, paused, captions off with no box on the frame and the upper lane empty, and ended, where the working column holds the single attention item Where was your attention? as seven equal boxes at least 44 square from Mostly the image to Mostly the sound. On 3 watch again, the stage plays the kept captions from the lower lane, and the working column holds Second viewing · your captions · revise any shot as it plays with Revise this caption and Finish viewing beneath. Make five frames, playing, paused, revising, with playback paused, the caption dropped from the frame into the strip, the skeleton open for the current shot in the working column, and that shot's lower-lane segment in mark, ended with the attention item, and finish disabled until one full playthrough. Follow sections 3.3, 4.4, and 4.7.
```

## Prompt 7. Single-column screens

```
Build the single-column screens from spec Table 4, where the working column widens across the layout. Intro, author intro, and control intro are each one instruction card at 14 on a 1.7 line with a 20 heading and a single primary button, Begin for the intro and Start for the other two. List has the heading What sounds did you notice?, an entry field with the placeholder Type a sound and press enter, each submitted entry rendered as a bracketed uppercase token in the participant's own words, and Done listing. Make it empty, with one or more listed, and confirming, where Done listing has asked Nothing else you noticed? with Nothing else and Add more as the two answers. No autocomplete, no suggestion, no count. Measures has item rows with the item text on the left and a discrete scale on the right drawn as equal boxes at least 44 square with pole labels beneath the end boxes, five boxes for most items, seven for the attention item, two for Is this the caption you'd want for this moment?, then the open item What makes this place sound like itself? as a free-text field, then Continue, disabled until every scaled item has an answer. Make it untouched, partly answered with filled boxes, and complete. Probe and done has the probe question from Table 6 as a free-text field, a short session summary, Submit, and Download session log. Make it empty, filled, and complete. Use placeholder item wording. Follow sections 4.5 and 4.7.
```

## Prompt 8. Follow-up

```
Build the follow-up from spec section 8 and Table 9 as phone-first single-column frames in the same identity, with no skeleton anywhere. Intro has an instruction card, a participant identifier field, and Begin. Recognition has a still frame of the clip, the heading Which sounds were in this clip?, a list of bracketed sound names each with a box at least 44 square, and Next, in untouched, some checked, and complete. Sound only has the heading Which clip is this sound from?, a Play button, the excerpt with no image, a row of six clip stills to choose from, and Next, in not played, played, and chosen. Check has the looping clip with the placed caption in the lower third exactly as in the kiosk, the eyebrow Caption A or Caption B, the heading Which caption would you rather have?, Show the other caption as the single toggle, and I'd rather have this one, in preparing comparison with Preparing comparison… on the button, caption A showing, caption B showing, and choice enabled only once both captions have been on the frame. Done is a summary. Then show the check frame once more at desktop width.
```

## Prompt 9. Interactions

```
Make the prototype live where the canvas allows it, following spec Table 5 and section 7. On the shaping screen, tapping a row strikes or restores it and recomputes the ordering columns, holding a row ghosts its one-source sentence into the strip and releases back to what the strip held, tapping a column reorders the rows to that ordering, and dragging the fold handle moves through Itemized, Grouped, Scene, and Atmospheric. Each of those marks the current caption stale and disables Keep this caption, and returning the list to the state the caption was written for makes it fresh again without a new request. Show caption switches the button to Writing… and then writes a caption into the strip. Keep this caption slides the caption up from the strip into the frame with a short linear motion, fills the lower-lane segment, and loads the next shot. On the second viewing, Revise this caption pauses playback, drops the caption from the frame into the strip, and opens the skeleton. Row reorders and folds use the same short linear motion. Add a reduced-motion variant in which every move cuts and moved rows are marked instead.
```

## Prompt 10. Audit

```
Audit every frame against spec Table 10 and spec-identity.md, row by row. For each row report pass or name the exact frames that fail, then fix every failure. Check in particular that no icon, shadow, gradient, or border radius exists anywhere, that mark appears only on the primary action and on the lower-lane segment being shaped or revised, that no numeral appears on the skeleton, its columns, or its fold, that every string matches spec Table 6 exactly, that every text container has room for the Korean strings that will replace the English placeholders, and that the parked and placed captions are identical in face, size, box, and ground.
```

## While running

Table 1 covers the situations that come up between prompts.

**Table 1. Between prompts**

| Situation | Do this |
|---|---|
| a stage is right and you want to try another direction | say "Save what we have and try a different approach" before the next prompt |
| one element needs a small fix | leave an inline comment on the canvas, and paste it into chat if it is not picked up |
| the tool asks questions | answer only what the spec leaves open, and point it back to the section for the rest |
| the output contains something the spec does not list | tell it to remove the element, not refine it |
| chat upstream error | open a new chat tab inside the same project and continue from the next prompt |
