# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-09-08
- Surfaces: multi-clip calibration, three-video interactive viewing, one final survey.
- Evidence: `identity.css`, `regen.css`, `regen.html`, `.omx/plans/long-video-caption-phase.md`.

## Brand
Quiet, precise and reassuring. Reuse the existing near-white ground, dark ink,
restrained blue accent and offline system font stacks. Avoid decorative gradients,
new branding, unrelated colors and large animated transitions.

## Product goals
Make calibration and viewing feel like one study with two clear stages. Keep
attention on the media. Success means visible progress, understandable controls,
stable captions and recoverable interruptions. No live-video capture or weight tuning.

## Personas and jobs
Participants report what they notice, set caption preferences and watch three
five-minute videos. Researchers prepare media and inspect versioned observations.
The interface must work on laptops, touch screens and slow connections.

## Information architecture
Calibration preferences → repeated clip viewing/observations → calibration complete
→ three interactive videos → one final experience survey → receipt. Progress is
informational; forward transitions follow acknowledged server state.

## Design principles
Reuse existing tokens. One primary action per screen. Keep requested controls
distinct from applied captions. Never cover playback controls with captions.

## Visual language
Use `identity.css` as the token owner: color, type, radius, 44px targets and focus.
Use the existing 880px reading column; allow a wider watching surface with video
beside a compact control panel. Eight-pixel spacing rhythm. Black video stage,
reserved caption band, white controls. Respect reduced motion.

## Components
Reuse primary/secondary buttons, eyebrow/head/lede hierarchy and bordered panels.
Add a two-stage progress rail, typed survey fields, visual-point editor and two-axis
pad. Native sliders are synchronized keyboard-accessible alternatives to dragging.
All surfaces use the same button geometry and caption/control terminology.

## Accessibility
Target readable contrast using existing tokens, visible focus and native input
semantics. Label both axes with words and values. Announce loading/errors politely;
do not announce every playback tick. Caption area remains stable. No color-only status.

## Responsive behavior
At narrow widths stack controls below video, keep full-width survey groups and
avoid horizontal scrolling. Fullscreen includes both video and modulation controls.
Pointer interactions also have keyboard equivalents; targets remain at least 44px.

## Interaction states
Loading explains the next action. Pending media preserves calibration and offers
retry. Errors preserve input and offer retry. Offline exposure data stays queued.
Completion shows a receipt. Disabled actions explain what remains. No placeholder
media is presented as prepared study footage.

## Content voice
Plain instructions addressed to the participant. Use “Acoustic detail”,
“Source and scene detail”, “Calibration”, and “Your viewing experience”. Technical
provenance belongs in researcher exports. Draft survey wording is documented.

## Implementation constraints
FastAPI and vanilla JS/CSS, existing local assets, no new packages. New protocol
has separate versioned state; legacy UI remains supported. Verify API transitions,
browser flow, narrow layout and theme-token reuse. Real media/model performance
requires the three prepared videos and a configured model.

## Open questions
- Calibration clip count/content: researcher configuration.
- Final wording/translations: pilot review before participant recruitment.
- Actual long-video files: researcher preparation; readiness gate remains closed.
