# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-09-23
- Surfaces: paired short-clip calibration, per-video questions, chapter-level PRSS.
- Evidence: `app.py`, `regen.js`, `study_api.py`, `study.js`, `items/default.json`, `identity.css`.

## Brand
Quiet, precise and reassuring. Reuse the existing near-white ground, dark ink,
restrained blue accent and offline system font stacks. Avoid decorative gradients,
new branding, unrelated colors and large animated transitions.

## Product goals
Make calibration and viewing feel like one study with two clear stages. Keep
attention on the media. Success means visible progress, understandable controls,
stable captions and recoverable interruptions. No live-video capture or weight tuning.

## Personas and jobs
Participants report what they notice, set caption preferences and watch the configured
five-minute videos. Researchers prepare media and inspect versioned observations.
The interface must work on laptops, touch screens and slow connections.

## Information architecture
The integrated regen interface has two chapters:
1. For each short clip, watch the prepared captions, answer the clip questions and
   visual/sound observations, watch the same clip with updated captions, and answer
   the post-viewing questions. Repeat for the remaining clips, then answer overall
   PRSS once for the short-clip chapter.
2. Watch each longer video with adjustable captions, then answer four questions
   about that video. Repeat for all configured videos, then answer overall PRSS and
   questions about the controls and effort once before receiving a receipt.

The longer-video chapter uses one viewing per video to limit participant burden.
The standalone observation-only calibration entry remains a separate protocol.
Progress identifies both chapter and clip, with totals from the frozen session;
forward transitions follow acknowledged server state. New v4 configurations allow
any positive short/long clip counts; legacy A/B configurations remain supported. Survey drafts belong to the specific clip and viewing. Active
phases keep their assigned protocol and frozen instruments; an unstarted viewing
phase receives the current instrument and matching flow together at handoff.
Phase 1 questionnaire ratings now inform the frozen Phase 2 caption-writing
guidance as well as being retained for analysis. Credibility and distraction
responses inform caution and conciseness; paired ART and PRSS supply experience
context. Explicit live controls take precedence. The numeric controls remain at
the neutral midpoint until adjusted because the questionnaires do not directly
measure acoustic/source-detail preference. See `PERSONALIZATION.md`.

## Design principles
Reuse existing tokens. One primary action per screen. Keep requested controls
distinct from applied captions. Never cover playback controls with captions.
Separate questions about the just-watched clip from questions about the whole
chapter. Preserve configured item wording, response scales, and provenance; use
chapter-specific instructions to explain the overall PRSS reference period.

## Visual language
Use `identity.css` as the token owner: color, type, radius, 44px targets and focus.
Use the existing 880px reading column; allow a wider watching surface with video
beside a compact control panel. Eight-pixel spacing rhythm. Black video stage,
reserved caption band, white controls. Respect reduced motion.

## Components
Reuse primary/secondary buttons, eyebrow/head/lede hierarchy and bordered panels.
Add a two-stage progress rail, typed survey fields and a visual-point editor.
Viewing controls use two labeled native sliders, presets and Reset. Each slider
supports pointer and keyboard input for its own detail dimension.
All surfaces use the same button geometry and caption/control terminology.

Steering feedback uses a compact volume-style overlay at the top of the video,
away from its captions and playback controls. It displays both **selected** detail
levels as filled bars and values, with no implication of sound volume or
model confidence. Input on either slider reveals
the overlay; holding a pointer keeps it visible; it fades after 1.5 seconds of
idle time. New input restarts that hold. Reduced motion removes the transition.
The overlay is non-interactive and duplicates the accessible native slider values.
A persistent line beside the caption identifies its actual applied levels, or
explicitly identifies a prepared fallback. A control change never changes that
line until a new caption is displayed. These rules apply on mobile and fullscreen.

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
requires the configured prepared videos and a configured model.

## Open questions
- Calibration clip count/content: researcher configuration.
- Existing PRSS wording refers to a place; chapter-wide administration and draft
  translations remain research-instrument review questions. Do not silently rewrite
  configured items or describe the pilot items as validated.
- Each run needs validated media; viewing remains gated until every configured video is ready.
