# Steering level overlay — 2026-09-14

The viewing player now shows a volume-style overlay when either detail control
changes. Both selected levels have filled bars and numeric percentages. The
overlay stays visible during a held pointer, holds approximately 1.5 seconds
after input/commit, then fades over 220 ms (140 ms fade-in). New input resets
the hold. Reduced motion switches visibility without animated fading.

Sliders appear before the two-axis pad with Brief/Detailed and General/Specific
endpoints. The non-interactive overlay has no keyboard focus and is hidden from
screen readers; native labelled sliders expose the selected value. A separate
line under the caption records its actual applied levels or explicitly identifies
the prepared fallback. This display does not measure audio volume or confidence.

## Evidence

- 21 focused Chromium checks passed, including mouse and touch hold/release,
  touch cancellation, keyboard, repeat-input timing, minimum/maximum values,
  preset/reset, selected-versus-applied status, fallback, fullscreen, Korean
  mobile layout, pointer transparency and reduced motion.
- 40 existing localization, study and continuation tests passed.
- Repository-wide Ruff and formatting checks, mypy (175 source files), JS syntax
  and `git diff --check` passed.
- Visual verdict: 94/100. Desktop and mobile layouts keep the meter clear of the
  caption band and playback controls.

Browser evidence: `/tmp/regen-meter-20260914-v2/result.json`,
`desktop-input.png` and `mobile-input-ko.png` in the same directory.
The browser test uses a short real-media excerpt and deterministic API/caption
fixtures to isolate interface state and timing. It does not claim new inference,
full study completion, multi-user capacity or physical-device verification.
No participant response records were created by this check.

The existing service reads the frontend files per request; new page loads receive
the revision. Tabs already open must reload to use it.

## Changed files

`study.js` owns overlay lifecycle/input feedback and applied-caption status;
`study.css` owns fading, filled sliders and responsive placement; `study-ko.json`
contains Korean copy. `DESIGN.md` records the interaction contract and
`tests/browser_steering_meter.mjs` provides the repeatable UI check.

The implementation reuses native sliders and the current control submission
path; it adds no package, model call or playback/protocol change. Remaining
limits: Chromium automation with emulated touch; physical-device and Firefox/
WebKit testing have not been performed.
