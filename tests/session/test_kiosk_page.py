"""Source-level invariants of the kiosk page (build contract 12, "pages").

No browser runs here. These assertions read kiosk.html / kiosk.css / kiosk.js
as text and pin the properties that a server test cannot see and a screenshot
would not catch: every Table 6 string present verbatim and defined once, the
identity's "never" list (radius, shadow, gradient, icon, emoji, off-palette
color) absent, mark confined to its two places, every log event type emitted,
Keep gated on freshness by comparison, and the inventory requested only after
the listing.submit POST has resolved — the gating on which the whole design's
"show nothing the tool knows first" rests (spec 2).
"""

from __future__ import annotations

import re
from importlib.resources import files


def _source(name: str) -> str:
    return files("dpo.session").joinpath(name).read_text(encoding="utf-8")


def _html() -> str:
    return _source("kiosk.html")


def _css() -> str:
    return _source("kiosk.css")


def _js() -> str:
    return _source("kiosk.js")


def _strings_block(js: str) -> str:
    """The text of the STRINGS object, the one place copy may be defined."""
    start = js.index("const STRINGS = {")
    end = js.index("\n};", start)
    return js[start:end]


def _function_body(js: str, signature: str) -> str:
    start = js.index(signature)
    end = js.index("\n}\n", start)
    return js[start:end]


# spec Table 6, verbatim. The clip count and the shot label are patterns
# ("Clip 2 of 6" → "Clip {n} of {m}"; build contract 9), asserted separately.
TABLE_6 = [
    "Caption session",
    "Automatic",
    "Yours",
    "First viewing",
    "Second viewing · your captions · revise any shot as it plays",
    "Where was your attention?",
    "Mostly the image",
    "Mostly the sound",
    "What sounds did you notice?",
    "Type a sound and press enter",
    "Nothing else you noticed?",
    "Nothing else",
    "Add more",
    "Itemized",
    "Grouped",
    "Scene",
    "Atmospheric",
    "Underneath",
    "Stands out",
    "Only here",
    "Eye",
    "Ear",
    "Near",
    "Far",
    "Adjust, then show the caption",
    "Written before your last change",
    "If only this were mentioned",
    "Is this the caption you'd want for this moment?",
    "What makes this place sound like itself?",
    "Begin",
    "Start",
    "Done listing",
    "Show caption",
    "Keep this caption",
    "Revise this caption",
    "Finish viewing",
    "Continue",
    "Submit",
    "Download session log",
    "Replay shot",
    "Pause loop",
    "Writing…",
    "Across the clips, what, if anything, would you still change about the captions "
    "that the controls didn't let you change?",
    "Caption request failed. Check the connection and try again.",
]

# build contract 6: the event vocabulary the log is analysed by.
EVENT_TYPES = [
    "session.begin",
    "screen.enter",
    "viewing.play",
    "viewing.pause",
    "viewing.ended",
    "attention.answer",
    "listing.add",
    "listing.confirm",
    "listing.more",
    "listing.submit",
    "shape.open",
    "skeleton.strike",
    "skeleton.restore",
    "skeleton.hold",
    "skeleton.balance",
    "skeleton.fold",
    "loop.pause",
    "loop.resume",
    "loop.replay",
    "caption.request",
    "caption.written",
    "caption.failed",
    "caption.keep",
    "revise.open",
    "measures.answer",
    "measures.open",
    "measures.submit",
    "probe.submit",
    "log.download",
    "session.done",
]

# spec-identity.md Table 2: the seven tokens and no others.
PALETTE = {"#111311", "#F2F2EC", "#E9EAE4", "#1A1C1B", "#6F726E", "#C3C6C0", "#EAC93C"}

HEX_RE = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\U0001f900-\U0001f9ff]"
)


class TestCopy:
    def test_every_table_6_string_is_in_strings_verbatim(self) -> None:
        block = _strings_block(_js())
        for text in TABLE_6:
            assert f'"{text}"' in block, text
        assert '"Clip {n} of {m}"' in block
        assert '"Shot {n} of {m}"' in block

    def test_copy_is_defined_once(self) -> None:
        js = _js()
        assert js.count("const STRINGS = {") == 1
        block = _strings_block(js)
        outside = js.replace(block, "")
        for text in TABLE_6:
            assert f'"{text}"' not in outside, f"{text!r} is defined outside STRINGS"

    def test_the_shell_is_titled_from_table_6(self) -> None:
        assert "<title>Caption session</title>" in _html()


class TestSurfaces:
    def test_no_radius_shadow_gradient_or_image(self) -> None:
        for name in ("kiosk.html", "kiosk.css", "kiosk.js"):
            text = _source(name)
            for forbidden in ("border-radius", "box-shadow", "gradient", "<img"):
                assert forbidden not in text, f"{forbidden} in {name}"
            assert not EMOJI_RE.search(text), f"emoji in {name}"

    def test_no_color_outside_the_seven_tokens(self) -> None:
        for name in ("kiosk.html", "kiosk.css", "kiosk.js"):
            found = {match.upper() for match in HEX_RE.findall(_source(name))}
            assert found <= PALETTE, f"{name}: {found - PALETTE}"
        assert {match.upper() for match in HEX_RE.findall(_source("identity.css"))} == PALETTE

    def test_mark_appears_only_on_the_primary_action_and_the_current_segment(self) -> None:
        # kiosk.css and kiosk.js never touch mark; identity.css gives it to
        # exactly two rules (spec-identity.md Palette).
        assert "--mark" not in _css()
        assert "--mark" not in _js()
        identity = _source("identity.css")
        selectors = []
        for rule in re.finditer(r"([^{}]+)\{([^{}]*)\}", identity):
            if "var(--mark)" in rule.group(2):
                selectors.append(" ".join(rule.group(1).split()))
        assert selectors == [".button.primary", ".lane .segment.current"]

    def test_the_only_svg_is_the_orderings_drawing_in_rect_and_line(self) -> None:
        assert "<svg" not in _html() and "<svg" not in _css()
        js = _js()
        assert js.count('svgNode("svg"') == 1
        drawing = _function_body(js, "function drawOrderings()")
        assert 'svgNode("svg"' in drawing
        shapes = set(re.findall(r'svgNode\("(\w+)"', js))
        assert shapes <= {"svg", "rect", "line", "polyline"}
        # No numerals on the component (spec 4.2): the columns carry no text.
        assert 'svgNode("text"' not in js
        assert "aria-label" not in drawing

    def test_the_parked_and_placed_caption_share_one_element(self) -> None:
        html = _html()
        assert html.count('class="caption-object') == 1
        js = _js()
        assert 'moveCaption($("frame"), "caption-object placed")' in js
        assert 'moveCaption($("strip"), "caption-object parked")' in js
        # Short linear motion, cut under reduced motion (spec-identity.md Motion).
        assert "const MOTION_MS = 160;" in js
        assert "prefers-reduced-motion: reduce" in js


class TestBehavior:
    def test_every_event_type_is_emitted(self) -> None:
        js = _js()
        for event_type in EVENT_TYPES:
            assert f'"{event_type}"' in js, event_type

    def test_keep_is_disabled_unless_fresh_by_comparison(self) -> None:
        js = _js()
        assert "keep.disabled = !isFresh();" in js
        fresh = _function_body(js, "function isFresh()")
        # spec 4.4: a comparison of keys, not a flag; spec 4.1: the audition
        # never enables Keep.
        assert "s.strip.key === currentKey(s)" in fresh
        assert "!s.audition" in fresh
        keep = _function_body(js, "async function keepCaption()")
        assert "if (!isFresh()) return;" in keep

    def test_settings_key_matches_the_server_byte_for_byte(self) -> None:
        body = _function_body(_js(), "function settingsKey(level, admitted, order)")
        assert 'return "itemized|" + order.join(",")' in body
        assert 'return "grouped|" + order.join(",") + "|" + admitted.slice().sort().join(",")' in body
        assert "return level;" in body

    def test_inventory_is_requested_only_after_the_submit_post_resolves(self) -> None:
        submit = _function_body(_js(), "async function submitListing()")
        emitted = submit.index('api.emit("listing.submit"')
        flushed = submit.index("await api.flush(")
        fetched = submit.index("await api.inventory(")
        assert emitted < flushed < fetched
        # Nothing between the flush and the fetch may skip the gate.
        assert "if (appended)" in submit[flushed:fetched]
        # The inventory is never requested from anywhere the listing has not
        # been submitted: the only other callers are the resume paths.
        js = _js()
        assert js.count("api.inventory(") == 3

    def test_autosave_is_debounced_and_flushed_with_keepalive(self) -> None:
        js = _js()
        assert "const AUTOSAVE_MS = 250;" in js
        assert "keepalive: !!keepalive" in js
        assert '"visibilitychange"' in js and '"pagehide"' in js
        assert 'method: "POST"' in js and '"/api/events"' in js

    def test_hold_and_fold_thresholds_are_the_spec_values(self) -> None:
        js = _js()
        assert "const HOLD_MS = 400;" in js
        assert "const FOLD_STEP_PX = 40;" in js
        assert 'foldBy(1, "pinch")' in js and 'foldBy(-1, "pinch")' in js
        assert 'foldBy(1, "key")' in js and 'foldBy(-1, "key")' in js
        assert 'setLevel(LEVELS[index], "drag")' in js

    def test_resume_re_enters_the_interrupted_screen(self) -> None:
        js = _js()
        assert '"/api/state?participant="' in js
        assert "enter(snapshot.screen);" in js

    def test_a_revise_loop_on_the_last_shot_wraps_instead_of_ending_the_viewing(self) -> None:
        # While revising the screen is still watch2 (spec 3.3), so the clip's
        # `ended` must be keyed on the shaping state: otherwise the last shot's
        # loop ends the viewing and clears the skeleton from under the finger.
        ended = _function_body(_js(), "function onVideoEnded()")
        assert "if (state.shaping) {" in ended
        assert 'screen === "shape"' not in ended
        assert ended.index("if (state.shaping) {") < ended.index('api.emit("viewing.ended"')

    def test_keep_locks_the_list_and_advances_the_snapshot_before_the_motion(self) -> None:
        js = _js()
        keep = _function_body(js, "async function keepCaption()")
        # spec 4.8 "Keeping locks the list": the lock is set before the event,
        # and every gesture on the skeleton honours it.
        assert keep.index("s.locked = true;") < keep.index('api.emit("caption.keep"')
        for signature in (
            "function strikeOrRestore(id)",
            "function selectOrdering(index)",
            "async function keepCaption()",
        ):
            assert "if (!s || s.locked) return;" in _function_body(js, signature), signature
        assert "s.locked || level === s.level" in _function_body(js, "function setLevel(level, via)")
        assert "s.locked || s.audition" in _function_body(js, "function beginAudition(node, id, pressedAt)")
        assert "!s.locked && !s.writing" in _function_body(js, "function canShow(s)")
        # The snapshot advances before the slide, so a flush or crash during it
        # resumes on the next shot (or watch2), never on the shot just kept.
        assert keep.index("state.snapshot.shot_index = next;") < keep.index("await placeCaption(")
        assert keep.index('state.snapshot.screen = "watch2";') < keep.index("await placeCaption(")
        assert "if (!state.viewing) state.viewing" not in keep

    def test_revise_is_latched_against_re_entry(self) -> None:
        js = _js()
        revise = _function_body(js, "async function onRevise()")
        assert "if (!state.viewing || state.shaping || state.revising) return;" in revise
        assert revise.index("state.revising = true;") < revise.index("video.pause();")
        assert revise.count("state.revising = false;") >= 3
        assert "viewing.error = STRINGS.error;" in revise
        # The frame, Enter, and Space honour the latch too.
        assert "if (state.viewing && !state.shaping && !state.revising) togglePlayback();" in js
        assert 'if (event.key !== " " || !state.viewing || state.shaping || state.revising) return;' in js
        assert "!state.viewing || state.revising) return;" in _function_body(js, "function togglePlayback()")

    def test_a_resume_that_cannot_fetch_the_inventory_shows_the_error_and_retries(self) -> None:
        js = _js()
        shape = _function_body(js, "function renderShape()")
        assert "text: STRINGS.error" in shape
        assert "INVENTORY_RETRY_MS" in shape
        assert "const INVENTORY_RETRY_MS = 2000;" in js
        boot = _function_body(js, "async function boot()")
        assert "document.title = STRINGS.appTitle;" in boot
        assert "text: STRINGS.error" in boot.split("await api.session()", 1)[1]

    def test_leaving_a_stage_screen_silences_the_clip(self) -> None:
        assert 'if (!stage) $("video").pause();' in _function_body(_js(), "function render()")

    def test_the_gating_event_is_emitted_once_per_listing(self) -> None:
        submit = _function_body(_js(), "async function submitListing()")
        assert 'if (!listing.submitted) api.emit("listing.submit"' in submit
        assert "listing.submitted = true;" in submit

    def test_show_caption_needs_something_to_write_and_a_change_clears_the_error(self) -> None:
        js = _js()
        assert "rowsOf(s).length > 0" in _function_body(js, "function canShow(s)")
        assert "disabled: !canShow(s)," in _function_body(js, "function renderActions()")
        assert "if (!canShow(s)) return;" in _function_body(js, "async function showCaption()")
        assert "s.error = null;" in _function_body(js, "function afterChange()")
        assert "s.error = null;" in _function_body(js, "function setLevel(level, via)")
        # A fresh caption clears the reduced-motion marks (spec 4.4 reserves
        # the mark for rows that moved since the caption was written).
        show = _function_body(js, "async function showCaption()")
        assert show.index("s.strip = { caption: written.caption") < show.index("s.movedNow = new Set();")

    def test_the_hold_is_timed_from_the_press(self) -> None:
        js = _js()
        assert "since: pressedAt" in _function_body(js, "function beginAudition(node, id, pressedAt)")
        assert js.count("const pressedAt = performance.now();") == 2
        assert js.count("beginAudition(node, id, pressedAt);") == 2

    def test_typed_text_is_snapshotted_as_it_is_typed(self) -> None:
        js = _js()
        assert js.count('area.addEventListener("input"') == 2
        assert "state.snapshot.probe = area.value;\n    api.save();" in js
        assert "record.open = area.value;\n    api.save();" in js

    def test_the_motion_is_identitys_moving_class(self) -> None:
        js = _js()
        assert "style.transition" not in js
        assert js.count('classList.add("moving")') == 2
        assert ".moving { transition: transform var(--motion) var(--linear); }" in _source("identity.css")
        rows = _function_body(js, "function renderRows(animate)")
        assert rows.index("drawOrderings();") < rows.index("if (animate) flipRows(before);")

    def test_a_pinch_never_strikes_and_the_keyboard_keeps_its_place(self) -> None:
        js = _js()
        assert "let pinched = false;" in js
        assert "pointers.size < 2 && !pinched) strikeOrRestore(id);" in js
        assert "if (!pointers.size) pinched = false;" in _function_body(js, "function liftPointer(pointerId)")
        change = _function_body(js, "function afterChange()")
        assert "const target = focusedTarget();" in change and "refocus(target);" in change
        assert 'id: "pause-loop", "aria-pressed": s.paused ? "true" : "false"' in js
        assert "reflectPause();" in _function_body(js, "function replayShot(emit)")
        assert "s.paused = false;" in _function_body(js, "function replayShot(emit)")

    def test_the_done_screen_has_its_one_action_as_the_primary(self) -> None:
        # spec-identity.md Actions: one filled primary per screen. On probe the
        # download sits secondary beside Submit; on done it is the only action.
        js = _js()
        assert 'class: primary ? "button primary" : "button",' in _function_body(
            js, "function downloadLink(primary)"
        )
        assert "downloadLink(true)" in _function_body(js, "function renderDone()")
        assert "downloadLink()" in _function_body(js, "function renderProbe()")


class TestKiosk:
    def test_the_page_is_a_kiosk(self) -> None:
        html = _html()
        assert "user-scalable=no" in html
        assert '<video id="video" playsinline preload="auto">' in html
        js = _js()
        assert "requestFullscreen" in js
        assert '"contextmenu"' in js
        assert ".skeleton { touch-action: none;" in _css()

    def test_no_external_assets(self) -> None:
        html = _html()
        assert "http://" not in html and "https://" not in html
        assert "@import" not in _css() and "url(" not in _css()
        assert "fonts.googleapis" not in _css()
        assert html.count('href="/static/') == 2 and 'src="/static/kiosk.js"' in html

    def test_missing_participant_shows_one_line(self) -> None:
        assert '"Open this page with a participant identifier."' in _js()
        # ... on a landing screen that asks for it: the title, a field for the
        # identifier, and Continue, which reopens the page with it in the URL.
        landing = _js()[_js().index("function renderLanding") :].split("\n}\n", 1)[0]
        assert "STRINGS.appTitle" in landing
        assert 'class: "field"' in landing and "STRINGS.continue" in landing
        assert "window.location.search" in landing

    def test_targets_meet_the_minimums(self) -> None:
        css = _css()
        assert "min-height: 44px" in css.split(".fold-handle {", 1)[1].split("}", 1)[0]
        assert "min-height: 44px" in css.split(".orderings .column {", 1)[1].split("}", 1)[0]
        assert "const COLUMN_MIN_W = 40;" in _js()
        assert "Math.max(COLUMN_MIN_W," in _js()

    def test_focus_is_visible_on_every_target(self) -> None:
        # The ordering columns sit inside a scrolling box that clips an outset
        # ring, so theirs is drawn inside; the stage frame is a target during
        # the viewings (tap toggles playback) and so can take focus.
        css = _css()
        assert ".orderings .column:focus-visible { outline-offset: -2px; }" in css
        assert 'id="frame" tabindex="0"' in _html()
        js = _js()
        assert "function stageFocusable(on)" in js
        assert 'if (event.key !== "Enter" || !state.viewing || state.shaping) return;' in js

    def test_role_heads_are_set_as_tokens(self) -> None:
        # spec-identity.md Table 1: role head "as token" — identity's tracking, not wider.
        assert ".row.head" not in _css()
        assert "letter-spacing" not in _css()
