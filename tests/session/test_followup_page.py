"""Source-level invariants of the follow-up page (spec 8, Table 9; contract §11).

No server exercises these files, so the defects they guard would otherwise be
invisible to every other test: a string drifting from Table 6, a color or a
radius creeping in, a Next that could be pressed before the excerpt was
heard, a choice that could be made before both captions had been on the
frame, or a clip id reaching the participant through a still's alt text.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from importlib.resources import as_file, files

import pytest

# Spec Table 6, the rows the follow-up uses.
TABLE_6_FOLLOWUP = (
    "Caption session",
    "Begin",
    "Next",
    "Play",
    "Show the other caption",
    "I'd rather have this one",
    "Preparing comparison…",
    "Which sounds were in this clip?",
    "Which clip is this sound from?",
    "Which caption would you rather have?",
    "Caption A",
    "Caption B",
)

# spec-identity.md Table 2: the seven tokens and no others.
TOKENS = {"#111311", "#F2F2EC", "#E9EAE4", "#1A1C1B", "#6F726E", "#C3C6C0", "#EAC93C"}

EMOJI = re.compile("[\U0001f000-\U0001ffff☀-➿⬀-⯿]")


def _source(name: str) -> str:
    return files("dpo.session").joinpath(name).read_text(encoding="utf-8")


def _html() -> str:
    return _source("followup.html")


def _css() -> str:
    return _source("followup.css")


def _js() -> str:
    return _source("followup.js")


def _strings_block(js: str) -> str:
    return js.split("const STRINGS = {", 1)[1].split("\n};", 1)[0]


def _selectors(css: str) -> set[str]:
    """Every selector that opens a rule, comma-split and stripped; at-rules skipped."""
    bare = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    found: set[str] = set()
    for group in re.findall(r"([^{}]+)\{", bare):
        for selector in group.split(","):
            selector = selector.strip()
            if selector and not selector.startswith("@") and not selector.startswith(":root"):
                found.add(selector)
    return found


def _tag(html: str, marker: str) -> str:
    """The opening tag that contains ``marker``."""
    start = html.rindex("<", 0, html.index(marker))
    return html[start : html.index(">", start) + 1]


# ---- copy ------------------------------------------------------------------


def test_every_table_6_string_is_in_strings_verbatim() -> None:
    block = _strings_block(_js())
    for text in TABLE_6_FOLLOWUP:
        assert f'"{text}"' in block, text
    # The clip count pattern ("Clip 2 of 6") names a still by position.
    assert 'clipCount: "Clip {n} of {m}"' in block


def test_strings_are_defined_once_and_the_shell_carries_no_copy() -> None:
    js = _js()
    assert js.count("const STRINGS = {") == 1
    body = _html().split("<body>", 1)[1]
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    assert re.sub(r"<[^>]+>", "", body).strip() == ""


def test_notices_speak_in_the_interfaces_voice_without_apology() -> None:
    block = _strings_block(_js()).lower()
    for word in ("sorry", "apolog", "oops", "please"):
        assert word not in block


# ---- identity --------------------------------------------------------------


@pytest.mark.parametrize("name", ["followup.html", "followup.css", "followup.js"])
def test_no_radius_shadow_gradient_svg_icon_or_emoji(name: str) -> None:
    source = _source(name)
    for forbidden in ("border-radius", "box-shadow", "gradient", "<svg", "http://", "https://"):
        assert forbidden not in source, forbidden
    assert not EMOJI.search(source)


@pytest.mark.parametrize("name", ["followup.html", "followup.css", "followup.js"])
def test_only_the_seven_tokens_are_named(name: str) -> None:
    hexes = set(re.findall(r"#[0-9A-Fa-f]{3,8}\b", _source(name)))
    assert hexes <= TOKENS, hexes - TOKENS


@pytest.mark.parametrize("name", ["followup.html", "followup.css", "followup.js"])
def test_mark_appears_only_where_identity_puts_it(name: str) -> None:
    # spec-identity.md: mark is the primary action and the shaped lane segment, both
    # owned by identity.css; the follow-up never reaches for it.
    assert "--mark" not in _source(name)


def test_followup_css_redefines_nothing_from_identity() -> None:
    identity = _selectors(_source("identity.css"))
    followup = _selectors(_css())
    assert identity
    assert followup
    assert not (identity & followup), identity & followup


def test_phone_first_single_column_wide_enough_for_the_kiosk_frame() -> None:
    css = _css()
    # 640 (kiosk.css --stage-w) + 2 x 16 padding: the check frame is unscaled
    # on a desktop, so the placed caption is the identity's 22 on 30 there.
    assert "max-width: 672px" in css
    assert "padding: 16px 16px 48px" in css
    assert "margin: 0 auto" in css
    assert "width=device-width" in _html()


def test_the_only_images_are_clip_stills() -> None:
    html, js = _html(), _js()
    assert html.count("<img") == 1
    assert 'class="still"' in _tag(html, 'id="recognition-still"')
    assert "<img" not in js
    # The six choice stills are created once, from the still route.
    assert js.count('element("img")') == 1
    assert "still.src = stillUrl(clip.clip_id)" in js


# ---- interactive elements --------------------------------------------------


def test_every_interactive_element_is_a_button_or_a_labeled_input() -> None:
    html, js = _html(), _js()
    assert "<a " not in html
    for tag in re.findall(r"<button[^>]*>", html):
        assert 'type="button"' in tag, tag
    for tag in re.findall(r"<input[^>]*>", html):
        input_id = re.search(r'id="([^"]+)"', tag)
        assert input_id is not None, tag
        assert f'for="{input_id.group(1)}"' in html
    # The rows and the stills are buttons, and clicks land only on buttons.
    assert 'element("button", "check")' in js
    assert 'element("button", "choice")' in js
    receivers = set(re.findall(r'([\w.]+)\.addEventListener\("click"', js))
    assert receivers == {
        "row",
        "choice",
        "dom.begin",
        "dom.recognitionNext",
        "dom.play",
        "dom.soundNext",
        "dom.toggle",
        "dom.choose",
    }


def test_recognition_rows_are_tokens_beside_a_44_box() -> None:
    js = _js()
    # identity .check .box is 44 square; the token carries its brackets as text.
    assert 'element("span", "box")' in js
    assert 'element("span", "token", "[" + sound + "]")' in js
    assert 'row.setAttribute("role", "checkbox")' in js
    # A checked box is filled (spec-identity.md Table 3): identity fills
    # `.check .box[aria-checked="true"]`, so the row's state must reach the box.
    assert '.check .box[aria-checked="true"]' in _source("identity.css")
    click = js.split('row.addEventListener("click"', 1)[1].split("\n      });", 1)[0]
    assert 'row.setAttribute("aria-checked", checked);' in click
    assert 'box.setAttribute("aria-checked", checked);' in click


def test_recognition_next_is_always_enabled_and_the_empty_answer_is_recorded() -> None:
    html, js = _html(), _js()
    assert "disabled" not in _tag(html, 'id="recognition-next"')
    assert "recognitionNext.disabled" not in js
    assert "state.recognition[step.item.clip_id] = step.item.sounds.filter(" in js


def test_the_excerpt_plays_with_no_image_and_no_controls() -> None:
    html = _html()
    audio = _tag(html, 'id="excerpt"')
    assert audio.startswith("<audio") and "controls" not in audio
    section = html.split('id="sound-only"', 1)[1].split("</section>", 1)[0]
    assert "<img" not in section and "<video" not in section


def test_sound_only_next_waits_for_a_play_and_a_choice() -> None:
    js = _js()
    assert "dom.soundNext.disabled = !(current.played > 0 && current.chosen !== null);" in js
    assert "disabled" in _tag(_html(), 'id="sound-next"')
    # A play is counted when playback actually began, not when the button was pressed.
    assert "current.played += 1;" in js.split("dom.excerpt.play(),", 1)[1].split("updateSound", 1)[0]
    assert "played: current.played" in js


def test_the_check_frame_is_the_kiosk_stage_scaled_not_reflowed() -> None:
    # identity's 22px two-line box holds a caption only at the kiosk's stage
    # width, so the follow-up lays the frame out at that width and scales the
    # picture; the constant must be the kiosk's.
    kiosk_css = files("dpo.session").joinpath("kiosk.css").read_text(encoding="utf-8")
    # On a larger screen the kiosk's stage grows past the tablet's width; the
    # follow-up still lays the frame out at that width, so the two constants
    # must be one number, whatever it is.
    kiosk_px = re.search(r"--kiosk-stage-w:\s*(\d+)px", kiosk_css)
    followup_px = re.search(r"const KIOSK_STAGE_WIDTH = (\d+);", _js())
    assert kiosk_px and followup_px and kiosk_px.group(1) == followup_px.group(1) == "640"
    css = _css()
    stage = css.split(".stage {", 1)[1].split("}", 1)[0]
    assert "aspect-ratio: 16 / 9" in stage and "overflow: hidden" in stage
    frame = css.split("\n.frame {", 1)[1].split("}", 1)[0]
    assert "width: 640px" in frame and "height: 360px" in frame
    assert "transform: scale(var(--frame-scale, 1))" in frame
    js = _js()
    # Never scaled up: at the kiosk's width or wider the picture is the kiosk's.
    assert (
        'dom.stage.style.setProperty("--frame-scale", String(Math.min(1, width / KIOSK_STAGE_WIDTH)));' in js
    )
    assert "dom.check.hidden = false;\n  fitStage();" in js
    assert 'window.addEventListener("resize", fitStage);' in js


def test_the_check_shows_the_kiosk_caption_object_on_a_frame() -> None:
    html, js = _html(), _js()
    frame = html.split('class="frame dark"', 1)[1].split("</div>\n      </div>", 1)[0]
    video = _tag(frame, 'id="clip"')
    assert "playsinline" in video and "loop" in video and "controls" not in video
    assert frame.index("<video") < frame.index('class="caption-object placed"')
    assert 'class="prose" id="placed-prose"' in frame
    # The caption follows the shot's timing from a rAF loop.
    assert "requestAnimationFrame(loop)" in js
    assert "at >= candidate.start_ms && at < candidate.end_ms" in js
    assert 'dom.placed.hidden = text === "";' in js


def test_the_choice_reads_preparing_then_waits_for_both_captions() -> None:
    js = _js()
    show = js.split("function showCheck(", 1)[1].split("\nfunction ", 1)[0]
    assert "dom.choose.textContent = STRINGS.preparing;" in show
    assert "dom.choose.disabled = true;" in show
    assert "dom.toggle.disabled = true;" in show
    prepare = js.split("function prepareCheck(", 1)[1].split("\nfunction ", 1)[0]
    assert "dom.choose.textContent = STRINGS.choose;" in prepare
    assert 'addEventListener("canplay"' in js and "prepareCheck();" in js
    assert "dom.choose.disabled = !(current.prepared && current.seen.A && current.seen.B);" in js
    assert "disabled" in _tag(_html(), 'id="choose"')
    # A set has "been on the frame" (spec 8) only when the loop painted one of
    # its captions: the mark lives in loop() and nowhere else.
    assert js.count("current.seen[current.showing] = true;") == 1
    loop = js.split("function loop(", 1)[1].split("\nfunction ", 1)[0]
    assert 'if (text !== "" && !current.seen[current.showing]) {' in loop
    assert "current.seen[current.showing] = true;" in loop
    # The last shot holds through the media's tail, as the kiosk's shotAt() does.
    assert "(last && at >= last.end_ms ? last : null)" in loop


def test_the_single_toggle_swaps_the_set_and_the_eyebrow() -> None:
    js = _js()
    toggle = js.split('dom.toggle.addEventListener("click"', 1)[1].split("\n});", 1)[0]
    assert 'current.showing = current.showing === "A" ? "B" : "A";' in toggle
    assert "current.seen" not in toggle
    assert "current.toggles += 1;" in toggle
    assert 'current.showing === "A" ? STRINGS.captionA : STRINGS.captionB' in toggle
    assert "chosen: current.showing, toggles: current.toggles" in js


# ---- resume and responses --------------------------------------------------


def test_progress_is_saved_under_a_key_that_names_the_participant() -> None:
    js = _js()
    assert 'const STORAGE_PREFIX = "dpo.caption-session-followup/v1:";' in js
    assert "return STORAGE_PREFIX + id;" in js
    assert "localStorage.setItem(storageKey(participant), JSON.stringify(state));" in js
    assert "state = loadState(participant) || freshState(participant);" in js


def test_resume_is_derived_from_the_answers_not_a_stored_index() -> None:
    js = _js()
    resume = js.split("function resumeIndex(", 1)[1].split("\nfunction ", 1)[0]
    assert "steps.findIndex((step) => !answered(step))" in resume
    answered = js.split("function answered(", 1)[1].split("\nfunction ", 1)[0]
    assert "step.item.clip_id in state.recognition" in answered
    assert "step.item.excerpt_id in state.sound_only" in answered
    assert "step.item.clip_id in state.check" in answered
    begin = js.split("async function begin(", 1)[1].split("\nfunction ", 1)[0]
    assert "state.step = resumeIndex();" in begin
    assert "Math.min(state.step" not in js
    # "done" reached with an answer missing shows that step, never the owed POST.
    show = js.split("function showStep(", 1)[1].split("\nfunction ", 1)[0]
    assert "else if (resumeIndex() < state.step) {" in show


def test_failed_media_is_named_and_the_check_clip_is_asked_for_again() -> None:
    js = _js()
    assert "dom.recognitionStill.onerror = () => notice(STRINGS.mediaFailed);" in js
    assert 'still.addEventListener("error", () => notice(STRINGS.mediaFailed));' in js
    error = js.split('dom.clip.addEventListener("error"', 1)[1].split("\n});", 1)[0]
    assert "notice(STRINGS.mediaFailed);" in error
    assert "dom.clip.load();" in error and "MEDIA_RETRY_MS" in error
    assert "const MEDIA_RETRY_MS = 3000;" in js


def test_a_record_already_on_the_server_counts_as_saved() -> None:
    assert "saved = response.ok || response.status === 409;" in _js()


def test_responses_are_posted_once_in_the_contract_shape() -> None:
    js = _js()
    assert js.count('"/api/followup/responses"') == 1
    post = js.split('"/api/followup/responses"', 1)[1].split("});", 1)[0]
    assert 'method: "POST"' in post
    assert 'const RESPONSES_SCHEMA = "dpo.caption-session-followup/v1";' in js
    document = js.split("function responsesDocument(", 1)[1].split("\nasync function", 1)[0]
    for key in (
        "schema: RESPONSES_SCHEMA",
        "participant,",
        "checked:",
        "chosen_clip_id:",
        "played:",
        "chosen:",
        "toggles:",
    ):
        assert key in document, key
    # The done screen is entered only after the save succeeds.
    assert "state.submitted = true;" in js.split("if (!saved) {", 1)[1]


def test_a_missing_kiosk_session_is_named_with_the_way_forward() -> None:
    js = _js()
    assert "response.status === 409" in js
    assert "notice(STRINGS.noSession);" in js
    assert "begin again" in _strings_block(js)


def test_stills_are_named_by_position_and_never_by_id() -> None:
    js = _js()
    assert "dom.recognitionStill.alt = clipLabel(clipId);" in js
    assert "still.alt = clipLabel(clip.clip_id);" in js
    for line in js.splitlines():
        if "textContent" in line or ".alt =" in line:
            assert "clip_id" not in line or "clipLabel(" in line, line


def test_nothing_of_the_kiosk_reaches_the_follow_up() -> None:
    for name in ("followup.html", "followup.css", "followup.js"):
        source = _source(name)
        for forbidden in ("/api/inventory", "/api/caption", "/api/events", ".row", 'class="row"', ".lane"):
            assert forbidden not in source, (name, forbidden)
    html = _html()
    for static in ("/static/identity.css", "/static/followup.css", "/static/followup.js"):
        assert static in html
    assert html.count("/static/") == 3


def test_the_script_parses() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    with as_file(files("dpo.session").joinpath("followup.js")) as path:
        subprocess.run([node, "--check", str(path)], check=True)
